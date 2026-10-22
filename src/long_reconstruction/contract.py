"""Typed local prediction contract for HiP3R reconstruction."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import sys
from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional at import time.
    np = None


ArrayLike = Any
_FINITE_VALIDATED_TOKEN = object()


def _is_torch_tensor(value: ArrayLike) -> bool:
    torch_module = sys.modules.get("torch")
    module_spec = getattr(torch_module, "__spec__", None)
    if bool(getattr(module_spec, "_initializing", False)):
        return False
    is_tensor = getattr(torch_module, "is_tensor", None)
    if not callable(is_tensor):
        return False
    try:
        return bool(is_tensor(value))
    except (AttributeError, NameError):
        # Torch publishes parts of its module namespace before import finishes.
        return False


def _shape(name: str, value: ArrayLike) -> Tuple[int, ...]:
    if not hasattr(value, "shape"):
        raise TypeError(f"{name} must be a torch.Tensor or numpy.ndarray-like object with shape")
    return tuple(int(dim) for dim in value.shape)


def _is_numpy_array(value: ArrayLike) -> bool:
    return np is not None and isinstance(value, np.ndarray)


def _check_supported_array(name: str, value: ArrayLike) -> None:
    if _is_torch_tensor(value) or _is_numpy_array(value):
        return
    raise TypeError(f"{name} must be a torch.Tensor or numpy.ndarray, got {type(value).__name__}")


def _check_numeric_or_bool(name: str, value: ArrayLike) -> None:
    _check_supported_array(name, value)
    if _is_torch_tensor(value):
        torch_module = sys.modules["torch"]
        try:
            torch_module.isfinite(value)
        except TypeError:
            if value.dtype == torch_module.bool:
                return
            raise TypeError(f"{name} must contain numeric finite values") from None
        return

    if not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
        raise TypeError(f"{name} must contain numeric finite values")


def _check_finite(name: str, value: ArrayLike) -> None:
    _check_numeric_or_bool(name, value)
    if _is_torch_tensor(value):
        torch_module = sys.modules["torch"]
        try:
            finite = torch_module.isfinite(value)
        except TypeError:
            if value.dtype == torch_module.bool:
                return
            raise TypeError(f"{name} must contain numeric finite values") from None
        if not bool(finite.all().item()):
            raise ValueError(f"{name} contains non-finite values")
        return

    if value.dtype == np.bool_:
        return
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _check_shape(
    name: str,
    value: ArrayLike,
    expected: Tuple[int, ...],
    *,
    skip_finite: bool = False,
) -> None:
    actual = _shape(name, value)
    if actual != expected:
        raise ValueError(f"{name} must have shape {expected}, got {actual}")
    if skip_finite:
        _check_numeric_or_bool(name, value)
    else:
        _check_finite(name, value)


def _to_numpy(name: str, value: ArrayLike):
    if np is None:
        raise RuntimeError("numpy is required for ChunkPrediction.to_numpy_dict()")
    _check_supported_array(name, value)
    if _is_torch_tensor(value):
        return value.detach().cpu().numpy()
    return value


@dataclass(frozen=True)
class ChunkPrediction:
    """Validated local chunk prediction consumed by hip3r long reconstruction."""

    world_points: ArrayLike
    world_points_conf: ArrayLike
    extrinsic_c2w: ArrayLike
    intrinsic: ArrayLike
    depth: Optional[ArrayLike] = None
    depth_conf: Optional[ArrayLike] = None
    images: Optional[ArrayLike] = None
    mask: Optional[ArrayLike] = None
    _finite_validated_fields: frozenset[str] = field(
        default_factory=frozenset,
        init=False,
        compare=False,
        repr=False,
    )
    _skip_finite_validation_token: InitVar[object | None] = field(
        default=None,
        kw_only=True,
        repr=False,
    )
    _validated_fields: InitVar[Iterable[str] | None] = field(
        default=None,
        kw_only=True,
        repr=False,
    )

    @classmethod
    def _from_finite_validated_arrays(
        cls,
        *,
        _token: object,
        _validated_fields: Iterable[str],
        **arrays: ArrayLike,
    ) -> "ChunkPrediction":
        if _token is not _FINITE_VALIDATED_TOKEN:
            raise ValueError("invalid finite-validation token")
        validated = frozenset(str(name) for name in _validated_fields)
        unknown = validated.difference(arrays)
        if unknown:
            raise ValueError(
                f"finite-validated fields are missing from arrays: {sorted(unknown)}"
            )
        return cls(
            **arrays,
            _skip_finite_validation_token=_token,
            _validated_fields=validated,
        )

    def _has_finite_validated_fields(self, *names: str) -> bool:
        return all(name in self._finite_validated_fields for name in names)

    def __post_init__(
        self,
        _skip_finite_validation_token: object | None,
        _validated_fields: Iterable[str] | None,
    ) -> None:
        if (
            _skip_finite_validation_token is not None
            and _skip_finite_validation_token is not _FINITE_VALIDATED_TOKEN
        ):
            raise ValueError("invalid finite-validation token")
        if _skip_finite_validation_token is None and _validated_fields is not None:
            raise ValueError("finite-validated fields require the validation token")
        finite_validated_fields = (
            frozenset(str(name) for name in _validated_fields)
            if _skip_finite_validation_token is _FINITE_VALIDATED_TOKEN
            else frozenset()
        )
        object.__setattr__(
            self, "_finite_validated_fields", finite_validated_fields
        )
        world_shape = _shape("world_points", self.world_points)
        if len(world_shape) != 4 or world_shape[-1] != 3:
            raise ValueError(f"world_points must have shape [S,H,W,3], got {world_shape}")
        if "world_points" in finite_validated_fields:
            _check_numeric_or_bool("world_points", self.world_points)
        else:
            _check_finite("world_points", self.world_points)

        sequence, height, width, _ = world_shape
        _check_shape(
            "world_points_conf",
            self.world_points_conf,
            (sequence, height, width),
            skip_finite="world_points_conf" in finite_validated_fields,
        )
        _check_shape(
            "extrinsic_c2w",
            self.extrinsic_c2w,
            (sequence, 4, 4),
            skip_finite="extrinsic_c2w" in finite_validated_fields,
        )
        _check_shape(
            "intrinsic",
            self.intrinsic,
            (sequence, 3, 3),
            skip_finite="intrinsic" in finite_validated_fields,
        )

        if self.depth is not None:
            depth_shape = _shape("depth", self.depth)
            if depth_shape not in ((sequence, height, width), (sequence, height, width, 1)):
                raise ValueError(
                    f"depth must have shape {(sequence, height, width)} or "
                    f"{(sequence, height, width, 1)}, got {depth_shape}"
                )
            if "depth" in finite_validated_fields:
                _check_numeric_or_bool("depth", self.depth)
            else:
                _check_finite("depth", self.depth)

        if self.depth_conf is not None:
            _check_shape(
                "depth_conf",
                self.depth_conf,
                (sequence, height, width),
                skip_finite="depth_conf" in finite_validated_fields,
            )

        if self.images is not None:
            image_shape = _shape("images", self.images)
            if image_shape not in ((sequence, 3, height, width), (sequence, height, width, 3)):
                raise ValueError(
                    f"images must have shape {(sequence, 3, height, width)} or "
                    f"{(sequence, height, width, 3)}, got {image_shape}"
                )
            if "images" in finite_validated_fields:
                _check_numeric_or_bool("images", self.images)
            else:
                _check_finite("images", self.images)

        if self.mask is not None:
            _check_shape(
                "mask",
                self.mask,
                (sequence, height, width),
                skip_finite="mask" in finite_validated_fields,
            )

    def to_numpy_dict(self) -> Dict[str, Any]:
        """Return all populated fields as numpy arrays."""
        result = {
            "world_points": _to_numpy("world_points", self.world_points),
            "world_points_conf": _to_numpy("world_points_conf", self.world_points_conf),
            "extrinsic_c2w": _to_numpy("extrinsic_c2w", self.extrinsic_c2w),
            "intrinsic": _to_numpy("intrinsic", self.intrinsic),
        }
        for name in ("depth", "depth_conf", "images", "mask"):
            value = getattr(self, name)
            if value is not None:
                result[name] = _to_numpy(name, value)
        return result
