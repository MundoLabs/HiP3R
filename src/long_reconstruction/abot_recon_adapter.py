"""ABot-Recon adapter for local long-reconstruction chunk predictions.

ABot-Recon is an external source checkout.  The adapter never downloads model
weights: callers must provide a local checkpoint path or set
``ABOT_RECON_CHECKPOINT``.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from long_reconstruction.contract import ChunkPrediction
from long_reconstruction.vggt_omega_adapter import (
    _check_finite,
    _squeeze_single_batch,
    _to_cpu_tensor,
)


ABOT_RECON_REPO_ROOT_ENV = "ABOT_RECON_REPO_ROOT"
ABOT_RECON_CHECKPOINT_ENV = "ABOT_RECON_CHECKPOINT"
DEFAULT_HEIGHT = 280
DEFAULT_WIDTH = 504
DEFAULT_MAX_FRAMES = 22_000
DEFAULT_ATTENTION_BACKEND = "auto"
DEFAULT_CONFIDENCE_THRESHOLD = 0.0

_AMP_DTYPE_ALIASES = {
    "float32": "fp32",
    "fp32": "fp32",
    "float16": "fp16",
    "fp16": "fp16",
    "bfloat16": "bf16",
    "bf16": "bf16",
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}


def _amp_dtype_from_spec(dtype: Any) -> str:
    key = dtype if isinstance(dtype, torch.dtype) else str(dtype).lower()
    if key not in _AMP_DTYPE_ALIASES:
        raise ValueError(f"unsupported ABot-Recon inference dtype: {dtype}")
    return _AMP_DTYPE_ALIASES[key]


def _resolve_repo_root(repo_root: Any = None) -> Path:
    source = (
        repo_root
        if repo_root is not None
        else os.environ.get(ABOT_RECON_REPO_ROOT_ENV)
    )
    if source is None:
        raise ValueError(
            "ABot-Recon requires --abot-recon-repo-root or "
            f"{ABOT_RECON_REPO_ROOT_ENV}"
        )
    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ABot-Recon repo_root does not exist: {root}")
    if not (root / "abot_recon" / "api.py").is_file():
        raise FileNotFoundError(f"ABot-Recon source is missing under: {root}")
    return root


def _resolve_checkpoint(checkpoint: Any = None) -> Path | None:
    source = (
        checkpoint
        if checkpoint is not None
        else os.environ.get(ABOT_RECON_CHECKPOINT_ENV)
    )
    if source is None:
        return None
    path = Path(source).expanduser().absolute()
    if not path.is_file():
        raise FileNotFoundError(f"ABot-Recon checkpoint not found: {path}")
    return path


def _path_is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _module_origin_paths(module: Any) -> list[Path]:
    origins = []
    path = getattr(module, "__file__", None)
    if path:
        origins.append(Path(path))
    package_paths = getattr(module, "__path__", None)
    if package_paths:
        origins.extend(Path(path) for path in package_paths)
    return origins


def _evict_abot_modules_outside_repo(root: Path) -> None:
    for name, module in list(sys.modules.items()):
        if name != "abot_recon" and not name.startswith("abot_recon."):
            continue
        origins = _module_origin_paths(module)
        if origins and not any(_path_is_relative_to(origin, root) for origin in origins):
            sys.modules.pop(name, None)


@contextmanager
def _repo_import_context(repo_root: Path):
    _evict_abot_modules_outside_repo(repo_root)
    root = os.fspath(repo_root)
    inserted = False
    if root not in sys.path:
        sys.path.insert(0, root)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(root)
            except ValueError:
                pass


def _import_from_repo(repo_root: Path, module_name: str):
    with _repo_import_context(repo_root):
        return importlib.import_module(module_name)


def _squeeze_confidence(value: Any) -> torch.Tensor:
    tensor = _to_cpu_tensor(value, dtype=torch.float32)
    if tensor.dim() == 5 and tensor.shape[0] == 1 and tensor.shape[-1] == 1:
        tensor = tensor[0, ..., 0]
    elif tensor.dim() == 4 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    elif tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.dim() != 3:
        raise ValueError(
            f"ABot-Recon confidence must have shape [S,H,W], got {tuple(tensor.shape)}"
        )
    return tensor


def _recover_intrinsics(local_points: torch.Tensor) -> torch.Tensor:
    """Match Pi3's centered-principal-point ray calibration helper."""
    sequence, height, width, _ = local_points.shape
    dtype = local_points.dtype
    device = local_points.device
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    z = local_points[..., 2].abs().clamp_min(1e-10)
    x_projection = local_points[..., 0] / z
    y_projection = local_points[..., 1] / z
    centered_x = grid_x - (width - 1) / 2.0
    centered_y = grid_y - (height - 1) / 2.0
    fx = (x_projection * centered_x).sum((-2, -1)) / x_projection.square().sum(
        (-2, -1)
    ).clamp_min(1e-10)
    fy = (y_projection * centered_y).sum((-2, -1)) / y_projection.square().sum(
        (-2, -1)
    ).clamp_min(1e-10)
    intrinsic = torch.zeros((sequence, 3, 3), dtype=dtype, device=device)
    intrinsic[:, 0, 0] = fx
    intrinsic[:, 1, 1] = fy
    intrinsic[:, 0, 2] = (width - 1) / 2.0
    intrinsic[:, 1, 2] = (height - 1) / 2.0
    intrinsic[:, 2, 2] = 1.0
    return intrinsic


def _finite_or_zero(tensor: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))


def _result_value(result: Any, name: str) -> Any:
    if isinstance(result, Mapping):
        return result.get(name)
    return getattr(result, name, None)


class ABotReconChunkAdapter:
    """Run an external ABot-Recon checkout and emit ``ChunkPrediction``."""

    def __init__(
        self,
        model: Any = None,
        *,
        repo_root: Any = None,
        checkpoint: Any = None,
        device: Any = None,
        inference_dtype: Any = "bfloat16",
        amp_dtype: Any = None,
        attention_backend: str = DEFAULT_ATTENTION_BACKEND,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        max_frames: int = DEFAULT_MAX_FRAMES,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.repo_root = _resolve_repo_root(repo_root)
        self.checkpoint = _resolve_checkpoint(checkpoint)
        self.device = str(device) if device is not None else "cuda"
        self.amp_dtype = _amp_dtype_from_spec(
            inference_dtype if amp_dtype is None else amp_dtype
        )
        self.attention_backend = str(attention_backend)
        if self.attention_backend not in {"auto", "paged", "sdpa"}:
            raise ValueError("attention_backend must be one of: auto, paged, sdpa")
        self.height = int(height)
        self.width = int(width)
        self.max_frames = int(max_frames)
        self.confidence_threshold = float(confidence_threshold)
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self._last_inference_call: dict[str, Any] | None = None

        if model is None:
            if self.checkpoint is None:
                raise ValueError(
                    "ABot-Recon checkpoint is required; pass checkpoint or set "
                    f"{ABOT_RECON_CHECKPOINT_ENV}"
                )
            module = _import_from_repo(self.repo_root, "abot_recon")
            abot_recon = getattr(module, "ABotRecon")
            model = abot_recon.from_pretrained(
                self.checkpoint,
                local_files_only=True,
                device=self.device,
                amp_dtype=self.amp_dtype,
                attention_backend=self.attention_backend,
                loop_closure=False,
                height=self.height,
                width=self.width,
                max_frames=self.max_frames,
            )
            initialization = "external_abot_recon_from_pretrained"
        else:
            initialization = "provided_model"
        self.model = model
        self.load_report = {
            "backend": "abot_recon",
            "source_type": initialization,
            "repo_root": str(self.repo_root),
            "checkpoint": str(self.checkpoint) if self.checkpoint is not None else None,
            "local_files_only": True,
            "implicit_download": False,
            "device": self.device,
            "amp_dtype": self.amp_dtype,
            "attention_backend": self.attention_backend,
            "height": self.height,
            "width": self.width,
            "max_frames": self.max_frames,
            "confidence_threshold": self.confidence_threshold,
            "model_initialization_mode": initialization,
            "preprocess_protocol": self.preprocess_protocol(),
        }

    def preprocess_protocol(self) -> dict[str, Any]:
        return {
            "backend": "abot_recon",
            "loader": "abot_recon.ABotRecon.infer",
            "height": self.height,
            "width": self.width,
            "path_ordering": "input_order_preserved",
            "pose_contract": "native_camera_to_world_opencv",
            "intrinsic_recovery": "centered_principal_point_ray_least_squares",
            "confidence_threshold": self.confidence_threshold,
            "loop_closure": False,
            "attention_backend": self.attention_backend,
            "implicit_download": False,
        }

    def activation_report(self) -> dict[str, Any]:
        return {
            "backend": "abot_recon",
            "repo_root": str(self.repo_root),
            "checkpoint": str(self.checkpoint) if self.checkpoint is not None else None,
            "load_report": dict(self.load_report),
            "preprocess_protocol": self.preprocess_protocol(),
            "last_inference_call": dict(self._last_inference_call or {}),
        }

    def infer_paths(
        self,
        image_paths: Iterable[os.PathLike | str],
        **_: Any,
    ) -> ChunkPrediction:
        paths = [os.fspath(path) for path in image_paths]
        if not paths:
            raise ValueError("image_paths is empty")
        result = self.model.infer(
            paths,
            output_local_points=True,
            output_world_points=True,
            output_confidence=True,
            confidence_threshold=0.0,
            loop_closure=False,
        )
        prediction = self._prediction_from_result(result)
        self._last_inference_call = {
            "num_paths": len(paths),
            "paths": paths,
            "output_shape": list(prediction.world_points.shape),
        }
        return prediction

    def infer_loop_paths(
        self,
        image_paths: Iterable[os.PathLike | str],
        **kwargs: Any,
    ) -> ChunkPrediction:
        return self.infer_paths(image_paths, **kwargs)

    def _prediction_from_result(
        self,
        result: Any,
    ) -> ChunkPrediction:
        missing = [
            name
            for name in ("local_points", "world_points", "confidence", "camera_poses")
            if _result_value(result, name) is None
        ]
        if missing:
            raise KeyError(
                f"ABot-Recon result is missing required field(s): {', '.join(missing)}"
            )

        local_points = _squeeze_single_batch(
            "local_points",
            _result_value(result, "local_points"),
            4,
        ).float()
        world_points = _squeeze_single_batch(
            "world_points",
            _result_value(result, "world_points"),
            4,
        ).float()
        poses = _squeeze_single_batch(
            "camera_poses",
            _result_value(result, "camera_poses"),
            3,
        ).float()
        confidence = _squeeze_confidence(_result_value(result, "confidence")).float()
        if world_points.shape != local_points.shape:
            raise ValueError(
                f"ABot-Recon world_points shape {tuple(world_points.shape)} "
                f"does not match local_points shape {tuple(local_points.shape)}"
            )
        if confidence.shape != local_points.shape[:3]:
            raise ValueError(
                f"ABot-Recon confidence shape {tuple(confidence.shape)} "
                f"does not match local_points shape {tuple(local_points.shape[:3])}"
            )
        if poses.shape[0] != local_points.shape[0]:
            raise ValueError(
                f"ABot-Recon camera_poses has {poses.shape[0]} frame(s), "
                f"but dense outputs have {local_points.shape[0]}"
            )
        finite_local = torch.isfinite(local_points).all(-1)
        finite_world = torch.isfinite(world_points).all(-1)
        finite_confidence = torch.isfinite(confidence)
        depth = local_points[..., 2]
        valid = (
            finite_local
            & finite_world
            & torch.isfinite(depth)
            & (depth > 0)
            & finite_confidence
            & (confidence >= self.confidence_threshold)
        )
        local_points = _finite_or_zero(local_points)
        world_points = _finite_or_zero(world_points)
        depth = _finite_or_zero(depth)
        confidence = torch.where(
            valid,
            _finite_or_zero(confidence),
            torch.zeros_like(confidence),
        )
        intrinsic = _recover_intrinsics(local_points)

        for name, value in (
            ("world_points", world_points),
            ("local_points", local_points),
            ("camera_poses", poses),
            ("intrinsic", intrinsic),
            ("depth", depth),
            ("confidence", confidence),
        ):
            _check_finite(name, value)
        return ChunkPrediction(
            world_points=world_points.cpu(),
            world_points_conf=confidence.cpu(),
            extrinsic_c2w=poses.cpu(),
            intrinsic=intrinsic.cpu(),
            depth=depth.cpu(),
            depth_conf=confidence.cpu(),
            mask=valid.cpu(),
        )
