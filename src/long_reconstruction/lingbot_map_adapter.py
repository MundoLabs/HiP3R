"""LingBot-MAP adapter for local chunk predictions."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

from long_reconstruction.contract import ChunkPrediction

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]


_DTYPE_ALIASES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def _first_parameter_device(model: Any) -> torch.device:
    if hasattr(model, "parameters"):
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass
    return torch.device("cpu")


def _dtype_from_spec(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        if dtype not in {torch.float32, torch.bfloat16, torch.float16}:
            raise ValueError(f"unsupported inference dtype: {dtype}")
        return dtype
    key = str(dtype).lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"unsupported inference dtype: {dtype}")
    return _DTYPE_ALIASES[key]


def _to_device_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def _to_cpu_tensor(value: Any, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _check_finite(name: str, value: torch.Tensor) -> None:
    try:
        finite = torch.isfinite(value)
    except TypeError:
        if value.dtype == torch.bool:
            return
        raise TypeError(f"{name} must contain numeric finite values") from None
    if not bool(finite.all().item()):
        raise ValueError(f"{name} contains non-finite values")


def _image_size_hw(images: torch.Tensor) -> Tuple[int, int]:
    if images.dim() == 4:
        if images.shape[1] != 3:
            raise ValueError(f"images must have shape [S,3,H,W], got {tuple(images.shape)}")
        return int(images.shape[-2]), int(images.shape[-1])
    if images.dim() == 5:
        if images.shape[0] != 1 or images.shape[2] != 3:
            raise ValueError(f"images must have shape [1,S,3,H,W], got {tuple(images.shape)}")
        return int(images.shape[-2]), int(images.shape[-1])
    raise ValueError(f"images must have shape [S,3,H,W] or [1,S,3,H,W], got {tuple(images.shape)}")


def _frame_tensor(images: Any, device: torch.device) -> torch.Tensor:
    tensor = _to_device_tensor(images, device)
    _image_size_hw(tensor)
    if tensor.dim() == 5:
        tensor = tensor[0]
    return tensor


def _squeeze_single_batch(name: str, value: Any, expected_unbatched_ndim: int) -> torch.Tensor:
    tensor = _to_cpu_tensor(value)
    shape = tuple(tensor.shape)
    if len(shape) == expected_unbatched_ndim:
        _check_finite(name, tensor)
        return tensor
    if len(shape) == expected_unbatched_ndim + 1:
        if shape[0] != 1:
            raise ValueError(f"{name} batch dimension must be 1 for chunk predictions, got {shape[0]}")
        tensor = tensor[0]
        _check_finite(name, tensor)
        return tensor
    raise ValueError(f"{name} has unsupported shape {shape}")


def _squeeze_depth(depth: Any) -> torch.Tensor:
    tensor = _to_cpu_tensor(depth)
    shape = tuple(tensor.shape)
    if tensor.dim() == 5:
        if tensor.shape[0] != 1:
            raise ValueError(f"depth batch dimension must be 1 for chunk predictions, got {tensor.shape[0]}")
        tensor = tensor[0]
    elif tensor.dim() == 4 and tensor.shape[-1] != 1:
        if tensor.shape[0] != 1:
            raise ValueError(f"depth has unsupported shape {shape}")
        tensor = tensor[0]
    elif tensor.dim() not in (3, 4):
        raise ValueError(f"depth has unsupported shape {shape}")
    if tensor.dim() == 4:
        if tensor.shape[-1] != 1:
            raise ValueError(f"depth has unsupported shape {tuple(tensor.shape)}")
        tensor = tensor[..., 0]
    _check_finite("depth", tensor)
    return tensor


def _squeeze_optional(name: str, value: Any, expected_unbatched_ndim: int):
    if value is None:
        return None
    return _squeeze_single_batch(name, value, expected_unbatched_ndim)


def _homogenize_w2c(extrinsic_w2c: torch.Tensor) -> torch.Tensor:
    if extrinsic_w2c.shape[-2:] == (4, 4):
        return extrinsic_w2c
    if extrinsic_w2c.shape[-2:] != (3, 4):
        raise ValueError(f"extrinsic_w2c must have shape [...,3,4] or [...,4,4], got {tuple(extrinsic_w2c.shape)}")
    bottom = torch.zeros(
        extrinsic_w2c.shape[:-2] + (1, 4),
        dtype=extrinsic_w2c.dtype,
        device=extrinsic_w2c.device,
    )
    bottom[..., 0, 3] = 1
    return torch.cat([extrinsic_w2c, bottom], dim=-2)


def _w2c_to_c2w(extrinsic_w2c: torch.Tensor) -> torch.Tensor:
    extrinsic_w2c = _to_cpu_tensor(extrinsic_w2c, dtype=torch.float32)
    _check_finite("extrinsic_w2c", extrinsic_w2c)
    return torch.linalg.inv(_homogenize_w2c(extrinsic_w2c))


def _decoded_pose_to_c2w(decoded_pose: torch.Tensor, pose_convention: str) -> torch.Tensor:
    """Convert the decoder output to C2W under an explicit convention."""
    decoded_pose = _to_cpu_tensor(decoded_pose, dtype=torch.float32)
    _check_finite("decoded_pose", decoded_pose)
    homogeneous = _homogenize_w2c(decoded_pose)
    if pose_convention == "decoded_c2w":
        return homogeneous
    if pose_convention == "decoded_w2c":
        return torch.linalg.inv(homogeneous)
    raise ValueError(f"unsupported LingBot-MAP pose convention: {pose_convention!r}")


def _unproject_depth_to_world_points(
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic_c2w: torch.Tensor,
) -> torch.Tensor:
    depth = depth.float()
    intrinsic = intrinsic.float()
    extrinsic_c2w = extrinsic_c2w.float()
    sequence, height, width = depth.shape

    y, x = torch.meshgrid(
        torch.arange(height, dtype=depth.dtype),
        torch.arange(width, dtype=depth.dtype),
        indexing="ij",
    )
    x = x.view(1, height, width).expand(sequence, height, width)
    y = y.view(1, height, width).expand(sequence, height, width)

    fx = intrinsic[:, 0, 0].view(sequence, 1, 1)
    fy = intrinsic[:, 1, 1].view(sequence, 1, 1)
    cx = intrinsic[:, 0, 2].view(sequence, 1, 1)
    cy = intrinsic[:, 1, 2].view(sequence, 1, 1)
    camera_points = torch.stack(
        (
            (x - cx) * depth / fx,
            (y - cy) * depth / fy,
            depth,
        ),
        dim=-1,
    )
    rotation = extrinsic_c2w[:, :3, :3]
    translation = extrinsic_c2w[:, :3, 3]
    return torch.einsum("sij,shwj->shwi", rotation, camera_points) + translation[:, None, None, :]


def _filter_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _call_with_supported_kwargs(fn: Any, *args: Any, **kwargs: Any) -> Any:
    return fn(*args, **_filter_kwargs(fn, kwargs))


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


def _evict_lingbot_modules_outside_repo(root: Path) -> None:
    for name, module in list(sys.modules.items()):
        if name != "lingbot_map" and not name.startswith("lingbot_map."):
            continue
        origins = _module_origin_paths(module)
        if origins and not any(_path_is_relative_to(origin, root) for origin in origins):
            sys.modules.pop(name, None)


@contextmanager
def _repo_import_context(repo_root: Optional[os.PathLike | str]):
    if repo_root is None:
        yield
        return
    root_path = Path(repo_root).expanduser().resolve()
    root = os.fspath(root_path)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"LingBot-MAP repo_root does not exist: {root}")
    _evict_lingbot_modules_outside_repo(root_path)
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


def _import_from_repo(repo_root: Optional[os.PathLike | str], module_name: str):
    with _repo_import_context(repo_root):
        return importlib.import_module(module_name)


def _resolve_keyframe_interval(raw: Any, num_frames: int, threshold: int) -> int:
    if threshold <= 0:
        raise ValueError(f"auto_keyframe_threshold must be positive, got {threshold}")
    if raw is None or raw == 0 or (isinstance(raw, str) and raw.lower() == "auto"):
        if num_frames <= threshold:
            return 1
        return (num_frames + threshold - 1) // threshold
    value = int(raw)
    if value <= 0:
        raise ValueError(f"keyframe_interval must be positive or auto-like, got {raw!r}")
    return value


class LingbotMapChunkAdapter:
    """Run LingBot-MAP on one local chunk and return the HiP3R prediction contract."""

    def __init__(
        self,
        model: Any = None,
        *,
        repo_root: Optional[os.PathLike | str] = None,
        checkpoint: Any = None,
        device: Optional[Any] = None,
        mode: str = "windowed",
        geometry_source: str = "depth_unprojection",
        pose_convention: str = "decoded_c2w",
        dtype: Any = "bfloat16",
        inference_dtype: Any = None,
        image_size: int = 518,
        patch_size: int = 14,
        enable_3d_rope: bool = True,
        max_frame_num: int = 1024,
        num_scale_frames: int = 8,
        keyframe_interval: Any = "auto",
        auto_keyframe_threshold: int = 320,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: Optional[int] = None,
        window_size: int = 64,
        overlap_size: Optional[int] = None,
        overlap_keyframes: Optional[int] = None,
        overlap: Optional[int] = None,
        camera_num_iterations: int = 4,
        camera_iterations: Optional[int] = None,
        use_sdpa: bool = True,
        flow_threshold: float = 0.0,
        max_non_keyframe_gap: int = 30,
        output_cpu: bool = True,
        output_device: Optional[Any] = None,
        checkpoint_strict: bool = False,
        map_location: Any = "cpu",
    ) -> None:
        self.repo_root = repo_root if repo_root is not None else DEFAULT_REPO_ROOT
        self.mode = str(mode)
        if self.mode not in {"streaming", "windowed"}:
            raise ValueError(f"LingBot-MAP mode must be 'streaming' or 'windowed', got {self.mode!r}")
        self.geometry_source = str(geometry_source)
        if self.geometry_source not in {"auto", "native", "depth_unprojection"}:
            raise ValueError(
                "LingBot-MAP geometry_source must be 'auto', 'native', or "
                f"'depth_unprojection', got {self.geometry_source!r}"
            )
        self.pose_convention = str(pose_convention)
        if self.pose_convention not in {"decoded_c2w", "decoded_w2c"}:
            raise ValueError(
                "LingBot-MAP pose_convention must be 'decoded_c2w' or "
                f"'decoded_w2c', got {self.pose_convention!r}"
            )
        self._activation_counts: Counter[str] = Counter()
        self._last_inference_report: dict[str, Any] | None = None

        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.enable_3d_rope = bool(enable_3d_rope)
        self.max_frame_num = int(max_frame_num)
        self.num_scale_frames = int(num_scale_frames)
        self.keyframe_interval = keyframe_interval
        self.auto_keyframe_threshold = int(auto_keyframe_threshold)
        self.kv_cache_sliding_window = int(kv_cache_sliding_window)
        self.kv_cache_scale_frames = (
            int(kv_cache_scale_frames) if kv_cache_scale_frames is not None else self.num_scale_frames
        )
        self.window_size = int(window_size)
        if overlap_size is None and overlap is not None:
            overlap_size = overlap
        self.overlap_size = int(overlap_size) if overlap_size is not None else None
        self.overlap_keyframes = int(overlap_keyframes) if overlap_keyframes is not None else None
        self.camera_num_iterations = int(camera_iterations if camera_iterations is not None else camera_num_iterations)
        self.use_sdpa = bool(use_sdpa)
        self.flow_threshold = float(flow_threshold)
        self.max_non_keyframe_gap = int(max_non_keyframe_gap)
        if self.mode == "streaming" and self.flow_threshold != 0.0:
            raise ValueError(
                "flow_threshold is only supported by LingBot-MAP windowed mode; "
                "use mode='windowed' or leave flow_threshold at 0"
            )
        self.output_cpu = bool(output_cpu)
        self.output_device = torch.device(output_device) if output_device is not None else None
        self.inference_dtype = _dtype_from_spec(dtype if inference_dtype is None else inference_dtype)

        if model is None:
            model = self._build_model()
        self.model = model
        self.device = torch.device(device) if device is not None else _first_parameter_device(model)

        if checkpoint is not None:
            self.load_report = load_lingbot_map_checkpoint(
                self.model,
                checkpoint,
                strict=checkpoint_strict,
                map_location=map_location,
            )
        else:
            self.load_report = {
                "source_type": "none",
                "state_dict_type": "none",
                "total_keys": 0,
                "matched_keys": 0,
                "stripped_prefix": "",
                "strict": checkpoint_strict,
                "missing_keys": [],
                "unexpected_keys": [],
            }

        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def _build_model(self) -> Any:
        module_name = (
            "lingbot_map.models.gct_stream_window"
            if self.mode == "windowed"
            else "lingbot_map.models.gct_stream"
        )
        module = _import_from_repo(self.repo_root, module_name)
        kwargs = {
            "img_size": self.image_size,
            "patch_size": self.patch_size,
            "enable_3d_rope": self.enable_3d_rope,
            "max_frame_num": self.max_frame_num,
            "kv_cache_sliding_window": self.kv_cache_sliding_window,
            "kv_cache_scale_frames": self.kv_cache_scale_frames,
            "kv_cache_cross_frame_special": True,
            "kv_cache_include_scale_frames": True,
            "use_sdpa": self.use_sdpa,
            "camera_num_iterations": self.camera_num_iterations,
        }
        return module.GCTStream(**_filter_kwargs(module.GCTStream, kwargs))

    def infer_paths(self, image_paths: Iterable[os.PathLike | str], preprocess_mode: str = "crop") -> ChunkPrediction:
        paths = [os.fspath(path) for path in image_paths]
        if not paths:
            raise ValueError("image_paths is empty")
        module = _import_from_repo(self.repo_root, "lingbot_map.utils.load_fn")
        images = _call_with_supported_kwargs(
            module.load_and_preprocess_images,
            paths,
            mode=preprocess_mode,
            image_size=self.image_size,
            patch_size=self.patch_size,
        )
        return self.infer_tensor(images)

    def infer_loop_paths(
        self, image_paths: Iterable[os.PathLike | str], preprocess_mode: str = "crop"
    ) -> ChunkPrediction:
        return self.infer_paths(image_paths, preprocess_mode=preprocess_mode)

    def infer_tensor(self, images: Any) -> ChunkPrediction:
        images = _frame_tensor(images, self.device)
        image_size_hw = _image_size_hw(images)
        resolved_keyframe_interval = _resolve_keyframe_interval(
            self.keyframe_interval,
            int(images.shape[0]),
            self.auto_keyframe_threshold,
        )
        output_device = torch.device("cpu") if self.output_cpu else self.device
        if self.output_device is not None:
            output_device = self.output_device
        autocast = nullcontext()
        if self.device.type == "cuda" and self.inference_dtype != torch.float32:
            autocast = torch.autocast(device_type="cuda", dtype=self.inference_dtype)

        with torch.no_grad(), autocast:
            if self.mode == "streaming":
                output = self.model.inference_streaming(
                    images,
                    num_scale_frames=self.num_scale_frames,
                    keyframe_interval=resolved_keyframe_interval,
                    output_device=output_device,
                )
            else:
                output = self.model.inference_windowed(
                    images,
                    window_size=self.window_size,
                    overlap_size=self.overlap_size,
                    overlap_keyframes=self.overlap_keyframes,
                    num_scale_frames=self.num_scale_frames,
                    keyframe_interval=resolved_keyframe_interval,
                    flow_threshold=self.flow_threshold,
                    max_non_keyframe_gap=self.max_non_keyframe_gap,
                    output_device=output_device,
                )

        if not isinstance(output, Mapping):
            raise TypeError(f"LingBot-MAP model must return a mapping, got {type(output).__name__}")
        native_window_count = None
        if self.mode == "windowed":
            chunk_scales = output.get("chunk_scales")
            native_window_count = (
                int(chunk_scales.shape[1]) if chunk_scales is not None else 1
            )
            self._activation_counts["windowed:calls"] += 1
            self._activation_counts["windowed:native_windows"] += native_window_count
            if native_window_count > 1:
                self._activation_counts["windowed:multi_window_calls"] += 1
        prediction = self._prediction_from_output(output, image_size_hw, images)
        assert self._last_inference_report is not None
        self._last_inference_report.update(
            {
                "mode": self.mode,
                "num_frames": int(images.shape[0]),
                "resolved_keyframe_interval": resolved_keyframe_interval,
                "native_window_size": self.window_size if self.mode == "windowed" else None,
                "native_overlap_size": self.overlap_size if self.mode == "windowed" else None,
                "native_overlap_keyframes": (
                    self.overlap_keyframes if self.mode == "windowed" else None
                ),
                "native_window_count": native_window_count,
                "native_windowing_effective": bool(
                    native_window_count is not None and native_window_count > 1
                ),
            }
        )
        return prediction

    def activation_report(self) -> dict[str, Any]:
        """Return effective LingBot-MAP inference-path diagnostics."""
        return {
            "backend": "lingbot_map",
            "configured_geometry_source": self.geometry_source,
            "configured_pose_convention": self.pose_convention,
            "configured_mode": self.mode,
            "configured_keyframe_interval": self.keyframe_interval,
            "auto_keyframe_threshold": self.auto_keyframe_threshold,
            "effective_geometry_source_counts": {
                key.removeprefix("geometry:"): value
                for key, value in sorted(self._activation_counts.items())
                if key.startswith("geometry:")
            },
            "effective_confidence_source_counts": {
                key.removeprefix("confidence:"): value
                for key, value in sorted(self._activation_counts.items())
                if key.startswith("confidence:")
            },
            "inference_calls": sum(
                value
                for key, value in self._activation_counts.items()
                if key.startswith("geometry:")
            ),
            "windowed_calls": self._activation_counts["windowed:calls"],
            "windowed_multi_window_calls": self._activation_counts[
                "windowed:multi_window_calls"
            ],
            "windowed_native_windows": self._activation_counts[
                "windowed:native_windows"
            ],
            "last_inference": dict(self._last_inference_report or {}),
        }

    def _prediction_from_output(
        self,
        output: Mapping[str, Any],
        image_size_hw: Tuple[int, int],
        input_images: torch.Tensor,
    ) -> ChunkPrediction:
        if "pose_enc" not in output:
            raise KeyError("LingBot-MAP output is missing required key 'pose_enc'")

        pose_module = _import_from_repo(self.repo_root, "lingbot_map.utils.pose_enc")
        pose_enc = _to_cpu_tensor(output["pose_enc"], dtype=torch.float32)
        _check_finite("pose_enc", pose_enc)
        decoded_pose, intrinsic = pose_module.pose_encoding_to_extri_intri(pose_enc, image_size_hw)
        extrinsic_c2w = _squeeze_single_batch(
            "extrinsic_c2w",
            _decoded_pose_to_c2w(decoded_pose, self.pose_convention),
            3,
        )
        intrinsic = _squeeze_single_batch("intrinsic", intrinsic, 3).float()

        depth = _squeeze_depth(output["depth"]) if "depth" in output else None
        depth_conf = _squeeze_optional("depth_conf", output.get("depth_conf"), 3)
        native_world_points = _squeeze_optional("world_points", output.get("world_points"), 4)
        native_world_points_conf = _squeeze_optional(
            "world_points_conf", output.get("world_points_conf"), 3
        )

        use_native = self.geometry_source == "native" or (
            self.geometry_source == "auto" and native_world_points is not None
        )
        if use_native:
            if native_world_points is None:
                raise KeyError(
                    "LingBot-MAP geometry_source='native' requires output key "
                    "'world_points'"
                )
            world_points = native_world_points
            effective_geometry_source = "native_world_points"
        else:
            if depth is None:
                raise KeyError(
                    "LingBot-MAP depth-unprojection geometry requires output key 'depth'"
                )
            world_points = _unproject_depth_to_world_points(depth, intrinsic, extrinsic_c2w)
            effective_geometry_source = "depth_unprojection"

        if use_native and native_world_points_conf is not None:
            world_points_conf = native_world_points_conf
            effective_confidence_source = "native_world_points_conf"
        else:
            if depth_conf is None:
                missing = "world_points_conf" if use_native else "depth_conf"
                raise KeyError(
                    "LingBot-MAP selected geometry has no compatible confidence: "
                    f"missing {missing!r} and fallback 'depth_conf'"
                )
            world_points_conf = depth_conf
            effective_confidence_source = "depth_conf"

        images = _squeeze_optional("images", output.get("images"), 4)
        if images is None:
            images = _to_cpu_tensor(input_images)
        mask = _squeeze_optional("mask", output.get("mask"), 3)

        prediction = ChunkPrediction(
            world_points=world_points.float().cpu(),
            world_points_conf=world_points_conf.float().cpu(),
            extrinsic_c2w=extrinsic_c2w.float().cpu(),
            intrinsic=intrinsic.float().cpu(),
            depth=depth.float().cpu() if depth is not None else None,
            depth_conf=depth_conf.float().cpu() if depth_conf is not None else None,
            images=images.float().cpu(),
            mask=mask.cpu() if mask is not None else None,
        )
        self._activation_counts[f"geometry:{effective_geometry_source}"] += 1
        self._activation_counts[f"confidence:{effective_confidence_source}"] += 1
        self._last_inference_report = {
            "effective_geometry_source": effective_geometry_source,
            "effective_confidence_source": effective_confidence_source,
            "native_world_points_available": native_world_points is not None,
            "native_world_points_conf_available": native_world_points_conf is not None,
            "depth_available": depth is not None,
            "depth_conf_available": depth_conf is not None,
            "pose_source": (
                "pose_enc_decoded_c2w"
                if self.pose_convention == "decoded_c2w"
                else "pose_enc_decoded_w2c_then_inverse"
            ),
        }
        return prediction


def _load_checkpoint_object(source: Any, map_location: Any):
    if isinstance(source, Mapping):
        return source, "mapping"

    path = os.fspath(source)
    if os.path.exists(path):
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(path, device=str(map_location)), "safetensors_file"
        try:
            return torch.load(path, map_location=map_location, weights_only=False), "file"
        except TypeError:
            return torch.load(path, map_location=map_location), "file"

    raise FileNotFoundError(f"checkpoint path does not exist: {path}")


def _extract_state_dict(checkpoint: Mapping[str, Any]) -> Tuple[Mapping[str, Any], str]:
    if "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
        return checkpoint["model"], "model_state_dict"
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        return checkpoint["state_dict"], "state_dict"
    return checkpoint, "plain_state_dict"


def _strip_prefix(state_dict: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    if not prefix:
        return dict(state_dict)
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped[key[len(prefix) :]] = value
        else:
            stripped[key] = value
    return stripped


def _best_state_dict_for_model(state_dict: Mapping[str, Any], model_keys: set[str]):
    candidates = ("", "net.", "model.", "module.", "model.net.", "model.module.")
    best_prefix = ""
    best_state_dict = dict(state_dict)
    best_matches = len(set(best_state_dict).intersection(model_keys))
    for prefix in candidates[1:]:
        candidate = _strip_prefix(state_dict, prefix)
        matches = len(set(candidate).intersection(model_keys))
        if matches > best_matches:
            best_prefix = prefix
            best_state_dict = candidate
            best_matches = matches
    return best_state_dict, best_prefix, best_matches


def load_lingbot_map_checkpoint(
    model: torch.nn.Module,
    checkpoint: Any,
    *,
    strict: bool = False,
    map_location: Any = "cpu",
) -> Dict[str, Any]:
    """Load a LingBot-MAP checkpoint and return a structured report."""
    checkpoint_obj, source_type = _load_checkpoint_object(checkpoint, map_location)
    if not isinstance(checkpoint_obj, Mapping):
        raise TypeError(f"checkpoint must resolve to a mapping, got {type(checkpoint_obj).__name__}")

    raw_state_dict, state_dict_type = _extract_state_dict(checkpoint_obj)
    model_keys = set(model.state_dict().keys())
    normalized_state_dict, stripped_prefix, matched_keys = _best_state_dict_for_model(raw_state_dict, model_keys)
    if matched_keys == 0:
        raise RuntimeError("checkpoint has zero keys matching the LingBot-MAP model state_dict")

    incompatible = model.load_state_dict(normalized_state_dict, strict=strict)
    missing_keys = list(getattr(incompatible, "missing_keys", incompatible[0] if incompatible else []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", incompatible[1] if incompatible else []))
    return {
        "source_type": source_type,
        "state_dict_type": state_dict_type,
        "total_keys": len(raw_state_dict),
        "matched_keys": matched_keys,
        "stripped_prefix": stripped_prefix,
        "strict": strict,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }
