"""VGGT-Omega adapter for local chunk predictions."""

from __future__ import annotations

import importlib
import json
import multiprocessing
import os
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import torch

from long_reconstruction.contract import ChunkPrediction


# Bundled base models live in ``src``.
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]

_DTYPE_ALIASES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}

_CPU_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _dtype_from_spec(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        if dtype not in {torch.float32, torch.bfloat16, torch.float16}:
            raise ValueError(f"unsupported inference dtype: {dtype}")
        return dtype
    key = str(dtype).lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"unsupported inference dtype: {dtype}")
    return _DTYPE_ALIASES[key]


def _first_parameter_device(model: Any) -> torch.device:
    if hasattr(model, "parameters"):
        try:
            return next(model.parameters()).device
        except StopIteration:
            pass
    return torch.device("cpu")


@contextmanager
def _repo_on_path(repo_root: Path):
    root = str(repo_root)
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
    with _repo_on_path(repo_root):
        return importlib.import_module(module_name)


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


def _squeeze_optional(name: str, value: Any, expected_unbatched_ndim: int):
    if value is None:
        return None
    return _squeeze_single_batch(name, value, expected_unbatched_ndim)


def _squeeze_depth(depth: Any) -> torch.Tensor:
    tensor = _to_cpu_tensor(depth)
    if tensor.dim() == 5:
        if tensor.shape[0] != 1:
            raise ValueError(f"depth batch dimension must be 1 for chunk predictions, got {tensor.shape[0]}")
        tensor = tensor[0]
    elif tensor.dim() == 4 and tensor.shape[-1] != 1:
        if tensor.shape[0] != 1:
            raise ValueError(f"depth has unsupported shape {tuple(tensor.shape)}")
        tensor = tensor[0]
    elif tensor.dim() not in (3, 4):
        raise ValueError(f"depth has unsupported shape {tuple(tensor.shape)}")
    if tensor.dim() == 4:
        if tensor.shape[-1] != 1:
            raise ValueError(f"depth has unsupported shape {tuple(tensor.shape)}")
        tensor = tensor[..., 0]
    _check_finite("depth", tensor)
    return tensor


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


def _load_checkpoint_object(
    source: Any,
    map_location: Any,
    *,
    checkpoint_mmap: bool = False,
):
    if isinstance(source, Mapping):
        return source, "mapping"

    path = os.fspath(source)
    if os.path.exists(path):
        if path.endswith(".safetensors"):
            if checkpoint_mmap:
                raise ValueError(
                    "checkpoint_mmap is only supported for PyTorch zip checkpoints"
                )
            from safetensors.torch import load_file

            return load_file(path, device=str(map_location)), "safetensors_file"
        try:
            return (
                torch.load(
                    path,
                    map_location=map_location,
                    weights_only=False,
                    mmap=checkpoint_mmap,
                ),
                "file_mmap" if checkpoint_mmap else "file",
            )
        except TypeError as exc:
            if checkpoint_mmap:
                raise RuntimeError(
                    "checkpoint mmap was requested but this torch.load does not support it"
                ) from exc
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
    candidates = (
        "",
        "net.model.",
        "model.net.model.",
        "model.",
        "module.",
        "model.module.",
        "net.",
        "model.net.",
    )
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


def _load_vggt_omega_checkpoint_from_object(
    model: torch.nn.Module,
    checkpoint_obj: Mapping[str, Any],
    *,
    source_type: str,
    strict: bool,
) -> Dict[str, Any]:
    if not isinstance(checkpoint_obj, Mapping):
        raise TypeError(f"checkpoint must resolve to a mapping, got {type(checkpoint_obj).__name__}")

    raw_state_dict, state_dict_type = _extract_state_dict(checkpoint_obj)
    model_keys = set(model.state_dict().keys())
    normalized_state_dict, stripped_prefix, matched_keys = _best_state_dict_for_model(raw_state_dict, model_keys)
    if matched_keys == 0:
        raise RuntimeError("checkpoint has zero keys matching the VGGT-Omega model state_dict")

    incompatible = model.load_state_dict(normalized_state_dict, strict=strict)
    missing_keys = list(getattr(incompatible, "missing_keys", incompatible[0] if incompatible else []))
    unexpected_keys = list(getattr(incompatible, "unexpected_keys", incompatible[1] if incompatible else []))
    return {
        "backend": "vggt_omega",
        "source_type": source_type,
        "state_dict_type": state_dict_type,
        "total_keys": len(raw_state_dict),
        "matched_keys": matched_keys,
        "stripped_prefix": stripped_prefix,
        "strict": strict,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


def load_vggt_omega_checkpoint(
    model: torch.nn.Module,
    checkpoint: Any,
    *,
    strict: bool = False,
    map_location: Any = "cpu",
    checkpoint_mmap: bool = False,
) -> Dict[str, Any]:
    """Load a VGGT-Omega checkpoint and return a structured report."""
    checkpoint_obj, source_type = _load_checkpoint_object(
        checkpoint,
        map_location,
        checkpoint_mmap=checkpoint_mmap,
    )
    return _load_vggt_omega_checkpoint_from_object(
        model,
        checkpoint_obj,
        source_type=source_type,
        strict=strict,
    )


@dataclass(frozen=True)
class VggtOmegaAdapterSpec:
    """Spawn-safe construction arguments for a lazily loaded adapter."""

    checkpoint: Optional[str] = None
    repo_root: str = os.fspath(DEFAULT_REPO_ROOT)
    inference_dtype: str = "bfloat16"
    image_resolution: int = 512
    patch_size: int = 16
    preprocess_mode: str = "balanced"
    checkpoint_strict: bool = False
    map_location: str = "cpu"
    checkpoint_mmap: bool = False
    checkpoint_direct_device_init: bool = False
    cpu_threads_per_worker: Optional[int] = None

    def __post_init__(self) -> None:
        if self.checkpoint is not None:
            try:
                object.__setattr__(self, "checkpoint", os.fspath(self.checkpoint))
            except TypeError:
                raise TypeError("checkpoint must be a path or None in a spawn-safe adapter spec") from None
        try:
            object.__setattr__(self, "repo_root", os.fspath(self.repo_root))
        except TypeError:
            raise TypeError("repo_root must be a path in a spawn-safe adapter spec") from None
        for name in ("inference_dtype", "preprocess_mode", "map_location"):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f"{name} must be a string in a spawn-safe adapter spec")
        if self.image_resolution <= 0 or self.patch_size <= 0:
            raise ValueError("image_resolution and patch_size must be positive")
        if self.cpu_threads_per_worker is not None and self.cpu_threads_per_worker < 1:
            raise ValueError("cpu_threads_per_worker must be positive when configured")
        _dtype_from_spec(self.inference_dtype)

    def create(self, device: Any = None) -> "VggtOmegaChunkAdapter":
        """Construct and load an adapter on ``device``."""
        kwargs = asdict(self)
        kwargs.pop("cpu_threads_per_worker")
        return VggtOmegaChunkAdapter(device=device, **kwargs)


class LazyVggtOmegaChunkAdapter:
    """Delay model and checkpoint loading until the first inference call."""

    def __init__(self, spec: VggtOmegaAdapterSpec, device: Any = None) -> None:
        if not isinstance(spec, VggtOmegaAdapterSpec):
            raise TypeError("spec must be a VggtOmegaAdapterSpec")
        self.spec = spec
        self.device = device
        self._adapter: Optional[VggtOmegaChunkAdapter] = None

    @property
    def loaded(self) -> bool:
        return self._adapter is not None

    def load(self) -> "VggtOmegaChunkAdapter":
        if self._adapter is None:
            self._adapter = self.spec.create(self.device)
        return self._adapter

    def infer_paths(self, image_paths: Iterable[os.PathLike | str], preprocess_mode: Optional[str] = None):
        return self.load().infer_paths(image_paths, preprocess_mode=preprocess_mode)

    def infer_loop_paths(self, image_paths: Iterable[os.PathLike | str], preprocess_mode: Optional[str] = None):
        return self.infer_paths(image_paths, preprocess_mode=preprocess_mode)

    def infer_tensor(self, images: Any):
        return self.load().infer_tensor(images)

    def infer_chunks_parallel(
        self,
        chunks: Iterable[Iterable[os.PathLike | str]],
        *,
        gpu_ids: Sequence[int],
        result_root: Optional[os.PathLike | str] = None,
        progress: Any = None,
    ):
        chunk_list = [tuple(paths) for paths in chunks]
        devices = tuple(f"cuda:{int(gpu_id)}" for gpu_id in gpu_ids)
        temp_kwargs = {}
        if result_root is not None:
            root = Path(result_root)
            root.mkdir(parents=True, exist_ok=True)
            temp_kwargs["dir"] = str(root)
        with tempfile.TemporaryDirectory(prefix="vggt-omega-run-", **temp_kwargs) as exchange:
            if progress is not None:
                progress(
                    f"dispatching {len(chunk_list)} local chunks to devices={list(devices)}"
                )
            predictions = infer_chunks_parallel(
                chunk_list,
                self.spec,
                devices=devices,
                exchange_dir=exchange,
            )
        assignments = [
            {
                "chunk_index": index,
                "gpu_id": int(gpu_ids[index % min(len(gpu_ids), len(chunk_list))]),
            }
            for index in range(len(chunk_list))
        ]
        return predictions, {
            "exchange": "atomic_disk",
            "chunk_assignments": assignments,
        }

    def infer_chunks_parallel_to_artifacts(
        self,
        chunks: Iterable[Iterable[os.PathLike | str]],
        *,
        gpu_ids: Sequence[int],
        artifact_root: os.PathLike | str,
        artifact_relative_paths: Sequence[str],
        artifact_version: int,
        artifact_field_names: Sequence[str],
        artifact_field_dtype_overrides: Mapping[str, Any] | None = None,
        result_root: Optional[os.PathLike | str] = None,
        progress: Any = None,
    ):
        chunk_list = [tuple(paths) for paths in chunks]
        devices = tuple(f"cuda:{int(gpu_id)}" for gpu_id in gpu_ids)
        temp_kwargs = {}
        if result_root is not None:
            root = Path(result_root)
            root.mkdir(parents=True, exist_ok=True)
            temp_kwargs["dir"] = str(root)
        with tempfile.TemporaryDirectory(prefix="vggt-omega-run-", **temp_kwargs) as exchange:
            if progress is not None:
                progress(
                    "dispatching direct artifact publication for "
                    f"{len(chunk_list)} chunks to devices={list(devices)}"
                )
            entries = infer_chunks_parallel(
                chunk_list,
                self.spec,
                devices=devices,
                exchange_dir=exchange,
                artifact_root=artifact_root,
                artifact_relative_paths=artifact_relative_paths,
                artifact_version=artifact_version,
                artifact_field_names=artifact_field_names,
                artifact_field_dtype_overrides=artifact_field_dtype_overrides,
            )
            metadata_exchange_bytes = sum(
                path.stat().st_size
                for path in Path(exchange).glob("*.artifact.json")
            )
        assignments = [
            {
                "chunk_index": index,
                "gpu_id": int(gpu_ids[index % min(len(gpu_ids), len(chunk_list))]),
            }
            for index in range(len(chunk_list))
        ]
        return entries, {
            "exchange": "metadata_only_atomic_json",
            "parent_materialized_predictions": False,
            "artifact_publication_owner": "gpu_worker",
            "metadata_exchange_bytes": metadata_exchange_bytes,
            "chunk_assignments": assignments,
        }

    def infer_forward_phases_parallel_to_artifacts(
        self,
        chunk_paths: Iterable[Iterable[os.PathLike | str]],
        loop_paths: Iterable[Iterable[os.PathLike | str]],
        *,
        gpu_ids: Sequence[int],
        artifact_root: os.PathLike | str,
        chunk_relative_paths: Sequence[str],
        loop_relative_paths: Sequence[str],
        artifact_version: int,
        chunk_field_names: Sequence[str],
        loop_field_names: Sequence[str],
        artifact_field_dtype_overrides: Mapping[str, Any] | None = None,
        result_root: Optional[os.PathLike | str] = None,
        progress: Any = None,
    ):
        """Publish chunk and loop phases with one model load per GPU worker."""
        chunks = [tuple(paths) for paths in chunk_paths]
        loops = [tuple(paths) for paths in loop_paths]
        devices = tuple(f"cuda:{int(gpu_id)}" for gpu_id in gpu_ids)
        temp_kwargs = {}
        if result_root is not None:
            result_path = Path(result_root)
            result_path.mkdir(parents=True, exist_ok=True)
            temp_kwargs["dir"] = str(result_path)
        with tempfile.TemporaryDirectory(
            prefix="vggt-omega-forward-phases-", **temp_kwargs
        ) as exchange:
            if progress is not None:
                progress(
                    "dispatching persistent direct artifact publication for "
                    f"chunks={len(chunks)} loops={len(loops)} "
                    f"devices={list(devices)}"
                )
            chunk_entries, loop_entries, diagnostics = (
                _infer_forward_phases_parallel_to_artifacts(
                    chunks,
                    loops,
                    self.spec,
                    devices=devices,
                    exchange_dir=exchange,
                    artifact_root=artifact_root,
                    chunk_relative_paths=chunk_relative_paths,
                    loop_relative_paths=loop_relative_paths,
                    artifact_version=artifact_version,
                    chunk_field_names=chunk_field_names,
                    loop_field_names=loop_field_names,
                    artifact_field_dtype_overrides=artifact_field_dtype_overrides,
                )
            )
        diagnostics["chunk_assignments"] = [
            {
                "chunk_index": index,
                "gpu_id": int(gpu_ids[index % len(devices)]),
            }
            for index in range(len(chunks))
        ]
        diagnostics["loop_assignments"] = [
            {
                "loop_index": index,
                "gpu_id": int(gpu_ids[index % len(devices)]),
            }
            for index in range(len(loops))
        ]
        return chunk_entries, loop_entries, diagnostics


class VggtOmegaChunkAdapter:
    """Run VGGT-Omega on one chunk and return the HiP3R local prediction contract."""

    def __init__(
        self,
        model: Any = None,
        *,
        repo_root: Any = None,
        checkpoint: Any = None,
        device: Any = None,
        inference_dtype: Any = "bfloat16",
        image_resolution: int = 512,
        patch_size: int = 16,
        preprocess_mode: str = "balanced",
        checkpoint_strict: bool = False,
        map_location: Any = "cpu",
        checkpoint_mmap: bool = False,
        checkpoint_direct_device_init: bool = False,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else DEFAULT_REPO_ROOT
        self.image_resolution = int(image_resolution)
        self.patch_size = int(patch_size)
        self.preprocess_mode = str(preprocess_mode)
        self.inference_dtype = _dtype_from_spec(inference_dtype)
        requested_device = torch.device(device) if device is not None else None
        if checkpoint_direct_device_init:
            if model is not None:
                raise ValueError(
                    "checkpoint_direct_device_init requires adapter-owned model construction"
                )
            if checkpoint is None:
                raise ValueError(
                    "checkpoint_direct_device_init requires a checkpoint"
                )
            if not checkpoint_mmap:
                raise ValueError(
                    "checkpoint_direct_device_init requires checkpoint_mmap"
                )
            if requested_device is None or requested_device.type != "cuda":
                raise ValueError(
                    "checkpoint_direct_device_init requires an explicit CUDA device"
                )
            if torch.device(map_location).type != "cpu":
                raise ValueError(
                    "checkpoint_direct_device_init requires CPU checkpoint map_location"
                )

        checkpoint_obj = None
        checkpoint_source_type = None
        adapter_config = {"dense_adapter_enabled": False}
        if model is None and checkpoint is not None:
            checkpoint_obj, checkpoint_source_type = _load_checkpoint_object(
                checkpoint,
                map_location,
                checkpoint_mmap=checkpoint_mmap,
            )
            if not isinstance(checkpoint_obj, Mapping):
                raise TypeError(f"checkpoint must resolve to a mapping, got {type(checkpoint_obj).__name__}")
            raw_state_dict, _ = _extract_state_dict(checkpoint_obj)
            adapters_module = _import_from_repo(self.repo_root, "vggt_omega.models.adapters")
            adapter_config = adapters_module.infer_dense_adapter_config_from_state_dict(raw_state_dict)
            if checkpoint_direct_device_init and checkpoint_source_type != "file_mmap":
                raise ValueError(
                    "checkpoint_direct_device_init requires an effective mmap PyTorch file"
                )

        if model is None:
            module = _import_from_repo(self.repo_root, "vggt_omega.models")
            if checkpoint_direct_device_init:
                with torch.device(requested_device):
                    model = module.VGGTOmega(**adapter_config)
                model_initialization_mode = "target_cuda_before_checkpoint_load"
            else:
                model = module.VGGTOmega(**adapter_config)
                model_initialization_mode = "default_device_then_move"
        else:
            model_initialization_mode = "provided_model"
        self.model = model
        self.device = requested_device if requested_device is not None else _first_parameter_device(model)

        if checkpoint is not None:
            if checkpoint_obj is None:
                self.load_report = load_vggt_omega_checkpoint(
                    self.model,
                    checkpoint,
                    strict=checkpoint_strict,
                    map_location=map_location,
                    checkpoint_mmap=checkpoint_mmap,
                )
            else:
                self.load_report = _load_vggt_omega_checkpoint_from_object(
                    self.model,
                    checkpoint_obj,
                    source_type=checkpoint_source_type,
                    strict=checkpoint_strict,
                )
        else:
            self.load_report = {
                "backend": "vggt_omega",
                "source_type": "none",
                "state_dict_type": "none",
                "total_keys": 0,
                "matched_keys": 0,
                "stripped_prefix": "",
                "strict": checkpoint_strict,
                "missing_keys": [],
                "unexpected_keys": [],
            }
        self.load_report["model_initialization_mode"] = model_initialization_mode

        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def infer_paths(self, image_paths: Iterable[os.PathLike | str], preprocess_mode: Optional[str] = None) -> ChunkPrediction:
        paths = [os.fspath(path) for path in image_paths]
        if not paths:
            raise ValueError("image_paths is empty")
        module = _import_from_repo(self.repo_root, "vggt_omega.utils.load_fn")
        images = module.load_and_preprocess_images(
            paths,
            mode=preprocess_mode or self.preprocess_mode,
            image_resolution=self.image_resolution,
            patch_size=self.patch_size,
        )
        return self.infer_tensor(images)

    def infer_loop_paths(
        self, image_paths: Iterable[os.PathLike | str], preprocess_mode: Optional[str] = None
    ) -> ChunkPrediction:
        return self.infer_paths(image_paths, preprocess_mode=preprocess_mode)

    def infer_tensor(self, images: Any) -> ChunkPrediction:
        images = _frame_tensor(images, self.device)
        image_size_hw = _image_size_hw(images)
        autocast = nullcontext()
        if self.device.type == "cuda" and self.inference_dtype != torch.float32:
            autocast = torch.autocast(device_type="cuda", dtype=self.inference_dtype)
        with torch.no_grad(), autocast:
            output = self.model(images)
        if not isinstance(output, Mapping):
            raise TypeError(f"VGGT-Omega model must return a mapping, got {type(output).__name__}")
        return self._prediction_from_output(output, image_size_hw, images)

    def _prediction_from_output(
        self,
        output: Mapping[str, Any],
        image_size_hw: Tuple[int, int],
        input_images: torch.Tensor,
    ) -> ChunkPrediction:
        if "pose_enc" not in output:
            raise KeyError("VGGT-Omega output is missing required key 'pose_enc'")

        pose_module = _import_from_repo(self.repo_root, "vggt_omega.utils.pose_enc")
        pose_enc = _to_cpu_tensor(output["pose_enc"], dtype=torch.float32)
        _check_finite("pose_enc", pose_enc)
        extrinsic_w2c, intrinsic = pose_module.pose_encoding_to_extri_intri(pose_enc, image_size_hw)
        extrinsic_c2w = _squeeze_single_batch("extrinsic_c2w", _w2c_to_c2w(extrinsic_w2c), 3)
        intrinsic = _squeeze_single_batch("intrinsic", intrinsic, 3).float()

        depth = _squeeze_depth(output["depth"]) if "depth" in output else None
        depth_conf = _squeeze_optional("depth_conf", output.get("depth_conf"), 3)
        world_points = _squeeze_optional("world_points", output.get("world_points"), 4)
        world_points_conf = _squeeze_optional("world_points_conf", output.get("world_points_conf"), 3)

        if world_points is None:
            if depth is None:
                raise KeyError("VGGT-Omega output is missing 'world_points' and fallback 'depth'")
            world_points = _unproject_depth_to_world_points(depth, intrinsic, extrinsic_c2w)

        if world_points_conf is None:
            if depth_conf is None:
                raise KeyError("VGGT-Omega output is missing 'world_points_conf' and fallback 'depth_conf'")
            world_points_conf = depth_conf

        images = _squeeze_optional("images", output.get("images"), 4)
        if images is None:
            images = _to_cpu_tensor(input_images)
        mask = _squeeze_optional("mask", output.get("mask"), 3)

        return ChunkPrediction(
            world_points=world_points.float().cpu(),
            world_points_conf=world_points_conf.float().cpu(),
            extrinsic_c2w=extrinsic_c2w.float().cpu(),
            intrinsic=intrinsic.float().cpu(),
            depth=depth.float().cpu() if depth is not None else None,
            depth_conf=depth_conf.float().cpu() if depth_conf is not None else None,
            images=images.float().cpu(),
            mask=mask.cpu() if mask is not None else None,
        )


def _atomic_torch_save(value: Any, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_json_save(value: Mapping[str, Any], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(value), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parallel_chunk_worker(
    spec: VggtOmegaAdapterSpec,
    device: str,
    indexed_chunks: Sequence[Tuple[int, Tuple[str, ...]]],
    exchange_dir: str,
    artifact_spec: Optional[Mapping[str, Any]] = None,
) -> None:
    exchange = Path(exchange_dir)
    try:
        adapter = LazyVggtOmegaChunkAdapter(spec, device=device)
        for index, paths in indexed_chunks:
            try:
                prediction = adapter.infer_paths(paths)
                if artifact_spec is None:
                    _atomic_torch_save(prediction, exchange / f"{index:08d}.pt")
                else:
                    from long_reconstruction.runner import (
                        _write_prediction_artifact,
                    )

                    relative_paths = artifact_spec["relative_paths"]
                    entry = _write_prediction_artifact(
                        Path(artifact_spec["root"]),
                        str(relative_paths[index]),
                        prediction,
                        version=int(artifact_spec["version"]),
                        field_names=tuple(artifact_spec["field_names"]),
                        field_dtype_overrides=artifact_spec.get(
                            "field_dtype_overrides"
                        ),
                    )
                    _atomic_json_save(
                        {"index": index, "entry": entry},
                        exchange / f"{index:08d}.artifact.json",
                    )
                del prediction
            except BaseException as exc:
                _atomic_json_save(
                    {
                        "index": index,
                        "device": device,
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    exchange / f"{index:08d}.error.json",
                )
                raise
    except BaseException:
        raise


def _parallel_forward_phases_worker(
    spec: VggtOmegaAdapterSpec,
    device: str,
    worker_index: int,
    phase_assignments: Sequence[
        Tuple[str, Sequence[Tuple[int, Tuple[str, ...], str, Tuple[str, ...]]]]
    ],
    exchange_dir: str,
    artifact_root: str,
    artifact_version: int,
    artifact_field_dtype_overrides: Mapping[str, Any],
) -> None:
    exchange = Path(exchange_dir)
    phase_timings: dict[str, dict[str, float | int]] = {}
    try:
        if spec.cpu_threads_per_worker is not None:
            torch.set_num_threads(spec.cpu_threads_per_worker)
        adapter = LazyVggtOmegaChunkAdapter(spec, device=device)
        for phase_name, tasks in phase_assignments:
            phase_started = time.perf_counter()
            for index, paths, relative_path, field_names in tasks:
                try:
                    prediction = adapter.infer_paths(paths)
                    from long_reconstruction.runner import (
                        _write_prediction_artifact,
                    )

                    entry = _write_prediction_artifact(
                        Path(artifact_root),
                        relative_path,
                        prediction,
                        version=artifact_version,
                        field_names=field_names,
                        field_dtype_overrides=artifact_field_dtype_overrides,
                    )
                    _atomic_json_save(
                        {"index": index, "entry": entry},
                        exchange / f"{phase_name}.{index:08d}.artifact.json",
                    )
                    del prediction
                except BaseException as exc:
                    _atomic_json_save(
                        {
                            "phase": phase_name,
                            "index": index,
                            "device": device,
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                        exchange / f"{phase_name}.{index:08d}.error.json",
                    )
                    raise
            phase_timings[phase_name] = {
                "started": phase_started,
                "ended": time.perf_counter(),
                "tasks": len(tasks),
            }
        _atomic_json_save(
            {
                "worker_index": worker_index,
                "device": device,
                "model_loads": int(bool(sum(item["tasks"] for item in phase_timings.values()))),
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "cpu_thread_environment": {
                    key: os.environ.get(key) for key in _CPU_THREAD_ENV_KEYS
                },
                "checkpoint_source_type": adapter.load().load_report.get(
                    "source_type", "unknown"
                ),
                "model_initialization_mode": adapter.load().load_report.get(
                    "model_initialization_mode", "unknown"
                ),
                "phases": phase_timings,
            },
            exchange / f"worker.{worker_index:04d}.timing.json",
        )
    except BaseException:
        raise


def _infer_forward_phases_parallel_to_artifacts(
    chunk_paths: Sequence[Tuple[os.PathLike | str, ...]],
    loop_paths: Sequence[Tuple[os.PathLike | str, ...]],
    spec: VggtOmegaAdapterSpec,
    *,
    devices: Sequence[Any],
    exchange_dir: os.PathLike | str,
    artifact_root: os.PathLike | str,
    chunk_relative_paths: Sequence[str],
    loop_relative_paths: Sequence[str],
    artifact_version: int,
    chunk_field_names: Sequence[str],
    loop_field_names: Sequence[str],
    artifact_field_dtype_overrides: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run two forward phases in the same spawned GPU workers."""
    if not isinstance(spec, VggtOmegaAdapterSpec):
        raise TypeError("spec must be a VggtOmegaAdapterSpec")
    device_names = tuple(str(device) for device in devices)
    if not device_names:
        raise ValueError("devices must contain at least one device")
    if len(set(device_names)) != len(device_names):
        raise ValueError("devices must not contain duplicates")
    if int(artifact_version) != 2:
        raise ValueError("persistent direct publication requires artifact_version=2")
    phases = (
        (
            "chunk",
            tuple(tuple(os.fspath(path) for path in paths) for paths in chunk_paths),
            tuple(str(path) for path in chunk_relative_paths),
            tuple(str(name) for name in chunk_field_names),
        ),
        (
            "loop",
            tuple(tuple(os.fspath(path) for path in paths) for paths in loop_paths),
            tuple(str(path) for path in loop_relative_paths),
            tuple(str(name) for name in loop_field_names),
        ),
    )
    if not chunk_paths or not loop_paths:
        raise ValueError("persistent forward requires non-empty chunk and loop phases")
    all_relative_paths: list[str] = []
    for phase_name, paths, relative_paths, field_names in phases:
        if len(paths) != len(relative_paths):
            raise ValueError(f"{phase_name}_relative_paths must match task count")
        if any(not item for item in paths):
            raise ValueError(f"{phase_name} tasks must contain image paths")
        if not field_names:
            raise ValueError(f"{phase_name}_field_names must not be empty")
        for relative_path in relative_paths:
            parsed = Path(relative_path)
            if parsed.is_absolute() or ".." in parsed.parts:
                raise ValueError("artifact relative paths must stay below artifact_root")
        all_relative_paths.extend(relative_paths)
    if len(set(all_relative_paths)) != len(all_relative_paths):
        raise ValueError("artifact relative paths must not contain duplicates")

    worker_count = min(len(device_names), max(len(item[1]) for item in phases))
    exchange = Path(exchange_dir)
    exchange.mkdir(parents=True, exist_ok=True)
    per_worker: list[list[tuple[str, list[tuple[int, Tuple[str, ...], str, Tuple[str, ...]]]]]] = [
        [] for _ in range(worker_count)
    ]
    for phase_name, paths, relative_paths, field_names in phases:
        indexed = [
            (index, paths[index], relative_paths[index], field_names)
            for index in range(len(paths))
        ]
        for worker in range(worker_count):
            per_worker[worker].append((phase_name, indexed[worker::worker_count]))

    context = multiprocessing.get_context("spawn")
    processes = []
    original_thread_environment = {
        key: os.environ.get(key) for key in _CPU_THREAD_ENV_KEYS
    }
    try:
        if spec.cpu_threads_per_worker is not None:
            for key in _CPU_THREAD_ENV_KEYS:
                os.environ[key] = str(spec.cpu_threads_per_worker)
        try:
            for worker in range(worker_count):
                process = context.Process(
                    target=_parallel_forward_phases_worker,
                    args=(
                        spec,
                        device_names[worker],
                        worker,
                        per_worker[worker],
                        str(exchange),
                        os.fspath(artifact_root),
                        int(artifact_version),
                        dict(artifact_field_dtype_overrides or {}),
                    ),
                    name=f"vggt-omega-forward-{worker}",
                )
                process.start()
                processes.append(process)
        finally:
            for key, value in original_thread_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        for process in processes:
            process.join()

        errors = []
        for error_path in sorted(exchange.glob("*.error.json")):
            try:
                errors.append(json.loads(error_path.read_text(encoding="utf-8")))
            except Exception as exc:
                errors.append({"index": "unknown", "message": f"could not read {error_path}: {exc}"})
        failed = [(process.name, process.exitcode) for process in processes if process.exitcode != 0]
        if errors or failed:
            details = "; ".join(
                f"{item.get('phase', 'unknown')} {item.get('index')} on "
                f"{item.get('device', 'unknown')}: {item.get('type', 'Error')}: "
                f"{item.get('message')}"
                for item in errors
            )
            raise RuntimeError(
                f"persistent VGGT-Omega forward failed ({failed}): "
                f"{details or 'worker exited'}"
            )

        phase_entries: dict[str, list[dict[str, Any]]] = {}
        for phase_name, paths, _, _ in phases:
            entries = []
            for index in range(len(paths)):
                result_path = exchange / f"{phase_name}.{index:08d}.artifact.json"
                if not result_path.is_file():
                    raise RuntimeError(
                        f"persistent forward produced no {phase_name} metadata for {index}"
                    )
                envelope = json.loads(result_path.read_text(encoding="utf-8"))
                if int(envelope.get("index", -1)) != index:
                    raise RuntimeError(
                        f"persistent {phase_name} metadata order mismatch: "
                        f"expected {index}, got {envelope.get('index')}"
                    )
                entries.append(envelope["entry"])
            phase_entries[phase_name] = entries

        worker_timings = [
            json.loads((exchange / f"worker.{worker:04d}.timing.json").read_text(encoding="utf-8"))
            for worker in range(worker_count)
        ]
        phase_intervals = {}
        for phase_name, _, _, _ in phases:
            timings = [
                item["phases"][phase_name]
                for item in worker_timings
                if int(item["phases"][phase_name]["tasks"]) > 0
            ]
            phase_intervals[phase_name] = {
                "started": min(float(item["started"]) for item in timings),
                "ended": max(float(item["ended"]) for item in timings),
            }
        overlap_seconds = max(
            0.0,
            min(phase_intervals["chunk"]["ended"], phase_intervals["loop"]["ended"])
            - max(phase_intervals["chunk"]["started"], phase_intervals["loop"]["started"]),
        )
        metadata_exchange_bytes = sum(
            path.stat().st_size for path in exchange.glob("*.json")
        )
        model_loads = sum(int(item["model_loads"]) for item in worker_timings)
        checkpoint_source_types = sorted(
            {str(item["checkpoint_source_type"]) for item in worker_timings}
        )
        model_initialization_modes = sorted(
            {str(item["model_initialization_mode"]) for item in worker_timings}
        )
        torch_num_threads = sorted(
            {int(item["torch_num_threads"]) for item in worker_timings}
        )
        torch_num_interop_threads = sorted(
            {int(item["torch_num_interop_threads"]) for item in worker_timings}
        )
        cpu_thread_environment = {
            key: sorted(
                {
                    item["cpu_thread_environment"].get(key)
                    for item in worker_timings
                    if item["cpu_thread_environment"].get(key) is not None
                }
            )
            for key in _CPU_THREAD_ENV_KEYS
        }
        separate_phase_model_loads = sum(
            min(len(device_names), len(paths)) for _, paths, _, _ in phases
        )
        return phase_entries["chunk"], phase_entries["loop"], {
            "mode": "persistent_parallel_worker_direct_artifact_v1",
            "exchange": "metadata_only_atomic_json",
            "parent_materialized_predictions": False,
            "artifact_publication_owner": "gpu_worker",
            "actual_workers": worker_count,
            "model_loads": model_loads,
            "checkpoint_mmap_configured": bool(spec.checkpoint_mmap),
            "checkpoint_mmap_effective": (
                checkpoint_source_types == ["file_mmap"]
            ),
            "checkpoint_source_types": checkpoint_source_types,
            "checkpoint_direct_device_init_configured": bool(
                spec.checkpoint_direct_device_init
            ),
            "checkpoint_direct_device_init_effective": (
                model_initialization_modes == ["target_cuda_before_checkpoint_load"]
            ),
            "checkpoint_model_initialization_modes": model_initialization_modes,
            "torch_num_threads": torch_num_threads,
            "torch_num_interop_threads": torch_num_interop_threads,
            "cpu_thread_environment": cpu_thread_environment,
            "cpu_threads_per_worker_configured": spec.cpu_threads_per_worker,
            "cpu_threads_per_worker_effective": (
                spec.cpu_threads_per_worker is not None
                and torch_num_threads == [spec.cpu_threads_per_worker]
                and all(
                    cpu_thread_environment[key]
                    == [str(spec.cpu_threads_per_worker)]
                    for key in _CPU_THREAD_ENV_KEYS
                )
            ),
            "separate_loop_model_loads_avoided": (
                separate_phase_model_loads - model_loads
            ),
            "chunk_wall_seconds": (
                phase_intervals["chunk"]["ended"] - phase_intervals["chunk"]["started"]
            ),
            "loop_wall_seconds": (
                phase_intervals["loop"]["ended"] - phase_intervals["loop"]["started"]
            ),
            "phase_overlap_seconds": overlap_seconds,
            "combined_wall_seconds": (
                max(item["ended"] for item in phase_intervals.values())
                - min(item["started"] for item in phase_intervals.values())
            ),
            "metadata_exchange_bytes": metadata_exchange_bytes,
        }
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()


def infer_chunks_parallel(
    chunks: Iterable[Iterable[os.PathLike | str]],
    spec: VggtOmegaAdapterSpec,
    *,
    devices: Sequence[Any] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
    exchange_dir: Optional[os.PathLike | str] = None,
    artifact_root: Optional[os.PathLike | str] = None,
    artifact_relative_paths: Optional[Sequence[str]] = None,
    artifact_version: Optional[int] = None,
    artifact_field_names: Optional[Sequence[str]] = None,
    artifact_field_dtype_overrides: Optional[Mapping[str, Any]] = None,
) -> list[ChunkPrediction] | list[dict[str, Any]]:
    """Infer chunks on dedicated spawned GPU processes and return index-ordered results."""
    indexed = [(index, tuple(os.fspath(path) for path in paths)) for index, paths in enumerate(chunks)]
    if not indexed:
        return []
    if not isinstance(spec, VggtOmegaAdapterSpec):
        raise TypeError("spec must be a VggtOmegaAdapterSpec")
    device_names = tuple(str(device) for device in devices)
    if not device_names:
        raise ValueError("devices must contain at least one device")
    if len(set(device_names)) != len(device_names):
        raise ValueError("devices must not contain duplicates")
    for index, paths in indexed:
        if not paths:
            raise ValueError(f"chunk {index} has no image paths")

    artifact_options = (
        artifact_relative_paths,
        artifact_version,
        artifact_field_names,
    )
    direct_publication = artifact_root is not None
    if direct_publication != all(item is not None for item in artifact_options):
        raise ValueError(
            "artifact_root, artifact_relative_paths, artifact_version, and "
            "artifact_field_names must be provided together"
        )
    artifact_spec: Optional[dict[str, Any]] = None
    if direct_publication:
        relative_paths = tuple(str(path) for path in artifact_relative_paths or ())
        if len(relative_paths) != len(indexed):
            raise ValueError(
                "artifact_relative_paths must contain one path per chunk"
            )
        if int(artifact_version) != 2:
            raise ValueError("direct artifact publication requires artifact_version=2")
        if not artifact_field_names:
            raise ValueError("artifact_field_names must not be empty")
        if len(set(relative_paths)) != len(relative_paths):
            raise ValueError("artifact_relative_paths must not contain duplicates")
        if any(Path(path).is_absolute() or ".." in Path(path).parts for path in relative_paths):
            raise ValueError("artifact_relative_paths must stay below artifact_root")
        artifact_spec = {
            "root": os.fspath(artifact_root),
            "relative_paths": relative_paths,
            "version": int(artifact_version),
            "field_names": tuple(str(name) for name in artifact_field_names),
            "field_dtype_overrides": {
                str(name): str(dtype)
                for name, dtype in (artifact_field_dtype_overrides or {}).items()
            },
        }

    worker_count = min(len(device_names), len(indexed))
    assignments = [indexed[worker::worker_count] for worker in range(worker_count)]
    temporary_owner = None
    if exchange_dir is None:
        temporary_owner = tempfile.TemporaryDirectory(prefix="vggt-omega-chunks-")
        exchange = Path(temporary_owner.name)
    else:
        exchange = Path(exchange_dir)
        exchange.mkdir(parents=True, exist_ok=True)

    processes = []
    try:
        context = multiprocessing.get_context("spawn")
        for worker in range(worker_count):
            process = context.Process(
                target=_parallel_chunk_worker,
                args=(
                    spec,
                    device_names[worker],
                    assignments[worker],
                    str(exchange),
                    artifact_spec,
                ),
                name=f"vggt-omega-{worker}",
            )
            process.start()
            processes.append(process)
        for process in processes:
            process.join()

        errors = []
        for error_path in sorted(exchange.glob("*.error.json")):
            try:
                errors.append(json.loads(error_path.read_text(encoding="utf-8")))
            except Exception as exc:
                errors.append({"index": "unknown", "message": f"could not read {error_path}: {exc}"})
        failed = [(process.name, process.exitcode) for process in processes if process.exitcode != 0]
        if errors or failed:
            details = "; ".join(
                f"chunk {item.get('index')} on {item.get('device', 'unknown')}: "
                f"{item.get('type', 'Error')}: {item.get('message')}"
                for item in errors
            )
            raise RuntimeError(f"parallel VGGT-Omega inference failed ({failed}): {details or 'worker exited'}")

        results = []
        for index, _ in indexed:
            if direct_publication:
                result_path = exchange / f"{index:08d}.artifact.json"
                if not result_path.is_file():
                    raise RuntimeError(
                        "parallel VGGT-Omega inference produced no artifact "
                        f"metadata for chunk {index}"
                    )
                envelope = json.loads(result_path.read_text(encoding="utf-8"))
                if int(envelope.get("index", -1)) != index:
                    raise RuntimeError(
                        "parallel VGGT-Omega artifact metadata order mismatch: "
                        f"expected {index}, got {envelope.get('index')}"
                    )
                results.append(envelope["entry"])
                continue
            result_path = exchange / f"{index:08d}.pt"
            if not result_path.is_file():
                raise RuntimeError(f"parallel VGGT-Omega inference produced no result for chunk {index}")
            try:
                results.append(torch.load(result_path, map_location="cpu", weights_only=False))
            except TypeError:
                results.append(torch.load(result_path, map_location="cpu"))
        return results
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        if temporary_owner is not None:
            temporary_owner.cleanup()
