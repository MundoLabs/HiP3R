"""Plain VGGT adapter for local long-reconstruction chunk predictions.

The plain VGGT source tree is intentionally external to this release checkout.
It must be provided with ``--vggt-repo-root`` or ``VGGT_REPO_ROOT``; checkpoints
must be local files supplied by the ``checkpoint`` argument or ``VGGT_CHECKPOINT``.
"""

from __future__ import annotations

import importlib
import inspect
import os
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from long_reconstruction.contract import ChunkPrediction
from long_reconstruction.vggt_omega_adapter import (
    _best_state_dict_for_model,
    _check_finite,
    _dtype_from_spec,
    _extract_state_dict,
    _first_parameter_device,
    _frame_tensor,
    _image_size_hw,
    _load_checkpoint_object,
    _repo_on_path,
    _squeeze_depth,
    _squeeze_optional,
    _squeeze_single_batch,
    _to_cpu_tensor,
    _unproject_depth_to_world_points,
    _w2c_to_c2w,
)


VGGT_REPO_ROOT_ENV = "VGGT_REPO_ROOT"
VGGT_CHECKPOINT_ENV = "VGGT_CHECKPOINT"
DEFAULT_IMAGE_RESOLUTION = 518
DEFAULT_PATCH_SIZE = 14
DEFAULT_PREPROCESS_MODE = "crop"


def _resolve_repo_root(repo_root: Any = None) -> Path:
    source = repo_root if repo_root is not None else os.environ.get(VGGT_REPO_ROOT_ENV)
    if source is None:
        raise ValueError(
            "plain VGGT requires --vggt-repo-root or VGGT_REPO_ROOT; "
            "no repository path is bundled in this release"
        )
    root = Path(source).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"VGGT repo_root does not exist: {root}")
    return root


def _import_from_repo(repo_root: Path, module_name: str):
    with _repo_on_path(repo_root):
        return importlib.import_module(module_name)


def _instantiate_vggt_model(repo_root: Path) -> torch.nn.Module:
    try:
        module = _import_from_repo(repo_root, "vggt.models.vggt")
        model_cls = getattr(module, "VGGT")
    except (ImportError, AttributeError):
        module = _import_from_repo(repo_root, "vggt.models")
        model_cls = getattr(module, "VGGT")
    return model_cls()


def _load_vggt_checkpoint_from_object(
    model: torch.nn.Module,
    checkpoint_obj: Mapping[str, Any],
    *,
    source_type: str,
    strict: bool,
) -> dict[str, Any]:
    if not isinstance(checkpoint_obj, Mapping):
        raise TypeError(
            f"checkpoint must resolve to a mapping, got {type(checkpoint_obj).__name__}"
        )

    raw_state_dict, state_dict_type = _extract_state_dict(checkpoint_obj)
    model_keys = set(model.state_dict().keys())
    normalized_state_dict, stripped_prefix, matched_keys = _best_state_dict_for_model(
        raw_state_dict,
        model_keys,
    )
    if matched_keys == 0:
        raise RuntimeError("checkpoint has zero keys matching the VGGT model state_dict")

    incompatible = model.load_state_dict(normalized_state_dict, strict=strict)
    missing_keys = list(
        getattr(incompatible, "missing_keys", incompatible[0] if incompatible else [])
    )
    unexpected_keys = list(
        getattr(incompatible, "unexpected_keys", incompatible[1] if incompatible else [])
    )
    return {
        "backend": "vggt",
        "source_type": source_type,
        "state_dict_type": state_dict_type,
        "total_keys": len(raw_state_dict),
        "matched_keys": matched_keys,
        "stripped_prefix": stripped_prefix,
        "strict": strict,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "implicit_download": False,
    }


def load_vggt_checkpoint(
    model: torch.nn.Module,
    checkpoint: Any,
    *,
    strict: bool = False,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    """Load a local plain VGGT checkpoint and return a checkpoint loading report."""
    checkpoint_obj, source_type = _load_checkpoint_object(checkpoint, map_location)
    return _load_vggt_checkpoint_from_object(
        model,
        checkpoint_obj,
        source_type=source_type,
        strict=strict,
    )


def _preprocess_kwargs(
    fn: Any,
    *,
    mode: str,
    image_resolution: int,
    patch_size: int,
) -> tuple[dict[str, Any], str]:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return {}, "paths_only_unknown_signature"
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if accepts_kwargs or "mode" in parameters:
        kwargs["mode"] = mode
    if accepts_kwargs or "image_resolution" in parameters:
        kwargs["image_resolution"] = image_resolution
    if accepts_kwargs or "patch_size" in parameters:
        kwargs["patch_size"] = patch_size
    return kwargs, (
        "paths_with_" + "_".join(sorted(kwargs)) if kwargs else "paths_only"
    )


class VggtChunkAdapter:
    """Run an external plain VGGT checkout and emit ``ChunkPrediction``."""

    def __init__(
        self,
        model: Any = None,
        *,
        repo_root: Any = None,
        checkpoint: Any = None,
        device: Any = None,
        inference_dtype: Any = "bfloat16",
        image_resolution: int = DEFAULT_IMAGE_RESOLUTION,
        patch_size: int = DEFAULT_PATCH_SIZE,
        preprocess_mode: str = DEFAULT_PREPROCESS_MODE,
        checkpoint_strict: bool = False,
        map_location: Any = "cpu",
    ) -> None:
        self.repo_root = _resolve_repo_root(repo_root)
        # Preserve the user-facing suffix when the checkpoint is a Hugging Face
        # cache symlink.  Its resolved blob target is extensionless, which would
        # otherwise make the shared loader fall back to ``torch.load`` instead
        # of the safetensors reader.
        self.checkpoint = Path(checkpoint).expanduser().absolute() if checkpoint is not None else None
        self.image_resolution = int(image_resolution)
        self.patch_size = int(patch_size)
        self.preprocess_mode = str(preprocess_mode)
        self.inference_dtype = _dtype_from_spec(inference_dtype)
        requested_device = torch.device(device) if device is not None else None
        self._last_preprocess_call: dict[str, Any] | None = None

        if model is None:
            model = _instantiate_vggt_model(self.repo_root)
            model_initialization_mode = "external_repo_vggt_default_constructor"
        else:
            model_initialization_mode = "provided_model"
        self.model = model
        self.device = requested_device if requested_device is not None else _first_parameter_device(model)

        if self.checkpoint is not None:
            self.load_report = load_vggt_checkpoint(
                self.model,
                self.checkpoint,
                strict=checkpoint_strict,
                map_location=map_location,
            )
            self.load_report["checkpoint"] = str(self.checkpoint)
        else:
            self.load_report = {
                "backend": "vggt",
                "source_type": "none",
                "state_dict_type": "none",
                "total_keys": 0,
                "matched_keys": 0,
                "stripped_prefix": "",
                "strict": checkpoint_strict,
                "missing_keys": [],
                "unexpected_keys": [],
                "implicit_download": False,
                "checkpoint": None,
            }
        self.load_report["repo_root"] = str(self.repo_root)
        self.load_report["model_initialization_mode"] = model_initialization_mode
        self.load_report["preprocess_protocol"] = self.preprocess_protocol()

        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def preprocess_protocol(self) -> dict[str, Any]:
        """Return the fixed plain VGGT preprocessing protocol."""
        return {
            "backend": "vggt",
            "loader": "vggt.utils.load_fn.load_and_preprocess_images",
            "mode": self.preprocess_mode,
            "image_resolution": self.image_resolution,
            "patch_size": self.patch_size,
            "path_ordering": "sorted_lexicographic",
            "pose_decode": "vggt.utils.pose_enc.pose_encoding_to_extri_intri",
            "pose_decode_output": "world_to_camera",
            "pose_contract": "explicit_inverse_to_camera_to_world",
            "geometry_source": "VGGT.forward raw output",
            "bundle_adjustment": "disabled",
            "implicit_download": False,
        }

    def activation_report(self) -> dict[str, Any]:
        """Return effective plain VGGT inference-path diagnostics."""
        return {
            "backend": "vggt",
            "repo_root": str(self.repo_root),
            "checkpoint": str(self.checkpoint) if self.checkpoint is not None else None,
            "load_report": dict(self.load_report),
            "preprocess_protocol": self.preprocess_protocol(),
            "last_preprocess_call": dict(self._last_preprocess_call or {}),
        }

    def infer_paths(
        self,
        image_paths: Iterable[os.PathLike | str],
        preprocess_mode: Optional[str] = None,
    ) -> ChunkPrediction:
        paths = sorted(os.fspath(path) for path in image_paths)
        if not paths:
            raise ValueError("image_paths is empty")
        module = _import_from_repo(self.repo_root, "vggt.utils.load_fn")
        kwargs, call_signature = _preprocess_kwargs(
            module.load_and_preprocess_images,
            mode=preprocess_mode or self.preprocess_mode,
            image_resolution=self.image_resolution,
            patch_size=self.patch_size,
        )
        images = module.load_and_preprocess_images(paths, **kwargs)
        self._last_preprocess_call = {
            "num_paths": len(paths),
            "paths": paths,
            "call_signature": call_signature,
            "kwargs": dict(kwargs),
        }
        return self.infer_tensor(images)

    def infer_loop_paths(
        self,
        image_paths: Iterable[os.PathLike | str],
        preprocess_mode: Optional[str] = None,
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
            raise TypeError(f"VGGT model must return a mapping, got {type(output).__name__}")
        return self._prediction_from_output(output, image_size_hw, images)

    def _prediction_from_output(
        self,
        output: Mapping[str, Any],
        image_size_hw: Tuple[int, int],
        input_images: torch.Tensor,
    ) -> ChunkPrediction:
        if "pose_enc" not in output:
            raise KeyError("VGGT output is missing required key 'pose_enc'")

        pose_module = _import_from_repo(self.repo_root, "vggt.utils.pose_enc")
        pose_enc = _to_cpu_tensor(output["pose_enc"], dtype=torch.float32)
        _check_finite("pose_enc", pose_enc)
        extrinsic_w2c, intrinsic = pose_module.pose_encoding_to_extri_intri(
            pose_enc,
            image_size_hw,
        )
        extrinsic_c2w = _squeeze_single_batch(
            "extrinsic_c2w",
            _w2c_to_c2w(extrinsic_w2c),
            3,
        )
        intrinsic = _squeeze_single_batch("intrinsic", intrinsic, 3).float()

        depth = _squeeze_depth(output["depth"]) if "depth" in output else None
        depth_conf = _squeeze_optional("depth_conf", output.get("depth_conf"), 3)
        world_points = _squeeze_optional("world_points", output.get("world_points"), 4)
        world_points_conf = _squeeze_optional(
            "world_points_conf",
            output.get("world_points_conf"),
            3,
        )

        if world_points is None:
            if depth is None:
                raise KeyError("VGGT output is missing 'world_points' and fallback 'depth'")
            world_points = _unproject_depth_to_world_points(depth, intrinsic, extrinsic_c2w)

        if world_points_conf is None:
            if depth_conf is None:
                raise KeyError(
                    "VGGT output is missing 'world_points_conf' and fallback 'depth_conf'"
                )
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
