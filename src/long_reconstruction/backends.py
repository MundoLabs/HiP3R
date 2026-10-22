"""Local chunk backend factory for long reconstruction."""

from __future__ import annotations

from typing import Any, Tuple


LOCAL_BACKENDS = ("vggt", "vggt_omega", "lingbot_map", "abot_recon")

BACKEND_CHUNK_DEFAULTS = {
    "vggt": (60, 30),
    "vggt_omega": (60, 30),
    "lingbot_map": (120, 32),
    "abot_recon": (60, 30),
}


def normalize_backend_name(backend: str | None) -> str:
    name = "vggt_omega" if backend is None else str(backend).strip().lower().replace("-", "_")
    aliases = {
        "omega": "vggt_omega",
        "vggt": "vggt",
        "vggt_omega": "vggt_omega",
        "lingbot": "lingbot_map",
        "lingbotmap": "lingbot_map",
        "lingbot_map": "lingbot_map",
        "abot": "abot_recon",
        "abot_recon": "abot_recon",
    }
    if name not in aliases:
        raise ValueError(f"unsupported local backend {backend!r}; expected one of {LOCAL_BACKENDS}")
    return aliases[name]


def create_lingbot_map_backend(
    *,
    checkpoint: Any = None,
    device: Any = None,
    repo_root: Any = None,
    mode: str = "windowed",
    dtype: Any = "bfloat16",
    inference_dtype: Any = None,
    model: Any = None,
    **kwargs: Any,
) -> Tuple[Any, dict[str, Any]]:
    """Build the LingBot-MAP local backend and return ``(adapter, report)``."""
    from long_reconstruction.lingbot_map_adapter import LingbotMapChunkAdapter

    adapter = LingbotMapChunkAdapter(
        model,
        repo_root=repo_root,
        checkpoint=checkpoint,
        device=device,
        mode=mode,
        dtype=dtype if inference_dtype is None else inference_dtype,
        **kwargs,
    )
    return adapter, dict(adapter.load_report)


def create_vggt_backend(
    *,
    checkpoint: Any,
    device: Any = None,
    repo_root: Any = None,
    inference_dtype: Any = "bfloat16",
    model: Any = None,
    **kwargs: Any,
) -> Tuple[Any, dict[str, Any]]:
    """Build the external plain VGGT local backend and return ``(adapter, report)``."""
    from long_reconstruction.vggt_adapter import VggtChunkAdapter

    adapter = VggtChunkAdapter(
        model,
        repo_root=repo_root,
        checkpoint=checkpoint,
        device=device,
        inference_dtype=inference_dtype,
        **kwargs,
    )
    return adapter, dict(adapter.load_report)


def create_abot_recon_backend(
    *,
    checkpoint: Any = None,
    device: Any = None,
    repo_root: Any = None,
    inference_dtype: Any = "bfloat16",
    model: Any = None,
    **kwargs: Any,
) -> Tuple[Any, dict[str, Any]]:
    """Build the external ABot-Recon local backend and return ``(adapter, report)``."""
    from long_reconstruction.abot_recon_adapter import ABotReconChunkAdapter

    adapter = ABotReconChunkAdapter(
        model,
        repo_root=repo_root,
        checkpoint=checkpoint,
        device=device,
        inference_dtype=inference_dtype,
        **kwargs,
    )
    return adapter, dict(adapter.load_report)


def create_vggt_omega_backend(
    *,
    checkpoint: Any,
    device: Any = None,
    repo_root: Any = None,
    inference_dtype: Any = "bfloat16",
    model: Any = None,
    lazy_load: bool = False,
    cpu_threads_per_worker: int | None = None,
    **kwargs: Any,
) -> Tuple[Any, dict[str, Any]]:
    """Build the VGGT-Omega local backend and return ``(adapter, report)``."""
    from long_reconstruction.vggt_omega_adapter import (
        LazyVggtOmegaChunkAdapter,
        VggtOmegaAdapterSpec,
        VggtOmegaChunkAdapter,
    )

    if lazy_load:
        if model is not None:
            raise ValueError("lazy VGGT-Omega loading does not support an in-memory model")
        spec_kwargs = dict(
            checkpoint=checkpoint,
            inference_dtype=str(inference_dtype),
            cpu_threads_per_worker=cpu_threads_per_worker,
            **kwargs,
        )
        if repo_root is not None:
            spec_kwargs["repo_root"] = repo_root
        spec = VggtOmegaAdapterSpec(**spec_kwargs)
        adapter = LazyVggtOmegaChunkAdapter(spec, device=device)
        return adapter, {
            "backend": "vggt_omega",
            "source_type": "lazy",
            "checkpoint": str(checkpoint),
            "loaded": False,
        }
    if cpu_threads_per_worker is not None:
        raise ValueError("cpu_threads_per_worker requires lazy VGGT-Omega workers")
    adapter = VggtOmegaChunkAdapter(
        model,
        repo_root=repo_root,
        checkpoint=checkpoint,
        device=device,
        inference_dtype=inference_dtype,
        **kwargs,
    )
    return adapter, dict(adapter.load_report)


def create_local_backend(backend: str | None = "vggt_omega", **kwargs: Any) -> Tuple[Any, dict[str, Any]]:
    """Create a local chunk backend by name and return ``(adapter, load_report)``."""
    name = normalize_backend_name(backend)
    if name == "vggt":
        return create_vggt_backend(**kwargs)
    if name == "lingbot_map":
        return create_lingbot_map_backend(**kwargs)
    if name == "abot_recon":
        return create_abot_recon_backend(**kwargs)
    if name == "vggt_omega":
        return create_vggt_omega_backend(**kwargs)
    raise AssertionError(f"unhandled local backend {name!r}")
