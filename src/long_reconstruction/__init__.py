"""Long-sequence reconstruction models, geometry, and optimization."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "ChunkPrediction": "contract",
    "ChunkRange": "runner",
    "DEFAULT_HIERARCHICAL_COARSE_MAX_NFEV": "runner",
    "ReconstructionConfig": "runner",
    "ReconstructionPostProcessor": "postprocess",
    "ReconstructionRunner": "runner",
    "LongReconstructionResult": "runner",
    "LoopPair": "runner",
    "PostProcessResult": "postprocess",
    "Sim3": "sim3",
    "Sim3Edge": "postprocess",
    "estimate_sim3": "sim3",
    "HierarchicalPoseGraphConfig": "hierarchical_pose_graph",
    "HierarchicalPoseGraphResult": "hierarchical_pose_graph",
    "generate_chunk_ranges": "runner",
    "load_loop_pairs": "runner",
    "optimize_sim3_pose_graph": "pose_graph",
    "optimize_hierarchical_pose_graph": "hierarchical_pose_graph",
    "residual_edge_weight": "postprocess",
    "SaladRetrievalConfig": "loop_retrieval",
    "frame_to_chunk": "loop_retrieval",
    "map_retrievals_to_loop_pairs": "loop_retrieval",
    "retrieve_salad_loop_pairs": "loop_retrieval",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(f"long_reconstruction.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
