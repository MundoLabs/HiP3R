"""End-to-end long-sequence runner with optional hip3r Sim3 post-processing."""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import json
import fcntl
import errno
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from numpy.lib import format as np_format

try:
    import orjson as _orjson
except ImportError:  # pragma: no cover - exercised in minimal environments.
    _orjson = None

from long_reconstruction.contract import ChunkPrediction, _FINITE_VALIDATED_TOKEN
from long_reconstruction.hierarchical_pose_graph import (
    HierarchicalAlphaBasis,
    HierarchicalPoseGraphConfig,
    apply_hierarchical_alpha,
    build_hierarchical_basis,
)
from long_reconstruction.postprocess import (
    ReconstructionPostProcessor,
    PostProcessResult,
    Sim3Edge,
    estimate_adjacent_edge,
    estimate_loop_edge,
    residual_edge_weight,
)
from long_reconstruction.sim3 import Sim3


_HIERARCHICAL_TRAJECTORY_FIELD_PROFILES = frozenset(
    {
        "hierarchical_trajectory_v1",
        "hierarchical_trajectory_conf_fp16_v1",
        "hierarchical_trajectory_world_q16_conf_fp16_v1",
    }
)

_WORLD_POINTS_Q16_ENCODING_SCHEMA = "world_points_per_field_axis_symmetric_q16_v1"
_WORLD_POINTS_Q16_MAX = 32767

# The coarse graph may need substantially more evaluations than the frame-level
# solve on long or weakly conditioned sequences.  Keep this public so every
# entry point resolves and records the same production default.
DEFAULT_HIERARCHICAL_COARSE_MAX_NFEV = 5000


def optimize_sim3_pose_graph(*args: Any, **kwargs: Any):
    from long_reconstruction.pose_graph import optimize_sim3_pose_graph

    return optimize_sim3_pose_graph(*args, **kwargs)


def prepare_torch_pose_graph_analytic_runtime(*args: Any, **kwargs: Any):
    from long_reconstruction.pose_graph import (
        prepare_torch_pose_graph_analytic_runtime as _prepare_runtime,
    )

    return _prepare_runtime(*args, **kwargs)


@dataclass(frozen=True)
class ChunkRange:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class LoopPair:
    chunk_a: int
    chunk_b: int
    frame_a: int | None = None
    frame_b: int | None = None
    similarity: float | None = None


class _CompactBasisComponentCacheMiss(RuntimeError):
    """Request one full-manifest fallback to rebuild a missing component cache."""


class HierarchicalSolverConvergenceError(RuntimeError):
    """Raised when an explicitly required hierarchical solve did not converge."""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class HierarchicalBasisActivationError(RuntimeError):
    """Raised when hierarchical basis diagnostics fail an activation contract."""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class ReconstructionConfig:
    chunk_size: int = 60
    overlap: int = 30
    loop_window_size: int = 8
    loop_edge_policy: str = "all"
    loop_edge_budget: int | None = None
    postprocess: str = "pose_graph"
    confidence_threshold: float | None = 0.0
    max_alignment_points: int = 20000
    max_sim3_iterations: int = 10
    huber_delta: float | None = None
    residual_edge_weighting: bool = False
    residual_edge_quality_quantile: str = "p95_only"
    use_jac_sparsity: bool = False
    residual_scale: float = 0.05
    min_edge_weight: float = 0.05
    max_edge_weight: float = 1.0
    chain_global_scale_gauge_centering_v1: bool = False
    chain_global_scale_gauge_centering_span_cap: float = 50.0
    save_chunk_predictions: bool = True
    reliability_trace: bool = False
    sim3_schedule: str = "deferred"
    hierarchical_fine_edge_weight: float = 1.0
    hierarchical_coarse_prior_weight: float = 0.25
    hierarchical_normalize_residual_families: bool = False
    hierarchical_frame_stride: int = 1
    hierarchical_max_nfev: int = 100
    hierarchical_solver_ftol: float = 1e-8
    hierarchical_solver_xtol: float = 1e-8
    hierarchical_solver_gtol: float = 1e-8
    hierarchical_joint_lsmr_tolerance: float | None = None
    hierarchical_coarse_max_nfev: int = DEFAULT_HIERARCHICAL_COARSE_MAX_NFEV
    hierarchical_robust_loss: str = "huber"
    hierarchical_f_scale: float = 1.0
    hierarchical_translation_scale: float = 1.0
    hierarchical_max_log_scale_delta: float = 4.605170185988092
    hierarchical_apply_frame_corrections: bool = True
    hierarchical_frame_correction_alpha: float = 1.0
    hierarchical_freeze_coarse_alignments: bool = False
    hierarchical_frozen_optimizer_variable_rows_only: bool = True
    hierarchical_joint_coarse_warm_start: bool = False
    hierarchical_max_frames_per_solve: int = 2000
    hierarchical_consensus_keyframe_stride: int = 20
    hierarchical_long_sequence_mode: str = "overlap_consensus"
    hierarchical_keyframe_selection: str = "uniform"
    hierarchical_keyframe_target_stride: int = 5
    hierarchical_keyframe_min_overlap_shared: int = 3
    hierarchical_chunk_scale_trust_mode: str = "fixed"
    hierarchical_chunk_scale_trust_gain: float = 2.0
    hierarchical_chunk_scale_trust_min_ratio: float = 0.2
    hierarchical_chunk_scale_trust_max_ratio: float = 2.0
    hierarchical_solver_workers: int = 1
    hierarchical_incremental_frozen_residual: bool = True
    hierarchical_analytic_frozen_jacobian: bool = True
    hierarchical_analytic_dynamic_jacobian: bool = False
    hierarchical_analytic_dynamic_jacobian_backend: str = "closed_form"
    hierarchical_analytic_dynamic_jacobian_local_rotation_projection: str = "strict"
    hierarchical_vectorize_dynamic_residual: bool = False
    hierarchical_analytic_jacobian_threads: int = 8
    hierarchical_analytic_jacobian_backend: str = "closed_form"
    hierarchical_analytic_fast_residual_logmap: bool = True
    hierarchical_factorized_v4_keyframe_continuation: bool = False
    hierarchical_factorized_v4_keyframe_continuation_objective_projection: (
        bool
    ) = False
    hierarchical_joint_family_diagnostics: bool = False
    hierarchical_joint_full_sparse_exact_basis: bool = False
    hierarchical_joint_frame_relaxation_max_nfev: int = 0
    hierarchical_joint_subspace_max_nfev: int = 0
    hierarchical_joint_subspace_control_stride: int = 64
    hierarchical_joint_subspace_conditional_frame_polish: bool = False
    hierarchical_joint_subspace_min_coarse_squared_residual: float = 0.0
    hierarchical_joint_subspace_stage_b_rotation_trust_ratio: (
        float | None
    ) = None
    hierarchical_joint_subspace_stage_b_translation_trust_ratio: (
        float | None
    ) = None
    hierarchical_joint_subspace_stage_b_p64_gauge_fix: bool = False
    hierarchical_joint_subspace_stage_b_p64_sparse_basis: bool = False
    hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle: bool = False
    hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking: (
        bool
    ) = False
    hierarchical_joint_subspace_unrestricted_repair_max_nfev: int = 0
    hierarchical_joint_schur_diagnostics: bool = False
    hierarchical_joint_schur_damping: float = 1e-4
    hierarchical_joint_schur_cg_maxiter: int = 200
    hierarchical_joint_schur_cg_rtol: float = 1e-4
    hierarchical_joint_schur_two_level_aggregate_size: int | None = None
    hierarchical_joint_schur_full_space_block_pcg_probe: bool = False
    hierarchical_joint_schur_step: bool = False
    hierarchical_joint_schur_step_final_solver: bool = False
    hierarchical_joint_schur_frozen_constrained_step_oracle: bool = False
    hierarchical_joint_schur_one_active_set_restoration_at_first_reject: (
        bool
    ) = False
    hierarchical_joint_schur_step_public_pilot_max_nfev: int = 0
    hierarchical_joint_schur_step_max_outer_iterations: int = 1
    hierarchical_joint_schur_step_initial_alpha: float = 0.0625
    hierarchical_joint_schur_step_backtracks: int = 4
    hierarchical_joint_schur_step_freeze_coarse_scales: bool = False
    hierarchical_joint_schur_step_min_accepted_alpha: float | None = None
    hierarchical_joint_schur_step_frame_low_frequency_trust_ratio: (
        float | None
    ) = None
    hierarchical_joint_schur_step_frame_low_frequency_trust_aggregate_size: (
        int
    ) = 64
    hierarchical_joint_schur_step_min_low_frequency_efficiency_ratio: (
        float | None
    ) = None
    hierarchical_joint_schur_step_max_initial_jacobi_low_frequency_efficiency: (
        float | None
    ) = None
    hierarchical_joint_schur_step_max_initial_diagonal_low_frequency_efficiency: (
        float | None
    ) = None
    hierarchical_joint_schur_step_max_initial_diagonal_reduction_fraction: (
        float | None
    ) = None
    hierarchical_joint_schur_step_max_graph_cost_ratio: float = 1.02
    hierarchical_joint_schur_step_max_adjacent_p95_ratio: float | None = None
    hierarchical_joint_schur_step_max_loop_p95_ratio: float | None = None
    hierarchical_joint_schur_step_max_loop_max_ratio: float | None = None
    hierarchical_coarse_analytic_jacobian: bool = True
    hierarchical_require_solver_convergence: bool = False
    hierarchical_coarse_accept_max_nfev_optimality: float | None = None
    hierarchical_coarse_x_scale: str = "unit"
    hierarchical_coarse_residual_component_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    hierarchical_record_residual_components: bool = False
    hierarchical_coarse_lsmr_tolerance: float | None = None
    hierarchical_coarse_loop_continuation_stages: int = 1
    hierarchical_coarse_adjacent_first_stage: bool = False
    hierarchical_coarse_max_loop_span: int | None = None
    hierarchical_coarse_adaptive_loop_continuation: bool = False
    hierarchical_coarse_adaptive_full_max_nfev: int | None = None
    hierarchical_coarse_adaptive_budget_fallback_max_nfev: int | None = None
    hierarchical_coarse_prefer_boundary_restart_for_active_bounds: bool = False
    hierarchical_coarse_continuation_max_nfev: int = 100
    hierarchical_coarse_boundary_restart_passes: int = 0
    hierarchical_coarse_boundary_restart_max_nfev: int = 1000
    local_chunk_workers: int = 1
    loop_forward_workers: int = 1
    postprocess_loop_workers: int = 32
    artifact_load_workers: int = 32
    forward_artifact_write_workers: int = 1
    forward_artifact_field_profile: str = "full"
    forward_worker_direct_artifact_publish: bool = False
    forward_persistent_gpu_workers: bool = False
    local_chunk_gpu_ids: tuple[int, ...] = ()
    parallel_result_root: str | None = None
    postprocess_lock_path: str | None = None
    forward_artifact_version: int = 2

    def __post_init__(self) -> None:
        if self.chunk_size < 2:
            raise ValueError("chunk_size must be at least 2")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        if self.loop_window_size < 1:
            raise ValueError("loop_window_size must be positive")
        if self.loop_edge_policy not in {
            "all",
            "recent",
            "uniform",
            "descriptor",
            "geometry_utility",
        }:
            raise ValueError(
                "loop_edge_policy must be one of: all, recent, uniform, descriptor, geometry_utility"
            )
        if self.loop_edge_budget is not None and self.loop_edge_budget < 0:
            raise ValueError("loop_edge_budget must be nonnegative")
        if self.loop_edge_policy != "all" and self.loop_edge_budget is None:
            raise ValueError(f"loop_edge_budget is required for policy={self.loop_edge_policy}")
        if self.residual_edge_quality_quantile not in {"p95_only", "p95_else_p90"}:
            raise ValueError(
                "residual_edge_quality_quantile must be 'p95_only' or 'p95_else_p90'"
            )
        if (
            not np.isfinite(self.chain_global_scale_gauge_centering_span_cap)
            or self.chain_global_scale_gauge_centering_span_cap <= 0.0
        ):
            raise ValueError(
                "chain_global_scale_gauge_centering_span_cap must be positive finite"
            )
        if self.hierarchical_factorized_v4_keyframe_continuation:
            if not self.chain_global_scale_gauge_centering_v1:
                raise ValueError(
                    "hierarchical_factorized_v4_keyframe_continuation requires "
                    "chain_global_scale_gauge_centering_v1"
                )
            if self.hierarchical_long_sequence_mode != "keyframe_fine_graph":
                raise ValueError(
                    "hierarchical_factorized_v4_keyframe_continuation requires "
                    "hierarchical_long_sequence_mode='keyframe_fine_graph'"
                )
            if not self.hierarchical_freeze_coarse_alignments:
                raise ValueError(
                    "hierarchical_factorized_v4_keyframe_continuation requires "
                    "frozen coarse alignments"
                )
        if (
            self.hierarchical_factorized_v4_keyframe_continuation_objective_projection
            and not self.hierarchical_factorized_v4_keyframe_continuation
        ):
            raise ValueError(
                "hierarchical_factorized_v4_keyframe_continuation_objective_projection "
                "requires hierarchical_factorized_v4_keyframe_continuation"
            )
        if self.postprocess not in {"none", "adjacent", "pose_graph", "hierarchical_pose_graph"}:
            raise ValueError(
                "postprocess must be one of: none, adjacent, pose_graph, hierarchical_pose_graph"
            )
        if self.sim3_schedule not in {"deferred", "incremental"}:
            raise ValueError("sim3_schedule must be one of: deferred, incremental")
        if self.hierarchical_fine_edge_weight < 0.0:
            raise ValueError("hierarchical_fine_edge_weight must be non-negative")
        if self.hierarchical_coarse_prior_weight < 0.0:
            raise ValueError("hierarchical_coarse_prior_weight must be non-negative")
        if self.hierarchical_frame_stride < 1:
            raise ValueError("hierarchical_frame_stride must be positive")
        if self.hierarchical_max_nfev < 1:
            raise ValueError("hierarchical_max_nfev must be positive")
        if self.hierarchical_joint_subspace_max_nfev < 0:
            raise ValueError(
                "hierarchical_joint_subspace_max_nfev must be non-negative"
            )
        if (
            self.hierarchical_joint_full_sparse_exact_basis
            and self.hierarchical_freeze_coarse_alignments
        ):
            raise ValueError(
                "hierarchical_joint_full_sparse_exact_basis requires free "
                "coarse alignments"
            )
        direct_sparse_basis_conflicts = (
            self.hierarchical_joint_subspace_max_nfev > 0
            or self.hierarchical_joint_frame_relaxation_max_nfev > 0
            or self.hierarchical_joint_schur_diagnostics
            or self.hierarchical_joint_schur_full_space_block_pcg_probe
            or self.hierarchical_joint_schur_step
            or self.hierarchical_joint_schur_step_final_solver
            or self.hierarchical_joint_schur_frozen_constrained_step_oracle
            or (
                self
                .hierarchical_joint_schur_one_active_set_restoration_at_first_reject
            )
            or self.hierarchical_joint_schur_step_public_pilot_max_nfev > 0
            or (
                self.hierarchical_joint_subspace_stage_b_rotation_trust_ratio
                is not None
            )
            or (
                self.hierarchical_joint_subspace_stage_b_translation_trust_ratio
                is not None
            )
            or self.hierarchical_joint_subspace_stage_b_p64_gauge_fix
            or self.hierarchical_joint_subspace_stage_b_p64_sparse_basis
            or self.hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle
            or (
                self
                .hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking
            )
            or self.hierarchical_joint_subspace_unrestricted_repair_max_nfev
            > 0
        )
        if (
            self.hierarchical_joint_full_sparse_exact_basis
            and direct_sparse_basis_conflicts
        ):
            raise ValueError(
                "hierarchical_joint_full_sparse_exact_basis cannot be combined "
                "with subspace, Stage-B, frame relaxation, Schur, public pilot, "
                "or unrestricted repair controls"
            )
        if (
            self.hierarchical_joint_subspace_max_nfev
            >= self.hierarchical_max_nfev
        ):
            raise ValueError(
                "hierarchical_joint_subspace_max_nfev must be smaller than "
                "hierarchical_max_nfev"
            )
        if self.hierarchical_joint_subspace_control_stride < 2:
            raise ValueError(
                "hierarchical_joint_subspace_control_stride must be at least 2"
            )
        if (
            not np.isfinite(
                self.hierarchical_joint_subspace_min_coarse_squared_residual
            )
            or self.hierarchical_joint_subspace_min_coarse_squared_residual
            < 0.0
        ):
            raise ValueError(
                "hierarchical_joint_subspace_min_coarse_squared_residual "
                "must be finite and non-negative"
            )
        if (
            self.hierarchical_joint_subspace_min_coarse_squared_residual > 0.0
            and self.hierarchical_joint_subspace_max_nfev == 0
        ):
            raise ValueError(
                "hierarchical_joint_subspace_min_coarse_squared_residual "
                "requires hierarchical_joint_subspace_max_nfev"
            )
        stage_b_trust_ratios = (
            self.hierarchical_joint_subspace_stage_b_rotation_trust_ratio,
            self.hierarchical_joint_subspace_stage_b_translation_trust_ratio,
        )
        if (stage_b_trust_ratios[0] is None) != (
            stage_b_trust_ratios[1] is None
        ):
            raise ValueError(
                "hierarchical_joint_subspace_stage_b rotation and "
                "translation trust ratios must be configured together"
            )
        if stage_b_trust_ratios[0] is not None and any(
            not np.isfinite(value) or value <= 0.0
            for value in stage_b_trust_ratios
        ):
            raise ValueError(
                "hierarchical_joint_subspace_stage_b trust ratios must be "
                "finite and positive"
            )
        if (
            stage_b_trust_ratios[0] is not None
            and not self.hierarchical_joint_subspace_conditional_frame_polish
        ):
            raise ValueError(
                "hierarchical_joint_subspace_stage_b trust ratios require "
                "hierarchical_joint_subspace_conditional_frame_polish"
            )
        if (
            self.hierarchical_joint_subspace_stage_b_p64_gauge_fix
            and not self.hierarchical_joint_subspace_conditional_frame_polish
        ):
            raise ValueError(
                "hierarchical_joint_subspace_stage_b_p64_gauge_fix requires "
                "hierarchical_joint_subspace_conditional_frame_polish"
            )
        if (
            self.hierarchical_joint_subspace_stage_b_p64_sparse_basis
            and not self.hierarchical_joint_subspace_conditional_frame_polish
        ):
            raise ValueError(
                "hierarchical_joint_subspace_stage_b_p64_sparse_basis "
                "requires hierarchical_joint_subspace_conditional_frame_polish"
            )
        if (
            self.hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle
            and not self.hierarchical_joint_subspace_conditional_frame_polish
        ):
            raise ValueError(
                "hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle "
                "requires hierarchical_joint_subspace_conditional_frame_polish"
            )
        if (
            self.hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle
            and not self.hierarchical_joint_subspace_stage_b_p64_sparse_basis
        ):
            raise ValueError(
                "hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle "
                "requires hierarchical_joint_subspace_stage_b_p64_sparse_basis"
            )
        if (
            self.hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking
            and not self.hierarchical_joint_subspace_conditional_frame_polish
        ):
            raise ValueError(
                "hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking "
                "requires hierarchical_joint_subspace_conditional_frame_polish"
            )
        if (
            self.hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking
            and not self.hierarchical_joint_subspace_stage_b_p64_sparse_basis
        ):
            raise ValueError(
                "hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking "
                "requires hierarchical_joint_subspace_stage_b_p64_sparse_basis"
            )
        if (
            self.hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking
            and not self.hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle
        ):
            raise ValueError(
                "hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking "
                "requires hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle"
            )
        if (
            self.hierarchical_joint_subspace_stage_b_p64_gauge_fix
            and stage_b_trust_ratios[0] is not None
        ):
            raise ValueError(
                "hierarchical_joint_subspace_stage_b_p64_gauge_fix cannot be "
                "combined with stage_b trust ratios"
            )
        if (
            self.hierarchical_joint_subspace_stage_b_p64_sparse_basis
            and stage_b_trust_ratios[0] is not None
        ):
            raise ValueError(
                "hierarchical_joint_subspace_stage_b_p64_sparse_basis cannot be "
                "combined with stage_b trust ratios"
            )
        if (
            self.hierarchical_joint_subspace_stage_b_p64_sparse_basis
            and self.hierarchical_joint_subspace_stage_b_p64_gauge_fix
        ):
            raise ValueError(
                "hierarchical_joint_subspace_stage_b_p64_sparse_basis cannot be "
                "combined with p64 gauge fix"
            )
        if self.hierarchical_joint_subspace_unrestricted_repair_max_nfev < 0:
            raise ValueError(
                "hierarchical_joint_subspace_unrestricted_repair_max_nfev "
                "must be non-negative"
            )
        if (
            self.hierarchical_joint_subspace_unrestricted_repair_max_nfev > 0
            and not self.hierarchical_joint_subspace_conditional_frame_polish
        ):
            raise ValueError(
                "hierarchical_joint_subspace_unrestricted_repair_max_nfev "
                "requires hierarchical_joint_subspace_conditional_frame_polish"
            )
        if (
            stage_b_trust_ratios[0] is not None
            and self.hierarchical_joint_subspace_unrestricted_repair_max_nfev
            > 0
        ):
            raise ValueError(
                "hierarchical_joint_subspace_stage_b trust ratios cannot be "
                "combined with unrestricted repair"
            )
        if (
            self.hierarchical_joint_subspace_stage_b_p64_gauge_fix
            and self.hierarchical_joint_subspace_unrestricted_repair_max_nfev
            > 0
        ):
            raise ValueError(
                "hierarchical_joint_subspace_stage_b_p64_gauge_fix cannot be "
                "combined with unrestricted repair"
            )
        if (
            self.hierarchical_joint_subspace_stage_b_p64_sparse_basis
            and self.hierarchical_joint_subspace_unrestricted_repair_max_nfev
            > 0
        ):
            raise ValueError(
                "hierarchical_joint_subspace_stage_b_p64_sparse_basis cannot be "
                "combined with unrestricted repair"
            )
        if (
            self.hierarchical_joint_subspace_max_nfev > 0
            and self.hierarchical_freeze_coarse_alignments
        ):
            raise ValueError(
                "hierarchical_joint_subspace_max_nfev requires free coarse "
                "alignments"
            )
        if (
            self.hierarchical_joint_subspace_conditional_frame_polish
            and self.hierarchical_joint_subspace_max_nfev == 0
        ):
            raise ValueError(
                "hierarchical_joint_subspace_conditional_frame_polish "
                "requires hierarchical_joint_subspace_max_nfev"
            )
        if self.hierarchical_joint_subspace_max_nfev > 0 and (
            self.hierarchical_joint_frame_relaxation_max_nfev > 0
            or self.hierarchical_joint_schur_diagnostics
            or self.hierarchical_joint_schur_full_space_block_pcg_probe
            or self.hierarchical_joint_schur_step
            or self.hierarchical_joint_schur_step_final_solver
            or self.hierarchical_joint_schur_frozen_constrained_step_oracle
            or (
                self
                .hierarchical_joint_schur_one_active_set_restoration_at_first_reject
            )
            or self.hierarchical_joint_schur_step_public_pilot_max_nfev > 0
        ):
            raise ValueError(
                "hierarchical_joint_subspace_max_nfev cannot be combined "
                "with frame relaxation, Schur diagnostics/steps, or a "
                "public pilot"
            )
        if self.hierarchical_joint_schur_full_space_block_pcg_probe and (
            self.hierarchical_joint_full_sparse_exact_basis
            or self.hierarchical_joint_subspace_max_nfev > 0
            or (
                self.hierarchical_joint_subspace_stage_b_rotation_trust_ratio
                is not None
            )
            or (
                self.hierarchical_joint_subspace_stage_b_translation_trust_ratio
                is not None
            )
            or self.hierarchical_joint_subspace_stage_b_p64_gauge_fix
            or self.hierarchical_joint_subspace_stage_b_p64_sparse_basis
            or self.hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle
            or (
                self
                .hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking
            )
            or self.hierarchical_joint_subspace_unrestricted_repair_max_nfev
            > 0
        ):
            raise ValueError(
                "hierarchical_joint_schur_full_space_block_pcg_probe cannot "
                "be combined with full sparse exact basis, subspace, or "
                "Stage-B controls"
            )
        if (
            self.hierarchical_joint_schur_full_space_block_pcg_probe
            and not self.hierarchical_joint_schur_diagnostics
        ):
            raise ValueError(
                "hierarchical_joint_schur_full_space_block_pcg_probe requires "
                "hierarchical_joint_schur_diagnostics"
            )
        if (
            self.hierarchical_joint_schur_full_space_block_pcg_probe
            and self.hierarchical_joint_schur_step
        ):
            raise ValueError(
                "hierarchical_joint_schur_full_space_block_pcg_probe cannot "
                "be combined with hierarchical_joint_schur_step"
            )
        if (
            self.hierarchical_joint_schur_full_space_block_pcg_probe
            and self.hierarchical_joint_schur_step_freeze_coarse_scales
        ):
            raise ValueError(
                "hierarchical_joint_schur_full_space_block_pcg_probe cannot "
                "be combined with "
                "hierarchical_joint_schur_step_freeze_coarse_scales"
            )
        if (
            self.hierarchical_joint_schur_step_final_solver
            and not self.hierarchical_joint_schur_step
        ):
            raise ValueError(
                "hierarchical_joint_schur_step_final_solver requires "
                "hierarchical_joint_schur_step"
            )
        if (
            self.hierarchical_joint_schur_frozen_constrained_step_oracle
            and not self.hierarchical_joint_schur_step
        ):
            raise ValueError(
                "hierarchical_joint_schur_frozen_constrained_step_oracle "
                "requires hierarchical_joint_schur_step"
            )
        if self.hierarchical_joint_schur_one_active_set_restoration_at_first_reject and (
            not self.hierarchical_joint_schur_step_final_solver
            or not self.hierarchical_joint_schur_frozen_constrained_step_oracle
        ):
            raise ValueError(
                "hierarchical_joint_schur_one_active_set_restoration_at_first_reject "
                "requires hierarchical_joint_schur_step_final_solver and "
                "hierarchical_joint_schur_frozen_constrained_step_oracle"
            )
        if (
            self.hierarchical_joint_schur_step_final_solver
            and not self.hierarchical_analytic_dynamic_jacobian
        ):
            raise ValueError(
                "hierarchical_joint_schur_step_final_solver requires "
                "hierarchical_analytic_dynamic_jacobian"
            )
        if (
            self.hierarchical_joint_schur_step_final_solver
            and self.hierarchical_joint_schur_step_public_pilot_max_nfev > 0
        ):
            raise ValueError(
                "hierarchical_joint_schur_step_final_solver cannot be combined "
                "with hierarchical_joint_schur_step_public_pilot_max_nfev"
            )
        if (
            self.hierarchical_joint_schur_step_final_solver
            and self.hierarchical_joint_frame_relaxation_max_nfev > 0
        ):
            raise ValueError(
                "hierarchical_joint_schur_step_final_solver cannot be combined "
                "with hierarchical_joint_frame_relaxation_max_nfev"
            )
        if (
            self.hierarchical_joint_schur_step_final_solver
            and self.hierarchical_joint_schur_full_space_block_pcg_probe
        ):
            raise ValueError(
                "hierarchical_joint_schur_step_final_solver cannot be combined "
                "with hierarchical_joint_schur_full_space_block_pcg_probe"
            )
        if (
            self.hierarchical_joint_schur_step_final_solver
            and self.hierarchical_joint_schur_step_freeze_coarse_scales
        ):
            raise ValueError(
                "hierarchical_joint_schur_step_final_solver currently requires "
                "hierarchical_joint_schur_step_freeze_coarse_scales=False"
            )
        for name, value in (
            ("hierarchical_solver_ftol", self.hierarchical_solver_ftol),
            ("hierarchical_solver_xtol", self.hierarchical_solver_xtol),
            ("hierarchical_solver_gtol", self.hierarchical_solver_gtol),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.hierarchical_joint_lsmr_tolerance is not None and (
            not np.isfinite(self.hierarchical_joint_lsmr_tolerance)
            or self.hierarchical_joint_lsmr_tolerance <= 0.0
        ):
            raise ValueError(
                "hierarchical_joint_lsmr_tolerance must be None or finite and positive"
            )
        if self.hierarchical_f_scale <= 0.0:
            raise ValueError("hierarchical_f_scale must be positive")
        if self.hierarchical_translation_scale <= 0.0:
            raise ValueError("hierarchical_translation_scale must be positive")
        if not 0.0 <= self.hierarchical_frame_correction_alpha <= 1.0:
            raise ValueError("hierarchical_frame_correction_alpha must be in [0, 1]")
        if self.hierarchical_max_frames_per_solve < 1:
            raise ValueError("hierarchical_max_frames_per_solve must be positive")
        if self.hierarchical_consensus_keyframe_stride < 1:
            raise ValueError("hierarchical_consensus_keyframe_stride must be positive")
        if self.hierarchical_long_sequence_mode not in {
            "overlap_consensus",
            "keyframe_fine_graph",
            "keyframe_then_full",
        }:
            raise ValueError(
                "hierarchical_long_sequence_mode must be 'overlap_consensus', "
                "'keyframe_fine_graph', or 'keyframe_then_full'"
            )
        if (
            self.hierarchical_long_sequence_mode == "keyframe_then_full"
            and self.hierarchical_freeze_coarse_alignments
        ):
            raise ValueError(
                "hierarchical_long_sequence_mode='keyframe_then_full' requires "
                "free coarse alignments"
            )
        if self.hierarchical_keyframe_selection not in {"uniform", "motion_equalized"}:
            raise ValueError(
                "hierarchical_keyframe_selection must be 'uniform' or 'motion_equalized'"
            )
        if self.hierarchical_keyframe_target_stride < 2:
            raise ValueError("hierarchical_keyframe_target_stride must be at least 2")
        if self.hierarchical_keyframe_min_overlap_shared < 1:
            raise ValueError(
                "hierarchical_keyframe_min_overlap_shared must be positive"
            )
        if self.hierarchical_chunk_scale_trust_mode not in {
            "fixed",
            "edge_consistency",
        }:
            raise ValueError(
                "hierarchical_chunk_scale_trust_mode must be 'fixed' "
                "or 'edge_consistency'"
            )
        if self.hierarchical_chunk_scale_trust_gain <= 0.0:
            raise ValueError("hierarchical_chunk_scale_trust_gain must be positive")
        if not (
            0.0
            < self.hierarchical_chunk_scale_trust_min_ratio
            <= self.hierarchical_chunk_scale_trust_max_ratio
        ):
            raise ValueError(
                "hierarchical_chunk_scale_trust ratios must satisfy 0 < min <= max"
            )
        if (
            self.hierarchical_coarse_max_nfev is not None
            and self.hierarchical_coarse_max_nfev < 1
        ):
            raise ValueError("hierarchical_coarse_max_nfev must be positive")
        if self.hierarchical_coarse_x_scale not in {"unit", "jac"}:
            raise ValueError("hierarchical_coarse_x_scale must be 'unit' or 'jac'")
        weights = np.asarray(self.hierarchical_coarse_residual_component_weights, dtype=np.float64)
        if weights.shape != (3,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("hierarchical_coarse_residual_component_weights must be three finite positive values")
        if (
            self.hierarchical_coarse_lsmr_tolerance is not None
            and self.hierarchical_coarse_lsmr_tolerance <= 0.0
        ):
            raise ValueError("hierarchical_coarse_lsmr_tolerance must be positive")
        if (
            self.hierarchical_coarse_accept_max_nfev_optimality is not None
            and (
                not np.isfinite(
                    self.hierarchical_coarse_accept_max_nfev_optimality
                )
                or self.hierarchical_coarse_accept_max_nfev_optimality <= 0.0
            )
        ):
            raise ValueError(
                "hierarchical_coarse_accept_max_nfev_optimality must be "
                "positive and finite"
            )
        if (
            self.hierarchical_joint_schur_two_level_aggregate_size is not None
            and self.hierarchical_joint_schur_two_level_aggregate_size < 2
        ):
            raise ValueError(
                "hierarchical_joint_schur_two_level_aggregate_size must be None or >= 2"
            )
        if self.hierarchical_joint_schur_step_max_outer_iterations < 1:
            raise ValueError(
                "hierarchical_joint_schur_step_max_outer_iterations must be positive"
            )
        if self.hierarchical_joint_schur_step_public_pilot_max_nfev < 0:
            raise ValueError(
                "hierarchical_joint_schur_step_public_pilot_max_nfev must be "
                "non-negative"
            )
        if (
            self.hierarchical_joint_schur_step_public_pilot_max_nfev
            >= self.hierarchical_max_nfev
        ):
            raise ValueError(
                "hierarchical_joint_schur_step_public_pilot_max_nfev must be "
                "smaller than hierarchical_max_nfev"
            )
        if (
            self.hierarchical_joint_schur_step_public_pilot_max_nfev > 0
            and not self.hierarchical_joint_schur_step
        ):
            raise ValueError(
                "hierarchical_joint_schur_step_public_pilot_max_nfev requires "
                "hierarchical_joint_schur_step"
            )
        if (
            self.hierarchical_joint_schur_step_public_pilot_max_nfev > 0
            and self.hierarchical_joint_frame_relaxation_max_nfev > 0
        ):
            raise ValueError(
                "hierarchical_joint_schur_step_public_pilot_max_nfev cannot "
                "be combined with hierarchical_joint_frame_relaxation_max_nfev"
            )
        if (
            self.hierarchical_joint_schur_step_min_accepted_alpha is not None
            and (
                not np.isfinite(
                    self.hierarchical_joint_schur_step_min_accepted_alpha
                )
                or not 0.0
                < self.hierarchical_joint_schur_step_min_accepted_alpha
                <= self.hierarchical_joint_schur_step_initial_alpha
            )
        ):
            raise ValueError(
                "hierarchical_joint_schur_step_min_accepted_alpha must be "
                "None or in (0, hierarchical_joint_schur_step_initial_alpha]"
            )
        if (
            self.hierarchical_joint_schur_step_frame_low_frequency_trust_ratio
            is not None
            and (
                not np.isfinite(
                    self.hierarchical_joint_schur_step_frame_low_frequency_trust_ratio
                )
                or self.hierarchical_joint_schur_step_frame_low_frequency_trust_ratio
                <= 0.0
            )
        ):
            raise ValueError(
                "hierarchical_joint_schur_step_frame_low_frequency_trust_ratio "
                "must be None or finite and positive"
            )
        if (
            self.hierarchical_joint_schur_step_frame_low_frequency_trust_aggregate_size
            < 2
        ):
            raise ValueError(
                "hierarchical_joint_schur_step_frame_low_frequency_trust_aggregate_size "
                "must be >= 2"
            )
        if (
            self.hierarchical_joint_schur_step_min_low_frequency_efficiency_ratio
            is not None
            and (
                not np.isfinite(
                    self.hierarchical_joint_schur_step_min_low_frequency_efficiency_ratio
                )
                or not 0.0
                < self.hierarchical_joint_schur_step_min_low_frequency_efficiency_ratio
                <= 1.0
            )
        ):
            raise ValueError(
                "hierarchical_joint_schur_step_min_low_frequency_efficiency_ratio "
                "must be None or in (0, 1]"
            )
        if (
            self.hierarchical_joint_schur_step_max_initial_jacobi_low_frequency_efficiency
            is not None
            and (
                not np.isfinite(
                    self.hierarchical_joint_schur_step_max_initial_jacobi_low_frequency_efficiency
                )
                or self.hierarchical_joint_schur_step_max_initial_jacobi_low_frequency_efficiency
                <= 0.0
            )
        ):
            raise ValueError(
                "hierarchical_joint_schur_step_max_initial_jacobi_low_frequency_efficiency "
                "must be None or finite and positive"
            )
        if (
            self.hierarchical_joint_schur_step_max_initial_diagonal_low_frequency_efficiency
            is not None
            and (
                not np.isfinite(
                    self.hierarchical_joint_schur_step_max_initial_diagonal_low_frequency_efficiency
                )
                or self.hierarchical_joint_schur_step_max_initial_diagonal_low_frequency_efficiency
                <= 0.0
            )
        ):
            raise ValueError(
                "hierarchical_joint_schur_step_max_initial_diagonal_low_frequency_efficiency "
                "must be None or finite and positive"
            )
        if (
            self.hierarchical_joint_schur_step_max_initial_diagonal_reduction_fraction
            is not None
            and (
                not np.isfinite(
                    self.hierarchical_joint_schur_step_max_initial_diagonal_reduction_fraction
                )
                or not 0.0
                < self.hierarchical_joint_schur_step_max_initial_diagonal_reduction_fraction
                <= 1.0
            )
        ):
            raise ValueError(
                "hierarchical_joint_schur_step_max_initial_diagonal_reduction_fraction "
                "must be None or in (0, 1]"
            )
        if self.hierarchical_coarse_loop_continuation_stages < 1:
            raise ValueError(
                "hierarchical_coarse_loop_continuation_stages must be positive"
            )
        if (
            self.hierarchical_coarse_max_loop_span is not None
            and self.hierarchical_coarse_max_loop_span < 1
        ):
            raise ValueError("hierarchical_coarse_max_loop_span must be positive")
        if (
            self.hierarchical_coarse_adaptive_full_max_nfev is not None
            and self.hierarchical_coarse_adaptive_full_max_nfev < 1
        ):
            raise ValueError(
                "hierarchical_coarse_adaptive_full_max_nfev must be positive"
            )
        if (
            self.hierarchical_coarse_adaptive_budget_fallback_max_nfev is not None
            and self.hierarchical_coarse_adaptive_budget_fallback_max_nfev < 1
        ):
            raise ValueError(
                "hierarchical_coarse_adaptive_budget_fallback_max_nfev must be positive"
            )
        if self.hierarchical_coarse_continuation_max_nfev < 1:
            raise ValueError(
                "hierarchical_coarse_continuation_max_nfev must be positive"
            )
        if self.hierarchical_coarse_boundary_restart_passes < 0:
            raise ValueError(
                "hierarchical_coarse_boundary_restart_passes must be nonnegative"
            )
        if self.hierarchical_coarse_boundary_restart_max_nfev < 1:
            raise ValueError(
                "hierarchical_coarse_boundary_restart_max_nfev must be positive"
            )
        if self.hierarchical_solver_workers < 1:
            raise ValueError("hierarchical_solver_workers must be positive")
        if self.local_chunk_workers < 1:
            raise ValueError("local_chunk_workers must be positive")
        if self.loop_forward_workers < 1:
            raise ValueError("loop_forward_workers must be positive")
        if self.postprocess_loop_workers < 1:
            raise ValueError("postprocess_loop_workers must be positive")
        if self.artifact_load_workers < 1:
            raise ValueError("artifact_load_workers must be positive")
        if self.forward_artifact_write_workers < 1:
            raise ValueError("forward_artifact_write_workers must be positive")
        if self.forward_artifact_field_profile not in {
            "full",
            *_HIERARCHICAL_TRAJECTORY_FIELD_PROFILES,
        }:
            raise ValueError(
                "unsupported forward_artifact_field_profile: "
                f"{self.forward_artifact_field_profile!r}"
            )
        if self.forward_artifact_field_profile in _HIERARCHICAL_TRAJECTORY_FIELD_PROFILES:
            if self.forward_artifact_version != 2:
                raise ValueError(
                    f"{self.forward_artifact_field_profile} requires "
                    "forward_artifact_version=2"
                )
            if self.postprocess != "hierarchical_pose_graph":
                raise ValueError(
                    f"{self.forward_artifact_field_profile} requires "
                    "postprocess='hierarchical_pose_graph'"
                )
            if self.save_chunk_predictions:
                raise ValueError(
                    f"{self.forward_artifact_field_profile} requires "
                    "save_chunk_predictions=False"
                )
        if self.forward_worker_direct_artifact_publish:
            if self.forward_artifact_version != 2:
                raise ValueError(
                    "forward_worker_direct_artifact_publish requires "
                    "forward_artifact_version=2"
                )
            if (
                self.forward_artifact_field_profile
                not in _HIERARCHICAL_TRAJECTORY_FIELD_PROFILES
            ):
                raise ValueError(
                    "forward_worker_direct_artifact_publish requires "
                    "forward_artifact_field_profile to be a hierarchical "
                    "trajectory profile"
                )
            if self.local_chunk_workers <= 1:
                raise ValueError(
                    "forward_worker_direct_artifact_publish requires "
                    "local_chunk_workers > 1"
                )
        if self.forward_persistent_gpu_workers:
            if not self.forward_worker_direct_artifact_publish:
                raise ValueError(
                    "forward_persistent_gpu_workers requires "
                    "forward_worker_direct_artifact_publish=True"
                )
            if self.loop_forward_workers <= 1:
                raise ValueError(
                    "forward_persistent_gpu_workers requires loop_forward_workers > 1"
                )
            if self.local_chunk_workers != self.loop_forward_workers:
                raise ValueError(
                    "forward_persistent_gpu_workers requires equal chunk and loop workers"
                )
        if any(gpu_id < 0 for gpu_id in self.local_chunk_gpu_ids):
            raise ValueError("local_chunk_gpu_ids must be nonnegative")
        if len(set(self.local_chunk_gpu_ids)) != len(self.local_chunk_gpu_ids):
            raise ValueError("local_chunk_gpu_ids must not contain duplicates")
        if self.local_chunk_workers > 1 and len(self.local_chunk_gpu_ids) < self.local_chunk_workers:
            raise ValueError("local_chunk_gpu_ids must provide at least one GPU per worker")
        if self.loop_forward_workers > 1 and len(self.local_chunk_gpu_ids) < self.loop_forward_workers:
            raise ValueError("local_chunk_gpu_ids must provide at least one GPU per loop worker")
        if self.forward_artifact_version not in {1, 2}:
            raise ValueError("forward_artifact_version must be 1 or 2")


@dataclass(frozen=True)
class LongReconstructionResult:
    chunk_ranges: list[ChunkRange]
    chunks: list[ChunkPrediction]
    alignments: list[Sim3]
    edges: list[Sim3Edge]
    trajectory_c2w: np.ndarray
    intrinsic: np.ndarray
    diagnostics: dict[str, Any]
    reliability_trace: dict[str, Any] | None = None
    frame_poses: list[Sim3] | None = None


@dataclass(frozen=True)
class ForwardStageResult:
    """Small, RAM-safe result returned by :meth:`ReconstructionRunner.run_forward`."""

    artifact_dir: Path
    protocol_id: str
    num_frames: int
    num_chunks: int
    num_loop_constraints: int
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class BasisStageResult:
    """Alpha-independent post-processing basis published by the CPU replay stage."""

    basis_dir: Path
    forward_protocol_id: str
    basis_protocol_id: str
    num_frames: int
    num_chunks: int
    num_edges: int
    diagnostics: dict[str, Any]


def generate_chunk_ranges(num_frames: int, chunk_size: int, overlap: int) -> list[ChunkRange]:
    """Cover all frames with fixed-size chunks and one possibly shorter tail chunk."""
    if num_frames <= 0:
        return []
    if chunk_size < 2:
        raise ValueError("chunk_size must be at least 2")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    if num_frames <= chunk_size:
        return [ChunkRange(0, num_frames)]

    stride = chunk_size - overlap
    ranges = []
    for start in range(0, num_frames, stride):
        end = min(start + chunk_size, num_frames)
        ranges.append(ChunkRange(start, end))
        if end == num_frames:
            break
    return ranges


def chunk_ranges_records(ranges: Sequence[ChunkRange]) -> list[dict[str, int]]:
    return [{"start": int(item.start), "end": int(item.end)} for item in ranges]


def chunk_ranges_identity(ranges: Sequence[ChunkRange]) -> str:
    records = chunk_ranges_records(ranges)
    if not records:
        return "ranges_empty"
    return (
        f"ranges_{len(records)}"
        f"_first_{records[0]['start']}_{records[0]['end']}"
        f"_last_{records[-1]['start']}_{records[-1]['end']}"
    )


def summarize_chunk_ranges(
    ranges: Sequence[ChunkRange],
    *,
    num_frames: int,
) -> dict[str, int | bool | None]:
    overlaps = [
        int(previous.end - current.start)
        for previous, current in zip(ranges, ranges[1:])
    ]
    return {
        "num_frames": int(num_frames),
        "num_chunks": len(ranges),
        "total_processed_frames": sum(int(item.size) for item in ranges),
        "max_chunk_size": max((int(item.size) for item in ranges), default=0),
        "min_adjacent_overlap": min(overlaps) if overlaps else None,
        "max_adjacent_overlap": max(overlaps) if overlaps else None,
        "full_coverage": bool(
            (not ranges and num_frames == 0)
            or (
                bool(ranges)
                and ranges[0].start == 0
                and ranges[-1].end == num_frames
                and all(
                    current.start <= previous.end
                    for previous, current in zip(ranges, ranges[1:])
                )
            )
        ),
    }


def validate_explicit_chunk_ranges(
    num_frames: int,
    ranges: Sequence[ChunkRange | Mapping[str, Any] | Sequence[int]],
    *,
    max_chunk_size: int,
    min_overlap: int = 0,
    expected_count: int | None = None,
    expected_total_processed_frames: int | None = None,
) -> list[ChunkRange]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive for explicit chunk ranges")
    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be positive")
    if min_overlap < 0:
        raise ValueError("min_overlap must be non-negative")
    parsed: list[ChunkRange] = []
    for index, item in enumerate(ranges):
        if isinstance(item, ChunkRange):
            parsed.append(item)
        elif isinstance(item, Mapping):
            parsed.append(ChunkRange(int(item["start"]), int(item["end"])))
        else:
            values = list(item)
            if len(values) != 2:
                raise ValueError(f"explicit range {index} must have start and end")
            parsed.append(ChunkRange(int(values[0]), int(values[1])))
    if not parsed:
        raise ValueError("explicit chunk ranges are empty")
    if expected_count is not None and len(parsed) != int(expected_count):
        raise ValueError(
            f"explicit chunk range count {len(parsed)} != expected {expected_count}"
        )
    if parsed[0].start != 0:
        raise ValueError("explicit chunk ranges must start at frame 0")
    if parsed[-1].end != num_frames:
        raise ValueError("explicit chunk ranges must end at num_frames")
    for index, item in enumerate(parsed):
        if item.start < 0 or item.end > num_frames or item.start >= item.end:
            raise ValueError(f"explicit chunk range {index} is out of bounds: {item}")
        if item.size > max_chunk_size:
            raise ValueError(
                f"explicit chunk range {index} size {item.size} exceeds "
                f"max_chunk_size {max_chunk_size}"
            )
        if index > 0:
            previous = parsed[index - 1]
            if item.start <= previous.start:
                raise ValueError("explicit chunk range starts must be strictly ordered")
            if item.start > previous.end:
                raise ValueError("explicit chunk ranges leave an uncovered frame gap")
            adjacent_overlap = previous.end - item.start
            if adjacent_overlap < min_overlap:
                raise ValueError(
                    f"explicit chunk overlap {adjacent_overlap} below minimum "
                    f"{min_overlap} at pair {index - 1}:{index}"
                )
    total_processed = sum(item.size for item in parsed)
    if (
        expected_total_processed_frames is not None
        and total_processed != int(expected_total_processed_frames)
    ):
        raise ValueError(
            "explicit chunk total processed frames "
            f"{total_processed} != expected {expected_total_processed_frames}"
        )
    return parsed


def load_loop_pairs(path: str | Path | None) -> list[LoopPair]:
    if path is None:
        return []
    payload = json.loads(Path(path).read_text())
    raw_pairs = payload.get("loop_pairs", payload) if isinstance(payload, dict) else payload
    pairs = []
    for item in raw_pairs:
        if isinstance(item, dict):
            pairs.append(LoopPair(**item))
        elif len(item) == 2:
            pairs.append(LoopPair(chunk_a=int(item[0]), chunk_b=int(item[1])))
        else:
            raise ValueError(f"unsupported loop pair: {item}")
    return pairs


class ReconstructionRunner:
    """Run a trainable local model and keep hip3r alignment as optional post-processing."""

    def __init__(
        self,
        adapter: Any,
        config: ReconstructionConfig,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.progress = progress

    def run(
        self,
        image_paths: Iterable[str | Path],
        *,
        loop_pairs: Sequence[LoopPair] | None = None,
        output_dir: str | Path | None = None,
        trajectory_only: bool = False,
    ) -> LongReconstructionResult:
        paths = [Path(path) for path in image_paths]
        if not paths:
            raise ValueError("image_paths is empty")
        ranges = generate_chunk_ranges(len(paths), self.config.chunk_size, self.config.overlap)
        chunks, trace_chunks, chunk_diagnostics = self._infer_local_chunks(paths, ranges)
        incremental_adjacent_edges = self._build_incremental_adjacent_edges(
            chunks,
            ranges,
        )

        with self._postprocess_lock():
            self._report(f"estimating {len(loop_pairs or [])} loop edge(s)")
            loops, trace_loops = self._build_loop_edges(paths, ranges, chunks, loop_pairs or [])
            selected_loops, loop_selection = select_loop_edges(
                loops,
                loop_pairs or [],
                policy=self.config.loop_edge_policy,
                budget=self.config.loop_edge_budget,
            )
            self._report(f"running postprocess={self.config.postprocess}")
            post_result = self._postprocess(
                chunks,
                selected_loops,
                ranges,
                adjacent_edges=incremental_adjacent_edges or None,
                trajectory_only=trajectory_only,
            )
        trajectory, intrinsic = _merge_frame_outputs(post_result.chunks, ranges, len(paths))
        result = LongReconstructionResult(
            chunk_ranges=ranges,
            chunks=list(post_result.chunks),
            alignments=post_result.alignments,
            edges=post_result.edges,
            trajectory_c2w=trajectory,
            intrinsic=intrinsic,
            diagnostics={
                **post_result.diagnostics,
                "num_frames": len(paths),
                "num_requested_loops": len(loop_pairs or []),
                "loop_edge_selection": loop_selection,
                "local_chunk_inference": chunk_diagnostics,
                "trajectory_only": bool(trajectory_only),
            },
            reliability_trace=(
                {
                    "version": 1,
                    "read_only": True,
                    "chunks": trace_chunks,
                    "loop_forwards": trace_loops,
                    "edges": [_edge_trace(edge) for edge in post_result.edges],
                    "summary": _reliability_summary(
                        [*trace_chunks, *trace_loops], post_result.edges
                    ),
                }
                if self.config.reliability_trace
                else None
            ),
            frame_poses=post_result.frame_poses,
        )
        if output_dir is not None:
            self._report(f"saving reconstruction to {output_dir}")
            save_long_reconstruction(result, output_dir, paths, self.config)
        return result

    def run_forward(
        self,
        image_paths: Iterable[str | Path],
        *,
        loop_pairs: Sequence[LoopPair] | None = None,
        artifact_dir: str | Path,
        protocol: dict[str, Any],
        explicit_ranges: Sequence[ChunkRange | Mapping[str, Any] | Sequence[int]] | None = None,
    ) -> ForwardStageResult:
        """Run only model forwards and durably publish their ordered outputs.

        ``forward.complete.json`` is written last and is the sole commit marker.
        Serial chunk inference is spilled chunk-by-chunk; it is never accumulated.
        """
        paths = [Path(path) for path in image_paths]
        if not paths:
            raise ValueError("image_paths is empty")
        protocol_field_profile = str(
            protocol.get("forward_artifact_field_profile", "full")
        )
        if protocol_field_profile != self.config.forward_artifact_field_profile:
            raise ValueError(
                "forward artifact field profile mismatch: protocol requests "
                f"{protocol_field_profile!r}, runtime config uses "
                f"{self.config.forward_artifact_field_profile!r}"
            )
        pairs = list(loop_pairs or [])
        if explicit_ranges is None:
            ranges = generate_chunk_ranges(
                len(paths), self.config.chunk_size, self.config.overlap
            )
        else:
            expected_summary = protocol.get("explicit_ranges_summary", {})
            ranges = validate_explicit_chunk_ranges(
                len(paths),
                explicit_ranges,
                max_chunk_size=int(protocol.get("max_chunk_size", self.config.chunk_size)),
                min_overlap=int(protocol.get("explicit_ranges_min_overlap", 0)),
                expected_count=expected_summary.get("num_chunks"),
                expected_total_processed_frames=expected_summary.get(
                    "total_processed_frames"
                ),
            )
            protocol_ranges = protocol.get("explicit_ranges")
            if protocol_ranges is None:
                raise ValueError(
                    "explicit_ranges protocol entry is required when "
                    "run_forward receives explicit_ranges"
                )
            protocol_ranges = validate_explicit_chunk_ranges(
                len(paths),
                protocol_ranges,
                max_chunk_size=int(protocol.get("max_chunk_size", self.config.chunk_size)),
                min_overlap=int(protocol.get("explicit_ranges_min_overlap", 0)),
                expected_count=len(ranges),
                expected_total_processed_frames=sum(item.size for item in ranges),
            )
            if chunk_ranges_records(protocol_ranges) != chunk_ranges_records(ranges):
                raise ValueError("explicit_ranges argument does not match protocol")
            expected_identity = protocol.get("explicit_ranges_identity")
            if expected_identity != chunk_ranges_identity(ranges):
                raise ValueError("explicit_ranges protocol identity mismatch")
            expected_summary = dict(expected_summary)
            observed_summary = summarize_chunk_ranges(ranges, num_frames=len(paths))
            for key in (
                "num_frames",
                "num_chunks",
                "total_processed_frames",
                "max_chunk_size",
                "min_adjacent_overlap",
                "full_coverage",
            ):
                if key in expected_summary and expected_summary[key] != observed_summary[key]:
                    raise ValueError(f"explicit_ranges summary mismatch for {key}")
        root = Path(artifact_dir)
        root.mkdir(parents=True, exist_ok=True)
        (root / "forward.complete.json").unlink(missing_ok=True)
        for name in ("chunks", "loops"):
            shutil.rmtree(root / name, ignore_errors=True)
            (root / name).mkdir()

        chunk_entries: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        chunk_artifact_fields, loop_artifact_fields = _forward_artifact_fields(
            self.config.forward_artifact_field_profile
        )
        artifact_dtype_overrides = _forward_artifact_dtype_overrides(
            self.config.forward_artifact_field_profile
        )
        chunk_publication_started = time.perf_counter()
        chunk_inference_seconds: float | None = None
        chunk_artifact_write_seconds: float | None = None
        direct_publication = self.config.forward_worker_direct_artifact_publish
        loop_requests: list[
            tuple[LoopPair, list[int], list[int], list[int], list[int]]
        ] = []
        for pair in pairs:
            if not (0 <= pair.chunk_a < len(ranges) and 0 <= pair.chunk_b < len(ranges)):
                raise ValueError(f"loop pair chunk index out of range: {pair}")
            if pair.chunk_a == pair.chunk_b:
                raise ValueError(f"loop pair must connect different chunks: {pair}")
            local_a, global_a = _loop_window(
                ranges[pair.chunk_a], pair.frame_a, self.config.loop_window_size
            )
            local_b, global_b = _loop_window(
                ranges[pair.chunk_b], pair.frame_b, self.config.loop_window_size
            )
            loop_requests.append((pair, local_a, local_b, global_a, global_b))

        persistent_phase_diagnostics: dict[str, Any] | None = None
        persistent_published_loops: list[dict[str, Any]] = []
        persistent_without_loops = (
            self.config.forward_persistent_gpu_workers and not loop_requests
        )
        if self.config.forward_persistent_gpu_workers and loop_requests:
            infer_phases = getattr(
                self.adapter, "infer_forward_phases_parallel_to_artifacts", None
            )
            if infer_phases is None:
                raise TypeError(
                    f"{type(self.adapter).__name__} does not support persistent "
                    "parallel direct artifact publication"
                )
            loop_paths = [
                [paths[item] for item in request[3] + request[4]]
                for request in loop_requests
            ]
            chunk_entries, persistent_published_loops, persistent_phase_diagnostics = (
                infer_phases(
                    [paths[item.start : item.end] for item in ranges],
                    loop_paths,
                    gpu_ids=self.config.local_chunk_gpu_ids[
                        : self.config.local_chunk_workers
                    ],
                    artifact_root=root,
                    chunk_relative_paths=[
                        f"chunks/chunk_{index:06d}" for index in range(len(ranges))
                    ],
                    loop_relative_paths=[
                        f"loops/loop_{index:06d}" for index in range(len(loop_paths))
                    ],
                    artifact_version=self.config.forward_artifact_version,
                    chunk_field_names=chunk_artifact_fields,
                    loop_field_names=loop_artifact_fields,
                    artifact_field_dtype_overrides=artifact_dtype_overrides,
                    result_root=self.config.parallel_result_root,
                    progress=self._report,
                )
            )
            _validate_direct_artifact_entries(
                chunk_entries,
                artifact_root=root,
                relative_paths=[
                    f"chunks/chunk_{index:06d}" for index in range(len(ranges))
                ],
                field_names=chunk_artifact_fields,
            )
            _validate_direct_artifact_entries(
                persistent_published_loops,
                artifact_root=root,
                relative_paths=[
                    f"loops/loop_{index:06d}" for index in range(len(loop_paths))
                ],
                field_names=loop_artifact_fields,
            )
            chunk_entries = [
                {**entry, "index": index, "start": item.start, "end": item.end}
                for index, (item, entry) in enumerate(zip(ranges, chunk_entries))
            ]
            chunk_diagnostics = {
                **persistent_phase_diagnostics,
                "mode": "persistent_parallel_worker_direct_artifact_v1",
                "requested_workers": self.config.local_chunk_workers,
                "actual_workers": persistent_phase_diagnostics["actual_workers"],
            }
            chunk_write_workers = int(persistent_phase_diagnostics["actual_workers"])
        elif self.config.local_chunk_workers == 1:
            chunk_inference_seconds = 0.0
            chunk_artifact_write_seconds = 0.0
            for index, item in enumerate(ranges):
                self._report(f"local chunk {index + 1}/{len(ranges)} frames={item.start}:{item.end}")
                inference_started = time.perf_counter()
                prediction = ChunkPrediction(
                    **self.adapter.infer_paths(paths[item.start : item.end]).to_numpy_dict()
                )
                chunk_inference_seconds += time.perf_counter() - inference_started
                artifact_write_started = time.perf_counter()
                entry = _write_prediction_artifact(
                    root, f"chunks/chunk_{index:06d}", prediction,
                    version=self.config.forward_artifact_version,
                    field_names=chunk_artifact_fields,
                    field_dtype_overrides=artifact_dtype_overrides,
                )
                chunk_artifact_write_seconds += (
                    time.perf_counter() - artifact_write_started
                )
                chunk_entries.append(
                    {**entry, "index": index, "start": item.start, "end": item.end}
                )
                trace = self._pop_reliability_trace()
                if trace is not None:
                    traces.append({"chunk_index": index, **trace})
                del prediction
            chunk_diagnostics = {"mode": "serial", "requested_workers": 1, "actual_workers": 1}
            chunk_write_workers = 1
        elif direct_publication:
            chunk_entries, chunk_diagnostics = self._infer_local_chunks_to_artifacts(
                paths,
                ranges,
                artifact_root=root,
                field_names=chunk_artifact_fields,
                field_dtype_overrides=artifact_dtype_overrides,
            )
            if persistent_without_loops:
                chunk_diagnostics = {
                    **chunk_diagnostics,
                    "persistent_gpu_workers_skipped_reason": "no_loop_requests",
                }
            traces = []
            chunk_write_workers = int(chunk_diagnostics["actual_workers"])
        else:
            inference_started = time.perf_counter()
            chunks, traces, chunk_diagnostics = self._infer_local_chunks(paths, ranges)
            chunk_inference_seconds = time.perf_counter() - inference_started

            def publish_chunk(
                indexed: tuple[int, ChunkRange, ChunkPrediction],
            ) -> dict[str, Any]:
                index, item, prediction = indexed
                entry = _write_prediction_artifact(
                    root, f"chunks/chunk_{index:06d}", prediction,
                    version=self.config.forward_artifact_version,
                    field_names=chunk_artifact_fields,
                    field_dtype_overrides=artifact_dtype_overrides,
                )
                return {**entry, "index": index, "start": item.start, "end": item.end}

            indexed_chunks = [
                (index, item, prediction)
                for index, (item, prediction) in enumerate(zip(ranges, chunks))
            ]
            chunk_write_workers = min(
                self.config.forward_artifact_write_workers, len(indexed_chunks)
            )
            artifact_write_started = time.perf_counter()
            if chunk_write_workers > 1:
                self._report(
                    "parallel forward artifact publication "
                    f"chunks={len(indexed_chunks)} workers={chunk_write_workers}"
                )
                with ThreadPoolExecutor(
                    max_workers=chunk_write_workers,
                    thread_name_prefix="hip3r-forward-artifact-write",
                ) as executor:
                    chunk_entries = list(executor.map(publish_chunk, indexed_chunks))
            else:
                chunk_entries = [publish_chunk(item) for item in indexed_chunks]
            chunk_artifact_write_seconds = time.perf_counter() - artifact_write_started
            del indexed_chunks
            del chunks
        chunk_publication_seconds = (
            float(persistent_phase_diagnostics["chunk_wall_seconds"])
            if persistent_phase_diagnostics is not None
            else time.perf_counter() - chunk_publication_started
        )

        loop_phase_started = time.perf_counter()
        loop_predictions: list[ChunkPrediction] = []
        published_loops: list[dict[str, Any]] = []
        loop_diagnostics: dict[str, Any]
        if persistent_phase_diagnostics is not None:
            published_loops = persistent_published_loops
            loop_diagnostics = {
                **persistent_phase_diagnostics,
                "mode": "persistent_parallel_worker_direct_artifact_v1",
                "requested_workers": self.config.loop_forward_workers,
                "actual_workers": persistent_phase_diagnostics["actual_workers"],
            }
        elif direct_publication and self.config.loop_forward_workers > 1 and loop_requests:
            infer_direct = getattr(
                self.adapter, "infer_chunks_parallel_to_artifacts", None
            )
            if infer_direct is None:
                raise TypeError(
                    f"{type(self.adapter).__name__} does not support direct "
                    "parallel artifact publication"
                )
            loop_paths = [
                [paths[item] for item in request[3] + request[4]]
                for request in loop_requests
            ]
            worker_count = min(self.config.loop_forward_workers, len(loop_paths))
            published_loops, diagnostics = infer_direct(
                loop_paths,
                gpu_ids=self.config.local_chunk_gpu_ids[:worker_count],
                artifact_root=root,
                artifact_relative_paths=[
                    f"loops/loop_{index:06d}" for index in range(len(loop_paths))
                ],
                artifact_version=self.config.forward_artifact_version,
                artifact_field_names=loop_artifact_fields,
                artifact_field_dtype_overrides=artifact_dtype_overrides,
                result_root=self.config.parallel_result_root,
                progress=self._report,
            )
            _validate_direct_artifact_entries(
                published_loops,
                artifact_root=root,
                relative_paths=[
                    f"loops/loop_{index:06d}" for index in range(len(loop_paths))
                ],
                field_names=loop_artifact_fields,
            )
            loop_diagnostics = {
                "mode": "parallel_worker_direct_artifact_v2",
                "requested_workers": self.config.loop_forward_workers,
                "actual_workers": worker_count,
                **dict(diagnostics or {}),
            }
        elif self.config.loop_forward_workers > 1 and loop_requests:
            infer_parallel = getattr(self.adapter, "infer_chunks_parallel", None)
            if infer_parallel is None:
                raise TypeError(
                    f"{type(self.adapter).__name__} does not support parallel loop inference"
                )
            loop_paths = [
                [paths[item] for item in request[3] + request[4]]
                for request in loop_requests
            ]
            worker_count = min(self.config.loop_forward_workers, len(loop_paths))
            self._report(
                "parallel loop windows "
                f"count={len(loop_paths)} workers={worker_count} "
                f"gpus={list(self.config.local_chunk_gpu_ids[:worker_count])}"
            )
            predictions, diagnostics = infer_parallel(
                loop_paths,
                gpu_ids=self.config.local_chunk_gpu_ids[:worker_count],
                result_root=self.config.parallel_result_root,
                progress=self._report,
            )
            loop_predictions = [
                ChunkPrediction(**prediction.to_numpy_dict()) for prediction in predictions
            ]
            loop_diagnostics = {
                "mode": "parallel",
                "requested_workers": self.config.loop_forward_workers,
                "actual_workers": worker_count,
                **dict(diagnostics or {}),
            }
        else:
            infer_loop_paths = getattr(self.adapter, "infer_loop_paths", self.adapter.infer_paths)
            for index, request in enumerate(loop_requests):
                prediction = ChunkPrediction(
                    **infer_loop_paths(
                        [paths[item] for item in request[3] + request[4]]
                    ).to_numpy_dict()
                )
                if direct_publication:
                    published_loops.append(
                        _write_prediction_artifact(
                            root,
                            f"loops/loop_{index:06d}",
                            prediction,
                            version=self.config.forward_artifact_version,
                            field_names=loop_artifact_fields,
                            field_dtype_overrides=artifact_dtype_overrides,
                        )
                    )
                    del prediction
                else:
                    loop_predictions.append(prediction)
            loop_diagnostics = {
                "mode": (
                    "serial_direct_artifact_v2" if direct_publication else "serial"
                ),
                "requested_workers": self.config.loop_forward_workers,
                "actual_workers": 1 if loop_requests else 0,
            }

        loop_publication_started = (
            None
            if persistent_phase_diagnostics is not None
            else (loop_phase_started if direct_publication else time.perf_counter())
        )

        def publish_loop(indexed: tuple[int, ChunkPrediction]) -> dict[str, Any]:
            index, joint = indexed
            entry = _write_prediction_artifact(
                root, f"loops/loop_{index:06d}", joint,
                version=self.config.forward_artifact_version,
                field_names=loop_artifact_fields,
                field_dtype_overrides=artifact_dtype_overrides,
            )
            return entry

        indexed_loops = list(enumerate(loop_predictions))
        if direct_publication:
            loop_write_workers = int(loop_diagnostics["actual_workers"])
        else:
            loop_write_workers = min(
                self.config.forward_artifact_write_workers, len(indexed_loops)
            ) if indexed_loops else 0
            if loop_write_workers > 1:
                with ThreadPoolExecutor(
                    max_workers=loop_write_workers,
                    thread_name_prefix="hip3r-forward-artifact-write",
                ) as executor:
                    published_loops = list(executor.map(publish_loop, indexed_loops))
            else:
                published_loops = [publish_loop(item) for item in indexed_loops]
        del indexed_loops
        del loop_predictions

        loop_entries: list[dict[str, Any]] = []
        for index, ((pair, local_a, local_b, global_a, global_b), entry) in enumerate(
            zip(loop_requests, published_loops)
        ):
            loop_entries.append(
                {
                    **entry,
                    "index": index,
                    "pair": pair.__dict__,
                    "local_a": local_a,
                    "local_b": local_b,
                    "global_a": global_a,
                    "global_b": global_b,
                }
            )
            trace = self._pop_reliability_trace()
            if trace is not None:
                traces.append({"loop_index": index, **trace})
        loop_publication_seconds = (
            float(persistent_phase_diagnostics["loop_wall_seconds"])
            if persistent_phase_diagnostics is not None
            else time.perf_counter() - loop_publication_started
        )

        backend_activation_fn = getattr(self.adapter, "activation_report", None)
        backend_activation = (
            dict(backend_activation_fn()) if callable(backend_activation_fn) else None
        )

        protocol_id = _protocol_id(protocol)
        manifest = {
            "schema": f"hip3r_forward_artifacts_v{self.config.forward_artifact_version}",
            "protocol": _json_safe(protocol),
            "protocol_id": protocol_id,
            "images": [str(path) for path in paths],
            "chunks": chunk_entries,
            "loops": loop_entries,
            "diagnostics": {
                "local_chunk_inference": chunk_diagnostics,
                "loop_window_inference": loop_diagnostics,
                "local_backend_activation": backend_activation,
                "forward_artifact_publication": {
                    "mode": (
                        "worker_direct_atomic_v2"
                        if direct_publication
                        else (
                            "bounded_ordered_parallel"
                            if max(chunk_write_workers, loop_write_workers) > 1
                            else "serial"
                        )
                    ),
                    "requested_workers": (
                        self.config.local_chunk_workers
                        if direct_publication
                        else self.config.forward_artifact_write_workers
                    ),
                    "configured_parent_write_workers": (
                        self.config.forward_artifact_write_workers
                    ),
                    "chunk_actual_workers": chunk_write_workers,
                    "loop_actual_workers": loop_write_workers,
                    "chunk_wall_seconds": chunk_publication_seconds,
                    "loop_wall_seconds": loop_publication_seconds,
                    "chunk_inference_seconds": chunk_inference_seconds,
                    "chunk_artifact_write_seconds": chunk_artifact_write_seconds,
                    "chunk_unattributed_seconds": (
                        max(
                            0.0,
                            chunk_publication_seconds
                            - chunk_inference_seconds
                            - chunk_artifact_write_seconds,
                        )
                        if chunk_inference_seconds is not None
                        and chunk_artifact_write_seconds is not None
                        else None
                    ),
                    "field_profile": self.config.forward_artifact_field_profile,
                    "field_dtype_overrides": {
                        name: dtype.str
                        for name, dtype in artifact_dtype_overrides.items()
                    },
                    "chunk_fields": list(chunk_artifact_fields),
                    "loop_fields": list(loop_artifact_fields),
                    "chunk_bytes": _artifact_entries_nbytes(chunk_entries),
                    "loop_bytes": _artifact_entries_nbytes(loop_entries),
                    "worker_direct_artifact_publish": direct_publication,
                    "persistent_gpu_workers_configured": (
                        self.config.forward_persistent_gpu_workers
                    ),
                    "persistent_gpu_workers_effective": (
                        persistent_phase_diagnostics is not None
                    ),
                    "persistent_gpu_worker_mode": (
                        persistent_phase_diagnostics["mode"]
                        if persistent_phase_diagnostics is not None
                        else (
                            "not_applicable_no_loop_requests"
                            if persistent_without_loops
                            else "disabled"
                        )
                    ),
                    "chunk_wall_scope": (
                        "inference_and_publication"
                        if direct_publication
                        else "publication_only"
                    ),
                    "loop_wall_scope": (
                        "inference_and_publication"
                        if direct_publication
                        else "publication_only"
                    ),
                },
                "reliability_traces": traces,
            },
        }
        _atomic_write_json(root / "manifest.json", manifest)
        manifest_identity = _file_identity_file(root / "manifest.json")
        if self.config.forward_artifact_version == 2:
            _write_validation_receipt_from_forward_manifest(
                root,
                manifest,
                manifest_id=manifest_identity,
            )
        complete = {
            "schema": f"hip3r_forward_complete_v{self.config.forward_artifact_version}",
            "protocol_id": protocol_id,
            "manifest": "manifest.json",
            "manifest_id": manifest_identity,
            "num_frames": len(paths),
            "num_chunks": len(chunk_entries),
            "num_loop_constraints": len(loop_entries),
            "persistent_gpu_workers": {
                "configured": self.config.forward_persistent_gpu_workers,
                "effective": persistent_phase_diagnostics is not None,
                "mode": (
                    persistent_phase_diagnostics["mode"]
                    if persistent_phase_diagnostics is not None
                    else (
                        "not_applicable_no_loop_requests"
                        if persistent_without_loops
                        else "disabled"
                    )
                ),
                "actual_workers": (
                    int(persistent_phase_diagnostics["actual_workers"])
                    if persistent_phase_diagnostics is not None
                    else 0
                ),
                "model_loads": (
                    int(persistent_phase_diagnostics["model_loads"])
                    if persistent_phase_diagnostics is not None
                    else 0
                ),
                "separate_loop_model_loads_avoided": (
                    int(
                        persistent_phase_diagnostics[
                            "separate_loop_model_loads_avoided"
                        ]
                    )
                    if persistent_phase_diagnostics is not None
                    else 0
                ),
            },
        }
        _atomic_write_json(root / "forward.complete.json", complete)
        return ForwardStageResult(
            artifact_dir=root,
            protocol_id=protocol_id,
            num_frames=len(paths),
            num_chunks=len(chunk_entries),
            num_loop_constraints=len(loop_entries),
            diagnostics=manifest["diagnostics"],
        )

    def run_basis(
        self,
        artifact_dir: str | Path,
        *,
        forward_protocol: dict[str, Any],
        basis_dir: str | Path,
        basis_protocol: dict[str, Any],
        edge_transformer: Callable[
            [Sequence[Sim3Edge], Sequence[Sim3Edge]],
            tuple[Sequence[Sim3Edge], Sequence[Sim3Edge], dict[str, Any]],
        ]
        | None = None,
        edge_transformer_protocol: dict[str, Any] | None = None,
        edge_cache_namespace: str | None = None,
        edge_stability_strata: int = 0,
        reuse_coarse_anchor_with_edge_transform: bool = False,
        coarse_anchor_cache_key: str | None = None,
    ) -> BasisStageResult:
        """Build and publish an alpha-independent post-processing basis."""
        if (edge_transformer is None) != (edge_transformer_protocol is None):
            raise ValueError(
                "edge_transformer and edge_transformer_protocol must be provided together"
            )
        if edge_transformer is not None and self.config.postprocess != "hierarchical_pose_graph":
            raise ValueError("edge_transformer requires postprocess='hierarchical_pose_graph'")
        if reuse_coarse_anchor_with_edge_transform and edge_transformer is None:
            raise ValueError(
                "reuse_coarse_anchor_with_edge_transform requires an edge transformer"
            )
        if coarse_anchor_cache_key is not None and not (
            reuse_coarse_anchor_with_edge_transform
            and edge_transformer is not None
        ):
            raise ValueError(
                "coarse_anchor_cache_key requires transformed-edge Coarse-anchor reuse"
            )
        if (
            reuse_coarse_anchor_with_edge_transform
            and not self.config.hierarchical_freeze_coarse_alignments
        ):
            raise ValueError(
                "transformed-edge Coarse-anchor reuse requires frozen Coarse alignments"
            )
        if self.config.postprocess == "pose_graph":
            return self._run_pose_graph_basis(
                artifact_dir,
                forward_protocol=forward_protocol,
                basis_dir=basis_dir,
                basis_protocol=basis_protocol,
            )
        if self.config.postprocess != "hierarchical_pose_graph":
            raise ValueError(
                "basis stage requires postprocess='pose_graph' or "
                "'hierarchical_pose_graph'"
            )
        root = Path(artifact_dir)
        manifest = _load_forward_manifest(root, forward_protocol)
        if edge_stability_strata not in (0, 4):
            raise ValueError("edge_stability_strata must be zero or four")
        edge_cache_protocol = _edge_cache_protocol(
            self.config, edge_stability_strata=edge_stability_strata
        )
        if edge_cache_namespace is not None:
            edge_cache_protocol = {
                **edge_cache_protocol,
                "namespace": str(edge_cache_namespace),
            }
        edge_cache_manifest_id = str(
            json.loads((root / "forward.complete.json").read_text()).get(
                "manifest_id", ""
            )
        )
        edge_cache_key = _postprocess_edge_cache_path(
            root,
            edge_cache_manifest_id,
            edge_cache_protocol,
        ).stem
        edge_cache_load_started = time.perf_counter()
        edge_cache = _load_postprocess_edge_cache(
            root,
            manifest,
            edge_cache_protocol,
            workers=self.config.artifact_load_workers,
        )
        edge_cache_load_seconds = time.perf_counter() - edge_cache_load_started
        coarse_cache_protocol = None
        coarse_cache = None
        needs_coarse_anchor = bool(
            self.config.hierarchical_freeze_coarse_alignments
            or self.config.hierarchical_joint_coarse_warm_start
        )
        if (
            needs_coarse_anchor
            and self.config.chain_global_scale_gauge_centering_v1
            and edge_cache is not None
            and not edge_cache.get("loops")
        ):
            needs_coarse_anchor = False
        if coarse_anchor_cache_key is not None:
            if not needs_coarse_anchor:
                raise ValueError("explicit Coarse-anchor cache is not used by this graph")
            coarse_cache = _load_coarse_anchor_cache_by_key(
                root,
                str(coarse_anchor_cache_key),
                self.config,
                expected_num_chunks=len(manifest.get("chunks", [])),
            )
            if coarse_cache is None:
                raise FileNotFoundError(
                    f"validated Coarse-anchor cache is missing: {coarse_anchor_cache_key}"
                )
            coarse_cache_protocol = coarse_cache["protocol"]
        elif (
            edge_cache is not None
            and needs_coarse_anchor
            and (
                edge_transformer is None
                or reuse_coarse_anchor_with_edge_transform
            )
        ):
            coarse_cache_protocol = _coarse_anchor_cache_protocol(
                self.config,
                edge_cache_key=edge_cache_key,
                edge_cache_file_identity=edge_cache["cache_file_identity"],
            )
            coarse_cache = _load_coarse_anchor_cache(
                root,
                coarse_cache_protocol,
                expected_num_chunks=len(manifest.get("chunks", [])),
            )
        analytic_prepare_executor: ThreadPoolExecutor | None = None
        analytic_prepare_future = None
        analytic_prepare_started: float | None = None
        if (
            self.config.use_jac_sparsity
            and self.config.hierarchical_freeze_coarse_alignments
            and self.config.hierarchical_coarse_analytic_jacobian
            and coarse_cache is None
        ):
            analytic_prepare_started = time.perf_counter()
            analytic_prepare_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="hip3r-coarse-analytic-prepare",
            )
            analytic_prepare_future = analytic_prepare_executor.submit(
                prepare_torch_pose_graph_analytic_runtime
            )
        phase_started = time.perf_counter()
        try:
            manifest, paths, ranges, chunks, artifact_load_diagnostics = (
                self._load_basis_forward_inputs(
                    root,
                    forward_protocol,
                    manifest=manifest,
                    required_fields=(
                        ("extrinsic_c2w",)
                        if edge_cache is not None
                        else _BASIS_CHUNK_REQUIRED_FIELDS
                    ),
                )
            )
        except BaseException:
            if analytic_prepare_executor is not None:
                analytic_prepare_executor.shutdown(wait=True)
            raise
        basis_phase_seconds = {
            "edge_cache_validation_loading": edge_cache_load_seconds,
            "artifact_validation_loading": time.perf_counter() - phase_started,
        }
        analytic_prepare_wait_seconds = 0.0
        if analytic_prepare_future is not None:
            prepare_wait_started = time.perf_counter()
            try:
                analytic_prepare_future.result()
            finally:
                assert analytic_prepare_executor is not None
                analytic_prepare_executor.shutdown(wait=True)
            analytic_prepare_wait_seconds = time.perf_counter() - prepare_wait_started
            artifact_load_diagnostics = {
                **artifact_load_diagnostics,
                "coarse_analytic_runtime_preparation": {
                    "mode": "background_overlap",
                    "elapsed_since_submit_seconds": (
                        time.perf_counter() - analytic_prepare_started
                        if analytic_prepare_started is not None
                        else 0.0
                    ),
                    "foreground_wait_seconds": analytic_prepare_wait_seconds,
                },
            }
        basis_phase_seconds["coarse_analytic_runtime_foreground_wait"] = (
            analytic_prepare_wait_seconds
        )
        basis_root = Path(basis_dir)
        basis_root.mkdir(parents=True, exist_ok=True)
        (basis_root / "basis.complete.json").unlink(missing_ok=True)
        (basis_root / "basis.gate_failed.json").unlink(missing_ok=True)
        forward_protocol_id = _protocol_id(forward_protocol)
        basis_protocol_id = _protocol_id(basis_protocol)

        def write_gate_failed(
            *,
            reason: str,
            message: str,
            exception: BaseException | None = None,
            convergence_gate: dict[str, Any] | None = None,
            hierarchical_diagnostics: dict[str, Any] | None = None,
            solver_diagnostics: dict[str, Any] | None = None,
            postprocess_edge_cache: dict[str, Any] | None = None,
            coarse_anchor_cache: dict[str, Any] | None = None,
        ) -> None:
            payload: dict[str, Any] = {
                "schema": "hip3r_hierarchical_basis_gate_failed_v1",
                "reason": reason,
                "message": message,
                "forward_protocol_id": forward_protocol_id,
                "basis_protocol_id": basis_protocol_id,
                "basis_stage_phase_seconds": _json_safe_diagnostic(
                    basis_phase_seconds
                ),
            }
            if exception is not None:
                payload["exception_type"] = type(exception).__name__
            if convergence_gate is not None:
                payload["convergence_gate"] = _json_safe_diagnostic(convergence_gate)
            if solver_diagnostics is None and hierarchical_diagnostics is not None:
                solver_diagnostics = _hierarchical_solver_failure_diagnostics(
                    hierarchical_diagnostics
                )
            if solver_diagnostics is not None:
                payload["solver_diagnostics"] = _json_safe_diagnostic(
                    solver_diagnostics
                )
            if hierarchical_diagnostics is not None:
                payload["hierarchical_diagnostics"] = _json_safe_diagnostic(
                    hierarchical_diagnostics
                )
            if postprocess_edge_cache is not None:
                payload["postprocess_edge_cache"] = _json_safe_diagnostic(
                    postprocess_edge_cache
                )
            if coarse_anchor_cache is not None:
                payload["coarse_anchor_cache"] = _json_safe_diagnostic(
                    coarse_anchor_cache
                )
            _atomic_write_json(basis_root / "basis.gate_failed.json", payload)

        phase_started = time.perf_counter()
        if edge_cache is None:
            loops, _, loop_selection, loop_diagnostics = (
                self._estimate_staged_loop_edges(
                    root, manifest, chunks,
                    stability_strata=edge_stability_strata,
                )
            )
            adjacent_edges = None
            edge_cache_diagnostics = {
                "mode": "miss",
                "schema": _POSTPROCESS_EDGE_CACHE_SCHEMA,
                "key": edge_cache_key,
                "load_wall_seconds": edge_cache_load_seconds,
            }
        else:
            loops = edge_cache["loops"]
            loop_selection = edge_cache["loop_selection"]
            loop_diagnostics = edge_cache["loop_diagnostics"]
            edge_cache_diagnostics = {
                "mode": "hit",
                "schema": _POSTPROCESS_EDGE_CACHE_SCHEMA,
                "key": edge_cache_key,
                "input_identity_workers": edge_cache["input_identity_workers"],
                "load_wall_seconds": edge_cache_load_seconds,
            }
            adjacent_edges = edge_cache["adjacent"]
        basis_phase_seconds["loop_edge_estimation"] = time.perf_counter() - phase_started
        raw_loops = list(loops)
        raw_adjacent_edges = None if adjacent_edges is None else list(adjacent_edges)
        edge_transform_diagnostics: dict[str, Any] | None = None
        if edge_transformer is not None:
            if adjacent_edges is None:
                processor = ReconstructionPostProcessor(
                    overlap=self.config.overlap,
                    confidence_threshold=self.config.confidence_threshold,
                    max_points=self.config.max_alignment_points,
                    max_iterations=self.config.max_sim3_iterations,
                    huber_delta=self.config.huber_delta,
                    stability_strata=edge_stability_strata,
                    use_jac_sparsity=self.config.use_jac_sparsity,
                )
                adjacent_edges = processor.build_adjacent_edges(
                    chunks,
                    chunk_ranges=ranges,
                    workers=self.config.postprocess_loop_workers,
                )
                raw_adjacent_edges = list(adjacent_edges)
            transformed_adjacent, transformed_loops, transform_diagnostics = (
                edge_transformer(tuple(adjacent_edges), tuple(loops))
            )
            adjacent_edges = list(transformed_adjacent)
            loops = list(transformed_loops)
            if len(adjacent_edges) != len(raw_adjacent_edges or ()):
                raise ValueError("edge transformer changed adjacent-edge count")
            allows_loop_filtering = bool(
                edge_transformer_protocol.get("allows_loop_edge_filtering", False)
            )
            if len(loops) != len(raw_loops) and not (
                allows_loop_filtering and len(loops) < len(raw_loops)
            ):
                raise ValueError("edge transformer changed loop-edge count")
            edge_transform_diagnostics = {
                "protocol": _json_safe(edge_transformer_protocol),
                "protocol_id": _protocol_id(edge_transformer_protocol),
                "diagnostics": _json_safe(transform_diagnostics),
                "allows_loop_edge_filtering": allows_loop_filtering,
            }
        phase_started = time.perf_counter()
        try:
            basis, edges, coarse_anchor_for_cache = self._build_hierarchical_basis(
                chunks,
                loops,
                ranges,
                adjacent_edges=adjacent_edges,
                coarse_anchor_result=coarse_cache,
                edge_stability_strata=edge_stability_strata,
            )
        except Exception as exc:
            basis_phase_seconds["graph_solve"] = time.perf_counter() - phase_started
            diagnostics = getattr(exc, "diagnostics", None)
            write_gate_failed(
                reason="basis_stage_exception",
                message=str(exc),
                exception=exc,
                hierarchical_diagnostics=diagnostics
                if isinstance(diagnostics, dict)
                else None,
                postprocess_edge_cache=edge_cache_diagnostics,
            )
            raise
        basis_phase_seconds["graph_solve"] = time.perf_counter() - phase_started
        coarse_cache_diagnostics = None
        try:
            if edge_cache is None:
                edge_cache_publication = _write_postprocess_edge_cache(
                    root,
                    manifest,
                    edge_cache_protocol,
                    adjacent=(
                        raw_adjacent_edges
                        if raw_adjacent_edges is not None
                        else edges[: max(len(chunks) - 1, 0)]
                    ),
                    loops=raw_loops,
                    loop_selection=loop_selection,
                    loop_diagnostics=loop_diagnostics,
                    workers=self.config.artifact_load_workers,
                )
                if edge_cache_publication is not None:
                    edge_cache_diagnostics["published"] = True
                    if (
                        needs_coarse_anchor
                        and edge_transformer is None
                        and not (
                            self.config.chain_global_scale_gauge_centering_v1
                            and not loops
                        )
                    ):
                        coarse_cache_protocol = _coarse_anchor_cache_protocol(
                            self.config,
                            edge_cache_key=str(edge_cache_publication["key"]),
                            edge_cache_file_identity=str(
                                edge_cache_publication["cache_file_identity"]
                            ),
                        )
            factorized_no_loop_cache_disabled = bool(
                self.config.chain_global_scale_gauge_centering_v1 and not loops
            )
            coarse_cache_diagnostics = {
                "mode": (
                    "disabled_factorized_adjacent_chain_no_absolute_coarse_solve"
                    if factorized_no_loop_cache_disabled
                    else ("hit" if coarse_cache is not None else "miss")
                ),
                "schema": _COARSE_ANCHOR_CACHE_SCHEMA,
                "key": (
                    _protocol_id(coarse_cache_protocol)
                    if coarse_cache_protocol is not None
                    else None
                ),
                "reused_with_edge_transform": bool(
                    reuse_coarse_anchor_with_edge_transform
                ),
                "lookup": (
                    "explicit_key"
                    if coarse_anchor_cache_key is not None
                    else "derived_from_current_edge_cache"
                ),
            }
            if (
                coarse_cache is None
                and coarse_cache_protocol is not None
                and coarse_anchor_for_cache is not None
            ):
                # Publish the independent coarse solve that initialized the fine
                # problem.  In Joint mode the final coarse variables have moved and
                # are not a valid cache value, whereas this pre-Joint anchor is.
                coarse_cache_diagnostics["published"] = _write_coarse_anchor_cache(
                    root,
                    coarse_cache_protocol,
                    alignments=coarse_anchor_for_cache["alignments"],
                    diagnostics=coarse_anchor_for_cache["diagnostics"],
                )
            if self.config.hierarchical_require_solver_convergence:
                basis = replace(
                    basis,
                    diagnostics=_annotate_coarse_max_nfev_stationarity_acceptance(
                        basis.diagnostics,
                        threshold=(
                            self.config.hierarchical_coarse_accept_max_nfev_optimality
                        ),
                    ),
                )
                try:
                    convergence_gate = _require_hierarchical_solver_convergence(
                        basis.diagnostics
                    )
                except HierarchicalSolverConvergenceError as exc:
                    write_gate_failed(
                        reason="solver_not_converged",
                        message=str(exc),
                        exception=exc,
                        convergence_gate=getattr(exc, "diagnostics", {}),
                        hierarchical_diagnostics=basis.diagnostics,
                        postprocess_edge_cache=edge_cache_diagnostics,
                        coarse_anchor_cache=coarse_cache_diagnostics,
                    )
                    raise
                basis = replace(
                    basis,
                    diagnostics={
                        **basis.diagnostics,
                        "convergence_gate": convergence_gate,
                    },
                )
        except HierarchicalSolverConvergenceError:
            raise
        except Exception as exc:
            diagnostics = getattr(exc, "diagnostics", None)
            write_gate_failed(
                reason="basis_stage_exception",
                message=str(exc),
                exception=exc,
                hierarchical_diagnostics=(
                    diagnostics if isinstance(diagnostics, dict) else basis.diagnostics
                ),
                postprocess_edge_cache=edge_cache_diagnostics,
                coarse_anchor_cache=coarse_cache_diagnostics,
            )
            raise
        basis_serialization_started = time.perf_counter()
        basis_json = _basis_to_json(basis)
        basis_serialization_wall_seconds = (
            time.perf_counter() - basis_serialization_started
        )
        basis_diagnostics_json = basis_json["diagnostics"]
        edge_serialization_started = time.perf_counter()
        edges_json = [
            _edge_to_json(
                edge,
                diagnostics_are_json_safe=edge_cache is not None,
            )
            for edge in edges
        ]
        edge_serialization_wall_seconds = (
            time.perf_counter() - edge_serialization_started
        )
        basis_manifest = {
            "schema": "hip3r_hierarchical_basis_v1",
            "forward_protocol_id": forward_protocol_id,
            "basis_protocol": _json_safe(basis_protocol),
            "basis_protocol_id": basis_protocol_id,
            "images": [str(path) for path in paths],
            "chunk_ranges": [item.__dict__ for item in ranges],
            "basis": basis_json,
            "edges": edges_json,
            "diagnostics": _json_safe({
                "basis": basis_diagnostics_json,
                "basis_stage_phase_seconds": basis_phase_seconds,
                "forward_artifact_loading": artifact_load_diagnostics,
                "loop_edge_selection": loop_selection,
                "staged_loop_edges": loop_diagnostics,
                "postprocess_edge_cache": edge_cache_diagnostics,
                "coarse_anchor_cache": coarse_cache_diagnostics,
                "edge_transform": edge_transform_diagnostics,
                "local_chunk_inference": manifest["diagnostics"]["local_chunk_inference"],
                "basis_manifest_json_backend": (
                    "orjson" if _orjson is not None else "stdlib_json"
                ),
                "basis_manifest_payload_validation": "construction_time_json_safe",
                "basis_sim3_serialization": "batched_contiguous_matrix_build",
                "basis_sim3_serialization_wall_seconds": (
                    basis_serialization_wall_seconds
                ),
                "basis_edge_diagnostics_validation": (
                    "cache_json_decode_safe"
                    if edge_cache is not None
                    else "recursive_json_safe"
                ),
                "basis_edge_serialization_wall_seconds": (
                    edge_serialization_wall_seconds
                ),
            }),
        }
        publication_started = time.perf_counter()
        _atomic_write_json(
            basis_root / "basis_manifest.json",
            basis_manifest,
            compact=True,
            payload_is_json_safe=True,
        )
        manifest_identity = _file_identity_file(basis_root / "basis_manifest.json")
        manifest_publication_seconds = time.perf_counter() - publication_started
        complete = {
            "schema": "hip3r_hierarchical_basis_complete_v1",
            "forward_protocol_id": forward_protocol_id,
            "basis_protocol_id": basis_protocol_id,
            "manifest": "basis_manifest.json",
            "manifest_id": manifest_identity,
            "num_frames": len(paths),
            "num_chunks": len(ranges),
            "num_edges": len(edges),
            "manifest_publication_seconds": manifest_publication_seconds,
            "manifest_json_backend": (
                "orjson" if _orjson is not None else "stdlib_json"
            ),
        }
        _atomic_write_json(basis_root / "basis.complete.json", complete)
        return BasisStageResult(
            basis_dir=basis_root,
            forward_protocol_id=forward_protocol_id,
            basis_protocol_id=basis_protocol_id,
            num_frames=len(paths),
            num_chunks=len(ranges),
            num_edges=len(edges),
            diagnostics=basis_manifest["diagnostics"],
        )

    def _run_pose_graph_basis(
        self,
        artifact_dir: str | Path,
        *,
        forward_protocol: dict[str, Any],
        basis_dir: str | Path,
        basis_protocol: dict[str, Any],
    ) -> BasisStageResult:
        """Run pose-graph post-processing once and publish its immutable outputs."""
        basis_root = Path(basis_dir)
        basis_root.mkdir(parents=True, exist_ok=True)
        (basis_root / "basis.complete.json").unlink(missing_ok=True)
        (basis_root / "basis_manifest.json").unlink(missing_ok=True)
        result = self.run_postprocess(
            artifact_dir,
            protocol=forward_protocol,
            output_dir=basis_root,
        )
        result_files = {
            str(path.relative_to(basis_root)): _file_identity_file(path)
            for path in sorted(basis_root.rglob("*"))
            if path.is_file()
        }
        forward_protocol_id = _protocol_id(forward_protocol)
        basis_protocol_id = _protocol_id(basis_protocol)
        basis_manifest = {
            "schema": "hip3r_pose_graph_basis_v1",
            "forward_protocol_id": forward_protocol_id,
            "basis_protocol": _json_safe(basis_protocol),
            "basis_protocol_id": basis_protocol_id,
            "chunk_ranges": [item.__dict__ for item in result.chunk_ranges],
            "alignments": [_sim3_to_json(item) for item in result.alignments],
            "edges": [_edge_to_json(edge) for edge in result.edges],
            "result_files": result_files,
            "diagnostics": _json_safe(result.diagnostics),
        }
        _atomic_write_json(basis_root / "basis_manifest.json", basis_manifest)
        manifest_identity = _file_identity_file(basis_root / "basis_manifest.json")
        complete = {
            "schema": "hip3r_pose_graph_basis_complete_v1",
            "forward_protocol_id": forward_protocol_id,
            "basis_protocol_id": basis_protocol_id,
            "manifest": "basis_manifest.json",
            "manifest_id": manifest_identity,
            "num_frames": len(result.trajectory_c2w),
            "num_chunks": len(result.chunk_ranges),
            "num_edges": len(result.edges),
        }
        _atomic_write_json(basis_root / "basis.complete.json", complete)
        return BasisStageResult(
            basis_dir=basis_root,
            forward_protocol_id=forward_protocol_id,
            basis_protocol_id=basis_protocol_id,
            num_frames=len(result.trajectory_c2w),
            num_chunks=len(result.chunk_ranges),
            num_edges=len(result.edges),
            diagnostics=result.diagnostics,
        )

    def build_trajectory_replay_artifact(
        self,
        artifact_dir: str | Path,
        *,
        forward_protocol: dict[str, Any],
        replay_dir: str | Path,
    ) -> dict[str, Any]:
        """Publish the minimal forward contract consumed by trajectory-only export."""
        source_root = Path(artifact_dir)
        target_root = Path(replay_dir)
        if source_root.resolve() == target_root.resolve():
            raise ValueError("trajectory replay directory must differ from forward artifacts")
        if target_root.exists() and any(target_root.iterdir()):
            raise FileExistsError(
                f"trajectory replay directory is not empty: {target_root}"
            )

        manifest = _load_forward_manifest(source_root, forward_protocol)
        if manifest.get("schema") != "hip3r_forward_artifacts_v2":
            raise ValueError("trajectory replay requires forward artifact version 2")

        target_root.mkdir(parents=True, exist_ok=True)
        required_fields = _TRAJECTORY_EXPORT_REQUIRED_FIELDS
        copied_fields = 0
        copied_bytes = 0
        for index, entry in enumerate(manifest["chunks"]):
            _load_prediction_artifact_fields(
                source_root,
                entry,
                expected_index=index,
                required_names=required_fields,
            )
            fields = {field["name"]: field for field in entry["fields"]}
            for name in required_fields:
                field = fields[name]
                relative = Path(str(field["path"]))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        f"trajectory replay field path escapes artifact root: {relative}"
                    )
                source = source_root / relative
                destination = target_root / relative
                _atomic_copy_file(source, destination)
                if _file_identity_file(destination) != field["file_identity"]:
                    raise ValueError(
                        f"trajectory replay identity mismatch after copy: {destination}"
                    )
                copied_fields += 1
                copied_bytes += int(destination.stat().st_size)

        source_complete = json.loads(
            (source_root / "forward.complete.json").read_text()
        )
        source_manifest = source_root / source_complete.get("manifest", "manifest.json")
        replay_manifest = target_root / "manifest.json"
        _atomic_copy_file(source_manifest, replay_manifest)
        manifest_id = _file_identity_file(replay_manifest)
        if manifest_id != manifest["_validated_manifest_id"]:
            raise ValueError("trajectory replay manifest identity mismatch after copy")
        complete = {
            "schema": _TRAJECTORY_REPLAY_COMPLETE_SCHEMA,
            "protocol_id": _protocol_id(forward_protocol),
            "manifest": "manifest.json",
            "manifest_id": manifest_id,
            "source_forward_manifest_id": manifest_id,
            "num_frames": len(manifest["images"]),
            "num_chunks": len(manifest["chunks"]),
            "num_loop_constraints": len(manifest["loops"]),
            "required_fields": list(required_fields),
            "field_count": copied_fields,
            "field_bytes": copied_bytes,
        }
        _atomic_write_json(target_root / "trajectory_replay.complete.json", complete)
        return complete

    def run_alpha_export(
        self,
        artifact_dir: str | Path,
        *,
        forward_protocol: dict[str, Any],
        basis_dir: str | Path,
        basis_protocol: dict[str, Any],
        alpha_protocol: dict[str, Any],
        output_dir: str | Path,
    ) -> LongReconstructionResult:
        """Replay a published post-processing basis and export outputs."""
        root = Path(artifact_dir)
        basis_root = Path(basis_dir)
        manifest, forward_artifact_contract = _load_forward_manifest_for_alpha(
            root, forward_protocol
        )
        paths = list(manifest["images"])
        ranges = [
            ChunkRange(int(item["start"]), int(item["end"]))
            for item in manifest["chunks"]
        ]
        basis_manifest = _load_basis_manifest(
            basis_root,
            forward_protocol_id=_protocol_id(forward_protocol),
            basis_protocol=basis_protocol,
            basis_protocol_id=_protocol_id(basis_protocol),
        )
        if basis_manifest["schema"] == "hip3r_pose_graph_basis_v1":
            return self._export_pose_graph_basis(
                ranges=ranges,
                paths=[Path(item) for item in paths],
                basis_root=basis_root,
                basis_manifest=basis_manifest,
                forward_protocol=forward_protocol,
                basis_protocol=basis_protocol,
                alpha_protocol=alpha_protocol,
                output_dir=output_dir,
            )
        export_phase_started = time.perf_counter()
        trajectory_only_export = not self.config.save_chunk_predictions
        trajectory_component_cache = {
            "schema": _TRAJECTORY_COMPONENT_CACHE_SCHEMA,
            "mode": "not_applicable",
        }
        if trajectory_only_export:
            required_fields = _TRAJECTORY_EXPORT_REQUIRED_FIELDS
            chunk_entries = list(zip(manifest["chunks"], ranges))
            requested_workers = self.config.artifact_load_workers
            actual_workers = min(requested_workers, len(chunk_entries)) if chunk_entries else 0

            cached_chunks = _try_load_trajectory_component_cache(
                root,
                manifest_id=str(manifest["_validated_manifest_id"]),
                manifest=manifest,
                ranges=ranges,
                workers=requested_workers,
            )
            if cached_chunks is not None:
                chunks = cached_chunks
                trajectory_component_cache["mode"] = "hit"
                artifact_load_diagnostics = {
                    "mode": "trajectory_component_cache_hit",
                    "requested_workers": requested_workers,
                    "actual_workers": 0,
                    "num_chunks": len(chunk_entries),
                    "scope": "trajectory_chunks",
                    "required_fields": list(required_fields),
                    "cache_schema": _TRAJECTORY_COMPONENT_CACHE_SCHEMA,
                }
            else:

                def load_trajectory_chunk(
                    task: tuple[int, tuple[dict[str, Any], ChunkRange]],
                ) -> ChunkPrediction:
                    index, (entry, chunk_range) = task
                    fields = _load_prediction_artifact_fields(
                        root,
                        entry,
                        expected_index=index,
                        required_names=required_fields,
                    )
                    length = chunk_range.size
                    return ChunkPrediction._from_finite_validated_arrays(
                        _token=_FINITE_VALIDATED_TOKEN,
                        _validated_fields=fields.finite_validated_fields,
                        world_points=np.zeros((length, 1, 1, 3), dtype=np.float32),
                        world_points_conf=np.zeros((length, 1, 1), dtype=np.float32),
                        extrinsic_c2w=fields.arrays["extrinsic_c2w"],
                        intrinsic=fields.arrays["intrinsic"],
                    )

                trajectory_tasks = list(enumerate(chunk_entries))
                if actual_workers > 1:
                    with ThreadPoolExecutor(
                        max_workers=actual_workers,
                        thread_name_prefix="hip3r-artifact-load",
                    ) as executor:
                        chunks = list(executor.map(load_trajectory_chunk, trajectory_tasks))
                else:
                    chunks = [load_trajectory_chunk(task) for task in trajectory_tasks]
                cache_mode = _publish_trajectory_component_cache(
                    root,
                    manifest_id=str(manifest["_validated_manifest_id"]),
                    manifest=manifest,
                    chunks=chunks,
                    ranges=ranges,
                    workers=requested_workers,
                )
                trajectory_component_cache["mode"] = cache_mode
                artifact_load_diagnostics = {
                    "mode": "parallel" if actual_workers > 1 else "serial",
                    "requested_workers": requested_workers,
                    "actual_workers": actual_workers,
                    "num_chunks": len(chunk_entries),
                    "scope": "trajectory_chunks",
                    "required_fields": list(required_fields),
                    "cache_schema": _TRAJECTORY_COMPONENT_CACHE_SCHEMA,
                    "cache_mode": cache_mode,
                }
        else:
            _, _, _, chunks, artifact_load_diagnostics = self._load_forward_inputs(
                root, forward_protocol
            )
        artifact_loading_seconds = time.perf_counter() - export_phase_started
        export_phase_started = time.perf_counter()
        try:
            basis, edge_transforms, basis_component_cache = _load_basis_with_component_cache(
                basis_root, basis_manifest
            )
        except _CompactBasisComponentCacheMiss:
            basis_manifest = _load_basis_manifest(
                basis_root,
                forward_protocol_id=_protocol_id(forward_protocol),
                basis_protocol=basis_protocol,
                basis_protocol_id=_protocol_id(basis_protocol),
                allow_replay_cache=False,
            )
            basis, edge_transforms, basis_component_cache = _load_basis_with_component_cache(
                basis_root, basis_manifest
            )
            basis_component_cache["replay_manifest_fallback"] = True
        basis_deserialization_seconds = time.perf_counter() - export_phase_started
        export_phase_started = time.perf_counter()
        edges = _edges_from_component_transforms(
            basis_manifest["edges"], edge_transforms
        )
        edge_deserialization_seconds = time.perf_counter() - export_phase_started
        alpha = float(alpha_protocol["alpha"])
        export_phase_started = time.perf_counter()
        corrected_chunks = apply_hierarchical_alpha(
            chunks,
            ranges,
            basis,
            alpha=alpha,
            apply_frame_corrections=bool(
                alpha_protocol.get("apply_frame_corrections", True)
            ),
        )
        alpha_application_seconds = time.perf_counter() - export_phase_started
        export_phase_started = time.perf_counter()
        trajectory, intrinsic = _merge_frame_outputs(corrected_chunks, ranges, len(paths))
        merge_output_seconds = time.perf_counter() - export_phase_started
        result = LongReconstructionResult(
            chunk_ranges=ranges,
            chunks=[] if trajectory_only_export else list(corrected_chunks),
            alignments=basis.coarse_alignments,
            edges=edges,
            trajectory_c2w=trajectory,
            intrinsic=intrinsic,
            diagnostics={
                "status": basis.diagnostics.get("status", "unknown"),
                "postprocess": "hierarchical_pose_graph",
                "optimizer_used": bool(basis.diagnostics.get("optimizer_used", False)),
                "num_frames": len(paths),
                "num_requested_loops": len(manifest["loops"]),
                "num_edges": len(edges),
                "forward_protocol_id": _protocol_id(forward_protocol),
                "basis_protocol_id": _protocol_id(basis_protocol),
                "alpha_protocol_id": _protocol_id(alpha_protocol),
                "alpha_protocol": _json_safe(alpha_protocol),
                "alpha_export_mode": (
                    "verified_trajectory_only" if trajectory_only_export else "full_chunks"
                ),
                "alpha_application_implementation": (
                    "partial_alpha_global_numpy_so3_batch_numerical_equivalent"
                    if 0.0 < alpha < 1.0
                    else "scalar_component_exact"
                ),
                "text_export_implementation": (
                    "parallel_independent_artifacts"
                    if _parallel_artifact_publication_enabled(
                        len(paths),
                        save_chunk_predictions=bool(self.config.save_chunk_predictions),
                    )
                    else "parallel_two_file_savetxt"
                    if len(paths) >= 1000
                    else "serial_savetxt"
                ),
                "alpha_export_phase_seconds": {
                    "artifact_validation_loading": artifact_loading_seconds,
                    "basis_deserialization": basis_deserialization_seconds,
                    "edge_deserialization": edge_deserialization_seconds,
                    "alpha_application": alpha_application_seconds,
                    "merge_output": merge_output_seconds,
                },
                "forward_artifact_loading": artifact_load_diagnostics,
                "forward_artifact_contract": forward_artifact_contract,
                "trajectory_component_cache": trajectory_component_cache,
                "basis_component_cache": basis_component_cache,
                "basis_replay_manifest": {
                    "schema": _BASIS_REPLAY_CACHE_SCHEMA,
                    "mode": (
                        "compact_hit"
                        if basis_manifest.get("_replay_compact")
                        else "full_manifest"
                    ),
                },
                "hierarchical": {
                    **basis.diagnostics,
                    "frame_corrections_applied": bool(
                        alpha_protocol.get("apply_frame_corrections", True)
                    ),
                    "frame_correction_alpha": alpha,
                },
                "loop_edge_selection": basis_manifest["diagnostics"].get(
                    "loop_edge_selection"
                ),
                "local_chunk_inference": manifest["diagnostics"]["local_chunk_inference"],
            },
            frame_poses=basis.frame_poses,
        )
        publication_started = time.perf_counter()
        save_long_reconstruction(result, output_dir, paths, self.config)
        result.diagnostics["alpha_export_phase_seconds"]["publication"] = (
            time.perf_counter() - publication_started
        )
        _atomic_write_json(
            Path(output_dir) / "export.complete.json",
            {
                "schema": "hip3r_alpha_export_complete_v1",
                "forward_protocol_id": _protocol_id(forward_protocol),
                "basis_protocol_id": _protocol_id(basis_protocol),
                "alpha_protocol_id": _protocol_id(alpha_protocol),
                "alpha": alpha,
                "alpha_application_implementation": result.diagnostics[
                    "alpha_application_implementation"
                ],
                "text_export_implementation": result.diagnostics[
                    "text_export_implementation"
                ],
                "trajectory_component_cache": result.diagnostics[
                    "trajectory_component_cache"
                ],
                "basis_replay_manifest": result.diagnostics[
                    "basis_replay_manifest"
                ],
                "forward_artifact_contract": forward_artifact_contract,
                "forward_manifest_id": manifest[
                    "_validated_manifest_id"
                ],
                "num_frames": len(paths),
                "num_chunks": len(ranges),
                "alpha_export_phase_seconds": result.diagnostics[
                    "alpha_export_phase_seconds"
                ],
            },
        )
        return result

    def _export_pose_graph_basis(
        self,
        *,
        ranges: list[ChunkRange],
        paths: Sequence[Path],
        basis_root: Path,
        basis_manifest: dict[str, Any],
        forward_protocol: dict[str, Any],
        basis_protocol: dict[str, Any],
        alpha_protocol: dict[str, Any],
        output_dir: str | Path,
    ) -> LongReconstructionResult:
        """Materialize an immutable pose-graph basis without rerunning optimization."""
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "export.complete.json").unlink(missing_ok=True)
        for relative in basis_manifest["result_files"]:
            source = basis_root / relative
            destination = output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        trajectory = np.loadtxt(output_root / "camera_poses.txt").reshape(len(paths), 4, 4)
        intrinsic = np.loadtxt(output_root / "intrinsics.txt").reshape(len(paths), 3, 3)
        diagnostics = {
            **dict(basis_manifest.get("diagnostics") or {}),
            "forward_protocol_id": _protocol_id(forward_protocol),
            "basis_protocol_id": _protocol_id(basis_protocol),
            "alpha_protocol_id": _protocol_id(alpha_protocol),
            "alpha_protocol": _json_safe(alpha_protocol),
            "export_reused_pose_graph_basis": True,
        }
        result = LongReconstructionResult(
            chunk_ranges=ranges,
            chunks=[],
            alignments=[_sim3_from_json(item) for item in basis_manifest["alignments"]],
            edges=[_edge_from_json(item) for item in basis_manifest["edges"]],
            trajectory_c2w=trajectory,
            intrinsic=intrinsic,
            diagnostics=diagnostics,
        )
        _atomic_write_json(
            output_root / "export.complete.json",
            {
                "schema": "hip3r_alpha_export_complete_v1",
                "forward_protocol_id": _protocol_id(forward_protocol),
                "basis_protocol_id": _protocol_id(basis_protocol),
                "alpha_protocol_id": _protocol_id(alpha_protocol),
                "alpha": float(alpha_protocol["alpha"]),
                "num_frames": len(paths),
                "num_chunks": len(ranges),
            },
        )
        return result

    def run_postprocess(
        self,
        artifact_dir: str | Path,
        *,
        protocol: dict[str, Any],
        output_dir: str | Path | None = None,
        cleanup: bool = False,
    ) -> LongReconstructionResult:
        """Validate and consume a completed forward stage without using the adapter."""
        root = Path(artifact_dir)
        phase_started = time.perf_counter()
        manifest, paths, ranges, chunks, artifact_load_diagnostics = self._load_forward_inputs(
            root, protocol
        )
        phase_seconds = {
            "artifact_validation_loading": time.perf_counter() - phase_started,
        }
        pairs = [LoopPair(**item["pair"]) for item in manifest["loops"]]
        phase_started = time.perf_counter()
        loop_tasks = [
            (index, item, pair)
            for index, (item, pair) in enumerate(zip(manifest["loops"], pairs))
        ]
        actual_loop_workers = min(self.config.postprocess_loop_workers, len(loop_tasks))
        endpoint_workers = (
            2
            if loop_tasks
            and self.config.postprocess_loop_workers >= 2 * max(actual_loop_workers, 1)
            else 1
        )

        def load_and_estimate_loop(task: tuple[int, dict[str, Any], LoopPair]) -> Sim3Edge:
            index, item, pair = task
            if "fields" in item:
                fields = _load_prediction_artifact_fields(
                    root,
                    item,
                    expected_index=index,
                    required_names=_LOOP_JOINT_REQUIRED_FIELDS,
                )
                joint = _prediction_from_consumed_fields(
                    fields.arrays,
                    required_names=_LOOP_JOINT_REQUIRED_FIELDS,
                    validated_names=fields.finite_validated_fields,
                )
            else:
                joint = _load_prediction_artifact(root, item, expected_index=index)
            edge = estimate_loop_edge(
                joint,
                _slice_prediction(chunks[pair.chunk_a], item["local_a"]),
                _slice_prediction(chunks[pair.chunk_b], item["local_b"]),
                endpoint_a_index=pair.chunk_a,
                endpoint_b_index=pair.chunk_b,
                confidence_threshold=self.config.confidence_threshold,
                max_points=self.config.max_alignment_points,
                max_iterations=self.config.max_sim3_iterations,
                huber_delta=self.config.huber_delta,
                endpoint_workers=endpoint_workers,
            )
            if self.config.residual_edge_weighting:
                diagnostics = dict(edge.diagnostics)
                diagnostics["weight"] = residual_edge_weight(
                    diagnostics,
                    residual_key=(
                        "residual_quality"
                        if self.config.residual_edge_quality_quantile == "p95_else_p90"
                        else "residual_p95"
                    ),
                    residual_scale=self.config.residual_scale,
                    min_weight=self.config.min_edge_weight,
                    max_weight=self.config.max_edge_weight,
                )
                diagnostics["weight_mode"] = (
                    "endpoint_max_available_p95_else_p90"
                    if self.config.residual_edge_quality_quantile == "p95_else_p90"
                    else "endpoint_max_residual_p95"
                )
                edge = replace(edge, diagnostics=diagnostics)
            return edge

        if actual_loop_workers > 1:
            with ThreadPoolExecutor(
                max_workers=actual_loop_workers,
                thread_name_prefix="hip3r-loop-edge",
            ) as executor:
                # executor.map yields in input order, independent of completion order.
                loops = list(executor.map(load_and_estimate_loop, loop_tasks))
        else:
            loops = [load_and_estimate_loop(task) for task in loop_tasks]
        phase_seconds["loop_edge_estimation"] = time.perf_counter() - phase_started
        selected_loops, loop_selection = select_loop_edges(
            loops,
            pairs,
            policy=self.config.loop_edge_policy,
            budget=self.config.loop_edge_budget,
        )
        phase_started = time.perf_counter()
        with self._postprocess_lock():
            post_result = self._postprocess(chunks, selected_loops, ranges)
        phase_seconds["graph_postprocess"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        trajectory, intrinsic = _merge_frame_outputs(post_result.chunks, ranges, len(paths))
        result = LongReconstructionResult(
            chunk_ranges=ranges,
            chunks=list(post_result.chunks),
            alignments=post_result.alignments,
            edges=post_result.edges,
            trajectory_c2w=trajectory,
            intrinsic=intrinsic,
            diagnostics={
                **post_result.diagnostics,
                "num_frames": len(paths),
                "num_requested_loops": len(pairs),
                "loop_edge_selection": loop_selection,
                "local_chunk_inference": manifest["diagnostics"]["local_chunk_inference"],
                "forward_artifact_loading": artifact_load_diagnostics,
                "postprocess_loop_edges": {
                    "mode": "parallel" if actual_loop_workers > 1 else "serial",
                    "requested_workers": self.config.postprocess_loop_workers,
                    "actual_workers": actual_loop_workers,
                    "endpoint_workers_per_edge": endpoint_workers,
                    "num_edges": len(loop_tasks),
                    "scope": "loop_joints",
                    "required_fields": list(_LOOP_JOINT_REQUIRED_FIELDS),
                },
                "postprocess_phase_seconds": phase_seconds,
            },
            frame_poses=post_result.frame_poses,
        )
        phase_seconds["merge_output"] = time.perf_counter() - phase_started
        if output_dir is not None:
            save_long_reconstruction(result, output_dir, paths, self.config)
        if cleanup:
            shutil.rmtree(root)
        return result

    def _load_forward_inputs(
        self,
        root: Path,
        protocol: dict[str, Any],
    ) -> tuple[
        dict[str, Any],
        list[Path],
        list[ChunkRange],
        list[ChunkPrediction],
        dict[str, Any],
    ]:
        manifest = _load_forward_manifest(root, protocol)
        paths = [Path(item) for item in manifest["images"]]
        ranges = [
            ChunkRange(int(item["start"]), int(item["end"])) for item in manifest["chunks"]
        ]
        chunk_entries = list(manifest["chunks"])
        requested_workers = self.config.artifact_load_workers
        actual_workers = min(requested_workers, len(chunk_entries)) if chunk_entries else 0

        def load_chunk(task: tuple[int, dict[str, Any]]) -> ChunkPrediction:
            index, item = task
            return _load_prediction_artifact(root, item, expected_index=index)

        tasks = list(enumerate(chunk_entries))
        if actual_workers > 1:
            with ThreadPoolExecutor(
                max_workers=actual_workers,
                thread_name_prefix="hip3r-artifact-load",
            ) as executor:
                # executor.map preserves manifest order while loads finish independently.
                chunks = list(executor.map(load_chunk, tasks))
        else:
            chunks = [load_chunk(task) for task in tasks]
        diagnostics = {
            "mode": "parallel" if actual_workers > 1 else "serial",
            "requested_workers": requested_workers,
            "actual_workers": actual_workers,
            "num_chunks": len(chunk_entries),
        }
        return manifest, paths, ranges, chunks, diagnostics

    def _load_basis_forward_inputs(
        self,
        root: Path,
        protocol: dict[str, Any],
        *,
        manifest: dict[str, Any] | None = None,
        required_fields: Sequence[str] | None = None,
    ) -> tuple[
        dict[str, Any],
        list[Path],
        list[ChunkRange],
        list[ChunkPrediction],
        dict[str, Any],
    ]:
        manifest = manifest or _load_forward_manifest(root, protocol)
        required_fields = required_fields or _BASIS_CHUNK_REQUIRED_FIELDS
        chunk_entries = list(manifest["chunks"])
        if any("fields" not in entry for entry in chunk_entries):
            return self._load_forward_inputs(root, protocol)
        paths = [Path(item) for item in manifest["images"]]
        ranges = [
            ChunkRange(int(item["start"]), int(item["end"])) for item in chunk_entries
        ]
        requested_workers = self.config.artifact_load_workers
        actual_workers = min(requested_workers, len(chunk_entries)) if chunk_entries else 0
        validation_receipt = _ArtifactValidationReceipt(root)

        def load_chunk(task: tuple[int, dict[str, Any]]) -> ChunkPrediction:
            index, item = task
            fields = _load_prediction_artifact_fields(
                root,
                item,
                expected_index=index,
                required_names=required_fields,
                validation_receipt=validation_receipt,
            )
            return _prediction_from_consumed_fields(
                fields.arrays,
                required_names=required_fields,
                validated_names=fields.finite_validated_fields,
            )

        tasks = list(enumerate(chunk_entries))
        if actual_workers > 1:
            with ThreadPoolExecutor(
                max_workers=actual_workers,
                thread_name_prefix="hip3r-artifact-load",
            ) as executor:
                chunks = list(executor.map(load_chunk, tasks))
        else:
            chunks = [load_chunk(task) for task in tasks]
        validation_receipt.flush()
        diagnostics = {
            "mode": "parallel" if actual_workers > 1 else "serial",
            "requested_workers": requested_workers,
            "actual_workers": actual_workers,
            "num_chunks": len(chunk_entries),
            "scope": "basis_chunks",
            "required_fields": list(required_fields),
            "validation_receipt_hits": validation_receipt.hits,
            "validation_receipt_misses": validation_receipt.misses,
            "validation_receipt_schema": _VALIDATION_RECEIPT_SCHEMA,
        }
        return manifest, paths, ranges, chunks, diagnostics

    def _estimate_staged_loop_edges(
        self,
        root: Path,
        manifest: dict[str, Any],
        chunks: Sequence[ChunkPrediction],
        *,
        stability_strata: int = 0,
    ) -> tuple[list[Sim3Edge], list[LoopPair], dict[str, Any], dict[str, Any]]:
        pairs = [LoopPair(**item["pair"]) for item in manifest["loops"]]
        loop_tasks = [
            (index, item, pair)
            for index, (item, pair) in enumerate(zip(manifest["loops"], pairs))
        ]
        actual_loop_workers = min(self.config.postprocess_loop_workers, len(loop_tasks))
        endpoint_workers = (
            2
            if loop_tasks
            and self.config.postprocess_loop_workers >= 2 * max(actual_loop_workers, 1)
            else 1
        )
        validation_receipt = _ArtifactValidationReceipt(root)

        def load_and_estimate_loop(task: tuple[int, dict[str, Any], LoopPair]) -> Sim3Edge:
            index, item, pair = task
            if "fields" in item:
                fields = _load_prediction_artifact_fields(
                    root,
                    item,
                    expected_index=index,
                    required_names=_LOOP_JOINT_REQUIRED_FIELDS,
                    validation_receipt=validation_receipt,
                )
                joint = _prediction_from_consumed_fields(
                    fields.arrays,
                    required_names=_LOOP_JOINT_REQUIRED_FIELDS,
                    validated_names=fields.finite_validated_fields,
                )
            else:
                joint = _load_prediction_artifact(root, item, expected_index=index)
            edge = estimate_loop_edge(
                joint,
                _slice_prediction(chunks[pair.chunk_a], item["local_a"]),
                _slice_prediction(chunks[pair.chunk_b], item["local_b"]),
                endpoint_a_index=pair.chunk_a,
                endpoint_b_index=pair.chunk_b,
                confidence_threshold=self.config.confidence_threshold,
                max_points=self.config.max_alignment_points,
                max_iterations=self.config.max_sim3_iterations,
                huber_delta=self.config.huber_delta,
                endpoint_workers=endpoint_workers,
                stability_strata=stability_strata,
                endpoint_a_frame_indices=item.get("global_a"),
                endpoint_b_frame_indices=item.get("global_b"),
            )
            if self.config.residual_edge_weighting:
                diagnostics = dict(edge.diagnostics)
                diagnostics["weight"] = residual_edge_weight(
                    diagnostics,
                    residual_key=(
                        "residual_quality"
                        if self.config.residual_edge_quality_quantile == "p95_else_p90"
                        else "residual_p95"
                    ),
                    residual_scale=self.config.residual_scale,
                    min_weight=self.config.min_edge_weight,
                    max_weight=self.config.max_edge_weight,
                )
                diagnostics["weight_mode"] = (
                    "endpoint_max_available_p95_else_p90"
                    if self.config.residual_edge_quality_quantile == "p95_else_p90"
                    else "endpoint_max_residual_p95"
                )
                edge = replace(edge, diagnostics=diagnostics)
            return edge

        if actual_loop_workers > 1:
            with ThreadPoolExecutor(
                max_workers=actual_loop_workers,
                thread_name_prefix="hip3r-loop-edge",
            ) as executor:
                loops = list(executor.map(load_and_estimate_loop, loop_tasks))
        else:
            loops = [load_and_estimate_loop(task) for task in loop_tasks]
        validation_receipt.flush()
        selected_loops, loop_selection = select_loop_edges(
            loops,
            pairs,
            policy=self.config.loop_edge_policy,
            budget=self.config.loop_edge_budget,
        )
        diagnostics = {
            "mode": "parallel" if actual_loop_workers > 1 else "serial",
            "requested_workers": self.config.postprocess_loop_workers,
            "actual_workers": actual_loop_workers,
            "endpoint_workers_per_edge": endpoint_workers,
            "num_edges": len(loop_tasks),
            "scope": "loop_joints",
            "required_fields": list(_LOOP_JOINT_REQUIRED_FIELDS),
            "validation_receipt_hits": validation_receipt.hits,
            "validation_receipt_misses": validation_receipt.misses,
            "validation_receipt_schema": _VALIDATION_RECEIPT_SCHEMA,
        }
        return selected_loops, pairs, loop_selection, diagnostics

    def _build_hierarchical_basis(
        self,
        chunks: Sequence[ChunkPrediction],
        loops: Sequence[Sim3Edge],
        ranges: Sequence[ChunkRange],
        *,
        adjacent_edges: Sequence[Sim3Edge] | None = None,
        coarse_anchor_result: dict[str, Any] | None = None,
        edge_stability_strata: int = 0,
    ) -> tuple[
        HierarchicalAlphaBasis,
        list[Sim3Edge],
        dict[str, Any] | None,
    ]:
        phase_started = time.perf_counter()
        processor = ReconstructionPostProcessor(
            overlap=self.config.overlap,
            confidence_threshold=self.config.confidence_threshold,
            max_points=self.config.max_alignment_points,
            max_iterations=self.config.max_sim3_iterations,
            huber_delta=self.config.huber_delta,
            stability_strata=edge_stability_strata,
            use_jac_sparsity=self.config.use_jac_sparsity,
            chain_global_scale_gauge_centering_v1=(
                self.config.chain_global_scale_gauge_centering_v1
            ),
            chain_global_scale_gauge_centering_span_cap=(
                self.config.chain_global_scale_gauge_centering_span_cap
            ),
            chain_global_scale_gauge_centering_defer_translation_gate_to_factorized_v4=(
                self.config.chain_global_scale_gauge_centering_v1
            ),
        )
        adjacent = (
            list(adjacent_edges)
            if adjacent_edges is not None
            else processor.build_adjacent_edges(
                chunks,
                chunk_ranges=ranges,
                workers=self.config.postprocess_loop_workers,
            )
        )
        adjacent_worker_count = (
            0
            if adjacent_edges is not None
            else min(self.config.postprocess_loop_workers, max(len(chunks) - 1, 0))
        )
        build_phase_seconds = {
            "adjacent_edge_estimation": time.perf_counter() - phase_started,
        }
        phase_started = time.perf_counter()
        chain_gauge_diagnostics: dict[str, Any] | None = None
        if self.config.chain_global_scale_gauge_centering_v1:
            initial_alignments, chain_gauge_diagnostics = (
                processor._chain_adjacent_alignments_global_scale_gauge_centered(
                    len(chunks),
                    adjacent,
                    span_cap=self.config.chain_global_scale_gauge_centering_span_cap,
                    defer_translation_residual_to_factorized_frame_chart=True,
                )
            )
        else:
            initial_alignments = processor._chain_adjacent_alignments(len(chunks), adjacent)
        graph_edges = adjacent + list(loops)
        build_phase_seconds["coarse_initialization"] = time.perf_counter() - phase_started
        hierarchical_initial = initial_alignments
        coarse_anchor_diagnostics: dict[str, Any] | None = None
        coarse_anchor_for_cache: dict[str, Any] | None = None
        phase_started = time.perf_counter()
        factorized_no_loop_coarse_skip = bool(
            self.config.chain_global_scale_gauge_centering_v1 and not loops
        )
        if (
            self.config.hierarchical_freeze_coarse_alignments
            or self.config.hierarchical_joint_coarse_warm_start
        ):
            if factorized_no_loop_coarse_skip:
                if coarse_anchor_result is not None:
                    raise ValueError(
                        "factorized_frame_chart_residual_v4 no-loop path must "
                        "not reuse an absolute coarse-anchor result"
                    )
                anchored = {
                    "alignments": list(initial_alignments),
                    "diagnostics": {
                        "status": "ok",
                        "optimizer_used": False,
                        "solver_mode": (
                            "factorized_adjacent_chain_no_absolute_coarse_solve"
                        ),
                        "num_function_evaluations": 0,
                        "num_edges": len(graph_edges),
                        "num_loop_edges": 0,
                        "factorized_frame_chart_residual_v4": True,
                    },
                }
            else:
                anchored = (
                    coarse_anchor_result
                    if coarse_anchor_result is not None
                    else _optimize_coarse_anchor_with_loop_continuation(
                        len(chunks),
                        graph_edges,
                        initial_alignments,
                        config=self.config,
                    )
                )
            hierarchical_initial = anchored["alignments"]
            coarse_anchor_diagnostics = dict(anchored.get("diagnostics") or {})
            if not factorized_no_loop_coarse_skip:
                coarse_anchor_for_cache = {
                    "alignments": list(hierarchical_initial),
                    "diagnostics": coarse_anchor_diagnostics,
                }
        build_phase_seconds["coarse_pose_graph"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        basis = build_hierarchical_basis(
            chunks,
            ranges,
            graph_edges,
            hierarchical_initial,
            config=HierarchicalPoseGraphConfig(
                fine_edge_weight=self.config.hierarchical_fine_edge_weight,
                coarse_prior_weight=self.config.hierarchical_coarse_prior_weight,
                normalize_residual_families=(
                    self.config.hierarchical_normalize_residual_families
                ),
                frame_stride=self.config.hierarchical_frame_stride,
                max_nfev=self.config.hierarchical_max_nfev,
                solver_ftol=self.config.hierarchical_solver_ftol,
                solver_xtol=self.config.hierarchical_solver_xtol,
                solver_gtol=self.config.hierarchical_solver_gtol,
                joint_lsmr_tolerance=(
                    self.config.hierarchical_joint_lsmr_tolerance
                ),
                robust_loss=self.config.hierarchical_robust_loss,
                f_scale=self.config.hierarchical_f_scale,
                translation_scale=self.config.hierarchical_translation_scale,
                record_residual_components=self.config.hierarchical_record_residual_components,
                max_log_scale_delta=self.config.hierarchical_max_log_scale_delta,
                apply_frame_corrections=self.config.hierarchical_apply_frame_corrections,
                frame_correction_alpha=self.config.hierarchical_frame_correction_alpha,
                freeze_coarse_alignments=self.config.hierarchical_freeze_coarse_alignments,
                frozen_optimizer_variable_rows_only=(
                    self.config.hierarchical_frozen_optimizer_variable_rows_only
                ),
                factorized_frame_chart_residual_v4=(
                    self.config.chain_global_scale_gauge_centering_v1
                ),
                factorized_frame_chart_residual_v4_keyframe_continuation=(
                    self.config.hierarchical_factorized_v4_keyframe_continuation
                ),
                factorized_frame_chart_residual_v4_keyframe_continuation_objective_projection=(
                    self.config
                    .hierarchical_factorized_v4_keyframe_continuation_objective_projection
                ),
                max_frames_per_solve=self.config.hierarchical_max_frames_per_solve,
                consensus_keyframe_stride=self.config.hierarchical_consensus_keyframe_stride,
                long_sequence_mode=self.config.hierarchical_long_sequence_mode,
                keyframe_selection=self.config.hierarchical_keyframe_selection,
                keyframe_target_stride=self.config.hierarchical_keyframe_target_stride,
                keyframe_min_overlap_shared=(
                    self.config.hierarchical_keyframe_min_overlap_shared
                ),
                chunk_scale_trust_mode=self.config.hierarchical_chunk_scale_trust_mode,
                chunk_scale_trust_gain=self.config.hierarchical_chunk_scale_trust_gain,
                chunk_scale_trust_min_ratio=(
                    self.config.hierarchical_chunk_scale_trust_min_ratio
                ),
                chunk_scale_trust_max_ratio=(
                    self.config.hierarchical_chunk_scale_trust_max_ratio
                ),
                use_jac_sparsity=self.config.use_jac_sparsity,
                solver_workers=self.config.hierarchical_solver_workers,
                incremental_frozen_residual=(
                    self.config.hierarchical_incremental_frozen_residual
                ),
                analytic_frozen_jacobian=(
                    self.config.hierarchical_analytic_frozen_jacobian
                ),
                analytic_dynamic_jacobian=(
                    self.config.hierarchical_analytic_dynamic_jacobian
                ),
                analytic_dynamic_jacobian_backend=(
                    self.config.hierarchical_analytic_dynamic_jacobian_backend
                ),
                analytic_dynamic_jacobian_local_rotation_projection=(
                    self.config.hierarchical_analytic_dynamic_jacobian_local_rotation_projection
                ),
                vectorize_dynamic_residual=(
                    self.config.hierarchical_vectorize_dynamic_residual
                ),
                analytic_jacobian_threads=(
                    self.config.hierarchical_analytic_jacobian_threads
                ),
                analytic_jacobian_backend=(
                    self.config.hierarchical_analytic_jacobian_backend
                ),
                analytic_fast_residual_logmap=(
                    self.config.hierarchical_analytic_fast_residual_logmap
                ),
                joint_family_diagnostics=(
                    self.config.hierarchical_joint_family_diagnostics
                ),
                joint_full_sparse_exact_basis=(
                    self.config.hierarchical_joint_full_sparse_exact_basis
                ),
                joint_frame_relaxation_max_nfev=(
                    self.config.hierarchical_joint_frame_relaxation_max_nfev
                ),
                joint_subspace_max_nfev=(
                    self.config.hierarchical_joint_subspace_max_nfev
                ),
                joint_subspace_control_stride=(
                    self.config.hierarchical_joint_subspace_control_stride
                ),
                joint_subspace_conditional_frame_polish=(
                    self.config.hierarchical_joint_subspace_conditional_frame_polish
                ),
                joint_subspace_min_coarse_squared_residual=(
                    self.config.hierarchical_joint_subspace_min_coarse_squared_residual
                ),
                joint_subspace_stage_b_rotation_trust_ratio=(
                    self.config.hierarchical_joint_subspace_stage_b_rotation_trust_ratio
                ),
                joint_subspace_stage_b_translation_trust_ratio=(
                    self.config.hierarchical_joint_subspace_stage_b_translation_trust_ratio
                ),
                joint_subspace_stage_b_p64_gauge_fix=(
                    self.config.hierarchical_joint_subspace_stage_b_p64_gauge_fix
                ),
                joint_subspace_stage_b_p64_sparse_basis=(
                    self.config.hierarchical_joint_subspace_stage_b_p64_sparse_basis
                ),
                joint_subspace_stage_b_p64_sparse_basis_outer_cycle=(
                    self.config
                    .hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle
                ),
                joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking=(
                    self.config
                    .hierarchical_joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking
                ),
                joint_subspace_unrestricted_repair_max_nfev=(
                    self.config.hierarchical_joint_subspace_unrestricted_repair_max_nfev
                ),
                joint_schur_diagnostics=(
                    self.config.hierarchical_joint_schur_diagnostics
                ),
                joint_schur_damping=self.config.hierarchical_joint_schur_damping,
                joint_schur_cg_maxiter=(
                    self.config.hierarchical_joint_schur_cg_maxiter
                ),
                joint_schur_cg_rtol=self.config.hierarchical_joint_schur_cg_rtol,
                joint_schur_two_level_aggregate_size=(
                    self.config.hierarchical_joint_schur_two_level_aggregate_size
                ),
                joint_schur_full_space_block_pcg_probe=(
                    self.config.hierarchical_joint_schur_full_space_block_pcg_probe
                ),
                joint_schur_step=self.config.hierarchical_joint_schur_step,
                joint_schur_step_final_solver=(
                    self.config.hierarchical_joint_schur_step_final_solver
                ),
                joint_schur_frozen_constrained_step_oracle=(
                    self.config
                    .hierarchical_joint_schur_frozen_constrained_step_oracle
                ),
                joint_schur_one_active_set_restoration_at_first_reject=(
                    self.config
                    .hierarchical_joint_schur_one_active_set_restoration_at_first_reject
                ),
                joint_schur_step_public_pilot_max_nfev=(
                    self.config.hierarchical_joint_schur_step_public_pilot_max_nfev
                ),
                joint_schur_step_max_outer_iterations=(
                    self.config.hierarchical_joint_schur_step_max_outer_iterations
                ),
                joint_schur_step_initial_alpha=(
                    self.config.hierarchical_joint_schur_step_initial_alpha
                ),
                joint_schur_step_backtracks=(
                    self.config.hierarchical_joint_schur_step_backtracks
                ),
                joint_schur_step_freeze_coarse_scales=(
                    self.config.hierarchical_joint_schur_step_freeze_coarse_scales
                ),
                joint_schur_step_min_accepted_alpha=(
                    self.config.hierarchical_joint_schur_step_min_accepted_alpha
                ),
                joint_schur_step_frame_low_frequency_trust_ratio=(
                    self.config.hierarchical_joint_schur_step_frame_low_frequency_trust_ratio
                ),
                joint_schur_step_frame_low_frequency_trust_aggregate_size=(
                    self.config
                    .hierarchical_joint_schur_step_frame_low_frequency_trust_aggregate_size
                ),
                joint_schur_step_min_low_frequency_efficiency_ratio=(
                    self.config.hierarchical_joint_schur_step_min_low_frequency_efficiency_ratio
                ),
                joint_schur_step_max_initial_jacobi_low_frequency_efficiency=(
                    self.config.hierarchical_joint_schur_step_max_initial_jacobi_low_frequency_efficiency
                ),
                joint_schur_step_max_initial_diagonal_low_frequency_efficiency=(
                    self.config.hierarchical_joint_schur_step_max_initial_diagonal_low_frequency_efficiency
                ),
                joint_schur_step_max_initial_diagonal_reduction_fraction=(
                    self.config.hierarchical_joint_schur_step_max_initial_diagonal_reduction_fraction
                ),
                joint_schur_step_max_graph_cost_ratio=(
                    self.config.hierarchical_joint_schur_step_max_graph_cost_ratio
                ),
                joint_schur_step_max_adjacent_p95_ratio=(
                    self.config.hierarchical_joint_schur_step_max_adjacent_p95_ratio
                ),
                joint_schur_step_max_loop_p95_ratio=(
                    self.config.hierarchical_joint_schur_step_max_loop_p95_ratio
                ),
                joint_schur_step_max_loop_max_ratio=(
                    self.config.hierarchical_joint_schur_step_max_loop_max_ratio
                ),
            ),
        )
        basis.diagnostics["joint_coarse_warm_start"] = bool(
            self.config.hierarchical_joint_coarse_warm_start
        )
        build_phase_seconds["fine_basis_build"] = time.perf_counter() - phase_started
        basis_diagnostics = {
            **basis.diagnostics,
            "hierarchical_build_phase_seconds": build_phase_seconds,
            "adjacent_edge_workers": {
                "mode": (
                    "cached"
                    if adjacent_edges is not None
                    else ("parallel" if adjacent_worker_count > 1 else "serial")
                ),
                "requested_workers": self.config.postprocess_loop_workers,
                "actual_workers": adjacent_worker_count,
                "num_edges": len(adjacent),
            },
        }
        if coarse_anchor_diagnostics is not None:
            basis_diagnostics["coarse_anchor"] = coarse_anchor_diagnostics
        if chain_gauge_diagnostics is not None:
            basis_diagnostics["chain_global_scale_gauge_centering_v1"] = (
                chain_gauge_diagnostics
            )
        if self.config.chain_global_scale_gauge_centering_v1:
            v4_diagnostics = basis_diagnostics.get(
                "factorized_frame_chart_residual_v4"
            )
            if (
                not isinstance(v4_diagnostics, dict)
                or v4_diagnostics.get("enabled") is not True
                or v4_diagnostics.get("valid") is not True
                or v4_diagnostics.get("fallback_to_absolute_residual") is not False
            ):
                raise HierarchicalBasisActivationError(
                    "factorized_frame_chart_residual_v4 activation failed: "
                    f"{v4_diagnostics}",
                    basis_diagnostics,
                )
        basis = replace(
            basis,
            diagnostics=basis_diagnostics,
        )
        return basis, graph_edges, coarse_anchor_for_cache

    def _report(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def _pop_reliability_trace(self):
        if not self.config.reliability_trace:
            return None
        pop_trace = getattr(self.adapter, "pop_reliability_trace", None)
        return pop_trace() if pop_trace is not None else None

    def _infer_local_chunks(
        self,
        paths: Sequence[Path],
        ranges: Sequence[ChunkRange],
    ) -> tuple[list[ChunkPrediction], list[dict[str, Any]], dict[str, Any]]:
        if self.config.local_chunk_workers == 1:
            chunks = []
            traces = []
            for index, item in enumerate(ranges):
                self._report(
                    f"local chunk {index + 1}/{len(ranges)} frames={item.start}:{item.end}"
                )
                prediction = self.adapter.infer_paths(paths[item.start : item.end])
                chunks.append(ChunkPrediction(**prediction.to_numpy_dict()))
                trace = self._pop_reliability_trace()
                if trace is not None:
                    traces.append(
                        {
                            "chunk_index": index,
                            "frame_start": item.start,
                            "frame_end": item.end,
                            **trace,
                        }
                    )
            return chunks, traces, {
                "mode": "serial",
                "requested_workers": 1,
                "actual_workers": 1,
            }

        infer_parallel = getattr(self.adapter, "infer_chunks_parallel", None)
        if infer_parallel is None:
            raise TypeError(
                f"{type(self.adapter).__name__} does not support parallel local chunk inference"
            )
        if self.config.reliability_trace:
            raise ValueError("reliability_trace is not supported with parallel local chunks")
        chunk_paths = [paths[item.start : item.end] for item in ranges]
        self._report(
            "parallel local chunks "
            f"count={len(chunk_paths)} workers={self.config.local_chunk_workers} "
            f"gpus={list(self.config.local_chunk_gpu_ids)}"
        )
        predictions, diagnostics = infer_parallel(
            chunk_paths,
            gpu_ids=self.config.local_chunk_gpu_ids[: self.config.local_chunk_workers],
            result_root=self.config.parallel_result_root,
            progress=self._report,
        )
        if len(predictions) != len(ranges):
            raise RuntimeError(
                f"parallel chunk inference returned {len(predictions)} results for {len(ranges)} ranges"
            )
        chunks = [ChunkPrediction(**prediction.to_numpy_dict()) for prediction in predictions]
        return chunks, [], {
            "mode": "parallel",
            "requested_workers": self.config.local_chunk_workers,
            "actual_workers": min(self.config.local_chunk_workers, len(ranges)),
            "gpu_ids": list(
                self.config.local_chunk_gpu_ids[
                    : min(self.config.local_chunk_workers, len(ranges))
                ]
            ),
            **dict(diagnostics or {}),
        }

    def _infer_local_chunks_to_artifacts(
        self,
        paths: Sequence[Path],
        ranges: Sequence[ChunkRange],
        *,
        artifact_root: Path,
        field_names: Sequence[str],
        field_dtype_overrides: Mapping[str, np.dtype[Any] | str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        infer_direct = getattr(
            self.adapter, "infer_chunks_parallel_to_artifacts", None
        )
        if infer_direct is None:
            raise TypeError(
                f"{type(self.adapter).__name__} does not support direct "
                "parallel artifact publication"
            )
        chunk_paths = [paths[item.start : item.end] for item in ranges]
        relative_paths = [
            f"chunks/chunk_{index:06d}" for index in range(len(ranges))
        ]
        worker_count = min(self.config.local_chunk_workers, len(ranges))
        entries, diagnostics = infer_direct(
            chunk_paths,
            gpu_ids=self.config.local_chunk_gpu_ids[:worker_count],
            artifact_root=artifact_root,
            artifact_relative_paths=relative_paths,
            artifact_version=self.config.forward_artifact_version,
            artifact_field_names=field_names,
            artifact_field_dtype_overrides=field_dtype_overrides,
            result_root=self.config.parallel_result_root,
            progress=self._report,
        )
        _validate_direct_artifact_entries(
            entries,
            artifact_root=artifact_root,
            relative_paths=relative_paths,
            field_names=field_names,
        )
        return [
            {**entry, "index": index, "start": item.start, "end": item.end}
            for index, (item, entry) in enumerate(zip(ranges, entries))
        ], {
            "mode": "parallel_worker_direct_artifact_v2",
            "requested_workers": self.config.local_chunk_workers,
            "actual_workers": worker_count,
            "gpu_ids": list(self.config.local_chunk_gpu_ids[:worker_count]),
            **dict(diagnostics or {}),
        }

    def _build_incremental_adjacent_edges(
        self,
        chunks: Sequence[ChunkPrediction],
        ranges: Sequence[ChunkRange],
    ) -> list[Sim3Edge]:
        if self.config.sim3_schedule != "incremental" or self.config.postprocess == "none":
            return []
        if len(ranges) != len(chunks):
            raise ValueError(f"chunk range count {len(ranges)} != chunk count {len(chunks)}")
        edges = []
        for index in range(1, len(chunks)):
            self._report(f"estimating incremental adjacent edge {index - 1}->{index}")
            edges.append(
                estimate_adjacent_edge(
                    chunks[index - 1],
                    chunks[index],
                    overlap=self.config.overlap,
                    previous_range=ranges[index - 1],
                    next_range=ranges[index],
                    previous_index=index - 1,
                    next_index=index,
                    confidence_threshold=self.config.confidence_threshold,
                    max_points=self.config.max_alignment_points,
                    max_iterations=self.config.max_sim3_iterations,
                    huber_delta=self.config.huber_delta,
                )
            )
        return edges

    @contextmanager
    def _postprocess_lock(self):
        lock_path = self.config.postprocess_lock_path
        if lock_path is None:
            yield
            return
        path = Path(lock_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+") as handle:
            self._report(f"waiting for postprocess lock {path}")
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EAGAIN, errno.EACCES, errno.EINTR}:
                        raise
                    time.sleep(5.0)
            self._report(f"acquired postprocess lock {path}")
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                self._report(f"released postprocess lock {path}")

    def _postprocess(
        self,
        chunks: Sequence[ChunkPrediction],
        loops: Sequence[Sim3Edge],
        ranges: Sequence[ChunkRange],
        *,
        adjacent_edges: Sequence[Sim3Edge] | None = None,
        trajectory_only: bool = False,
    ) -> PostProcessResult:
        if self.config.postprocess == "none":
            return PostProcessResult(
                chunks=(
                    _trajectory_only_chunks(chunks)
                    if trajectory_only
                    else list(chunks)
                ),
                alignments=[Sim3.identity() for _ in chunks],
                edges=[],
                diagnostics={
                    "status": "ok",
                    "postprocess": "none",
                    "optimizer_used": False,
                    "trajectory_only": bool(trajectory_only),
                },
            )
        processor = ReconstructionPostProcessor(
            overlap=self.config.overlap,
            confidence_threshold=self.config.confidence_threshold,
            max_points=self.config.max_alignment_points,
            max_iterations=self.config.max_sim3_iterations,
            huber_delta=self.config.huber_delta,
            use_jac_sparsity=self.config.use_jac_sparsity,
            chain_global_scale_gauge_centering_v1=(
                self.config.chain_global_scale_gauge_centering_v1
            ),
            chain_global_scale_gauge_centering_span_cap=(
                self.config.chain_global_scale_gauge_centering_span_cap
            ),
            pose_graph_optimizer=(
                optimize_sim3_pose_graph
                if self.config.postprocess == "pose_graph"
                else None
            ),
        )
        if self.config.postprocess != "hierarchical_pose_graph":
            result = processor.process(
                _trajectory_only_chunks(chunks) if trajectory_only else chunks,
                chunk_ranges=ranges,
                adjacent_constraints=adjacent_edges,
                loop_constraints=loops,
            )
            if trajectory_only:
                result.diagnostics["trajectory_only"] = True
            return result

        basis, graph_edges, _ = self._build_hierarchical_basis(
            chunks,
            loops,
            ranges,
            adjacent_edges=adjacent_edges,
        )
        if self.config.hierarchical_require_solver_convergence:
            _require_hierarchical_solver_convergence(basis.diagnostics)
        alpha_chunks = _trajectory_only_chunks(chunks) if trajectory_only else chunks
        corrected_chunks = apply_hierarchical_alpha(
            alpha_chunks,
            ranges,
            basis,
            alpha=self.config.hierarchical_frame_correction_alpha,
            apply_frame_corrections=self.config.hierarchical_apply_frame_corrections,
        )
        hierarchical_diagnostics = {
            **basis.diagnostics,
            "frame_corrections_applied": bool(
                self.config.hierarchical_apply_frame_corrections
            ),
            "frame_correction_alpha": float(self.config.hierarchical_frame_correction_alpha),
        }
        return PostProcessResult(
            chunks=corrected_chunks,
            alignments=basis.coarse_alignments,
            edges=graph_edges,
            diagnostics={
                "status": hierarchical_diagnostics.get("status", "unknown"),
                "postprocess": "hierarchical_pose_graph",
                "optimizer_used": bool(hierarchical_diagnostics.get("optimizer_used", False)),
                "num_adjacent_edges": sum(edge.kind == "adjacent" for edge in graph_edges),
                "num_loop_edges": len(loops),
                "hierarchical": hierarchical_diagnostics,
                "trajectory_only": bool(trajectory_only),
            },
            frame_poses=basis.frame_poses,
        )

    def _build_loop_edges(
        self,
        paths: Sequence[Path],
        ranges: Sequence[ChunkRange],
        chunks: Sequence[ChunkPrediction],
        pairs: Sequence[LoopPair],
    ) -> tuple[list[Sim3Edge], list[dict[str, Any]]]:
        edges = []
        traces = []
        for pair in pairs:
            if not (0 <= pair.chunk_a < len(chunks) and 0 <= pair.chunk_b < len(chunks)):
                raise ValueError(f"loop pair chunk index out of range: {pair}")
            if pair.chunk_a == pair.chunk_b:
                raise ValueError(f"loop pair must connect different chunks: {pair}")
            local_a, global_a = _loop_window(ranges[pair.chunk_a], pair.frame_a, self.config.loop_window_size)
            local_b, global_b = _loop_window(ranges[pair.chunk_b], pair.frame_b, self.config.loop_window_size)
            infer_loop_paths = getattr(self.adapter, "infer_loop_paths", self.adapter.infer_paths)
            joint = infer_loop_paths([paths[index] for index in global_a + global_b]).to_numpy_dict()
            trace = self._pop_reliability_trace()
            if trace is not None:
                traces.append(
                    {
                        "chunk_a": pair.chunk_a,
                        "chunk_b": pair.chunk_b,
                        "frames_a": global_a,
                        "frames_b": global_b,
                        **trace,
                    }
                )
            a_prediction = _slice_prediction(chunks[pair.chunk_a], local_a)
            b_prediction = _slice_prediction(chunks[pair.chunk_b], local_b)
            edge = estimate_loop_edge(
                ChunkPrediction(**joint),
                a_prediction,
                b_prediction,
                endpoint_a_index=pair.chunk_a,
                endpoint_b_index=pair.chunk_b,
                confidence_threshold=self.config.confidence_threshold,
                max_points=self.config.max_alignment_points,
                max_iterations=self.config.max_sim3_iterations,
                huber_delta=self.config.huber_delta,
                endpoint_workers=(2 if self.config.postprocess_loop_workers > 1 else 1),
            )
            if self.config.residual_edge_weighting:
                diagnostics = dict(edge.diagnostics)
                diagnostics["weight"] = residual_edge_weight(
                    diagnostics,
                    residual_key=(
                        "residual_quality"
                        if self.config.residual_edge_quality_quantile == "p95_else_p90"
                        else "residual_p95"
                    ),
                    residual_scale=self.config.residual_scale,
                    min_weight=self.config.min_edge_weight,
                    max_weight=self.config.max_edge_weight,
                )
                diagnostics["weight_mode"] = (
                    "endpoint_max_available_p95_else_p90"
                    if self.config.residual_edge_quality_quantile == "p95_else_p90"
                    else "endpoint_max_residual_p95"
                )
                edge = replace(edge, diagnostics=diagnostics)
            edges.append(edge)
        return edges, traces


def _loop_window(chunk_range: ChunkRange, frame: int | None, window_size: int) -> tuple[list[int], list[int]]:
    center = (chunk_range.start + chunk_range.end - 1) // 2 if frame is None else int(frame)
    if not chunk_range.start <= center < chunk_range.end:
        raise ValueError(f"loop frame {center} is outside chunk range {chunk_range}")
    size = min(window_size, chunk_range.size)
    start = max(chunk_range.start, min(center - size // 2, chunk_range.end - size))
    global_indices = list(range(start, start + size))
    local_indices = [index - chunk_range.start for index in global_indices]
    return local_indices, global_indices


def _slice_prediction(prediction: ChunkPrediction, indices: Sequence[int]) -> ChunkPrediction:
    values = prediction.to_numpy_dict()
    sliced = {name: value[list(indices)] for name, value in values.items()}
    validated_fields = [
        name
        for name in sliced
        if prediction._has_finite_validated_fields(name)
    ]
    if validated_fields:
        return ChunkPrediction._from_finite_validated_arrays(
            _token=_FINITE_VALIDATED_TOKEN,
            _validated_fields=validated_fields,
            **sliced,
        )
    return ChunkPrediction(**sliced)


def _merge_frame_outputs(
    chunks: Sequence[ChunkPrediction], ranges: Sequence[ChunkRange], num_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    trajectory = np.zeros((num_frames, 4, 4), dtype=np.float64)
    intrinsic = np.zeros((num_frames, 3, 3), dtype=np.float64)
    written = np.zeros(num_frames, dtype=bool)
    for chunk, chunk_range in zip(chunks, ranges):
        poses = np.asarray(chunk.extrinsic_c2w)
        intrinsics = np.asarray(chunk.intrinsic)
        for local_index, global_index in enumerate(range(chunk_range.start, chunk_range.end)):
            if not written[global_index]:
                trajectory[global_index] = poses[local_index]
                intrinsic[global_index] = intrinsics[local_index]
                written[global_index] = True
    if not written.all():
        raise RuntimeError(f"chunk merge left {int((~written).sum())} frames uncovered")
    return trajectory, intrinsic


def _trajectory_only_chunks(
    chunks: Sequence[ChunkPrediction],
) -> list[ChunkPrediction]:
    """Drop dense arrays after graph construction while preserving camera outputs.

    Hierarchical pose-only export still needs a valid ``ChunkPrediction`` so the
    existing correction code can update C2W matrices.  Per-frame 1x1 zero
    geometry keeps that contract without materializing a second dense map.
    """
    output: list[ChunkPrediction] = []
    for index, chunk in enumerate(chunks):
        poses = np.asarray(chunk.extrinsic_c2w)
        intrinsics = np.asarray(chunk.intrinsic)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(
                f"chunk {index} extrinsic_c2w must have shape [S,4,4], got {poses.shape}"
            )
        if intrinsics.shape != (poses.shape[0], 3, 3):
            raise ValueError(
                f"chunk {index} intrinsic shape {intrinsics.shape} does not match "
                f"{poses.shape[0]} pose(s)"
            )
        length = poses.shape[0]
        output.append(
            ChunkPrediction(
                world_points=np.zeros((length, 1, 1, 3), dtype=np.float32),
                world_points_conf=np.zeros((length, 1, 1), dtype=np.float32),
                extrinsic_c2w=poses,
                intrinsic=intrinsics,
            )
        )
    return output


def save_long_reconstruction(
    result: LongReconstructionResult,
    output_dir: str | Path,
    image_paths: Sequence[Path],
    config: ReconstructionConfig,
) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    text_exports = (
        (
            root / "camera_poses.txt",
            result.trajectory_c2w.reshape(len(image_paths), -1),
        ),
        (
            root / "intrinsics.txt",
            result.intrinsic.reshape(len(image_paths), -1),
        ),
    )
    parallel_artifact_publication = _parallel_artifact_publication_enabled(
        len(image_paths), save_chunk_predictions=bool(config.save_chunk_predictions)
    )

    manifest = {
        "config": config.__dict__,
        "images": [str(path) for path in image_paths],
        "chunk_ranges": [item.__dict__ for item in result.chunk_ranges],
        "alignments": [item.as_matrix().tolist() for item in result.alignments],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "kind": edge.kind,
                "transform": edge.transform.as_matrix().tolist(),
                "diagnostics": _json_safe(edge.diagnostics),
            }
            for edge in result.edges
        ],
        "diagnostics": _json_safe(result.diagnostics),
    }
    loop_selection = result.diagnostics.get("loop_edge_selection")

    def save_text_export(item: tuple[Path, np.ndarray]) -> None:
        np.savetxt(*item)

    if parallel_artifact_publication:
        def save_frame_poses() -> None:
            assert result.frame_poses is not None
            np.save(
                root / "frame_pose_sim3.npy",
                _sim3_batch_to_matrices(result.frame_poses),
            )

        def save_reconstruction_manifest() -> None:
            _atomic_write_json(
                root / "reconstruction.json",
                manifest,
                compact=True,
                payload_is_json_safe=True,
            )

        def save_loop_selection() -> None:
            (root / "loop_edge_selection.json").write_text(
                json.dumps(_json_safe(loop_selection), indent=2, sort_keys=True) + "\n"
            )

        publication_tasks: list[Callable[[], None]] = [
            lambda item=item: save_text_export(item) for item in text_exports
        ]
        if result.frame_poses is not None:
            publication_tasks.append(save_frame_poses)
        publication_tasks.append(save_reconstruction_manifest)
        if loop_selection is not None:
            publication_tasks.append(save_loop_selection)
        if result.reliability_trace is not None:
            publication_tasks.append(
                lambda: (root / "reliability_trace.json").write_text(
                    json.dumps(_json_safe(result.reliability_trace), indent=2) + "\n"
                )
            )
        with ThreadPoolExecutor(
            max_workers=len(publication_tasks),
            thread_name_prefix="hip3r-artifact-export",
        ) as executor:
            futures = [executor.submit(task) for task in publication_tasks]
            for future in futures:
                future.result()
        return

    if len(image_paths) >= 1000:
        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="hip3r-text-export"
        ) as executor:
            list(executor.map(save_text_export, text_exports))
    else:
        for text_export in text_exports:
            np.savetxt(*text_export)
    if config.save_chunk_predictions:
        chunk_dir = root / "chunks"
        chunk_dir.mkdir(exist_ok=True)
        for index, chunk in enumerate(result.chunks):
            np.savez_compressed(chunk_dir / f"chunk_{index:04d}.npz", **chunk.to_numpy_dict())
    if result.frame_poses is not None:
        np.save(
            root / "frame_pose_sim3.npy",
            _sim3_batch_to_matrices(result.frame_poses),
        )

    _atomic_write_json(
        root / "reconstruction.json",
        manifest,
        compact=True,
        payload_is_json_safe=True,
    )
    if loop_selection is not None:
        (root / "loop_edge_selection.json").write_text(
            json.dumps(_json_safe(loop_selection), indent=2, sort_keys=True) + "\n"
        )
    if result.reliability_trace is not None:
        (root / "reliability_trace.json").write_text(
            json.dumps(_json_safe(result.reliability_trace), indent=2) + "\n"
        )


def _parallel_artifact_publication_enabled(
    num_frames: int, *, save_chunk_predictions: bool
) -> bool:
    return num_frames >= 1000 and not save_chunk_predictions


def _edge_trace(edge: Sim3Edge) -> dict[str, Any]:
    return {
        "source": edge.source,
        "target": edge.target,
        "kind": edge.kind,
        "diagnostics": _json_safe(edge.diagnostics),
    }


def _sim3_to_json(transform: Sim3) -> dict[str, Any]:
    return {"matrix": transform.as_matrix().tolist()}


def _sim3_batch_to_json(transforms: Sequence[Sim3]) -> list[dict[str, Any]]:
    """Serialize validated Sim3 objects with one contiguous matrix build."""

    return [{"matrix": matrix} for matrix in _sim3_batch_to_matrices(transforms).tolist()]


def _sim3_batch_to_matrices(transforms: Sequence[Sim3]) -> np.ndarray:
    """Build contiguous Sim3 matrices from already validated components."""

    count = len(transforms)
    if count == 0:
        return np.empty((0, 4, 4), dtype=np.float64)
    scales = np.asarray([item.scale for item in transforms], dtype=np.float64)
    rotations = np.asarray(
        [item.rotation for item in transforms], dtype=np.float64
    ).reshape(count, 3, 3)
    translations = np.asarray(
        [item.translation for item in transforms], dtype=np.float64
    ).reshape(count, 3)
    matrices = np.zeros((count, 4, 4), dtype=np.float64)
    matrices[:, :3, :3] = scales[:, None, None] * rotations
    matrices[:, :3, 3] = translations
    matrices[:, 3, 3] = 1.0
    return matrices


def _sim3_from_json(payload: dict[str, Any]) -> Sim3:
    matrix = np.asarray(payload["matrix"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Sim3 matrix must have shape (4, 4), got {matrix.shape}")
    scaled_rotation = matrix[:3, :3]
    scale = float(abs(np.linalg.det(scaled_rotation)) ** (1.0 / 3.0))
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("Sim3 matrix has invalid scale")
    rotation = scaled_rotation / scale
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return Sim3(scale=scale, rotation=rotation, translation=matrix[:3, 3])


def _sim3_batch_from_json(payloads: Sequence[dict[str, Any]]) -> list[Sim3]:
    """Deserialize scalar-exact Sim3 values with one final validity scan."""

    count = len(payloads)
    scales = np.empty(count, dtype=np.float64)
    rotations = np.empty((count, 3, 3), dtype=np.float64)
    translations = np.empty((count, 3), dtype=np.float64)
    for index, payload in enumerate(payloads):
        matrix = np.asarray(payload["matrix"], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Sim3 matrix must have shape (4, 4), got {matrix.shape}")
        scaled_rotation = matrix[:3, :3]
        scale = float(abs(np.linalg.det(scaled_rotation)) ** (1.0 / 3.0))
        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError("Sim3 matrix has invalid scale")
        rotation = scaled_rotation / scale
        u, _, vt = np.linalg.svd(rotation)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        scales[index] = scale
        rotations[index] = rotation
        translations[index] = matrix[:3, 3]

    return _sim3_batch_from_components(scales, rotations, translations)


def _sim3_batch_from_components(
    scales: np.ndarray, rotations: np.ndarray, translations: np.ndarray
) -> list[Sim3]:
    scales = np.asarray(scales, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    count = len(scales)
    if scales.shape != (count,):
        raise ValueError("Sim3 scales must have shape (N,)")
    if rotations.shape != (count, 3, 3):
        raise ValueError("Sim3 rotations must have shape (N,3,3)")
    if translations.shape != (count, 3):
        raise ValueError("Sim3 translations must have shape (N,3)")
    if not np.all(np.isfinite(scales)) or np.any(np.abs(scales) < 1e-12):
        raise ValueError("Sim3 scales contain invalid values")
    if not np.all(np.isfinite(rotations)):
        raise ValueError("rotation contains non-finite values")
    if not np.all(np.isfinite(translations)):
        raise ValueError("translation contains non-finite values")
    result = []
    for scale, rotation, translation in zip(scales, rotations, translations):
        transform = object.__new__(Sim3)
        object.__setattr__(transform, "scale", float(scale))
        object.__setattr__(transform, "rotation", rotation)
        object.__setattr__(transform, "translation", translation)
        result.append(transform)
    return result


def _edge_to_json(
    edge: Sim3Edge, *, diagnostics_are_json_safe: bool = False
) -> dict[str, Any]:
    return {
        "source": int(edge.source),
        "target": int(edge.target),
        "kind": edge.kind,
        "transform": _sim3_to_json(edge.transform),
        "diagnostics": (
            edge.diagnostics
            if diagnostics_are_json_safe
            else _json_safe(edge.diagnostics)
        ),
    }


def _edge_from_json(payload: dict[str, Any]) -> Sim3Edge:
    return Sim3Edge(
        source=int(payload["source"]),
        target=int(payload["target"]),
        kind=str(payload["kind"]),
        transform=_sim3_from_json(payload["transform"]),
        diagnostics=dict(payload.get("diagnostics") or {}),
    )


def _edges_from_component_transforms(
    payloads: Sequence[dict[str, Any]], transforms: Sequence[Sim3]
) -> list[Sim3Edge]:
    if len(payloads) != len(transforms):
        raise ValueError("cached edge Sim3 count mismatch")
    return [
        Sim3Edge(
            source=int(payload["source"]),
            target=int(payload["target"]),
            kind=str(payload["kind"]),
            transform=transform,
            diagnostics=dict(payload.get("diagnostics") or {}),
        )
        for payload, transform in zip(payloads, transforms)
    ]


def _edge_to_cache_json(edge: Sim3Edge) -> dict[str, Any]:
    return {
        "source": int(edge.source),
        "target": int(edge.target),
        "kind": edge.kind,
        "transform": {
            "scale": float(edge.transform.scale),
            "rotation": edge.transform.rotation.tolist(),
            "translation": edge.transform.translation.tolist(),
        },
        "diagnostics": _json_safe(edge.diagnostics),
    }


def _edge_from_cache_json(payload: dict[str, Any]) -> Sim3Edge:
    transform = payload["transform"]
    return Sim3Edge(
        source=int(payload["source"]),
        target=int(payload["target"]),
        kind=str(payload["kind"]),
        transform=Sim3(
            scale=float(transform["scale"]),
            rotation=np.asarray(transform["rotation"], dtype=np.float64),
            translation=np.asarray(transform["translation"], dtype=np.float64),
        ),
        diagnostics=dict(payload.get("diagnostics") or {}),
    )


def _basis_to_json(basis: HierarchicalAlphaBasis) -> dict[str, Any]:
    return {
        "coarse_alignments": _sim3_batch_to_json(basis.coarse_alignments),
        "frame_poses": _sim3_batch_to_json(basis.frame_poses),
        "apply_point_corrections": bool(basis.apply_point_corrections),
        "diagnostics": _json_safe(basis.diagnostics),
    }


def _basis_from_json(payload: dict[str, Any]) -> HierarchicalAlphaBasis:
    return HierarchicalAlphaBasis(
        coarse_alignments=_sim3_batch_from_json(
            payload.get("coarse_alignments", [])
        ),
        frame_poses=_sim3_batch_from_json(payload.get("frame_poses", [])),
        apply_point_corrections=bool(payload.get("apply_point_corrections", True)),
        diagnostics=dict(payload.get("diagnostics") or {}),
    )


def _basis_component_arrays(
    basis: HierarchicalAlphaBasis, edges: Sequence[Sim3Edge]
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for prefix, transforms in (
        ("coarse", basis.coarse_alignments),
        ("frame", basis.frame_poses),
    ):
        arrays[f"{prefix}_scales"] = np.asarray(
            [item.scale for item in transforms], dtype=np.float64
        )
        arrays[f"{prefix}_rotations"] = np.asarray(
            [item.rotation for item in transforms], dtype=np.float64
        ).reshape(len(transforms), 3, 3)
        arrays[f"{prefix}_translations"] = np.asarray(
            [item.translation for item in transforms], dtype=np.float64
        ).reshape(len(transforms), 3)
    arrays["edge_scales"] = np.asarray(
        [item.transform.scale for item in edges], dtype=np.float64
    )
    arrays["edge_rotations"] = np.asarray(
        [item.transform.rotation for item in edges], dtype=np.float64
    ).reshape(len(edges), 3, 3)
    arrays["edge_translations"] = np.asarray(
        [item.transform.translation for item in edges], dtype=np.float64
    ).reshape(len(edges), 3)
    return arrays


def _basis_from_component_arrays(
    payload: dict[str, Any], arrays: dict[str, np.ndarray]
) -> HierarchicalAlphaBasis:
    expected_counts = {
        "coarse": int(
            payload.get("coarse_count", len(payload.get("coarse_alignments", [])))
        ),
        "frame": int(
            payload.get("frame_count", len(payload.get("frame_poses", [])))
        ),
    }
    transforms: dict[str, list[Sim3]] = {}
    for prefix, count in expected_counts.items():
        scales = arrays[f"{prefix}_scales"]
        rotations = arrays[f"{prefix}_rotations"]
        translations = arrays[f"{prefix}_translations"]
        if scales.shape != (count,):
            raise ValueError(f"cached {prefix} Sim3 count mismatch")
        transforms[prefix] = _sim3_batch_from_components(
            scales, rotations, translations
        )
    return HierarchicalAlphaBasis(
        coarse_alignments=transforms["coarse"],
        frame_poses=transforms["frame"],
        apply_point_corrections=bool(payload.get("apply_point_corrections", True)),
        diagnostics=dict(payload.get("diagnostics") or {}),
    )


def _basis_component_cache_paths(
    root: Path, manifest_id: str
) -> tuple[Path, Path, Path]:
    directory = root / ".basis_component_cache"
    return (
        directory / f"{manifest_id}.npz",
        directory / f"{manifest_id}.json",
        directory / f"{manifest_id}.lock",
    )


def _try_load_basis_component_cache(
    payload: dict[str, Any],
    edge_payloads: Sequence[dict[str, Any]],
    npz_path: Path,
    metadata_path: Path,
    manifest_id: str,
) -> tuple[HierarchicalAlphaBasis, list[Sim3]] | None:
    try:
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("schema") != _BASIS_COMPONENT_CACHE_SCHEMA
            or metadata.get("manifest_id") != manifest_id
            or metadata.get("cache_file_identity") != _file_identity_file(npz_path)
        ):
            return None
        with np.load(npz_path, allow_pickle=False) as archive:
            names = {
                "coarse_scales",
                "coarse_rotations",
                "coarse_translations",
                "frame_scales",
                "frame_rotations",
                "frame_translations",
                "edge_scales",
                "edge_rotations",
                "edge_translations",
            }
            if set(archive.files) != names:
                return None
            arrays = {name: archive[name] for name in names}
        basis = _basis_from_component_arrays(payload, arrays)
        if arrays["edge_scales"].shape != (len(edge_payloads),):
            raise ValueError("cached edge Sim3 count mismatch")
        edge_transforms = _sim3_batch_from_components(
            arrays["edge_scales"],
            arrays["edge_rotations"],
            arrays["edge_translations"],
        )
        return basis, edge_transforms
    except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError, ValueError):
        return None


def _load_basis_with_component_cache(
    root: Path, manifest: dict[str, Any]
) -> tuple[HierarchicalAlphaBasis, list[Sim3], dict[str, Any]]:
    payload = manifest["basis"]
    edge_payloads = manifest["edges"]
    manifest_id = str(manifest["_validated_manifest_id"])
    npz_path, metadata_path, lock_path = _basis_component_cache_paths(
        root, manifest_id
    )
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                cached = _try_load_basis_component_cache(
                    payload,
                    edge_payloads,
                    npz_path,
                    metadata_path,
                    manifest_id,
                )
                if cached is not None:
                    basis, edge_transforms = cached
                    return basis, edge_transforms, {
                        "mode": "hit",
                        "schema": _BASIS_COMPONENT_CACHE_SCHEMA,
                    }
                if manifest.get("_replay_compact"):
                    raise _CompactBasisComponentCacheMiss
                basis = _basis_from_json(payload)
                edges = [_edge_from_json(item) for item in edge_payloads]
                try:
                    _atomic_write_npz(npz_path, _basis_component_arrays(basis, edges))
                    _atomic_write_json(
                        metadata_path,
                        {
                            "schema": _BASIS_COMPONENT_CACHE_SCHEMA,
                            "manifest_id": manifest_id,
                            "cache_file_identity": _file_identity_file(npz_path),
                        },
                    )
                    mode = "miss_published"
                except OSError:
                    mode = "miss_read_only"
                return basis, [edge.transform for edge in edges], {
                    "mode": mode,
                    "schema": _BASIS_COMPONENT_CACHE_SCHEMA,
                }
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        basis = _basis_from_json(payload)
        edges = [_edge_from_json(item) for item in edge_payloads]
        return basis, [edge.transform for edge in edges], {
            "mode": "unavailable",
            "schema": _BASIS_COMPONENT_CACHE_SCHEMA,
        }


def _trajectory_component_cache_paths(
    root: Path, manifest_id: str
) -> tuple[Path, Path, Path]:
    directory = root / ".trajectory_component_cache"
    return (
        directory / f"{manifest_id}.npz",
        directory / f"{manifest_id}.json",
        directory / f"{manifest_id}.lock",
    )


def _trajectory_chunks_from_component_arrays(
    arrays: dict[str, np.ndarray], ranges: Sequence[ChunkRange]
) -> list[ChunkPrediction]:
    offsets = np.asarray(arrays["offsets"], dtype=np.int64)
    expected_offsets = np.concatenate(
        (
            np.zeros(1, dtype=np.int64),
            np.cumsum([item.size for item in ranges], dtype=np.int64),
        )
    )
    if not np.array_equal(offsets, expected_offsets):
        raise ValueError("trajectory component cache chunk offsets mismatch")
    extrinsic = np.asarray(arrays["extrinsic_c2w"])
    intrinsic = np.asarray(arrays["intrinsic"])
    total = int(expected_offsets[-1])
    if extrinsic.shape != (total, 4, 4) or intrinsic.shape != (total, 3, 3):
        raise ValueError("trajectory component cache array shape mismatch")
    if not np.isfinite(extrinsic).all() or not np.isfinite(intrinsic).all():
        raise ValueError("trajectory component cache contains non-finite values")
    chunks: list[ChunkPrediction] = []
    for index, item in enumerate(ranges):
        start = int(offsets[index])
        end = int(offsets[index + 1])
        chunks.append(
            ChunkPrediction._from_finite_validated_arrays(
                _token=_FINITE_VALIDATED_TOKEN,
                _validated_fields=("extrinsic_c2w", "intrinsic"),
                world_points=np.zeros((item.size, 1, 1, 3), dtype=np.float32),
                world_points_conf=np.zeros((item.size, 1, 1), dtype=np.float32),
                extrinsic_c2w=extrinsic[start:end],
                intrinsic=intrinsic[start:end],
            )
        )
    return chunks


def _trajectory_component_input_identities(
    root: Path, manifest: dict[str, Any], *, workers: int
) -> dict[str, dict[str, Any]]:
    tasks: list[tuple[str, str]] = []
    required = {"extrinsic_c2w", "intrinsic"}
    for artifact in manifest.get("chunks", []):
        for field in artifact.get("fields", []):
            if field.get("name") in required:
                tasks.append((str(field["path"]), str(field["file_identity"])))

    def identify(task: tuple[str, str]) -> tuple[str, dict[str, Any]]:
        relative, expected_file_identity = task
        return relative, _ArtifactValidationReceipt._identity(
            root / relative, expected_file_identity
        )

    actual_workers = min(max(int(workers), 1), len(tasks)) if tasks else 0
    if actual_workers > 1:
        with ThreadPoolExecutor(
            max_workers=actual_workers,
            thread_name_prefix="hip3r-trajectory-cache-identity",
        ) as executor:
            return dict(executor.map(identify, tasks))
    return dict(identify(task) for task in tasks)


def _try_load_trajectory_component_cache(
    root: Path,
    *,
    manifest_id: str,
    manifest: dict[str, Any],
    ranges: Sequence[ChunkRange],
    workers: int,
) -> list[ChunkPrediction] | None:
    npz_path, metadata_path, _ = _trajectory_component_cache_paths(
        root, manifest_id
    )
    try:
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("schema") != _TRAJECTORY_COMPONENT_CACHE_SCHEMA
            or metadata.get("manifest_id") != manifest_id
            or metadata.get("cache_file_identity") != _file_identity_file(npz_path)
            or metadata.get("input_identities")
            != _trajectory_component_input_identities(
                root, manifest, workers=workers
            )
        ):
            return None
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != {"offsets", "extrinsic_c2w", "intrinsic"}:
                return None
            arrays = {name: archive[name] for name in archive.files}
        return _trajectory_chunks_from_component_arrays(arrays, ranges)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError, ValueError):
        return None


def _publish_trajectory_component_cache(
    root: Path,
    *,
    manifest_id: str,
    manifest: dict[str, Any],
    chunks: Sequence[ChunkPrediction],
    ranges: Sequence[ChunkRange],
    workers: int,
) -> str:
    npz_path, metadata_path, lock_path = _trajectory_component_cache_paths(
        root, manifest_id
    )
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            if _try_load_trajectory_component_cache(
                root,
                manifest_id=manifest_id,
                manifest=manifest,
                ranges=ranges,
                workers=workers,
            ) is not None:
                return "hit_after_lock"
            offsets = np.concatenate(
                (
                    np.zeros(1, dtype=np.int64),
                    np.cumsum([item.size for item in ranges], dtype=np.int64),
                )
            )
            arrays = {
                "offsets": offsets,
                "extrinsic_c2w": np.concatenate(
                    [np.asarray(chunk.extrinsic_c2w) for chunk in chunks], axis=0
                ),
                "intrinsic": np.concatenate(
                    [np.asarray(chunk.intrinsic) for chunk in chunks], axis=0
                ),
            }
            _atomic_write_npz(npz_path, arrays)
            _atomic_write_json(
                metadata_path,
                {
                    "schema": _TRAJECTORY_COMPONENT_CACHE_SCHEMA,
                    "manifest_id": manifest_id,
                    "cache_file_identity": _file_identity_file(npz_path),
                    "input_identities": _trajectory_component_input_identities(
                        root, manifest, workers=workers
                    ),
                },
            )
            return "miss_published"
    except OSError:
        return "miss_read_only"


def select_loop_edges(
    edges: Sequence[Sim3Edge],
    pairs: Sequence[LoopPair],
    *,
    policy: str = "all",
    budget: int | None = None,
) -> tuple[list[Sim3Edge], dict[str, Any]]:
    """Select already-estimated loop edges using inference-only evidence."""
    policies = {"all", "recent", "uniform", "descriptor", "geometry_utility"}
    if policy not in policies:
        raise ValueError(f"unsupported loop edge policy: {policy!r}")
    if len(edges) != len(pairs):
        raise ValueError(f"loop edge/pair count mismatch: edges={len(edges)} pairs={len(pairs)}")
    if budget is not None and budget < 0:
        raise ValueError("loop edge budget must be nonnegative")
    if policy != "all" and budget is None:
        raise ValueError(f"loop edge budget is required for policy={policy}")

    candidates = [
        _loop_candidate(index, edge, pair)
        for index, (edge, pair) in enumerate(zip(edges, pairs))
    ]
    _normalize_loop_features(candidates)
    limit = len(candidates) if budget is None else min(budget, len(candidates))

    if policy == "all":
        ranked = candidates
    elif policy == "recent":
        ranked = sorted(
            candidates,
            key=lambda item: (-item["recent_key"], item["index"]),
        )
    elif policy == "uniform":
        ranked = [candidates[index] for index in _uniform_indices(len(candidates), limit)]
    elif policy == "descriptor":
        ranked = sorted(
            candidates,
            key=lambda item: (-item["features"]["descriptor_similarity"], item["index"]),
        )
    else:
        ranked = sorted(candidates, key=lambda item: (-item["score"], item["index"]))

    selected = ranked[:limit]
    selected_indices = {item["index"] for item in selected}
    for rank, item in enumerate(selected):
        item["rank"] = rank
    records = []
    for item in candidates:
        records.append(
            {
                "loop_id": item["loop_id"],
                "selected": item["index"] in selected_indices,
                "rank": item.get("rank"),
                "score": item["score"],
                "features": item["features"],
            }
        )

    selected_ids = [item["loop_id"] for item in selected]
    rejected_ids = [item["loop_id"] for item in candidates if item["index"] not in selected_indices]
    protocol = {
        "schema": "hip3r_loop_edge_selection_v1",
        "policy": policy,
        "budget": budget,
        "feature_sources": ["loop_pair_similarity", "sim3_edge_diagnostics"],
        "feature_normalization": "finite_cohort_minmax_and_clipped_unit_interval_v1",
        "geometry_weights": {
            "descriptor_similarity": 0.35,
            "residual_quality": 0.25,
            "direction_quality": 0.15,
            "support": 0.15,
            "status_ok": 0.10,
        },
        "evidence_scope": "inference_only",
    }
    diagnostics = {
        "schema": protocol["schema"],
        "policy": policy,
        "activation": bool(budget is not None and len(candidates) > limit),
        "budget": budget,
        "num_candidates": len(candidates),
        "num_selected": len(selected),
        "selected_loop_ids": selected_ids,
        "rejected_loop_ids": rejected_ids,
        "protocol_id": _protocol_id(protocol),
        "protocol": protocol,
        "candidates": records,
    }
    return [edges[item["index"]] for item in selected], diagnostics


def _loop_candidate(index: int, edge: Sim3Edge, pair: LoopPair) -> dict[str, Any]:
    if edge.kind != "loop":
        raise ValueError(f"loop selector received non-loop edge at index {index}: {edge.kind!r}")
    diag = edge.diagnostics
    subedges = [diag.get("joint_to_a", {}), diag.get("joint_to_b", {})]
    used = [item.get("num_used") for item in subedges if item.get("num_used") is not None]
    return {
        "index": index,
        "loop_id": f"loop_{index:04d}_{pair.chunk_a}_{pair.chunk_b}",
        "recent_key": max(pair.chunk_a, pair.chunk_b),
        "raw": {
            "descriptor_similarity": pair.similarity,
            "residual_quality": diag.get("residual_p95"),
            "direction_quality": diag.get("direction_error"),
            "support": min(used) if used else None,
            "status_ok": diag.get("status") == "ok",
        },
    }


def _normalize_loop_features(candidates: list[dict[str, Any]]) -> None:
    residual_values = [item["raw"]["residual_quality"] for item in candidates]
    direction_values = [item["raw"]["direction_quality"] for item in candidates]
    support_values = [item["raw"]["support"] for item in candidates]
    for index, item in enumerate(candidates):
        similarity = _unit_value(item["raw"]["descriptor_similarity"])
        features = {
            "descriptor_similarity": similarity,
            "residual_quality": _cohort_unit(residual_values, index, lower_is_better=True),
            "direction_quality": _cohort_unit(direction_values, index, lower_is_better=True),
            "support": _cohort_unit(support_values, index),
            "status_ok": float(item["raw"]["status_ok"]),
        }
        item["features"] = features
        item["score"] = float(
            0.35 * features["descriptor_similarity"]
            + 0.25 * features["residual_quality"]
            + 0.15 * features["direction_quality"]
            + 0.15 * features["support"]
            + 0.10 * features["status_ok"]
        )


def _unit_value(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number):
        return 0.0
    return float(np.clip(number, 0.0, 1.0))


def _cohort_unit(values: Sequence[Any], index: int, *, lower_is_better: bool = False) -> float:
    numeric = [_finite_float(value) for value in values]
    finite = [value for value in numeric if value is not None]
    value = numeric[index]
    if value is None or not finite:
        return 0.0
    low, high = min(finite), max(finite)
    normalized = 1.0 if high == low else (float(value) - low) / (high - low)
    if lower_is_better:
        normalized = 1.0 - normalized
    return float(np.clip(normalized, 0.0, 1.0))


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _uniform_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def _reliability_summary(
    chunks: Sequence[dict[str, Any]], edges: Sequence[Sim3Edge]
) -> dict[str, Any]:
    attention_samples = [
        sample
        for chunk in chunks
        for module in chunk.get("attention", [])
        for sample in module.get("samples", [])
    ]
    residual_samples = [
        sample
        for chunk in chunks
        for module in chunk.get("token_residual", [])
        for sample in module.get("samples", [])
    ]
    confidence_means = [
        float(chunk["confidence"]["mean"])
        for chunk in chunks
        if chunk.get("confidence", {}).get("mean") is not None
    ]
    edge_residuals = [
        float(edge.diagnostics["residual_p90"])
        for edge in edges
        if edge.diagnostics.get("residual_p90") is not None
    ]
    return {
        "attention_entropy_mean": _mean_or_none(
            sample.get("entropy_mean") for sample in attention_samples
        ),
        "attention_max_response_mean": _mean_or_none(
            sample.get("max_response_mean") for sample in attention_samples
        ),
        "token_residual_rms_mean": _mean_or_none(
            sample.get("rms") for sample in residual_samples
        ),
        "confidence_mean": _mean_or_none(confidence_means),
        "edge_residual_p90_mean": _mean_or_none(edge_residuals),
        "num_attention_samples": len(attention_samples),
        "num_token_residual_samples": len(residual_samples),
        "num_edges": len(edges),
    }


def _mean_or_none(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _filter_coarse_edges_by_max_loop_span(
    edges: Sequence[Sim3Edge],
    max_loop_span: int | None,
) -> tuple[tuple[Sim3Edge, ...], dict[str, Any]]:
    source = tuple(edges)
    num_loop_before = sum(1 for edge in source if edge.kind == "loop")
    if max_loop_span is None:
        return source, {
            "max_loop_span": None,
            "num_loop_edges_before_span_filter": num_loop_before,
            "num_loop_edges_dropped_by_span": 0,
        }
    kept = tuple(
        edge
        for edge in source
        if edge.kind != "loop" or abs(edge.source - edge.target) <= int(max_loop_span)
    )
    num_loop_after = sum(1 for edge in kept if edge.kind == "loop")
    return kept, {
        "max_loop_span": int(max_loop_span),
        "num_loop_edges_before_span_filter": num_loop_before,
        "num_loop_edges_dropped_by_span": num_loop_before - num_loop_after,
    }


def _optimize_coarse_anchor_with_loop_continuation(
    num_chunks: int,
    edges: Sequence[Sim3Edge],
    initial_alignments: Sequence[Sim3],
    *,
    config: ReconstructionConfig,
) -> dict[str, Any]:
    """Admit loop edges by increasing chunk span before the final full solve."""

    configured_full_max_nfev = int(
        config.hierarchical_coarse_max_nfev
        if config.hierarchical_coarse_max_nfev is not None
        else config.hierarchical_max_nfev
    )
    edges, span_filter = _filter_coarse_edges_by_max_loop_span(
        edges, config.hierarchical_coarse_max_loop_span
    )

    def solve(
        stage_edges: Sequence[Sim3Edge],
        stage_initial: Sequence[Sim3],
        *,
        max_nfev: int,
    ) -> dict[str, Any]:
        return optimize_sim3_pose_graph(
            num_chunks,
            stage_edges,
            stage_initial,
            max_nfev=max_nfev,
            robust_loss=config.hierarchical_robust_loss,
            f_scale=config.hierarchical_f_scale,
            use_jac_sparsity=config.use_jac_sparsity,
            analytic_jacobian=config.hierarchical_coarse_analytic_jacobian,
            x_scale=config.hierarchical_coarse_x_scale,
            lsmr_tolerance=config.hierarchical_coarse_lsmr_tolerance,
            residual_component_weights=config.hierarchical_coarse_residual_component_weights,
            record_residual_components=config.hierarchical_record_residual_components,
        )

    configured_stages = int(config.hierarchical_coarse_loop_continuation_stages)
    adjacent_first = bool(config.hierarchical_coarse_adjacent_first_stage)
    loop_indices = [index for index, edge in enumerate(edges) if edge.kind == "loop"]
    if configured_stages == 1 and not adjacent_first:
        return solve(edges, initial_alignments, max_nfev=configured_full_max_nfev)
    if not loop_indices:
        result = solve(edges, initial_alignments, max_nfev=configured_full_max_nfev)
        diagnostics = dict(result.get("diagnostics") or {})
        diagnostics["loop_continuation"] = {
            "mode": "disabled",
            "requested_stages": configured_stages,
            "effective_stages": 1,
            "adaptive_enabled": bool(
                config.hierarchical_coarse_adaptive_loop_continuation
            ),
            "adjacent_first_stage": adjacent_first,
            "adaptive_triggered": False,
            "actual_stages": 1,
            "num_loop_edges": len(loop_indices),
            "reason": "no_loop_edges",
            **span_filter,
        }
        return {**result, "diagnostics": diagnostics}

    ordered_loop_indices = sorted(
        loop_indices,
        key=lambda index: (
            abs(edges[index].source - edges[index].target),
            min(edges[index].source, edges[index].target),
            max(edges[index].source, edges[index].target),
            index,
        ),
    )
    ordered_spans = [
        abs(edges[index].source - edges[index].target)
        for index in ordered_loop_indices
    ]
    shortest_span = ordered_spans[0]
    shortest_span_count = sum(span == shortest_span for span in ordered_spans)
    adaptive_enabled = bool(
        config.hierarchical_coarse_adaptive_loop_continuation
    )
    adaptive_triggered = (
        adaptive_enabled
        and ordered_spans[-1] > shortest_span
        and shortest_span_count * configured_stages
        >= len(ordered_spans) * (configured_stages - 1)
    )
    effective_stages = (
        min(configured_stages * 2, len(ordered_loop_indices))
        if adaptive_triggered
        else configured_stages
    )
    full_max_nfev = int(
        config.hierarchical_coarse_adaptive_full_max_nfev
        if adaptive_triggered
        and config.hierarchical_coarse_adaptive_full_max_nfev is not None
        else configured_full_max_nfev
    )
    actual_stages = min(effective_stages, len(ordered_loop_indices))
    non_loop_indices = {
        index for index, edge in enumerate(edges) if edge.kind != "loop"
    }
    current_alignments = list(initial_alignments)
    stage_diagnostics: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None
    if adjacent_first:
        adjacent_edges = [
            edge
            for index, edge in enumerate(edges)
            if index in non_loop_indices
        ]
        stage_max_nfev = int(config.hierarchical_coarse_continuation_max_nfev)
        stage_started = time.perf_counter()
        final_result = solve(
            adjacent_edges,
            current_alignments,
            max_nfev=stage_max_nfev,
        )
        stage_wall_seconds = time.perf_counter() - stage_started
        current_alignments = list(final_result["alignments"])
        solver_diagnostics = dict(final_result.get("diagnostics") or {})
        stage_diagnostics.append(
            {
                "stage_index": 0,
                "stage_kind": "adjacent_only",
                "final_stage": False,
                "num_edges": len(adjacent_edges),
                "num_loop_edges": 0,
                "max_loop_span": None,
                "max_nfev": stage_max_nfev,
                "status": solver_diagnostics.get("status"),
                "success": solver_diagnostics.get("success"),
                "termination_status": solver_diagnostics.get(
                    "termination_status"
                ),
                "num_function_evaluations": solver_diagnostics.get(
                    "num_function_evaluations"
                ),
                "initial_cost": solver_diagnostics.get("initial_cost"),
                "final_cost": solver_diagnostics.get("final_cost"),
                "optimality": solver_diagnostics.get("optimality"),
                "active_bound_count": solver_diagnostics.get(
                    "active_bound_count"
                ),
                "wall_seconds": stage_wall_seconds,
            }
        )
    for stage_index in range(1, actual_stages + 1):
        admitted_count = -(
            -(len(ordered_loop_indices) * stage_index) // actual_stages
        )
        admitted_loop_indices = set(ordered_loop_indices[:admitted_count])
        selected_indices = non_loop_indices | admitted_loop_indices
        stage_edges = [
            edge for index, edge in enumerate(edges) if index in selected_indices
        ]
        final_stage = stage_index == actual_stages
        stage_max_nfev = (
            full_max_nfev
            if final_stage
            else int(config.hierarchical_coarse_continuation_max_nfev)
        )
        stage_started = time.perf_counter()
        final_result = solve(
            stage_edges,
            current_alignments,
            max_nfev=stage_max_nfev,
        )
        stage_wall_seconds = time.perf_counter() - stage_started
        current_alignments = list(final_result["alignments"])
        solver_diagnostics = dict(final_result.get("diagnostics") or {})
        admitted_spans = [
            abs(edges[index].source - edges[index].target)
            for index in admitted_loop_indices
        ]
        stage_diagnostics.append(
            {
                "stage_index": stage_index,
                "stage_kind": (
                    "full_graph_initial" if final_stage else "loop_admission"
                ),
                "final_stage": final_stage,
                "num_edges": len(stage_edges),
                "num_loop_edges": admitted_count,
                "max_loop_span": max(admitted_spans) if admitted_spans else None,
                "max_nfev": stage_max_nfev,
                "status": solver_diagnostics.get("status"),
                "success": solver_diagnostics.get("success"),
                "termination_status": solver_diagnostics.get("termination_status"),
                "num_function_evaluations": solver_diagnostics.get(
                    "num_function_evaluations"
                ),
                "initial_cost": solver_diagnostics.get("initial_cost"),
                "final_cost": solver_diagnostics.get("final_cost"),
                "optimality": solver_diagnostics.get("optimality"),
                "active_bound_count": solver_diagnostics.get(
                    "active_bound_count"
                ),
                "wall_seconds": stage_wall_seconds,
            }
        )

    assert final_result is not None
    adaptive_budget_fallback_attempted = False
    adaptive_budget_fallback_deferred_to_boundary_restart = False
    adaptive_budget_fallback_max_nfev = (
        config.hierarchical_coarse_adaptive_budget_fallback_max_nfev
    )
    solver_diagnostics = dict(final_result.get("diagnostics") or {})
    adaptive_budget_fallback_deferred_to_boundary_restart = bool(
        adaptive_triggered
        and config.hierarchical_coarse_prefer_boundary_restart_for_active_bounds
        and adaptive_budget_fallback_max_nfev is not None
        and solver_diagnostics.get("success") is not True
        and int(solver_diagnostics.get("termination_status") or 0) == 0
        and int(solver_diagnostics.get("active_bound_count") or 0) > 0
        and int(config.hierarchical_coarse_boundary_restart_passes) > 0
    )
    if (
        adaptive_triggered
        and adaptive_budget_fallback_max_nfev is not None
        and solver_diagnostics.get("success") is not True
        and int(solver_diagnostics.get("termination_status") or 0) == 0
        and not adaptive_budget_fallback_deferred_to_boundary_restart
    ):
        stage_started = time.perf_counter()
        final_result = solve(
            edges,
            current_alignments,
            max_nfev=int(adaptive_budget_fallback_max_nfev),
        )
        stage_wall_seconds = time.perf_counter() - stage_started
        current_alignments = list(final_result["alignments"])
        solver_diagnostics = dict(final_result.get("diagnostics") or {})
        adaptive_budget_fallback_attempted = True
        stage_diagnostics.append(
            {
                "stage_index": actual_stages + 1,
                "stage_kind": "full_graph_adaptive_budget_fallback",
                "final_stage": True,
                "num_edges": len(edges),
                "num_loop_edges": len(loop_indices),
                "max_loop_span": max(ordered_spans),
                "max_nfev": int(adaptive_budget_fallback_max_nfev),
                "status": solver_diagnostics.get("status"),
                "success": solver_diagnostics.get("success"),
                "termination_status": solver_diagnostics.get(
                    "termination_status"
                ),
                "num_function_evaluations": solver_diagnostics.get(
                    "num_function_evaluations"
                ),
                "initial_cost": solver_diagnostics.get("initial_cost"),
                "final_cost": solver_diagnostics.get("final_cost"),
                "optimality": solver_diagnostics.get("optimality"),
                "active_bound_count": solver_diagnostics.get(
                    "active_bound_count"
                ),
                "wall_seconds": stage_wall_seconds,
            }
        )
    boundary_restart_passes = int(
        config.hierarchical_coarse_boundary_restart_passes
    )
    boundary_restarts_attempted = 0
    boundary_restart_stop_reason = "not_requested"
    if boundary_restart_passes > 0:
        boundary_restart_stop_reason = "pass_limit"
        for restart_index in range(1, boundary_restart_passes + 1):
            solver_diagnostics = dict(final_result.get("diagnostics") or {})
            if solver_diagnostics.get("success") is True:
                boundary_restart_stop_reason = "converged"
                break
            if int(solver_diagnostics.get("active_bound_count") or 0) == 0:
                boundary_restart_stop_reason = "no_active_bounds"
                break
            stage_started = time.perf_counter()
            final_result = solve(
                edges,
                current_alignments,
                max_nfev=int(
                    config.hierarchical_coarse_boundary_restart_max_nfev
                ),
            )
            stage_wall_seconds = time.perf_counter() - stage_started
            current_alignments = list(final_result["alignments"])
            solver_diagnostics = dict(final_result.get("diagnostics") or {})
            boundary_restarts_attempted += 1
            stage_diagnostics.append(
                {
                    "stage_index": (
                        actual_stages
                        + int(adaptive_budget_fallback_attempted)
                        + restart_index
                    ),
                    "stage_kind": "full_graph_boundary_restart",
                    "restart_index": restart_index,
                    "final_stage": True,
                    "num_edges": len(edges),
                    "num_loop_edges": len(loop_indices),
                    "max_loop_span": max(
                        abs(edges[index].source - edges[index].target)
                        for index in loop_indices
                    ),
                    "max_nfev": int(
                        config.hierarchical_coarse_boundary_restart_max_nfev
                    ),
                    "status": solver_diagnostics.get("status"),
                    "success": solver_diagnostics.get("success"),
                    "termination_status": solver_diagnostics.get(
                        "termination_status"
                    ),
                    "num_function_evaluations": solver_diagnostics.get(
                        "num_function_evaluations"
                    ),
                    "initial_cost": solver_diagnostics.get("initial_cost"),
                    "final_cost": solver_diagnostics.get("final_cost"),
                    "optimality": solver_diagnostics.get("optimality"),
                    "active_bound_count": solver_diagnostics.get(
                        "active_bound_count"
                    ),
                    "wall_seconds": stage_wall_seconds,
                }
            )
        else:
            solver_diagnostics = dict(final_result.get("diagnostics") or {})
            if solver_diagnostics.get("success") is True:
                boundary_restart_stop_reason = "converged"
            elif int(solver_diagnostics.get("active_bound_count") or 0) == 0:
                boundary_restart_stop_reason = "no_active_bounds"

    diagnostics = dict(final_result.get("diagnostics") or {})
    diagnostics["loop_continuation"] = {
        "mode": "cumulative_loop_span",
        "ordering": "chunk_span_then_endpoints_then_original_index",
        "requested_stages": configured_stages,
        "effective_stages": effective_stages,
        "adaptive_enabled": adaptive_enabled,
        "adjacent_first_stage": adjacent_first,
        **span_filter,
        "adaptive_triggered": adaptive_triggered,
        "adaptive_rule": "shortest_span_fills_all_nonfinal_configured_stages",
        "shortest_loop_span": shortest_span,
        "shortest_span_loop_count": shortest_span_count,
        "shortest_span_loop_fraction": shortest_span_count / len(ordered_spans),
        "configured_full_max_nfev": configured_full_max_nfev,
        "adaptive_full_max_nfev": (
            None
            if config.hierarchical_coarse_adaptive_full_max_nfev is None
            else int(config.hierarchical_coarse_adaptive_full_max_nfev)
        ),
        "effective_full_max_nfev": full_max_nfev,
        "adaptive_budget_fallback_max_nfev": (
            None
            if adaptive_budget_fallback_max_nfev is None
            else int(adaptive_budget_fallback_max_nfev)
        ),
        "adaptive_budget_fallback_attempted": adaptive_budget_fallback_attempted,
        "prefer_boundary_restart_for_active_bounds": bool(
            config.hierarchical_coarse_prefer_boundary_restart_for_active_bounds
        ),
        "adaptive_budget_fallback_deferred_to_boundary_restart": (
            adaptive_budget_fallback_deferred_to_boundary_restart
        ),
        "actual_stages": actual_stages,
        "num_loop_edges": len(loop_indices),
        "intermediate_max_nfev": int(
            config.hierarchical_coarse_continuation_max_nfev
        ),
        "boundary_restart_trigger": "nonconverged_with_active_bounds",
        "boundary_restart_passes_requested": boundary_restart_passes,
        "boundary_restart_max_nfev": int(
            config.hierarchical_coarse_boundary_restart_max_nfev
        ),
        "boundary_restarts_attempted": boundary_restarts_attempted,
        "boundary_restart_stop_reason": boundary_restart_stop_reason,
        "stages": stage_diagnostics,
    }
    return {**final_result, "diagnostics": diagnostics}


def _solver_convergence_component(
    name: str, diagnostics: dict[str, Any]
) -> dict[str, Any]:
    status = str(diagnostics.get("status", "unknown"))
    optimizer_used_value = diagnostics.get("optimizer_used", False)
    optimizer_used = optimizer_used_value is True
    solver_mode = diagnostics.get("solver_mode")
    try:
        num_loop_edges = int(diagnostics.get("num_loop_edges"))
    except (TypeError, ValueError):
        num_loop_edges = None
    try:
        num_function_evaluations = int(
            diagnostics.get("num_function_evaluations")
        )
    except (TypeError, ValueError):
        num_function_evaluations = None
    factorized_no_loop_coarse_skip = (
        name == "coarse"
        and status == "ok"
        and optimizer_used_value is False
        and solver_mode == "factorized_adjacent_chain_no_absolute_coarse_solve"
        and diagnostics.get("factorized_frame_chart_residual_v4") is True
        and num_loop_edges == 0
        and num_function_evaluations == 0
    )
    benign_without_solver = status in {
        "empty",
        "no_edges",
        "no_variables",
        "no_constraints",
    } or (
        status == "ok" and solver_mode == "overlap_consensus"
    ) or factorized_no_loop_coarse_skip
    stationarity_acceptance = diagnostics.get(
        "max_nfev_stationarity_acceptance"
    )
    accepted_at_max_nfev = (
        name == "coarse"
        and isinstance(stationarity_acceptance, dict)
        and stationarity_acceptance.get("schema")
        == "hip3r_max_nfev_stationarity_acceptance_v1"
        and stationarity_acceptance.get("accepted") is True
        and isinstance(stationarity_acceptance.get("criteria"), dict)
        and bool(stationarity_acceptance["criteria"])
        and all(
            value is True
            for value in stationarity_acceptance["criteria"].values()
        )
    )
    reasons: list[str] = []
    if optimizer_used:
        if diagnostics.get("success") is not True and not accepted_at_max_nfev:
            reasons.append("success_is_not_true")
        if status != "ok" and not accepted_at_max_nfev:
            reasons.append(f"status={status}")
        termination_status = diagnostics.get("termination_status")
        if (
            termination_status is not None
            and int(termination_status) not in {1, 2, 3, 4}
            and not accepted_at_max_nfev
        ):
            reasons.append(f"termination_status={termination_status}")
    elif not benign_without_solver:
        reasons.append("optimizer_not_used")
        reasons.append(f"status={status}")
    for field in ("solution_finite", "residual_finite", "cost_finite"):
        if field in diagnostics and diagnostics[field] is not True:
            reasons.append(f"{field}=false")
    for field in ("initial_cost", "final_cost", "scipy_cost", "optimality"):
        if field in diagnostics and not np.isfinite(float(diagnostics[field])):
            reasons.append(f"{field}=nonfinite")
    return {
        "name": name,
        "valid": not reasons,
        "status": status,
        "success": diagnostics.get("success"),
        "termination_status": diagnostics.get("termination_status"),
        "solver_mode": solver_mode,
        "message": diagnostics.get("message"),
        "num_function_evaluations": diagnostics.get("num_function_evaluations"),
        "initial_cost": diagnostics.get("initial_cost"),
        "final_cost": diagnostics.get("final_cost"),
        "scipy_cost": diagnostics.get("scipy_cost"),
        "optimality": diagnostics.get("optimality"),
        "active_bound_count": diagnostics.get("active_bound_count"),
        "max_nfev_stationarity_acceptance": stationarity_acceptance,
        "reasons": reasons,
    }


def _annotate_coarse_max_nfev_stationarity_acceptance(
    diagnostics: dict[str, Any], *, threshold: float | None
) -> dict[str, Any]:
    """Audit a narrowly-scoped acceptance of a finite near-stationary solve.

    SciPy status 0 only says that the evaluation budget was exhausted. For very
    long graphs a solution can already be numerically stationary at that point.
    Preserve the original solver status and attach an explicit policy decision;
    the convergence gate recognizes only this internally generated annotation.
    """
    if threshold is None:
        return diagnostics
    result = dict(diagnostics)
    coarse = result.get("coarse_anchor")
    if not isinstance(coarse, dict):
        return result
    annotated = dict(coarse)

    def finite(field: str) -> bool:
        try:
            return bool(np.isfinite(float(annotated.get(field))))
        except (TypeError, ValueError):
            return False

    message = str(annotated.get("message", "")).lower()
    initial_cost = annotated.get("initial_cost")
    final_cost = annotated.get("final_cost")
    criteria = {
        "optimizer_used": annotated.get("optimizer_used") is True,
        "status_not_converged": annotated.get("status") == "not_converged",
        "success_false": annotated.get("success") is False,
        "termination_status_zero": annotated.get("termination_status") == 0,
        "maximum_evaluations_message": (
            "maximum number of function evaluations" in message
        ),
        "no_active_bounds": annotated.get("active_bound_count") == 0,
        "solution_finite": annotated.get("solution_finite") is True,
        "residual_finite": annotated.get("residual_finite") is True,
        "cost_finite": annotated.get("cost_finite") is True,
        "finite_initial_and_final_cost": (
            finite("initial_cost") and finite("final_cost")
        ),
        "cost_nonincreasing": (
            finite("initial_cost")
            and finite("final_cost")
            and float(final_cost) <= float(initial_cost)
        ),
        "optimality_below_threshold": (
            finite("optimality")
            and float(annotated["optimality"]) <= float(threshold)
        ),
        "gradient_inf_norm_below_threshold": (
            finite("gradient_inf_norm")
            and float(annotated["gradient_inf_norm"]) <= float(threshold)
        ),
    }
    annotated["max_nfev_stationarity_acceptance"] = {
        "schema": "hip3r_max_nfev_stationarity_acceptance_v1",
        "accepted": all(criteria.values()),
        "threshold": float(threshold),
        "criteria": criteria,
        "original_status": annotated.get("status"),
        "original_success": annotated.get("success"),
        "original_termination_status": annotated.get("termination_status"),
        "optimality": annotated.get("optimality"),
        "gradient_inf_norm": annotated.get("gradient_inf_norm"),
    }
    result["coarse_anchor"] = annotated
    return result


def _require_hierarchical_solver_convergence(
    diagnostics: dict[str, Any]
) -> dict[str, Any]:
    components = [_solver_convergence_component("fine", diagnostics)]
    coarse = diagnostics.get("coarse_anchor")
    if isinstance(coarse, dict):
        components.insert(0, _solver_convergence_component("coarse", coarse))
    gate = {
        "schema": "hip3r_solver_convergence_gate_v1",
        "required": True,
        "valid": all(component["valid"] for component in components),
        "components": components,
    }
    if not gate["valid"]:
        failed = [
            f"{component['name']}({','.join(component['reasons'])})"
            for component in components
            if not component["valid"]
        ]
        raise HierarchicalSolverConvergenceError(
            "hierarchical solver convergence gate failed: " + "; ".join(failed),
            gate,
        )
    return gate


def _hierarchical_solver_failure_diagnostics(
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    fine_fields = (
        "status",
        "optimizer_used",
        "solver_mode",
        "success",
        "termination_status",
        "message",
        "num_function_evaluations",
        "num_jacobian_evaluations",
        "initial_cost",
        "final_cost",
        "scipy_cost",
        "optimality",
        "active_bound_count",
        "num_function_evaluations_total",
        "num_jacobian_evaluations_total",
        "gradient_inf_norm",
        "solution_step_norm",
        "solution_finite",
        "residual_finite",
        "cost_finite",
    )
    return {
        "coarse_anchor": diagnostics.get("coarse_anchor"),
        "fine": {
            field: diagnostics.get(field)
            for field in fine_fields
            if field in diagnostics
        },
        "factorized_frame_chart_residual_v4": diagnostics.get(
            "factorized_frame_chart_residual_v4"
        ),
        "chain_global_scale_gauge_centering_v1": diagnostics.get(
            "chain_global_scale_gauge_centering_v1"
        ),
        "convergence_gate": diagnostics.get("convergence_gate"),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.floating):
        return _json_safe(value.item())
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite float is not valid JSON")
    return value


def _json_safe_diagnostic(value: Any) -> Any:
    """Best-effort JSON-safe diagnostics that preserves non-finite evidence."""

    if isinstance(value, dict):
        return {str(key): _json_safe_diagnostic(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_diagnostic(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe_diagnostic(value.tolist())
    if isinstance(value, np.floating):
        return _json_safe_diagnostic(value.item())
    if isinstance(value, np.integer):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return {"nonfinite_float": str(value)}
    return value


def _protocol_id(protocol: dict[str, Any]) -> str:
    payload = _json_safe(protocol)
    parts = [str(payload.get("schema", "protocol"))]
    for key in (
        "postprocess",
        "chunk_size",
        "overlap",
        "max_chunk_size",
        "forward_artifact_field_profile",
        "alpha",
        "apply_frame_corrections",
    ):
        if key in payload:
            parts.append(f"{key}_{payload[key]}")
    if "explicit_ranges_summary" in payload:
        summary = dict(payload.get("explicit_ranges_summary") or {})
        parts.append(f"ranges_{summary.get('num_chunks', 'unknown')}")
        parts.append(f"frames_{summary.get('num_frames', 'unknown')}")
    label = "__".join(parts)
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in label)


def _file_identity_file(path: Path) -> str:
    stat = path.stat()
    return f"size_{int(stat.st_size)}_mtime_ns_{int(stat.st_mtime_ns)}"


_NPY_SCAN_BLOCK_BYTES = 8 * 1024 * 1024
_VALIDATION_RECEIPT_SCHEMA = "hip3r_forward_validation_receipt_v1"
_TRAJECTORY_COMPONENT_CACHE_SCHEMA = "hip3r_trajectory_component_cache_v1"
_TRAJECTORY_REPLAY_COMPLETE_SCHEMA = "hip3r_trajectory_replay_complete_v1"
_BASIS_REPLAY_CACHE_SCHEMA = "hip3r_basis_replay_cache_v1"
_POSTPROCESS_EDGE_CACHE_SCHEMA = "hip3r_postprocess_edge_cache_v3"
_COARSE_ANCHOR_CACHE_SCHEMA = "hip3r_coarse_anchor_cache_v1"
_BASIS_COMPONENT_CACHE_SCHEMA = "hip3r_basis_component_cache_v2"


def _numerical_environment_identity() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "scipy", "torch"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema": "numerical_environment_identity_v1",
        "distribution_versions": versions,
    }


def _edge_cache_protocol(
    config: ReconstructionConfig, *, edge_stability_strata: int = 0
) -> dict[str, Any]:
    protocol = {
        "schema": _POSTPROCESS_EDGE_CACHE_SCHEMA,
        "implementation": "paired_confidence_linear_sample_irls_residual_quality_v3",
        "adjacent_pairing_policy": "global_frame_intersection_v1",
        "overlap": int(config.overlap),
        "use_scale": True,
        "confidence_threshold": config.confidence_threshold,
        "max_alignment_points": config.max_alignment_points,
        "max_sim3_iterations": int(config.max_sim3_iterations),
        "huber_delta": config.huber_delta,
        "residual_edge_weighting": bool(config.residual_edge_weighting),
        "residual_edge_quality_quantile": config.residual_edge_quality_quantile,
        "residual_scale": float(config.residual_scale),
        "min_edge_weight": float(config.min_edge_weight),
        "max_edge_weight": float(config.max_edge_weight),
        "loop_edge_policy": config.loop_edge_policy,
        "loop_edge_budget": config.loop_edge_budget,
    }
    if edge_stability_strata:
        protocol["edge_stability"] = {
            "schema": "sim3_estimator_stability/v1",
            "num_strata": int(edge_stability_strata),
            "partition": "global_frame_xy_mod_v1",
            "aggregation": "median_components_then_median_strata_v1",
        }
    return protocol


def _postprocess_edge_cache_path(
    root: Path,
    manifest_id: str,
    protocol: dict[str, Any],
) -> Path:
    key = _protocol_id(
        {
            "manifest_id": manifest_id,
            "edge_protocol": protocol,
        }
    )
    return root / ".postprocess_edge_cache" / f"{key}.json"


def _edge_cache_input_identities(
    root: Path,
    manifest: dict[str, Any],
    *,
    workers: int = 1,
) -> dict[str, dict[str, Any]]:
    tasks: list[tuple[str, str]] = []
    required = {"world_points", "world_points_conf"}
    for section in ("chunks", "loops"):
        for artifact in manifest.get(section, []):
            for field in artifact.get("fields", []):
                if field.get("name") not in required:
                    continue
                relative = str(field["path"])
                tasks.append((relative, str(field["file_identity"])))

    def identify(task: tuple[str, str]) -> tuple[str, dict[str, Any]]:
        relative, expected_file_identity = task
        return relative, _ArtifactValidationReceipt._identity(
            root / relative, expected_file_identity
        )

    actual_workers = min(max(int(workers), 1), len(tasks)) if tasks else 0
    if actual_workers > 1:
        with ThreadPoolExecutor(
            max_workers=actual_workers,
            thread_name_prefix="hip3r-edge-cache-identity",
        ) as executor:
            return dict(executor.map(identify, tasks))
    return dict(identify(task) for task in tasks)


def _load_postprocess_edge_cache(
    root: Path,
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    *,
    workers: int = 1,
) -> dict[str, Any] | None:
    complete = json.loads((root / "forward.complete.json").read_text())
    manifest_id = str(complete.get("manifest_id", ""))
    path = _postprocess_edge_cache_path(root, manifest_id, protocol)
    try:
        envelope = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("schema") != _POSTPROCESS_EDGE_CACHE_SCHEMA
        or payload.get("manifest_id") != manifest_id
        or payload.get("edge_protocol") != protocol
    ):
        return None
    try:
        if payload.get("input_identities") != _edge_cache_input_identities(
            root, manifest, workers=workers
        ):
            return None
        adjacent = [_edge_from_cache_json(item) for item in payload["adjacent"]]
        loops = [_edge_from_cache_json(item) for item in payload["loops"]]
        if len(adjacent) != max(len(manifest.get("chunks", [])) - 1, 0):
            return None
        if any(edge.kind != "adjacent" for edge in adjacent):
            return None
        if any(edge.kind != "loop" for edge in loops):
            return None
        selected_ids = payload["loop_selection"].get("selected_loop_ids", [])
        if len(loops) != len(selected_ids):
            return None
        return {
            "adjacent": adjacent,
            "loops": loops,
            "loop_selection": dict(payload["loop_selection"]),
            "loop_diagnostics": dict(payload["loop_diagnostics"]),
            "path": path,
            "cache_file_identity": _file_identity_file(path),
            "input_identity_workers": min(
                max(int(workers), 1), len(payload["input_identities"])
            ),
        }
    except (KeyError, OSError, TypeError, ValueError):
        return None


def _write_postprocess_edge_cache(
    root: Path,
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    *,
    adjacent: Sequence[Sim3Edge],
    loops: Sequence[Sim3Edge],
    loop_selection: dict[str, Any],
    loop_diagnostics: dict[str, Any],
    workers: int = 1,
) -> dict[str, Any] | None:
    try:
        complete = json.loads((root / "forward.complete.json").read_text())
        manifest_id = str(complete.get("manifest_id", ""))
        payload = {
            "schema": _POSTPROCESS_EDGE_CACHE_SCHEMA,
            "manifest_id": manifest_id,
            "edge_protocol": protocol,
            "input_identities": _edge_cache_input_identities(
                root, manifest, workers=workers
            ),
            "adjacent": [_edge_to_cache_json(edge) for edge in adjacent],
            "loops": [_edge_to_cache_json(edge) for edge in loops],
            "loop_selection": loop_selection,
            "loop_diagnostics": loop_diagnostics,
        }
        path = _postprocess_edge_cache_path(root, manifest_id, protocol)
        _atomic_write_json(
            path,
            {
                "payload": payload,
            },
        )
    except OSError:
        # Read-only forward artifacts remain valid; only the replay speedup is lost.
        return None
    return {
        "path": path,
        "key": path.stem,
        "cache_file_identity": _file_identity_file(path),
    }


def _coarse_anchor_cache_protocol(
    config: ReconstructionConfig,
    *,
    edge_cache_key: str,
    edge_cache_file_identity: str,
) -> dict[str, Any]:
    protocol = {
        "schema": _COARSE_ANCHOR_CACHE_SCHEMA,
        "implementation": "sparse_torch_block_analytic_exact_residual_v1",
        "numerical_environment": _numerical_environment_identity(),
        "edge_cache_key": edge_cache_key,
        "edge_cache_file_identity": str(edge_cache_file_identity),
        "max_nfev": int(
            config.hierarchical_coarse_max_nfev
            if config.hierarchical_coarse_max_nfev is not None
            else config.hierarchical_max_nfev
        ),
        "robust_loss": config.hierarchical_robust_loss,
        "f_scale": float(config.hierarchical_f_scale),
        "use_jac_sparsity": bool(config.use_jac_sparsity),
        "analytic_jacobian": bool(config.hierarchical_coarse_analytic_jacobian),
        "x_scale": config.hierarchical_coarse_x_scale,
        "lsmr_tolerance": config.hierarchical_coarse_lsmr_tolerance,
    }
    if tuple(config.hierarchical_coarse_residual_component_weights) != (1.0, 1.0, 1.0):
        protocol["residual_component_weights"] = list(config.hierarchical_coarse_residual_component_weights)
    if config.hierarchical_record_residual_components:
        protocol["record_residual_components"] = True
    if (
        config.hierarchical_coarse_loop_continuation_stages != 1
        or config.hierarchical_coarse_adjacent_first_stage
    ):
        protocol["continuation_max_nfev"] = (
            config.hierarchical_coarse_continuation_max_nfev
        )
    if config.hierarchical_coarse_loop_continuation_stages != 1:
        protocol["loop_continuation_stages"] = (
            config.hierarchical_coarse_loop_continuation_stages
        )
    if config.hierarchical_coarse_adjacent_first_stage:
        protocol["adjacent_first_stage"] = True
    if config.hierarchical_coarse_max_loop_span is not None:
        protocol["max_loop_span"] = int(config.hierarchical_coarse_max_loop_span)
    if config.hierarchical_coarse_adaptive_loop_continuation:
        protocol["adaptive_loop_continuation"] = True
        if config.hierarchical_coarse_adaptive_full_max_nfev is not None:
            protocol["adaptive_full_max_nfev"] = int(
                config.hierarchical_coarse_adaptive_full_max_nfev
            )
        if (
            config.hierarchical_coarse_adaptive_budget_fallback_max_nfev
            is not None
        ):
            protocol["adaptive_budget_fallback_max_nfev"] = int(
                config.hierarchical_coarse_adaptive_budget_fallback_max_nfev
            )
        if config.hierarchical_coarse_prefer_boundary_restart_for_active_bounds:
            protocol["prefer_boundary_restart_for_active_bounds"] = True
    if config.hierarchical_coarse_boundary_restart_passes != 0:
        protocol["boundary_restart_passes"] = (
            config.hierarchical_coarse_boundary_restart_passes
        )
        protocol["boundary_restart_max_nfev"] = (
            config.hierarchical_coarse_boundary_restart_max_nfev
        )
    if config.chain_global_scale_gauge_centering_v1:
        protocol["chain_global_scale_gauge_centering_v1"] = True
        protocol["chain_global_scale_gauge_centering_scheme"] = (
            "factorized_frame_chart_residual_v4"
        )
        protocol["factorized_frame_chart_residual_v4_integration"] = (
            "defer_translation_gate_skip_no_loop_absolute_coarse_v1"
        )
        protocol["chain_global_scale_gauge_centering_span_cap"] = float(
            config.chain_global_scale_gauge_centering_span_cap
        )
    return protocol


def _coarse_anchor_cache_path(root: Path, protocol: dict[str, Any]) -> Path:
    return root / ".coarse_anchor_cache" / f"{_protocol_id(protocol)}.json"


def _load_coarse_anchor_cache(
    root: Path,
    protocol: dict[str, Any],
    *,
    expected_num_chunks: int,
) -> dict[str, Any] | None:
    path = _coarse_anchor_cache_path(root, protocol)
    try:
        envelope = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("schema") != _COARSE_ANCHOR_CACHE_SCHEMA
        or payload.get("protocol") != protocol
    ):
        return None
    try:
        alignments = [
            Sim3(
                scale=float(item["scale"]),
                rotation=np.asarray(item["rotation"], dtype=np.float64),
                translation=np.asarray(item["translation"], dtype=np.float64),
            )
            for item in payload["alignments"]
        ]
        if len(alignments) != expected_num_chunks:
            return None
        return {
            "alignments": alignments,
            "diagnostics": dict(payload["diagnostics"]),
            "path": path,
            "protocol": protocol,
        }
    except (KeyError, TypeError, ValueError):
        return None


def _load_coarse_anchor_cache_by_key(
    root: Path,
    key: str,
    config: ReconstructionConfig,
    *,
    expected_num_chunks: int,
) -> dict[str, Any] | None:
    path = root / ".coarse_anchor_cache" / f"{key}.json"
    try:
        envelope = json.loads(path.read_text())
        payload = envelope["payload"]
        protocol = payload["protocol"]
        edge_cache_key = str(protocol["edge_cache_key"])
        edge_file_identity = str(protocol["edge_cache_file_identity"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError, TypeError):
        return None
    if _protocol_id(protocol) != key:
        return None
    if protocol != _coarse_anchor_cache_protocol(
        config,
        edge_cache_key=edge_cache_key,
        edge_cache_file_identity=edge_file_identity,
    ):
        return None
    return _load_coarse_anchor_cache(
        root,
        protocol,
        expected_num_chunks=expected_num_chunks,
    )


def _write_coarse_anchor_cache(
    root: Path,
    protocol: dict[str, Any],
    *,
    alignments: Sequence[Sim3],
    diagnostics: dict[str, Any],
) -> bool:
    payload = {
        "schema": _COARSE_ANCHOR_CACHE_SCHEMA,
        "protocol": protocol,
        "alignments": [
            {
                "scale": float(item.scale),
                "rotation": item.rotation.tolist(),
                "translation": item.translation.tolist(),
            }
            for item in alignments
        ],
        "diagnostics": diagnostics,
    }
    try:
        _atomic_write_json(
            _coarse_anchor_cache_path(root, protocol),
            {
                "payload": payload,
            },
        )
    except OSError:
        return False
    return True


class _ArtifactValidationReceipt:
    """Metadata-gated cache for previously identity- and finite-validated fields."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / ".validation_receipt.json"
        complete = json.loads((root / "forward.complete.json").read_text())
        self.manifest_id = str(complete.get("manifest_id", ""))
        self._lock = threading.Lock()
        self._dirty = False
        self.hits = 0
        self.misses = 0
        self.entries: dict[str, dict[str, Any]] = {}
        try:
            payload = json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            payload = {}
        if (
            payload.get("schema") == _VALIDATION_RECEIPT_SCHEMA
            and payload.get("manifest_id") == self.manifest_id
            and isinstance(payload.get("entries"), dict)
        ):
            self.entries = dict(payload["entries"])

    @staticmethod
    def _identity(path: Path, expected_file_identity: str) -> dict[str, Any]:
        stat = path.stat()
        return {
            "file_identity": expected_file_identity,
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "ctime_ns": int(stat.st_ctime_ns),
        }

    def is_valid(self, path: Path, expected_file_identity: str) -> bool:
        relative = str(path.relative_to(self.root))
        try:
            identity = self._identity(path, expected_file_identity)
        except OSError:
            with self._lock:
                self.misses += 1
            return False
        with self._lock:
            valid = self.entries.get(relative) == identity
            if valid:
                self.hits += 1
            else:
                self.misses += 1
            return valid

    def record(self, path: Path, expected_file_identity: str) -> None:
        relative = str(path.relative_to(self.root))
        identity = self._identity(path, expected_file_identity)
        with self._lock:
            if self.entries.get(relative) != identity:
                self.entries[relative] = identity
                self._dirty = True

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            payload = {
                "schema": _VALIDATION_RECEIPT_SCHEMA,
                "manifest_id": self.manifest_id,
                "entries": dict(self.entries),
            }
            self._dirty = False
        try:
            _atomic_write_json(self.path, payload)
        except OSError:
            # The forward artifact may be mounted read-only. Validation already
            # succeeded; inability to persist an execution cache is non-fatal.
            return


def _write_validation_receipt_from_forward_manifest(
    root: Path,
    manifest: dict[str, Any],
    *,
    manifest_id: str,
) -> None:
    """Publish metadata identities for fields validated during forward export."""

    entries: dict[str, dict[str, Any]] = {}
    for section in ("chunks", "loops"):
        for artifact in manifest.get(section, []):
            for field in artifact.get("fields", []):
                relative = str(field["path"])
                path = root / relative
                entries[relative] = _ArtifactValidationReceipt._identity(
                    path,
                    str(field["file_identity"]),
                )
    _atomic_write_json(
        root / ".validation_receipt.json",
        {
            "schema": _VALIDATION_RECEIPT_SCHEMA,
            "manifest_id": manifest_id,
            "entries": entries,
        },
    )


def _read_npy_header(handle: Any, path: Path) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    version = np_format.read_magic(handle)
    if version == (1, 0):
        shape, fortran_order, dtype = np_format.read_array_header_1_0(handle)
    elif version == (2, 0):
        shape, fortran_order, dtype = np_format.read_array_header_2_0(handle)
    else:
        raise ValueError(f"unsupported npy format version {version}: {path}")
    return tuple(int(item) for item in shape), bool(fortran_order), np.dtype(dtype)


def _npy_payload_nbytes(shape: Sequence[int], dtype: np.dtype[Any]) -> int:
    elements = 1
    for dim in shape:
        if dim < 0:
            raise ValueError(f"npy shape contains negative dimension: {tuple(shape)}")
        elements *= int(dim)
    return elements * int(dtype.itemsize)


def _npy_scan_block_size(dtype: np.dtype[Any]) -> int:
    itemsize = int(dtype.itemsize)
    if itemsize <= 0:
        raise TypeError("npy dtype must have a positive itemsize")
    return max(itemsize, (_NPY_SCAN_BLOCK_BYTES // itemsize) * itemsize)


def _check_npy_numeric_or_bool(name: str, dtype: np.dtype[Any]) -> None:
    if dtype == np.dtype(np.bool_):
        return
    if not np.issubdtype(dtype, np.number):
        raise TypeError(f"{name} must contain numeric finite values")


def _load_v2_npy_memmap(
    path: Path,
    field: dict[str, Any],
    *,
    name: str,
    check_finite: bool,
    validation_receipt: _ArtifactValidationReceipt | None = None,
) -> np.memmap:
    if check_finite and validation_receipt is not None and validation_receipt.is_valid(
        path, str(field["file_identity"])
    ):
        with path.open("rb") as raw:
            shape, fortran_order, dtype = _read_npy_header(raw, path)
            data_offset = raw.tell()
        if dtype.str != field["dtype"] or list(shape) != field["shape"]:
            raise ValueError(f"forward artifact header mismatch: {path}")
        _check_npy_numeric_or_bool(name, dtype)
        return np.memmap(
            path,
            dtype=dtype,
            mode="r",
            offset=data_offset,
            shape=shape,
            order="F" if fortran_order else "C",
        )
    has_extra_payload = False
    with path.open("rb") as raw:
        shape, fortran_order, dtype = _read_npy_header(raw, path)
        data_offset = raw.tell()
        if dtype.str != field["dtype"]:
            raise ValueError(f"forward artifact dtype mismatch: {path}")
        if list(shape) != field["shape"]:
            raise ValueError(f"forward artifact shape mismatch: {path}")
        _check_npy_numeric_or_bool(name, dtype)

        remaining = _npy_payload_nbytes(shape, dtype)
        block_size = _npy_scan_block_size(dtype)
        scan_finite = check_finite and dtype.kind in {"f", "c"}
        while remaining:
            data = raw.read(min(block_size, remaining))
            if not data:
                raise ValueError(f"forward artifact payload is truncated: {path}")
            remaining -= len(data)
            if scan_finite and not bool(np.isfinite(np.frombuffer(data, dtype=dtype)).all()):
                raise ValueError(f"{name} contains non-finite values")

        tail = raw.read(1)
        if tail:
            has_extra_payload = True
            for _ in iter(lambda: raw.read(_NPY_SCAN_BLOCK_BYTES), b""):
                pass

        if _file_identity_file(path) != field["file_identity"]:
            raise ValueError(f"forward artifact identity mismatch: {path}")
        if has_extra_payload:
            raise ValueError(f"forward artifact payload size mismatch: {path}")
        if check_finite and validation_receipt is not None:
            validation_receipt.record(path, str(field["file_identity"]))
        return np.memmap(
            raw,
            dtype=dtype,
            mode="r",
            offset=data_offset,
            shape=shape,
            order="F" if fortran_order else "C",
        )


def _atomic_write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    compact: bool = False,
    payload_is_json_safe: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        safe_payload = payload if payload_is_json_safe else _json_safe(payload)
        if _orjson is not None:
            options = _orjson.OPT_SORT_KEYS
            if not compact:
                options |= _orjson.OPT_INDENT_2
            encoded = _orjson.dumps(
                safe_payload,
                option=options,
            ) + b"\n"
        else:
            kwargs: dict[str, Any] = {
                "sort_keys": True,
                "allow_nan": False,
            }
            if compact:
                kwargs["separators"] = (",", ":")
            else:
                kwargs["indent"] = 2
            encoded = (json.dumps(safe_payload, **kwargs) + "\n").encode("utf-8")
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


_PREDICTION_FIELD_ORDER = (
    "world_points",
    "world_points_conf",
    "extrinsic_c2w",
    "intrinsic",
    "depth",
    "depth_conf",
    "images",
    "mask",
)

_BASIS_CHUNK_REQUIRED_FIELDS = (
    "world_points",
    "world_points_conf",
    "extrinsic_c2w",
)
_LOOP_JOINT_REQUIRED_FIELDS = ("world_points", "world_points_conf")
_TRAJECTORY_EXPORT_REQUIRED_FIELDS = ("extrinsic_c2w", "intrinsic")


def _forward_artifact_fields(profile: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if profile == "full":
        return _PREDICTION_FIELD_ORDER, _PREDICTION_FIELD_ORDER
    if profile in _HIERARCHICAL_TRAJECTORY_FIELD_PROFILES:
        return (
            tuple(dict.fromkeys((*_BASIS_CHUNK_REQUIRED_FIELDS, *_TRAJECTORY_EXPORT_REQUIRED_FIELDS))),
            _LOOP_JOINT_REQUIRED_FIELDS,
        )
    raise ValueError(f"unsupported forward artifact field profile: {profile}")


def _forward_artifact_dtype_overrides(profile: str) -> dict[str, np.dtype[Any]]:
    if profile == "hierarchical_trajectory_conf_fp16_v1":
        return {"world_points_conf": np.dtype(np.float16)}
    if profile == "hierarchical_trajectory_world_q16_conf_fp16_v1":
        return {
            "world_points": np.dtype(np.int16),
            "world_points_conf": np.dtype(np.float16),
        }
    if profile == "full" or profile in _HIERARCHICAL_TRAJECTORY_FIELD_PROFILES:
        return {}
    raise ValueError(f"unsupported forward artifact field profile: {profile}")


def _artifact_entries_nbytes(entries: Sequence[dict[str, Any]]) -> int:
    return sum(
        int(field.get("nbytes", 0))
        for entry in entries
        for field in entry.get("fields", [])
    )


def _validate_direct_artifact_entries(
    entries: Sequence[dict[str, Any]],
    *,
    artifact_root: Path,
    relative_paths: Sequence[str],
    field_names: Sequence[str],
) -> None:
    if len(entries) != len(relative_paths):
        raise RuntimeError(
            "direct artifact publication returned "
            f"{len(entries)} entries for {len(relative_paths)} requests"
        )
    expected_fields = list(field_names)
    for index, (entry, relative_path) in enumerate(zip(entries, relative_paths)):
        if entry.get("path") != relative_path:
            raise RuntimeError(
                "direct artifact publication path mismatch at index "
                f"{index}: expected {relative_path!r}, got {entry.get('path')!r}"
            )
        fields = entry.get("fields")
        if not isinstance(fields, list):
            raise RuntimeError(
                f"direct artifact publication entry {index} has no field list"
            )
        names = [field.get("name") for field in fields]
        if names != expected_fields:
            raise RuntimeError(
                "direct artifact publication field mismatch at index "
                f"{index}: expected {expected_fields}, got {names}"
            )
        for field in fields:
            name = str(field["name"])
            expected_path = f"{relative_path}/{name}.npy"
            if field.get("path") != expected_path:
                raise RuntimeError(
                    "direct artifact publication path mismatch for field at "
                    f"index {index}: expected {expected_path!r}, "
                    f"got {field.get('path')!r}"
                )
            if not isinstance(field.get("file_identity"), str) or not field["file_identity"]:
                raise RuntimeError(
                    f"direct artifact publication has invalid file identity at index {index}"
                )
            if int(field.get("nbytes", 0)) <= 0:
                raise RuntimeError(
                    f"direct artifact publication has invalid size at index {index}"
                )
            path = artifact_root / expected_path
            if not path.is_file():
                raise RuntimeError(
                    f"direct artifact publication produced no file: {path}"
                )
            if path.stat().st_size != int(field["nbytes"]):
                raise RuntimeError(
                    f"direct artifact publication size mismatch: {path}"
                )


@dataclass(frozen=True)
class _LoadedPredictionFields:
    arrays: dict[str, np.ndarray]
    finite_validated_fields: frozenset[str]


def _zero_intrinsics(length: int) -> np.ndarray:
    return np.zeros((length, 3, 3), dtype=np.float32)


def _identity_extrinsics(length: int) -> np.ndarray:
    return np.repeat(np.eye(4, dtype=np.float32)[None], length, axis=0)


def _prediction_from_consumed_fields(
    fields: dict[str, np.ndarray],
    *,
    required_names: Sequence[str],
    validated_names: Sequence[str] | frozenset[str],
) -> ChunkPrediction:
    missing = set(required_names).difference(fields)
    if missing:
        raise ValueError(f"forward artifact fields are missing: {sorted(missing)}")
    if "world_points" in fields:
        length = int(np.asarray(fields["world_points"]).shape[0])
    elif "extrinsic_c2w" in fields:
        length = int(np.asarray(fields["extrinsic_c2w"]).shape[0])
    else:
        raise ValueError("consumed fields cannot determine prediction length")
    arrays = {
        "world_points": fields.get(
            "world_points", np.zeros((length, 1, 1, 3), dtype=np.float32)
        ),
        "world_points_conf": fields.get(
            "world_points_conf", np.zeros((length, 1, 1), dtype=np.float32)
        ),
        "extrinsic_c2w": fields.get("extrinsic_c2w", _identity_extrinsics(length)),
        "intrinsic": fields.get("intrinsic", _zero_intrinsics(length)),
    }
    validated = frozenset(validated_names)
    unknown = validated.difference(fields)
    if unknown:
        raise ValueError(
            f"finite-validated fields were not loaded: {sorted(unknown)}"
        )
    return ChunkPrediction._from_finite_validated_arrays(
        _token=_FINITE_VALIDATED_TOKEN,
        _validated_fields=validated,
        **arrays,
    )


def _encode_world_points_q16(array: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(array)
    if source.dtype != np.float32 or source.ndim < 2 or source.shape[-1] != 3:
        raise ValueError(
            "world_points q16 encoding requires float32 shape (..., 3), "
            f"got dtype={source.dtype}, shape={source.shape}"
        )
    if not bool(np.isfinite(source).all()):
        raise ValueError("world_points q16 encoding requires finite values")
    reduction_axes = tuple(range(source.ndim - 1))
    minimum = np.asarray(np.min(source, axis=reduction_axes), dtype=np.float64)
    maximum = np.asarray(np.max(source, axis=reduction_axes), dtype=np.float64)
    center = (minimum + maximum) * 0.5
    span = maximum - minimum
    scale = np.where(
        span > 0.0,
        span / (2.0 * _WORLD_POINTS_Q16_MAX),
        1.0,
    )
    quantized = np.rint((source.astype(np.float64) - center) / scale)
    np.clip(
        quantized,
        -_WORLD_POINTS_Q16_MAX,
        _WORLD_POINTS_Q16_MAX,
        out=quantized,
    )
    return quantized.astype(np.int16), {
        "schema": _WORLD_POINTS_Q16_ENCODING_SCHEMA,
        "axis": "last_xyz",
        "center": center.tolist(),
        "scale": scale.tolist(),
        "qmin": -_WORLD_POINTS_Q16_MAX,
        "qmax": _WORLD_POINTS_Q16_MAX,
        "rounding": "numpy_rint_ties_to_even",
        "reconstruction_dtype": np.dtype(np.float32).str,
    }


def _decode_world_points_q16(
    quantized: np.ndarray,
    encoding: Mapping[str, Any],
) -> np.ndarray:
    values = np.asarray(quantized)
    if (
        encoding.get("schema") != _WORLD_POINTS_Q16_ENCODING_SCHEMA
        or encoding.get("axis") != "last_xyz"
        or encoding.get("qmin") != -_WORLD_POINTS_Q16_MAX
        or encoding.get("qmax") != _WORLD_POINTS_Q16_MAX
        or encoding.get("rounding") != "numpy_rint_ties_to_even"
        or encoding.get("reconstruction_dtype") != np.dtype(np.float32).str
    ):
        raise ValueError("unsupported world_points q16 encoding metadata")
    if values.dtype != np.int16 or values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError(
            "encoded world_points must have int16 shape (..., 3), "
            f"got dtype={values.dtype}, shape={values.shape}"
        )
    center = np.asarray(encoding.get("center"), dtype=np.float64)
    scale = np.asarray(encoding.get("scale"), dtype=np.float64)
    if (
        center.shape != (3,)
        or scale.shape != (3,)
        or not bool(np.isfinite(center).all())
        or not bool(np.isfinite(scale).all())
        or not bool((scale > 0.0).all())
    ):
        raise ValueError("invalid world_points q16 center/scale metadata")
    reconstructed = (values.astype(np.float64) * scale + center).astype(np.float32)
    if not bool(np.isfinite(reconstructed).all()):
        raise ValueError("world_points q16 reconstruction produced non-finite values")
    return reconstructed


def _write_prediction_artifact(
    root: Path,
    relative_path: str,
    prediction: ChunkPrediction,
    *,
    version: int,
    field_names: Sequence[str] | None = None,
    field_dtype_overrides: Mapping[str, np.dtype[Any] | str] | None = None,
) -> dict[str, Any]:
    arrays = prediction.to_numpy_dict()
    selected_names = tuple(field_names or _PREDICTION_FIELD_ORDER)
    dtype_overrides = {
        str(name): np.dtype(dtype)
        for name, dtype in (field_dtype_overrides or {}).items()
    }
    unknown_overrides = set(dtype_overrides).difference(selected_names)
    if unknown_overrides:
        raise ValueError(
            "artifact dtype overrides target unselected fields: "
            f"{sorted(unknown_overrides)}"
        )
    full_profile = selected_names == _PREDICTION_FIELD_ORDER
    missing = set(selected_names).difference(arrays)
    if missing and not full_profile:
        raise ValueError(
            f"prediction is missing requested artifact fields: {sorted(missing)}"
        )
    if version == 2:
        directory = root / relative_path
        directory.mkdir(parents=True, exist_ok=True)
        fields = []
        selected = set(selected_names)
        for name in _PREDICTION_FIELD_ORDER:
            if name not in arrays or name not in selected:
                continue
            path = directory / f"{name}.npy"
            encoding = None
            if name == "world_points" and dtype_overrides.get(name) == np.dtype(
                np.int16
            ):
                published, encoding = _encode_world_points_q16(arrays[name])
            else:
                with np.errstate(over="ignore", invalid="ignore"):
                    published = np.asarray(
                        arrays[name], dtype=dtype_overrides.get(name)
                    )
            if name in dtype_overrides and not bool(np.isfinite(published).all()):
                raise ValueError(
                    f"artifact dtype conversion produced non-finite values for {name}"
                )
            _atomic_write_npy(path, published)
            fields.append(
                {
                    "name": name,
                    "path": f"{relative_path}/{name}.npy",
                    "file_identity": _file_identity_file(path),
                    "dtype": published.dtype.str,
                    "shape": list(published.shape),
                    "nbytes": int(path.stat().st_size),
                    **({"encoding": encoding} if encoding is not None else {}),
                }
            )
        _fsync_directory(directory)
        return {"path": relative_path, "fields": fields}
    if version != 1:
        raise ValueError(f"unsupported forward artifact version: {version}")
    if selected_names != _PREDICTION_FIELD_ORDER:
        raise ValueError("field-subset publication requires forward artifact v2")
    relative_path = f"{relative_path}.npz"
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return {"path": relative_path, "file_identity": _file_identity_file(path)}


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    os.close(fd)
    try:
        np.save(temporary, array, allow_pickle=False)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with source.open("rb") as source_handle, os.fdopen(fd, "wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=8 * 1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(temporary, destination)
        shutil.copystat(source, destination)
        with destination.open("rb") as handle:
            os.fsync(handle.fileno())
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(fd)
    try:
        np.savez(temporary, **arrays)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_prediction_artifact(
    root: Path, entry: dict[str, Any], *, expected_index: int
) -> ChunkPrediction:
    if int(entry.get("index", -1)) != expected_index:
        raise ValueError(
            f"forward artifact order mismatch: expected index {expected_index}, "
            f"got {entry.get('index')}"
        )
    if "fields" in entry:
        fields = entry["fields"]
        names = [item.get("name") for item in fields]
        expected_names = [name for name in _PREDICTION_FIELD_ORDER if name in names]
        if names != expected_names or len(names) != len(set(names)):
            raise ValueError(
                f"forward artifact field order mismatch: expected {expected_names}, got {names}"
            )
        arrays = {}
        for field in fields:
            name = field["name"]
            path = root / field["path"]
            if not path.is_file():
                raise FileNotFoundError(f"forward artifact is missing: {path}")
            loaded = _load_v2_npy_memmap(
                path,
                field,
                name=name,
                check_finite=True,
            )
            if "encoding" in field:
                if name != "world_points":
                    raise ValueError(f"unsupported encoded forward artifact field: {name}")
                loaded = _decode_world_points_q16(loaded, field["encoding"])
            arrays[name] = loaded
        return ChunkPrediction._from_finite_validated_arrays(
            _token=_FINITE_VALIDATED_TOKEN,
            _validated_fields=arrays.keys(),
            **arrays,
        )

    path = root / entry["path"]
    if not path.is_file():
        raise FileNotFoundError(f"forward artifact is missing: {path}")
    actual = _file_identity_file(path)
    if actual != entry["file_identity"]:
        raise ValueError(f"forward artifact identity mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return ChunkPrediction(**{name: payload[name] for name in payload.files})


def _load_prediction_artifact_fields(
    root: Path,
    entry: dict[str, Any],
    *,
    expected_index: int,
    required_names: Sequence[str],
    validation_receipt: _ArtifactValidationReceipt | None = None,
) -> _LoadedPredictionFields:
    """Verify and materialize only the v2 fields consumed by this stage."""

    required = set(required_names)
    if "fields" not in entry:
        prediction = _load_prediction_artifact(root, entry, expected_index=expected_index)
        values = prediction.to_numpy_dict()
        missing = required.difference(values)
        if missing:
            raise ValueError(f"forward artifact fields are missing: {sorted(missing)}")
        return _LoadedPredictionFields(
            arrays={name: values[name] for name in required_names},
            finite_validated_fields=frozenset(),
        )

    if int(entry.get("index", -1)) != expected_index:
        raise ValueError(
            f"forward artifact order mismatch: expected index {expected_index}, "
            f"got {entry.get('index')}"
        )
    fields = entry["fields"]
    names = [item.get("name") for item in fields]
    expected_names = [name for name in _PREDICTION_FIELD_ORDER if name in names]
    if names != expected_names or len(names) != len(set(names)):
        raise ValueError(
            f"forward artifact field order mismatch: expected {expected_names}, got {names}"
        )
    missing = required.difference(names)
    if missing:
        raise ValueError(f"forward artifact fields are missing: {sorted(missing)}")

    arrays: dict[str, np.ndarray] = {}
    for field in fields:
        name = field["name"]
        if name not in required:
            continue
        path = root / field["path"]
        if not path.is_file():
            raise FileNotFoundError(f"forward artifact is missing: {path}")
        loaded = _load_v2_npy_memmap(
            path,
            field,
            name=name,
            check_finite=True,
            validation_receipt=validation_receipt,
        )
        if "encoding" in field:
            if name != "world_points":
                raise ValueError(f"unsupported encoded forward artifact field: {name}")
            loaded = _decode_world_points_q16(loaded, field["encoding"])
        arrays[name] = loaded
    return _LoadedPredictionFields(
        arrays={name: arrays[name] for name in required_names},
        finite_validated_fields=frozenset(required_names),
    )


def _load_forward_manifest(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    complete_path = root / "forward.complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(f"forward stage is incomplete: {complete_path} is missing")
    complete = json.loads(complete_path.read_text())
    complete_schema = complete.get("schema")
    if complete_schema not in {"hip3r_forward_complete_v1", "hip3r_forward_complete_v2"}:
        raise ValueError("unsupported forward completion schema")
    expected_identity = _protocol_id(protocol)
    if complete.get("protocol_id") != expected_identity:
        raise ValueError(
            "forward protocol identity mismatch: "
            f"expected {expected_identity}, got {complete.get('protocol_id')}"
        )
    manifest_path = root / complete.get("manifest", "manifest.json")
    manifest_id = _file_identity_file(manifest_path)
    if manifest_id != complete.get("manifest_id"):
        raise ValueError("forward manifest identity mismatch")
    manifest = json.loads(manifest_path.read_text())
    expected_manifest_schema = complete_schema.replace("_complete_", "_artifacts_")
    if manifest.get("schema") != expected_manifest_schema:
        raise ValueError("unsupported forward artifact schema")
    if manifest.get("protocol") != _json_safe(protocol):
        raise ValueError("forward manifest protocol metadata mismatch")
    if manifest.get("protocol_id") != expected_identity:
        raise ValueError("forward manifest protocol identity mismatch")
    if len(manifest.get("chunks", [])) != int(complete.get("num_chunks", -1)):
        raise ValueError("forward chunk count mismatch")
    if len(manifest.get("loops", [])) != int(complete.get("num_loop_constraints", -1)):
        raise ValueError("forward loop count mismatch")
    if "explicit_ranges" in protocol:
        range_summary = dict(protocol.get("explicit_ranges_summary") or {})
        min_overlap = int(protocol.get("explicit_ranges_min_overlap", 0))
        max_chunk_size = int(protocol.get("max_chunk_size", protocol.get("chunk_size", 0)))
        if max_chunk_size <= 0:
            raise ValueError("explicit range protocol is missing max_chunk_size")
        ranges = validate_explicit_chunk_ranges(
            len(manifest.get("images", [])),
            protocol["explicit_ranges"],
            max_chunk_size=max_chunk_size,
            min_overlap=min_overlap,
            expected_count=range_summary.get("num_chunks"),
            expected_total_processed_frames=range_summary.get(
                "total_processed_frames"
            ),
        )
        manifest_ranges = validate_explicit_chunk_ranges(
            len(manifest.get("images", [])),
            manifest.get("chunks", []),
            max_chunk_size=max_chunk_size,
            min_overlap=min_overlap,
            expected_count=len(ranges),
            expected_total_processed_frames=sum(item.size for item in ranges),
        )
        if chunk_ranges_records(manifest_ranges) != chunk_ranges_records(ranges):
            raise ValueError("forward explicit ranges do not match manifest chunks")
        if protocol.get("explicit_ranges_identity") != chunk_ranges_identity(ranges):
            raise ValueError("forward explicit range identity mismatch")
        observed_summary = summarize_chunk_ranges(
            ranges, num_frames=len(manifest.get("images", []))
        )
        for key in (
            "num_frames",
            "num_chunks",
            "total_processed_frames",
            "max_chunk_size",
            "min_adjacent_overlap",
            "full_coverage",
        ):
            if key in range_summary and range_summary[key] != observed_summary[key]:
                raise ValueError(f"forward explicit range summary mismatch for {key}")
    field_profile = protocol.get("forward_artifact_field_profile")
    if field_profile == "hierarchical_trajectory_world_q16_conf_fp16_v1":
        publication = manifest.get("diagnostics", {}).get(
            "forward_artifact_publication", {}
        )
        expected_overrides = {"world_points": "<i2", "world_points_conf": "<f2"}
        if (
            publication.get("field_profile")
            != "hierarchical_trajectory_world_q16_conf_fp16_v1"
            or publication.get("field_dtype_overrides") != expected_overrides
        ):
            raise ValueError("world q16 forward artifact activation diagnostic mismatch")
        for section in ("chunks", "loops"):
            for entry in manifest[section]:
                fields = {field.get("name"): field for field in entry.get("fields", [])}
                world = fields.get("world_points", {})
                confidence = fields.get("world_points_conf", {})
                if (
                    world.get("dtype") != "<i2"
                    or world.get("encoding", {}).get("schema")
                    != _WORLD_POINTS_Q16_ENCODING_SCHEMA
                    or confidence.get("dtype") != "<f2"
                ):
                    raise ValueError("world q16 forward artifact field activation mismatch")
                if any(
                    "encoding" in field
                    for name, field in fields.items()
                    if name != "world_points"
                ):
                    raise ValueError("world q16 encoding is only valid for world_points")
    else:
        for section in ("chunks", "loops"):
            for entry in manifest[section]:
                if any("encoding" in field for field in entry.get("fields", [])):
                    raise ValueError(
                        "encoded forward artifact field requires the world q16 profile"
                    )
    manifest["_validated_manifest_id"] = manifest_id
    return manifest


def _load_trajectory_replay_manifest(
    root: Path, protocol: dict[str, Any]
) -> dict[str, Any]:
    complete_path = root / "trajectory_replay.complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(
            f"trajectory replay is incomplete: {complete_path} is missing"
        )
    complete = json.loads(complete_path.read_text())
    if complete.get("schema") != _TRAJECTORY_REPLAY_COMPLETE_SCHEMA:
        raise ValueError("unsupported trajectory replay completion schema")
    expected_identity = _protocol_id(protocol)
    if complete.get("protocol_id") != expected_identity:
        raise ValueError(
            "trajectory replay protocol identity mismatch: "
            f"expected {expected_identity}, got {complete.get('protocol_id')}"
        )
    if complete.get("required_fields") != list(_TRAJECTORY_EXPORT_REQUIRED_FIELDS):
        raise ValueError("trajectory replay required fields mismatch")
    manifest_path = root / complete.get("manifest", "manifest.json")
    manifest_id = _file_identity_file(manifest_path)
    if (
        manifest_id != complete.get("manifest_id")
        or manifest_id != complete.get("source_forward_manifest_id")
    ):
        raise ValueError("trajectory replay manifest identity mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "hip3r_forward_artifacts_v2":
        raise ValueError("unsupported trajectory replay source manifest schema")
    if manifest.get("protocol") != _json_safe(protocol):
        raise ValueError("trajectory replay source protocol metadata mismatch")
    if manifest.get("protocol_id") != expected_identity:
        raise ValueError("trajectory replay source protocol identity mismatch")
    if len(manifest.get("images", [])) != int(complete.get("num_frames", -1)):
        raise ValueError("trajectory replay frame count mismatch")
    if len(manifest.get("chunks", [])) != int(complete.get("num_chunks", -1)):
        raise ValueError("trajectory replay chunk count mismatch")
    if len(manifest.get("loops", [])) != int(
        complete.get("num_loop_constraints", -1)
    ):
        raise ValueError("trajectory replay loop count mismatch")
    expected_field_count = len(manifest["chunks"]) * len(
        _TRAJECTORY_EXPORT_REQUIRED_FIELDS
    )
    if int(complete.get("field_count", -1)) != expected_field_count:
        raise ValueError("trajectory replay field count mismatch")
    expected_field_bytes = 0
    for index, entry in enumerate(manifest["chunks"]):
        if int(entry.get("index", -1)) != index:
            raise ValueError("trajectory replay chunk order mismatch")
        names = {field.get("name") for field in entry.get("fields", [])}
        missing = set(_TRAJECTORY_EXPORT_REQUIRED_FIELDS).difference(names)
        if missing:
            raise ValueError(
                f"trajectory replay source fields are missing: {sorted(missing)}"
            )
        expected_field_bytes += sum(
            int(field["nbytes"])
            for field in entry["fields"]
            if field.get("name") in _TRAJECTORY_EXPORT_REQUIRED_FIELDS
        )
    if int(complete.get("field_bytes", -1)) != expected_field_bytes:
        raise ValueError("trajectory replay field byte count mismatch")
    manifest["_validated_manifest_id"] = manifest_id
    return manifest


def _load_forward_manifest_for_alpha(
    root: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    full_marker = root / "forward.complete.json"
    replay_marker = root / "trajectory_replay.complete.json"
    if full_marker.is_file() and replay_marker.is_file():
        raise ValueError("ambiguous forward artifact contract for alpha export")
    if replay_marker.is_file():
        return _load_trajectory_replay_manifest(root, protocol), "trajectory_replay_v1"
    manifest = _load_forward_manifest(root, protocol)
    version = str(manifest["schema"]).rsplit("_v", 1)[-1]
    return manifest, f"full_forward_v{version}"


def _basis_replay_cache_path(root: Path, manifest_id: str) -> Path:
    return root / ".basis_replay_cache" / f"{manifest_id}.json"


def _compact_basis_replay_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    basis = manifest["basis"]
    return {
        "schema": manifest["schema"],
        "forward_protocol_id": manifest["forward_protocol_id"],
        "basis_protocol": manifest["basis_protocol"],
        "basis_protocol_id": manifest["basis_protocol_id"],
        "chunk_ranges": manifest["chunk_ranges"],
        "edges": manifest["edges"],
        "diagnostics": manifest.get("diagnostics") or {},
        "basis": {
            "coarse_count": len(basis.get("coarse_alignments", [])),
            "frame_count": len(basis.get("frame_poses", [])),
            "apply_point_corrections": bool(basis.get("apply_point_corrections", True)),
            "diagnostics": basis.get("diagnostics") or {},
        },
    }


def _publish_basis_replay_cache(
    root: Path, manifest: dict[str, Any], *, manifest_id: str
) -> None:
    payload = _compact_basis_replay_manifest(manifest)
    try:
        _atomic_write_json(
            _basis_replay_cache_path(root, manifest_id),
            {
                "schema": _BASIS_REPLAY_CACHE_SCHEMA,
                "manifest_id": manifest_id,
                "payload": payload,
            },
            compact=True,
            payload_is_json_safe=True,
        )
    except OSError:
        return


def _try_load_basis_replay_cache(
    root: Path,
    *,
    manifest_id: str,
    forward_protocol_id: str,
    basis_protocol: dict[str, Any],
    basis_protocol_id: str,
    num_chunks: int,
    num_edges: int,
) -> dict[str, Any] | None:
    try:
        envelope = json.loads(_basis_replay_cache_path(root, manifest_id).read_text())
        payload = envelope["payload"]
        if (
            envelope.get("schema") != _BASIS_REPLAY_CACHE_SCHEMA
            or envelope.get("manifest_id") != manifest_id
            or payload.get("schema") != "hip3r_hierarchical_basis_v1"
            or payload.get("forward_protocol_id") != forward_protocol_id
            or payload.get("basis_protocol") != _json_safe(basis_protocol)
            or payload.get("basis_protocol_id") != basis_protocol_id
            or len(payload.get("chunk_ranges", [])) != num_chunks
            or len(payload.get("edges", [])) != num_edges
        ):
            return None
        payload["_validated_manifest_id"] = manifest_id
        payload["_replay_compact"] = True
        return payload
    except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError, TypeError):
        return None


def _load_basis_manifest(
    root: Path,
    *,
    forward_protocol_id: str,
    basis_protocol: dict[str, Any],
    basis_protocol_id: str,
    allow_replay_cache: bool = True,
) -> dict[str, Any]:
    complete_path = root / "basis.complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError(f"basis stage is incomplete: {complete_path} is missing")
    complete = json.loads(complete_path.read_text())
    complete_schema = complete.get("schema")
    if complete_schema not in {
        "hip3r_hierarchical_basis_complete_v1",
        "hip3r_pose_graph_basis_complete_v1",
    }:
        raise ValueError("unsupported basis completion schema")
    if complete.get("forward_protocol_id") != forward_protocol_id:
        raise ValueError(
            "basis forward protocol identity mismatch: "
            f"expected {forward_protocol_id}, got {complete.get('forward_protocol_id')}"
        )
    if complete.get("basis_protocol_id") != basis_protocol_id:
        raise ValueError(
            "basis protocol identity mismatch: "
            f"expected {basis_protocol_id}, got {complete.get('basis_protocol_id')}"
        )
    manifest_path = root / complete.get("manifest", "basis_manifest.json")
    manifest_id = _file_identity_file(manifest_path)
    if manifest_id != complete.get("manifest_id"):
        raise ValueError("basis manifest identity mismatch")
    if allow_replay_cache and complete_schema == "hip3r_hierarchical_basis_complete_v1":
        replay = _try_load_basis_replay_cache(
            root,
            manifest_id=manifest_id,
            forward_protocol_id=forward_protocol_id,
            basis_protocol=basis_protocol,
            basis_protocol_id=basis_protocol_id,
            num_chunks=int(complete.get("num_chunks", -1)),
            num_edges=int(complete.get("num_edges", -1)),
        )
        if replay is not None:
            return replay
    manifest = json.loads(manifest_path.read_text())
    expected_manifest_schema = {
        "hip3r_hierarchical_basis_complete_v1": "hip3r_hierarchical_basis_v1",
        "hip3r_pose_graph_basis_complete_v1": "hip3r_pose_graph_basis_v1",
    }[complete_schema]
    if manifest.get("schema") != expected_manifest_schema:
        raise ValueError("unsupported basis manifest schema")
    if manifest.get("forward_protocol_id") != forward_protocol_id:
        raise ValueError("basis manifest forward protocol identity mismatch")
    if manifest.get("basis_protocol") != _json_safe(basis_protocol):
        raise ValueError("basis manifest protocol metadata mismatch")
    if manifest.get("basis_protocol_id") != basis_protocol_id:
        raise ValueError("basis manifest protocol identity mismatch")
    if len(manifest.get("chunk_ranges", [])) != int(complete.get("num_chunks", -1)):
        raise ValueError("basis chunk count mismatch")
    if len(manifest.get("edges", [])) != int(complete.get("num_edges", -1)):
        raise ValueError("basis edge count mismatch")
    manifest["_validated_manifest_id"] = manifest_id
    if complete_schema == "hip3r_hierarchical_basis_complete_v1":
        _publish_basis_replay_cache(root, manifest, manifest_id=manifest_id)
    if complete_schema == "hip3r_pose_graph_basis_complete_v1":
        result_files = manifest.get("result_files")
        if not isinstance(result_files, dict) or not result_files:
            raise ValueError("pose-graph basis result files are missing")
        for relative, expected_file_identity in result_files.items():
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("pose-graph basis result path escapes basis directory")
            path = root / relative
            if _file_identity_file(path) != expected_file_identity:
                raise ValueError(f"pose-graph basis result identity mismatch: {path}")
    return manifest


__all__ = [
    "BasisStageResult",
    "ChunkRange",
    "DEFAULT_HIERARCHICAL_COARSE_MAX_NFEV",
    "ReconstructionRunner",
    "ReconstructionConfig",
    "ForwardStageResult",
    "LongReconstructionResult",
    "LoopPair",
    "chunk_ranges_identity",
    "chunk_ranges_records",
    "generate_chunk_ranges",
    "load_loop_pairs",
    "save_long_reconstruction",
    "select_loop_edges",
    "summarize_chunk_ranges",
    "validate_explicit_chunk_ranges",
]
