"""Joint fine-to-coarse pose graph for long VGGT-Omega reconstruction.

The regular hip3r post-processor optimizes one Sim3 node per local chunk.  This
module keeps that graph as the coarse layer and adds one shared Sim3 pose node
per global frame.  The fine layer is constrained by all local frame-to-frame
measurements emitted by the local VGGT-Omega forward, while every occurrence
of a frame in an overlapping chunk is tied to the same global frame node.

The implementation deliberately keeps the two coordinate systems explicit:

* ``A_c`` maps chunk ``c`` coordinates to the global map;
* ``P_{c,i}`` maps camera ``i`` to chunk ``c`` coordinates;
* ``F_g`` maps global frame ``g`` camera coordinates to the global map.

The consistency prior is ``F_g ~= A_c compose P_{c,i}``.  Consequently the
same optimized frame pose can be used to transform both the camera trajectory
and the corresponding point map, instead of correcting the trajectory only.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace as _dataclass_replace
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np


def _scipy_rotation_types() -> tuple[Any, Any]:
    from scipy.spatial.transform import Rotation as scipy_rotation
    from scipy.spatial.transform import Slerp as scipy_slerp

    return scipy_rotation, scipy_slerp


class _LazyRotation:
    @staticmethod
    def from_matrix(*args: Any, **kwargs: Any) -> Any:
        return _scipy_rotation_types()[0].from_matrix(*args, **kwargs)

    @staticmethod
    def from_rotvec(*args: Any, **kwargs: Any) -> Any:
        return _scipy_rotation_types()[0].from_rotvec(*args, **kwargs)


Rotation = _LazyRotation


def Slerp(*args: Any, **kwargs: Any) -> Any:
    return _scipy_rotation_types()[1](*args, **kwargs)


_FACTORIZED_V4_KEYFRAME_CONTINUATION_STRIDE_MULTIPLIERS = (4, 2, 1)
_FACTORIZED_V4_KEYFRAME_CONTINUATION_BUDGET_FRACTIONS = (0.15, 0.25, 0.60)
_FACTORIZED_V4_KEYFRAME_CONTINUATION_V2_STRIDE_MULTIPLIERS = (2, 1)
_FACTORIZED_V4_KEYFRAME_CONTINUATION_V2_BUDGET_FRACTIONS = (0.125, 0.875)
_FACTORIZED_V4_KEYFRAME_CONTINUATION_V2_ALPHA_GRID = (1.0, 0.5, 0.25, 0.125, 0.0)

try:
    import numexpr as _numexpr
except ImportError:  # pragma: no cover - minimal environments use NumPy fallback.
    _numexpr = None

_NUMEXPR_EXACT_THREADS = 0
if _numexpr is not None:
    _NUMEXPR_EXACT_THREADS = min(4, int(_numexpr.get_num_threads()))
    _numexpr.set_num_threads(_NUMEXPR_EXACT_THREADS)

from long_reconstruction.postprocess import (  # noqa: E402
    C2W_FIELDS,
    POINT_FIELDS,
    Sim3Edge,
    _get_field,
    _replace_chunk_fields,
)
from long_reconstruction.sim3 import Sim3  # noqa: E402

if TYPE_CHECKING:
    from scipy.sparse import lil_matrix


_MAX_LOG_SCALE_DELTA = float(np.log(100.0))
_FACTORIZED_FRAME_CHART_MAX_CHUNK_HOPS = 4


class _ExactSparseOutlierHuberLoss:
    """Exact Huber arithmetic with one indexed gather for sparse outliers."""

    def __init__(self, num_residuals: int) -> None:
        self.rho = np.empty((3, int(num_residuals)), dtype=np.float64)
        self.calls = 0
        self.outliers_total = 0
        self.outliers_max = 0

    def __call__(self, z: np.ndarray) -> np.ndarray:
        values = np.asarray(z, dtype=np.float64)
        if values.shape != (self.rho.shape[1],):
            raise ValueError("Huber loss residual count changed during solve")
        outliers = np.flatnonzero(values > 1.0)
        num_outliers = len(outliers)
        self.calls += 1
        self.outliers_total += num_outliers
        self.outliers_max = max(self.outliers_max, num_outliers)
        self.rho[0] = values
        self.rho[1].fill(1.0)
        self.rho[2].fill(0.0)
        if num_outliers:
            selected = values[outliers]
            self.rho[0, outliers] = 2.0 * selected**0.5 - 1.0
            self.rho[1, outliers] = selected**-0.5
            self.rho[2, outliers] = -0.5 * selected**-1.5
        return self.rho


class _MixedPhysicalRobustLinearLoss:
    """Apply the configured robust loss only to physical residual rows."""

    def __init__(
        self,
        num_physical_residuals: int,
        num_total_residuals: int,
        *,
        physical_loss: str,
    ) -> None:
        if not 0 <= num_physical_residuals <= num_total_residuals:
            raise ValueError("physical residual count exceeds total rows")
        self.num_physical_residuals = int(num_physical_residuals)
        self.physical_loss = str(physical_loss)
        self.rho = np.empty((3, int(num_total_residuals)), dtype=np.float64)
        self.calls = 0

    def __call__(self, z: np.ndarray) -> np.ndarray:
        values = np.asarray(z, dtype=np.float64)
        if values.shape != (self.rho.shape[1],):
            raise ValueError("mixed loss residual count changed during solve")
        physical_stop = self.num_physical_residuals
        physical_terms = _scipy_compatible_loss_terms(
            values[:physical_stop], self.physical_loss
        )
        for row, terms in enumerate(physical_terms):
            self.rho[row, :physical_stop] = terms
        self.rho[0, physical_stop:] = values[physical_stop:]
        self.rho[1, physical_stop:] = 1.0
        self.rho[2, physical_stop:] = 0.0
        self.calls += 1
        return self.rho


def _scipy_compatible_loss_terms(
    squared_scaled_residuals: np.ndarray, loss: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rho, rho', and rho'' for SciPy's built-in robust losses."""

    squared = np.asarray(squared_scaled_residuals, dtype=np.float64)
    if loss == "linear":
        rho = squared
        rho_prime = np.ones_like(squared)
        rho_double_prime = np.zeros_like(squared)
    elif loss == "huber":
        rho = squared.copy()
        rho_prime = np.ones_like(squared)
        rho_double_prime = np.zeros_like(squared)
        outliers = squared > 1.0
        selected = squared[outliers]
        rho[outliers] = 2.0 * np.sqrt(selected) - 1.0
        rho_prime[outliers] = selected**-0.5
        rho_double_prime[outliers] = -0.5 * selected**-1.5
    elif loss == "soft_l1":
        one_plus = 1.0 + squared
        rho = 2.0 * (np.sqrt(one_plus) - 1.0)
        rho_prime = one_plus**-0.5
        rho_double_prime = -0.5 * one_plus**-1.5
    elif loss == "cauchy":
        one_plus = 1.0 + squared
        rho = np.log(one_plus)
        rho_prime = 1.0 / one_plus
        rho_double_prime = -(rho_prime**2)
    elif loss == "arctan":
        rho = np.arctan(squared)
        one_plus_squared = 1.0 + squared**2
        rho_prime = 1.0 / one_plus_squared
        rho_double_prime = -2.0 * squared / one_plus_squared**2
    else:
        raise ValueError(f"unsupported robust loss for Schur trial: {loss!r}")
    return rho, rho_prime, rho_double_prime


def _scipy_compatible_robust_cost(
    residuals: np.ndarray,
    *,
    loss: str,
    f_scale: float,
) -> float:
    """Evaluate SciPy's least-squares objective without rebuilding a Jacobian."""

    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    scale = float(f_scale)
    rho, _, _ = _scipy_compatible_loss_terms(np.square(values / scale), loss)
    return float(0.5 * scale**2 * np.sum(rho))


def _scipy_compatible_robust_linearization_weights(
    residuals: np.ndarray,
    *,
    loss: str,
    f_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Jacobian row scales and residual weights used by SciPy's gradient."""

    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    scale = float(f_scale)
    _, rho_prime, rho_double_prime = _scipy_compatible_loss_terms(
        np.square(values / scale), loss
    )
    rho_double_prime = rho_double_prime / scale**2
    jacobian_scale = rho_prime + 2.0 * rho_double_prime * values**2
    jacobian_scale[jacobian_scale < np.finfo(np.float64).eps] = np.finfo(
        np.float64
    ).eps
    np.sqrt(jacobian_scale, out=jacobian_scale)
    return jacobian_scale, rho_prime


def _project_gradient_for_bounds(
    x: np.ndarray,
    gradient: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    active_atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Return SciPy-style bound active mask and projected KKT gradient."""

    values = np.asarray(x, dtype=np.float64).reshape(-1)
    grad = np.asarray(gradient, dtype=np.float64).reshape(-1)
    lower_bounds = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper_bounds = np.asarray(upper, dtype=np.float64).reshape(-1)
    if not (
        values.shape == grad.shape == lower_bounds.shape == upper_bounds.shape
    ):
        raise ValueError("projected gradient inputs must have matching shapes")

    active_mask = np.zeros(values.shape, dtype=np.int8)
    lower_active = np.isfinite(lower_bounds) & np.isclose(
        values, lower_bounds, rtol=0.0, atol=active_atol
    )
    upper_active = np.isfinite(upper_bounds) & np.isclose(
        values, upper_bounds, rtol=0.0, atol=active_atol
    )
    active_mask[lower_active] = -1
    active_mask[upper_active] = 1

    projected = grad.copy()
    projected[(active_mask < 0) & (grad > 0.0)] = 0.0
    projected[(active_mask > 0) & (grad < 0.0)] = 0.0
    return active_mask, projected


def _joint_schur_select_restoration_trial(
    trials: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select the largest production-equivalent alpha for Round22 canary."""

    passed_trials = [
        trial
        for trial in trials
        if trial.get("production_equivalent_gate_passed") is True
    ]
    if not passed_trials:
        return None
    return max(passed_trials, key=lambda trial: float(trial["alpha"]))


def _joint_schur_rank_boundary_restoration_readiness(
    projection: Any,
) -> tuple[bool, str | None]:
    """Return whether a rank-boundary projection is safe to promote once."""

    if not isinstance(projection, dict):
        return False, "rank_boundary_projection_unavailable"
    if projection.get("valid") is not True:
        reason = projection.get("reason")
        return (
            False,
            "rank_boundary_projection_invalid"
            if reason is None
            else f"rank_boundary_projection_invalid:{reason}",
        )
    active_count = int(projection.get("active_constraint_count") or 0)
    correction_l2 = float(projection.get("metric_correction_l2") or 0.0)
    if active_count > 0:
        return True, None
    if correction_l2 > 0.0:
        return True, None
    return False, "rank_boundary_projection_no_active_constraints_or_correction"


def _joint_schur_restoration_has_refresh_budget(
    outer_iteration: int,
    requested_outer_iterations: int,
) -> bool:
    """Return whether a restoration can be followed by one Schur refresh."""

    return int(outer_iteration) + 1 < int(requested_outer_iterations)


def _joint_schur_record_restoration_refresh(
    restoration_diagnostics: dict[str, Any],
    outer_diagnostics: list[dict[str, Any]],
    *,
    outer_iteration: int,
    accepted: bool,
    trial_count: int,
    refresh_stop_reason: str,
) -> None:
    """Record the single refresh attempt after Round22 restoration."""

    restoration_diagnostics["refresh_performed"] = True
    restoration_diagnostics["refresh_outer_iteration"] = int(outer_iteration)
    restoration_diagnostics["refresh_accepted"] = bool(accepted)
    restoration_diagnostics["refresh_trial_count"] = int(trial_count)
    restoration_diagnostics["refresh_stop_reason"] = str(refresh_stop_reason)
    if (
        outer_diagnostics
        and outer_diagnostics[-1].get("outer_iteration") == int(outer_iteration)
    ):
        outer_diagnostics[-1]["one_active_set_restoration_refresh"] = {
            "performed": True,
            "accepted": bool(accepted),
            "stop_reason": str(refresh_stop_reason),
        }


@dataclass(frozen=True)
class HierarchicalPoseGraphConfig:
    """Numerical controls for the joint fine/coarse optimization."""

    fine_edge_weight: float = 1.0
    coarse_prior_weight: float = 0.25
    normalize_residual_families: bool = False
    frame_stride: int = 1
    max_nfev: int = 100
    solver_ftol: float = 1e-8
    solver_xtol: float = 1e-8
    solver_gtol: float = 1e-8
    joint_lsmr_tolerance: float | None = None
    robust_loss: str = "huber"
    f_scale: float = 1.0
    translation_scale: float = 1.0
    record_residual_components: bool = False
    max_log_scale_delta: float = _MAX_LOG_SCALE_DELTA
    apply_frame_corrections: bool = True
    frame_correction_alpha: float = 1.0
    freeze_coarse_alignments: bool = False
    frozen_optimizer_variable_rows_only: bool = True
    factorized_frame_chart_residual_v4: bool = False
    factorized_frame_chart_residual_v4_keyframe_continuation: bool = False
    factorized_frame_chart_residual_v4_keyframe_continuation_objective_projection: (
        bool
    ) = False
    factorized_frame_chart_residual_v4_initial_frame_deltas: (
        Sequence[Sim3] | None
    ) = None
    factorized_frame_chart_residual_v4_initial_frame_delta_candidates: (
        Sequence[tuple[float, Sequence[Sim3]]] | None
    ) = None
    max_frames_per_solve: int = 2000
    consensus_keyframe_stride: int = 20
    long_sequence_mode: str = "overlap_consensus"
    keyframe_selection: str = "uniform"
    keyframe_target_stride: int = 5
    keyframe_min_overlap_shared: int = 3
    chunk_scale_trust_mode: str = "fixed"
    chunk_scale_trust_gain: float = 2.0
    chunk_scale_trust_min_ratio: float = 0.2
    chunk_scale_trust_max_ratio: float = 2.0
    use_jac_sparsity: bool = False
    solver_workers: int = 1
    precompute_frozen_residual_constants: bool = True
    vectorize_frozen_residual: bool = True
    incremental_frozen_residual: bool = True
    analytic_frozen_jacobian: bool = True
    analytic_dynamic_jacobian: bool = False
    analytic_dynamic_jacobian_backend: str = "closed_form"
    analytic_dynamic_jacobian_local_rotation_projection: str = "strict"
    vectorize_dynamic_residual: bool = False
    analytic_jacobian_threads: int = 8
    analytic_jacobian_backend: str = "closed_form"
    analytic_fast_residual_logmap: bool = True
    joint_family_diagnostics: bool = False
    joint_full_sparse_exact_basis: bool = False
    joint_frame_relaxation_max_nfev: int = 0
    joint_subspace_max_nfev: int = 0
    joint_subspace_control_stride: int = 64
    joint_subspace_conditional_frame_polish: bool = False
    joint_subspace_min_coarse_squared_residual: float = 0.0
    joint_subspace_stage_b_rotation_trust_ratio: float | None = None
    joint_subspace_stage_b_translation_trust_ratio: float | None = None
    joint_subspace_stage_b_p64_gauge_fix: bool = False
    joint_subspace_stage_b_p64_sparse_basis: bool = False
    joint_subspace_stage_b_p64_sparse_basis_outer_cycle: bool = False
    joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking: (
        bool
    ) = False
    joint_subspace_unrestricted_repair_max_nfev: int = 0
    joint_schur_diagnostics: bool = False
    joint_schur_damping: float = 1e-4
    joint_schur_cg_maxiter: int = 200
    joint_schur_cg_rtol: float = 1e-4
    joint_schur_two_level_aggregate_size: int | None = None
    joint_schur_full_space_block_pcg_probe: bool = False
    joint_schur_step: bool = False
    joint_schur_step_final_solver: bool = False
    joint_schur_frozen_constrained_step_oracle: bool = False
    joint_schur_one_active_set_restoration_at_first_reject: bool = False
    joint_schur_step_public_pilot_max_nfev: int = 0
    joint_schur_step_max_outer_iterations: int = 1
    joint_schur_step_initial_alpha: float = 0.0625
    joint_schur_step_backtracks: int = 4
    joint_schur_step_freeze_coarse_scales: bool = False
    joint_schur_step_min_accepted_alpha: float | None = None
    joint_schur_step_frame_low_frequency_trust_ratio: float | None = None
    joint_schur_step_frame_low_frequency_trust_aggregate_size: int = 64
    joint_schur_step_min_low_frequency_efficiency_ratio: float | None = None
    joint_schur_step_max_initial_jacobi_low_frequency_efficiency: (
        float | None
    ) = None
    joint_schur_step_max_initial_diagonal_low_frequency_efficiency: (
        float | None
    ) = None
    joint_schur_step_max_initial_diagonal_reduction_fraction: (
        float | None
    ) = None
    joint_schur_step_max_graph_cost_ratio: float = 1.02
    joint_schur_step_max_adjacent_p95_ratio: float | None = None
    joint_schur_step_max_loop_p95_ratio: float | None = None
    joint_schur_step_max_loop_max_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.fine_edge_weight < 0.0:
            raise ValueError("fine_edge_weight must be non-negative")
        if self.coarse_prior_weight < 0.0:
            raise ValueError("coarse_prior_weight must be non-negative")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be positive")
        if self.max_nfev < 1:
            raise ValueError("max_nfev must be positive")
        if self.joint_full_sparse_exact_basis and self.freeze_coarse_alignments:
            raise ValueError(
                "joint_full_sparse_exact_basis requires free coarse alignments"
            )
        if self.factorized_frame_chart_residual_v4:
            if not self.freeze_coarse_alignments:
                raise ValueError(
                    "factorized_frame_chart_residual_v4 requires "
                    "freeze_coarse_alignments=True"
                )
            if self.long_sequence_mode not in {
                "overlap_consensus",
                "keyframe_fine_graph",
            }:
                raise ValueError(
                    "factorized_frame_chart_residual_v4 supports only frozen/keyframe "
                    "adjacent-chain solves"
                )
            if (
                self.joint_full_sparse_exact_basis
                or self.joint_frame_relaxation_max_nfev > 0
                or self.joint_subspace_max_nfev > 0
                or self.joint_schur_diagnostics
                or self.joint_schur_step
                or self.joint_schur_step_final_solver
                or self.joint_schur_frozen_constrained_step_oracle
                or self.joint_schur_one_active_set_restoration_at_first_reject
                or self.joint_schur_step_public_pilot_max_nfev > 0
            ):
                raise ValueError(
                    "factorized_frame_chart_residual_v4 cannot be combined with joint "
                    "subspace, sparse, frame-relaxation, or Schur controls"
                )
        if self.factorized_frame_chart_residual_v4_keyframe_continuation:
            if not (
                self.freeze_coarse_alignments
                and self.factorized_frame_chart_residual_v4
                and self.long_sequence_mode == "keyframe_fine_graph"
            ):
                raise ValueError(
                    "factorized_frame_chart_residual_v4_keyframe_continuation "
                    "requires frozen factorized v4 keyframe_fine_graph"
                )
            if self.max_nfev < len(
                _FACTORIZED_V4_KEYFRAME_CONTINUATION_STRIDE_MULTIPLIERS
            ):
                raise ValueError(
                    "factorized_frame_chart_residual_v4_keyframe_continuation "
                    "requires max_nfev budget for all continuation levels"
                )
        if (
            self.factorized_frame_chart_residual_v4_keyframe_continuation_objective_projection
            and not self.factorized_frame_chart_residual_v4_keyframe_continuation
        ):
            raise ValueError(
                "factorized_frame_chart_residual_v4_keyframe_continuation_"
                "objective_projection requires keyframe continuation"
            )
        if (
            self.factorized_frame_chart_residual_v4_initial_frame_deltas is not None
            and not self.factorized_frame_chart_residual_v4
        ):
            raise ValueError(
                "factorized_frame_chart_residual_v4_initial_frame_deltas "
                "requires factorized_frame_chart_residual_v4=True"
            )
        if (
            self.factorized_frame_chart_residual_v4_initial_frame_delta_candidates
            is not None
        ):
            if not self.factorized_frame_chart_residual_v4:
                raise ValueError(
                    "factorized_frame_chart_residual_v4_initial_frame_delta_"
                    "candidates requires factorized_frame_chart_residual_v4=True"
                )
            if self.factorized_frame_chart_residual_v4_initial_frame_deltas is not None:
                raise ValueError(
                    "factorized_frame_chart_residual_v4 initial frame deltas "
                    "and candidates are mutually exclusive"
                )
        if self.joint_full_sparse_exact_basis and (
            self.joint_subspace_max_nfev > 0
            or self.joint_frame_relaxation_max_nfev > 0
            or self.joint_schur_diagnostics
            or self.joint_schur_full_space_block_pcg_probe
            or self.joint_schur_step
            or self.joint_schur_step_final_solver
            or self.joint_schur_frozen_constrained_step_oracle
            or self.joint_schur_one_active_set_restoration_at_first_reject
            or self.joint_schur_step_public_pilot_max_nfev > 0
            or self.joint_subspace_stage_b_rotation_trust_ratio is not None
            or self.joint_subspace_stage_b_translation_trust_ratio is not None
            or self.joint_subspace_stage_b_p64_gauge_fix
            or self.joint_subspace_stage_b_p64_sparse_basis
            or self.joint_subspace_stage_b_p64_sparse_basis_outer_cycle
            or (
                self
                .joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking
            )
            or self.joint_subspace_unrestricted_repair_max_nfev > 0
        ):
            raise ValueError(
                "joint_full_sparse_exact_basis cannot be combined with "
                "subspace, Stage-B, frame relaxation, Schur, public pilot, "
                "or unrestricted repair controls"
            )
        if self.joint_schur_full_space_block_pcg_probe and (
            self.joint_full_sparse_exact_basis
            or self.joint_subspace_max_nfev > 0
            or self.joint_subspace_stage_b_rotation_trust_ratio is not None
            or self.joint_subspace_stage_b_translation_trust_ratio is not None
            or self.joint_subspace_stage_b_p64_gauge_fix
            or self.joint_subspace_stage_b_p64_sparse_basis
            or self.joint_subspace_stage_b_p64_sparse_basis_outer_cycle
            or (
                self
                .joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking
            )
            or self.joint_subspace_unrestricted_repair_max_nfev > 0
        ):
            raise ValueError(
                "joint_schur_full_space_block_pcg_probe cannot be combined "
                "with full sparse exact basis, subspace, or Stage-B controls"
            )
        if self.joint_frame_relaxation_max_nfev < 0:
            raise ValueError("joint_frame_relaxation_max_nfev must be non-negative")
        if self.joint_subspace_max_nfev < 0:
            raise ValueError("joint_subspace_max_nfev must be non-negative")
        if self.joint_subspace_max_nfev >= self.max_nfev:
            raise ValueError(
                "joint_subspace_max_nfev must be smaller than max_nfev"
            )
        if self.joint_subspace_control_stride < 2:
            raise ValueError("joint_subspace_control_stride must be at least 2")
        if (
            not np.isfinite(self.joint_subspace_min_coarse_squared_residual)
            or self.joint_subspace_min_coarse_squared_residual < 0.0
        ):
            raise ValueError(
                "joint_subspace_min_coarse_squared_residual must be finite "
                "and non-negative"
            )
        if (
            self.joint_subspace_min_coarse_squared_residual > 0.0
            and self.joint_subspace_max_nfev == 0
        ):
            raise ValueError(
                "joint_subspace_min_coarse_squared_residual requires "
                "joint_subspace_max_nfev"
            )
        stage_b_trust_ratios = (
            self.joint_subspace_stage_b_rotation_trust_ratio,
            self.joint_subspace_stage_b_translation_trust_ratio,
        )
        if (stage_b_trust_ratios[0] is None) != (
            stage_b_trust_ratios[1] is None
        ):
            raise ValueError(
                "joint_subspace_stage_b rotation and translation trust "
                "ratios must be configured together"
            )
        if stage_b_trust_ratios[0] is not None and any(
            not np.isfinite(value) or value <= 0.0
            for value in stage_b_trust_ratios
        ):
            raise ValueError(
                "joint_subspace_stage_b trust ratios must be finite and "
                "positive"
            )
        if (
            stage_b_trust_ratios[0] is not None
            and not self.joint_subspace_conditional_frame_polish
        ):
            raise ValueError(
                "joint_subspace_stage_b trust ratios require "
                "joint_subspace_conditional_frame_polish"
            )
        if (
            self.joint_subspace_stage_b_p64_gauge_fix
            and not self.joint_subspace_conditional_frame_polish
        ):
            raise ValueError(
                "joint_subspace_stage_b_p64_gauge_fix requires "
                "joint_subspace_conditional_frame_polish"
            )
        if (
            self.joint_subspace_stage_b_p64_sparse_basis
            and not self.joint_subspace_conditional_frame_polish
        ):
            raise ValueError(
                "joint_subspace_stage_b_p64_sparse_basis requires "
                "joint_subspace_conditional_frame_polish"
            )
        if (
            self.joint_subspace_stage_b_p64_sparse_basis_outer_cycle
            and not self.joint_subspace_conditional_frame_polish
        ):
            raise ValueError(
                "joint_subspace_stage_b_p64_sparse_basis_outer_cycle requires "
                "joint_subspace_conditional_frame_polish"
            )
        if (
            self.joint_subspace_stage_b_p64_sparse_basis_outer_cycle
            and not self.joint_subspace_stage_b_p64_sparse_basis
        ):
            raise ValueError(
                "joint_subspace_stage_b_p64_sparse_basis_outer_cycle requires "
                "joint_subspace_stage_b_p64_sparse_basis"
            )
        if (
            self.joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking
            and not self.joint_subspace_conditional_frame_polish
        ):
            raise ValueError(
                "joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking "
                "requires joint_subspace_conditional_frame_polish"
            )
        if (
            self.joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking
            and not self.joint_subspace_stage_b_p64_sparse_basis
        ):
            raise ValueError(
                "joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking "
                "requires joint_subspace_stage_b_p64_sparse_basis"
            )
        if (
            self.joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking
            and not self.joint_subspace_stage_b_p64_sparse_basis_outer_cycle
        ):
            raise ValueError(
                "joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking "
                "requires joint_subspace_stage_b_p64_sparse_basis_outer_cycle"
            )
        if (
            self.joint_subspace_stage_b_p64_gauge_fix
            and stage_b_trust_ratios[0] is not None
        ):
            raise ValueError(
                "joint_subspace_stage_b_p64_gauge_fix cannot be combined "
                "with stage_b trust ratios"
            )
        if (
            self.joint_subspace_stage_b_p64_sparse_basis
            and stage_b_trust_ratios[0] is not None
        ):
            raise ValueError(
                "joint_subspace_stage_b_p64_sparse_basis cannot be combined "
                "with stage_b trust ratios"
            )
        if (
            self.joint_subspace_stage_b_p64_sparse_basis
            and self.joint_subspace_stage_b_p64_gauge_fix
        ):
            raise ValueError(
                "joint_subspace_stage_b_p64_sparse_basis cannot be combined "
                "with p64 gauge fix"
            )
        if self.joint_subspace_unrestricted_repair_max_nfev < 0:
            raise ValueError(
                "joint_subspace_unrestricted_repair_max_nfev must be "
                "non-negative"
            )
        if (
            self.joint_subspace_unrestricted_repair_max_nfev > 0
            and not self.joint_subspace_conditional_frame_polish
        ):
            raise ValueError(
                "joint_subspace_unrestricted_repair_max_nfev requires "
                "joint_subspace_conditional_frame_polish"
            )
        if (
            stage_b_trust_ratios[0] is not None
            and self.joint_subspace_unrestricted_repair_max_nfev > 0
        ):
            raise ValueError(
                "joint_subspace_stage_b trust ratios cannot be combined "
                "with unrestricted repair"
            )
        if (
            self.joint_subspace_stage_b_p64_gauge_fix
            and self.joint_subspace_unrestricted_repair_max_nfev > 0
        ):
            raise ValueError(
                "joint_subspace_stage_b_p64_gauge_fix cannot be combined "
                "with unrestricted repair"
            )
        if (
            self.joint_subspace_stage_b_p64_sparse_basis
            and self.joint_subspace_unrestricted_repair_max_nfev > 0
        ):
            raise ValueError(
                "joint_subspace_stage_b_p64_sparse_basis cannot be combined "
                "with unrestricted repair"
            )
        if self.joint_subspace_max_nfev > 0 and self.freeze_coarse_alignments:
            raise ValueError(
                "joint_subspace_max_nfev requires free coarse alignments"
            )
        if (
            self.joint_subspace_conditional_frame_polish
            and self.joint_subspace_max_nfev == 0
        ):
            raise ValueError(
                "joint_subspace_conditional_frame_polish requires "
                "joint_subspace_max_nfev"
            )
        if self.joint_subspace_max_nfev > 0 and (
            self.joint_frame_relaxation_max_nfev > 0
            or self.joint_schur_diagnostics
            or self.joint_schur_full_space_block_pcg_probe
            or self.joint_schur_step
            or self.joint_schur_step_final_solver
            or self.joint_schur_frozen_constrained_step_oracle
            or self.joint_schur_one_active_set_restoration_at_first_reject
            or self.joint_schur_step_public_pilot_max_nfev > 0
        ):
            raise ValueError(
                "joint_subspace_max_nfev cannot be combined with frame "
                "relaxation, Schur diagnostics/steps, or a public pilot"
            )
        if not np.isfinite(self.joint_schur_damping) or self.joint_schur_damping <= 0.0:
            raise ValueError("joint_schur_damping must be finite and positive")
        if self.joint_schur_cg_maxiter < 1:
            raise ValueError("joint_schur_cg_maxiter must be positive")
        if self.joint_schur_step_public_pilot_max_nfev < 0:
            raise ValueError(
                "joint_schur_step_public_pilot_max_nfev must be non-negative"
            )
        if self.joint_schur_step_public_pilot_max_nfev >= self.max_nfev:
            raise ValueError(
                "joint_schur_step_public_pilot_max_nfev must be smaller than "
                "max_nfev"
            )
        if (
            self.joint_schur_step_public_pilot_max_nfev > 0
            and not self.joint_schur_step
        ):
            raise ValueError(
                "joint_schur_step_public_pilot_max_nfev requires joint_schur_step"
            )
        if (
            self.joint_schur_step_public_pilot_max_nfev > 0
            and self.joint_frame_relaxation_max_nfev > 0
        ):
            raise ValueError(
                "joint_schur_step_public_pilot_max_nfev cannot be combined "
                "with joint_frame_relaxation_max_nfev"
            )
        if self.joint_schur_step_final_solver and not self.joint_schur_step:
            raise ValueError(
                "joint_schur_step_final_solver requires joint_schur_step"
            )
        if (
            self.joint_schur_frozen_constrained_step_oracle
            and not self.joint_schur_step
        ):
            raise ValueError(
                "joint_schur_frozen_constrained_step_oracle requires "
                "joint_schur_step"
            )
        if self.joint_schur_one_active_set_restoration_at_first_reject and (
            not self.joint_schur_step_final_solver
            or not self.joint_schur_frozen_constrained_step_oracle
        ):
            raise ValueError(
                "joint_schur_one_active_set_restoration_at_first_reject "
                "requires joint_schur_step_final_solver and "
                "joint_schur_frozen_constrained_step_oracle"
            )
        if (
            self.joint_schur_step_final_solver
            and not self.analytic_dynamic_jacobian
        ):
            raise ValueError(
                "joint_schur_step_final_solver requires analytic_dynamic_jacobian"
            )
        if (
            self.joint_schur_step_final_solver
            and self.joint_schur_step_public_pilot_max_nfev > 0
        ):
            raise ValueError(
                "joint_schur_step_final_solver cannot be combined with "
                "joint_schur_step_public_pilot_max_nfev"
            )
        if (
            self.joint_schur_step_final_solver
            and self.joint_frame_relaxation_max_nfev > 0
        ):
            raise ValueError(
                "joint_schur_step_final_solver cannot be combined with "
                "joint_frame_relaxation_max_nfev"
            )
        if (
            self.joint_schur_step_final_solver
            and self.joint_schur_full_space_block_pcg_probe
        ):
            raise ValueError(
                "joint_schur_step_final_solver cannot be combined with "
                "joint_schur_full_space_block_pcg_probe"
            )
        if (
            self.joint_schur_step_final_solver
            and self.joint_schur_step_freeze_coarse_scales
        ):
            raise ValueError(
                "joint_schur_step_final_solver currently requires "
                "joint_schur_step_freeze_coarse_scales=False"
            )
        if not np.isfinite(self.joint_schur_cg_rtol) or self.joint_schur_cg_rtol <= 0.0:
            raise ValueError("joint_schur_cg_rtol must be finite and positive")
        if (
            self.joint_schur_two_level_aggregate_size is not None
            and self.joint_schur_two_level_aggregate_size < 2
        ):
            raise ValueError(
                "joint_schur_two_level_aggregate_size must be None or >= 2"
            )
        if (
            self.joint_schur_full_space_block_pcg_probe
            and not self.joint_schur_diagnostics
        ):
            raise ValueError(
                "joint_schur_full_space_block_pcg_probe requires "
                "joint_schur_diagnostics"
            )
        if (
            self.joint_schur_full_space_block_pcg_probe
            and self.joint_schur_step
        ):
            raise ValueError(
                "joint_schur_full_space_block_pcg_probe cannot be combined "
                "with joint_schur_step"
            )
        if (
            self.joint_schur_full_space_block_pcg_probe
            and self.joint_schur_step_freeze_coarse_scales
        ):
            raise ValueError(
                "joint_schur_full_space_block_pcg_probe cannot be combined "
                "with joint_schur_step_freeze_coarse_scales"
            )
        if (
            not np.isfinite(self.joint_schur_step_initial_alpha)
            or not 0.0 < self.joint_schur_step_initial_alpha <= 1.0
        ):
            raise ValueError("joint_schur_step_initial_alpha must be in (0, 1]")
        if self.joint_schur_step_backtracks < 1:
            raise ValueError("joint_schur_step_backtracks must be positive")
        if (
            self.joint_schur_step_min_accepted_alpha is not None
            and (
                not np.isfinite(self.joint_schur_step_min_accepted_alpha)
                or not 0.0
                < self.joint_schur_step_min_accepted_alpha
                <= self.joint_schur_step_initial_alpha
            )
        ):
            raise ValueError(
                "joint_schur_step_min_accepted_alpha must be None or in "
                "(0, joint_schur_step_initial_alpha]"
            )
        if (
            self.joint_schur_step_frame_low_frequency_trust_ratio is not None
            and (
                not np.isfinite(
                    self.joint_schur_step_frame_low_frequency_trust_ratio
                )
                or self.joint_schur_step_frame_low_frequency_trust_ratio <= 0.0
            )
        ):
            raise ValueError(
                "joint_schur_step_frame_low_frequency_trust_ratio must be "
                "None or finite and positive"
            )
        if self.joint_schur_step_frame_low_frequency_trust_aggregate_size < 2:
            raise ValueError(
                "joint_schur_step_frame_low_frequency_trust_aggregate_size "
                "must be >= 2"
            )
        if (
            self.joint_schur_step_min_low_frequency_efficiency_ratio is not None
            and (
                not np.isfinite(
                    self.joint_schur_step_min_low_frequency_efficiency_ratio
                )
                or not 0.0
                < self.joint_schur_step_min_low_frequency_efficiency_ratio
                <= 1.0
            )
        ):
            raise ValueError(
                "joint_schur_step_min_low_frequency_efficiency_ratio must be "
                "None or in (0, 1]"
            )
        if (
            self.joint_schur_step_max_initial_jacobi_low_frequency_efficiency
            is not None
            and (
                not np.isfinite(
                    self.joint_schur_step_max_initial_jacobi_low_frequency_efficiency
                )
                or self.joint_schur_step_max_initial_jacobi_low_frequency_efficiency
                <= 0.0
            )
        ):
            raise ValueError(
                "joint_schur_step_max_initial_jacobi_low_frequency_efficiency "
                "must be None or finite and positive"
            )
        if (
            self.joint_schur_step_max_initial_diagonal_low_frequency_efficiency
            is not None
            and (
                not np.isfinite(
                    self.joint_schur_step_max_initial_diagonal_low_frequency_efficiency
                )
                or self.joint_schur_step_max_initial_diagonal_low_frequency_efficiency
                <= 0.0
            )
        ):
            raise ValueError(
                "joint_schur_step_max_initial_diagonal_low_frequency_efficiency "
                "must be None or finite and positive"
            )
        if (
            self.joint_schur_step_max_initial_diagonal_reduction_fraction
            is not None
            and (
                not np.isfinite(
                    self.joint_schur_step_max_initial_diagonal_reduction_fraction
                )
                or not 0.0
                < self.joint_schur_step_max_initial_diagonal_reduction_fraction
                <= 1.0
            )
        ):
            raise ValueError(
                "joint_schur_step_max_initial_diagonal_reduction_fraction "
                "must be None or in (0, 1]"
            )
        if self.joint_schur_step_max_outer_iterations < 1:
            raise ValueError(
                "joint_schur_step_max_outer_iterations must be positive"
            )
        if (
            not np.isfinite(self.joint_schur_step_max_graph_cost_ratio)
            or self.joint_schur_step_max_graph_cost_ratio < 1.0
        ):
            raise ValueError(
                "joint_schur_step_max_graph_cost_ratio must be finite and >= 1"
            )
        for name, value in (
            (
                "joint_schur_step_max_adjacent_p95_ratio",
                self.joint_schur_step_max_adjacent_p95_ratio,
            ),
            (
                "joint_schur_step_max_loop_p95_ratio",
                self.joint_schur_step_max_loop_p95_ratio,
            ),
            (
                "joint_schur_step_max_loop_max_ratio",
                self.joint_schur_step_max_loop_max_ratio,
            ),
        ):
            if value is not None and (not np.isfinite(value) or value < 1.0):
                raise ValueError(f"{name} must be None or finite and >= 1")
        for name, value in (
            ("solver_ftol", self.solver_ftol),
            ("solver_xtol", self.solver_xtol),
            ("solver_gtol", self.solver_gtol),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.joint_lsmr_tolerance is not None and (
            not np.isfinite(self.joint_lsmr_tolerance)
            or self.joint_lsmr_tolerance <= 0.0
        ):
            raise ValueError("joint_lsmr_tolerance must be None or finite and positive")
        if self.f_scale <= 0.0:
            raise ValueError("f_scale must be positive")
        if self.translation_scale <= 0.0:
            raise ValueError("translation_scale must be positive")
        if not np.isfinite(self.max_log_scale_delta) or self.max_log_scale_delta <= 0.0:
            raise ValueError("max_log_scale_delta must be finite and positive")
        if not 0.0 <= self.frame_correction_alpha <= 1.0:
            raise ValueError("frame_correction_alpha must be in [0, 1]")
        if self.max_frames_per_solve < 1:
            raise ValueError("max_frames_per_solve must be positive")
        if self.consensus_keyframe_stride < 1:
            raise ValueError("consensus_keyframe_stride must be positive")
        if self.long_sequence_mode not in {
            "overlap_consensus",
            "keyframe_fine_graph",
            "keyframe_then_full",
        }:
            raise ValueError(
                "long_sequence_mode must be 'overlap_consensus', "
                "'keyframe_fine_graph', or 'keyframe_then_full'"
            )
        if (
            self.long_sequence_mode == "keyframe_then_full"
            and self.freeze_coarse_alignments
        ):
            raise ValueError("keyframe_then_full requires free coarse alignments")
        if self.keyframe_selection not in {"uniform", "motion_equalized"}:
            raise ValueError(
                "keyframe_selection must be 'uniform' or 'motion_equalized'"
            )
        if self.keyframe_target_stride < 2:
            raise ValueError("keyframe_target_stride must be at least 2")
        if self.keyframe_min_overlap_shared < 1:
            raise ValueError("keyframe_min_overlap_shared must be positive")
        if self.chunk_scale_trust_mode not in {"fixed", "edge_consistency"}:
            raise ValueError(
                "chunk_scale_trust_mode must be 'fixed' or 'edge_consistency'"
            )
        if not np.isfinite(self.chunk_scale_trust_gain) or self.chunk_scale_trust_gain <= 0.0:
            raise ValueError("chunk_scale_trust_gain must be finite and positive")
        if not 0.0 < self.chunk_scale_trust_min_ratio <= self.chunk_scale_trust_max_ratio:
            raise ValueError(
                "chunk_scale_trust ratios must satisfy 0 < min <= max"
            )
        if self.solver_workers < 1:
            raise ValueError("solver_workers must be positive")
        if self.analytic_jacobian_threads < 1:
            raise ValueError("analytic_jacobian_threads must be positive")
        if self.analytic_jacobian_backend not in {"torch", "closed_form"}:
            raise ValueError(
                "analytic_jacobian_backend must be 'torch' or 'closed_form'"
            )
        if self.analytic_dynamic_jacobian_backend not in {"torch", "closed_form"}:
            raise ValueError(
                "analytic_dynamic_jacobian_backend must be 'torch' or 'closed_form'"
            )


@dataclass(frozen=True)
class HierarchicalPoseGraphResult:
    chunks: list[Any]
    coarse_alignments: list[Sim3]
    frame_poses: list[Sim3]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _JointSchurCoarseGridCache:
    """Reusable SPD coarse solve for a fresh Schur operator's preconditioner."""

    aggregate_size: int
    frame_base: int
    prolongation: Any
    cholesky: np.ndarray
    condition: float
    age_outer_iterations: int = 0


@dataclass(frozen=True)
class HierarchicalAlphaBasis:
    """Alpha-independent hierarchical solve that can be replayed for exports."""

    coarse_alignments: list[Sim3]
    frame_poses: list[Sim3]
    apply_point_corrections: bool
    diagnostics: dict[str, Any]


def _frozen_frame_values(
    vector: np.ndarray,
    *,
    num_frames: int,
    fixed_frame: Sim3,
) -> np.ndarray:
    frame_values = np.empty((num_frames, 6), dtype=np.float64)
    frame_values[0] = _se3_to_parameters(fixed_frame)
    frame_values[1:] = np.asarray(vector, dtype=np.float64).reshape(num_frames - 1, 6)
    return frame_values


def _compute_frozen_vectorized_residual(
    vector: np.ndarray,
    *,
    num_frames: int,
    fixed_frame: Sim3,
    frozen_vectorized: dict[str, np.ndarray],
    fine_edge_weight: float,
    coarse_prior_weight: float,
    translation_scale: float,
    fast_rotvec: bool = False,
) -> np.ndarray:
    frame_values = _frozen_frame_values(
        vector,
        num_frames=num_frames,
        fixed_frame=fixed_frame,
    )
    frame_rot = Rotation.from_rotvec(frame_values[:, :3]).as_matrix()
    frame_trans = frame_values[:, 3:6]
    fine_residual = _compute_frozen_fine_residual_rows(
        np.arange(len(frozen_vectorized["fine_a"]), dtype=np.int64),
        frame_rot,
        frame_trans,
        frozen_vectorized=frozen_vectorized,
        fine_edge_weight=fine_edge_weight,
        translation_scale=translation_scale,
        fast_rotvec=fast_rotvec,
    )
    prior_residual = _compute_frozen_prior_residual_rows(
        np.arange(len(frozen_vectorized["prior_frame"]), dtype=np.int64),
        frame_rot,
        frame_trans,
        frozen_vectorized=frozen_vectorized,
        coarse_prior_weight=coarse_prior_weight,
        translation_scale=translation_scale,
        fast_rotvec=fast_rotvec,
    )
    return np.concatenate(
        [
            frozen_vectorized["coarse"],
            fine_residual.reshape(-1),
            prior_residual.reshape(-1),
        ]
    )


def _batch_transpose_left_matmul_3x3_exact(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Compute ``left.T @ right`` with einsum's exact three-term order."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 3 or left.shape[1:] != (3, 3):
        raise ValueError("batched matrix operands must both have shape (N, 3, 3)")
    result = np.empty_like(left)
    for row in range(3):
        for column in range(3):
            if _numexpr is None:
                result[:, row, column] = (
                    left[:, 0, row] * right[:, 0, column]
                    + left[:, 1, row] * right[:, 1, column]
                    + left[:, 2, row] * right[:, 2, column]
                )
            else:
                result[:, row, column] = _numexpr.evaluate(
                    "left0 * right0 + left1 * right1 + left2 * right2",
                    local_dict={
                        "left0": left[:, 0, row],
                        "right0": right[:, 0, column],
                        "left1": left[:, 1, row],
                        "right1": right[:, 1, column],
                        "left2": left[:, 2, row],
                        "right2": right[:, 2, column],
                    },
                )
    return result


def _compute_frozen_fine_residual_rows(
    fine_indices: np.ndarray,
    frame_rot: np.ndarray,
    frame_trans: np.ndarray,
    *,
    frozen_vectorized: dict[str, np.ndarray],
    fine_edge_weight: float,
    translation_scale: float,
    fast_rotvec: bool = False,
) -> np.ndarray:
    if fine_indices.size == 0:
        return np.zeros((0, 6), dtype=np.float64)
    fine_a = frozen_vectorized["fine_a"][fine_indices]
    fine_b = frozen_vectorized["fine_b"][fine_indices]
    predicted_rot = _batch_transpose_left_matmul_3x3_exact(
        frame_rot[fine_b], frame_rot[fine_a]
    )
    predicted_trans = np.einsum(
        "nji,nj->ni",
        frame_rot[fine_b],
        frame_trans[fine_a] - frame_trans[fine_b],
    )
    fine_error_rot = _batch_transpose_left_matmul_3x3_exact(
        frozen_vectorized["fine_rot"][fine_indices], predicted_rot
    )
    fine_error_trans = np.einsum(
        "nji,nj->ni",
        frozen_vectorized["fine_rot"][fine_indices],
        predicted_trans - frozen_vectorized["fine_trans"][fine_indices],
    )
    return np.concatenate(
        [
            (
                _numpy_matrix_to_rotvec_hybrid(fine_error_rot)
                if fast_rotvec
                else Rotation.from_matrix(fine_error_rot).as_rotvec()
            ),
            fine_error_trans * float(translation_scale),
        ],
        axis=1,
    ) * np.sqrt(fine_edge_weight)


def _compute_frozen_prior_residual_rows(
    prior_indices: np.ndarray,
    frame_rot: np.ndarray,
    frame_trans: np.ndarray,
    *,
    frozen_vectorized: dict[str, np.ndarray],
    coarse_prior_weight: float,
    translation_scale: float,
    fast_rotvec: bool = False,
) -> np.ndarray:
    if prior_indices.size == 0:
        return np.zeros((0, 6), dtype=np.float64)
    prior_frame = frozen_vectorized["prior_frame"][prior_indices]
    prior_error_rot = _batch_transpose_left_matmul_3x3_exact(
        frozen_vectorized["prior_rot"][prior_indices],
        frame_rot[prior_frame],
    )
    prior_error_trans = np.einsum(
        "nji,nj->ni",
        frozen_vectorized["prior_rot"][prior_indices],
        frame_trans[prior_frame] - frozen_vectorized["prior_trans"][prior_indices],
    )
    return np.concatenate(
        [
            (
                _numpy_matrix_to_rotvec_hybrid(prior_error_rot)
                if fast_rotvec
                else Rotation.from_matrix(prior_error_rot).as_rotvec()
            ),
            prior_error_trans * float(translation_scale),
        ],
        axis=1,
    ) * np.sqrt(coarse_prior_weight)


def _compute_frozen_fine_rotation_rows(
    fine_indices: np.ndarray,
    frame_rot: np.ndarray,
    *,
    frozen_vectorized: dict[str, np.ndarray],
    fine_edge_weight: float,
) -> np.ndarray:
    if fine_indices.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    fine_a = frozen_vectorized["fine_a"][fine_indices]
    fine_b = frozen_vectorized["fine_b"][fine_indices]
    predicted_rot = np.einsum(
        "nji,njk->nik", frame_rot[fine_b], frame_rot[fine_a]
    )
    fine_error_rot = np.einsum(
        "nji,njk->nik", frozen_vectorized["fine_rot"][fine_indices], predicted_rot
    )
    return Rotation.from_matrix(fine_error_rot).as_rotvec() * np.sqrt(
        fine_edge_weight
    )


def _compute_frozen_fine_translation_rows(
    fine_indices: np.ndarray,
    frame_rot: np.ndarray,
    frame_trans: np.ndarray,
    *,
    frozen_vectorized: dict[str, np.ndarray],
    fine_edge_weight: float,
    translation_scale: float,
) -> np.ndarray:
    if fine_indices.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    fine_a = frozen_vectorized["fine_a"][fine_indices]
    fine_b = frozen_vectorized["fine_b"][fine_indices]
    predicted_trans = np.einsum(
        "nji,nj->ni",
        frame_rot[fine_b],
        frame_trans[fine_a] - frame_trans[fine_b],
    )
    fine_error_trans = np.einsum(
        "nji,nj->ni",
        frozen_vectorized["fine_rot"][fine_indices],
        predicted_trans - frozen_vectorized["fine_trans"][fine_indices],
    )
    return (
        fine_error_trans
        * float(translation_scale)
        * np.sqrt(fine_edge_weight)
    )


def _compute_frozen_prior_rotation_rows(
    prior_indices: np.ndarray,
    frame_rot: np.ndarray,
    *,
    frozen_vectorized: dict[str, np.ndarray],
    coarse_prior_weight: float,
) -> np.ndarray:
    if prior_indices.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    prior_frame = frozen_vectorized["prior_frame"][prior_indices]
    prior_error_rot = np.einsum(
        "nji,njk->nik",
        frozen_vectorized["prior_rot"][prior_indices],
        frame_rot[prior_frame],
    )
    return Rotation.from_matrix(prior_error_rot).as_rotvec() * np.sqrt(
        coarse_prior_weight
    )


def _compute_frozen_prior_translation_rows(
    prior_indices: np.ndarray,
    frame_trans: np.ndarray,
    *,
    frozen_vectorized: dict[str, np.ndarray],
    coarse_prior_weight: float,
    translation_scale: float,
) -> np.ndarray:
    if prior_indices.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    prior_frame = frozen_vectorized["prior_frame"][prior_indices]
    prior_error_trans = np.einsum(
        "nji,nj->ni",
        frozen_vectorized["prior_rot"][prior_indices],
        frame_trans[prior_frame] - frozen_vectorized["prior_trans"][prior_indices],
    )
    return (
        prior_error_trans
        * float(translation_scale)
        * np.sqrt(coarse_prior_weight)
    )


def _torch_hat(vector: Any) -> Any:
    import torch

    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(vector.shape[:-1] + (3, 3))


def _torch_rotvec_to_matrix(vector: Any) -> Any:
    import torch

    theta2 = torch.sum(vector * vector, dim=-1)
    theta = torch.sqrt(torch.clamp(theta2, min=1e-16))
    small = theta2 < 1e-8
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta,
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / torch.clamp(theta2, min=1e-16),
    )
    skew = _torch_hat(vector)
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device).expand(
        skew.shape
    )
    return identity + a[..., None, None] * skew + b[..., None, None] * (
        skew @ skew
    )


def _torch_matrix_to_rotvec(matrix: Any) -> Any:
    import torch

    omega = 0.5 * torch.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        dim=-1,
    )
    sine2 = torch.sum(omega * omega, dim=-1)
    sine = torch.sqrt(torch.clamp(sine2, min=1e-16))
    cosine = torch.clamp(
        (matrix.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    theta = torch.atan2(sine, cosine)
    small = sine2 < 1e-8
    scale = torch.where(
        small,
        1.0 + theta * theta / 6.0 + 7.0 * theta**4 / 360.0,
        theta / sine,
    )
    return omega * scale[..., None]


def _torch_fine_residual(parameters: Any, measured_rot: Any, measured_trans: Any) -> Any:
    import torch

    rotation_a = _torch_rotvec_to_matrix(parameters[:3])
    rotation_b = _torch_rotvec_to_matrix(parameters[6:9])
    predicted_rotation = rotation_b.transpose(-1, -2) @ rotation_a
    predicted_translation = rotation_b.transpose(-1, -2) @ (
        parameters[3:6] - parameters[9:12]
    )
    error_rotation = measured_rot.transpose(-1, -2) @ predicted_rotation
    error_translation = measured_rot.transpose(-1, -2) @ (
        predicted_translation - measured_trans
    )
    return torch.cat((_torch_matrix_to_rotvec(error_rotation), error_translation))


def _torch_prior_residual(parameters: Any, prior_rot: Any, prior_trans: Any) -> Any:
    import torch

    rotation = _torch_rotvec_to_matrix(parameters[:3])
    return torch.cat(
        (
            _torch_matrix_to_rotvec(prior_rot.transpose(-1, -2) @ rotation),
            prior_rot.transpose(-1, -2) @ (parameters[3:6] - prior_trans),
        )
    )


def _torch_dynamic_fine_residual(
    parameters: Any,
    local_a_rot: Any,
    local_a_trans: Any,
    local_b_rot: Any,
    local_b_trans: Any,
    row_scale: Any,
) -> Any:
    import torch

    coarse = parameters[:7]
    frame_parameters = parameters[7:]
    coarse_rotation = _torch_rotvec_to_matrix(coarse[:3])
    coarse_scale = torch.exp(coarse[6])
    expected_a_rot = coarse_rotation @ local_a_rot
    expected_b_rot = coarse_rotation @ local_b_rot
    expected_a_trans = _torch_scale_matvec(
        coarse_scale, coarse_rotation, local_a_trans
    ) + coarse[3:6]
    expected_b_trans = _torch_scale_matvec(
        coarse_scale, coarse_rotation, local_b_trans
    ) + coarse[3:6]
    measured_rot = expected_b_rot.transpose(-1, -2) @ expected_a_rot
    measured_trans = expected_b_rot.transpose(-1, -2) @ (
        expected_a_trans - expected_b_trans
    )
    return row_scale * _torch_fine_residual(
        frame_parameters,
        measured_rot,
        measured_trans,
    )


def _torch_scale_matvec(scale: Any, matrix: Any, vector: Any) -> Any:
    return scale * (matrix @ vector)


def _torch_dynamic_prior_residual(
    parameters: Any,
    local_rot: Any,
    local_trans: Any,
    row_scale: Any,
) -> Any:
    import torch

    coarse = parameters[:7]
    frame_parameters = parameters[7:]
    coarse_rotation = _torch_rotvec_to_matrix(coarse[:3])
    coarse_scale = torch.exp(coarse[6])
    prior_rot = coarse_rotation @ local_rot
    prior_trans = (
        _torch_scale_matvec(coarse_scale, coarse_rotation, local_trans)
        + coarse[3:6]
    )
    return row_scale * _torch_prior_residual(
        frame_parameters,
        prior_rot,
        prior_trans,
    )


def _numpy_hat(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.moveaxis(np.asarray(vector, dtype=np.float64), -1, 0)
    zero = np.zeros_like(x)
    return np.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), axis=-1
    ).reshape(vector.shape[:-1] + (3, 3))


def _squared_norm3_exact(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return (
        vector[..., 0] * vector[..., 0]
        + vector[..., 1] * vector[..., 1]
        + vector[..., 2] * vector[..., 2]
    )


def _trace3_exact(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]


def _numpy_matrix_to_rotvec(
    matrix: np.ndarray, *, cosine: np.ndarray | None = None
) -> np.ndarray:
    omega = 0.5 * np.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        axis=-1,
    )
    sine2 = _squared_norm3_exact(omega)
    sine = np.sqrt(np.maximum(sine2, 1e-16))
    if cosine is None:
        cosine = np.clip((_trace3_exact(matrix) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arctan2(sine, cosine)
    small = sine2 < 1e-8
    scale = np.empty_like(sine)
    scale[small] = (
        1.0
        + theta[small] * theta[small] / 6.0
        + 7.0 * theta[small] ** 4 / 360.0
    )
    scale[~small] = theta[~small] / sine[~small]
    return omega * scale[..., None]


def _numpy_matrix_to_rotvec_hybrid(matrix: np.ndarray) -> np.ndarray:
    """Fast SO(3) log map with SciPy's stable branch near pi.

    The closed form is both faster and numerically well-conditioned away from
    pi.  Near pi, the skew-symmetric part approaches zero and cannot recover a
    stable axis, so retain SciPy's quaternion-based implementation only for
    those rows.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    cosine = np.clip((_trace3_exact(matrix) - 1.0) * 0.5, -1.0, 1.0)
    near_pi = cosine <= -0.9999
    result = _numpy_matrix_to_rotvec(matrix, cosine=cosine)
    if np.any(near_pi):
        result[near_pi] = Rotation.from_matrix(matrix[near_pi]).as_rotvec()
    return result


def _numpy_rotvec_to_matrix(vector: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues exponential without importing SciPy."""

    vector = np.asarray(vector, dtype=np.float64)
    theta2 = _squared_norm3_exact(vector)
    theta = np.sqrt(theta2)
    small = theta2 < 1e-8
    a = np.empty_like(theta)
    b = np.empty_like(theta)
    a[small] = 1.0 - theta2[small] / 6.0 + theta2[small] ** 2 / 120.0
    b[small] = 0.5 - theta2[small] / 24.0 + theta2[small] ** 2 / 720.0
    a[~small] = np.sin(theta[~small]) / theta[~small]
    b[~small] = (1.0 - np.cos(theta[~small])) / theta2[~small]
    x, y, z = np.moveaxis(vector, -1, 0)
    zero = np.zeros_like(x)
    skew = np.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), axis=-1
    ).reshape(vector.shape[:-1] + (3, 3))
    identity = np.broadcast_to(np.eye(3, dtype=np.float64), skew.shape)
    return identity + a[..., None, None] * skew + b[..., None, None] * np.einsum(
        "...ij,...jk->...ik", skew, skew
    )


def _numpy_so3_right_jacobian(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    theta2 = _squared_norm3_exact(vector)
    theta = np.sqrt(theta2)
    small = theta2 < 1e-8
    a = np.empty_like(theta)
    b = np.empty_like(theta)
    a[small] = (
        0.5 - theta2[small] / 24.0 + theta2[small] ** 2 / 720.0
    )
    b[small] = (
        1.0 / 6.0 - theta2[small] / 120.0 + theta2[small] ** 2 / 5040.0
    )
    a[~small] = (1.0 - np.cos(theta[~small])) / theta2[~small]
    b[~small] = (theta[~small] - np.sin(theta[~small])) / (
        theta2[~small] * theta[~small]
    )
    skew = _numpy_hat(vector)
    return (
        np.eye(3, dtype=np.float64)
        - a[..., None, None] * skew
        + b[..., None, None] * (skew @ skew)
    )


def _numpy_so3_log_inverse_jacobian(
    vector: np.ndarray, *, right: bool
) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    theta2 = _squared_norm3_exact(vector)
    theta = np.sqrt(theta2)
    small = theta2 < 1e-8
    coefficient = np.empty_like(theta)
    coefficient[small] = (
        1.0 / 12.0
        + theta2[small] / 720.0
        + theta2[small] ** 2 / 30240.0
    )
    coefficient[~small] = 1.0 / theta2[~small] - (
        (1.0 + np.cos(theta[~small]))
        / (2.0 * theta[~small] * np.sin(theta[~small]))
    )
    skew = _numpy_hat(vector)
    return (
        np.eye(3, dtype=np.float64)
        + (0.5 if right else -0.5) * skew
        + coefficient[..., None, None] * (skew @ skew)
    )


def _numpy_so3_log_inverse_jacobian_pair(
    vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return right/left inverse Jacobians with shared exact intermediates."""

    vector = np.asarray(vector, dtype=np.float64)
    theta2 = _squared_norm3_exact(vector)
    theta = np.sqrt(theta2)
    small = theta2 < 1e-8
    coefficient = np.empty_like(theta)
    coefficient[small] = (
        1.0 / 12.0
        + theta2[small] / 720.0
        + theta2[small] ** 2 / 30240.0
    )
    coefficient[~small] = 1.0 / theta2[~small] - (
        (1.0 + np.cos(theta[~small]))
        / (2.0 * theta[~small] * np.sin(theta[~small]))
    )
    skew = _numpy_hat(vector)
    quadratic = coefficient[..., None, None] * (skew @ skew)
    identity = np.eye(3, dtype=np.float64)
    return (
        identity + 0.5 * skew + quadratic,
        identity + -0.5 * skew + quadratic,
    )


def _closed_form_fine_jacobian(
    parameters: np.ndarray,
    measured_rot: np.ndarray,
) -> np.ndarray:
    rotation_a = Rotation.from_rotvec(parameters[:, :3]).as_matrix()
    rotation_b = Rotation.from_rotvec(parameters[:, 6:9]).as_matrix()
    measured_inverse = np.swapaxes(measured_rot, -1, -2)
    rotation_b_inverse = np.swapaxes(rotation_b, -1, -2)
    error_rotation = measured_inverse @ rotation_b_inverse @ rotation_a
    error_rotvec = _numpy_matrix_to_rotvec(error_rotation)
    right_a = _numpy_so3_right_jacobian(parameters[:, :3])
    right_b = _numpy_so3_right_jacobian(parameters[:, 6:9])
    jacobian = np.zeros((len(parameters), 6, 12), dtype=np.float64)
    jacobian[:, :3, :3] = (
        _numpy_so3_log_inverse_jacobian(error_rotvec, right=True) @ right_a
    )
    jacobian[:, :3, 6:9] = (
        _numpy_so3_log_inverse_jacobian(error_rotvec, right=False)
        @ (-measured_inverse @ right_b)
    )
    relative_translation = (
        rotation_b_inverse
        @ (parameters[:, 3:6] - parameters[:, 9:12])[..., None]
    )[..., 0]
    jacobian[:, 3:6, 6:9] = (
        measured_inverse @ _numpy_hat(relative_translation) @ right_b
    )
    translation_block = measured_inverse @ rotation_b_inverse
    jacobian[:, 3:6, 3:6] = translation_block
    jacobian[:, 3:6, 9:12] = -translation_block
    return jacobian


def _closed_form_prior_jacobian(
    parameters: np.ndarray,
    prior_rot: np.ndarray,
) -> np.ndarray:
    rotation = Rotation.from_rotvec(parameters[:, :3]).as_matrix()
    prior_inverse = np.swapaxes(prior_rot, -1, -2)
    error_rotvec = _numpy_matrix_to_rotvec(prior_inverse @ rotation)
    jacobian = np.zeros((len(parameters), 6, 6), dtype=np.float64)
    jacobian[:, :3, :3] = (
        _numpy_so3_log_inverse_jacobian(error_rotvec, right=True)
        @ _numpy_so3_right_jacobian(parameters[:, :3])
    )
    jacobian[:, 3:6, 3:6] = prior_inverse
    return jacobian


def _closed_form_fine_prior_jacobian_blocks_from_frames(
    frame_values: np.ndarray,
    *,
    fine_a: np.ndarray,
    fine_b: np.ndarray,
    fine_measured_rot: np.ndarray,
    prior_frame: np.ndarray,
    prior_rot: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Evaluate only the seven nonzero 3x3 blocks, reusing each frame once."""

    frame_values = np.asarray(frame_values, dtype=np.float64)
    frame_rotations = Rotation.from_rotvec(frame_values[:, :3]).as_matrix()
    frame_right_jacobians = _numpy_so3_right_jacobian(frame_values[:, :3])

    rotation_a = frame_rotations[fine_a]
    rotation_b = frame_rotations[fine_b]
    measured_inverse = np.swapaxes(fine_measured_rot, -1, -2)
    rotation_b_inverse = np.swapaxes(rotation_b, -1, -2)
    error_rotation = measured_inverse @ rotation_b_inverse @ rotation_a
    error_rotvec = _numpy_matrix_to_rotvec(error_rotation)
    log_inverse_right, log_inverse_left = (
        _numpy_so3_log_inverse_jacobian_pair(error_rotvec)
    )
    fine_rotation_a = log_inverse_right @ frame_right_jacobians[fine_a]
    fine_rotation_b = (
        log_inverse_left @ (-measured_inverse @ frame_right_jacobians[fine_b])
    )
    relative_translation = (
        rotation_b_inverse
        @ (frame_values[fine_a, 3:6] - frame_values[fine_b, 3:6])[..., None]
    )[..., 0]
    fine_translation_rotation_b = (
        measured_inverse
        @ _numpy_hat(relative_translation)
        @ frame_right_jacobians[fine_b]
    )
    translation_block = measured_inverse @ rotation_b_inverse
    prior_inverse = np.swapaxes(prior_rot, -1, -2)
    prior_error_rotvec = _numpy_matrix_to_rotvec(
        prior_inverse @ frame_rotations[prior_frame]
    )
    prior_rotation = (
        _numpy_so3_log_inverse_jacobian(prior_error_rotvec, right=True)
        @ frame_right_jacobians[prior_frame]
    )
    return (
        fine_rotation_a,
        fine_rotation_b,
        fine_translation_rotation_b,
        translation_block,
        -translation_block,
        prior_rotation,
        prior_inverse,
    )


def _closed_form_fine_prior_jacobian_from_frames(
    frame_values: np.ndarray,
    *,
    fine_a: np.ndarray,
    fine_b: np.ndarray,
    fine_measured_rot: np.ndarray,
    prior_frame: np.ndarray,
    prior_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize dense local blocks for derivation and regression checks."""

    blocks = _closed_form_fine_prior_jacobian_blocks_from_frames(
        frame_values,
        fine_a=fine_a,
        fine_b=fine_b,
        fine_measured_rot=fine_measured_rot,
        prior_frame=prior_frame,
        prior_rot=prior_rot,
    )
    fine_jacobian = np.zeros((len(fine_a), 6, 12), dtype=np.float64)
    fine_jacobian[:, :3, :3] = blocks[0]
    fine_jacobian[:, :3, 6:9] = blocks[1]
    fine_jacobian[:, 3:6, 6:9] = blocks[2]
    fine_jacobian[:, 3:6, 3:6] = blocks[3]
    fine_jacobian[:, 3:6, 9:12] = blocks[4]
    prior_jacobian = np.zeros((len(prior_frame), 6, 6), dtype=np.float64)
    prior_jacobian[:, :3, :3] = blocks[5]
    prior_jacobian[:, 3:6, 3:6] = blocks[6]
    return fine_jacobian, prior_jacobian


class _FrozenTorchAnalyticJacobian:
    def __init__(
        self,
        *,
        num_frames: int,
        fixed_frame: Sim3,
        frozen_vectorized: dict[str, np.ndarray],
        fine_edge_weight: float,
        coarse_prior_weight: float,
        translation_scale: float,
        threads: int = 8,
        backend: str = "torch",
    ) -> None:
        self.backend = str(backend)
        if self.backend not in {"torch", "closed_form"}:
            raise ValueError("unsupported analytic Jacobian backend")
        self.threads_requested = int(threads)
        self.threads_before = 0
        self.threads_actual = 0
        self.num_frames = int(num_frames)
        self.fixed_frame = fixed_frame
        self.fixed_parameters = _se3_to_parameters(fixed_frame)
        self.fine_a = np.asarray(frozen_vectorized["fine_a"], dtype=np.int64)
        self.fine_b = np.asarray(frozen_vectorized["fine_b"], dtype=np.int64)
        self.prior_frame = np.asarray(
            frozen_vectorized["prior_frame"], dtype=np.int64
        )
        self.coarse_rows = int(frozen_vectorized["coarse"].size)
        self.num_fine = len(self.fine_a)
        self.num_prior = len(self.prior_frame)
        self.row_count = self.coarse_rows + 6 * (self.num_fine + self.num_prior)
        self.num_variables = 6 * max(self.num_frames - 1, 0)
        self.fine_rot_numpy = np.asarray(
            frozen_vectorized["fine_rot"], dtype=np.float64
        )
        self.fine_trans_numpy = np.asarray(
            frozen_vectorized["fine_trans"], dtype=np.float64
        )
        self.prior_rot_numpy = np.asarray(
            frozen_vectorized["prior_rot"], dtype=np.float64
        )
        self.prior_trans_numpy = np.asarray(
            frozen_vectorized["prior_trans"], dtype=np.float64
        )
        self.fine_row_scale_numpy = np.asarray(
            [np.sqrt(fine_edge_weight)] * 3
            + [np.sqrt(fine_edge_weight) * translation_scale] * 3,
            dtype=np.float64,
        )
        self.prior_row_scale_numpy = np.asarray(
            [np.sqrt(coarse_prior_weight)] * 3
            + [np.sqrt(coarse_prior_weight) * translation_scale] * 3,
            dtype=np.float64,
        )
        self.closed_form_block_scales = (
            self.fine_row_scale_numpy[0],
            self.fine_row_scale_numpy[0],
            self.fine_row_scale_numpy[3],
            self.fine_row_scale_numpy[3],
            self.fine_row_scale_numpy[3],
            self.prior_row_scale_numpy[0],
            self.prior_row_scale_numpy[3],
        )
        if self.backend == "torch":
            import torch
            from torch.func import jacrev, vmap

            self.torch = torch
            self.threads_before = int(torch.get_num_threads())
            self.threads_actual = min(self.threads_requested, self.threads_before)
            self.fine_rot = torch.from_numpy(self.fine_rot_numpy)
            self.fine_trans = torch.from_numpy(self.fine_trans_numpy)
            self.prior_rot = torch.from_numpy(self.prior_rot_numpy)
            self.prior_trans = torch.from_numpy(self.prior_trans_numpy)
            self.fine_row_scale = torch.from_numpy(self.fine_row_scale_numpy)
            self.prior_row_scale = torch.from_numpy(self.prior_row_scale_numpy)
            self.fine_jacobian = vmap(jacrev(_torch_fine_residual, argnums=0))
            self.prior_jacobian = vmap(jacrev(_torch_prior_residual, argnums=0))
        self.specs: list[tuple[str, np.ndarray, slice, slice]] = []
        rows: list[np.ndarray] = []
        columns: list[np.ndarray] = []
        fine_base = self.coarse_rows + 6 * np.arange(self.num_fine)
        prior_base = self.coarse_rows + 6 * self.num_fine + 6 * np.arange(
            self.num_prior
        )

        def append(
            kind: str,
            bases: np.ndarray,
            frames: np.ndarray,
            residual_slice: slice,
            local_slice: slice,
            variable_offset: int,
        ) -> None:
            mask = np.flatnonzero(frames > 0)
            residual_components = np.arange(
                residual_slice.start, residual_slice.stop
            )
            variable_components = np.arange(local_slice.stop - local_slice.start)
            rows.append(
                (
                    bases[mask, None, None]
                    + residual_components[None, :, None]
                    + np.zeros((1, 1, len(variable_components)), dtype=np.int64)
                ).reshape(-1)
            )
            columns.append(
                (
                    6 * (frames[mask, None, None] - 1)
                    + variable_offset
                    + variable_components[None, None, :]
                    + np.zeros((1, len(residual_components), 1), dtype=np.int64)
                ).reshape(-1)
            )
            self.specs.append((kind, mask, residual_slice, local_slice))

        append("fine", fine_base, self.fine_a, slice(0, 3), slice(0, 3), 0)
        append("fine", fine_base, self.fine_b, slice(0, 3), slice(6, 9), 0)
        append("fine", fine_base, self.fine_b, slice(3, 6), slice(6, 9), 0)
        append("fine", fine_base, self.fine_a, slice(3, 6), slice(3, 6), 3)
        append("fine", fine_base, self.fine_b, slice(3, 6), slice(9, 12), 3)
        append("prior", prior_base, self.prior_frame, slice(0, 3), slice(0, 3), 0)
        append("prior", prior_base, self.prior_frame, slice(3, 6), slice(3, 6), 3)
        self.rows = np.concatenate(rows)
        self.columns = np.concatenate(columns)
        from scipy.sparse import coo_matrix

        topology = coo_matrix(
            (
                np.arange(len(self.rows), dtype=np.int64),
                (self.rows, self.columns),
            ),
            shape=(self.row_count, self.num_variables),
        ).tocsr()
        if topology.nnz != len(self.rows):
            raise ValueError("analytic Jacobian topology contains duplicate entries")
        self.csr_value_order = np.asarray(topology.data, dtype=np.intp)
        self.csr_indices = topology.indices.copy()
        self.csr_indptr = topology.indptr.copy()
        self.calls = 0
        self.wall_seconds = 0.0
        self.preparation_wall_seconds = 0.0
        self.derivative_wall_seconds = 0.0
        self.assembly_wall_seconds = 0.0
        self.last_nnz = len(self.rows)

    def restore_threads(self) -> None:
        if (
            self.backend == "torch"
            and self.threads_before > 0
            and self.threads_actual != self.threads_before
        ):
            self.torch.set_num_threads(self.threads_before)

    def activate_threads(self) -> None:
        if (
            self.backend == "torch"
            and self.threads_before > 0
            and self.threads_actual != self.threads_before
            and int(self.torch.get_num_threads()) != self.threads_actual
        ):
            self.torch.set_num_threads(self.threads_actual)

    def __call__(self, vector: np.ndarray) -> Any:
        self.activate_threads()
        started = perf_counter()
        frame_values = _frozen_frame_values(
            vector,
            num_frames=self.num_frames,
            fixed_frame=self.fixed_frame,
        )
        prepared = perf_counter()
        if self.backend == "torch":
            fine_parameters_numpy = np.concatenate(
                (frame_values[self.fine_a], frame_values[self.fine_b]), axis=1
            )
            fine_parameters = self.torch.from_numpy(fine_parameters_numpy)
            frame_tensor = self.torch.from_numpy(frame_values)
            fine_jacobian = self.fine_jacobian(
                fine_parameters, self.fine_rot, self.fine_trans
            ) * self.fine_row_scale[None, :, None]
            prior_jacobian = self.prior_jacobian(
                frame_tensor[self.prior_frame], self.prior_rot, self.prior_trans
            ) * self.prior_row_scale[None, :, None]
        else:
            closed_form_blocks = (
                _closed_form_fine_prior_jacobian_blocks_from_frames(
                    frame_values,
                    fine_a=self.fine_a,
                    fine_b=self.fine_b,
                    fine_measured_rot=self.fine_rot_numpy,
                    prior_frame=self.prior_frame,
                    prior_rot=self.prior_rot_numpy,
                )
            )
        differentiated = perf_counter()
        values = []
        for index, (kind, mask, residual_slice, local_slice) in enumerate(self.specs):
            if self.backend == "torch":
                source = fine_jacobian if kind == "fine" else prior_jacobian
                selected = source[mask, residual_slice, local_slice]
                selected = selected.detach().numpy()
            else:
                selected = (
                    closed_form_blocks[index][mask]
                    * self.closed_form_block_scales[index]
                )
            values.append(np.asarray(selected).reshape(-1))
        values_flat = np.concatenate(values)
        from scipy.sparse import csr_matrix

        jacobian = csr_matrix(
            (
                values_flat[self.csr_value_order],
                self.csr_indices,
                self.csr_indptr,
            ),
            shape=(self.row_count, self.num_variables),
            copy=False,
        )
        finished = perf_counter()
        self.calls += 1
        self.wall_seconds += finished - started
        self.preparation_wall_seconds += prepared - started
        self.derivative_wall_seconds += differentiated - prepared
        self.assembly_wall_seconds += finished - differentiated
        return jacobian


class _DynamicTorchAnalyticJacobian:
    """Opt-in Torch local-block Jacobian for unfrozen coarse Joint-CF solves."""

    _KIND_COARSE = 0
    _KIND_FINE = 1
    _KIND_PRIOR = 2

    def __init__(
        self,
        *,
        num_chunks: int,
        num_frames: int,
        fixed_root: Sim3,
        fixed_frame: Sim3,
        coarse_edges: Sequence[Sim3Edge],
        fine_edges: Sequence[tuple[int, int, int, np.ndarray, np.ndarray]],
        prior_records: Sequence[tuple[int, int, np.ndarray]],
        coarse_offsets: dict[int, int],
        frame_offsets: dict[int, int],
        fine_edge_weight: float,
        coarse_prior_weight: float,
        translation_scale: float,
        local_rotation_projection: str = "strict",
        threads: int = 8,
    ) -> None:
        import torch
        from torch.func import jacrev, vmap

        self.backend = "torch"
        self.local_rotation_projection = str(local_rotation_projection)
        self.threads_requested = int(threads)
        self.threads_before = int(torch.get_num_threads())
        self.threads_actual = min(self.threads_requested, self.threads_before)
        self.torch = torch
        self.num_chunks = int(num_chunks)
        self.num_frames = int(num_frames)
        self.frame_base = 7 * max(self.num_chunks - 1, 0)
        self.num_variables = self.frame_base + 6 * max(self.num_frames - 1, 0)
        expected_coarse_offsets = {
            index: 7 * (index - 1) for index in range(1, self.num_chunks)
        }
        expected_frame_offsets = {
            frame: self.frame_base + 6 * (frame - 1)
            for frame in range(1, self.num_frames)
        }
        if dict(coarse_offsets) != expected_coarse_offsets:
            raise ValueError("dynamic analytic Jacobian coarse layout mismatch")
        if dict(frame_offsets) != expected_frame_offsets:
            raise ValueError("dynamic analytic Jacobian frame layout mismatch")

        self.fixed_root_parameters = torch.as_tensor(
            _sim3_to_parameters(fixed_root), dtype=torch.float64
        )
        self.fixed_frame_parameters = torch.as_tensor(
            _se3_to_parameters(fixed_frame), dtype=torch.float64
        )
        self.sources = np.asarray(
            [edge.source for edge in coarse_edges], dtype=np.int64
        )
        self.targets = np.asarray(
            [edge.target for edge in coarse_edges], dtype=np.int64
        )
        self.num_coarse = len(self.sources)
        measured_inverse = [edge.transform.inverse() for edge in coarse_edges]
        self.measured_inverse_rot = torch.as_tensor(
            np.stack([item.rotation for item in measured_inverse], axis=0)
            if measured_inverse
            else np.zeros((0, 3, 3), dtype=np.float64),
            dtype=torch.float64,
        )
        self.measured_inverse_trans = torch.as_tensor(
            np.stack([item.translation for item in measured_inverse], axis=0)
            if measured_inverse
            else np.zeros((0, 3), dtype=np.float64),
            dtype=torch.float64,
        )
        self.measured_inverse_log_scale = torch.as_tensor(
            np.log([abs(item.scale) for item in measured_inverse])
            if measured_inverse
            else np.zeros(0, dtype=np.float64),
            dtype=torch.float64,
        )
        self.coarse_sqrt_weights = torch.as_tensor(
            np.sqrt(
                [
                    float(edge.diagnostics.get("weight", 1.0))
                    for edge in coarse_edges
                ]
            )
            if coarse_edges
            else np.zeros(0, dtype=np.float64),
            dtype=torch.float64,
        )
        self.coarse_row_scale = torch.stack(
            (
                self.coarse_sqrt_weights,
                self.coarse_sqrt_weights,
                self.coarse_sqrt_weights,
                self.coarse_sqrt_weights * float(translation_scale),
                self.coarse_sqrt_weights * float(translation_scale),
                self.coarse_sqrt_weights * float(translation_scale),
                self.coarse_sqrt_weights,
            ),
            dim=1,
        )

        self.fine_chunks = np.asarray(
            [item[0] for item in fine_edges], dtype=np.int64
        )
        self.fine_a = np.asarray([item[1] for item in fine_edges], dtype=np.int64)
        self.fine_b = np.asarray([item[2] for item in fine_edges], dtype=np.int64)
        self.num_fine = len(self.fine_a)
        fine_a_rot, fine_a_trans = self._pose_components(
            [item[3] for item in fine_edges],
            projection_mode=self.local_rotation_projection,
        )
        fine_b_rot, fine_b_trans = self._pose_components(
            [item[4] for item in fine_edges],
            projection_mode=self.local_rotation_projection,
        )
        self.fine_a_rot = torch.as_tensor(fine_a_rot, dtype=torch.float64)
        self.fine_a_trans = torch.as_tensor(fine_a_trans, dtype=torch.float64)
        self.fine_b_rot = torch.as_tensor(fine_b_rot, dtype=torch.float64)
        self.fine_b_trans = torch.as_tensor(fine_b_trans, dtype=torch.float64)
        self.fine_row_scale = torch.as_tensor(
            [
                np.sqrt(fine_edge_weight),
                np.sqrt(fine_edge_weight),
                np.sqrt(fine_edge_weight),
                np.sqrt(fine_edge_weight) * float(translation_scale),
                np.sqrt(fine_edge_weight) * float(translation_scale),
                np.sqrt(fine_edge_weight) * float(translation_scale),
            ],
            dtype=torch.float64,
        )

        self.prior_chunks = np.asarray(
            [item[0] for item in prior_records], dtype=np.int64
        )
        self.prior_frame = np.asarray(
            [item[1] for item in prior_records], dtype=np.int64
        )
        self.num_prior = len(self.prior_frame)
        prior_rot, prior_trans = self._pose_components(
            [item[2] for item in prior_records],
            projection_mode=self.local_rotation_projection,
        )
        self.prior_rot = torch.as_tensor(prior_rot, dtype=torch.float64)
        self.prior_trans = torch.as_tensor(prior_trans, dtype=torch.float64)
        self.prior_row_scale = torch.as_tensor(
            [
                np.sqrt(coarse_prior_weight),
                np.sqrt(coarse_prior_weight),
                np.sqrt(coarse_prior_weight),
                np.sqrt(coarse_prior_weight) * float(translation_scale),
                np.sqrt(coarse_prior_weight) * float(translation_scale),
                np.sqrt(coarse_prior_weight) * float(translation_scale),
            ],
            dtype=torch.float64,
        )

        self.coarse_jacobian = vmap(jacrev(self._coarse_residual, argnums=0))
        self.fine_jacobian = vmap(jacrev(_torch_dynamic_fine_residual, argnums=0))
        self.prior_jacobian = vmap(
            jacrev(_torch_dynamic_prior_residual, argnums=0)
        )
        self.coarse_residual = vmap(self._coarse_residual)
        self.fine_residual = vmap(_torch_dynamic_fine_residual)
        self.prior_residual = vmap(_torch_dynamic_prior_residual)
        self.row_count = 7 * self.num_coarse + 6 * (
            self.num_fine + self.num_prior
        )
        self._build_topology(coarse_offsets, frame_offsets)
        self.calls = 0
        self.wall_seconds = 0.0
        self.preparation_wall_seconds = 0.0
        self.derivative_wall_seconds = 0.0
        self.assembly_wall_seconds = 0.0
        self.last_nnz = len(self.rows)
        self.residual_calls = 0
        self.residual_wall_seconds = 0.0

    def _build_component_exact_topology(
        self,
        coarse_offsets: dict[int, int],
        frame_offsets: dict[int, int],
    ) -> None:
        row_parts: list[np.ndarray] = []
        column_parts: list[np.ndarray] = []
        kind_parts: list[np.ndarray] = []
        item_parts: list[np.ndarray] = []
        local_row_parts: list[np.ndarray] = []
        local_column_parts: list[np.ndarray] = []

        def append_block(
            kind: int,
            row_starts: np.ndarray,
            column_starts: np.ndarray,
            local_row_start: int,
            local_column_start: int,
            global_column_start: int,
            row_width: int,
            column_width: int,
        ) -> None:
            valid = column_starts >= 0
            if not bool(np.any(valid)):
                return
            item_indices = np.flatnonzero(valid)
            local_rows = local_row_start + np.arange(row_width, dtype=np.int64)
            local_columns = local_column_start + np.arange(
                column_width, dtype=np.int64
            )
            rows = row_starts[valid, None, None] + local_row_start + np.arange(
                row_width, dtype=np.int64
            )[None, :, None]
            columns = (
                column_starts[valid, None, None]
                + global_column_start
                + np.arange(column_width, dtype=np.int64)[None, None, :]
            )
            shape = (len(item_indices), row_width, column_width)
            row_parts.append(np.broadcast_to(rows, shape).reshape(-1))
            column_parts.append(np.broadcast_to(columns, shape).reshape(-1))
            kind_parts.append(np.full(np.prod(shape), kind, dtype=np.int8))
            item_parts.append(
                np.broadcast_to(item_indices[:, None, None], shape).reshape(-1)
            )
            local_row_parts.append(
                np.broadcast_to(local_rows[None, :, None], shape).reshape(-1)
            )
            local_column_parts.append(
                np.broadcast_to(local_columns[None, None, :], shape).reshape(-1)
            )

        coarse_rows = 7 * np.arange(self.num_coarse, dtype=np.int64)
        coarse_source_offsets = np.asarray(
            [coarse_offsets.get(int(item), -1) for item in self.sources],
            dtype=np.int64,
        )
        coarse_target_offsets = np.asarray(
            [coarse_offsets.get(int(item), -1) for item in self.targets],
            dtype=np.int64,
        )
        for local_row, local_column, width in ((0, 0, 3), (3, 3, 3)):
            append_block(
                self._KIND_COARSE,
                coarse_rows,
                coarse_source_offsets,
                local_row,
                local_column,
                local_column,
                3,
                width,
            )
        append_block(
            self._KIND_COARSE,
            coarse_rows,
            coarse_source_offsets,
            6,
            6,
            6,
            1,
            1,
        )
        for local_row, local_column, global_column, row_width, column_width in (
            (0, 7, 0, 3, 3),
            (3, 7, 0, 3, 3),
            (3, 10, 3, 3, 3),
            (3, 13, 6, 3, 1),
            (6, 13, 6, 1, 1),
        ):
            append_block(
                self._KIND_COARSE,
                coarse_rows,
                coarse_target_offsets,
                local_row,
                local_column,
                global_column,
                row_width,
                column_width,
            )

        fine_rows = 7 * self.num_coarse + 6 * np.arange(
            self.num_fine, dtype=np.int64
        )
        fine_coarse_offsets = np.asarray(
            [coarse_offsets.get(int(item), -1) for item in self.fine_chunks],
            dtype=np.int64,
        )
        fine_a_offsets = np.asarray(
            [frame_offsets.get(int(item), -1) for item in self.fine_a],
            dtype=np.int64,
        )
        fine_b_offsets = np.asarray(
            [frame_offsets.get(int(item), -1) for item in self.fine_b],
            dtype=np.int64,
        )
        append_block(
            self._KIND_FINE,
            fine_rows,
            fine_coarse_offsets,
            3,
            6,
            6,
            3,
            1,
        )
        for offsets, blocks in (
            (fine_a_offsets, ((0, 7, 0, 3, 3), (3, 10, 3, 3, 3))),
            (
                fine_b_offsets,
                (
                    (0, 13, 0, 3, 3),
                    (3, 13, 0, 3, 3),
                    (3, 16, 3, 3, 3),
                ),
            ),
        ):
            for (
                local_row,
                local_column,
                global_column,
                row_width,
                column_width,
            ) in blocks:
                append_block(
                    self._KIND_FINE,
                    fine_rows,
                    offsets,
                    local_row,
                    local_column,
                    global_column,
                    row_width,
                    column_width,
                )

        prior_rows = (
            7 * self.num_coarse
            + 6 * self.num_fine
            + 6 * np.arange(self.num_prior, dtype=np.int64)
        )
        prior_coarse_offsets = np.asarray(
            [coarse_offsets.get(int(item), -1) for item in self.prior_chunks],
            dtype=np.int64,
        )
        prior_frame_offsets = np.asarray(
            [frame_offsets.get(int(item), -1) for item in self.prior_frame],
            dtype=np.int64,
        )
        for local_row, local_column, global_column, row_width, column_width in (
            (0, 0, 0, 3, 3),
            (3, 0, 0, 3, 3),
            (3, 3, 3, 3, 3),
            (3, 6, 6, 3, 1),
        ):
            append_block(
                self._KIND_PRIOR,
                prior_rows,
                prior_coarse_offsets,
                local_row,
                local_column,
                global_column,
                row_width,
                column_width,
            )
        for local_row, local_column, global_column in ((0, 7, 0), (3, 10, 3)):
            append_block(
                self._KIND_PRIOR,
                prior_rows,
                prior_frame_offsets,
                local_row,
                local_column,
                global_column,
                3,
                3,
            )

        self.rows = np.concatenate(row_parts) if row_parts else np.zeros(0, np.int64)
        self.columns = (
            np.concatenate(column_parts) if column_parts else np.zeros(0, np.int64)
        )
        self.kind_codes = (
            np.concatenate(kind_parts) if kind_parts else np.zeros(0, np.int8)
        )
        self.item_indices = (
            np.concatenate(item_parts) if item_parts else np.zeros(0, np.int64)
        )
        self.local_rows = (
            np.concatenate(local_row_parts)
            if local_row_parts
            else np.zeros(0, np.int64)
        )
        self.local_columns = (
            np.concatenate(local_column_parts)
            if local_column_parts
            else np.zeros(0, np.int64)
        )
        from scipy.sparse import coo_matrix

        topology = coo_matrix(
            (
                np.arange(len(self.rows), dtype=np.int64),
                (self.rows, self.columns),
            ),
            shape=(self.row_count, self.num_variables),
        ).tocsr()
        if topology.nnz != len(self.rows):
            raise ValueError("dynamic closed-form topology has duplicates")
        self.csr_value_order = np.asarray(topology.data, dtype=np.intp)
        self.csr_indices = topology.indices.copy()
        self.csr_indptr = topology.indptr.copy()

    @staticmethod
    def _pose_components(
        poses: Sequence[np.ndarray],
        *,
        projection_mode: str = "strict",
    ) -> tuple[np.ndarray, np.ndarray]:
        if projection_mode not in {"strict", "scalar_svd"}:
            raise ValueError("projection_mode_invalid")
        if not poses:
            return (
                np.zeros((0, 3, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64),
            )
        rotations = []
        translations = []
        for pose in poses:
            values = np.asarray(pose, dtype=np.float64)
            if values.shape[-2:] not in ((3, 4), (4, 4)):
                raise ValueError("dynamic analytic Jacobian c2w shape mismatch")
            rotation = values[:3, :3]
            translation = values[:3, 3]
            if not np.all(np.isfinite(rotation)) or not np.all(
                np.isfinite(translation)
            ):
                raise ValueError("local_pose_nonfinite")
            if projection_mode == "scalar_svd":
                u, _, vt = np.linalg.svd(rotation)
                rotation = u @ vt
                if np.linalg.det(rotation) < 0.0:
                    u[:, -1] *= -1.0
                    rotation = u @ vt
            rotations.append(rotation)
            translations.append(translation)
        rotation_array = np.stack(rotations, axis=0)
        translation_array = np.stack(translations, axis=0)
        if not np.all(np.isfinite(rotation_array)) or not np.all(
            np.isfinite(translation_array)
        ):
            raise ValueError("dynamic analytic Jacobian local poses are non-finite")
        if projection_mode == "scalar_svd":
            return rotation_array, translation_array
        identity = np.eye(3, dtype=np.float64)
        gram = np.swapaxes(rotation_array, -1, -2) @ rotation_array
        det = np.linalg.det(rotation_array)
        if not np.allclose(gram, identity, rtol=0.0, atol=1e-9) or not np.allclose(
            det, 1.0, rtol=0.0, atol=1e-9
        ):
            raise ValueError("local_rotation_invalid")
        return rotation_array, translation_array

    def _coarse_residual(
        self,
        parameters: Any,
        measured_rotation: Any,
        measured_translation: Any,
        measured_log_scale: Any,
        row_scale: Any,
    ) -> Any:
        import torch

        source = parameters[:7]
        target = parameters[7:]
        source_rotation = _torch_rotvec_to_matrix(source[:3])
        target_rotation = _torch_rotvec_to_matrix(target[:3])
        target_rotation_inverse = target_rotation.transpose(-1, -2)
        predicted_rotation = target_rotation_inverse @ source_rotation
        predicted_translation = torch.exp(-target[6]) * (
            target_rotation_inverse @ (source[3:6] - target[3:6])
        )
        error_rotation = measured_rotation @ predicted_rotation
        error_translation = (
            torch.exp(measured_log_scale) * (measured_rotation @ predicted_translation)
            + measured_translation
        )
        return row_scale * torch.cat(
            (
                _torch_matrix_to_rotvec(error_rotation),
                error_translation,
                (measured_log_scale + source[6] - target[6]).reshape(1),
            )
        )

    def _build_topology(
        self,
        coarse_offsets: dict[int, int],
        frame_offsets: dict[int, int],
    ) -> None:
        row_parts: list[np.ndarray] = []
        column_parts: list[np.ndarray] = []
        kind_parts: list[np.ndarray] = []
        item_parts: list[np.ndarray] = []
        local_row_parts: list[np.ndarray] = []
        local_column_parts: list[np.ndarray] = []

        def append(
            kind: int,
            item_count: int,
            row_starts: np.ndarray,
            column_starts: np.ndarray,
            local_column_start: int,
            row_width: int,
            column_width: int,
        ) -> None:
            if item_count == 0:
                return
            valid = column_starts >= 0
            if not bool(np.any(valid)):
                return
            item_indices = np.flatnonzero(valid)
            local_rows = np.arange(row_width, dtype=np.int64)
            local_columns = (
                local_column_start + np.arange(column_width, dtype=np.int64)
            )
            rows = row_starts[valid, None, None] + local_rows[None, :, None]
            columns = (
                column_starts[valid, None, None] + np.arange(column_width)[None, None, :]
            )
            shape = (len(item_indices), row_width, column_width)
            row_parts.append(np.broadcast_to(rows, shape).reshape(-1))
            column_parts.append(np.broadcast_to(columns, shape).reshape(-1))
            kind_parts.append(np.full(np.prod(shape), kind, dtype=np.int8))
            item_parts.append(
                np.broadcast_to(item_indices[:, None, None], shape).reshape(-1)
            )
            local_row_parts.append(
                np.broadcast_to(local_rows[None, :, None], shape).reshape(-1)
            )
            local_column_parts.append(
                np.broadcast_to(local_columns[None, None, :], shape).reshape(-1)
            )

        coarse_rows = 7 * np.arange(self.num_coarse, dtype=np.int64)
        source_offsets = np.asarray(
            [coarse_offsets.get(int(source), -1) for source in self.sources],
            dtype=np.int64,
        )
        target_offsets = np.asarray(
            [coarse_offsets.get(int(target), -1) for target in self.targets],
            dtype=np.int64,
        )
        append(
            self._KIND_COARSE, self.num_coarse, coarse_rows, source_offsets, 0, 7, 7
        )
        append(
            self._KIND_COARSE, self.num_coarse, coarse_rows, target_offsets, 7, 7, 7
        )

        fine_rows = 7 * self.num_coarse + 6 * np.arange(
            self.num_fine, dtype=np.int64
        )
        fine_coarse_offsets = np.asarray(
            [coarse_offsets.get(int(chunk), -1) for chunk in self.fine_chunks],
            dtype=np.int64,
        )
        fine_a_offsets = np.asarray(
            [frame_offsets.get(int(frame), -1) for frame in self.fine_a],
            dtype=np.int64,
        )
        fine_b_offsets = np.asarray(
            [frame_offsets.get(int(frame), -1) for frame in self.fine_b],
            dtype=np.int64,
        )
        append(
            self._KIND_FINE, self.num_fine, fine_rows, fine_coarse_offsets, 0, 6, 7
        )
        append(
            self._KIND_FINE, self.num_fine, fine_rows, fine_a_offsets, 7, 6, 6
        )
        append(
            self._KIND_FINE, self.num_fine, fine_rows, fine_b_offsets, 13, 6, 6
        )

        prior_rows = (
            7 * self.num_coarse
            + 6 * self.num_fine
            + 6 * np.arange(self.num_prior, dtype=np.int64)
        )
        prior_coarse_offsets = np.asarray(
            [coarse_offsets.get(int(chunk), -1) for chunk in self.prior_chunks],
            dtype=np.int64,
        )
        prior_frame_offsets = np.asarray(
            [frame_offsets.get(int(frame), -1) for frame in self.prior_frame],
            dtype=np.int64,
        )
        append(
            self._KIND_PRIOR,
            self.num_prior,
            prior_rows,
            prior_coarse_offsets,
            0,
            6,
            7,
        )
        append(
            self._KIND_PRIOR,
            self.num_prior,
            prior_rows,
            prior_frame_offsets,
            7,
            6,
            6,
        )

        if row_parts:
            self.rows = np.concatenate(row_parts)
            self.columns = np.concatenate(column_parts)
            self.kind_codes = np.concatenate(kind_parts)
            self.item_indices = np.concatenate(item_parts)
            self.local_rows = np.concatenate(local_row_parts)
            self.local_columns = np.concatenate(local_column_parts)
        else:
            self.rows = np.zeros(0, dtype=np.int64)
            self.columns = np.zeros(0, dtype=np.int64)
            self.kind_codes = np.zeros(0, dtype=np.int8)
            self.item_indices = np.zeros(0, dtype=np.int64)
            self.local_rows = np.zeros(0, dtype=np.int64)
            self.local_columns = np.zeros(0, dtype=np.int64)

        from scipy.sparse import coo_matrix

        topology = coo_matrix(
            (
                np.arange(len(self.rows), dtype=np.int64),
                (self.rows, self.columns),
            ),
            shape=(self.row_count, self.num_variables),
        ).tocsr()
        if topology.nnz != len(self.rows):
            raise ValueError("dynamic analytic Jacobian topology has duplicates")
        self.csr_value_order = np.asarray(topology.data, dtype=np.intp)
        self.csr_indices = topology.indices.copy()
        self.csr_indptr = topology.indptr.copy()

    def restore_threads(self) -> None:
        if (
            self.threads_before > 0
            and self.threads_actual != self.threads_before
        ):
            self.torch.set_num_threads(self.threads_before)

    def activate_threads(self) -> None:
        if (
            self.threads_before > 0
            and self.threads_actual != self.threads_before
            and int(self.torch.get_num_threads()) != self.threads_actual
        ):
            self.torch.set_num_threads(self.threads_actual)

    def __call__(self, vector: np.ndarray) -> Any:
        self.activate_threads()
        started = perf_counter()
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (self.num_variables,):
            raise ValueError("dynamic analytic Jacobian vector shape mismatch")
        free_coarse = self.torch.as_tensor(
            values[: self.frame_base].reshape(max(self.num_chunks - 1, 0), 7),
            dtype=self.torch.float64,
        )
        free_frames = self.torch.as_tensor(
            values[self.frame_base :].reshape(max(self.num_frames - 1, 0), 6),
            dtype=self.torch.float64,
        )
        coarse_nodes = self.torch.cat(
            (self.fixed_root_parameters.reshape(1, 7), free_coarse), dim=0
        )
        frame_nodes = self.torch.cat(
            (self.fixed_frame_parameters.reshape(1, 6), free_frames), dim=0
        )
        prepared = perf_counter()

        if self.num_coarse:
            coarse_parameters = self.torch.cat(
                (
                    coarse_nodes[self.sources],
                    coarse_nodes[self.targets],
                ),
                dim=1,
            )
            coarse_local = self.coarse_jacobian(
                coarse_parameters,
                self.measured_inverse_rot,
                self.measured_inverse_trans,
                self.measured_inverse_log_scale,
                self.coarse_row_scale,
            ).detach().numpy()
        else:
            coarse_local = np.zeros((0, 7, 14), dtype=np.float64)
        if self.num_fine:
            fine_parameters = self.torch.cat(
                (
                    coarse_nodes[self.fine_chunks],
                    frame_nodes[self.fine_a],
                    frame_nodes[self.fine_b],
                ),
                dim=1,
            )
            fine_local = self.fine_jacobian(
                fine_parameters,
                self.fine_a_rot,
                self.fine_a_trans,
                self.fine_b_rot,
                self.fine_b_trans,
                self.fine_row_scale.expand(self.num_fine, 6),
            ).detach().numpy()
        else:
            fine_local = np.zeros((0, 6, 19), dtype=np.float64)
        if self.num_prior:
            prior_parameters = self.torch.cat(
                (
                    coarse_nodes[self.prior_chunks],
                    frame_nodes[self.prior_frame],
                ),
                dim=1,
            )
            prior_local = self.prior_jacobian(
                prior_parameters,
                self.prior_rot,
                self.prior_trans,
                self.prior_row_scale.expand(self.num_prior, 6),
            ).detach().numpy()
        else:
            prior_local = np.zeros((0, 6, 13), dtype=np.float64)
        differentiated = perf_counter()

        flat_values = np.empty(len(self.rows), dtype=np.float64)
        for kind, source in (
            (self._KIND_COARSE, coarse_local),
            (self._KIND_FINE, fine_local),
            (self._KIND_PRIOR, prior_local),
        ):
            mask = self.kind_codes == kind
            if bool(np.any(mask)):
                flat_values[mask] = source[
                    self.item_indices[mask],
                    self.local_rows[mask],
                    self.local_columns[mask],
                ]
        from scipy.sparse import csr_matrix

        jacobian = csr_matrix(
            (
                flat_values[self.csr_value_order],
                self.csr_indices,
                self.csr_indptr,
            ),
            shape=(self.row_count, self.num_variables),
            copy=False,
        )
        finished = perf_counter()
        self.calls += 1
        self.wall_seconds += finished - started
        self.preparation_wall_seconds += prepared - started
        self.derivative_wall_seconds += differentiated - prepared
        self.assembly_wall_seconds += finished - differentiated
        return jacobian

    def residual(self, vector: np.ndarray) -> np.ndarray:
        """Evaluate the dynamic residual in batched Torch blocks."""

        self.activate_threads()
        started = perf_counter()
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (self.num_variables,):
            raise ValueError("dynamic analytic residual vector shape mismatch")
        free_coarse = self.torch.as_tensor(
            values[: self.frame_base].reshape(max(self.num_chunks - 1, 0), 7),
            dtype=self.torch.float64,
        )
        free_frames = self.torch.as_tensor(
            values[self.frame_base :].reshape(max(self.num_frames - 1, 0), 6),
            dtype=self.torch.float64,
        )
        coarse_nodes = self.torch.cat(
            (self.fixed_root_parameters.reshape(1, 7), free_coarse), dim=0
        )
        frame_nodes = self.torch.cat(
            (self.fixed_frame_parameters.reshape(1, 6), free_frames), dim=0
        )
        chunks = []
        if self.num_coarse:
            parameters = self.torch.cat(
                (coarse_nodes[self.sources], coarse_nodes[self.targets]), dim=1
            )
            chunks.append(
                self.coarse_residual(
                    parameters,
                    self.measured_inverse_rot,
                    self.measured_inverse_trans,
                    self.measured_inverse_log_scale,
                    self.coarse_row_scale,
                ).reshape(-1)
            )
        if self.num_fine:
            parameters = self.torch.cat(
                (
                    coarse_nodes[self.fine_chunks],
                    frame_nodes[self.fine_a],
                    frame_nodes[self.fine_b],
                ),
                dim=1,
            )
            chunks.append(
                self.fine_residual(
                    parameters,
                    self.fine_a_rot,
                    self.fine_a_trans,
                    self.fine_b_rot,
                    self.fine_b_trans,
                    self.fine_row_scale.expand(self.num_fine, 6),
                ).reshape(-1)
            )
        if self.num_prior:
            parameters = self.torch.cat(
                (coarse_nodes[self.prior_chunks], frame_nodes[self.prior_frame]),
                dim=1,
            )
            chunks.append(
                self.prior_residual(
                    parameters,
                    self.prior_rot,
                    self.prior_trans,
                    self.prior_row_scale.expand(self.num_prior, 6),
                ).reshape(-1)
            )
        result = (
            self.torch.cat(chunks).detach().numpy()
            if chunks
            else np.zeros(0, dtype=np.float64)
        )
        self.residual_calls += 1
        self.residual_wall_seconds += perf_counter() - started
        return result

    def diagnostics(self) -> dict[str, Any]:
        return {
            "analytic_dynamic_jacobian_enabled": True,
            "analytic_dynamic_jacobian_fallback_reason": None,
            "analytic_dynamic_jacobian_mode": (
                "dynamic_joint_cf_torch_local_blocks_scalar_svd_projected"
                if self.local_rotation_projection == "scalar_svd"
                else "dynamic_joint_cf_torch_local_blocks"
            ),
            "analytic_dynamic_jacobian_calls": int(self.calls),
            "analytic_dynamic_jacobian_wall_seconds": float(self.wall_seconds),
            "analytic_dynamic_jacobian_preparation_wall_seconds": float(
                self.preparation_wall_seconds
            ),
            "analytic_dynamic_jacobian_derivative_wall_seconds": float(
                self.derivative_wall_seconds
            ),
            "analytic_dynamic_jacobian_assembly_wall_seconds": float(
                self.assembly_wall_seconds
            ),
            "analytic_dynamic_jacobian_threads_requested": int(
                self.threads_requested
            ),
            "analytic_dynamic_jacobian_threads_actual": int(self.threads_actual),
            "analytic_dynamic_jacobian_backend": self.backend,
            "analytic_dynamic_jacobian_local_rotation_projection": (
                self.local_rotation_projection
            ),
            "analytic_dynamic_jacobian_csr_topology_precomputed": True,
            "analytic_dynamic_jacobian_root_columns_omitted": True,
            "analytic_dynamic_jacobian_component_exact_topology": False,
            "analytic_dynamic_jacobian_local_rotation_requirement": (
                "preprojected_so3_local_c2w"
            ),
            "analytic_dynamic_vectorized_residual_calls": int(
                self.residual_calls
            ),
            "analytic_dynamic_vectorized_residual_wall_seconds": float(
                self.residual_wall_seconds
            ),
        }


class _DynamicClosedFormAnalyticJacobian(_DynamicTorchAnalyticJacobian):
    """NumPy closed-form local-block Jacobian for dynamic Joint-CF solves."""

    def __init__(
        self,
        *,
        num_chunks: int,
        num_frames: int,
        fixed_root: Sim3,
        fixed_frame: Sim3,
        coarse_edges: Sequence[Sim3Edge],
        fine_edges: Sequence[tuple[int, int, int, np.ndarray, np.ndarray]],
        prior_records: Sequence[tuple[int, int, np.ndarray]],
        coarse_offsets: dict[int, int],
        frame_offsets: dict[int, int],
        fine_edge_weight: float,
        coarse_prior_weight: float,
        translation_scale: float,
        local_rotation_projection: str = "strict",
        threads: int = 1,
    ) -> None:
        self.backend = "closed_form"
        self.local_rotation_projection = str(local_rotation_projection)
        self.threads_requested = int(threads)
        self.threads_actual = 1
        self.num_chunks = int(num_chunks)
        self.num_frames = int(num_frames)
        self.frame_base = 7 * max(self.num_chunks - 1, 0)
        self.num_variables = self.frame_base + 6 * max(self.num_frames - 1, 0)
        expected_coarse_offsets = {
            index: 7 * (index - 1) for index in range(1, self.num_chunks)
        }
        expected_frame_offsets = {
            frame: self.frame_base + 6 * (frame - 1)
            for frame in range(1, self.num_frames)
        }
        if dict(coarse_offsets) != expected_coarse_offsets:
            raise ValueError("dynamic analytic Jacobian coarse layout mismatch")
        if dict(frame_offsets) != expected_frame_offsets:
            raise ValueError("dynamic analytic Jacobian frame layout mismatch")

        self.fixed_root_parameters = _sim3_to_parameters(fixed_root)
        self.fixed_frame_parameters = _se3_to_parameters(fixed_frame)
        self.sources = np.asarray(
            [edge.source for edge in coarse_edges], dtype=np.int64
        )
        self.targets = np.asarray(
            [edge.target for edge in coarse_edges], dtype=np.int64
        )
        self.num_coarse = len(self.sources)
        measured_inverse = [edge.transform.inverse() for edge in coarse_edges]
        self.measured_inverse_rot = (
            np.stack([item.rotation for item in measured_inverse], axis=0)
            if measured_inverse
            else np.zeros((0, 3, 3), dtype=np.float64)
        )
        self.measured_inverse_trans = (
            np.stack([item.translation for item in measured_inverse], axis=0)
            if measured_inverse
            else np.zeros((0, 3), dtype=np.float64)
        )
        self.measured_inverse_log_scale = (
            np.log([abs(item.scale) for item in measured_inverse])
            if measured_inverse
            else np.zeros(0, dtype=np.float64)
        )
        coarse_sqrt_weights = (
            np.sqrt(
                [
                    float(edge.diagnostics.get("weight", 1.0))
                    for edge in coarse_edges
                ]
            )
            if coarse_edges
            else np.zeros(0, dtype=np.float64)
        )
        self.coarse_row_scale = np.stack(
            (
                coarse_sqrt_weights,
                coarse_sqrt_weights,
                coarse_sqrt_weights,
                coarse_sqrt_weights * float(translation_scale),
                coarse_sqrt_weights * float(translation_scale),
                coarse_sqrt_weights * float(translation_scale),
                coarse_sqrt_weights,
            ),
            axis=1,
        )

        self.fine_chunks = np.asarray(
            [item[0] for item in fine_edges], dtype=np.int64
        )
        self.fine_a = np.asarray([item[1] for item in fine_edges], dtype=np.int64)
        self.fine_b = np.asarray([item[2] for item in fine_edges], dtype=np.int64)
        self.num_fine = len(self.fine_a)
        self.fine_a_rot, self.fine_a_trans = self._pose_components(
            [item[3] for item in fine_edges],
            projection_mode=self.local_rotation_projection,
        )
        self.fine_b_rot, self.fine_b_trans = self._pose_components(
            [item[4] for item in fine_edges],
            projection_mode=self.local_rotation_projection,
        )
        self.fine_measured_rot = (
            np.swapaxes(self.fine_b_rot, -1, -2) @ self.fine_a_rot
        )
        self.fine_local_translation = (
            np.swapaxes(self.fine_b_rot, -1, -2)
            @ (self.fine_a_trans - self.fine_b_trans)[..., None]
        )[..., 0]
        self.fine_row_scale = np.asarray(
            [
                np.sqrt(fine_edge_weight),
                np.sqrt(fine_edge_weight),
                np.sqrt(fine_edge_weight),
                np.sqrt(fine_edge_weight) * float(translation_scale),
                np.sqrt(fine_edge_weight) * float(translation_scale),
                np.sqrt(fine_edge_weight) * float(translation_scale),
            ],
            dtype=np.float64,
        )

        self.prior_chunks = np.asarray(
            [item[0] for item in prior_records], dtype=np.int64
        )
        self.prior_frame = np.asarray(
            [item[1] for item in prior_records], dtype=np.int64
        )
        self.num_prior = len(self.prior_frame)
        self.prior_rot, self.prior_trans = self._pose_components(
            [item[2] for item in prior_records],
            projection_mode=self.local_rotation_projection,
        )
        self.prior_row_scale = np.asarray(
            [
                np.sqrt(coarse_prior_weight),
                np.sqrt(coarse_prior_weight),
                np.sqrt(coarse_prior_weight),
                np.sqrt(coarse_prior_weight) * float(translation_scale),
                np.sqrt(coarse_prior_weight) * float(translation_scale),
                np.sqrt(coarse_prior_weight) * float(translation_scale),
            ],
            dtype=np.float64,
        )

        self.row_count = 7 * self.num_coarse + 6 * (
            self.num_fine + self.num_prior
        )
        self._build_component_exact_topology(coarse_offsets, frame_offsets)
        self.calls = 0
        self.wall_seconds = 0.0
        self.preparation_wall_seconds = 0.0
        self.derivative_wall_seconds = 0.0
        self.assembly_wall_seconds = 0.0
        self.last_nnz = len(self.rows)
        self.residual_calls = 0
        self.residual_wall_seconds = 0.0

    def restore_threads(self) -> None:
        return None

    def activate_threads(self) -> None:
        return None

    def _nodes(self, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (self.num_variables,):
            raise ValueError("dynamic analytic Jacobian vector shape mismatch")
        coarse_nodes = np.concatenate(
            (
                self.fixed_root_parameters.reshape(1, 7),
                values[: self.frame_base].reshape(max(self.num_chunks - 1, 0), 7),
            ),
            axis=0,
        )
        frame_nodes = np.concatenate(
            (
                self.fixed_frame_parameters.reshape(1, 6),
                values[self.frame_base :].reshape(max(self.num_frames - 1, 0), 6),
            ),
            axis=0,
        )
        return coarse_nodes, frame_nodes

    def __call__(self, vector: np.ndarray) -> Any:
        started = perf_counter()
        coarse_nodes, frame_nodes = self._nodes(vector)
        coarse_rot = _numpy_rotvec_to_matrix(coarse_nodes[:, :3])
        coarse_right = _numpy_so3_right_jacobian(coarse_nodes[:, :3])
        frame_rot = _numpy_rotvec_to_matrix(frame_nodes[:, :3])
        prepared = perf_counter()

        coarse_local = np.zeros((self.num_coarse, 7, 14), dtype=np.float64)
        if self.num_coarse:
            source_rot = coarse_rot[self.sources]
            target_rot_inv = np.swapaxes(coarse_rot[self.targets], -1, -2)
            error_rot = (
                self.measured_inverse_rot @ target_rot_inv @ source_rot
            )
            error_rotvec = _numpy_matrix_to_rotvec(error_rot)
            log_right, log_left = _numpy_so3_log_inverse_jacobian_pair(
                error_rotvec
            )
            coarse_local[:, :3, :3] = (
                log_right @ coarse_right[self.sources]
            )
            coarse_local[:, :3, 7:10] = (
                log_left
                @ (
                    -self.measured_inverse_rot
                    @ coarse_right[self.targets]
                )
            )
            target_inverse_scale = np.exp(-coarse_nodes[self.targets, 6])
            target_relative = (
                target_rot_inv
                @ (
                    coarse_nodes[self.sources, 3:6]
                    - coarse_nodes[self.targets, 3:6]
                )[..., None]
            )[..., 0]
            predicted_translation = target_inverse_scale[:, None] * target_relative
            measured_linear = (
                np.exp(self.measured_inverse_log_scale)[:, None, None]
                * self.measured_inverse_rot
            )
            translation_block = (
                measured_linear
                @ (
                    target_inverse_scale[:, None, None] * target_rot_inv
                )
            )
            coarse_local[:, 3:6, 3:6] = translation_block
            coarse_local[:, 3:6, 10:13] = -translation_block
            coarse_local[:, 3:6, 7:10] = (
                measured_linear
                @ _numpy_hat(predicted_translation)
                @ coarse_right[self.targets]
            )
            coarse_local[:, 3:6, 13] = -(
                measured_linear @ predicted_translation[..., None]
            )[..., 0]
            coarse_local[:, 6, 6] = 1.0
            coarse_local[:, 6, 13] = -1.0
            coarse_local *= self.coarse_row_scale[..., None]

        fine_local = np.zeros((self.num_fine, 6, 19), dtype=np.float64)
        prior_local = np.zeros((self.num_prior, 6, 13), dtype=np.float64)
        if self.num_fine or self.num_prior:
            blocks = _closed_form_fine_prior_jacobian_blocks_from_frames(
                frame_nodes,
                fine_a=self.fine_a,
                fine_b=self.fine_b,
                fine_measured_rot=self.fine_measured_rot,
                prior_frame=self.prior_frame,
                prior_rot=(
                    coarse_rot[self.prior_chunks] @ self.prior_rot
                    if self.num_prior
                    else np.zeros((0, 3, 3), dtype=np.float64)
                ),
            )
            if self.num_fine:
                fine_local[:, :3, 7:10] = blocks[0]
                fine_local[:, :3, 13:16] = blocks[1]
                fine_local[:, 3:6, 13:16] = blocks[2]
                fine_local[:, 3:6, 10:13] = blocks[3]
                fine_local[:, 3:6, 16:19] = blocks[4]
                measured_translation = (
                    np.exp(coarse_nodes[self.fine_chunks, 6])[:, None]
                    * self.fine_local_translation
                )
                fine_local[:, 3:6, 6] = -(
                    np.swapaxes(self.fine_measured_rot, -1, -2)
                    @ measured_translation[..., None]
                )[..., 0]
                fine_local *= self.fine_row_scale[None, :, None]
            if self.num_prior:
                prior_local[:, :3, 7:10] = blocks[5]
                prior_local[:, 3:6, 10:13] = blocks[6]
                prior_rotation = coarse_rot[self.prior_chunks] @ self.prior_rot
                error_rotvec = _numpy_matrix_to_rotvec(
                    np.swapaxes(prior_rotation, -1, -2)
                    @ frame_rot[self.prior_frame]
                )
                prior_local[:, :3, :3] = (
                    _numpy_so3_log_inverse_jacobian(error_rotvec, right=False)
                    @ (
                        -np.swapaxes(self.prior_rot, -1, -2)
                        @ coarse_right[self.prior_chunks]
                    )
                )
                coarse_relative = (
                    np.swapaxes(coarse_rot[self.prior_chunks], -1, -2)
                    @ (
                        frame_nodes[self.prior_frame, 3:6]
                        - coarse_nodes[self.prior_chunks, 3:6]
                    )[..., None]
                )[..., 0]
                prior_local[:, 3:6, :3] = (
                    np.swapaxes(self.prior_rot, -1, -2)
                    @ _numpy_hat(coarse_relative)
                    @ coarse_right[self.prior_chunks]
                )
                prior_inverse = (
                    np.swapaxes(self.prior_rot, -1, -2)
                    @ np.swapaxes(coarse_rot[self.prior_chunks], -1, -2)
                )
                prior_local[:, 3:6, 3:6] = -prior_inverse
                prior_local[:, 3:6, 6] = -(
                    np.exp(coarse_nodes[self.prior_chunks, 6])[:, None]
                    * (
                        np.swapaxes(self.prior_rot, -1, -2)
                        @ self.prior_trans[..., None]
                    )[..., 0]
                )
                prior_local *= self.prior_row_scale[None, :, None]
        differentiated = perf_counter()

        flat_values = np.empty(len(self.rows), dtype=np.float64)
        for kind, source in (
            (self._KIND_COARSE, coarse_local),
            (self._KIND_FINE, fine_local),
            (self._KIND_PRIOR, prior_local),
        ):
            mask = self.kind_codes == kind
            if bool(np.any(mask)):
                flat_values[mask] = source[
                    self.item_indices[mask],
                    self.local_rows[mask],
                    self.local_columns[mask],
                ]
        from scipy.sparse import csr_matrix

        jacobian = csr_matrix(
            (
                flat_values[self.csr_value_order],
                self.csr_indices,
                self.csr_indptr,
            ),
            shape=(self.row_count, self.num_variables),
            copy=False,
        )
        finished = perf_counter()
        self.calls += 1
        self.wall_seconds += finished - started
        self.preparation_wall_seconds += prepared - started
        self.derivative_wall_seconds += differentiated - prepared
        self.assembly_wall_seconds += finished - differentiated
        return jacobian

    def residual(self, vector: np.ndarray) -> np.ndarray:
        started = perf_counter()
        coarse_nodes, frame_nodes = self._nodes(vector)
        coarse_rot = _numpy_rotvec_to_matrix(coarse_nodes[:, :3])
        frame_rot = _numpy_rotvec_to_matrix(frame_nodes[:, :3])
        chunks: list[np.ndarray] = []
        if self.num_coarse:
            target_rot_inv = np.swapaxes(coarse_rot[self.targets], -1, -2)
            predicted_rot = target_rot_inv @ coarse_rot[self.sources]
            predicted_trans = (
                np.exp(-coarse_nodes[self.targets, 6])[:, None]
                * (
                    target_rot_inv
                    @ (
                        coarse_nodes[self.sources, 3:6]
                        - coarse_nodes[self.targets, 3:6]
                    )[..., None]
                )[..., 0]
            )
            error_rot = self.measured_inverse_rot @ predicted_rot
            error_trans = (
                np.exp(self.measured_inverse_log_scale)[:, None]
                * (
                    self.measured_inverse_rot @ predicted_trans[..., None]
                )[..., 0]
                + self.measured_inverse_trans
            )
            error_scale = (
                self.measured_inverse_log_scale
                + coarse_nodes[self.sources, 6]
                - coarse_nodes[self.targets, 6]
            )
            coarse_rows = np.concatenate(
                (
                    _numpy_matrix_to_rotvec(error_rot),
                    error_trans,
                    error_scale[:, None],
                ),
                axis=1,
            )
            chunks.append((coarse_rows * self.coarse_row_scale).reshape(-1))
        if self.num_fine:
            predicted_rot = (
                np.swapaxes(frame_rot[self.fine_b], -1, -2)
                @ frame_rot[self.fine_a]
            )
            predicted_trans = (
                np.swapaxes(frame_rot[self.fine_b], -1, -2)
                @ (
                    frame_nodes[self.fine_a, 3:6]
                    - frame_nodes[self.fine_b, 3:6]
                )[..., None]
            )[..., 0]
            measured_trans = (
                np.exp(coarse_nodes[self.fine_chunks, 6])[:, None]
                * self.fine_local_translation
            )
            measured_inverse = np.swapaxes(self.fine_measured_rot, -1, -2)
            fine_rows = np.concatenate(
                (
                    _numpy_matrix_to_rotvec(measured_inverse @ predicted_rot),
                    (
                        measured_inverse
                        @ (predicted_trans - measured_trans)[..., None]
                    )[..., 0],
                ),
                axis=1,
            )
            chunks.append((fine_rows * self.fine_row_scale).reshape(-1))
        if self.num_prior:
            prior_rot = coarse_rot[self.prior_chunks] @ self.prior_rot
            prior_trans = (
                np.exp(coarse_nodes[self.prior_chunks, 6])[:, None]
                * (
                    coarse_rot[self.prior_chunks]
                    @ self.prior_trans[..., None]
                )[..., 0]
                + coarse_nodes[self.prior_chunks, 3:6]
            )
            prior_inverse = np.swapaxes(prior_rot, -1, -2)
            prior_rows = np.concatenate(
                (
                    _numpy_matrix_to_rotvec(
                        prior_inverse @ frame_rot[self.prior_frame]
                    ),
                    (
                        prior_inverse
                        @ (
                            frame_nodes[self.prior_frame, 3:6] - prior_trans
                        )[..., None]
                    )[..., 0],
                ),
                axis=1,
            )
            chunks.append((prior_rows * self.prior_row_scale).reshape(-1))
        result = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)
        self.residual_calls += 1
        self.residual_wall_seconds += perf_counter() - started
        return result

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = super().diagnostics()
        diagnostics["analytic_dynamic_jacobian_mode"] = (
            "dynamic_joint_cf_closed_form_local_blocks_scalar_svd_projected"
            if self.local_rotation_projection == "scalar_svd"
            else "dynamic_joint_cf_closed_form_local_blocks"
        )
        diagnostics["analytic_dynamic_jacobian_backend"] = self.backend
        diagnostics["analytic_dynamic_jacobian_component_exact_topology"] = True
        return diagnostics


class _FrozenVectorizedResidualCache:
    def __init__(
        self,
        *,
        num_frames: int,
        fixed_frame: Sim3,
        frozen_vectorized: dict[str, np.ndarray],
        fine_edge_weight: float,
        coarse_prior_weight: float,
        translation_scale: float,
    ) -> None:
        self.num_frames = int(num_frames)
        self.fixed_frame = fixed_frame
        self.frozen_vectorized = frozen_vectorized
        self.fine_edge_weight = float(fine_edge_weight)
        self.coarse_prior_weight = float(coarse_prior_weight)
        self.translation_scale = float(translation_scale)
        self.num_variables = 6 * max(self.num_frames - 1, 0)
        self.coarse_rows = int(frozen_vectorized["coarse"].size)
        self.num_fine = int(len(frozen_vectorized["fine_a"]))
        self.num_prior = int(len(frozen_vectorized["prior_frame"]))
        self.fine_row0 = self.coarse_rows
        self.prior_row0 = self.fine_row0 + 6 * self.num_fine
        self.row_count = self.prior_row0 + 6 * self.num_prior
        fine_a = np.asarray(frozen_vectorized["fine_a"], dtype=np.int64)
        fine_b = np.asarray(frozen_vectorized["fine_b"], dtype=np.int64)
        prior_frame = np.asarray(frozen_vectorized["prior_frame"], dtype=np.int64)
        if (
            np.any(fine_a < 0)
            or np.any(fine_a >= self.num_frames)
            or np.any(fine_b < 0)
            or np.any(fine_b >= self.num_frames)
            or np.any(prior_frame < 0)
            or np.any(prior_frame >= self.num_frames)
        ):
            raise ValueError("frozen vectorized frame index is out of range")
        self.fine_a = fine_a
        self.fine_b = fine_b
        self.prior_frame = prior_frame
        self._inside_fd_batch = False
        self._baseline_vector: np.ndarray | None = None
        self._baseline_residual: np.ndarray | None = None
        self._baseline_frame_rot: np.ndarray | None = None
        self._baseline_frame_trans: np.ndarray | None = None
        self.batches = 0
        self.vectors = 0
        self.hits = 0
        self.full_fallbacks = 0
        self.baseline_updates = 0
        self.local_rows_recomputed_total = 0
        self.local_rows_recomputed_max = 0
        self.local_rotation_rows_recomputed_total = 0
        self.last_full_fallback_reason: str | None = None
        self.selection_wall_seconds = 0.0
        self.frame_update_wall_seconds = 0.0
        self.result_copy_write_wall_seconds = 0.0
        self.fine_rows_wall_seconds = 0.0
        self.prior_rows_wall_seconds = 0.0

    def full_residual(self, vector: np.ndarray) -> np.ndarray:
        return _compute_frozen_vectorized_residual(
            vector,
            num_frames=self.num_frames,
            fixed_frame=self.fixed_frame,
            frozen_vectorized=self.frozen_vectorized,
            fine_edge_weight=self.fine_edge_weight,
            coarse_prior_weight=self.coarse_prior_weight,
            translation_scale=self.translation_scale,
        )

    def residual(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=np.float64)
        if not self._inside_fd_batch:
            result = self.full_residual(values)
            frame_values = _frozen_frame_values(
                values,
                num_frames=self.num_frames,
                fixed_frame=self.fixed_frame,
            )
            self._baseline_vector = values.copy()
            self._baseline_residual = result.copy()
            self._baseline_frame_rot = Rotation.from_rotvec(
                frame_values[:, :3]
            ).as_matrix()
            self._baseline_frame_trans = frame_values[:, 3:6].copy()
            self.baseline_updates += 1
            return result
        return self._incremental_residual(values)

    def map(self, function: Any, iterable: Any) -> list[np.ndarray]:
        self.batches += 1
        previous = self._inside_fd_batch
        self._inside_fd_batch = True
        try:
            return [function(item) for item in iterable]
        finally:
            self._inside_fd_batch = previous

    def _incremental_residual(self, vector: np.ndarray) -> np.ndarray:
        self.vectors += 1
        if self._baseline_vector is None or self._baseline_residual is None:
            return self._full_fallback(vector, "missing_baseline")
        if vector.shape != self._baseline_vector.shape:
            return self._full_fallback(vector, "shape_mismatch")
        if not np.isfinite(vector).all():
            return self._full_fallback(vector, "nonfinite_candidate")
        phase_started = perf_counter()
        changed = np.flatnonzero(vector != self._baseline_vector)
        if changed.size == 0:
            self.hits += 1
            return self._baseline_residual.copy()
        if changed[-1] >= self.num_variables:
            return self._full_fallback(vector, "changed_variable_out_of_range")
        changed_frames = np.unique(1 + changed // 6)
        changed_rotation_frames = np.unique(1 + changed[changed % 6 < 3] // 6)
        changed_translation_frames = np.unique(1 + changed[changed % 6 >= 3] // 6)
        if changed_frames.size == 0 or np.any(changed_frames >= self.num_frames):
            return self._full_fallback(vector, "changed_frame_out_of_range")

        changed_rotation_frame_mask = np.zeros(self.num_frames, dtype=bool)
        changed_rotation_frame_mask[changed_rotation_frames] = True
        changed_translation_frame_mask = np.zeros(self.num_frames, dtype=bool)
        changed_translation_frame_mask[changed_translation_frames] = True
        fine_rotation_array = np.flatnonzero(
            changed_rotation_frame_mask[self.fine_a]
            | changed_rotation_frame_mask[self.fine_b]
        )
        fine_translation_array = np.flatnonzero(
            changed_rotation_frame_mask[self.fine_b]
            | changed_translation_frame_mask[self.fine_a]
            | changed_translation_frame_mask[self.fine_b]
        )
        prior_rotation_array = np.flatnonzero(
            changed_rotation_frame_mask[self.prior_frame]
        )
        prior_translation_array = np.flatnonzero(
            changed_translation_frame_mask[self.prior_frame]
        )
        self.selection_wall_seconds += perf_counter() - phase_started
        if self._baseline_frame_rot is None or self._baseline_frame_trans is None:
            return self._full_fallback(vector, "missing_baseline_frames")

        phase_started = perf_counter()
        frame_rot = self._baseline_frame_rot.copy()
        frame_trans = self._baseline_frame_trans.copy()
        frame_values = _frozen_frame_values(
            vector,
            num_frames=self.num_frames,
            fixed_frame=self.fixed_frame,
        )
        if changed_rotation_frames.size:
            frame_rot[changed_rotation_frames] = Rotation.from_rotvec(
                frame_values[changed_rotation_frames, :3]
            ).as_matrix()
        if changed_translation_frames.size:
            frame_trans[changed_translation_frames] = frame_values[
                changed_translation_frames, 3:6
            ]
        self.frame_update_wall_seconds += perf_counter() - phase_started

        phase_started = perf_counter()
        result = self._baseline_residual.copy()
        self.result_copy_write_wall_seconds += perf_counter() - phase_started
        phase_started = perf_counter()
        fine_rotation_rows = _compute_frozen_fine_rotation_rows(
            fine_rotation_array,
            frame_rot,
            frozen_vectorized=self.frozen_vectorized,
            fine_edge_weight=self.fine_edge_weight,
        )
        fine_translation_rows = _compute_frozen_fine_translation_rows(
            fine_translation_array,
            frame_rot,
            frame_trans,
            frozen_vectorized=self.frozen_vectorized,
            fine_edge_weight=self.fine_edge_weight,
            translation_scale=self.translation_scale,
        )
        self.fine_rows_wall_seconds += perf_counter() - phase_started
        phase_started = perf_counter()
        fine_result = result[self.fine_row0 : self.prior_row0].reshape(
            self.num_fine, 6
        )
        fine_result[fine_rotation_array, :3] = fine_rotation_rows
        fine_result[fine_translation_array, 3:] = fine_translation_rows
        self.result_copy_write_wall_seconds += perf_counter() - phase_started

        phase_started = perf_counter()
        prior_rotation_rows = _compute_frozen_prior_rotation_rows(
            prior_rotation_array,
            frame_rot,
            frozen_vectorized=self.frozen_vectorized,
            coarse_prior_weight=self.coarse_prior_weight,
        )
        prior_translation_rows = _compute_frozen_prior_translation_rows(
            prior_translation_array,
            frame_trans,
            frozen_vectorized=self.frozen_vectorized,
            coarse_prior_weight=self.coarse_prior_weight,
            translation_scale=self.translation_scale,
        )
        self.prior_rows_wall_seconds += perf_counter() - phase_started
        phase_started = perf_counter()
        prior_result = result[self.prior_row0 :].reshape(self.num_prior, 6)
        prior_result[prior_rotation_array, :3] = prior_rotation_rows
        prior_result[prior_translation_array, 3:] = prior_translation_rows
        self.result_copy_write_wall_seconds += perf_counter() - phase_started

        self.hits += 1
        local_rows = int(
            np.union1d(fine_rotation_array, fine_translation_array).size
            + np.union1d(prior_rotation_array, prior_translation_array).size
        )
        local_rotation_rows = int(
            fine_rotation_array.size + prior_rotation_array.size
        )
        self.local_rows_recomputed_total += local_rows
        self.local_rotation_rows_recomputed_total += local_rotation_rows
        self.local_rows_recomputed_max = max(
            self.local_rows_recomputed_max,
            local_rows,
        )
        return result

    def _full_fallback(self, vector: np.ndarray, reason: str) -> np.ndarray:
        self.full_fallbacks += 1
        self.last_full_fallback_reason = reason
        return self.full_residual(vector)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "incremental_frozen_residual_enabled": True,
            "incremental_frozen_residual_mode": "serial_map_fd_candidates",
            "incremental_frozen_residual_fallback_reason": self.last_full_fallback_reason,
            "incremental_frozen_residual_batches": int(self.batches),
            "incremental_frozen_residual_vectors": int(self.vectors),
            "incremental_frozen_residual_hits": int(self.hits),
            "incremental_frozen_residual_full_fallbacks": int(self.full_fallbacks),
            "incremental_frozen_residual_baseline_updates": int(self.baseline_updates),
            "incremental_frozen_residual_local_rows_recomputed_total": int(
                self.local_rows_recomputed_total
            ),
            "incremental_frozen_residual_local_rows_recomputed_max": int(
                self.local_rows_recomputed_max
            ),
            "incremental_frozen_residual_local_rotation_rows_recomputed_total": int(
                self.local_rotation_rows_recomputed_total
            ),
            "incremental_frozen_residual_phase_seconds": {
                "dependency_selection": float(self.selection_wall_seconds),
                "frame_update": float(self.frame_update_wall_seconds),
                "result_copy_write": float(self.result_copy_write_wall_seconds),
                "fine_rows": float(self.fine_rows_wall_seconds),
                "prior_rows": float(self.prior_rows_wall_seconds),
            },
        }


def _coarse_edge_residual_statistics(
    coarse_residual: np.ndarray,
    edges: Sequence[Sim3Edge],
) -> dict[str, dict[str, float | int | None]]:
    """Summarize per-edge residual norms without hiding sparse loop regressions."""

    values = np.asarray(coarse_residual, dtype=np.float64).reshape(len(edges), 7)
    squared_norms = np.sum(values * values, axis=1)
    norms = np.sqrt(squared_norms)
    kinds = np.asarray([edge.kind for edge in edges], dtype=object)
    statistics: dict[str, dict[str, float | int | None]] = {}
    for family, mask in (
        ("all", np.ones(len(edges), dtype=bool)),
        ("adjacent", kinds == "adjacent"),
        ("loop", kinds == "loop"),
    ):
        selected_norms = norms[mask]
        selected_squared = squared_norms[mask]
        if selected_norms.size == 0:
            statistics[family] = {
                "count": 0,
                "sum_squared": 0.0,
                "rms": None,
                "median": None,
                "p95": None,
                "max": None,
            }
            continue
        statistics[family] = {
            "count": int(selected_norms.size),
            "sum_squared": float(np.sum(selected_squared)),
            "rms": float(np.sqrt(np.mean(selected_squared))),
            "median": float(np.median(selected_norms)),
            "p95": float(np.quantile(selected_norms, 0.95)),
            "max": float(np.max(selected_norms)),
        }
    return statistics


def _coarse_edge_statistics_ratios(
    before: dict[str, dict[str, float | int | None]],
    after: dict[str, dict[str, float | int | None]],
) -> dict[str, dict[str, float | None]]:
    def ratio(before_value: float | int | None, after_value: float | int | None) -> float | None:
        if before_value is None or after_value is None:
            return None
        before_float = float(before_value)
        after_float = float(after_value)
        if before_float > 0.0:
            return float(after_float / before_float)
        return 1.0 if after_float == 0.0 else np.inf

    return {
        family: {
            metric: ratio(before[family][metric], after[family][metric])
            for metric in ("sum_squared", "rms", "median", "p95", "max")
        }
        for family in before
    }


def _nonnegative_cost_ratio(after: float, before: float) -> float:
    """Return a stable ratio when the reference cost is numerically zero."""
    before_float = float(before)
    after_float = float(after)
    if before_float < 0.0 or after_float < 0.0:
        raise ValueError("cost ratio inputs must be nonnegative")
    # Squared graph residuals below this level are dominated by roundoff. Keep
    # larger small values intact because topology gates deliberately use them.
    numerical_zero = 1e-30
    if before_float <= numerical_zero and after_float <= numerical_zero:
        return 1.0
    return float(after_float / max(before_float, numerical_zero))


def _joint_local_residual_statistics(
    residual: np.ndarray,
    *,
    num_coarse_edges: int,
    num_fine_edges: int,
    num_frame_priors: int,
    fine_edge_weight: float,
    coarse_prior_weight: float,
    translation_scale: float,
) -> dict[str, Any]:
    """Summarize local pose residuals with rotation and translation separated."""

    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    expected_rows = (
        7 * int(num_coarse_edges)
        + 6 * int(num_fine_edges)
        + 6 * int(num_frame_priors)
    )
    if values.size != expected_rows:
        raise ValueError(
            "joint residual row count does not match coarse/fine/prior layout"
        )
    if not np.isfinite(translation_scale) or translation_scale <= 0.0:
        raise ValueError("translation_scale must be finite and positive")

    coarse_stop = 7 * int(num_coarse_edges)
    fine_stop = coarse_stop + 6 * int(num_fine_edges)
    family_slices = {
        "fine": (
            values[coarse_stop:fine_stop].reshape(int(num_fine_edges), 6),
            float(fine_edge_weight),
        ),
        "prior": (
            values[fine_stop:].reshape(int(num_frame_priors), 6),
            float(coarse_prior_weight),
        ),
    }

    def norm_statistics(vectors: np.ndarray) -> dict[str, float | int | None]:
        if vectors.shape[0] == 0:
            return {
                "count": 0,
                "sum_squared": 0.0,
                "rms": None,
                "median": None,
                "p95": None,
                "max": None,
            }
        squared_norms = np.sum(vectors * vectors, axis=1)
        norms = np.sqrt(squared_norms)
        return {
            "count": int(len(norms)),
            "sum_squared": float(np.sum(squared_norms)),
            "rms": float(np.sqrt(np.mean(squared_norms))),
            "median": float(np.median(norms)),
            "p95": float(np.quantile(norms, 0.95)),
            "max": float(np.max(norms)),
        }

    families = {}
    for name, (blocks, weight) in family_slices.items():
        unweighting_available = bool(np.isfinite(weight) and weight > 0.0)
        residual_scale = np.sqrt(weight) if unweighting_available else 1.0
        rotation = blocks[:, :3] / residual_scale
        translation = (
            blocks[:, 3:] / (residual_scale * float(translation_scale))
        )
        families[name] = {
            "constraint_count": int(len(blocks)),
            "configured_weight": weight,
            "unweighting_available": unweighting_available,
            "rotation_radians": norm_statistics(rotation),
            "translation_scene_units": norm_statistics(translation),
        }
    return {
        "schema": "hierarchical_joint_local_residual_statistics_v1",
        "interpretation": (
            "unweighted_fine_relative_pose_and_frame_prior_residual_norms_"
            "with_rotation_and_translation_reported_separately"
        ),
        "families": families,
    }


def _joint_local_residual_statistics_ratios(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Compute candidate/initial ratios for local residual statistics."""

    def ratio(before_value: float | int | None, after_value: float | int | None) -> float | None:
        if before_value is None or after_value is None:
            return None
        before_float = float(before_value)
        after_float = float(after_value)
        if before_float > 0.0:
            return float(after_float / before_float)
        return 1.0 if after_float == 0.0 else np.inf

    metrics = ("sum_squared", "rms", "median", "p95", "max")
    return {
        "schema": "hierarchical_joint_local_residual_statistics_ratios_v1",
        "comparison": "candidate_over_initial",
        "families": {
            family: {
                component: {
                    metric: ratio(
                        before["families"][family][component][metric],
                        after["families"][family][component][metric],
                    )
                    for metric in metrics
                }
                for component in (
                    "rotation_radians",
                    "translation_scene_units",
                )
            }
            for family in ("fine", "prior")
        },
    }


def _sequential_pose_mode_energy_diagnostics(
    direction: np.ndarray,
    *,
    block_dimension: int,
    metric_diagonal: np.ndarray | None = None,
    aggregate_sizes: Sequence[int] = (4, 16, 64),
    component_groups: Sequence[tuple[str, int, int]] = (),
) -> dict[str, Any]:
    """Split a tangent direction into aggregate means and zero-mean local modes.

    This is a coordinate diagnostic, not an invariant geometric quantity.  The
    optional positive diagonal metric provides a cheap local-curvature-aware
    view in addition to the raw Euclidean partition.  Absolute block-norm
    statistics are reported as well as energy fractions: a nearly pure
    low-frequency direction can still be either a material trajectory update
    or a numerically negligible correction.
    """

    vector = np.asarray(direction, dtype=np.float64).reshape(-1)
    if block_dimension < 1 or vector.size % block_dimension != 0:
        raise ValueError("direction must contain complete pose tangent blocks")
    num_blocks = vector.size // block_dimension
    blocks = vector.reshape(num_blocks, block_dimension)
    if metric_diagonal is None:
        metric_blocks = np.ones_like(blocks)
    else:
        metric = np.asarray(metric_diagonal, dtype=np.float64).reshape(-1)
        if metric.shape != vector.shape or np.any(metric <= 0.0):
            raise ValueError("metric_diagonal must be positive and match direction")
        metric_blocks = metric.reshape(num_blocks, block_dimension)

    requested_sizes = [int(size) for size in aggregate_sizes if int(size) >= 2]
    if num_blocks >= 2:
        requested_sizes.append(num_blocks)
    effective_sizes = sorted(
        {min(size, num_blocks) for size in requested_sizes if num_blocks > 0}
    )
    normalized_component_groups: list[tuple[str, slice]] = []
    for name, start, stop in component_groups:
        group_start = int(start)
        group_stop = int(stop)
        if not name or not (0 <= group_start < group_stop <= block_dimension):
            raise ValueError("component_groups must be named valid block slices")
        normalized_component_groups.append(
            (str(name), slice(group_start, group_stop))
        )

    def block_norm_statistics(
        values: np.ndarray,
        weights: np.ndarray,
    ) -> dict[str, float]:
        norms = np.sqrt(np.sum(weights * values * values, axis=1))
        return {
            "rms": float(np.sqrt(np.mean(norms * norms))),
            "median": float(np.median(norms)),
            "p95": float(np.quantile(norms, 0.95)),
            "max": float(np.max(norms)),
        }

    def partition(aggregate_size: int, weights: np.ndarray) -> dict[str, Any]:
        low_frequency = np.empty_like(blocks)
        for start in range(0, num_blocks, aggregate_size):
            stop = min(start + aggregate_size, num_blocks)
            group_weights = weights[start:stop]
            weighted_sum = np.sum(group_weights * blocks[start:stop], axis=0)
            weight_sum = np.sum(group_weights, axis=0)
            low_frequency[start:stop] = weighted_sum / weight_sum
        zero_mean_local = blocks - low_frequency
        total_energy = float(np.sum(weights * blocks * blocks))
        low_frequency_energy = float(
            np.sum(weights * low_frequency * low_frequency)
        )
        zero_mean_local_energy = float(
            np.sum(weights * zero_mean_local * zero_mean_local)
        )
        denominator = total_energy if total_energy > 0.0 else 1.0
        result: dict[str, Any] = {
            "total_energy": total_energy,
            "low_frequency_energy": low_frequency_energy,
            "zero_mean_local_energy": zero_mean_local_energy,
            "low_frequency_energy_ratio": float(
                low_frequency_energy / denominator
            ),
            "zero_mean_local_energy_ratio": float(
                zero_mean_local_energy / denominator
            ),
            "energy_reconstruction_relative_error": float(
                abs(
                    total_energy
                    - low_frequency_energy
                    - zero_mean_local_energy
                )
                / denominator
            ),
            "total_block_norm": block_norm_statistics(blocks, weights),
            "low_frequency_block_norm": block_norm_statistics(
                low_frequency, weights
            ),
            "zero_mean_local_block_norm": block_norm_statistics(
                zero_mean_local, weights
            ),
        }
        if normalized_component_groups:
            result["component_groups"] = {
                name: {
                    "total_block_norm": block_norm_statistics(
                        blocks[:, group_slice], weights[:, group_slice]
                    ),
                    "low_frequency_block_norm": block_norm_statistics(
                        low_frequency[:, group_slice],
                        weights[:, group_slice],
                    ),
                    "zero_mean_local_block_norm": block_norm_statistics(
                        zero_mean_local[:, group_slice],
                        weights[:, group_slice],
                    ),
                }
                for name, group_slice in normalized_component_groups
            }
        return result

    return {
        "schema": "hierarchical_joint_sequential_pose_mode_energy_v2",
        "interpretation": (
            "piecewise_constant_aggregate_modes_plus_zero_mean_local_modes"
        ),
        "coordinate_caveat": (
            "Local tangent-coordinate diagnostic; values are not globally "
            "SE3-invariant. Jacobi metric uses the damped frame normal-matrix "
            "diagonal at this linearization. Euclidean rotation and "
            "translation groups retain their native radians and trajectory "
            "units and should not be combined into a physical distance."
        ),
        "num_blocks": int(num_blocks),
        "block_dimension": int(block_dimension),
        "levels": [
            {
                "aggregate_size": int(aggregate_size),
                "num_aggregates": int(
                    (num_blocks + aggregate_size - 1) // aggregate_size
                ),
                "euclidean": partition(
                    aggregate_size, np.ones_like(metric_blocks)
                ),
                "jacobi_metric": partition(aggregate_size, metric_blocks),
            }
            for aggregate_size in effective_sizes
        ],
    }


def _trust_frame_low_frequency_direction(
    coarse_direction: np.ndarray,
    frame_direction: np.ndarray,
    *,
    aggregate_size: int,
    maximum_component_block_rms_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Cap only piecewise-constant frame modes relative to the coarse step."""

    coarse = np.asarray(coarse_direction, dtype=np.float64).reshape(-1, 7)
    frame = np.asarray(frame_direction, dtype=np.float64).reshape(-1, 6)
    low = np.empty_like(frame)
    for start in range(0, len(frame), int(aggregate_size)):
        stop = min(start + int(aggregate_size), len(frame))
        low[start:stop] = np.mean(frame[start:stop], axis=0, keepdims=True)
    high = frame - low

    def component_block_rms(blocks: np.ndarray, component: slice) -> float:
        if len(blocks) == 0:
            return 0.0
        return float(
            np.linalg.norm(blocks[:, component]) / np.sqrt(len(blocks))
        )

    coarse_rotation_rms = component_block_rms(coarse, slice(0, 3))
    coarse_translation_rms = component_block_rms(coarse, slice(3, 6))
    low_rotation_rms = component_block_rms(low, slice(0, 3))
    low_translation_rms = component_block_rms(low, slice(3, 6))

    def trust_scale(low_rms: float, coarse_rms: float) -> float:
        if low_rms == 0.0:
            return 1.0
        return float(
            min(
                1.0,
                maximum_component_block_rms_ratio * coarse_rms / low_rms,
            )
        )

    rotation_scale = trust_scale(low_rotation_rms, coarse_rotation_rms)
    translation_scale = trust_scale(low_translation_rms, coarse_translation_rms)
    trusted_low = low.copy()
    trusted_low[:, :3] *= rotation_scale
    trusted_low[:, 3:6] *= translation_scale
    trusted = high + trusted_low
    trusted_low_rotation_rms = component_block_rms(
        trusted_low, slice(0, 3)
    )
    trusted_low_translation_rms = component_block_rms(
        trusted_low, slice(3, 6)
    )
    return trusted.reshape(-1), {
        "schema": "hierarchical_joint_frame_low_frequency_trust_v1",
        "interpretation": (
            "piecewise_constant_frame_modes_capped_relative_to_coarse_components"
        ),
        "aggregate_size": int(aggregate_size),
        "num_aggregates": int(
            (len(frame) + int(aggregate_size) - 1) // int(aggregate_size)
        ),
        "maximum_component_block_rms_ratio": float(
            maximum_component_block_rms_ratio
        ),
        "rotation_scale": rotation_scale,
        "translation_scale": translation_scale,
        "coarse_rotation_block_rms": coarse_rotation_rms,
        "coarse_translation_block_rms": coarse_translation_rms,
        "frame_low_rotation_block_rms_before": low_rotation_rms,
        "frame_low_translation_block_rms_before": low_translation_rms,
        "frame_low_rotation_block_rms_after": trusted_low_rotation_rms,
        "frame_low_translation_block_rms_after": trusted_low_translation_rms,
        "frame_high_direction_l2": float(np.linalg.norm(high)),
        "frame_low_direction_l2_before": float(np.linalg.norm(low)),
        "frame_low_direction_l2_after": float(np.linalg.norm(trusted_low)),
        "frame_direction_l2_before": float(np.linalg.norm(frame)),
        "frame_direction_l2_after": float(np.linalg.norm(trusted)),
        "coordinate_caveat": (
            "Projection uses local rotvec and translation tangent coordinates; "
            "rotation and translation are trusted separately and remain in their "
            "native units."
        ),
    }


def _joint_diagonal_low_frequency_efficiency_from_solver_result(
    result: Any,
    *,
    frame_base: int,
    num_variables: int,
    damping: float,
    aggregate_size: int,
    freeze_coarse_scales: bool,
) -> dict[str, Any]:
    """Estimate slow-frame usefulness before Schur factorization or PCG.

    The direction is a damped diagonal-normal step, followed by an exact
    one-dimensional minimization in the full local quadratic model. It is a
    cheap selector, not a replacement for the coupled Schur direction.
    """

    from scipy import sparse

    started = perf_counter()
    jacobian = result.jac
    if not sparse.issparse(jacobian):
        jacobian = sparse.csr_matrix(np.asarray(jacobian, dtype=np.float64))
    else:
        jacobian = jacobian.tocsr()
    gradient = np.asarray(result.grad, dtype=np.float64).reshape(-1)
    diagonal = np.asarray(
        jacobian.multiply(jacobian).sum(axis=0), dtype=np.float64
    ).reshape(-1)
    diagonal_floor = max(
        float(np.max(diagonal, initial=0.0)) * 1e-12,
        1e-12,
    )
    regularizer = float(damping) * np.maximum(diagonal, diagonal_floor)
    metric_diagonal = diagonal + regularizer
    direction = -gradient / metric_diagonal
    if freeze_coarse_scales:
        direction[np.arange(6, frame_base, 7, dtype=np.int64)] = 0.0

    directional_gradient = float(np.dot(gradient, direction))
    projected_direction = jacobian @ direction
    directional_curvature = float(
        np.dot(projected_direction, projected_direction)
        + np.dot(regularizer * direction, direction)
    )
    line_alpha = (
        float(np.clip(-directional_gradient / directional_curvature, 0.0, 1.0))
        if directional_gradient < 0.0 and directional_curvature > 0.0
        else 0.0
    )
    direction *= line_alpha
    predicted_reduction = float(
        -line_alpha * directional_gradient
        - 0.5 * line_alpha * line_alpha * directional_curvature
    )
    frame_direction = direction[frame_base:num_variables]
    frame_mode_energy = _sequential_pose_mode_energy_diagnostics(
        frame_direction,
        block_dimension=6,
        metric_diagonal=metric_diagonal[frame_base:num_variables],
        aggregate_sizes=(int(aggregate_size),),
        component_groups=(("rotation", 0, 3), ("translation", 3, 6)),
    )
    target_size = min(int(aggregate_size), int(frame_mode_energy["num_blocks"]))
    selected_level = next(
        (
            level
            for level in frame_mode_energy["levels"]
            if int(level["aggregate_size"]) == target_size
        ),
        None,
    )
    low_frequency_energy = (
        None
        if selected_level is None
        else float(selected_level["jacobi_metric"]["low_frequency_energy"])
    )
    efficiency = (
        float(predicted_reduction / low_frequency_energy)
        if low_frequency_energy is not None
        and low_frequency_energy > 0.0
        and predicted_reduction > 0.0
        else None
    )
    robust_cost = float(result.cost)
    predicted_reduction_fraction = (
        float(predicted_reduction / robust_cost)
        if robust_cost > 0.0 and np.isfinite(robust_cost)
        else None
    )
    return {
        "schema": "hierarchical_joint_initial_diagonal_efficiency_v1",
        "interpretation": (
            "full_quadratic_reduction_per_frame_low_frequency_energy_for_"
            "damped_diagonal_step"
        ),
        "caveat": (
            "A pre-factorization selector in local tangent coordinates, not a "
            "Hessian eigenvalue or a trajectory-quality estimate."
        ),
        "aggregate_size": target_size,
        "efficiency": efficiency,
        "predicted_quadratic_reduction": predicted_reduction,
        "predicted_quadratic_reduction_fraction": (
            predicted_reduction_fraction
        ),
        "frame_low_frequency_energy": low_frequency_energy,
        "line_alpha": line_alpha,
        "directional_gradient": directional_gradient,
        "directional_curvature": directional_curvature,
        "coarse_direction_l2": float(np.linalg.norm(direction[:frame_base])),
        "frame_direction_l2": float(np.linalg.norm(frame_direction)),
        "phase_seconds": float(perf_counter() - started),
    }


def _joint_schur_diagnostics_from_solver_result(
    result: Any,
    *,
    frame_base: int,
    num_variables: int,
    damping: float,
    cg_maxiter: int,
    cg_rtol: float,
    two_level_aggregate_size: int | None,
    two_level_cache: _JointSchurCoarseGridCache | None = None,
    prefer_two_level_direction: bool = False,
    freeze_coarse_scales: bool = False,
    full_space_block_pcg_probe: bool = False,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    _JointSchurCoarseGridCache | None,
]:
    """Measure a damped Schur-reduced coarse direction without changing the solve."""

    from scipy import sparse
    from scipy.linalg import solve_triangular
    from scipy.sparse.linalg import LinearOperator, cg, splu

    def solve_cg(
        operator: Any,
        rhs: np.ndarray,
        *,
        preconditioner: Any,
        callback: Any,
    ) -> tuple[np.ndarray, int]:
        common = {
            "M": preconditioner,
            "atol": 0.0,
            "maxiter": int(cg_maxiter),
            "callback": callback,
        }
        try:
            return cg(operator, rhs, rtol=float(cg_rtol), **common)
        except TypeError as exc:
            if "unexpected keyword argument 'rtol'" not in str(exc):
                raise
            return cg(operator, rhs, tol=float(cg_rtol), **common)

    total_started = perf_counter()
    jacobian = result.jac
    if not sparse.issparse(jacobian):
        jacobian = sparse.csr_matrix(np.asarray(jacobian, dtype=np.float64))
    else:
        jacobian = jacobian.tocsr()
    coarse_jacobian = jacobian[:, :frame_base]
    frame_jacobian = jacobian[:, frame_base:num_variables]
    gradient = np.asarray(result.grad, dtype=np.float64).reshape(-1)
    coarse_gradient = gradient[:frame_base]
    frame_gradient = gradient[frame_base:num_variables]

    assembly_started = perf_counter()
    coarse_diagonal = np.asarray(
        coarse_jacobian.multiply(coarse_jacobian).sum(axis=0), dtype=np.float64
    ).reshape(-1)
    frame_diagonal = np.asarray(
        frame_jacobian.multiply(frame_jacobian).sum(axis=0), dtype=np.float64
    ).reshape(-1)
    coarse_floor = max(float(np.max(coarse_diagonal, initial=0.0)) * 1e-12, 1e-12)
    frame_floor = max(float(np.max(frame_diagonal, initial=0.0)) * 1e-12, 1e-12)
    coarse_regularizer = float(damping) * np.maximum(coarse_diagonal, coarse_floor)
    frame_regularizer = float(damping) * np.maximum(frame_diagonal, frame_floor)
    coarse_hessian = (
        coarse_jacobian.T @ coarse_jacobian
        + sparse.diags(coarse_regularizer, format="csc")
    ).tocsc()
    frame_hessian = (
        frame_jacobian.T @ frame_jacobian
        + sparse.diags(frame_regularizer, format="csc")
    ).tocsc()
    assembly_seconds = perf_counter() - assembly_started

    factor_started = perf_counter()
    frame_factor = splu(frame_hessian)
    frame_factor_seconds = perf_counter() - factor_started
    coarse_factor_started = perf_counter()
    coarse_factor = splu(coarse_hessian)
    coarse_scale_indices = np.arange(6, frame_base, 7, dtype=np.int64)
    coarse_free_mask = np.ones(frame_base, dtype=bool)
    if freeze_coarse_scales:
        coarse_free_mask[coarse_scale_indices] = False
    coarse_free_indices = np.flatnonzero(coarse_free_mask)
    restricted_coarse_factor = (
        splu(coarse_hessian[coarse_free_indices][:, coarse_free_indices].tocsc())
        if freeze_coarse_scales
        else None
    )
    coarse_factor_seconds = perf_counter() - coarse_factor_started
    reduced_gradient_started = perf_counter()
    frame_response = frame_factor.solve(frame_gradient)
    reduced_gradient = coarse_gradient - np.asarray(
        coarse_jacobian.T @ (frame_jacobian @ frame_response), dtype=np.float64
    ).reshape(-1)
    reduced_gradient_seconds = perf_counter() - reduced_gradient_started

    def schur_matvec(vector: np.ndarray) -> np.ndarray:
        coarse_projection = coarse_jacobian @ vector
        frame_rhs = np.asarray(
            frame_jacobian.T @ coarse_projection, dtype=np.float64
        ).reshape(-1)
        eliminated = frame_factor.solve(frame_rhs)
        coupling = np.asarray(
            coarse_jacobian.T @ (frame_jacobian @ eliminated), dtype=np.float64
        ).reshape(-1)
        return np.asarray(coarse_hessian @ vector, dtype=np.float64).reshape(-1) - coupling

    schur_operator = LinearOperator(
        (frame_base, frame_base), matvec=schur_matvec, dtype=np.float64
    )
    preconditioner = LinearOperator(
        (frame_base, frame_base),
        matvec=coarse_factor.solve,
        dtype=np.float64,
    )
    if freeze_coarse_scales:
        assert restricted_coarse_factor is not None

        def restricted_schur_matvec(vector: np.ndarray) -> np.ndarray:
            expanded = np.zeros(frame_base, dtype=np.float64)
            expanded[coarse_free_indices] = vector
            return schur_matvec(expanded)[coarse_free_indices]

        solve_operator = LinearOperator(
            (len(coarse_free_indices), len(coarse_free_indices)),
            matvec=restricted_schur_matvec,
            dtype=np.float64,
        )
        solve_preconditioner = LinearOperator(
            (len(coarse_free_indices), len(coarse_free_indices)),
            matvec=restricted_coarse_factor.solve,
            dtype=np.float64,
        )
        solve_gradient = reduced_gradient[coarse_free_indices]
    else:
        solve_operator = schur_operator
        solve_preconditioner = preconditioner
        solve_gradient = reduced_gradient
    baseline_direction = None
    baseline_info = None
    baseline_iterations = 0
    baseline_seconds = 0.0

    def count_baseline_iteration(_: np.ndarray) -> None:
        nonlocal baseline_iterations
        baseline_iterations += 1

    two_level_enabled = bool(
        two_level_aggregate_size is not None and not freeze_coarse_scales
    )
    if not prefer_two_level_direction or not two_level_enabled:
        baseline_started = perf_counter()
        baseline_solve_direction, baseline_info = solve_cg(
            solve_operator,
            -solve_gradient,
            preconditioner=solve_preconditioner,
            callback=count_baseline_iteration,
        )
        if freeze_coarse_scales:
            baseline_direction = np.zeros(frame_base, dtype=np.float64)
            baseline_direction[coarse_free_indices] = baseline_solve_direction
        else:
            baseline_direction = baseline_solve_direction
        baseline_seconds = perf_counter() - baseline_started
    schur_rhs_norm = float(np.linalg.norm(solve_gradient))
    two_level_diagnostics = None
    coarse_grid_cache_out = None
    additive_direction = None
    additive_info = None
    additive_iterations = 0
    additive_cg_seconds = 0.0
    if two_level_enabled:
        two_level_started = perf_counter()
        num_coarse_blocks = frame_base // 7
        num_aggregates = int(
            np.ceil(num_coarse_blocks / float(two_level_aggregate_size))
        )
        coarse_grid_variables = 7 * num_aggregates
        coarse_grid_reused = bool(
            two_level_cache is not None
            and two_level_cache.aggregate_size == int(two_level_aggregate_size)
            and two_level_cache.frame_base == frame_base
        )
        if coarse_grid_reused:
            assert two_level_cache is not None
            prolongation = two_level_cache.prolongation
            coarse_grid_cholesky = two_level_cache.cholesky
            coarse_grid_condition = two_level_cache.condition
            coarse_grid_cache_out = _dataclass_replace(
                two_level_cache,
                age_outer_iterations=two_level_cache.age_outer_iterations + 1,
            )
        else:
            block_indices = np.arange(num_coarse_blocks, dtype=np.int64)
            component_indices = np.arange(7, dtype=np.int64)
            prolongation_rows = (
                7 * block_indices[:, None] + component_indices[None, :]
            ).reshape(-1)
            prolongation_columns = (
                7
                * (block_indices // int(two_level_aggregate_size))[:, None]
                + component_indices[None, :]
            ).reshape(-1)
            prolongation = sparse.csr_matrix(
                (
                    np.ones_like(prolongation_rows, dtype=np.float64),
                    (prolongation_rows, prolongation_columns),
                ),
                shape=(frame_base, coarse_grid_variables),
            )
            schur_prolongation_batches = []
            for column_start in range(0, coarse_grid_variables, 32):
                prolongation_batch = prolongation[
                    :, column_start : column_start + 32
                ]
                coarse_projection = coarse_jacobian @ prolongation_batch
                frame_rhs = frame_jacobian.T @ coarse_projection
                eliminated = frame_factor.solve(frame_rhs.toarray())
                coupling = coarse_jacobian.T @ (frame_jacobian @ eliminated)
                schur_prolongation_batches.append(
                    np.asarray(coarse_hessian @ prolongation_batch - coupling)
                )
            schur_prolongation = np.hstack(schur_prolongation_batches)
            coarse_grid_matrix = np.asarray(
                prolongation.T @ schur_prolongation, dtype=np.float64
            )
            coarse_grid_matrix = 0.5 * (
                coarse_grid_matrix + coarse_grid_matrix.T
            )
            coarse_grid_floor = max(
                float(np.max(np.diag(coarse_grid_matrix), initial=0.0)) * 1e-10,
                1e-12,
            )
            coarse_grid_matrix.flat[:: coarse_grid_variables + 1] += (
                coarse_grid_floor
            )
            coarse_grid_cholesky = np.linalg.cholesky(coarse_grid_matrix)
            coarse_grid_condition = float(np.linalg.cond(coarse_grid_matrix))
            coarse_grid_cache_out = _JointSchurCoarseGridCache(
                aggregate_size=int(two_level_aggregate_size),
                frame_base=frame_base,
                prolongation=prolongation,
                cholesky=coarse_grid_cholesky,
                condition=coarse_grid_condition,
            )

        def coarse_grid_solve(vector: np.ndarray) -> np.ndarray:
            coarse_rhs = np.asarray(prolongation.T @ vector).reshape(-1)
            intermediate = solve_triangular(
                coarse_grid_cholesky,
                coarse_rhs,
                lower=True,
                check_finite=False,
            )
            coarse_solution = solve_triangular(
                coarse_grid_cholesky.T,
                intermediate,
                lower=False,
                check_finite=False,
            )
            return np.asarray(prolongation @ coarse_solution).reshape(-1)

        def additive_precondition(vector: np.ndarray) -> np.ndarray:
            return coarse_factor.solve(vector) + coarse_grid_solve(vector)

        additive_iterations = 0

        def count_additive_iteration(_: np.ndarray) -> None:
            nonlocal additive_iterations
            additive_iterations += 1

        additive_preconditioner = LinearOperator(
            (frame_base, frame_base),
            matvec=additive_precondition,
            dtype=np.float64,
        )
        additive_cg_started = perf_counter()
        additive_direction, additive_info = solve_cg(
            schur_operator,
            -reduced_gradient,
            preconditioner=additive_preconditioner,
            callback=count_additive_iteration,
        )
        additive_cg_seconds = perf_counter() - additive_cg_started
        additive_residual = schur_matvec(additive_direction) + reduced_gradient
        additive_direction_denominator = (
            float(
                np.linalg.norm(additive_direction)
                * np.linalg.norm(baseline_direction)
            )
            if baseline_direction is not None
            else 0.0
        )

        coarse_grid_setup_seconds = additive_cg_started - two_level_started
        baseline_savings_seconds = baseline_seconds - additive_cg_seconds
        two_level_diagnostics = {
            "schema": "hierarchical_joint_schur_two_level_diagnostics_v1",
            "preconditioner_mode": "additive_local_plus_coarse_grid",
            "coarse_grid_reused": coarse_grid_reused,
            "coarse_grid_cache_age_outer_iterations": int(
                coarse_grid_cache_out.age_outer_iterations
            ),
            "coarse_grid_condition_source": (
                "reused_linearization"
                if coarse_grid_reused
                else "fresh_linearization"
            ),
            "aggregate_size": int(two_level_aggregate_size),
            "num_fine_blocks": int(num_coarse_blocks),
            "num_aggregates": int(num_aggregates),
            "coarse_grid_variables": int(coarse_grid_variables),
            "prolongation_nnz": int(prolongation.nnz),
            "coarse_grid_condition": coarse_grid_condition,
            "cg_info": int(additive_info),
            "cg_converged": bool(additive_info == 0),
            "cg_iterations": int(additive_iterations),
            "cg_relative_residual": (
                float(np.linalg.norm(additive_residual) / schur_rhs_norm)
                if schur_rhs_norm > 0.0
                else 0.0
            ),
            "direction_cosine_vs_baseline": (
                float(
                    np.dot(additive_direction, baseline_direction)
                    / additive_direction_denominator
                )
                if additive_direction_denominator > 0.0
                else None
            ),
            "estimated_break_even_schur_solves": (
                float(coarse_grid_setup_seconds / baseline_savings_seconds)
                if baseline_savings_seconds > 0.0
                else None
            ),
            "phase_seconds": {
                "coarse_grid_and_prolongation": float(coarse_grid_setup_seconds),
                "cg": float(additive_cg_seconds),
                "total": float(perf_counter() - two_level_started),
            },
        }
    if prefer_two_level_direction and additive_direction is not None:
        schur_direction = additive_direction
        cg_info = additive_info
        cg_iterations = additive_iterations
        cg_seconds = additive_cg_seconds
        linear_solver = "two_level_additive_pcg"
    else:
        assert baseline_direction is not None and baseline_info is not None
        schur_direction = baseline_direction
        cg_info = baseline_info
        cg_iterations = baseline_iterations
        cg_seconds = baseline_seconds
        linear_solver = "local_preconditioned_cg"
    frame_backsub_started = perf_counter()
    frame_direction_rhs = frame_gradient + np.asarray(
        frame_jacobian.T @ (coarse_jacobian @ schur_direction),
        dtype=np.float64,
    ).reshape(-1)
    frame_direction = -frame_factor.solve(frame_direction_rhs)
    full_directional_gradient = float(
        np.dot(coarse_gradient, schur_direction)
        + np.dot(frame_gradient, frame_direction)
    )
    projected_direction = (
        coarse_jacobian @ schur_direction + frame_jacobian @ frame_direction
    )
    full_directional_curvature = float(
        np.dot(projected_direction, projected_direction)
        + np.dot(coarse_regularizer * schur_direction, schur_direction)
        + np.dot(frame_regularizer * frame_direction, frame_direction)
    )
    full_predicted_reduction = float(
        -(full_directional_gradient + 0.5 * full_directional_curvature)
    )
    frame_backsub_seconds = perf_counter() - frame_backsub_started
    fixed_started = perf_counter()
    fixed_direction = coarse_factor.solve(-coarse_gradient)
    fixed_seconds = perf_counter() - fixed_started
    coarse_block_newton_decrement_squared = float(
        -np.dot(coarse_gradient, fixed_direction)
    )
    frame_block_newton_decrement_squared = float(
        np.dot(frame_gradient, frame_response)
    )
    reduced_coarse_newton_decrement_squared = float(
        -np.dot(reduced_gradient, schur_direction)
    )

    schur_residual = (
        schur_matvec(schur_direction) + reduced_gradient
    )[coarse_free_indices]
    direction_denominator = float(
        np.linalg.norm(schur_direction) * np.linalg.norm(fixed_direction)
    )
    direction_cosine = (
        float(np.dot(schur_direction, fixed_direction) / direction_denominator)
        if direction_denominator > 0.0
        else None
    )
    schur_product = schur_matvec(schur_direction)
    predicted_reduction = float(
        -(np.dot(reduced_gradient, schur_direction)
        + 0.5 * np.dot(schur_direction, schur_product))
    )
    num_coarse_blocks = frame_base // 7
    num_frame_blocks = (num_variables - frame_base) // 6
    coarse_direction_l2 = float(np.linalg.norm(schur_direction))
    frame_direction_l2 = float(np.linalg.norm(frame_direction))
    coarse_direction_dof_rms = coarse_direction_l2 / np.sqrt(frame_base)
    frame_direction_dof_rms = frame_direction_l2 / np.sqrt(
        num_variables - frame_base
    )
    coarse_direction_block_rms = coarse_direction_l2 / np.sqrt(num_coarse_blocks)
    frame_direction_block_rms = frame_direction_l2 / np.sqrt(num_frame_blocks)
    frame_mode_energy = _sequential_pose_mode_energy_diagnostics(
        frame_direction,
        block_dimension=6,
        metric_diagonal=frame_diagonal + frame_regularizer,
        component_groups=(("rotation", 0, 3), ("translation", 3, 6)),
    )
    frame_low_frequency_model_efficiency = {
        "schema": "hierarchical_joint_low_frequency_model_efficiency_v1",
        "interpretation": (
            "coupled_full_step_model_reduction_per_frame_low_frequency_energy"
        ),
        "caveat": (
            "A comparative slow-mode indicator, not a Hessian eigenvalue: the "
            "numerator also contains the coupled coarse direction and the "
            "Euclidean rotation/translation energies use different native units."
        ),
        "levels": [],
    }
    for level in frame_mode_energy["levels"]:
        euclidean = level["euclidean"]
        jacobi = level["jacobi_metric"]
        euclidean_low_energy = float(euclidean["low_frequency_energy"])
        jacobi_low_energy = float(jacobi["low_frequency_energy"])
        euclidean_components = euclidean["component_groups"]
        rotation_low_rms = float(
            euclidean_components["rotation"]["low_frequency_block_norm"]["rms"]
        )
        translation_low_rms = float(
            euclidean_components["translation"]["low_frequency_block_norm"]["rms"]
        )
        component_energy = rotation_low_rms**2 + translation_low_rms**2
        frame_low_frequency_model_efficiency["levels"].append(
            {
                "aggregate_size": int(level["aggregate_size"]),
                "euclidean_reduction_per_low_frequency_energy": (
                    float(full_predicted_reduction / euclidean_low_energy)
                    if euclidean_low_energy > 0.0
                    else None
                ),
                "jacobi_reduction_per_low_frequency_energy": (
                    float(full_predicted_reduction / jacobi_low_energy)
                    if jacobi_low_energy > 0.0
                    else None
                ),
                "euclidean_rotation_fraction_of_low_frequency_energy": (
                    float(rotation_low_rms**2 / component_energy)
                    if component_energy > 0.0
                    else None
                ),
                "euclidean_translation_fraction_of_low_frequency_energy": (
                    float(translation_low_rms**2 / component_energy)
                    if component_energy > 0.0
                    else None
                ),
            }
        )
    full_space_block_pcg_probe_diagnostics = None
    if full_space_block_pcg_probe:
        probe_started = perf_counter()
        regularizer = np.concatenate((coarse_regularizer, frame_regularizer))
        exact_full_direction = np.concatenate((schur_direction, frame_direction))
        full_gradient_norm = float(np.linalg.norm(gradient))
        normal_operator_applications_solver = 0
        preconditioner_applications = 0
        full_space_iterations = 0

        def full_normal_product(vector: np.ndarray) -> np.ndarray:
            projected = jacobian @ vector
            return np.asarray(
                jacobian.T @ projected + regularizer * vector,
                dtype=np.float64,
            ).reshape(-1)

        def counted_full_normal_product(vector: np.ndarray) -> np.ndarray:
            nonlocal normal_operator_applications_solver
            normal_operator_applications_solver += 1
            return full_normal_product(vector)

        def block_diagonal_precondition(vector: np.ndarray) -> np.ndarray:
            nonlocal preconditioner_applications
            preconditioner_applications += 1
            coarse_part = coarse_factor.solve(vector[:frame_base])
            frame_part = frame_factor.solve(vector[frame_base:num_variables])
            return np.concatenate((coarse_part, frame_part))

        def count_full_space_iteration(_: np.ndarray) -> None:
            nonlocal full_space_iterations
            full_space_iterations += 1

        full_operator = LinearOperator(
            (num_variables, num_variables),
            matvec=counted_full_normal_product,
            dtype=np.float64,
        )
        block_preconditioner = LinearOperator(
            (num_variables, num_variables),
            matvec=block_diagonal_precondition,
            dtype=np.float64,
        )
        solve_started = perf_counter()
        full_space_direction, full_space_info = solve_cg(
            full_operator,
            -gradient,
            preconditioner=block_preconditioner,
            callback=count_full_space_iteration,
        )
        solve_seconds = perf_counter() - solve_started
        candidate_h_direction = full_normal_product(full_space_direction)
        exact_h_direction = full_normal_product(exact_full_direction)
        candidate_normal_residual = candidate_h_direction + gradient
        exact_normal_residual = exact_h_direction + gradient
        candidate_predicted_reduction = float(
            -np.dot(gradient, full_space_direction)
            - 0.5 * np.dot(full_space_direction, candidate_h_direction)
        )
        candidate_relative_normal_residual = (
            float(np.linalg.norm(candidate_normal_residual) / full_gradient_norm)
            if full_gradient_norm > 0.0
            else 0.0
        )
        reference_relative_normal_residual = (
            float(np.linalg.norm(exact_normal_residual) / full_gradient_norm)
            if full_gradient_norm > 0.0
            else 0.0
        )
        comparison_relative_residual_threshold = max(2.0 * float(cg_rtol), 1e-10)
        comparison_valid = (
            full_space_info == 0
            and cg_info == 0
            and (
                candidate_relative_normal_residual
                <= comparison_relative_residual_threshold
            )
            and (
                reference_relative_normal_residual
                <= comparison_relative_residual_threshold
            )
        )

        def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator <= 0.0:
                return None
            return float(np.dot(left, right) / denominator)

        candidate_coarse = full_space_direction[:frame_base]
        candidate_frame = full_space_direction[frame_base:num_variables]
        full_space_block_pcg_probe_diagnostics = {
            "schema": "hierarchical_joint_schur_full_space_block_pcg_probe_v1",
            "reference": "damped_schur_cg_same_linearization",
            "candidate": "full_space_exact_block_jacobi_pcg",
            "interpretation": (
                "diagnostic_only_full_space_pcg_probe_on_the_frozen_robust_"
                "linearization_not_a_scipy_lsmr_internal_step"
            ),
            "solution_mutated": False,
            "not_scipy_lsmr_internal_step": True,
            "damping": float(damping),
            "normal_operator": "J_robust_T_J_robust_plus_damped_column_diagonal",
            "preconditioner_mode": "exact_block_diagonal_coarse_frame_lu",
            "cg_info": int(full_space_info),
            "cg_converged": bool(full_space_info == 0),
            "cg_iterations": int(full_space_iterations),
            "cg_maxiter": int(cg_maxiter),
            "cg_rtol": float(cg_rtol),
            "reference_cg_info": int(cg_info),
            "reference_cg_converged": bool(cg_info == 0),
            "reference_cg_iterations": int(cg_iterations),
            "reference_cg_rtol": float(cg_rtol),
            "comparison_valid": bool(comparison_valid),
            "comparison_relative_normal_residual_threshold": (
                float(comparison_relative_residual_threshold)
            ),
            "normal_operator_applications_solver": int(
                normal_operator_applications_solver
            ),
            "normal_operator_applications_posthoc": 2,
            "normal_operator_applications_total": int(
                normal_operator_applications_solver + 2
            ),
            "preconditioner_applications": int(preconditioner_applications),
            "relative_normal_residual": float(candidate_relative_normal_residual),
            "reference_relative_normal_residual": float(
                reference_relative_normal_residual
            ),
            "candidate_predicted_quadratic_reduction": (
                candidate_predicted_reduction
            ),
            "reference_predicted_quadratic_reduction": (
                full_predicted_reduction
            ),
            "model_reduction_ratio_vs_reference": (
                float(candidate_predicted_reduction / full_predicted_reduction)
                if full_predicted_reduction > 0.0
                else None
            ),
            "direction_cosine_vs_reference": cosine(
                full_space_direction, exact_full_direction
            ),
            "coarse_direction_cosine_vs_reference": cosine(
                candidate_coarse, schur_direction
            ),
            "frame_direction_cosine_vs_reference": cosine(
                candidate_frame, frame_direction
            ),
            "candidate_full_direction_l2": float(
                np.linalg.norm(full_space_direction)
            ),
            "reference_full_direction_l2": float(
                np.linalg.norm(exact_full_direction)
            ),
            "candidate_coarse_direction_block_rms": float(
                np.linalg.norm(candidate_coarse) / np.sqrt(num_coarse_blocks)
            ),
            "reference_coarse_direction_block_rms": float(
                coarse_direction_block_rms
            ),
            "candidate_frame_direction_block_rms": float(
                np.linalg.norm(candidate_frame) / np.sqrt(num_frame_blocks)
            ),
            "reference_frame_direction_block_rms": float(frame_direction_block_rms),
            "coarse_direction_block_rms_ratio_vs_reference": (
                float(
                    np.linalg.norm(candidate_coarse)
                    / np.sqrt(num_coarse_blocks)
                    / coarse_direction_block_rms
                )
                if coarse_direction_block_rms > 0.0
                else None
            ),
            "frame_direction_block_rms_ratio_vs_reference": (
                float(
                    np.linalg.norm(candidate_frame)
                    / np.sqrt(num_frame_blocks)
                    / frame_direction_block_rms
                )
                if frame_direction_block_rms > 0.0
                else None
            ),
            "phase_seconds": {
                "setup": float(solve_started - probe_started),
                "cg": float(solve_seconds),
                "residual_and_comparison": float(
                    perf_counter() - solve_started - solve_seconds
                ),
                "total": float(perf_counter() - probe_started),
            },
        }
    diagnostics = {
        "schema": "hierarchical_joint_schur_diagnostics_v1",
        "linearization": getattr(
            result,
            "linearization_source",
            "scipy_least_squares_result_linearization",
        ),
        "linear_solver": linear_solver,
        "solution_mutated": False,
        "damping": float(damping),
        "coarse_variables": int(frame_base),
        "frame_variables": int(num_variables - frame_base),
        "coarse_hessian_nnz": int(coarse_hessian.nnz),
        "frame_hessian_nnz": int(frame_hessian.nnz),
        "frame_factor_nnz": int(frame_factor.L.nnz + frame_factor.U.nnz),
        "coarse_gradient_l2": float(np.linalg.norm(coarse_gradient)),
        "frame_gradient_l2": float(np.linalg.norm(frame_gradient)),
        "reduced_coarse_gradient_l2": float(np.linalg.norm(reduced_gradient)),
        "reduced_coarse_free_gradient_l2": schur_rhs_norm,
        "frame_elimination_gradient_correction_l2": float(
            np.linalg.norm(coarse_gradient - reduced_gradient)
        ),
        "uncoupled_coarse_block_newton_decrement_squared": (
            coarse_block_newton_decrement_squared
        ),
        "uncoupled_frame_block_newton_decrement_squared": (
            frame_block_newton_decrement_squared
        ),
        "frame_to_coarse_block_newton_decrement_squared_ratio": (
            float(
                frame_block_newton_decrement_squared
                / coarse_block_newton_decrement_squared
            )
            if coarse_block_newton_decrement_squared > 0.0
            else None
        ),
        "reduced_coarse_newton_decrement_squared": (
            reduced_coarse_newton_decrement_squared
        ),
        "schur_direction_l2": coarse_direction_l2,
        "coarse_direction_constraint": (
            "freeze_log_scale_nullspace"
            if freeze_coarse_scales
            else "unconstrained"
        ),
        "frozen_coarse_scale_direction_count": int(
            len(coarse_scale_indices) if freeze_coarse_scales else 0
        ),
        "coarse_scale_direction_linf": float(
            np.max(np.abs(schur_direction[coarse_scale_indices]), initial=0.0)
        ),
        "frame_backsub_direction_l2": frame_direction_l2,
        "coarse_direction_dof_rms": float(coarse_direction_dof_rms),
        "frame_direction_dof_rms": float(frame_direction_dof_rms),
        "frame_to_coarse_direction_dof_rms_ratio": float(
            frame_direction_dof_rms / coarse_direction_dof_rms
        ),
        "coarse_direction_block_rms": float(coarse_direction_block_rms),
        "frame_direction_block_rms": float(frame_direction_block_rms),
        "frame_to_coarse_direction_block_rms_ratio": float(
            frame_direction_block_rms / coarse_direction_block_rms
        ),
        "frame_to_coarse_direction_total_l2_ratio": float(
            frame_direction_l2 / coarse_direction_l2
        ),
        "frame_direction_sequential_mode_energy": frame_mode_energy,
        "frame_low_frequency_model_efficiency": (
            frame_low_frequency_model_efficiency
        ),
        "full_direction_l2": float(
            np.hypot(coarse_direction_l2, frame_direction_l2)
        ),
        "full_directional_gradient": full_directional_gradient,
        "full_directional_curvature": full_directional_curvature,
        "full_predicted_quadratic_reduction": full_predicted_reduction,
        "fixed_frame_direction_l2": float(np.linalg.norm(fixed_direction)),
        "schur_fixed_direction_cosine": direction_cosine,
        "predicted_quadratic_reduction": predicted_reduction,
        "cg_info": int(cg_info),
        "cg_converged": bool(cg_info == 0),
        "cg_iterations": int(cg_iterations),
        "cg_maxiter": int(cg_maxiter),
        "cg_rtol": float(cg_rtol),
        "cg_relative_residual": (
            float(np.linalg.norm(schur_residual) / schur_rhs_norm)
            if schur_rhs_norm > 0.0
            else 0.0
        ),
        "phase_seconds": {
            "normal_matrix_assembly": float(assembly_seconds),
            "frame_factorization": float(frame_factor_seconds),
            "coarse_factorization": float(coarse_factor_seconds),
            "reduced_gradient": float(reduced_gradient_seconds),
            "schur_cg": float(cg_seconds),
            "frame_backsubstitution": float(frame_backsub_seconds),
            "fixed_frame_solve": float(fixed_seconds),
            "total": float(perf_counter() - total_started),
        },
        **(
            {"two_level": two_level_diagnostics}
            if two_level_diagnostics is not None
            else {}
        ),
        **(
            {"two_level_bypassed_reason": "coarse_scale_nullspace_constraint"}
            if freeze_coarse_scales and two_level_aggregate_size is not None
            else {}
        ),
        **(
            {"full_space_block_pcg_probe": full_space_block_pcg_probe_diagnostics}
            if full_space_block_pcg_probe_diagnostics is not None
            else {}
        ),
    }
    return diagnostics, schur_direction, frame_direction, coarse_grid_cache_out


def _joint_family_diagnostics_from_solver_result(
    result: Any,
    *,
    num_coarse_edges: int,
    num_fine_edges: int,
    num_frame_priors: int,
    frame_base: int,
    num_variables: int,
) -> dict[str, Any]:
    """Summarize final Joint linearization by residual family and variable block."""

    from scipy import sparse

    jacobian = result.jac
    residual = np.asarray(result.fun, dtype=np.float64).reshape(-1)
    row_slices = {
        "coarse": slice(0, 7 * int(num_coarse_edges)),
        "fine": slice(
            7 * int(num_coarse_edges),
            7 * int(num_coarse_edges) + 6 * int(num_fine_edges),
        ),
        "prior": slice(
            7 * int(num_coarse_edges) + 6 * int(num_fine_edges),
            7 * int(num_coarse_edges)
            + 6 * int(num_fine_edges)
            + 6 * int(num_frame_priors),
        ),
    }
    column_slices = {
        "coarse": slice(0, int(frame_base)),
        "frame": slice(int(frame_base), int(num_variables)),
    }

    def gradient(row_slice: slice, column_slice: slice) -> np.ndarray:
        if row_slice.stop == row_slice.start or column_slice.stop == column_slice.start:
            return np.zeros(column_slice.stop - column_slice.start, dtype=np.float64)
        block = jacobian[row_slice, column_slice]
        return np.asarray(
            block.T @ residual[row_slice], dtype=np.float64
        ).reshape(-1)

    def hessian_trace(row_slice: slice, column_slice: slice) -> float:
        if row_slice.stop == row_slice.start or column_slice.stop == column_slice.start:
            return 0.0
        block = jacobian[row_slice, column_slice]
        if sparse.issparse(block):
            return float(block.multiply(block).sum())
        values = np.asarray(block, dtype=np.float64)
        return float(np.sum(values * values))

    family_diagnostics = {}
    family_gradients: dict[str, dict[str, np.ndarray]] = {}
    for name, row_slice in row_slices.items():
        residual_block = residual[row_slice]
        block_gradients = {
            block_name: gradient(row_slice, column_slice)
            for block_name, column_slice in column_slices.items()
        }
        family_gradients[name] = block_gradients
        family_diagnostics[name] = {
            "row_count": int(row_slice.stop - row_slice.start),
            "squared_residual_sum": float(residual_block @ residual_block),
            "coarse_gradient_inf": float(
                np.linalg.norm(block_gradients["coarse"], ord=np.inf)
            ),
            "frame_gradient_inf": float(
                np.linalg.norm(block_gradients["frame"], ord=np.inf)
            ),
            "coarse_hessian_trace": hessian_trace(row_slice, column_slices["coarse"]),
            "frame_hessian_trace": hessian_trace(row_slice, column_slices["frame"]),
        }

    def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator == 0.0:
            return None
        return float(np.dot(left, right) / denominator)

    block_balance = {}
    block_dimensions = {"coarse": 7, "frame": 6}
    for block_name in column_slices:
        gradients = {
            family_name: family_gradients[family_name][block_name]
            for family_name in row_slices
        }
        total = sum(gradients.values(), np.zeros_like(next(iter(gradients.values()))))
        total_l2 = float(np.linalg.norm(total))
        family_l2_sum = float(sum(np.linalg.norm(item) for item in gradients.values()))
        block_dimension = block_dimensions[block_name]
        num_dofs = int(total.size)
        num_blocks = num_dofs // block_dimension
        dof_normalizer = np.sqrt(num_dofs) if num_dofs > 0 else 1.0
        block_normalizer = np.sqrt(num_blocks) if num_blocks > 0 else 1.0
        block_balance[block_name] = {
            "block_dimension": int(block_dimension),
            "num_dofs": num_dofs,
            "num_blocks": int(num_blocks),
            "total_gradient_l2": total_l2,
            "total_gradient_dof_rms": float(total_l2 / dof_normalizer),
            "total_gradient_block_rms": float(total_l2 / block_normalizer),
            "family_gradient_l2_sum": family_l2_sum,
            "family_gradient_dof_rms_sum": float(
                family_l2_sum / dof_normalizer
            ),
            "family_gradient_block_rms_sum": float(
                family_l2_sum / block_normalizer
            ),
            "cancellation_ratio": (
                float(total_l2 / family_l2_sum)
                if family_l2_sum > 0.0
                else 0.0
            ),
            "pairwise_cosine": {
                "coarse_fine": cosine(gradients["coarse"], gradients["fine"]),
                "coarse_prior": cosine(gradients["coarse"], gradients["prior"]),
                "fine_prior": cosine(gradients["fine"], gradients["prior"]),
            },
        }

    reconstructed_gradient = np.asarray(
        jacobian.T @ residual, dtype=np.float64
    ).reshape(-1)
    result_gradient = np.asarray(result.grad, dtype=np.float64).reshape(-1)
    gradient_shape_matches = reconstructed_gradient.shape == result_gradient.shape
    if gradient_shape_matches:
        gradient_delta = reconstructed_gradient - result_gradient
        gradient_error_inf = float(np.linalg.norm(gradient_delta, ord=np.inf))
        result_gradient_inf = float(np.linalg.norm(result_gradient, ord=np.inf))
        reconstructed_gradient_inf = float(
            np.linalg.norm(reconstructed_gradient, ord=np.inf)
        )
        if result_gradient_inf > 0.0:
            gradient_norm_ratio = float(reconstructed_gradient_inf / result_gradient_inf)
            gradient_relative_error = float(gradient_error_inf / result_gradient_inf)
        else:
            gradient_norm_ratio = 1.0 if reconstructed_gradient_inf == 0.0 else np.inf
            gradient_relative_error = 0.0 if gradient_error_inf == 0.0 else np.inf
    else:
        gradient_error_inf = np.inf
        result_gradient_inf = float(np.linalg.norm(result_gradient, ord=np.inf))
        reconstructed_gradient_inf = float(
            np.linalg.norm(reconstructed_gradient, ord=np.inf)
        )
        gradient_norm_ratio = np.inf
        gradient_relative_error = np.inf

    return {
        "schema": "hierarchical_joint_family_diagnostics_v1",
        "linearization": "solver_final_linearization",
        "source": "scipy_least_squares_result_jac_and_fun",
        "note": (
            "Values are computed from result.jac and result.fun at the solver "
            "final linearization; no fresh residual or Jacobian evaluation is run."
        ),
        "cancellation_interpretation": (
            "Small cross-family cancellation ratios are expected near a "
            "stationary point and do not by themselves prove poor conditioning "
            "or block dominance; compare curvature-normalized Schur diagnostics."
        ),
        "row_count": int(residual.size),
        "row_count_from_families": int(
            sum(item["row_count"] for item in family_diagnostics.values())
        ),
        "coarse_column_count": int(frame_base),
        "frame_column_count": int(num_variables - frame_base),
        "jacobian_sparse": bool(sparse.issparse(jacobian)),
        "families": family_diagnostics,
        "block_gradient_balance": block_balance,
        "total_gradient_reconstruction": {
            "source": "result.jac.T @ result.fun",
            "compared_to": "result.grad",
            "shape_matches": bool(gradient_shape_matches),
            "result_grad_inf": result_gradient_inf,
            "reconstructed_grad_inf": reconstructed_gradient_inf,
            "error_inf": gradient_error_inf,
            "relative_error": gradient_relative_error,
            "norm_ratio": gradient_norm_ratio,
        },
    }


def optimize_hierarchical_pose_graph(
    chunks: Sequence[Any],
    chunk_ranges: Sequence[Any],
    edges: Sequence[Sim3Edge],
    initial_alignments: Sequence[Sim3],
    *,
    config: HierarchicalPoseGraphConfig | None = None,
) -> HierarchicalPoseGraphResult:
    """Optimize chunk Sim3 nodes and shared global frame Sim3 nodes jointly.

    ``edges`` contains the existing hip3r adjacent and loop constraints.  Local
    frame constraints are derived from each chunk's predicted ``extrinsic_c2w``
    and are never used to replace those coarse constraints.  The root chunk and
    first global frame are fixed to remove the global gauge.
    """

    cfg = config or HierarchicalPoseGraphConfig()
    basis = build_hierarchical_basis(
        chunks,
        chunk_ranges,
        edges,
        initial_alignments,
        config=cfg,
    )
    corrected = apply_hierarchical_alpha(
        chunks,
        chunk_ranges,
        basis,
        alpha=cfg.frame_correction_alpha,
        apply_frame_corrections=cfg.apply_frame_corrections,
    )
    diagnostics = {
        **basis.diagnostics,
        "frame_corrections_applied": bool(cfg.apply_frame_corrections),
        "frame_correction_alpha": float(cfg.frame_correction_alpha),
    }
    return HierarchicalPoseGraphResult(
        corrected,
        basis.coarse_alignments,
        basis.frame_poses,
        diagnostics,
    )


def build_hierarchical_basis(
    chunks: Sequence[Any],
    chunk_ranges: Sequence[Any],
    edges: Sequence[Sim3Edge],
    initial_alignments: Sequence[Sim3],
    *,
    config: HierarchicalPoseGraphConfig | None = None,
) -> HierarchicalAlphaBasis:
    """Run the hierarchical solve once and return an alpha-independent basis."""

    cfg = config or HierarchicalPoseGraphConfig()
    chunk_list = list(chunks)
    ranges = list(chunk_ranges)
    num_chunks = len(chunk_list)
    if num_chunks == 0:
        return HierarchicalAlphaBasis(
            coarse_alignments=[],
            frame_poses=[],
            apply_point_corrections=True,
            diagnostics={"status": "empty"},
        )
    if len(ranges) != num_chunks:
        raise ValueError(f"expected {num_chunks} chunk ranges, got {len(ranges)}")
    if len(initial_alignments) != num_chunks:
        raise ValueError(f"expected {num_chunks} alignments, got {len(initial_alignments)}")
    _validate_edges(num_chunks, edges)
    factorized_frame_chart_residual_v4 = bool(
        cfg.factorized_frame_chart_residual_v4
    )
    if factorized_frame_chart_residual_v4:
        expected_adjacent = {(index, index - 1) for index in range(1, num_chunks)}
        actual_adjacent = {(int(edge.source), int(edge.target)) for edge in edges}
        if any(edge.kind != "adjacent" for edge in edges):
            raise ValueError(
                "factorized_frame_chart_residual_v4 supports only adjacent edges"
            )
        if actual_adjacent != expected_adjacent or len(edges) != len(
            expected_adjacent
        ):
            raise ValueError(
                "factorized_frame_chart_residual_v4 requires a loop-free complete "
                "adjacent chain"
            )

    starts_ends = [_range_start_end(item) for item in ranges]
    num_frames = max(end for _, end in starts_ends)
    _validate_ranges(starts_ends, num_frames)
    local_poses = [_chunk_c2w(chunk) for chunk in chunk_list]
    if any(poses.shape[0] != end - start for poses, (start, end) in zip(local_poses, starts_ends)):
        raise ValueError("chunk range lengths do not match chunk extrinsic_c2w lengths")

    fine_build_phase_seconds: dict[str, float] = {}
    fine_phase_started = perf_counter()
    scale_bound_alignments = list(initial_alignments)
    hierarchical_warm_start_diagnostics: dict[str, Any] | None = None
    frozen_coarse = list(initial_alignments) if cfg.freeze_coarse_alignments else None
    frozen_aligned_local_components = (
        _aligned_local_frame_pose_components(local_poses, frozen_coarse)
        if frozen_coarse is not None
        else None
    )
    initial_frame_poses = (
        _initial_frame_poses_from_aligned_components(
            frozen_aligned_local_components, starts_ends, num_frames
        )
        if frozen_aligned_local_components is not None
        else _initial_frame_poses(
            local_poses, starts_ends, list(initial_alignments), num_frames
        )
    )
    fine_build_phase_seconds["aligned_local_and_initial_frame_poses"] = (
        perf_counter() - fine_phase_started
    )
    fixed_root = initial_alignments[0]
    fixed_frame = initial_frame_poses[0]

    if num_frames > cfg.max_frames_per_solve and cfg.long_sequence_mode == "keyframe_fine_graph":
        return _solve_keyframe_fine_graph(
            local_poses,
            starts_ends,
            edges,
            list(initial_alignments),
            initial_frame_poses,
            num_frames,
            cfg=cfg,
        )
    if num_frames > cfg.max_frames_per_solve and cfg.long_sequence_mode == "keyframe_then_full":
        if cfg.freeze_coarse_alignments:
            raise ValueError(
                "keyframe_then_full currently requires free coarse alignments"
            )
        warm_start_started = perf_counter()
        warm_start_config = _dataclass_replace(
            cfg,
            freeze_coarse_alignments=True,
            long_sequence_mode="keyframe_fine_graph",
            analytic_frozen_jacobian=True,
            analytic_dynamic_jacobian=False,
            vectorize_dynamic_residual=False,
            joint_family_diagnostics=False,
            joint_frame_relaxation_max_nfev=0,
            joint_schur_diagnostics=False,
            joint_schur_step=False,
        )
        warm_start = _solve_keyframe_fine_graph(
            local_poses,
            starts_ends,
            edges,
            list(initial_alignments),
            initial_frame_poses,
            num_frames,
            cfg=warm_start_config,
        )
        warm_start_wall_seconds = perf_counter() - warm_start_started
        initial_alignments = list(warm_start.coarse_alignments)
        initial_frame_poses = list(warm_start.frame_poses)
        fixed_root = initial_alignments[0]
        fixed_frame = initial_frame_poses[0]
        warm_diagnostics = warm_start.diagnostics
        hierarchical_warm_start_diagnostics = {
            "schema": "hierarchical_joint_keyframe_then_full_warm_start_v2",
            "interpretation": (
                "frozen_coarse_low_frequency_frame_continuation_before_full_joint"
            ),
            "objective_scope": (
                "warm start freezes the public coarse anchor; the second stage "
                "restores free coarse nodes, every frame variable, and the "
                "original full Joint objective"
            ),
            "coarse_policy": "frozen_public_anchor_then_release",
            "scale_bound_center": "original_public_coarse_anchor",
            "wall_seconds": float(warm_start_wall_seconds),
            "success": bool(warm_diagnostics.get("success", False)),
            "num_keyframes": int(warm_diagnostics["num_keyframes"]),
            "num_variables": int(warm_diagnostics["num_variables"]),
            "num_function_evaluations": int(
                warm_diagnostics.get(
                    "num_function_evaluations_total",
                    warm_diagnostics.get("num_function_evaluations", 0),
                )
            ),
            "num_jacobian_evaluations": warm_diagnostics.get(
                "num_jacobian_evaluations_total",
                warm_diagnostics.get("num_jacobian_evaluations"),
            ),
            "solver_wall_seconds": float(
                warm_diagnostics.get("solver_wall_seconds", 0.0)
            ),
            "scipy_cost": float(warm_diagnostics["scipy_cost"]),
            "optimality": float(warm_diagnostics["optimality"]),
        }
        fine_build_phase_seconds["keyframe_hierarchy_warm_start"] = float(
            warm_start_wall_seconds
        )
    if cfg.freeze_coarse_alignments and num_frames > cfg.max_frames_per_solve:
        consensus_stride = cfg.consensus_keyframe_stride
        consensus_keyframes = _sparse_keyframe_indices(num_frames, consensus_stride)
        raw_consensus_frames = _consensus_frame_poses(
            local_poses,
            starts_ends,
            list(initial_alignments),
            num_frames,
            selected_frames=set(consensus_keyframes),
            fallback_frames=initial_frame_poses,
        )
        consensus_frames = _interpolate_sparse_frame_corrections(
            initial_frame_poses,
            raw_consensus_frames,
            stride=consensus_stride,
        )
        rotation_corrections = []
        translation_corrections = []
        for initial, consensus in zip(initial_frame_poses, consensus_frames):
            delta = consensus.compose(initial.inverse())
            rotation_corrections.append(
                float(np.linalg.norm(Rotation.from_matrix(delta.rotation).as_rotvec()))
            )
            translation_corrections.append(float(np.linalg.norm(delta.translation)))
        return HierarchicalAlphaBasis(
            coarse_alignments=list(initial_alignments),
            frame_poses=consensus_frames,
            apply_point_corrections=False,
            diagnostics={
                "status": "ok",
                "optimizer_used": False,
                "solver_mode": "overlap_consensus",
                "coarse_alignments_frozen": True,
                "num_chunks": int(num_chunks),
                "num_frames": int(num_frames),
                "max_frames_per_solve": int(cfg.max_frames_per_solve),
                "consensus_keyframe_stride": int(consensus_stride),
                "consensus_compute_mode": "sparse_keyframes",
                "consensus_frames_evaluated": int(len(consensus_keyframes)),
                "point_corrections_applied": False,
                "max_rotation_correction_rad": max(rotation_corrections, default=0.0),
                "max_translation_correction": max(translation_corrections, default=0.0),
            },
        )

    coarse_offsets = (
        {}
        if cfg.freeze_coarse_alignments
        else {index: 7 * (index - 1) for index in range(1, num_chunks)}
    )
    frame_base = 0 if cfg.freeze_coarse_alignments else 7 * max(num_chunks - 1, 0)
    frame_offsets = {
        frame: frame_base + 6 * (frame - 1) for frame in range(1, num_frames)
    }
    num_variables = frame_base + 6 * max(num_frames - 1, 0)

    fine_phase_started = perf_counter()
    fine_edges = _build_fine_edges(
        local_poses,
        starts_ends,
        frame_stride=cfg.frame_stride,
    )
    prior_records = _build_prior_records(
        local_poses,
        starts_ends,
    )
    if cfg.normalize_residual_families:
        if not fine_edges or not prior_records:
            raise ValueError(
                "residual-family normalization requires non-empty fine and prior constraints"
            )
        normalization_mass = max(len(edges), 1)
        fine_edge_weight = float(normalization_mass / len(fine_edges))
        coarse_prior_weight = float(normalization_mass / len(prior_records))
    else:
        fine_edge_weight = float(cfg.fine_edge_weight)
        coarse_prior_weight = float(cfg.coarse_prior_weight)
    fine_build_phase_seconds["edge_and_prior_records"] = (
        perf_counter() - fine_phase_started
    )
    frozen_fine_constraints = None
    frozen_prior_constraints = None
    frozen_vectorized = None
    factorized_fine_constraints = None
    factorized_prior_constraints = None
    factorized_constants_diagnostics = None
    if factorized_frame_chart_residual_v4:
        if frozen_coarse is None:
            raise ValueError(
                "factorized_frame_chart_residual_v4 requires frozen coarse "
                "alignments"
            )
        fine_phase_started = perf_counter()
        frame_anchor_records: list[tuple[int, int] | None] = [None] * num_frames
        for chunk_index, (start, end) in enumerate(starts_ends):
            for local_index, frame in enumerate(range(start, end)):
                if frame_anchor_records[frame] is None:
                    frame_anchor_records[frame] = (chunk_index, local_index)
        if any(record is None for record in frame_anchor_records):
            raise ValueError(
                "factorized_frame_chart_residual_v4 could not anchor every frame"
            )
        frame_anchor_records_strict = [
            record for record in frame_anchor_records if record is not None
        ]
        local_pose_sim3 = [
            [_c2w_to_sim3(pose) for pose in poses] for poses in local_poses
        ]
        adjacent_by_source = {int(edge.source): edge.transform for edge in edges}
        path_hop_max = 0

        def chunk_relative_by_adjacent_path(target: int, source: int) -> Sim3:
            nonlocal path_hop_max
            target = int(target)
            source = int(source)
            hops = abs(source - target)
            path_hop_max = max(path_hop_max, hops)
            if hops > _FACTORIZED_FRAME_CHART_MAX_CHUNK_HOPS:
                raise ValueError(
                    "factorized_frame_chart_residual_v4 path hop cap exceeded: "
                    f"{hops}>{_FACTORIZED_FRAME_CHART_MAX_CHUNK_HOPS}"
                )
            if source == target:
                return Sim3.identity()
            if source > target:
                result = Sim3.identity()
                for chunk in range(target + 1, source + 1):
                    result = result.compose(adjacent_by_source[chunk])
                return result
            return chunk_relative_by_adjacent_path(source, target).inverse()

        def stable_chart_relative(
            target_record: tuple[int, int],
            source_record: tuple[int, int],
        ) -> Sim3:
            target_chunk, target_local = target_record
            source_chunk, source_local = source_record
            target_pose = local_pose_sim3[target_chunk][target_local]
            source_pose = local_pose_sim3[source_chunk][source_local]
            chunk_relative = chunk_relative_by_adjacent_path(
                target_chunk, source_chunk
            )
            rotation = (
                target_pose.rotation.T
                @ chunk_relative.rotation
                @ source_pose.rotation
            )
            translation = frozen_coarse[target_chunk].scale * (
                target_pose.rotation.T
                @ (
                    chunk_relative.scale
                    * (chunk_relative.rotation @ source_pose.translation)
                    + chunk_relative.translation
                    - target_pose.translation
                )
            )
            return Sim3(
                scale=1.0,
                rotation=rotation,
                translation=translation,
            )

        factorized_fine_constraints = []
        fine_measurement_translation_norms: list[float] = []
        for chunk_index, frame_a, frame_b, _, _ in fine_edges:
            chunk_start = starts_ends[chunk_index][0]
            measured = stable_chart_relative(
                (chunk_index, frame_b - chunk_start),
                (chunk_index, frame_a - chunk_start),
            )
            base = stable_chart_relative(
                frame_anchor_records_strict[frame_b],
                frame_anchor_records_strict[frame_a],
            )
            fine_measurement_translation_norms.append(
                float(np.linalg.norm(measured.translation))
            )
            factorized_fine_constraints.append((frame_a, frame_b, measured, base))

        factorized_prior_constraints = []
        prior_identity_translation_norms: list[float] = []
        for chunk_index, frame, _ in prior_records:
            chunk_start = starts_ends[chunk_index][0]
            chart_error = stable_chart_relative(
                (chunk_index, frame - chunk_start),
                frame_anchor_records_strict[frame],
            )
            prior_identity_translation_norms.append(
                float(np.linalg.norm(chart_error.translation))
            )
            factorized_prior_constraints.append((frame, chart_error))

        coarse_identity_residuals = []
        for edge in edges:
            path_relative = chunk_relative_by_adjacent_path(
                edge.target, edge.source
            )
            error = edge.transform.inverse().compose(path_relative)
            coarse_identity_residuals.append(
                _sim3_residual(error, cfg.translation_scale)
            )
        coarse_identity = (
            np.concatenate(coarse_identity_residuals)
            if coarse_identity_residuals
            else np.zeros(0, dtype=np.float64)
        )
        coarse_path_identity_inf = float(
            np.max(np.abs(coarse_identity), initial=0.0)
        )
        factorized_constants_diagnostics = {
            "stable_constant_source": "adjacent_edge_path_and_chunk_local_poses",
            "chunk_path_max_hops": int(path_hop_max),
            "chunk_path_hop_cap": int(_FACTORIZED_FRAME_CHART_MAX_CHUNK_HOPS),
            "coarse_path_identity_inf": coarse_path_identity_inf,
            "coarse_path_identity_gate_pass": bool(
                coarse_path_identity_inf <= 1e-10
            ),
            "fine_measurement_translation_norm_max": float(
                max(fine_measurement_translation_norms, default=0.0)
            ),
            "prior_identity_translation_norm_max": float(
                max(prior_identity_translation_norms, default=0.0)
            ),
            "anchor_assignment": "first_chunk_occurrence_matches_initial_frame_poses",
        }
        fine_build_phase_seconds["factorized_frame_chart_constants"] = (
            perf_counter() - fine_phase_started
        )
    if (
        frozen_coarse is not None
        and cfg.precompute_frozen_residual_constants
        and not factorized_frame_chart_residual_v4
    ):
        fine_phase_started = perf_counter()
        num_fine_edges = len(fine_edges)
        assert frozen_aligned_local_components is not None
        component_lengths = np.fromiter(
            (len(item[0]) for item in frozen_aligned_local_components),
            dtype=np.int64,
            count=len(frozen_aligned_local_components),
        )
        component_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(component_lengths))
        )
        component_rotations = np.concatenate(
            [item[0] for item in frozen_aligned_local_components], axis=0
        )
        component_translations = np.concatenate(
            [item[1] for item in frozen_aligned_local_components], axis=0
        )
        chunk_starts = np.asarray([item[0] for item in starts_ends], dtype=np.int64)
        fine_chunks = np.fromiter(
            (item[0] for item in fine_edges), dtype=np.int64, count=num_fine_edges
        )
        frozen_fine_a = np.fromiter(
            (item[1] for item in fine_edges), dtype=np.int64, count=num_fine_edges
        )
        frozen_fine_b = np.fromiter(
            (item[2] for item in fine_edges), dtype=np.int64, count=num_fine_edges
        )
        fine_a_indices = (
            component_offsets[fine_chunks]
            + frozen_fine_a
            - chunk_starts[fine_chunks]
        )
        fine_b_indices = (
            component_offsets[fine_chunks]
            + frozen_fine_b
            - chunk_starts[fine_chunks]
        )
        frozen_expected_a_rot = component_rotations[fine_a_indices]
        frozen_expected_a_trans = component_translations[fine_a_indices]
        frozen_expected_b_rot = component_rotations[fine_b_indices]
        frozen_expected_b_trans = component_translations[fine_b_indices]
        frozen_fine_scales, frozen_fine_rotations, frozen_fine_translations = (
            _inverse_compose_component_arrays_exact(
                np.ones(num_fine_edges, dtype=np.float64),
                frozen_expected_b_rot,
                frozen_expected_b_trans,
                np.ones(num_fine_edges, dtype=np.float64),
                frozen_expected_a_rot,
                frozen_expected_a_trans,
            )
        )
        num_prior_records = len(prior_records)
        prior_chunks = np.fromiter(
            (item[0] for item in prior_records), dtype=np.int64, count=num_prior_records
        )
        frozen_prior_frames = np.fromiter(
            (item[1] for item in prior_records), dtype=np.int64, count=num_prior_records
        )
        prior_indices = (
            component_offsets[prior_chunks]
            + frozen_prior_frames
            - chunk_starts[prior_chunks]
        )
        frozen_prior_rotations = component_rotations[prior_indices]
        frozen_prior_translations = component_translations[prior_indices]
        if cfg.vectorize_frozen_residual:
            (
                frozen_fine_scales,
                frozen_fine_rotations,
                frozen_fine_translations,
            ) = _validate_sim3_batch_components(
                frozen_fine_scales,
                frozen_fine_rotations,
                frozen_fine_translations,
            )
            frozen_vectorized = {
                "coarse": _weighted_edge_residuals_batch_exact(
                    edges, frozen_coarse, cfg.translation_scale
                )
                if edges
                else np.zeros(0, dtype=np.float64),
                "fine_a": frozen_fine_a,
                "fine_b": frozen_fine_b,
                "fine_rot": frozen_fine_rotations,
                "fine_trans": frozen_fine_translations,
                "prior_frame": frozen_prior_frames,
                "prior_rot": frozen_prior_rotations,
                "prior_trans": frozen_prior_translations,
            }
        else:
            frozen_fine_transforms = _validated_sim3_batch(
                frozen_fine_scales,
                frozen_fine_rotations,
                frozen_fine_translations,
            )
            frozen_fine_constraints = [
                (frame_a, frame_b, transform)
                for frame_a, frame_b, transform in zip(
                    frozen_fine_a, frozen_fine_b, frozen_fine_transforms
                )
            ]
            frozen_prior_transforms = _validated_sim3_batch(
                np.ones(len(frozen_prior_frames), dtype=np.float64),
                np.asarray(frozen_prior_rotations, dtype=np.float64),
                np.asarray(frozen_prior_translations, dtype=np.float64),
            )
            frozen_prior_constraints = list(zip(frozen_prior_frames, frozen_prior_transforms))
        fine_build_phase_seconds["frozen_constraint_precompute"] = (
            perf_counter() - fine_phase_started
        )
    row_count = 7 * len(edges) + 6 * (len(fine_edges) + len(prior_records))
    frozen_coarse_rows_excluded_from_optimizer = bool(
        cfg.freeze_coarse_alignments and cfg.frozen_optimizer_variable_rows_only
    )
    optimizer_residual_row_start = (
        7 * len(edges) if frozen_coarse_rows_excluded_from_optimizer else 0
    )
    optimizer_row_count = row_count - optimizer_residual_row_start
    if optimizer_row_count < 0:
        raise RuntimeError("optimizer residual row slice is invalid")

    def optimizer_residual_view(values: np.ndarray) -> np.ndarray:
        residual_values = np.asarray(values, dtype=np.float64).reshape(-1)
        if residual_values.size == optimizer_row_count:
            return residual_values
        if residual_values.size != row_count:
            raise ValueError(
                "optimizer residual slice expected full or optimizer rows, got "
                f"{residual_values.size}"
            )
        if optimizer_residual_row_start == 0:
            return residual_values
        return residual_values[optimizer_residual_row_start:]

    def optimizer_jacobian_view(jacobian: Any) -> Any:
        if int(jacobian.shape[0]) == optimizer_row_count:
            return jacobian
        if int(jacobian.shape[0]) != row_count:
            raise ValueError(
                "optimizer Jacobian slice expected full or optimizer rows, got "
                f"{jacobian.shape[0]}"
            )
        if optimizer_residual_row_start == 0:
            return jacobian
        return jacobian[optimizer_residual_row_start:, :]

    def unpack(vector: np.ndarray) -> tuple[list[Sim3], list[Sim3]]:
        if frozen_coarse is not None:
            coarse = list(frozen_coarse)
        else:
            coarse = [fixed_root]
            for index in range(1, num_chunks):
                offset = coarse_offsets[index]
                coarse.append(_parameters_to_sim3(vector[offset : offset + 7]))
        frames = [fixed_frame]
        if num_frames > 1:
            frame_parameters = np.asarray(
                vector[
                    frame_offsets[1] : frame_offsets[1] + 6 * (num_frames - 1)
                ],
                dtype=np.float64,
            ).reshape(num_frames - 1, 6)
            frame_rotations = Rotation.from_rotvec(
                frame_parameters[:, :3]
            ).as_matrix()
            frames.extend(
                _validated_sim3_batch(
                    np.ones(num_frames - 1, dtype=np.float64),
                    frame_rotations,
                    frame_parameters[:, 3:6],
                )
            )
        return coarse, frames

    def unpack_factorized_frame_chart_residual_v4(
        vector: np.ndarray,
    ) -> tuple[list[Sim3], list[Sim3], list[Sim3]]:
        if frozen_coarse is None:
            raise ValueError(
                "factorized_frame_chart_residual_v4 requires frozen coarse "
                "alignments"
            )
        deltas = [Sim3.identity()]
        if num_frames > 1:
            delta_parameters = np.asarray(
                vector[
                    frame_offsets[1] : frame_offsets[1] + 6 * (num_frames - 1)
                ],
                dtype=np.float64,
            ).reshape(num_frames - 1, 6)
            delta_rotations = Rotation.from_rotvec(
                delta_parameters[:, :3]
            ).as_matrix()
            deltas.extend(
                _validated_sim3_batch(
                    np.ones(num_frames - 1, dtype=np.float64),
                    delta_rotations,
                    delta_parameters[:, 3:6],
                )
            )
        frames = [
            base_pose.compose(delta)
            for base_pose, delta in zip(initial_frame_poses, deltas)
        ]
        return list(frozen_coarse), frames, deltas

    residual_profile_lock = Lock()
    residual_calls_total = 0
    residual_calls_solver = 0
    residual_wall_seconds_total = 0.0
    residual_wall_seconds_solver = 0.0
    residual_profile_solver_active = False
    analytic_jacobian: _FrozenTorchAnalyticJacobian | None = None
    analytic_fallback_reason: str | None = None
    dynamic_analytic_jacobian: (
        _DynamicTorchAnalyticJacobian | _DynamicClosedFormAnalyticJacobian | None
    ) = None
    dynamic_analytic_fallback_reason: str | None = None
    if factorized_frame_chart_residual_v4:
        analytic_fallback_reason = (
            "factorized_frame_chart_residual_v4_uses_sparse_finite_difference"
        )
    elif not cfg.analytic_frozen_jacobian:
        analytic_fallback_reason = "config_disabled"
    elif frozen_vectorized is None:
        analytic_fallback_reason = "frozen_vectorized_unavailable"
    elif not cfg.use_jac_sparsity:
        analytic_fallback_reason = "jac_sparsity_disabled"
    elif not cfg.freeze_coarse_alignments:
        analytic_fallback_reason = "coarse_not_frozen"
    elif num_variables != 6 * max(num_frames - 1, 0):
        analytic_fallback_reason = "unexpected_variable_layout"
    else:
        fine_phase_started = perf_counter()
        try:
            analytic_jacobian = _FrozenTorchAnalyticJacobian(
                num_frames=num_frames,
                fixed_frame=fixed_frame,
                frozen_vectorized=frozen_vectorized,
                fine_edge_weight=fine_edge_weight,
                coarse_prior_weight=coarse_prior_weight,
                translation_scale=cfg.translation_scale,
                threads=cfg.analytic_jacobian_threads,
                backend=cfg.analytic_jacobian_backend,
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            analytic_fallback_reason = f"analytic_init_failed:{exc}"
        fine_build_phase_seconds["analytic_jacobian_initialization"] = (
            perf_counter() - fine_phase_started
        )
    if not cfg.analytic_dynamic_jacobian:
        dynamic_analytic_fallback_reason = "config_disabled"
    elif cfg.analytic_dynamic_jacobian_local_rotation_projection not in {
        "strict",
        "scalar_svd",
    }:
        dynamic_analytic_fallback_reason = "projection_mode_mismatch"
    elif not cfg.use_jac_sparsity:
        dynamic_analytic_fallback_reason = "jac_disabled"
    elif cfg.freeze_coarse_alignments:
        dynamic_analytic_fallback_reason = "coarse_frozen"
    elif cfg.solver_workers != 1:
        dynamic_analytic_fallback_reason = "solver_workers_gt_1"
    elif num_variables != 7 * max(num_chunks - 1, 0) + 6 * max(num_frames - 1, 0):
        dynamic_analytic_fallback_reason = "unexpected_variable_layout"
    else:
        fine_phase_started = perf_counter()
        try:
            dynamic_jacobian_type = (
                _DynamicClosedFormAnalyticJacobian
                if cfg.analytic_dynamic_jacobian_backend == "closed_form"
                else _DynamicTorchAnalyticJacobian
            )
            dynamic_analytic_jacobian = dynamic_jacobian_type(
                num_chunks=num_chunks,
                num_frames=num_frames,
                fixed_root=fixed_root,
                fixed_frame=fixed_frame,
                coarse_edges=edges,
                fine_edges=fine_edges,
                prior_records=prior_records,
                coarse_offsets=coarse_offsets,
                frame_offsets=frame_offsets,
                fine_edge_weight=fine_edge_weight,
                coarse_prior_weight=coarse_prior_weight,
                translation_scale=cfg.translation_scale,
                local_rotation_projection=(
                    cfg.analytic_dynamic_jacobian_local_rotation_projection
                ),
                threads=cfg.analytic_jacobian_threads,
            )
        except (ImportError, RuntimeError, ValueError) as exc:
            reason = str(exc)
            dynamic_analytic_fallback_reason = (
                reason
                if reason in {"local_rotation_invalid", "local_pose_nonfinite"}
                else f"analytic_init_failed:{exc}"
            )
        fine_build_phase_seconds["dynamic_analytic_jacobian_initialization"] = (
            perf_counter() - fine_phase_started
        )
    active_analytic_jacobian = (
        analytic_jacobian
        if analytic_jacobian is not None
        else dynamic_analytic_jacobian
    )
    incremental_cache: _FrozenVectorizedResidualCache | None = None
    incremental_fallback_reason: str | None = None
    if factorized_frame_chart_residual_v4:
        incremental_fallback_reason = (
            "factorized_frame_chart_residual_v4_uses_sparse_finite_difference"
        )
    elif active_analytic_jacobian is not None:
        incremental_fallback_reason = "analytic_jacobian_active"
    elif not cfg.incremental_frozen_residual:
        incremental_fallback_reason = "config_disabled"
    elif frozen_vectorized is None:
        incremental_fallback_reason = "frozen_vectorized_unavailable"
    elif not cfg.use_jac_sparsity:
        incremental_fallback_reason = "jac_sparsity_disabled"
    elif not cfg.freeze_coarse_alignments:
        incremental_fallback_reason = "coarse_not_frozen"
    elif cfg.solver_workers != 1:
        incremental_fallback_reason = "solver_workers_gt_1"
    elif num_variables != 6 * max(num_frames - 1, 0):
        incremental_fallback_reason = "unexpected_variable_layout"
    else:
        try:
            incremental_cache = _FrozenVectorizedResidualCache(
                num_frames=num_frames,
                fixed_frame=fixed_frame,
                frozen_vectorized=frozen_vectorized,
                fine_edge_weight=fine_edge_weight,
                coarse_prior_weight=coarse_prior_weight,
                translation_scale=cfg.translation_scale,
            )
        except ValueError as exc:
            incremental_fallback_reason = f"cache_init_failed:{exc}"

    def compute_residual(vector: np.ndarray) -> np.ndarray:
        if factorized_frame_chart_residual_v4:
            _, _, frame_deltas = unpack_factorized_frame_chart_residual_v4(vector)
            residuals: list[np.ndarray] = []
            if edges:
                residuals.append(np.zeros(7 * len(edges), dtype=np.float64))
            assert factorized_fine_constraints is not None
            for frame_a, frame_b, measured, base in factorized_fine_constraints:
                predicted = (
                    frame_deltas[frame_b]
                    .inverse()
                    .compose(base)
                    .compose(frame_deltas[frame_a])
                )
                error = measured.inverse().compose(predicted)
                residuals.append(
                    np.sqrt(fine_edge_weight)
                    * _se3_residual(error, cfg.translation_scale)
                )
            assert factorized_prior_constraints is not None
            for frame, chart_error in factorized_prior_constraints:
                error = chart_error.compose(frame_deltas[frame])
                residuals.append(
                    np.sqrt(coarse_prior_weight)
                    * _se3_residual(error, cfg.translation_scale)
                )
            if not residuals:
                return np.zeros(0, dtype=np.float64)
            return np.concatenate(residuals)
        if incremental_cache is not None:
            return incremental_cache.residual(vector)
        if frozen_vectorized is not None:
            return _compute_frozen_vectorized_residual(
                vector,
                num_frames=num_frames,
                fixed_frame=fixed_frame,
                frozen_vectorized=frozen_vectorized,
                fine_edge_weight=fine_edge_weight,
                coarse_prior_weight=coarse_prior_weight,
                translation_scale=cfg.translation_scale,
                fast_rotvec=(
                    analytic_jacobian is not None
                    and cfg.analytic_fast_residual_logmap
                ),
            )
        if cfg.vectorize_dynamic_residual and dynamic_analytic_jacobian is not None:
            return dynamic_analytic_jacobian.residual(vector)

        coarse, frames = unpack(vector)
        residuals: list[np.ndarray] = []
        for edge in edges:
            residuals.append(_weighted_edge_residual(edge, coarse, cfg.translation_scale))
        fine_constraints = frozen_fine_constraints
        if fine_constraints is None:
            fine_constraints = []
            for chunk_index, frame_a, frame_b, local_a, local_b in fine_edges:
                expected_a = _c2w_to_sim3(coarse[chunk_index].apply_c2w(local_a))
                expected_b = _c2w_to_sim3(coarse[chunk_index].apply_c2w(local_b))
                fine_constraints.append(
                    (frame_a, frame_b, expected_b.inverse().compose(expected_a))
                )
        for frame_a, frame_b, measured in fine_constraints:
            predicted = frames[frame_b].inverse().compose(frames[frame_a])
            error = measured.inverse().compose(predicted)
            residuals.append(
                np.sqrt(fine_edge_weight)
                * _se3_residual(error, cfg.translation_scale)
            )
        prior_constraints = frozen_prior_constraints
        if prior_constraints is None:
            prior_constraints = [
                (
                    frame,
                    _c2w_to_sim3(coarse[chunk_index].apply_c2w(local_pose)),
                )
                for chunk_index, frame, local_pose in prior_records
            ]
        for frame, prior in prior_constraints:
            predicted = frames[frame]
            error = prior.inverse().compose(predicted)
            residuals.append(
                np.sqrt(coarse_prior_weight)
                * _se3_residual(error, cfg.translation_scale)
            )
        if not residuals:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(residuals)

    def residual(vector: np.ndarray) -> np.ndarray:
        nonlocal residual_calls_solver
        nonlocal residual_calls_total
        nonlocal residual_wall_seconds_solver
        nonlocal residual_wall_seconds_total

        with residual_profile_lock:
            solver_call = residual_profile_solver_active
        start = perf_counter()
        try:
            return compute_residual(vector)
        finally:
            elapsed = perf_counter() - start
            with residual_profile_lock:
                residual_calls_total += 1
                residual_wall_seconds_total += elapsed
                if solver_call:
                    residual_calls_solver += 1
                    residual_wall_seconds_solver += elapsed

    if num_variables == 0 or optimizer_row_count == 0:
        zero_vector = np.zeros(num_variables, dtype=np.float64)
        if factorized_frame_chart_residual_v4:
            coarse, frames, _ = unpack_factorized_frame_chart_residual_v4(
                zero_vector
            )
        else:
            coarse, frames = unpack(zero_vector)
        no_solve_full_residual = (
            residual(zero_vector) if row_count else np.zeros(0, dtype=np.float64)
        )
        no_solve_optimizer_residual = optimizer_residual_view(
            no_solve_full_residual
        )
        no_solve_full_robust_cost = _scipy_compatible_robust_cost(
            no_solve_full_residual,
            loss=cfg.robust_loss,
            f_scale=cfg.f_scale,
        )
        no_solve_optimizer_robust_cost = _scipy_compatible_robust_cost(
            no_solve_optimizer_residual,
            loss=cfg.robust_loss,
            f_scale=cfg.f_scale,
        )
        no_solve_diagnostics = {
            "status": "no_variables" if num_variables == 0 else "no_constraints",
            "optimizer_used": False,
            "initial_cost": (
                float(np.mean(no_solve_full_residual**2))
                if no_solve_full_residual.size
                else 0.0
            ),
            "initial_robust_cost": no_solve_full_robust_cost,
            "initial_optimizer_robust_cost": no_solve_optimizer_robust_cost,
            "final_cost": (
                float(np.mean(no_solve_full_residual**2))
                if no_solve_full_residual.size
                else 0.0
            ),
            "final_robust_cost": no_solve_full_robust_cost,
            "final_optimizer_robust_cost": no_solve_optimizer_robust_cost,
            "scipy_cost": no_solve_optimizer_robust_cost,
            "scipy_cost_scope": (
                "variable_fine_prior_rows_only"
                if frozen_coarse_rows_excluded_from_optimizer
                else "full_residual_rows"
            ),
            "full_residual_row_count": int(row_count),
            "optimizer_row_count": int(optimizer_row_count),
            "optimizer_residual_row_start": int(optimizer_residual_row_start),
            "frozen_coarse_rows_excluded_from_optimizer": bool(
                frozen_coarse_rows_excluded_from_optimizer
            ),
            "optimizer_residual_scope": (
                "variable_fine_prior_rows"
                if frozen_coarse_rows_excluded_from_optimizer
                else "full_residual_rows"
            ),
            "robust_loss_row_count": int(optimizer_row_count),
            "robust_loss_scope": "optimizer_residual_rows",
            "num_chunks": num_chunks,
            "num_frames": num_frames,
            "num_coarse_edges": len(edges),
            "num_fine_edges": len(fine_edges),
            "num_frame_priors": len(prior_records),
            "point_corrections_applied": True,
        }
        if cfg.record_residual_components:
            no_solve_diagnostics["residual_components"] = {
                "scope": "full_residual_rows_after_all_residual_weights",
                "coarse_rows_constant": bool(cfg.freeze_coarse_alignments),
                "initial": _residual_component_diagnostics(
                    no_solve_full_residual,
                    len(edges),
                    len(fine_edges),
                    len(prior_records),
                    loss=cfg.robust_loss,
                    f_scale=cfg.f_scale,
                ),
                "final": _residual_component_diagnostics(
                    no_solve_full_residual,
                    len(edges),
                    len(fine_edges),
                    len(prior_records),
                    loss=cfg.robust_loss,
                    f_scale=cfg.f_scale,
                ),
            }
        if factorized_frame_chart_residual_v4:
            no_solve_diagnostics["factorized_frame_chart_residual_v4"] = {
                "schema": "factorized_frame_chart_residual_v4",
                "mode": "frozen_coarse_factorized_frame_chart_residual",
                "enabled": True,
                "valid": True,
                "reason": None,
                "residual_source": "adjacent_path_factorized_frame_chart",
                "fallback_to_absolute_residual": False,
                "identity_delta_local_residual_gate_pass": True,
                "final_materialized_pose_finite": _sim3_sequence_finite(
                    [*coarse, *frames]
                ),
            }
        return HierarchicalAlphaBasis(
            coarse_alignments=coarse,
            frame_poses=frames,
            apply_point_corrections=True,
            diagnostics=no_solve_diagnostics,
        )

    initial_parameterization_started = perf_counter()
    initial_vector = np.zeros(num_variables, dtype=np.float64)
    if not cfg.freeze_coarse_alignments:
        for index in range(1, num_chunks):
            offset = coarse_offsets[index]
            initial_vector[offset : offset + 7] = _sim3_to_parameters(initial_alignments[index])
    if num_frames > 1 and not factorized_frame_chart_residual_v4:
        initial_rotations = np.stack(
            [pose.rotation for pose in initial_frame_poses[1:]], axis=0
        )
        initial_translations = np.stack(
            [pose.translation for pose in initial_frame_poses[1:]], axis=0
        )
        initial_frame_parameters = np.concatenate(
            [
                Rotation.from_matrix(initial_rotations).as_rotvec(),
                initial_translations,
            ],
            axis=1,
        )
        initial_vector[frame_base:] = initial_frame_parameters.reshape(-1)
    elif (
        num_frames > 1
        and factorized_frame_chart_residual_v4
        and cfg.factorized_frame_chart_residual_v4_initial_frame_deltas is not None
    ):
        initial_vector[frame_base:] = _factorized_v4_initial_delta_parameters(
            cfg.factorized_frame_chart_residual_v4_initial_frame_deltas,
            num_frames,
        )
    initial_parameterization_wall_seconds = (
        perf_counter() - initial_parameterization_started
    )

    sparsity_build_started = perf_counter()
    sparsity = (
        _build_jacobian_sparsity(
            num_variables,
            edges,
            fine_edges,
            prior_records,
            coarse_offsets,
            frame_offsets,
            translation_scale=cfg.translation_scale,
            component_exact_frame_blocks=(
                incremental_cache is not None or analytic_jacobian is not None
            ),
        )
        if cfg.use_jac_sparsity and active_analytic_jacobian is None
        else None
    )
    optimizer_sparsity = (
        optimizer_jacobian_view(sparsity) if sparsity is not None else None
    )
    sparsity_build_seconds = perf_counter() - sparsity_build_started
    lower = np.full_like(initial_vector, -np.inf)
    upper = np.full_like(initial_vector, np.inf)
    scale_trust_radii = None
    hierarchy_scale_projection_count = 0
    hierarchy_scale_projection_max_violation = 0.0
    if not cfg.freeze_coarse_alignments:
        if cfg.chunk_scale_trust_mode == "edge_consistency":
            scale_trust_radii = _chunk_scale_trust_radii(
                num_chunks, edges, scale_bound_alignments, cfg=cfg
            )
        for index in range(1, num_chunks):
            scale_index = coarse_offsets[index] + 6
            scale_delta = (
                cfg.max_log_scale_delta
                if scale_trust_radii is None
                else float(scale_trust_radii[index])
            )
            scale_center = float(np.log(scale_bound_alignments[index].scale))
            lower[scale_index] = scale_center - scale_delta
            upper[scale_index] = scale_center + scale_delta
        if hierarchical_warm_start_diagnostics is not None and num_chunks > 1:
            scale_indices = np.fromiter(
                (coarse_offsets[index] + 6 for index in range(1, num_chunks)),
                dtype=np.int64,
                count=num_chunks - 1,
            )
            warm_scales = initial_vector[scale_indices]
            projected_scales = np.clip(
                warm_scales,
                lower[scale_indices],
                upper[scale_indices],
            )
            violations = np.maximum(
                lower[scale_indices] - warm_scales,
                warm_scales - upper[scale_indices],
            )
            hierarchy_scale_projection_count = int(
                np.count_nonzero(projected_scales != warm_scales)
            )
            hierarchy_scale_projection_max_violation = float(
                np.max(np.maximum(violations, 0.0), initial=0.0)
            )
            initial_vector[scale_indices] = projected_scales
            hierarchical_warm_start_diagnostics.update(
                {
                    "scale_projection_count": hierarchy_scale_projection_count,
                    "scale_projection_max_violation": (
                        hierarchy_scale_projection_max_violation
                    ),
                    "scale_projection_policy": (
                        "clip_keyframe_warm_start_to_original_public_bounds"
                    ),
                }
            )
    precomputed_initial_residual: np.ndarray | None = None
    factorized_v4_initialization_diagnostics = None
    if (
        factorized_frame_chart_residual_v4
        and (
            cfg.factorized_frame_chart_residual_v4_initial_frame_deltas is not None
            or cfg.factorized_frame_chart_residual_v4_initial_frame_delta_candidates
            is not None
        )
    ):
        public_zero_vector = initial_vector.copy()
        public_zero_vector[frame_base:] = 0.0
        public_zero_residual = residual(public_zero_vector)
        public_zero_cost = (
            float(np.mean(public_zero_residual**2))
            if public_zero_residual.size
            else 0.0
        )
        public_zero_robust_cost = _scipy_compatible_robust_cost(
            public_zero_residual,
            loss=cfg.robust_loss,
            f_scale=cfg.f_scale,
        )
        if (
            cfg.factorized_frame_chart_residual_v4_initial_frame_delta_candidates
            is not None
        ):
            candidates = []
            best_candidate: dict[str, Any] | None = None
            selected_vector = public_zero_vector
            selected_residual = public_zero_residual
            extra_residual_evaluations = 1
            for alpha, deltas in (
                cfg.factorized_frame_chart_residual_v4_initial_frame_delta_candidates
            ):
                candidate_vector = public_zero_vector.copy()
                alpha_value = float(alpha)
                if alpha_value == 0.0:
                    candidate_residual = public_zero_residual
                else:
                    candidate_vector[frame_base:] = (
                        _factorized_v4_initial_delta_parameters(
                            deltas, num_frames
                        )
                    )
                    candidate_residual = residual(candidate_vector)
                    extra_residual_evaluations += 1
                raw_cost = (
                    float(np.mean(candidate_residual**2))
                    if candidate_residual.size
                    else 0.0
                )
                robust_cost = _scipy_compatible_robust_cost(
                    candidate_residual,
                    loss=cfg.robust_loss,
                    f_scale=cfg.f_scale,
                )
                raw_pass = _factorized_v4_keyframe_level_nonincrease(
                    public_zero_cost,
                    raw_cost,
                )
                robust_pass = _factorized_v4_keyframe_level_nonincrease(
                    public_zero_robust_cost,
                    robust_cost,
                )
                candidate_record = {
                    "alpha": alpha_value,
                    "raw_cost": raw_cost,
                    "robust_cost": robust_cost,
                    "raw_pass": raw_pass,
                    "robust_pass": robust_pass,
                    "passes": bool(raw_pass and robust_pass),
                }
                candidates.append(candidate_record)
                if alpha_value <= 0.0 or not candidate_record["passes"]:
                    continue
                if best_candidate is None or (
                    robust_cost < best_candidate["robust_cost"]
                    - max(1e-12, 1e-10 * abs(best_candidate["robust_cost"]))
                    or (
                        np.isclose(
                            robust_cost,
                            best_candidate["robust_cost"],
                            rtol=1e-10,
                            atol=1e-12,
                        )
                        and alpha_value > best_candidate["alpha"]
                    )
                ):
                    best_candidate = {
                        **candidate_record,
                        "vector": candidate_vector,
                        "residual": candidate_residual,
                    }
            if best_candidate is not None:
                selected_vector = np.asarray(best_candidate["vector"], dtype=np.float64)
                selected_residual = np.asarray(
                    best_candidate["residual"], dtype=np.float64
                )
                selected_alpha = float(best_candidate["alpha"])
                selected_raw_cost = float(best_candidate["raw_cost"])
                selected_robust_cost = float(best_candidate["robust_cost"])
            else:
                selected_alpha = 0.0
                selected_raw_cost = public_zero_cost
                selected_robust_cost = public_zero_robust_cost
            initial_vector = selected_vector
            precomputed_initial_residual = selected_residual
            factorized_v4_initialization_diagnostics = {
                "schema": "factorized_frame_chart_residual_v4_initialization_v1",
                "public_zero_cost": public_zero_cost,
                "public_zero_robust_cost": public_zero_robust_cost,
                "warm_initial_cost": selected_raw_cost,
                "warm_initial_robust_cost": selected_robust_cost,
                "warm_start_accepted": bool(selected_alpha > 0.0),
                "solver_start": (
                    "projected_warm" if selected_alpha > 0.0 else "public_zero"
                ),
                "warm_rejected_reason": (
                    None
                    if selected_alpha > 0.0
                    else "no_positive_alpha_candidate_passed_public_zero_gate"
                ),
                "extra_residual_evaluations": int(extra_residual_evaluations),
                "solver_nfev_budget_unchanged": True,
                "objective_projection": {
                    "schema": (
                        "factorized_frame_chart_residual_v4_objective_projection_v2"
                    ),
                    "public_zero_raw_cost": public_zero_cost,
                    "public_zero_robust_cost": public_zero_robust_cost,
                    "candidates": candidates,
                    "selected_alpha": selected_alpha,
                    "selected_raw_cost": selected_raw_cost,
                    "selected_robust_cost": selected_robust_cost,
                    "alpha_zero_fail_closed": bool(selected_alpha == 0.0),
                    "extra_residual_evaluations": int(
                        extra_residual_evaluations
                    ),
                },
            }
        else:
            warm_initial_residual = residual(initial_vector)
            warm_initial_cost = (
                float(np.mean(warm_initial_residual**2))
                if warm_initial_residual.size
                else 0.0
            )
            warm_initial_robust_cost = _scipy_compatible_robust_cost(
                warm_initial_residual,
                loss=cfg.robust_loss,
                f_scale=cfg.f_scale,
            )
            warm_start_accepted = _factorized_v4_keyframe_level_nonincrease(
                public_zero_cost,
                warm_initial_cost,
            )
            factorized_v4_initialization_diagnostics = {
                "schema": "factorized_frame_chart_residual_v4_initialization_v1",
                "public_zero_cost": public_zero_cost,
                "public_zero_robust_cost": public_zero_robust_cost,
                "warm_initial_cost": warm_initial_cost,
                "warm_initial_robust_cost": warm_initial_robust_cost,
                "warm_start_accepted": bool(warm_start_accepted),
                "solver_start": "warm" if warm_start_accepted else "public_zero",
                "warm_rejected_reason": (
                    None
                    if warm_start_accepted
                    else "warm_initial_cost_exceeds_public_zero_cost"
                ),
                "extra_residual_evaluations": 2,
                "solver_nfev_budget_unchanged": True,
            }
            if warm_start_accepted:
                precomputed_initial_residual = warm_initial_residual
            else:
                initial_vector = public_zero_vector
                precomputed_initial_residual = public_zero_residual
    initial_residual = (
        precomputed_initial_residual
        if precomputed_initial_residual is not None
        else residual(initial_vector)
    )
    initial_optimizer_residual = optimizer_residual_view(initial_residual)
    exact_huber_loss = (
        _ExactSparseOutlierHuberLoss(optimizer_row_count)
        if cfg.robust_loss == "huber"
        else None
    )
    least_squares_kwargs = {
        "bounds": (lower, upper),
        "jac_sparsity": (
            None if active_analytic_jacobian is not None else optimizer_sparsity
        ),
        "loss": exact_huber_loss if exact_huber_loss is not None else cfg.robust_loss,
        "f_scale": float(cfg.f_scale),
        "max_nfev": int(cfg.max_nfev),
        "ftol": float(cfg.solver_ftol),
        "xtol": float(cfg.solver_xtol),
        "gtol": float(cfg.solver_gtol),
        "x_scale": "jac",
    }
    if cfg.joint_lsmr_tolerance is not None:
        least_squares_kwargs.update(
            {
                "tr_solver": "lsmr",
                "tr_options": {
                    "atol": float(cfg.joint_lsmr_tolerance),
                    "btol": float(cfg.joint_lsmr_tolerance),
                },
            }
        )
    with residual_profile_lock:
        residual_profile_solver_active = True
    solver_wall_start = perf_counter()
    solver_workers_runtime_fallback_reason: str | None = None
    frame_relaxation_result = None
    joint_subspace_result = None
    joint_subspace_diagnostics = None
    joint_subspace_expanded_start = None
    joint_subspace_frame_polish_mode_energy = None
    joint_subspace_stage_b_trust = None
    joint_subspace_unrestricted_repair = None
    joint_full_sparse_exact_basis_diagnostics = None
    public_pilot_result = None
    schur_diagnostics_raw = None
    schur_direction = None
    schur_frame_direction = None
    schur_step_diagnostics = None
    schur_frozen_constrained_step_oracle = None
    schur_oracle_extra_residual_evaluations = 0
    schur_oracle_extra_jacobian_evaluations = 0
    schur_trial_residual_evaluations = 0
    schur_active_set_restoration_diagnostics = None
    schur_active_set_restoration_applied = False
    schur_active_set_restoration_rank_direction = None
    schur_final_solver_result = None
    schur_final_solver_diagnostics = None
    schur_final_accepted_x = None
    schur_final_accepted_fun = None
    schur_final_accepted_cost = None
    schur_final_last_step_norm = None
    schur_final_last_coarse_block_rms = None
    schur_final_last_frame_block_rms = None
    schur_final_last_step_measure = None
    schur_final_last_actual_reduction = None
    schur_final_last_reduction_ratio = None
    schur_final_last_cost_before = None
    schur_final_last_alpha = None
    schur_linearization_results: list[Any] = []
    discarded_final_joint_results: list[Any] = []
    final_solve_initial_cache = None
    final_solve_initial_cache_reused = False
    result = None
    try:
        from scipy.optimize import least_squares

        def solve_once(
            start: np.ndarray,
            kwargs: dict[str, Any],
            *,
            initial_cache: Any | None = None,
        ) -> Any:
            nonlocal solver_workers_runtime_fallback_reason

            def solver_residual(vector: np.ndarray) -> np.ndarray:
                return optimizer_residual_view(residual(vector))

            if active_analytic_jacobian is not None:
                solve_residual = solver_residual

                def solve_jacobian(vector: np.ndarray) -> Any:
                    return optimizer_jacobian_view(active_analytic_jacobian(vector))

                if (
                    initial_cache is not None
                    and hasattr(initial_cache, "raw_fun")
                    and hasattr(initial_cache, "raw_jac")
                    and np.array_equal(initial_cache.x, start)
                ):

                    def solve_residual(vector: np.ndarray) -> np.ndarray:
                        if np.array_equal(vector, start):
                            return optimizer_residual_view(initial_cache.raw_fun).copy()
                        return solver_residual(vector)

                    def solve_jacobian(vector: np.ndarray) -> Any:
                        if np.array_equal(vector, start):
                            return optimizer_jacobian_view(initial_cache.raw_jac).copy()
                        return optimizer_jacobian_view(
                            active_analytic_jacobian(vector)
                        )

                return least_squares(
                    solve_residual,
                    start,
                    jac=solve_jacobian,
                    **kwargs,
                )
            if cfg.solver_workers > 1:
                with ThreadPoolExecutor(
                    max_workers=cfg.solver_workers,
                    thread_name_prefix="fafrs-jacobian",
                ) as executor:
                    try:
                        return least_squares(
                            solver_residual,
                            start,
                            workers=executor.map,
                            **kwargs,
                        )
                    except TypeError as exc:
                        if "unexpected keyword argument 'workers'" not in str(exc):
                            raise
                        solver_workers_runtime_fallback_reason = (
                            "scipy_least_squares_workers_unsupported"
                        )
                        return least_squares(solver_residual, start, **kwargs)
            if incremental_cache is not None:
                try:
                    return least_squares(
                        solver_residual,
                        start,
                        workers=incremental_cache.map,
                        **kwargs,
                    )
                except TypeError as exc:
                    if "unexpected keyword argument 'workers'" not in str(exc):
                        raise
                    solver_workers_runtime_fallback_reason = (
                        "scipy_least_squares_workers_unsupported"
                    )
                    return least_squares(solver_residual, start, **kwargs)
            return least_squares(solver_residual, start, **kwargs)

        def solve_joint_subspace(
            start: np.ndarray,
            *,
            max_nfev: int,
        ) -> tuple[Any, np.ndarray, dict[str, Any]]:
            """Jointly optimize coarse nodes and a low-frequency frame basis."""

            from scipy import sparse

            (
                control_frames,
                frame_prolongation,
                scalar_column_norms,
            ) = _joint_frame_subspace_prolongation(
                num_frames,
                starts_ends,
                stride=cfg.joint_subspace_control_stride,
            )
            reduced_frame_variables = int(frame_prolongation.shape[1])
            reduced_variables = int(frame_base + reduced_frame_variables)
            transition = sparse.block_diag(
                (
                    sparse.eye(frame_base, format="csr", dtype=np.float64),
                    frame_prolongation,
                ),
                format="csr",
            )
            if transition.shape != (num_variables, reduced_variables):
                raise RuntimeError("joint subspace transition has invalid shape")

            reduced_initial = np.zeros(reduced_variables, dtype=np.float64)
            reduced_initial[:frame_base] = start[:frame_base]
            reduced_lower = np.full(reduced_variables, -np.inf, dtype=np.float64)
            reduced_upper = np.full(reduced_variables, np.inf, dtype=np.float64)
            reduced_lower[:frame_base] = lower[:frame_base]
            reduced_upper[:frame_base] = upper[:frame_base]

            def expand(reduced: np.ndarray) -> np.ndarray:
                expanded = start.copy()
                expanded[:frame_base] = reduced[:frame_base]
                expanded[frame_base:] += frame_prolongation @ reduced[frame_base:]
                return expanded

            def reduced_residual(reduced: np.ndarray) -> np.ndarray:
                return residual(expand(reduced))

            reduced_kwargs = {
                **least_squares_kwargs,
                "bounds": (reduced_lower, reduced_upper),
                "max_nfev": int(max_nfev),
                "jac_sparsity": None,
            }
            if active_analytic_jacobian is not None:

                def reduced_jacobian(reduced: np.ndarray) -> Any:
                    full_jacobian = active_analytic_jacobian(expand(reduced))
                    projected = sparse.hstack(
                        (
                            full_jacobian[:, :frame_base],
                            full_jacobian[:, frame_base:num_variables]
                            @ frame_prolongation,
                        ),
                        format="csr",
                    )
                    return (
                        projected.tocsr()
                        if sparse.issparse(projected)
                        else sparse.csr_matrix(
                            np.asarray(projected, dtype=np.float64)
                        )
                    )

                reduced_result = least_squares(
                    reduced_residual,
                    reduced_initial,
                    jac=reduced_jacobian,
                    **reduced_kwargs,
                )
                jacobian_mode = "full_analytic_chain_rule"
            else:
                reduced_sparsity = None
                if sparsity is not None:
                    reduced_sparsity = (
                        sparsity.astype(np.int8)
                        @ transition.astype(bool).astype(np.int8)
                    ).astype(bool)
                reduced_kwargs["jac_sparsity"] = reduced_sparsity
                reduced_result = least_squares(
                    reduced_residual,
                    reduced_initial,
                    **reduced_kwargs,
                )
                jacobian_mode = (
                    "finite_difference_reduced_sparsity"
                    if reduced_sparsity is not None
                    else "finite_difference_dense"
                )

            raw_expanded = expand(reduced_result.x)
            raw_candidate_residual = np.asarray(
                reduced_result.fun, dtype=np.float64
            ).reshape(-1)
            start_residual = (
                initial_residual
                if np.array_equal(start, initial_vector)
                else residual(start)
            )
            initial_cost = _scipy_compatible_robust_cost(
                start_residual,
                loss=cfg.robust_loss,
                f_scale=cfg.f_scale,
            )
            coarse_row_stop = 7 * len(edges)
            coarse_before = start_residual[:coarse_row_stop]
            coarse_cost_before = float(coarse_before @ coarse_before)
            coarse_statistics_before = _coarse_edge_residual_statistics(
                coarse_before, edges
            )
            local_statistics_before = _joint_local_residual_statistics(
                start_residual,
                num_coarse_edges=len(edges),
                num_fine_edges=len(fine_edges),
                num_frame_priors=len(prior_records),
                fine_edge_weight=fine_edge_weight,
                coarse_prior_weight=coarse_prior_weight,
                translation_scale=cfg.translation_scale,
            )
            raw_direction = raw_expanded - start
            backtracking_trials: list[dict[str, Any]] = []
            selected_snapshot = None
            raw_snapshot = None
            for backtrack, alpha in enumerate((1.0, 0.5, 0.25, 0.125)):
                trial_vector = np.clip(start + alpha * raw_direction, lower, upper)
                trial_residual = (
                    raw_candidate_residual
                    if backtrack == 0
                    else residual(trial_vector)
                )
                trial_cost = _scipy_compatible_robust_cost(
                    trial_residual,
                    loss=cfg.robust_loss,
                    f_scale=cfg.f_scale,
                )
                trial_coarse = trial_residual[:coarse_row_stop]
                trial_coarse_cost = float(trial_coarse @ trial_coarse)
                trial_coarse_ratio = (
                    trial_coarse_cost / coarse_cost_before
                    if coarse_cost_before > 0.0
                    else 1.0
                    if trial_coarse_cost == 0.0
                    else np.inf
                )
                trial_statistics_ratios = _coarse_edge_statistics_ratios(
                    coarse_statistics_before,
                    _coarse_edge_residual_statistics(trial_coarse, edges),
                )
                trial_local_statistics = _joint_local_residual_statistics(
                    trial_residual,
                    num_coarse_edges=len(edges),
                    num_fine_edges=len(fine_edges),
                    num_frame_priors=len(prior_records),
                    fine_edge_weight=fine_edge_weight,
                    coarse_prior_weight=coarse_prior_weight,
                    translation_scale=cfg.translation_scale,
                )
                trial_local_statistics_ratios = (
                    _joint_local_residual_statistics_ratios(
                        local_statistics_before,
                        trial_local_statistics,
                    )
                )
                trial_family_checks = (
                    (
                        cfg.joint_schur_step_max_adjacent_p95_ratio,
                        trial_statistics_ratios["adjacent"]["p95"],
                    ),
                    (
                        cfg.joint_schur_step_max_loop_p95_ratio,
                        trial_statistics_ratios["loop"]["p95"],
                    ),
                    (
                        cfg.joint_schur_step_max_loop_max_ratio,
                        trial_statistics_ratios["loop"]["max"],
                    ),
                )
                trial_family_gate = all(
                    threshold is None
                    or observed is None
                    or observed <= threshold
                    for threshold, observed in trial_family_checks
                )
                trial_cost_gate = bool(
                    trial_cost
                    <= initial_cost + max(1e-12, 1e-10 * abs(initial_cost))
                )
                trial_aggregate_gate = bool(
                    trial_coarse_ratio
                    <= cfg.joint_schur_step_max_graph_cost_ratio
                )
                trial_accepted = bool(
                    np.all(np.isfinite(trial_vector))
                    and np.all(np.isfinite(trial_residual))
                    and np.isfinite(trial_cost)
                    and trial_cost_gate
                    and trial_aggregate_gate
                    and trial_family_gate
                )
                snapshot = (
                    trial_vector,
                    trial_residual,
                    trial_cost,
                    trial_coarse_cost,
                    trial_coarse_ratio,
                    trial_statistics_ratios,
                    trial_cost_gate,
                    trial_aggregate_gate,
                    trial_family_gate,
                    alpha,
                )
                if raw_snapshot is None:
                    raw_snapshot = snapshot
                backtracking_trials.append(
                    {
                        "backtrack": int(backtrack),
                        "alpha": float(alpha),
                        "full_robust_cost": float(trial_cost),
                        "robust_cost_nonincreasing": trial_cost_gate,
                        "coarse_graph_cost_ratio": float(trial_coarse_ratio),
                        "aggregate_graph_gate_passed": trial_aggregate_gate,
                        "coarse_edge_statistics_ratios": (
                            trial_statistics_ratios
                        ),
                        "local_residual_statistics_ratios": (
                            trial_local_statistics_ratios
                        ),
                        "edge_family_gate_passed": trial_family_gate,
                        "accepted": trial_accepted,
                    }
                )
                if trial_accepted:
                    selected_snapshot = snapshot
                    break
            assert raw_snapshot is not None
            accepted = selected_snapshot is not None
            (
                expanded,
                candidate_residual,
                candidate_cost,
                coarse_cost_after,
                coarse_cost_ratio,
                coarse_statistics_ratios,
                robust_cost_nonincreasing,
                aggregate_gate_passed,
                family_gate_passed,
                accepted_alpha,
            ) = selected_snapshot if selected_snapshot is not None else raw_snapshot
            candidate_local_statistics = _joint_local_residual_statistics(
                candidate_residual,
                num_coarse_edges=len(edges),
                num_fine_edges=len(fine_edges),
                num_frame_priors=len(prior_records),
                fine_edge_weight=fine_edge_weight,
                coarse_prior_weight=coarse_prior_weight,
                translation_scale=cfg.translation_scale,
            )
            candidate_local_statistics_ratios = (
                _joint_local_residual_statistics_ratios(
                    local_statistics_before,
                    candidate_local_statistics,
                )
            )
            physical_frame_step = np.asarray(
                expanded[frame_base:] - start[frame_base:], dtype=np.float64
            ).reshape(-1, 6)
            physical_block_norms = np.linalg.norm(physical_frame_step, axis=1)
            reduced_gradient = np.asarray(
                reduced_result.grad, dtype=np.float64
            ).reshape(-1)
            scale_indices = np.arange(6, frame_base, 7, dtype=np.int64)
            active_scale_bounds = int(
                np.count_nonzero(
                    np.isclose(
                        expanded[scale_indices],
                        lower[scale_indices],
                        rtol=0.0,
                        atol=1e-12,
                    )
                    | np.isclose(
                        expanded[scale_indices],
                        upper[scale_indices],
                        rtol=0.0,
                        atol=1e-12,
                    )
                )
            )
            diagnostics = {
                "schema": "hierarchical_joint_full_residual_subspace_v1",
                "interpretation": (
                    "same_full_residual_graph_with_frame_corrections_restricted_"
                    "to_a_column_normalized_piecewise_linear_subspace"
                ),
                "accepted": accepted,
                "decision": (
                    "continue_conditional_frame_polish_from_subspace_solution"
                    if accepted
                    and cfg.joint_subspace_conditional_frame_polish
                    else "continue_full_joint_from_subspace_solution"
                    if accepted
                    else "discard_subspace_solution_and_use_public_initialization"
                ),
                "accepted_alpha": (
                    float(accepted_alpha) if accepted else None
                ),
                "backtracking_trials": backtracking_trials,
                "jacobian_mode": jacobian_mode,
                "frame_parameterization": "f_initial_plus_normalized_P_times_z",
                "control_policy": "uniform_plus_chunk_and_sequence_endpoints",
                "control_stride": int(cfg.joint_subspace_control_stride),
                "num_control_frames": int(len(control_frames)),
                "num_full_variables": int(num_variables),
                "num_reduced_variables": reduced_variables,
                "num_full_residuals": int(row_count),
                "prolongation_nnz": int(frame_prolongation.nnz),
                "scalar_column_norm_min": float(np.min(scalar_column_norms)),
                "scalar_column_norm_median": float(
                    np.median(scalar_column_norms)
                ),
                "scalar_column_norm_max": float(np.max(scalar_column_norms)),
                "requested_max_nfev": int(max_nfev),
                "num_function_evaluations": int(reduced_result.nfev),
                "num_jacobian_evaluations": (
                    None
                    if getattr(reduced_result, "njev", None) is None
                    else int(reduced_result.njev)
                ),
                "success": bool(reduced_result.success),
                "termination_status": int(reduced_result.status),
                "message": str(reduced_result.message),
                "initial_full_robust_cost": float(initial_cost),
                "candidate_full_robust_cost": float(candidate_cost),
                "robust_cost_nonincreasing": robust_cost_nonincreasing,
                "coarse_graph_squared_residual_before": coarse_cost_before,
                "coarse_graph_squared_residual_after": coarse_cost_after,
                "coarse_graph_cost_ratio": float(coarse_cost_ratio),
                "maximum_coarse_graph_cost_ratio": float(
                    cfg.joint_schur_step_max_graph_cost_ratio
                ),
                "coarse_edge_statistics_ratios": coarse_statistics_ratios,
                "initial_local_residual_statistics": local_statistics_before,
                "candidate_local_residual_statistics": (
                    candidate_local_statistics
                ),
                "candidate_local_residual_statistics_ratios": (
                    candidate_local_statistics_ratios
                ),
                "aggregate_graph_gate_passed": aggregate_gate_passed,
                "edge_family_gate_passed": family_gate_passed,
                "coarse_gradient_inf": float(
                    np.linalg.norm(reduced_gradient[:frame_base], ord=np.inf)
                ),
                "subspace_frame_gradient_inf": float(
                    np.linalg.norm(reduced_gradient[frame_base:], ord=np.inf)
                ),
                "frame_step_block_rms": float(
                    np.linalg.norm(physical_frame_step)
                    / np.sqrt(max(len(physical_frame_step), 1))
                ),
                "frame_step_block_p95": float(
                    np.quantile(physical_block_norms, 0.95)
                ),
                "frame_step_block_max": float(
                    np.max(physical_block_norms, initial=0.0)
                ),
                "active_coarse_scale_bound_count": active_scale_bounds,
                "coordinate_caveat": (
                    "Frame block norms combine local rotvec radians and "
                    "trajectory translation units and are diagnostic only."
                ),
            }
            return reduced_result, expanded, diagnostics

        def fixed_block_bounds(
            start: np.ndarray, block_slice: slice
        ) -> tuple[np.ndarray, np.ndarray]:
            block_lower = lower.copy()
            block_upper = upper.copy()
            values = start[block_slice]
            radius = 1e-12 * (1.0 + np.abs(values))
            block_lower[block_slice] = np.maximum(
                block_lower[block_slice], values - radius
            )
            block_upper[block_slice] = np.minimum(
                block_upper[block_slice], values + radius
            )
            return block_lower, block_upper

        def relax_frames(start: np.ndarray, *, max_nfev: int) -> Any:
            relaxation_kwargs = {
                **least_squares_kwargs,
                "bounds": fixed_block_bounds(start, slice(0, frame_base)),
                "max_nfev": int(max_nfev),
            }
            return solve_once(start, relaxation_kwargs)

        def relax_frames_p64_proximal(
            start: np.ndarray,
            *,
            max_nfev: int,
            rotation_trust_radius: float,
            translation_trust_radius: float,
        ) -> tuple[Any, dict[str, Any]]:
            """Polish frames while penalizing their exact P64 complement."""

            from types import SimpleNamespace

            from scipy import sparse

            proximal_started = perf_counter()
            (
                control_frames,
                frame_prolongation,
                _,
            ) = _joint_frame_subspace_prolongation(
                num_frames,
                starts_ends,
                stride=cfg.joint_subspace_control_stride,
            )
            frame_variables = int(num_variables - frame_base)
            latent_variables = int(frame_prolongation.shape[1])
            if frame_prolongation.shape[0] != frame_variables:
                raise RuntimeError("P64 proximal frame shape mismatch")
            num_frame_blocks = max(num_frames - 1, 1)
            family_normalization = 1.0 / np.sqrt(num_frame_blocks)
            component_weights = np.tile(
                np.asarray(
                    [
                        *([family_normalization / rotation_trust_radius] * 3),
                        *(
                            [
                                family_normalization
                                / translation_trust_radius
                            ]
                            * 3
                        ),
                    ],
                    dtype=np.float64,
                ),
                num_frames - 1,
            )
            weight_matrix = sparse.diags(component_weights, format="csr")
            proximal_jacobian = sparse.hstack(
                (
                    sparse.csr_matrix((frame_variables, latent_variables)),
                    weight_matrix,
                ),
                format="csr",
            )
            augmented_variables = int(latent_variables + frame_variables)
            augmented_initial = np.zeros(
                augmented_variables, dtype=np.float64
            )
            augmented_lower = np.full(
                augmented_variables, -np.inf, dtype=np.float64
            )
            augmented_upper = np.full(
                augmented_variables, np.inf, dtype=np.float64
            )

            def split_augmented(
                vector: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                latent = vector[:latent_variables]
                complement = vector[latent_variables:]
                physical = start.copy()
                physical[frame_base:] += (
                    frame_prolongation @ latent + complement
                )
                return physical, latent, np.asarray(complement).reshape(-1)

            def augmented_residual(vector: np.ndarray) -> np.ndarray:
                physical, _, complement = split_augmented(vector)
                return np.concatenate(
                    (residual(physical), component_weights * complement)
                )

            total_residuals = int(row_count + frame_variables)
            mixed_loss = _MixedPhysicalRobustLinearLoss(
                row_count,
                total_residuals,
                physical_loss=cfg.robust_loss,
            )
            augmented_kwargs = {
                **least_squares_kwargs,
                "bounds": (augmented_lower, augmented_upper),
                "loss": mixed_loss,
                "max_nfev": int(max_nfev),
                "jac_sparsity": None,
            }
            jacobian_build_seconds = 0.0
            if active_analytic_jacobian is not None:

                def augmented_jacobian(vector: np.ndarray) -> Any:
                    nonlocal jacobian_build_seconds
                    started = perf_counter()
                    physical, _, _ = split_augmented(vector)
                    physical_jacobian = active_analytic_jacobian(physical)
                    if sparse.issparse(physical_jacobian):
                        frame_jacobian = physical_jacobian[
                            :, frame_base:num_variables
                        ].tocsr()
                    else:
                        frame_jacobian = sparse.csr_matrix(
                            np.asarray(
                                physical_jacobian, dtype=np.float64
                            )[:, frame_base:num_variables]
                        )
                    augmented = sparse.vstack(
                        (
                            sparse.hstack(
                                (
                                    frame_jacobian @ frame_prolongation,
                                    frame_jacobian,
                                ),
                                format="csr",
                            ),
                            proximal_jacobian,
                        ),
                        format="csr",
                    )
                    jacobian_build_seconds += perf_counter() - started
                    return augmented

                augmented_result = least_squares(
                    augmented_residual,
                    augmented_initial,
                    jac=augmented_jacobian,
                    **augmented_kwargs,
                )
                jacobian_mode = "mixed_loss_sparse_analytic_plus_exact_p64"
            else:
                if sparsity is not None:
                    frame_sparsity = sparsity[:, frame_base:num_variables]
                    if not sparse.issparse(frame_sparsity):
                        frame_sparsity = sparse.csr_matrix(frame_sparsity)
                    frame_sparsity = frame_sparsity.astype(bool).tocsr()
                    prolongation_pattern = frame_prolongation.astype(bool)
                    latent_sparsity = (
                        frame_sparsity.astype(np.int8)
                        @ prolongation_pattern.astype(np.int8)
                    ).astype(bool)
                    augmented_sparsity = sparse.vstack(
                        (
                            sparse.hstack(
                                (
                                    latent_sparsity,
                                    frame_sparsity,
                                ),
                                format="csr",
                            ),
                            proximal_jacobian.astype(bool),
                        ),
                        format="csr",
                    ).astype(bool)
                else:
                    raise RuntimeError(
                        "P64 proximal frame polish requires an analytic "
                        "Jacobian or sparse finite-difference pattern"
                    )
                augmented_kwargs["jac_sparsity"] = augmented_sparsity
                augmented_result = least_squares(
                    augmented_residual,
                    augmented_initial,
                    **augmented_kwargs,
                )
                jacobian_mode = (
                    "mixed_loss_sparse_finite_difference_plus_exact_p64"
                )

            physical_x, _latent_x, complement = split_augmented(
                augmented_result.x
            )
            physical_fun = residual(physical_x)
            physical_cost = _scipy_compatible_robust_cost(
                physical_fun,
                loss=cfg.robust_loss,
                f_scale=cfg.f_scale,
            )
            proximal_residual = component_weights * complement
            proximal_cost = float(
                0.5 * np.dot(proximal_residual, proximal_residual)
            )
            proximal_gradient_q = component_weights * proximal_residual
            physical_gradient = np.zeros(num_variables, dtype=np.float64)
            physical_gradient[frame_base:] = np.asarray(
                augmented_result.grad[latent_variables:], dtype=np.float64
            ) - proximal_gradient_q
            physical_frame_jacobian = augmented_result.jac[
                :row_count, latent_variables:
            ]
            if not sparse.issparse(physical_frame_jacobian):
                physical_frame_jacobian = sparse.csr_matrix(
                    np.asarray(
                        physical_frame_jacobian, dtype=np.float64
                    )
                )
            physical_jacobian = sparse.hstack(
                (
                    sparse.csr_matrix((row_count, frame_base)),
                    physical_frame_jacobian,
                ),
                format="csr",
            )
            physical_result = SimpleNamespace(
                x=np.asarray(physical_x, dtype=np.float64).copy(),
                fun=np.asarray(physical_fun, dtype=np.float64),
                jac=physical_jacobian,
                grad=physical_gradient,
                cost=float(physical_cost),
                optimality=float(
                    np.linalg.norm(physical_gradient, ord=np.inf)
                ),
                active_mask=np.zeros(num_variables, dtype=np.int8),
                nfev=int(augmented_result.nfev),
                njev=(
                    None
                    if getattr(augmented_result, "njev", None) is None
                    else int(augmented_result.njev)
                ),
                success=bool(augmented_result.success),
                status=int(augmented_result.status),
                message=str(augmented_result.message),
            )
            complement_blocks = complement.reshape(-1, 6)
            latent_gradient = np.asarray(
                augmented_result.grad[:latent_variables], dtype=np.float64
            ).reshape(-1)
            diagnostics = {
                "schema": "hierarchical_joint_stage_b_frame_trust_v2",
                "enabled": True,
                "valid": True,
                "mode": "p64_direct_z_plus_q_proximal",
                "parameterization": "delta_frame=P64_z+q",
                "interpretation": (
                    "stage_b_penalizes_only_the_frame_correction_component_"
                    "outside_the_p64_low_frequency_column_space"
                ),
                "exact_p64_complement": True,
                "family_normalization": "mean_over_frame_blocks",
                "physical_rows": int(row_count),
                "proximal_rows": frame_variables,
                "physical_variables": int(num_variables),
                "latent_variables": latent_variables,
                "high_frequency_variables": frame_variables,
                "augmented_variables": augmented_variables,
                "num_control_frames": int(len(control_frames)),
                "jacobian_mode": jacobian_mode,
                "proximal_jacobian_nnz": int(proximal_jacobian.nnz),
                "jacobian_build_seconds": float(jacobian_build_seconds),
                "mixed_loss_calls": int(mixed_loss.calls),
                "physical_scipy_cost": float(physical_cost),
                "proximal_cost": proximal_cost,
                "augmented_scipy_cost": float(physical_cost + proximal_cost),
                "augmented_result_cost": float(augmented_result.cost),
                "augmented_optimality": float(augmented_result.optimality),
                "latent_gradient_inf": float(
                    np.linalg.norm(latent_gradient, ord=np.inf)
                ),
                "projection_residual_rotation_p95_ratio": float(
                    np.quantile(
                        np.linalg.norm(complement_blocks[:, :3], axis=1)
                        / rotation_trust_radius,
                        0.95,
                    )
                ),
                "projection_residual_translation_p95_ratio": float(
                    np.quantile(
                        np.linalg.norm(complement_blocks[:, 3:], axis=1)
                        / translation_trust_radius,
                        0.95,
                    )
                ),
                "wall_seconds": float(perf_counter() - proximal_started),
            }
            return physical_result, diagnostics

        def relax_frames_p64_gauge_fix(
            start: np.ndarray,
            *,
            max_nfev: int,
            rotation_scale: float,
            translation_scale: float,
        ) -> tuple[Any, dict[str, Any]]:
            """Polish frames with an unbiased P64 gauge-fixing residual."""

            from types import SimpleNamespace

            from scipy import sparse

            gauge_started = perf_counter()
            (
                control_frames,
                frame_prolongation,
                _,
            ) = _joint_frame_subspace_prolongation(
                num_frames,
                starts_ends,
                stride=cfg.joint_subspace_control_stride,
            )
            frame_variables = int(num_variables - frame_base)
            latent_variables = int(frame_prolongation.shape[1])
            if frame_prolongation.shape[0] != frame_variables:
                raise RuntimeError("P64 gauge-fix frame shape mismatch")
            component_weights = np.tile(
                np.asarray(
                    [
                        *([1.0 / rotation_scale] * 3),
                        *([1.0 / translation_scale] * 3),
                    ],
                    dtype=np.float64,
                ),
                num_frames - 1,
            )
            d_squared = component_weights * component_weights
            weighted_prolongation = sparse.diags(
                d_squared, format="csr"
            ) @ frame_prolongation
            gauge_normal_matrix = (
                frame_prolongation.T @ weighted_prolongation
            ).tocsr()
            gauge_normal_diag = np.asarray(
                gauge_normal_matrix.diagonal(), dtype=np.float64
            )
            if (
                gauge_normal_diag.size != latent_variables
                or np.any(gauge_normal_diag <= 0.0)
                or not np.all(np.isfinite(gauge_normal_diag))
            ):
                raise RuntimeError("P64 gauge-fix normal diagonal invalid")
            gauge_row_scale = sparse.diags(
                1.0 / np.sqrt(gauge_normal_diag), format="csr"
            )
            gauge_matrix = (
                gauge_row_scale
                @ (frame_prolongation.T @ sparse.diags(d_squared, format="csr"))
            ).tocsr()
            gauge_jacobian = sparse.hstack(
                (
                    sparse.csr_matrix((latent_variables, latent_variables)),
                    gauge_matrix,
                ),
                format="csr",
            )
            augmented_variables = int(latent_variables + frame_variables)
            augmented_initial = np.zeros(
                augmented_variables, dtype=np.float64
            )
            augmented_lower = np.full(
                augmented_variables, -np.inf, dtype=np.float64
            )
            augmented_upper = np.full(
                augmented_variables, np.inf, dtype=np.float64
            )

            def split_augmented(
                vector: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                latent = vector[:latent_variables]
                complement = vector[latent_variables:]
                physical = start.copy()
                physical[frame_base:] += (
                    frame_prolongation @ latent + complement
                )
                return physical, latent, np.asarray(complement).reshape(-1)

            def gauge_residual_from_complement(
                complement: np.ndarray,
            ) -> np.ndarray:
                return np.asarray(gauge_matrix @ complement).reshape(-1)

            def augmented_residual(vector: np.ndarray) -> np.ndarray:
                physical, _, complement = split_augmented(vector)
                return np.concatenate(
                    (residual(physical), gauge_residual_from_complement(complement))
                )

            total_residuals = int(row_count + latent_variables)
            mixed_loss = _MixedPhysicalRobustLinearLoss(
                row_count,
                total_residuals,
                physical_loss=cfg.robust_loss,
            )
            augmented_kwargs = {
                **least_squares_kwargs,
                "bounds": (augmented_lower, augmented_upper),
                "loss": mixed_loss,
                "max_nfev": int(max_nfev),
                "jac_sparsity": None,
            }
            jacobian_build_seconds = 0.0
            if active_analytic_jacobian is not None:

                def augmented_jacobian(vector: np.ndarray) -> Any:
                    nonlocal jacobian_build_seconds
                    started = perf_counter()
                    physical, _, _ = split_augmented(vector)
                    physical_jacobian = active_analytic_jacobian(physical)
                    if sparse.issparse(physical_jacobian):
                        frame_jacobian = physical_jacobian[
                            :, frame_base:num_variables
                        ].tocsr()
                    else:
                        frame_jacobian = sparse.csr_matrix(
                            np.asarray(
                                physical_jacobian, dtype=np.float64
                            )[:, frame_base:num_variables]
                        )
                    augmented = sparse.vstack(
                        (
                            sparse.hstack(
                                (
                                    frame_jacobian @ frame_prolongation,
                                    frame_jacobian,
                                ),
                                format="csr",
                            ),
                            gauge_jacobian,
                        ),
                        format="csr",
                    )
                    jacobian_build_seconds += perf_counter() - started
                    return augmented

                augmented_result = least_squares(
                    augmented_residual,
                    augmented_initial,
                    jac=augmented_jacobian,
                    **augmented_kwargs,
                )
                jacobian_mode = (
                    "mixed_loss_sparse_analytic_plus_p64_gauge_fix"
                )
            else:
                if sparsity is not None:
                    frame_sparsity = sparsity[:, frame_base:num_variables]
                    if not sparse.issparse(frame_sparsity):
                        frame_sparsity = sparse.csr_matrix(frame_sparsity)
                    frame_sparsity = frame_sparsity.astype(bool).tocsr()
                    prolongation_pattern = frame_prolongation.astype(bool)
                    latent_sparsity = (
                        frame_sparsity.astype(np.int8)
                        @ prolongation_pattern.astype(np.int8)
                    ).astype(bool)
                    augmented_sparsity = sparse.vstack(
                        (
                            sparse.hstack(
                                (
                                    latent_sparsity,
                                    frame_sparsity,
                                ),
                                format="csr",
                            ),
                            gauge_jacobian.astype(bool),
                        ),
                        format="csr",
                    ).astype(bool)
                else:
                    raise RuntimeError(
                        "P64 gauge-fix frame polish requires an analytic "
                        "Jacobian or sparse finite-difference pattern"
                    )
                augmented_kwargs["jac_sparsity"] = augmented_sparsity
                augmented_result = least_squares(
                    augmented_residual,
                    augmented_initial,
                    **augmented_kwargs,
                )
                jacobian_mode = (
                    "mixed_loss_sparse_finite_difference_plus_p64_gauge_fix"
                )

            physical_x, _latent_x, complement = split_augmented(
                augmented_result.x
            )
            physical_fun = residual(physical_x)
            physical_cost = _scipy_compatible_robust_cost(
                physical_fun,
                loss=cfg.robust_loss,
                f_scale=cfg.f_scale,
            )
            gauge_residual = gauge_residual_from_complement(complement)
            gauge_cost = float(0.5 * np.dot(gauge_residual, gauge_residual))
            gauge_gradient_q = np.asarray(
                gauge_matrix.T @ gauge_residual, dtype=np.float64
            ).reshape(-1)
            physical_gradient = np.zeros(num_variables, dtype=np.float64)
            physical_gradient[frame_base:] = np.asarray(
                augmented_result.grad[latent_variables:], dtype=np.float64
            ) - gauge_gradient_q
            physical_frame_jacobian = augmented_result.jac[
                :row_count, latent_variables:
            ]
            if not sparse.issparse(physical_frame_jacobian):
                physical_frame_jacobian = sparse.csr_matrix(
                    np.asarray(
                        physical_frame_jacobian, dtype=np.float64
                    )
                )
            physical_jacobian = sparse.hstack(
                (
                    sparse.csr_matrix((row_count, frame_base)),
                    physical_frame_jacobian,
                ),
                format="csr",
            )
            physical_result = SimpleNamespace(
                x=np.asarray(physical_x, dtype=np.float64).copy(),
                fun=np.asarray(physical_fun, dtype=np.float64),
                jac=physical_jacobian,
                grad=physical_gradient,
                cost=float(physical_cost),
                optimality=float(
                    np.linalg.norm(physical_gradient, ord=np.inf)
                ),
                active_mask=np.zeros(num_variables, dtype=np.int8),
                nfev=int(augmented_result.nfev),
                njev=(
                    None
                    if getattr(augmented_result, "njev", None) is None
                    else int(augmented_result.njev)
                ),
                success=bool(augmented_result.success),
                status=int(augmented_result.status),
                message=str(augmented_result.message),
            )
            complement_blocks = complement.reshape(-1, 6)
            weighted_overlap = np.asarray(
                frame_prolongation.T @ (d_squared * complement),
                dtype=np.float64,
            ).reshape(-1)
            diagnostics = {
                "schema": "hierarchical_joint_stage_b_frame_trust_v3",
                "enabled": True,
                "valid": True,
                "mode": "p64_unbiased_low_frequency_gauge_fix",
                "parameterization": "delta_frame=P64_z+q",
                "role": "unbiased_gauge_fixing_not_high_frequency_regularization",
                "interpretation": (
                    "stage_b_chooses_a_p64_decomposition_gauge_without_"
                    "penalizing_any_physical_frame_delta"
                ),
                "unbiased_physical_optimum": True,
                "unbiasedness_argument": (
                    "for_any_physical_delta_choose_weighted_least_squares_z_"
                    "so_that_G_times_q_is_zero"
                ),
                "family_normalization": "diag_normalized_weighted_p64_overlap",
                "physical_rows": int(row_count),
                "gauge_rows": latent_variables,
                "physical_variables": int(num_variables),
                "latent_variables": latent_variables,
                "high_frequency_variables": frame_variables,
                "augmented_variables": augmented_variables,
                "num_control_frames": int(len(control_frames)),
                "jacobian_mode": jacobian_mode,
                "gauge_jacobian_nnz": int(gauge_jacobian.nnz),
                "gauge_matrix_nnz": int(gauge_matrix.nnz),
                "jacobian_build_seconds": float(jacobian_build_seconds),
                "mixed_loss_calls": int(mixed_loss.calls),
                "physical_scipy_cost": float(physical_cost),
                "gauge_cost": gauge_cost,
                "augmented_scipy_cost": float(physical_cost + gauge_cost),
                "augmented_result_cost": float(augmented_result.cost),
                "augmented_optimality": float(augmented_result.optimality),
                "num_function_evaluations": int(augmented_result.nfev),
                "num_jacobian_evaluations": (
                    None
                    if getattr(augmented_result, "njev", None) is None
                    else int(augmented_result.njev)
                ),
                "latent_gradient_inf": float(
                    np.linalg.norm(
                        augmented_result.grad[:latent_variables], ord=np.inf
                    )
                ),
                "gauge_inf": float(np.linalg.norm(gauge_residual, ord=np.inf)),
                "weighted_overlap_inf": float(
                    np.linalg.norm(weighted_overlap, ord=np.inf)
                ),
                "gauge_normal_diag_condition_proxy": float(
                    np.max(gauge_normal_diag) / np.min(gauge_normal_diag)
                ),
                "rotation_scale": float(rotation_scale),
                "translation_scale": float(translation_scale),
                "q_rotation_p95_ratio": float(
                    np.quantile(
                        np.linalg.norm(complement_blocks[:, :3], axis=1)
                        / rotation_scale,
                        0.95,
                    )
                ),
                "q_translation_p95_ratio": float(
                    np.quantile(
                        np.linalg.norm(complement_blocks[:, 3:], axis=1)
                        / translation_scale,
                        0.95,
                    )
                ),
                "wall_seconds": float(perf_counter() - gauge_started),
            }
            return physical_result, diagnostics

        def relax_frames_p64_sparse_basis(
            start: np.ndarray,
            *,
            max_nfev: int,
        ) -> tuple[Any, dict[str, Any]]:
            """Polish frames in a square sparse hierarchical coordinate basis."""

            from types import SimpleNamespace

            from scipy import sparse

            if active_analytic_jacobian is None:
                raise RuntimeError(
                    "P64 sparse exact basis frame polish requires an "
                    "analytic Jacobian"
                )
            sparse_basis_started = perf_counter()
            basis = _joint_frame_sparse_exact_basis(
                num_frames,
                starts_ends,
                stride=cfg.joint_subspace_control_stride,
            )
            frame_transform = basis["frame_transform"]
            frame_variables = int(num_variables - frame_base)
            if frame_transform.shape != (frame_variables, frame_variables):
                raise RuntimeError("P64 sparse exact basis frame shape mismatch")
            coordinate_initial = np.zeros(frame_variables, dtype=np.float64)
            coordinate_lower = np.full(
                frame_variables, -np.inf, dtype=np.float64
            )
            coordinate_upper = np.full(
                frame_variables, np.inf, dtype=np.float64
            )

            def physical_from_coordinates(
                coordinates: np.ndarray,
            ) -> np.ndarray:
                physical = start.copy()
                physical[frame_base:] += frame_transform @ coordinates
                return physical

            def coordinate_residual(coordinates: np.ndarray) -> np.ndarray:
                return residual(physical_from_coordinates(coordinates))

            solver_jacobian_seconds = 0.0

            def coordinate_jacobian(coordinates: np.ndarray) -> Any:
                nonlocal solver_jacobian_seconds
                started = perf_counter()
                physical = physical_from_coordinates(coordinates)
                physical_jacobian = active_analytic_jacobian(physical)
                if sparse.issparse(physical_jacobian):
                    frame_jacobian = physical_jacobian[
                        :, frame_base:num_variables
                    ].tocsr()
                else:
                    frame_jacobian = sparse.csr_matrix(
                        np.asarray(
                            physical_jacobian, dtype=np.float64
                        )[:, frame_base:num_variables]
                    )
                coordinate_jac = frame_jacobian @ frame_transform
                solver_jacobian_seconds += perf_counter() - started
                return coordinate_jac

            coordinate_kwargs = {
                **least_squares_kwargs,
                "bounds": (coordinate_lower, coordinate_upper),
                "max_nfev": int(max_nfev),
                "jac_sparsity": None,
            }
            coordinate_result = least_squares(
                coordinate_residual,
                coordinate_initial,
                jac=coordinate_jacobian,
                **coordinate_kwargs,
            )

            physical_x = physical_from_coordinates(coordinate_result.x)
            physical_fun = np.asarray(
                coordinate_result.fun, dtype=np.float64
            ).copy()
            physical_cost = _scipy_compatible_robust_cost(
                physical_fun,
                loss=cfg.robust_loss,
                f_scale=cfg.f_scale,
            )
            wrapper_started = perf_counter()
            raw_physical_jacobian = active_analytic_jacobian(physical_x)
            if sparse.issparse(raw_physical_jacobian):
                raw_frame_jacobian = raw_physical_jacobian[
                    :, frame_base:num_variables
                ].tocsr()
            else:
                raw_frame_jacobian = sparse.csr_matrix(
                    np.asarray(
                        raw_physical_jacobian, dtype=np.float64
                    )[:, frame_base:num_variables]
                )
            row_scale, gradient_weight = (
                _scipy_compatible_robust_linearization_weights(
                    physical_fun,
                    loss=cfg.robust_loss,
                    f_scale=cfg.f_scale,
                )
            )
            robust_frame_jacobian = raw_frame_jacobian.multiply(
                row_scale[:, None]
            ).tocsr()
            physical_jacobian = sparse.hstack(
                (
                    sparse.csr_matrix((row_count, frame_base)),
                    robust_frame_jacobian,
                ),
                format="csr",
            )
            physical_gradient = np.zeros(num_variables, dtype=np.float64)
            physical_gradient[frame_base:] = np.asarray(
                raw_frame_jacobian.T @ (gradient_weight * physical_fun),
                dtype=np.float64,
            ).reshape(-1)
            wrapper_jacobian_seconds = perf_counter() - wrapper_started
            wrapper_njev = int(getattr(coordinate_result, "njev", 0) or 0) + 1
            physical_result = SimpleNamespace(
                x=np.asarray(physical_x, dtype=np.float64).copy(),
                fun=np.asarray(physical_fun, dtype=np.float64),
                jac=physical_jacobian,
                grad=physical_gradient,
                cost=float(physical_cost),
                optimality=float(
                    np.linalg.norm(physical_gradient, ord=np.inf)
                ),
                active_mask=np.zeros(num_variables, dtype=np.int8),
                nfev=int(coordinate_result.nfev),
                njev=wrapper_njev,
                success=bool(coordinate_result.success),
                status=int(coordinate_result.status),
                message=str(coordinate_result.message),
            )
            scalar_transform = basis["scalar_transform"]
            scalar_prolongation = basis["scalar_prolongation"]
            scalar_detail = basis["scalar_detail"]
            diagnostics = {
                "schema": "hierarchical_joint_stage_b_frame_trust_v4",
                "enabled": True,
                "valid": True,
                "mode": "p64_sparse_exact_hierarchical_basis",
                "parameterization": "delta_frame=[P64,N_sparse]_c",
                "role": (
                    "unbiased_nonredundant_coordinate_transform_not_regularization"
                ),
                "interpretation": (
                    "stage_b_optimizes_all_frame_deltas_in_a_square_sparse_"
                    "hierarchical_basis_without_extra_residuals"
                ),
                "unbiased_physical_optimum": True,
                "added_residual_rows": 0,
                "physical_rows": int(row_count),
                "physical_variables": int(num_variables),
                "coordinate_variables": frame_variables,
                "augmented_variables": frame_variables,
                "scalar_shape": list(scalar_transform.shape),
                "frame_shape": list(frame_transform.shape),
                "num_control_frames": int(len(basis["control_frames"])),
                "num_detail_rows": int(len(basis["detail_rows"])),
                "control_frames": list(basis["control_frames"]),
                "detail_rows": list(basis["detail_rows"]),
                "scalar_prolongation_nnz": int(scalar_prolongation.nnz),
                "scalar_detail_nnz": int(scalar_detail.nnz),
                "scalar_transform_nnz": int(scalar_transform.nnz),
                "frame_transform_nnz": int(frame_transform.nnz),
                "pivot_abs_min": float(
                    np.min(np.abs(basis["pivot_values"]))
                ),
                "pivot_abs_max": float(
                    np.max(np.abs(basis["pivot_values"]))
                ),
                "orthogonality_inf": float(basis["orthogonality_inf"]),
                "orthogonality_relative_inf": float(
                    basis["orthogonality_relative_inf"]
                ),
                "detail_column_norms_unnormalized": (
                    basis["detail_column_norms_summary"]
                ),
                "detail_column_nnz": basis["detail_nnz_summary"],
                "dense_scalar_condition": basis["dense_scalar_condition"],
                "dense_scalar_condition_policy": (
                    "computed_for_scalar_rows_le_128"
                    if basis["dense_scalar_condition"] is not None
                    else "not_computed_for_large_sequence"
                ),
                "jacobian_mode": (
                    "sparse_analytic_physical_residual_times_sparse_basis"
                ),
                "solver_jacobian_seconds": float(solver_jacobian_seconds),
                "wrapper_extra_jacobian_seconds": float(
                    wrapper_jacobian_seconds
                ),
                "wrapper_extra_jacobian_evaluations": 1,
                "wrapper_extra_function_evaluations": 0,
                "num_function_evaluations": int(coordinate_result.nfev),
                "num_jacobian_evaluations": wrapper_njev,
                "physical_scipy_cost": float(physical_cost),
                "coordinate_result_cost": float(coordinate_result.cost),
                "coordinate_optimality": float(coordinate_result.optimality),
                "wall_seconds": float(
                    perf_counter() - sparse_basis_started
                ),
            }
            return physical_result, diagnostics

        def relax_coarse_only_reduced(
            start: np.ndarray,
            *,
            max_nfev: int,
        ) -> tuple[Any, dict[str, Any]]:
            """Optimize coarse variables while keeping fine frames frozen."""

            from types import SimpleNamespace

            from scipy import sparse

            if active_analytic_jacobian is None:
                raise RuntimeError(
                    "P64 sparse outer-cycle coarse stage requires an "
                    "analytic Jacobian"
                )
            coarse_started = perf_counter()
            coarse_initial = np.asarray(
                start[:frame_base], dtype=np.float64
            ).copy()
            coarse_lower = np.asarray(
                lower[:frame_base], dtype=np.float64
            ).copy()
            coarse_upper = np.asarray(
                upper[:frame_base], dtype=np.float64
            ).copy()

            def physical_from_coarse(coarse_values: np.ndarray) -> np.ndarray:
                physical = start.copy()
                physical[:frame_base] = coarse_values
                return physical

            def coarse_residual(coarse_values: np.ndarray) -> np.ndarray:
                return residual(physical_from_coarse(coarse_values))

            solver_jacobian_seconds = 0.0

            def coarse_jacobian(coarse_values: np.ndarray) -> Any:
                nonlocal solver_jacobian_seconds
                started = perf_counter()
                physical = physical_from_coarse(coarse_values)
                physical_jacobian = active_analytic_jacobian(physical)
                if sparse.issparse(physical_jacobian):
                    reduced_jacobian = physical_jacobian[
                        :, :frame_base
                    ].tocsr()
                else:
                    reduced_jacobian = sparse.csr_matrix(
                        np.asarray(
                            physical_jacobian, dtype=np.float64
                        )[:, :frame_base]
                    )
                solver_jacobian_seconds += perf_counter() - started
                return reduced_jacobian

            coarse_kwargs = {
                **least_squares_kwargs,
                "bounds": (coarse_lower, coarse_upper),
                "max_nfev": int(max_nfev),
                "jac_sparsity": None,
            }
            coarse_result = least_squares(
                coarse_residual,
                coarse_initial,
                jac=coarse_jacobian,
                **coarse_kwargs,
            )

            physical_x = physical_from_coarse(coarse_result.x)
            physical_fun = np.asarray(
                coarse_result.fun, dtype=np.float64
            ).copy()
            physical_cost = _scipy_compatible_robust_cost(
                physical_fun,
                loss=cfg.robust_loss,
                f_scale=cfg.f_scale,
            )
            wrapper_started = perf_counter()
            raw_physical_jacobian = active_analytic_jacobian(physical_x)
            if sparse.issparse(raw_physical_jacobian):
                raw_coarse_jacobian = raw_physical_jacobian[
                    :, :frame_base
                ].tocsr()
            else:
                raw_coarse_jacobian = sparse.csr_matrix(
                    np.asarray(
                        raw_physical_jacobian, dtype=np.float64
                    )[:, :frame_base]
                )
            row_scale, gradient_weight = (
                _scipy_compatible_robust_linearization_weights(
                    physical_fun,
                    loss=cfg.robust_loss,
                    f_scale=cfg.f_scale,
                )
            )
            robust_coarse_jacobian = raw_coarse_jacobian.multiply(
                row_scale[:, None]
            ).tocsr()
            physical_jacobian = sparse.hstack(
                (
                    robust_coarse_jacobian,
                    sparse.csr_matrix((row_count, num_variables - frame_base)),
                ),
                format="csr",
            )
            physical_gradient = np.zeros(num_variables, dtype=np.float64)
            physical_gradient[:frame_base] = np.asarray(
                raw_coarse_jacobian.T @ (gradient_weight * physical_fun),
                dtype=np.float64,
            ).reshape(-1)
            physical_active_mask = np.zeros(num_variables, dtype=np.int8)
            physical_active_mask[:frame_base] = np.asarray(
                coarse_result.active_mask, dtype=np.int8
            )
            wrapper_jacobian_seconds = perf_counter() - wrapper_started
            wrapper_njev = int(getattr(coarse_result, "njev", 0) or 0) + 1
            physical_result = SimpleNamespace(
                x=np.asarray(physical_x, dtype=np.float64).copy(),
                fun=np.asarray(physical_fun, dtype=np.float64),
                jac=physical_jacobian,
                grad=physical_gradient,
                cost=float(physical_cost),
                optimality=float(
                    np.linalg.norm(physical_gradient, ord=np.inf)
                ),
                active_mask=physical_active_mask,
                nfev=int(coarse_result.nfev),
                njev=wrapper_njev,
                success=bool(coarse_result.success),
                status=int(coarse_result.status),
                message=str(coarse_result.message),
            )
            coarse_step = np.asarray(
                physical_x[:frame_base] - start[:frame_base],
                dtype=np.float64,
            )
            scale_step = (
                coarse_step.reshape(-1, 7)[:, 6]
                if frame_base > 0
                else np.empty(0, dtype=np.float64)
            )
            lower_active = int(np.count_nonzero(coarse_result.active_mask < 0))
            upper_active = int(np.count_nonzero(coarse_result.active_mask > 0))
            diagnostics = {
                "schema": "hierarchical_joint_outer_cycle_coarse_only_v1",
                "enabled": True,
                "valid": True,
                "mode": "coarse_only_with_frames_frozen_after_sparse_basis",
                "parameterization": "delta_coarse=physical_coarse",
                "fixed_frame_variables": int(num_variables - frame_base),
                "coarse_variables": int(frame_base),
                "physical_variables": int(num_variables),
                "physical_rows": int(row_count),
                "jacobian_mode": "analytic_physical_residual_coarse_columns",
                "frame_leak_inf": 0.0,
                "coarse_step_linf": float(
                    np.max(np.abs(coarse_step), initial=0.0)
                ),
                "coarse_step_rms": float(
                    np.linalg.norm(coarse_step)
                    / np.sqrt(max(1, coarse_step.size))
                ),
                "scale_step_linf": float(
                    np.max(np.abs(scale_step), initial=0.0)
                ),
                "scale_step_rms": float(
                    np.linalg.norm(scale_step)
                    / np.sqrt(max(1, scale_step.size))
                ),
                "active_bounds": {
                    "lower": lower_active,
                    "upper": upper_active,
                    "total": lower_active + upper_active,
                },
                "solver_jacobian_seconds": float(solver_jacobian_seconds),
                "wrapper_extra_jacobian_seconds": float(
                    wrapper_jacobian_seconds
                ),
                "wrapper_extra_jacobian_evaluations": 1,
                "wrapper_extra_function_evaluations": 0,
                "num_function_evaluations": int(coarse_result.nfev),
                "num_jacobian_evaluations": wrapper_njev,
                "physical_scipy_cost": float(physical_cost),
                "coarse_result_cost": float(coarse_result.cost),
                "coarse_result_optimality": float(coarse_result.optimality),
                "wall_seconds": float(perf_counter() - coarse_started),
            }
            return physical_result, diagnostics

        def evaluate_outer_cycle_stage(
            *,
            stage: str,
            input_residual: np.ndarray,
            candidate_result: Any,
            cap: int,
            remaining_before: int,
        ) -> dict[str, Any]:
            output_residual = np.asarray(
                candidate_result.fun, dtype=np.float64
            )
            input_cost = _scipy_compatible_robust_cost(
                input_residual,
                loss=cfg.robust_loss,
                f_scale=cfg.f_scale,
            )
            output_cost = _scipy_compatible_robust_cost(
                output_residual,
                loss=cfg.robust_loss,
                f_scale=cfg.f_scale,
            )
            coarse_row_stop = 7 * len(edges)
            initial_coarse = initial_residual[:coarse_row_stop]
            output_coarse = output_residual[:coarse_row_stop]
            initial_coarse_cost = float(initial_coarse @ initial_coarse)
            output_coarse_cost = float(output_coarse @ output_coarse)
            coarse_ratio = (
                output_coarse_cost / initial_coarse_cost
                if initial_coarse_cost > 0.0
                else 1.0
                if output_coarse_cost == 0.0
                else np.inf
            )
            edge_ratios = _coarse_edge_statistics_ratios(
                _coarse_edge_residual_statistics(initial_coarse, edges),
                _coarse_edge_residual_statistics(output_coarse, edges),
            )
            family_checks = (
                (
                    cfg.joint_schur_step_max_adjacent_p95_ratio,
                    edge_ratios["adjacent"]["p95"],
                ),
                (
                    cfg.joint_schur_step_max_loop_p95_ratio,
                    edge_ratios["loop"]["p95"],
                ),
                (
                    cfg.joint_schur_step_max_loop_max_ratio,
                    edge_ratios["loop"]["max"],
                ),
            )
            edge_family_gate = all(
                threshold is None
                or observed is None
                or observed <= threshold
                for threshold, observed in family_checks
            )
            cost_gate = bool(
                np.isfinite(output_cost)
                and output_cost
                <= input_cost + max(1e-12, 1e-10 * abs(input_cost))
            )
            finite_gate = bool(
                np.all(np.isfinite(candidate_result.x))
                and np.all(np.isfinite(output_residual))
                and np.isfinite(output_cost)
            )
            solver_success = bool(candidate_result.success)
            topology_gate = bool(
                coarse_ratio <= cfg.joint_schur_step_max_graph_cost_ratio
                and edge_family_gate
            )
            accepted = bool(
                finite_gate and solver_success and cost_gate and topology_gate
            )
            return {
                "stage": stage,
                "input_original_physical_robust_cost": float(input_cost),
                "output_original_physical_robust_cost": float(output_cost),
                "cost_ratio": float(
                    output_cost / input_cost
                    if input_cost > 0.0
                    else 1.0
                    if output_cost == 0.0
                    else np.inf
                ),
                "cost_nonincrease_gate_passed": cost_gate,
                "finite_gate_passed": finite_gate,
                "solver_success_gate_passed": solver_success,
                "cumulative_topology_gate_passed": topology_gate,
                "aggregate_graph_gate_passed": bool(
                    coarse_ratio <= cfg.joint_schur_step_max_graph_cost_ratio
                ),
                "edge_family_gate_passed": edge_family_gate,
                "coarse_graph_cost_ratio": float(coarse_ratio),
                "coarse_edge_statistics_ratios": edge_ratios,
                "accepted": accepted,
                "solver": {
                    "success": solver_success,
                    "status": int(candidate_result.status),
                    "message": str(candidate_result.message),
                    "nfev": int(candidate_result.nfev),
                    "njev": (
                        None
                        if getattr(candidate_result, "njev", None) is None
                        else int(candidate_result.njev)
                    ),
                    "cap": int(cap),
                    "remaining_before": int(remaining_before),
                    "remaining_after": int(
                        remaining_before - int(candidate_result.nfev)
                    ),
                },
                "original_physical_cost_only": True,
            }

        def relax_sparse_basis_outer_cycle(
            start: np.ndarray,
            *,
            max_nfev: int,
        ) -> tuple[Any, dict[str, Any], list[Any]]:
            """Run one frozen A->B1->C->B2 block-coordinate outer cycle."""

            outer_started = perf_counter()
            remaining = int(max_nfev)
            discarded: list[Any] = []
            stage_a_nfev = (
                0
                if joint_subspace_result is None
                else int(joint_subspace_result.nfev)
            )
            stage_a_njev = (
                0
                if joint_subspace_result is None
                or getattr(joint_subspace_result, "njev", None) is None
                else int(joint_subspace_result.njev)
            )
            diagnostics: dict[str, Any] = {
                "schema": "hierarchical_joint_block_coordinate_outer_cycle_v1",
                "enabled": True,
                "valid": True,
                "mode": "p64_sparse_exact_basis_outer_cycle",
                "stage_order": ["B1", "C", "B2"],
                "original_physical_cost_only": True,
                "stage_budget_policy": (
                    "B1=remaining-2_C=remaining_after_B1-1_B2=remaining_after_C"
                ),
                "stages": {},
                "discarded_stages": [],
                "fallback": None,
                "final_selected_stage": None,
                "candidate_total_nfev": 0,
                "candidate_total_njev": 0,
                "stage_a_num_function_evaluations": stage_a_nfev,
                "stage_a_num_jacobian_evaluations": stage_a_njev,
                "diagnostic_residual_evaluations": 0,
                "max_nfev": int(cfg.max_nfev),
                "available_nfev_after_stage_a": remaining,
            }

            def finalize_outer_cycle(selected_stage: str) -> None:
                diagnostics["selected_stage"] = selected_stage
                diagnostics["final_selected_stage"] = selected_stage
                diagnostics["candidate_total_nfev_including_stage_a"] = (
                    stage_a_nfev + int(diagnostics["candidate_total_nfev"])
                )
                diagnostics["candidate_total_njev_including_stage_a"] = (
                    stage_a_njev + int(diagnostics["candidate_total_njev"])
                )
                diagnostics["candidate_total_budget_gate_passed"] = bool(
                    diagnostics["candidate_total_nfev_including_stage_a"]
                    <= int(cfg.max_nfev)
                )
                diagnostics["remaining_after_outer_cycle"] = int(remaining)
                for name, stage_record in diagnostics["stages"].items():
                    stage_record["selected"] = bool(name == selected_stage)
                    stage_record["discarded"] = bool(
                        name in set(diagnostics["discarded_stages"])
                    )
                diagnostics["wall_seconds"] = float(
                    perf_counter() - outer_started
                )

            def compact_stage_gate(stage_record: dict[str, Any]) -> dict[str, Any]:
                return {
                    "accepted": bool(stage_record["accepted"]),
                    "input_original_physical_robust_cost": float(
                        stage_record["input_original_physical_robust_cost"]
                    ),
                    "output_original_physical_robust_cost": float(
                        stage_record["output_original_physical_robust_cost"]
                    ),
                    "cost_ratio": float(stage_record["cost_ratio"]),
                    "cost_nonincrease_gate_passed": bool(
                        stage_record["cost_nonincrease_gate_passed"]
                    ),
                    "finite_gate_passed": bool(
                        stage_record["finite_gate_passed"]
                    ),
                    "solver_success_gate_passed": bool(
                        stage_record["solver_success_gate_passed"]
                    ),
                    "cumulative_topology_gate_passed": bool(
                        stage_record["cumulative_topology_gate_passed"]
                    ),
                    "aggregate_graph_gate_passed": bool(
                        stage_record["aggregate_graph_gate_passed"]
                    ),
                    "edge_family_gate_passed": bool(
                        stage_record["edge_family_gate_passed"]
                    ),
                    "coarse_graph_cost_ratio": float(
                        stage_record["coarse_graph_cost_ratio"]
                    ),
                    "coarse_edge_statistics_ratios": (
                        stage_record["coarse_edge_statistics_ratios"]
                    ),
                }

            start_residual = residual(start)
            diagnostics["diagnostic_residual_evaluations"] += 1
            if (
                cfg.joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking
            ):
                diagnostics["coarse_backtracking_extra_residual_evaluations"] = 0
            b1_cap = remaining - 2
            if b1_cap < 1:
                b1_cap = max(1, remaining)
                diagnostics["fallback"] = (
                    "insufficient_outer_cycle_budget_run_b1_only"
                )
            b1_remaining_before = remaining
            b1_result, b1_diagnostics = relax_frames_p64_sparse_basis(
                start, max_nfev=b1_cap
            )
            remaining -= int(b1_result.nfev)
            b1_stage = evaluate_outer_cycle_stage(
                stage="B1",
                input_residual=start_residual,
                candidate_result=b1_result,
                cap=b1_cap,
                remaining_before=b1_remaining_before,
            )
            b1_stage["sparse_basis"] = b1_diagnostics
            diagnostics["stages"]["B1"] = b1_stage
            diagnostics["candidate_total_nfev"] += int(b1_result.nfev)
            diagnostics["candidate_total_njev"] += int(b1_result.njev)
            if not b1_stage["accepted"]:
                diagnostics["discarded_stages"].append("B1")
                diagnostics["fallback"] = "public_joint_initialization"
                finalize_outer_cycle("public_fallback_after_B1")
                return b1_result, diagnostics, discarded
            if diagnostics["fallback"] is not None:
                finalize_outer_cycle("B1")
                return b1_result, diagnostics, discarded

            c_cap = remaining - 1
            if c_cap < 1:
                diagnostics["fallback"] = (
                    "insufficient_budget_for_C_"
                    "select_B1_not_last_accepted_after_incomplete_frame_"
                    "reequilibration"
                )
                finalize_outer_cycle("B1")
                return b1_result, diagnostics, discarded
            c_remaining_before = remaining
            c_result, c_diagnostics = relax_coarse_only_reduced(
                b1_result.x, max_nfev=c_cap
            )
            remaining -= int(c_result.nfev)
            c_stage = evaluate_outer_cycle_stage(
                stage="C",
                input_residual=b1_result.fun,
                candidate_result=c_result,
                cap=c_cap,
                remaining_before=c_remaining_before,
            )
            c_stage["coarse_only"] = c_diagnostics
            if (
                cfg.joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking
            ):
                from types import SimpleNamespace

                backtracking_info: dict[str, Any] = {
                    "enabled": True,
                    "alphas": [0.5, 0.25, 0.125, 0.0625, 0.03125],
                    "trigger": None,
                    "failure": None,
                    "selected_alpha": None,
                    "selected_trial_index": None,
                    "full_step_gate": compact_stage_gate(c_stage),
                    "trials": [],
                    "trial_result_linearization": (
                        "jac_grad_not_recomputed_for_trial_because_C_is_"
                        "never_selected_without_B2_frame_reequilibration"
                    ),
                }
                topology_only_failure = bool(
                    not c_stage["accepted"]
                    and c_stage["solver_success_gate_passed"]
                    and c_stage["finite_gate_passed"]
                    and c_stage["cost_nonincrease_gate_passed"]
                    and not c_stage["cumulative_topology_gate_passed"]
                )
                if c_stage["accepted"]:
                    backtracking_info["trigger"] = "full_C_step_already_accepted"
                    backtracking_info["failure"] = "not_triggered"
                elif topology_only_failure:
                    backtracking_info["trigger"] = (
                        "full_C_step_failed_only_cumulative_topology_gate"
                    )
                    c_full_result = c_result
                    c_full_stage = c_stage
                    c_delta = np.asarray(
                        c_full_result.x - b1_result.x, dtype=np.float64
                    )
                    for trial_index, alpha in enumerate(
                        backtracking_info["alphas"]
                    ):
                        trial_x = np.asarray(
                            b1_result.x + float(alpha) * c_delta,
                            dtype=np.float64,
                        )
                        trial_fun = residual(trial_x)
                        diagnostics["diagnostic_residual_evaluations"] += 1
                        diagnostics[
                            "coarse_backtracking_extra_residual_evaluations"
                        ] += 1
                        trial_cost = _scipy_compatible_robust_cost(
                            trial_fun,
                            loss=cfg.robust_loss,
                            f_scale=cfg.f_scale,
                        )
                        trial_result = SimpleNamespace(
                            x=trial_x,
                            fun=np.asarray(trial_fun, dtype=np.float64),
                            jac=None,
                            grad=None,
                            cost=float(trial_cost),
                            optimality=None,
                            active_mask=np.zeros(num_variables, dtype=np.int8),
                            nfev=int(c_full_result.nfev),
                            njev=getattr(c_full_result, "njev", None),
                            success=bool(c_full_result.success),
                            status=int(c_full_result.status),
                            message=(
                                "accepted coarse backtracking trial from "
                                f"full C solver: alpha={float(alpha):.5g}"
                            ),
                        )
                        trial_stage = evaluate_outer_cycle_stage(
                            stage="C",
                            input_residual=b1_result.fun,
                            candidate_result=trial_result,
                            cap=c_cap,
                            remaining_before=c_remaining_before,
                        )
                        trial_record = {
                            "trial_index": int(trial_index),
                            "alpha": float(alpha),
                            **compact_stage_gate(trial_stage),
                        }
                        backtracking_info["trials"].append(trial_record)
                        if trial_stage["accepted"]:
                            backtracking_info["selected_alpha"] = float(alpha)
                            backtracking_info["selected_trial_index"] = int(
                                trial_index
                            )
                            c_result = trial_result
                            c_stage = trial_stage
                            c_stage["coarse_only"] = c_diagnostics
                            break
                    if backtracking_info["selected_alpha"] is None:
                        backtracking_info["failure"] = (
                            "no_alpha_passed_existing_full_objective_and_"
                            "cumulative_topology_gates"
                        )
                        c_result = c_full_result
                        c_stage = c_full_stage
                    else:
                        backtracking_info["failure"] = None
                else:
                    backtracking_info["trigger"] = (
                        "full_C_step_failed_non_topology_gate"
                    )
                    backtracking_info["failure"] = "not_triggered"
                c_stage["coarse_backtracking"] = backtracking_info
            diagnostics["stages"]["C"] = c_stage
            if (
                cfg.joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking
            ):
                c_backtracking = c_stage.get("coarse_backtracking", {})
                diagnostics["coarse_backtracking_selected"] = bool(
                    c_backtracking.get("selected_alpha") is not None
                )
                diagnostics["coarse_backtracking_selected_alpha"] = (
                    c_backtracking.get("selected_alpha")
                )
            diagnostics["candidate_total_nfev"] += int(c_result.nfev)
            diagnostics["candidate_total_njev"] += int(c_result.njev)
            if not c_stage["accepted"]:
                discarded.append(c_result)
                diagnostics["discarded_stages"].append("C")
                diagnostics["fallback"] = (
                    "C_rejected_select_B1_not_last_accepted_after_"
                    "incomplete_frame_reequilibration"
                )
                finalize_outer_cycle("B1")
                return b1_result, diagnostics, discarded

            b2_cap = remaining
            if b2_cap < 1:
                discarded.append(c_result)
                diagnostics["discarded_stages"].append("C")
                diagnostics["fallback"] = (
                    "insufficient_budget_for_B2_select_B1_not_last_"
                    "accepted_after_incomplete_frame_reequilibration"
                )
                finalize_outer_cycle("B1")
                return b1_result, diagnostics, discarded
            b2_remaining_before = remaining
            b2_result, b2_diagnostics = relax_frames_p64_sparse_basis(
                c_result.x, max_nfev=b2_cap
            )
            remaining -= int(b2_result.nfev)
            b2_stage = evaluate_outer_cycle_stage(
                stage="B2",
                input_residual=c_result.fun,
                candidate_result=b2_result,
                cap=b2_cap,
                remaining_before=b2_remaining_before,
            )
            b2_stage["sparse_basis"] = b2_diagnostics
            diagnostics["stages"]["B2"] = b2_stage
            diagnostics["candidate_total_nfev"] += int(b2_result.nfev)
            diagnostics["candidate_total_njev"] += int(b2_result.njev)
            if not b2_stage["accepted"]:
                discarded.extend((c_result, b2_result))
                diagnostics["discarded_stages"].extend(("C", "B2"))
                diagnostics["fallback"] = (
                    "B2_rejected_select_B1_not_last_accepted_after_"
                    "incomplete_frame_reequilibration"
                )
                finalize_outer_cycle("B1")
                return b1_result, diagnostics, discarded

            discarded.extend((b1_result, c_result))
            diagnostics["discarded_stages"].extend(("B1", "C"))
            finalize_outer_cycle("B2")
            return b2_result, diagnostics, discarded

        def solve_full_sparse_exact_basis() -> tuple[Any, dict[str, Any]]:
            """Solve the full joint problem in sparse exact frame coordinates."""

            from types import SimpleNamespace

            from scipy import sparse

            full_basis_started = perf_counter()
            if active_analytic_jacobian is None:
                fallback_result = solve_once(initial_vector, least_squares_kwargs)
                return fallback_result, {
                    "schema": "hierarchical_joint_full_sparse_exact_basis_v1",
                    "enabled": True,
                    "valid": False,
                    "reason": "analytic_jacobian_unavailable",
                    "mode": "joint_full_sparse_exact_basis_public_fallback",
                    "requested_mode": "joint_full_sparse_exact_basis",
                    "fallback": "public_joint_initialization",
                    "num_function_evaluations": int(fallback_result.nfev),
                    "num_jacobian_evaluations": (
                        None
                        if getattr(fallback_result, "njev", None) is None
                        else int(fallback_result.njev)
                    ),
                    "wall_seconds": float(perf_counter() - full_basis_started),
                }

            basis = _joint_frame_sparse_exact_basis(
                num_frames,
                starts_ends,
                stride=cfg.joint_subspace_control_stride,
            )
            frame_transform = basis["frame_transform"]
            frame_variables = int(num_variables - frame_base)
            if frame_transform.shape != (frame_variables, frame_variables):
                raise RuntimeError("full sparse exact basis frame shape mismatch")

            coordinate_initial = np.zeros(num_variables, dtype=np.float64)
            coordinate_initial[:frame_base] = initial_vector[:frame_base]
            coordinate_lower = np.full(num_variables, -np.inf, dtype=np.float64)
            coordinate_upper = np.full(num_variables, np.inf, dtype=np.float64)
            coordinate_lower[:frame_base] = lower[:frame_base]
            coordinate_upper[:frame_base] = upper[:frame_base]

            def physical_from_coordinates(coordinates: np.ndarray) -> np.ndarray:
                physical = initial_vector.copy()
                physical[:frame_base] = coordinates[:frame_base]
                physical[frame_base:] = (
                    initial_vector[frame_base:]
                    + frame_transform @ coordinates[frame_base:]
                )
                return physical

            def coordinate_residual(coordinates: np.ndarray) -> np.ndarray:
                return residual(physical_from_coordinates(coordinates))

            solver_jacobian_seconds = 0.0

            def coordinate_jacobian(coordinates: np.ndarray) -> Any:
                nonlocal solver_jacobian_seconds
                started = perf_counter()
                physical = physical_from_coordinates(coordinates)
                physical_jacobian = active_analytic_jacobian(physical)
                if sparse.issparse(physical_jacobian):
                    coarse_jacobian = physical_jacobian[:, :frame_base].tocsr()
                    frame_jacobian = physical_jacobian[
                        :, frame_base:num_variables
                    ].tocsr()
                else:
                    physical_array = np.asarray(
                        physical_jacobian, dtype=np.float64
                    )
                    coarse_jacobian = sparse.csr_matrix(
                        physical_array[:, :frame_base]
                    )
                    frame_jacobian = sparse.csr_matrix(
                        physical_array[:, frame_base:num_variables]
                    )
                coordinate_jac = sparse.hstack(
                    (coarse_jacobian, frame_jacobian @ frame_transform),
                    format="csr",
                )
                solver_jacobian_seconds += perf_counter() - started
                return coordinate_jac

            coordinate_kwargs = {
                **least_squares_kwargs,
                "bounds": (coordinate_lower, coordinate_upper),
                "jac_sparsity": None,
            }
            coordinate_result = least_squares(
                coordinate_residual,
                coordinate_initial,
                jac=coordinate_jacobian,
                **coordinate_kwargs,
            )
            physical_x = physical_from_coordinates(coordinate_result.x)
            physical_fun = np.asarray(
                coordinate_result.fun, dtype=np.float64
            ).copy()
            physical_cost = _scipy_compatible_robust_cost(
                physical_fun,
                loss=cfg.robust_loss,
                f_scale=cfg.f_scale,
            )

            wrapper_started = perf_counter()
            raw_physical_jacobian = active_analytic_jacobian(physical_x)
            if sparse.issparse(raw_physical_jacobian):
                raw_physical_jacobian = raw_physical_jacobian.tocsr()
            else:
                raw_physical_jacobian = sparse.csr_matrix(
                    np.asarray(raw_physical_jacobian, dtype=np.float64)
                )
            row_scale, gradient_weight = (
                _scipy_compatible_robust_linearization_weights(
                    physical_fun,
                    loss=cfg.robust_loss,
                    f_scale=cfg.f_scale,
                )
            )
            physical_jacobian = raw_physical_jacobian.multiply(
                row_scale[:, None]
            ).tocsr()
            physical_gradient = np.asarray(
                raw_physical_jacobian.T @ (gradient_weight * physical_fun),
                dtype=np.float64,
            ).reshape(-1)
            wrapper_jacobian_seconds = perf_counter() - wrapper_started
            wrapper_njev = int(getattr(coordinate_result, "njev", 0) or 0) + 1
            physical_active_mask = np.zeros(num_variables, dtype=np.int8)
            physical_active_mask[:frame_base] = np.asarray(
                coordinate_result.active_mask[:frame_base], dtype=np.int8
            )
            physical_result = SimpleNamespace(
                x=np.asarray(physical_x, dtype=np.float64).copy(),
                fun=physical_fun,
                jac=physical_jacobian,
                grad=physical_gradient,
                cost=float(physical_cost),
                optimality=float(
                    np.linalg.norm(physical_gradient, ord=np.inf)
                ),
                active_mask=physical_active_mask,
                nfev=int(coordinate_result.nfev),
                njev=wrapper_njev,
                success=bool(coordinate_result.success),
                status=int(coordinate_result.status),
                message=str(coordinate_result.message),
            )
            scalar_transform = basis["scalar_transform"]
            scalar_prolongation = basis["scalar_prolongation"]
            scalar_detail = basis["scalar_detail"]
            initial_physical_cost = _scipy_compatible_robust_cost(
                initial_residual,
                loss=cfg.robust_loss,
                f_scale=cfg.f_scale,
            )
            diagnostics = {
                "schema": "hierarchical_joint_full_sparse_exact_basis_v1",
                "enabled": True,
                "valid": True,
                "mode": "joint_full_sparse_exact_basis",
                "parameterization": "y=[coarse_physical,frame_sparse_exact_basis]",
                "role": "direct_full_joint_unbiased_coordinate_transform",
                "unbiased_physical_optimum": True,
                "added_residual_rows": 0,
                "square_invertible_frame_transform": True,
                "physical_rows": int(row_count),
                "physical_variables": int(num_variables),
                "coordinate_variables": int(num_variables),
                "coarse_variables": int(frame_base),
                "frame_coordinate_variables": int(frame_variables),
                "scalar_shape": list(scalar_transform.shape),
                "frame_shape": list(frame_transform.shape),
                "num_control_frames": int(len(basis["control_frames"])),
                "num_detail_rows": int(len(basis["detail_rows"])),
                "scalar_prolongation_nnz": int(scalar_prolongation.nnz),
                "scalar_detail_nnz": int(scalar_detail.nnz),
                "scalar_transform_nnz": int(scalar_transform.nnz),
                "frame_transform_nnz": int(frame_transform.nnz),
                "pivot_abs_min": float(np.min(np.abs(basis["pivot_values"]))),
                "pivot_abs_max": float(np.max(np.abs(basis["pivot_values"]))),
                "orthogonality_inf": float(basis["orthogonality_inf"]),
                "orthogonality_relative_inf": float(
                    basis["orthogonality_relative_inf"]
                ),
                "dense_scalar_condition": basis["dense_scalar_condition"],
                "dense_scalar_condition_policy": (
                    "computed_for_scalar_rows_le_128"
                    if basis["dense_scalar_condition"] is not None
                    else "not_computed_for_large_sequence"
                ),
                "jacobian_mode": "analytic_physical_jacobian_times_sparse_basis",
                "initial_physical_scipy_cost": float(initial_physical_cost),
                "final_physical_scipy_cost": float(physical_cost),
                "coordinate_result_cost": float(coordinate_result.cost),
                "physical_gradient_inf": float(
                    np.linalg.norm(physical_gradient, ord=np.inf)
                ),
                "physical_coarse_gradient_inf": float(
                    np.linalg.norm(physical_gradient[:frame_base], ord=np.inf)
                    if frame_base
                    else 0.0
                ),
                "physical_frame_gradient_inf": float(
                    np.linalg.norm(
                        physical_gradient[frame_base:num_variables],
                        ord=np.inf,
                    )
                    if frame_variables
                    else 0.0
                ),
                "coordinate_optimality": float(coordinate_result.optimality),
                "physical_optimality": float(physical_result.optimality),
                "solver_jacobian_seconds": float(solver_jacobian_seconds),
                "wrapper_extra_function_evaluations": 0,
                "wrapper_extra_jacobian_evaluations": 1,
                "wrapper_extra_jacobian_seconds": float(
                    wrapper_jacobian_seconds
                ),
                "num_function_evaluations": int(coordinate_result.nfev),
                "num_jacobian_evaluations": wrapper_njev,
                "wall_seconds": float(perf_counter() - full_basis_started),
            }
            return physical_result, diagnostics

        def linearize_full(start: np.ndarray) -> Any:
            if active_analytic_jacobian is None:
                linearization = solve_once(
                    start,
                    {
                        **least_squares_kwargs,
                        "max_nfev": 1,
                    },
                )
            else:
                from types import SimpleNamespace

                from scipy import sparse

                values = residual(start)
                raw_jacobian = active_analytic_jacobian(start)
                row_scale, gradient_weight = (
                    _scipy_compatible_robust_linearization_weights(
                        values,
                        loss=cfg.robust_loss,
                        f_scale=cfg.f_scale,
                    )
                )
                if sparse.issparse(raw_jacobian):
                    robust_jacobian = raw_jacobian.multiply(
                        row_scale[:, None]
                    ).tocsr()
                else:
                    robust_jacobian = (
                        row_scale[:, None]
                        * np.asarray(raw_jacobian, dtype=np.float64)
                    )
                gradient = np.asarray(
                    raw_jacobian.T @ (gradient_weight * values),
                    dtype=np.float64,
                ).reshape(-1)
                linearization = SimpleNamespace(
                    x=start.copy(),
                    fun=values,
                    jac=robust_jacobian,
                    raw_fun=values,
                    raw_jac=raw_jacobian,
                    grad=gradient,
                    cost=_scipy_compatible_robust_cost(
                        values,
                        loss=cfg.robust_loss,
                        f_scale=cfg.f_scale,
                    ),
                    optimality=float(np.linalg.norm(gradient, ord=np.inf)),
                    success=False,
                    status=0,
                    message="direct analytic robust linearization",
                    linearization_source=(
                        "direct_analytic_jacobian_robust_linearization"
                    ),
                    nfev=1,
                    njev=1,
                )
            schur_linearization_results.append(linearization)
            return linearization

        def record_schur_final_accepted_step(
            *,
            candidate: np.ndarray,
            candidate_residual: np.ndarray,
            full_cost_after: float,
            step_start: np.ndarray,
            full_cost_before: float,
            actual_full_reduction: float,
            reduction_ratio: float | None,
            alpha: float,
        ) -> None:
            nonlocal schur_final_accepted_x
            nonlocal schur_final_accepted_fun
            nonlocal schur_final_accepted_cost
            nonlocal schur_final_last_coarse_block_rms
            nonlocal schur_final_last_frame_block_rms
            nonlocal schur_final_last_step_measure
            nonlocal schur_final_last_step_norm
            nonlocal schur_final_last_actual_reduction
            nonlocal schur_final_last_reduction_ratio
            nonlocal schur_final_last_cost_before
            nonlocal schur_final_last_alpha
            schur_final_accepted_x = np.asarray(
                candidate, dtype=np.float64
            ).copy()
            schur_final_accepted_fun = np.asarray(
                candidate_residual, dtype=np.float64
            ).copy()
            schur_final_accepted_cost = float(full_cost_after)
            accepted_step = (
                np.asarray(candidate, dtype=np.float64)
                - np.asarray(step_start, dtype=np.float64)
            )
            coarse_block_count = frame_base // 7
            frame_block_count = (num_variables - frame_base) // 6
            schur_final_last_coarse_block_rms = (
                float(
                    np.linalg.norm(accepted_step[:frame_base])
                    / np.sqrt(coarse_block_count)
                )
                if coarse_block_count > 0
                else 0.0
            )
            schur_final_last_frame_block_rms = (
                float(
                    np.linalg.norm(accepted_step[frame_base:num_variables])
                    / np.sqrt(frame_block_count)
                )
                if frame_block_count > 0
                else 0.0
            )
            schur_final_last_step_measure = max(
                schur_final_last_coarse_block_rms,
                schur_final_last_frame_block_rms,
            )
            schur_final_last_step_norm = float(np.linalg.norm(accepted_step))
            schur_final_last_actual_reduction = float(actual_full_reduction)
            schur_final_last_reduction_ratio = (
                None if reduction_ratio is None else float(reduction_ratio)
            )
            schur_final_last_cost_before = float(full_cost_before)
            schur_final_last_alpha = float(alpha)

        def build_schur_final_solver_result(
            final_x: np.ndarray,
            final_fun: np.ndarray,
            final_cost: float,
            *,
            accepted: bool,
            stop_reason: str,
        ) -> tuple[Any, dict[str, Any]]:
            from types import SimpleNamespace

            from scipy import sparse

            def block_measure(vector: np.ndarray) -> tuple[float, float, float]:
                coarse_block_count = frame_base // 7
                frame_block_count = (num_variables - frame_base) // 6
                coarse_rms = (
                    float(
                        np.linalg.norm(vector[:frame_base])
                        / np.sqrt(coarse_block_count)
                    )
                    if coarse_block_count > 0
                    else 0.0
                )
                frame_rms = (
                    float(
                        np.linalg.norm(vector[frame_base:num_variables])
                        / np.sqrt(frame_block_count)
                    )
                    if frame_block_count > 0
                    else 0.0
                )
                return coarse_rms, frame_rms, max(coarse_rms, frame_rms)

            wrapper_started = perf_counter()
            final_vector = np.asarray(final_x, dtype=np.float64).copy()
            final_values = np.asarray(final_fun, dtype=np.float64).reshape(-1).copy()
            if active_analytic_jacobian is None:
                raise RuntimeError(
                    "joint_schur_step_final_solver requires an analytic Jacobian"
                )
            raw_jacobian = active_analytic_jacobian(final_vector)
            if sparse.issparse(raw_jacobian):
                raw_jacobian = raw_jacobian.tocsr()
            else:
                raw_jacobian = sparse.csr_matrix(
                    np.asarray(raw_jacobian, dtype=np.float64)
                )
            row_scale, gradient_weight = (
                _scipy_compatible_robust_linearization_weights(
                    final_values,
                    loss=cfg.robust_loss,
                    f_scale=cfg.f_scale,
                )
            )
            robust_jacobian = raw_jacobian.multiply(row_scale[:, None]).tocsr()
            gradient = np.asarray(
                raw_jacobian.T @ (gradient_weight * final_values),
                dtype=np.float64,
            ).reshape(-1)
            active_mask, projected_gradient = _project_gradient_for_bounds(
                final_vector,
                gradient,
                lower,
                upper,
            )
            projected_gradient_inf = float(
                np.linalg.norm(projected_gradient, ord=np.inf)
            )
            raw_gradient_inf = float(np.linalg.norm(gradient, ord=np.inf))
            first_linearization = (
                schur_linearization_results[0]
                if schur_linearization_results
                else None
            )
            initial_projected_gradient_inf = None
            if first_linearization is not None and hasattr(
                first_linearization, "grad"
            ):
                _, initial_projected_gradient = _project_gradient_for_bounds(
                    np.asarray(first_linearization.x, dtype=np.float64),
                    np.asarray(first_linearization.grad, dtype=np.float64),
                    lower,
                    upper,
                )
                initial_projected_gradient_inf = float(
                    np.linalg.norm(initial_projected_gradient, ord=np.inf)
                )
            relative_gradient_decrease = (
                float(projected_gradient_inf / initial_projected_gradient_inf)
                if initial_projected_gradient_inf is not None
                and initial_projected_gradient_inf > 0.0
                else None
            )
            adequate_reduction = bool(
                accepted
                and schur_final_last_reduction_ratio is not None
                and schur_final_last_reduction_ratio >= 0.25
                and schur_final_last_actual_reduction is not None
                and schur_final_last_actual_reduction >= -max(
                        1e-12, 1e-10 * abs(float(final_cost))
                )
            )
            gtol_passed = bool(projected_gradient_inf <= cfg.solver_gtol)
            already_converged_without_accepted_step = bool(
                gtol_passed and not accepted
            )
            ftol_passed = bool(
                adequate_reduction
                and schur_final_last_actual_reduction is not None
                and schur_final_last_cost_before is not None
                and abs(schur_final_last_actual_reduction)
                <= cfg.solver_ftol
                * max(1.0, abs(float(schur_final_last_cost_before)))
            )
            (
                final_x_coarse_block_rms,
                final_x_frame_block_rms,
                final_x_measure,
            ) = block_measure(final_vector)
            xtol_passed = bool(
                adequate_reduction
                and schur_final_last_step_measure is not None
                and schur_final_last_step_measure
                <= cfg.solver_xtol
                * (cfg.solver_xtol + final_x_measure)
            )
            if gtol_passed:
                success = True
                status = 1
                if already_converged_without_accepted_step:
                    message = (
                        "schur_final_solver_projected_gradient_already_converged"
                    )
                    termination_reason = "already_converged_gtol"
                else:
                    message = "schur_final_solver_projected_gradient_converged"
                    termination_reason = "gtol"
            elif ftol_passed and xtol_passed:
                success = True
                status = 4
                message = "schur_final_solver_ftol_and_xtol_converged"
                termination_reason = "ftol_and_xtol"
            elif ftol_passed:
                success = True
                status = 2
                message = "schur_final_solver_ftol_converged"
                termination_reason = "ftol"
            elif xtol_passed:
                success = True
                status = 3
                message = "schur_final_solver_xtol_converged"
                termination_reason = "xtol"
            elif accepted:
                success = False
                status = 0
                message = (
                    "schur_final_solver_accepted_guarded_step_but_"
                    "projected_optimality_not_converged"
                )
                termination_reason = (
                    "accepted_step_without_ftol_xtol_or_gtol_convergence"
                )
            else:
                success = False
                status = 0
                message = "schur_final_solver_no_accepted_step"
                termination_reason = (
                    "maximum_outer_iterations_reached_without_acceptance"
                    if stop_reason == "requested_outer_iterations_completed"
                    else stop_reason
                )
            linearization_nfev = sum(
                int(getattr(stage_result, "nfev", 0) or 0)
                for stage_result in schur_linearization_results
            )
            linearization_njev_values = [
                getattr(stage_result, "njev", None)
                for stage_result in schur_linearization_results
            ]
            linearization_njev = (
                None
                if any(value is None for value in linearization_njev_values)
                else sum(int(value or 0) for value in linearization_njev_values)
            )
            wrapper_njev = 1
            result_njev = (
                None
                if linearization_njev is None
                else int(linearization_njev) + wrapper_njev
            )
            result_nfev = int(linearization_nfev) + int(
                schur_trial_residual_evaluations
            )
            cg_total_iterations = sum(
                int(
                    outer.get("schur_diagnostics", {}).get(
                        "cg_iterations", 0
                    )
                    or 0
                )
                for outer in schur_outer_diagnostics
            )
            cg_all_converged = all(
                outer.get("schur_diagnostics", {}).get("cg_converged")
                is True
                for outer in schur_outer_diagnostics
                if outer.get("schur_diagnostics") is not None
            )
            wrapper_seconds = perf_counter() - wrapper_started
            final_result = SimpleNamespace(
                x=final_vector,
                fun=final_values,
                jac=robust_jacobian,
                grad=gradient,
                cost=float(final_cost),
                optimality=projected_gradient_inf,
                active_mask=active_mask,
                nfev=result_nfev,
                njev=result_njev,
                success=success,
                status=status,
                message=message,
            )
            diagnostics = {
                "schema": "hierarchical_joint_schur_final_solver_v1",
                "enabled": True,
                "mode": (
                    "joint_schur_final_solver"
                    if success
                    else "joint_schur_final_solver_not_converged"
                ),
                "final_polish_mode": "none",
                "public_final_solve_called": False,
                "public_fallback": False,
                "accepted_final_candidate": bool(accepted),
                "termination_reason": termination_reason,
                "outer_stop_reason": stop_reason,
                "success": success,
                "termination_status": status,
                "message": message,
                "initial_projected_gradient_inf": (
                    initial_projected_gradient_inf
                ),
                "final_projected_gradient_inf": projected_gradient_inf,
                "final_raw_gradient_inf": raw_gradient_inf,
                "relative_projected_gradient": relative_gradient_decrease,
                "gtol_passed": gtol_passed,
                "already_converged_without_accepted_step": (
                    already_converged_without_accepted_step
                ),
                "ftol_passed": ftol_passed,
                "xtol_passed": xtol_passed,
                "adequate_reduction_gate_passed": adequate_reduction,
                "last_accepted_alpha": schur_final_last_alpha,
                "last_accepted_step_norm": schur_final_last_step_norm,
                "last_accepted_coarse_step_block_rms": (
                    schur_final_last_coarse_block_rms
                ),
                "last_accepted_frame_step_block_rms": (
                    schur_final_last_frame_block_rms
                ),
                "last_accepted_step_measure": schur_final_last_step_measure,
                "final_x_coarse_block_rms": final_x_coarse_block_rms,
                "final_x_frame_block_rms": final_x_frame_block_rms,
                "final_x_measure": final_x_measure,
                "xtol_step_measure": schur_final_last_step_measure,
                "xtol_x_measure": final_x_measure,
                "last_accepted_actual_reduction": (
                    schur_final_last_actual_reduction
                ),
                "last_accepted_reduction_ratio": (
                    schur_final_last_reduction_ratio
                ),
                "linearization_function_evaluations": int(linearization_nfev),
                "linearization_jacobian_evaluations": linearization_njev,
                "trial_residual_evaluations": int(
                    schur_trial_residual_evaluations
                ),
                "wrapper_extra_function_evaluations": 0,
                "wrapper_extra_jacobian_evaluations": wrapper_njev,
                "num_function_evaluations": int(result_nfev),
                "num_jacobian_evaluations": result_njev,
                "cg_total_iterations": int(cg_total_iterations),
                "cg_all_converged": bool(cg_all_converged),
                "active_bound_count": int(np.count_nonzero(active_mask)),
                "projected_kkt_policy": (
                    "lower_bound_positive_gradient_and_upper_bound_negative_"
                    "gradient_components_zeroed"
                ),
                "wrapper_extra_jacobian_seconds": float(wrapper_seconds),
            }
            return final_result, diagnostics

        joint_start = initial_vector
        final_solve_kwargs = least_squares_kwargs
        if cfg.joint_full_sparse_exact_basis:
            result, joint_full_sparse_exact_basis_diagnostics = (
                solve_full_sparse_exact_basis()
            )
        if (
            result is None
            and
            cfg.joint_subspace_max_nfev > 0
            and not cfg.freeze_coarse_alignments
            and frame_base > 0
            and num_variables > frame_base
        ):
            initial_subspace_coarse = initial_residual[: 7 * len(edges)]
            initial_subspace_coarse_cost = float(
                initial_subspace_coarse @ initial_subspace_coarse
            )
            if (
                cfg.joint_subspace_min_coarse_squared_residual > 0.0
                and initial_subspace_coarse_cost
                <= cfg.joint_subspace_min_coarse_squared_residual
            ):
                joint_subspace_diagnostics = {
                    "schema": "hierarchical_joint_subspace_early_skip_v1",
                    "accepted": False,
                    "decision": "early_skip_near_zero_coarse_denominator",
                    "initial_coarse_graph_squared_residual": (
                        initial_subspace_coarse_cost
                    ),
                    "minimum_coarse_graph_squared_residual": float(
                        cfg.joint_subspace_min_coarse_squared_residual
                    ),
                    "num_function_evaluations": 0,
                    "num_jacobian_evaluations": 0,
                    "wall_seconds": 0.0,
                    "fallback": "public_joint_initialization",
                }
            else:
                subspace_started = perf_counter()
                (
                    joint_subspace_result,
                    joint_subspace_expanded_start,
                    joint_subspace_diagnostics,
                ) = solve_joint_subspace(
                    initial_vector,
                    max_nfev=cfg.joint_subspace_max_nfev,
                )
                joint_subspace_diagnostics["wall_seconds"] = float(
                    perf_counter() - subspace_started
                )
                if joint_subspace_diagnostics["accepted"]:
                    joint_start = joint_subspace_expanded_start
                    final_solve_kwargs = {
                        **least_squares_kwargs,
                        "max_nfev": max(
                            1,
                            int(cfg.max_nfev)
                            - int(joint_subspace_result.nfev),
                        ),
                    }
        if result is None and cfg.joint_schur_step_public_pilot_max_nfev > 0:
            public_pilot_result = solve_once(
                initial_vector,
                {
                    **least_squares_kwargs,
                    "max_nfev": int(
                        cfg.joint_schur_step_public_pilot_max_nfev
                    ),
                },
            )
            if public_pilot_result.success:
                result = public_pilot_result
            else:
                joint_start = public_pilot_result.x.copy()
                final_solve_kwargs = {
                    **least_squares_kwargs,
                    "max_nfev": max(
                        1,
                        int(cfg.max_nfev) - int(public_pilot_result.nfev),
                    ),
                }
        if (
            result is None
            and (
                cfg.joint_frame_relaxation_max_nfev > 0
                or cfg.joint_schur_diagnostics
                or cfg.joint_schur_step
            )
            and not cfg.freeze_coarse_alignments
            and frame_base > 0
            and num_variables > frame_base
        ):
            if cfg.joint_frame_relaxation_max_nfev > 0:
                frame_relaxation_result = relax_frames(
                    initial_vector,
                    max_nfev=cfg.joint_frame_relaxation_max_nfev,
                )
                joint_start = frame_relaxation_result.x
            if cfg.joint_schur_diagnostics or cfg.joint_schur_step:
                schur_step_trials = []
                schur_outer_diagnostics = []
                accepted_outer_iterations = 0
                probe_analytic_unavailable = bool(
                    cfg.joint_schur_full_space_block_pcg_probe
                    and active_analytic_jacobian is None
                )
                current_frame_result = (
                    None
                    if probe_analytic_unavailable
                    else frame_relaxation_result
                    if frame_relaxation_result is not None
                    else public_pilot_result
                    if public_pilot_result is not None
                    else linearize_full(joint_start)
                )
                current_alpha = float(cfg.joint_schur_step_initial_alpha)
                coarse_grid_cache = None
                adaptive_outer = bool(
                    cfg.joint_schur_step
                    and cfg.joint_schur_step_max_outer_iterations > 1
                )
                effective_loop_p95_ratio = (
                    cfg.joint_schur_step_max_loop_p95_ratio
                    if cfg.joint_schur_step_max_loop_p95_ratio is not None
                    else 1.01
                    if cfg.joint_schur_step
                    else None
                )
                effective_loop_max_ratio = (
                    cfg.joint_schur_step_max_loop_max_ratio
                    if cfg.joint_schur_step_max_loop_max_ratio is not None
                    else 1.05
                    if cfg.joint_schur_step
                    else None
                )
                requested_outer_iterations = (
                    cfg.joint_schur_step_max_outer_iterations
                    if cfg.joint_schur_step
                    else 0
                    if probe_analytic_unavailable
                    else 1
                )
                stop_reason = "requested_outer_iterations_completed"
                initial_coarse_cost = None
                initial_coarse_statistics = None
                initial_full_cost = (
                    _scipy_compatible_robust_cost(
                        initial_residual,
                        loss=cfg.robust_loss,
                        f_scale=cfg.f_scale,
                    )
                    if current_frame_result is None
                    else float(current_frame_result.cost)
                )
                accepted_warm_cost = initial_full_cost
                initial_low_frequency_efficiency = None
                initial_jacobi_low_frequency_efficiency = None
                initial_diagonal_low_frequency_gate = None
                low_frequency_efficiency_aggregate_size = None
                if probe_analytic_unavailable:
                    schur_diagnostics_raw = {
                        "schema": "hierarchical_joint_schur_diagnostics_v1",
                        "linearization": "analytic_jacobian_unavailable",
                        "solution_mutated": False,
                        "status": "skipped",
                        "valid": False,
                        "reason": "analytic_jacobian_unavailable",
                        "full_space_block_pcg_probe": {
                            "schema": (
                                "hierarchical_joint_schur_full_space_block_pcg_probe_v1"
                            ),
                            "valid": False,
                            "reason": "analytic_jacobian_unavailable",
                            "solution_mutated": False,
                        },
                    }

                def schur_solve_usable(diagnostics: dict[str, Any]) -> bool:
                    relative_residual = diagnostics.get("cg_relative_residual")
                    return bool(
                        diagnostics.get("cg_converged") is True
                        and relative_residual is not None
                        and np.isfinite(relative_residual)
                        and relative_residual
                        <= max(2.0 * cfg.joint_schur_cg_rtol, 1e-10)
                    )

                def selected_low_frequency_efficiency(
                    diagnostics: dict[str, Any],
                    *,
                    metric: str,
                ) -> tuple[int | None, float | None]:
                    mode_energy = diagnostics.get(
                        "frame_direction_sequential_mode_energy", {}
                    )
                    num_blocks = int(mode_energy.get("num_blocks", 0))
                    target_size = min(64, num_blocks)
                    efficiency_levels = diagnostics.get(
                        "frame_low_frequency_model_efficiency", {}
                    ).get("levels", [])
                    for level in efficiency_levels:
                        if int(level.get("aggregate_size", -1)) != target_size:
                            continue
                        value = level.get(metric)
                        if value is None or not np.isfinite(value) or value <= 0.0:
                            return target_size, None
                        return target_size, float(value)
                    return (target_size if target_size > 0 else None), None

                def full_direction_quadratic_model(
                    linearization: Any,
                    coarse_direction: np.ndarray,
                    frame_direction: np.ndarray,
                ) -> tuple[float, float]:
                    """Evaluate the damped local model for the actual trial direction."""

                    from scipy import sparse

                    jacobian = linearization.jac
                    if not sparse.issparse(jacobian):
                        jacobian = sparse.csr_matrix(
                            np.asarray(jacobian, dtype=np.float64)
                        )
                    else:
                        jacobian = jacobian.tocsr()
                    coarse_jacobian = jacobian[:, :frame_base]
                    frame_jacobian = jacobian[:, frame_base:num_variables]
                    coarse_diagonal = np.asarray(
                        coarse_jacobian.multiply(coarse_jacobian).sum(axis=0),
                        dtype=np.float64,
                    ).reshape(-1)
                    frame_diagonal = np.asarray(
                        frame_jacobian.multiply(frame_jacobian).sum(axis=0),
                        dtype=np.float64,
                    ).reshape(-1)
                    coarse_floor = max(
                        float(np.max(coarse_diagonal, initial=0.0)) * 1e-12,
                        1e-12,
                    )
                    frame_floor = max(
                        float(np.max(frame_diagonal, initial=0.0)) * 1e-12,
                        1e-12,
                    )
                    coarse_regularizer = float(
                        cfg.joint_schur_damping
                    ) * np.maximum(coarse_diagonal, coarse_floor)
                    frame_regularizer = float(
                        cfg.joint_schur_damping
                    ) * np.maximum(frame_diagonal, frame_floor)
                    gradient = np.asarray(
                        linearization.grad, dtype=np.float64
                    ).reshape(-1)
                    directional_gradient = float(
                        np.dot(gradient[:frame_base], coarse_direction)
                        + np.dot(
                            gradient[frame_base:num_variables], frame_direction
                        )
                    )
                    projected_direction = (
                        coarse_jacobian @ coarse_direction
                        + frame_jacobian @ frame_direction
                    )
                    directional_curvature = float(
                        np.dot(projected_direction, projected_direction)
                        + np.dot(
                            coarse_regularizer * coarse_direction,
                            coarse_direction,
                        )
                        + np.dot(
                            frame_regularizer * frame_direction,
                            frame_direction,
                        )
                    )
                    return directional_gradient, directional_curvature

                oracle_o0_recorded = False
                oracle_first_no_accepted_recorded = False

                def ensure_frozen_step_oracle() -> dict[str, Any]:
                    nonlocal schur_frozen_constrained_step_oracle
                    if schur_frozen_constrained_step_oracle is None:
                        schur_frozen_constrained_step_oracle = {
                            "schema": (
                                "hierarchical_joint_schur_frozen_"
                                "constrained_step_oracle_v1"
                            ),
                            "enabled": True,
                            "valid": False,
                            "mode": "diagnostic_only_frozen_constrained_step_oracle",
                            "solution_mutated": False,
                            "gt_free": True,
                            "linearization_scope": (
                                "frozen_current_physical_point"
                            ),
                            "topology_linearization": (
                                "raw_physical_coarse_residual_and_jacobian"
                            ),
                            "metric": (
                                "frozen_robust_damped_column_diagonal_with_"
                                "coarse_frame_block_diagnostics"
                            ),
                            "alpha_grid_policy": (
                                "production_current_alpha_times_backtracks; "
                                "extended_tail_alphas_recorded_separately"
                            ),
                            "solver_accounting_policy": (
                                "solver_nfev_and_candidate_selection_unchanged; "
                                "oracle_residual_evaluations_counted_separately; "
                                "wall_time_and_analytic_jacobian_calls_may_increase"
                            ),
                            "checkpoints": [],
                            "oracle_extra_residual_evaluations": 0,
                            "oracle_extra_jacobian_evaluations": 0,
                        }
                    return schur_frozen_constrained_step_oracle

                def record_oracle_invalid(
                    checkpoint: str,
                    *,
                    outer_iteration: int,
                    reason: str,
                ) -> None:
                    oracle = ensure_frozen_step_oracle()
                    oracle["checkpoints"].append(
                        {
                            "checkpoint": checkpoint,
                            "outer_iteration": int(outer_iteration),
                            "valid": False,
                            "reason": reason,
                            "solution_mutated": False,
                        }
                    )

                def record_frozen_constrained_step_oracle(
                    checkpoint: str,
                    *,
                    outer_iteration: int,
                    linearization: Any,
                    start: np.ndarray,
                    coarse_before: np.ndarray,
                    coarse_statistics_before: dict[str, dict[str, float | int | None]],
                    full_cost_before: float,
                    coarse_direction: np.ndarray | None,
                    frame_direction: np.ndarray | None,
                    production_frame_direction: np.ndarray | None,
                    current_alpha_for_checkpoint: float,
                    direction_model_source: str,
                    stop_reason_at_checkpoint: str,
                ) -> None:
                    nonlocal oracle_o0_recorded
                    nonlocal oracle_first_no_accepted_recorded
                    nonlocal schur_oracle_extra_residual_evaluations
                    nonlocal schur_oracle_extra_jacobian_evaluations
                    nonlocal schur_active_set_restoration_rank_direction
                    if not cfg.joint_schur_frozen_constrained_step_oracle:
                        return
                    if checkpoint == "O0_initial":
                        if oracle_o0_recorded:
                            return
                        oracle_o0_recorded = True
                    elif checkpoint == "first_production_no_accepted_alpha":
                        if oracle_first_no_accepted_recorded:
                            return
                        oracle_first_no_accepted_recorded = True
                    else:
                        raise ValueError(f"unknown oracle checkpoint {checkpoint!r}")

                    if coarse_direction is None or frame_direction is None:
                        record_oracle_invalid(
                            checkpoint,
                            outer_iteration=outer_iteration,
                            reason="schur_direction_unavailable",
                        )
                        return
                    if production_frame_direction is None:
                        production_frame_direction = frame_direction
                    if active_analytic_jacobian is None:
                        record_oracle_invalid(
                            checkpoint,
                            outer_iteration=outer_iteration,
                            reason="analytic_jacobian_unavailable",
                        )
                        return
                    if len(edges) == 0 or frame_base <= 0:
                        record_oracle_invalid(
                            checkpoint,
                            outer_iteration=outer_iteration,
                            reason="coarse_topology_rows_unavailable",
                        )
                        return

                    from scipy import sparse

                    raw_jacobian = getattr(linearization, "raw_jac", None)
                    raw_jacobian_source = "linearization_raw_jacobian"
                    if raw_jacobian is None:
                        raw_jacobian = active_analytic_jacobian(start)
                        raw_jacobian_source = "extra_active_analytic_jacobian"
                        schur_oracle_extra_jacobian_evaluations += 1
                    if not sparse.issparse(raw_jacobian):
                        raw_jacobian = sparse.csr_matrix(
                            np.asarray(raw_jacobian, dtype=np.float64)
                        )
                    else:
                        raw_jacobian = raw_jacobian.tocsr()
                    robust_jacobian = getattr(linearization, "jac", None)
                    if robust_jacobian is None:
                        record_oracle_invalid(
                            checkpoint,
                            outer_iteration=outer_iteration,
                            reason="robust_jacobian_unavailable",
                        )
                        return
                    if not sparse.issparse(robust_jacobian):
                        robust_jacobian = sparse.csr_matrix(
                            np.asarray(robust_jacobian, dtype=np.float64)
                        )
                    else:
                        robust_jacobian = robust_jacobian.tocsr()
                    if raw_jacobian.shape != (row_count, num_variables):
                        record_oracle_invalid(
                            checkpoint,
                            outer_iteration=outer_iteration,
                            reason="raw_jacobian_shape_mismatch",
                        )
                        return
                    if robust_jacobian.shape != (row_count, num_variables):
                        record_oracle_invalid(
                            checkpoint,
                            outer_iteration=outer_iteration,
                            reason="robust_jacobian_shape_mismatch",
                        )
                        return

                    coarse_row_stop = 7 * len(edges)
                    coarse_values = np.asarray(
                        coarse_before, dtype=np.float64
                    ).reshape(len(edges), 7)
                    coarse_norms = np.linalg.norm(coarse_values, axis=1)
                    edge_kinds = np.asarray([edge.kind for edge in edges], dtype=object)
                    active_edge_indices: list[int] = []
                    active_records: list[dict[str, Any]] = []
                    for family, mask in (
                        ("all", np.ones(len(edges), dtype=bool)),
                        ("adjacent", edge_kinds == "adjacent"),
                        ("loop", edge_kinds == "loop"),
                    ):
                        selected = np.flatnonzero(mask)
                        if selected.size == 0:
                            continue
                        edge_index = int(selected[np.argmax(coarse_norms[selected])])
                        if coarse_norms[edge_index] <= 0.0:
                            continue
                        if edge_index not in active_edge_indices:
                            active_edge_indices.append(edge_index)
                        active_records.append(
                            {
                                "family": family,
                                "edge_index": edge_index,
                                "edge_kind": str(edges[edge_index].kind),
                                "residual_norm": float(coarse_norms[edge_index]),
                            }
                        )
                    if not active_edge_indices:
                        record_oracle_invalid(
                            checkpoint,
                            outer_iteration=outer_iteration,
                            reason="topology_active_set_empty",
                        )
                        return

                    active_rows = []
                    edge_to_matrix_row = {
                        int(edge_index): int(matrix_row)
                        for matrix_row, edge_index in enumerate(active_edge_indices)
                    }
                    for record in active_records:
                        record["matrix_row"] = edge_to_matrix_row[
                            int(record["edge_index"])
                        ]
                    for edge_index in active_edge_indices:
                        row_start = 7 * edge_index
                        row_stop = row_start + 7
                        unit = coarse_values[edge_index] / coarse_norms[edge_index]
                        active_rows.append(
                            np.asarray(
                                raw_jacobian[row_start:row_stop, :].T @ unit,
                                dtype=np.float64,
                            ).reshape(-1)
                        )
                    active_matrix = np.vstack(active_rows)
                    if not np.all(np.isfinite(active_matrix)):
                        record_oracle_invalid(
                            checkpoint,
                            outer_iteration=outer_iteration,
                            reason="topology_active_matrix_nonfinite",
                        )
                        return

                    robust_diagonal = np.asarray(
                        robust_jacobian.multiply(robust_jacobian).sum(axis=0),
                        dtype=np.float64,
                    ).reshape(-1)
                    metric_floor = max(
                        float(np.max(robust_diagonal, initial=0.0)) * 1e-12,
                        1e-12,
                    )
                    regularizer = float(cfg.joint_schur_damping) * np.maximum(
                        robust_diagonal,
                        metric_floor,
                    )
                    metric_diagonal = np.maximum(
                        robust_diagonal + regularizer,
                        metric_floor,
                    )
                    metric_inverse = 1.0 / metric_diagonal
                    raw_direction = np.concatenate(
                        (
                            np.asarray(coarse_direction, dtype=np.float64),
                            np.asarray(frame_direction, dtype=np.float64),
                        )
                    )
                    production_direction = np.concatenate(
                        (
                            np.asarray(coarse_direction, dtype=np.float64),
                            np.asarray(
                                production_frame_direction,
                                dtype=np.float64,
                            ),
                        )
                    )
                    active_derivative = active_matrix @ raw_direction
                    weighted_active = active_matrix * metric_inverse[None, :]
                    gram = weighted_active @ active_matrix.T
                    gram_floor = max(
                        float(np.max(np.diag(gram), initial=0.0)) * 1e-10,
                        1e-12,
                    )
                    gram = 0.5 * (gram + gram.T)
                    gram.flat[:: gram.shape[0] + 1] += gram_floor

                    def apply_metric_correction(
                        direction: np.ndarray,
                        rhs: np.ndarray,
                    ) -> tuple[np.ndarray, np.ndarray]:
                        multipliers = np.linalg.solve(gram, rhs)
                        correction = metric_inverse * (active_matrix.T @ multipliers)
                        return direction - correction, correction

                    try:
                        tangent_direction, tangent_correction = (
                            apply_metric_correction(
                                raw_direction,
                                active_derivative,
                            )
                        )
                        normal_direction, normal_correction = (
                            apply_metric_correction(
                                raw_direction,
                                active_derivative + np.abs(active_derivative),
                            )
                        )
                    except np.linalg.LinAlgError:
                        record_oracle_invalid(
                            checkpoint,
                            outer_iteration=outer_iteration,
                            reason="topology_metric_projection_failed",
                        )
                        return

                    assert initial_coarse_cost is not None
                    assert initial_coarse_statistics is not None
                    production_alpha_grid = tuple(
                        float(current_alpha_for_checkpoint * (0.5**backtrack))
                        for backtrack in range(cfg.joint_schur_step_backtracks)
                    )
                    coarse_cost_before_scalar = float(coarse_before @ coarse_before)

                    def _finite_float_or_none(value: Any) -> float | None:
                        if value is None:
                            return None
                        candidate = float(value)
                        if not np.isfinite(candidate):
                            return None
                        return candidate

                    def build_rank_boundary_budgeted_projection() -> tuple[
                        np.ndarray,
                        dict[str, Any],
                    ]:
                        alpha_ref_source = "production_grid_min_legal_alpha"
                        if cfg.joint_schur_step_min_accepted_alpha is not None:
                            alpha_ref = float(
                                cfg.joint_schur_step_min_accepted_alpha
                            )
                            alpha_ref_source = "configured_min_accepted_alpha"
                        else:
                            legal_alphas = [
                                float(alpha)
                                for alpha in production_alpha_grid
                                if cfg.joint_schur_step_min_accepted_alpha is None
                                or alpha >= cfg.joint_schur_step_min_accepted_alpha
                            ]
                            alpha_ref = (
                                float(min(legal_alphas))
                                if legal_alphas
                                else float(min(production_alpha_grid))
                            )
                        diagnostics: dict[str, Any] = {
                            "schema": (
                                "rank_boundary_budgeted_projection_v1"
                            ),
                            "protocol": (
                                "slack_aware_halfspace_metric_projection_"
                                "not_top_tail_clamping"
                            ),
                            "loop_quantile_method": "numpy_default_linear",
                            "rank_tie_break": "edge_index",
                            "band_selection": "rank_boundary_band_not_top_tail",
                            "top_tail_clamping": False,
                            "alpha_ref": alpha_ref,
                            "alpha_ref_source": alpha_ref_source,
                            "valid": False,
                            "reason": None,
                            "constraint_rows": [],
                            "active_constraint_count": 0,
                            "metric_correction_l2": 0.0,
                            "metric_projection_gram_condition": None,
                        }
                        initial_loop_p95 = _finite_float_or_none(
                            initial_coarse_statistics.get("loop", {}).get("p95")
                        )
                        if initial_loop_p95 is None or initial_loop_p95 <= 0.0:
                            diagnostics["reason"] = "initial_loop_p95_unavailable"
                            return raw_direction.copy(), diagnostics
                        if initial_coarse_cost is None or initial_coarse_cost <= 0.0:
                            diagnostics["reason"] = "initial_coarse_cost_unavailable"
                            return raw_direction.copy(), diagnostics
                        loop_edge_indices = np.flatnonzero(edge_kinds == "loop")
                        loop_count = int(loop_edge_indices.size)
                        diagnostics["loop_count"] = loop_count
                        if loop_count <= 0:
                            diagnostics["reason"] = "loop_edges_unavailable"
                            return raw_direction.copy(), diagnostics

                        loop_norms = coarse_norms[loop_edge_indices]
                        sort_order = np.lexsort((loop_edge_indices, loop_norms))
                        sorted_loop_edge_indices = loop_edge_indices[sort_order]
                        h = 0.95 * float(loop_count - 1)
                        m = int(min(16, max(2, np.ceil(0.0025 * loop_count))))
                        rank_start = int(max(0, np.floor(h) - m))
                        rank_stop = int(min(loop_count - 1, np.ceil(h) + m))
                        band_ranks = np.arange(rank_start, rank_stop + 1, dtype=int)
                        band_edge_ids = sorted_loop_edge_indices[band_ranks]
                        sigma = float(max(1.0, m / 2.0))
                        band_rank_offsets = (band_ranks.astype(np.float64) - h) / sigma
                        weights = np.exp(-0.5 * band_rank_offsets * band_rank_offsets)
                        weight_sum = float(np.sum(weights))
                        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
                            diagnostics["reason"] = "rank_weight_sum_nonpositive"
                            return raw_direction.copy(), diagnostics
                        weights = weights / weight_sum
                        positive_weights = weights[weights > 0.0]
                        weight_entropy = float(
                            -np.sum(positive_weights * np.log(positive_weights))
                        )
                        current_loop_p95 = float(np.quantile(loop_norms, 0.95))
                        diagnostics.update(
                            {
                                "h": h,
                                "m": m,
                                "sigma": sigma,
                                "rank_band_start": rank_start,
                                "rank_band_stop": rank_stop,
                                "band_ranks": [int(value) for value in band_ranks],
                                "band_edge_ids": [
                                    int(value) for value in band_edge_ids
                                ],
                                "band_weights": [
                                    float(value) for value in weights
                                ],
                                "band_weight_entropy": weight_entropy,
                                "initial_loop_p95": float(initial_loop_p95),
                                "current_loop_p95": current_loop_p95,
                                "exact_c95_defect": float(
                                    current_loop_p95 / initial_loop_p95 - 1.01
                                ),
                                "aggregate_initial_energy": float(
                                    initial_coarse_cost
                                ),
                                "aggregate_current_energy": (
                                    coarse_cost_before_scalar
                                ),
                                "aggregate_defect": float(
                                    coarse_cost_before_scalar
                                    / initial_coarse_cost
                                    - 1.02
                                ),
                            }
                        )

                        coarse_raw_jacobian = raw_jacobian[:coarse_row_stop, :]
                        rank_row = np.zeros(num_variables, dtype=np.float64)
                        for weight, edge_index in zip(weights, band_edge_ids):
                            norm = float(coarse_norms[int(edge_index)])
                            if norm <= 0.0:
                                continue
                            row_start = 7 * int(edge_index)
                            row_stop = row_start + 7
                            unit = coarse_values[int(edge_index)] / norm
                            rank_row += (
                                float(weight)
                                * np.asarray(
                                    raw_jacobian[row_start:row_stop, :].T @ unit,
                                    dtype=np.float64,
                                ).reshape(-1)
                                / initial_loop_p95
                            )
                        aggregate_row = np.asarray(
                            coarse_raw_jacobian.T
                            @ (2.0 * np.asarray(coarse_before, dtype=np.float64)),
                            dtype=np.float64,
                        ).reshape(-1) / float(initial_coarse_cost)
                        row_specs: list[tuple[str, float, np.ndarray]] = [
                            (
                                "loop_p95_rank_boundary",
                                float(current_loop_p95 / initial_loop_p95 - 1.01),
                                rank_row,
                            ),
                            (
                                "aggregate_graph_energy",
                                float(
                                    coarse_cost_before_scalar
                                    / initial_coarse_cost
                                    - 1.02
                                ),
                                aggregate_row,
                            ),
                        ]
                        constraint_rows = []
                        constraint_records: list[dict[str, Any]] = []
                        for name, defect, row in row_specs:
                            row_norm = float(np.linalg.norm(row))
                            if row_norm <= 0.0 or not np.all(np.isfinite(row)):
                                constraint_records.append(
                                    {
                                        "name": name,
                                        "valid": False,
                                        "reason": "row_zero_or_nonfinite",
                                        "defect": defect,
                                        "row_norm": row_norm,
                                    }
                                )
                                continue
                            target_derivative = float(-defect / alpha_ref)
                            raw_derivative = float(row @ raw_direction)
                            constraint_rows.append(row)
                            constraint_records.append(
                                {
                                    "name": name,
                                    "valid": True,
                                    "reason": None,
                                    "defect": defect,
                                    "target_derivative": target_derivative,
                                    "raw_derivative": raw_derivative,
                                    "row_norm": row_norm,
                                }
                            )
                        valid_indices = [
                            index
                            for index, record in enumerate(constraint_records)
                            if record["valid"]
                        ]
                        if not valid_indices:
                            diagnostics["reason"] = "no_valid_constraint_rows"
                            diagnostics["constraint_rows"] = constraint_records
                            return raw_direction.copy(), diagnostics
                        constraint_matrix = np.vstack(constraint_rows)
                        targets = np.asarray(
                            [
                                constraint_records[index]["target_derivative"]
                                for index in valid_indices
                            ],
                            dtype=np.float64,
                        )
                        raw_derivatives = constraint_matrix @ raw_direction
                        active_mask = raw_derivatives > targets
                        active_count = int(np.count_nonzero(active_mask))
                        projected_direction = raw_direction.copy()
                        correction = np.zeros_like(raw_direction)
                        condition: float | None = None
                        if active_count:
                            active_constraint_matrix = constraint_matrix[active_mask]
                            excess = raw_derivatives[active_mask] - targets[
                                active_mask
                            ]
                            weighted_rows = (
                                active_constraint_matrix * metric_inverse[None, :]
                            )
                            budget_gram = weighted_rows @ active_constraint_matrix.T
                            budget_floor = max(
                                float(
                                    np.max(np.diag(budget_gram), initial=0.0)
                                )
                                * 1e-10,
                                1e-12,
                            )
                            budget_gram = 0.5 * (
                                budget_gram + budget_gram.T
                            )
                            budget_gram.flat[
                                :: budget_gram.shape[0] + 1
                            ] += budget_floor
                            try:
                                multipliers = np.linalg.solve(
                                    budget_gram,
                                    excess,
                                )
                            except np.linalg.LinAlgError:
                                diagnostics["reason"] = (
                                    "rank_boundary_metric_projection_failed"
                                )
                                diagnostics["constraint_rows"] = (
                                    constraint_records
                                )
                                return raw_direction.copy(), diagnostics
                            correction = metric_inverse * (
                                active_constraint_matrix.T @ multipliers
                            )
                            projected_direction = raw_direction - correction
                            condition = float(np.linalg.cond(budget_gram))

                        projected_derivatives = constraint_matrix @ projected_direction
                        valid_cursor = 0
                        for index, record in enumerate(constraint_records):
                            if not record["valid"]:
                                continue
                            target = targets[valid_cursor]
                            raw_value = raw_derivatives[valid_cursor]
                            projected_value = projected_derivatives[valid_cursor]
                            active = bool(active_mask[valid_cursor])
                            record.update(
                                {
                                    "active": active,
                                    "excess": float(
                                        max(0.0, raw_value - target)
                                    ),
                                    "projected_derivative": float(
                                        projected_value
                                    ),
                                    "first_order_satisfied": bool(
                                        projected_value
                                        <= target
                                        + max(1e-8, 1e-8 * abs(target))
                                    ),
                                }
                            )
                            constraint_records[index] = record
                            valid_cursor += 1
                        diagnostics.update(
                            {
                                "valid": True,
                                "reason": None,
                                "constraint_rows": constraint_records,
                                "active_constraint_count": active_count,
                                "metric_correction_l2": float(
                                    np.linalg.norm(correction)
                                ),
                                "metric_projection_gram_condition": condition,
                            }
                        )
                        return projected_direction, diagnostics

                    (
                        rank_boundary_direction,
                        rank_boundary_diagnostics,
                    ) = build_rank_boundary_budgeted_projection()
                    if checkpoint == "first_production_no_accepted_alpha":
                        schur_active_set_restoration_rank_direction = (
                            np.asarray(
                                rank_boundary_direction,
                                dtype=np.float64,
                            ).copy()
                        )

                    def evaluate_oracle_direction(
                        label: str,
                        direction: np.ndarray,
                        *,
                        direction_source: str,
                        extra_fields: dict[str, Any] | None = None,
                    ) -> dict[str, Any]:
                        nonlocal schur_oracle_extra_residual_evaluations
                        coarse_part = direction[:frame_base]
                        frame_part = direction[frame_base:num_variables]
                        directional_gradient, directional_curvature = (
                            full_direction_quadratic_model(
                                linearization,
                                coarse_part,
                                frame_part,
                            )
                        )
                        direction_record = {
                            "name": label,
                            "direction_source": direction_source,
                            "coarse_step_l2": float(np.linalg.norm(coarse_part)),
                            "frame_step_l2": float(np.linalg.norm(frame_part)),
                            "full_step_l2": float(np.linalg.norm(direction)),
                            "active_topology_directional_derivative": [
                                float(value) for value in active_matrix @ direction
                            ],
                            "full_directional_gradient": directional_gradient,
                            "full_directional_curvature": directional_curvature,
                            "trials": [],
                        }
                        if extra_fields:
                            direction_record.update(extra_fields)
                        for backtrack, alpha in enumerate(production_alpha_grid):
                            proposed = start + float(alpha) * direction
                            candidate = np.clip(proposed, lower, upper)
                            clipped_coarse_variable_count = int(
                                np.count_nonzero(
                                    ~np.isclose(
                                        candidate[:frame_base],
                                        proposed[:frame_base],
                                        rtol=0.0,
                                        atol=1e-12,
                                    )
                                )
                            )
                            clipped_frame_variable_count = int(
                                np.count_nonzero(
                                    ~np.isclose(
                                        candidate[frame_base:num_variables],
                                        proposed[frame_base:num_variables],
                                        rtol=0.0,
                                        atol=1e-12,
                                    )
                                )
                            )
                            candidate_residual = compute_residual(candidate)
                            schur_oracle_extra_residual_evaluations += 1
                            candidate_coarse = candidate_residual[:coarse_row_stop]
                            candidate_coarse_cost = float(
                                candidate_coarse @ candidate_coarse
                            )
                            candidate_statistics = (
                                _coarse_edge_residual_statistics(
                                    candidate_coarse,
                                    edges,
                                )
                            )
                            step_ratios = _coarse_edge_statistics_ratios(
                                coarse_statistics_before,
                                candidate_statistics,
                            )
                            cumulative_ratios = _coarse_edge_statistics_ratios(
                                initial_coarse_statistics,
                                candidate_statistics,
                            )
                            coarse_cost_ratio = (
                                candidate_coarse_cost / coarse_cost_before_scalar
                                if coarse_cost_before_scalar > 0.0
                                else 1.0
                                if candidate_coarse_cost == 0.0
                                else np.inf
                            )
                            cumulative_coarse_cost_ratio = (
                                candidate_coarse_cost / initial_coarse_cost
                                if initial_coarse_cost > 0.0
                                else 1.0
                                if candidate_coarse_cost == 0.0
                                else np.inf
                            )
                            family_gate_checks = (
                                (
                                    cfg.joint_schur_step_max_adjacent_p95_ratio,
                                    step_ratios["adjacent"]["p95"],
                                ),
                                (
                                    effective_loop_p95_ratio,
                                    step_ratios["loop"]["p95"],
                                ),
                                (
                                    effective_loop_max_ratio,
                                    step_ratios["loop"]["max"],
                                ),
                            )
                            edge_family_gate_passed = all(
                                threshold is None
                                or observed is None
                                or observed <= threshold
                                for threshold, observed in family_gate_checks
                            )
                            cumulative_family_gate_checks = (
                                (
                                    cfg.joint_schur_step_max_adjacent_p95_ratio,
                                    cumulative_ratios["adjacent"]["p95"],
                                ),
                                (
                                    effective_loop_p95_ratio,
                                    cumulative_ratios["loop"]["p95"],
                                ),
                                (
                                    effective_loop_max_ratio,
                                    cumulative_ratios["loop"]["max"],
                                ),
                            )
                            cumulative_edge_family_gate_passed = all(
                                threshold is None
                                or observed is None
                                or observed <= threshold
                                for threshold, observed in cumulative_family_gate_checks
                            )
                            aggregate_gate_passed = bool(
                                coarse_cost_ratio
                                <= cfg.joint_schur_step_max_graph_cost_ratio
                            )
                            cumulative_aggregate_gate_passed = bool(
                                cumulative_coarse_cost_ratio
                                <= cfg.joint_schur_step_max_graph_cost_ratio
                            )
                            full_cost_after = _scipy_compatible_robust_cost(
                                candidate_residual,
                                loss=cfg.robust_loss,
                                f_scale=cfg.f_scale,
                            )
                            objective_gate_passed = bool(
                                full_cost_after
                                <= full_cost_before
                                + max(1e-12, 1e-10 * abs(full_cost_before))
                            )
                            topology_gate_passed = bool(
                                aggregate_gate_passed
                                and cumulative_aggregate_gate_passed
                                and cumulative_edge_family_gate_passed
                            )
                            predicted_reduction = float(
                                -(
                                    float(alpha) * directional_gradient
                                    + 0.5
                                    * float(alpha) ** 2
                                    * directional_curvature
                                )
                            )
                            reduction_ratio = (
                                float(
                                    (full_cost_before - full_cost_after)
                                    / predicted_reduction
                                )
                                if predicted_reduction > 0.0
                                else None
                            )
                            no_clipping_gate_passed = bool(
                                clipped_coarse_variable_count == 0
                                and clipped_frame_variable_count == 0
                            )
                            minimum_alpha_gate_passed = bool(
                                cfg.joint_schur_step_min_accepted_alpha is None
                                or alpha
                                >= cfg.joint_schur_step_min_accepted_alpha
                            )
                            direction_model_gate_passed = bool(
                                no_clipping_gate_passed
                                and predicted_reduction > 0.0
                                and minimum_alpha_gate_passed
                            )
                            hierarchy_gate_passed = bool(
                                aggregate_gate_passed
                                and cumulative_aggregate_gate_passed
                                and cumulative_edge_family_gate_passed
                                and direction_model_gate_passed
                            )
                            production_equivalent_gate_passed = bool(
                                hierarchy_gate_passed
                                and np.all(np.isfinite(candidate))
                                and np.all(np.isfinite(candidate_residual))
                                and np.isfinite(full_cost_after)
                                and objective_gate_passed
                                and reduction_ratio is not None
                                and reduction_ratio >= 0.25
                            )
                            direction_record["trials"].append(
                                {
                                    "backtrack": int(backtrack),
                                    "alpha": float(alpha),
                                    "alpha_role": "production",
                                    "finite": bool(
                                        np.all(np.isfinite(candidate))
                                        and np.all(
                                            np.isfinite(candidate_residual)
                                        )
                                        and np.isfinite(full_cost_after)
                                    ),
                                    "clipped_coarse_variable_count": (
                                        clipped_coarse_variable_count
                                    ),
                                    "clipped_frame_variable_count": (
                                        clipped_frame_variable_count
                                    ),
                                    "full_robust_cost": float(full_cost_after),
                                    "full_robust_cost_ratio": (
                                        float(full_cost_after / full_cost_before)
                                        if full_cost_before > 0.0
                                        else None
                                    ),
                                    "full_robust_cost_nonincreasing": (
                                        objective_gate_passed
                                    ),
                                    "objective_gate_passed": (
                                        objective_gate_passed
                                    ),
                                    "full_robust_actual_reduction": float(
                                        full_cost_before - full_cost_after
                                    ),
                                    "full_reduction_ratio": reduction_ratio,
                                    "coarse_graph_squared_residual": (
                                        candidate_coarse_cost
                                    ),
                                    "coarse_graph_cost_ratio": float(
                                        coarse_cost_ratio
                                    ),
                                    "cumulative_coarse_graph_cost_ratio": float(
                                        cumulative_coarse_cost_ratio
                                    ),
                                    "coarse_edge_statistics_ratios": step_ratios,
                                    "cumulative_coarse_edge_statistics_ratios": (
                                        cumulative_ratios
                                    ),
                                    "aggregate_graph_gate_passed": (
                                        aggregate_gate_passed
                                    ),
                                    "edge_family_gate_passed": (
                                        edge_family_gate_passed
                                    ),
                                    "edge_family_gate_role": (
                                        "advisory_step_diagnostic"
                                    ),
                                    "cumulative_aggregate_graph_gate_passed": (
                                        cumulative_aggregate_gate_passed
                                    ),
                                    "cumulative_edge_family_gate_passed": (
                                        cumulative_edge_family_gate_passed
                                    ),
                                    "topology_gate_passed": topology_gate_passed,
                                    "no_clipping_gate_passed": (
                                        no_clipping_gate_passed
                                    ),
                                    "minimum_alpha_gate_passed": (
                                        minimum_alpha_gate_passed
                                    ),
                                    "direction_model_gate_passed": (
                                        direction_model_gate_passed
                                    ),
                                    "hierarchy_gate_passed": hierarchy_gate_passed,
                                    "production_equivalent_gate_passed": (
                                        production_equivalent_gate_passed
                                    ),
                                    "objective_and_topology_gate_passed": bool(
                                        objective_gate_passed
                                        and topology_gate_passed
                                    ),
                                    "full_quadratic_predicted_reduction": (
                                        predicted_reduction
                                    ),
                                }
                            )
                        return direction_record

                    checkpoint_record = {
                        "checkpoint": checkpoint,
                        "outer_iteration": int(outer_iteration),
                        "valid": True,
                        "reason": None,
                        "stop_reason_at_checkpoint": stop_reason_at_checkpoint,
                        "solution_mutated": False,
                        "current_alpha": float(current_alpha_for_checkpoint),
                        "production_alpha_grid": list(production_alpha_grid),
                        "extended_tail_alpha_grid": [],
                        "production_backtracks": int(
                            cfg.joint_schur_step_backtracks
                        ),
                        "direction_model_source": direction_model_source,
                        "raw_jacobian_source": raw_jacobian_source,
                        "active_topology_rows": active_records,
                        "active_topology_derivative_raw": [
                            float(value) for value in active_derivative
                        ],
                        "active_topology_derivative_tangent": [
                            float(value)
                            for value in active_matrix @ tangent_direction
                        ],
                        "active_topology_derivative_normal_corrected": [
                            float(value)
                            for value in active_matrix @ normal_direction
                        ],
                        "metric_diagonal_min": float(np.min(metric_diagonal)),
                        "metric_diagonal_max": float(np.max(metric_diagonal)),
                        "metric_projection_gram_condition": float(
                            np.linalg.cond(gram)
                        ),
                        "tangent_correction_l2": float(
                            np.linalg.norm(tangent_correction)
                        ),
                        "normal_correction_l2": float(
                            np.linalg.norm(normal_correction)
                        ),
                        "directions": [
                            evaluate_oracle_direction(
                                "raw_schur",
                                raw_direction,
                                direction_source="raw_schur_backsubstitution",
                            ),
                            evaluate_oracle_direction(
                                "actual_production_direction",
                                production_direction,
                                direction_source=direction_model_source,
                            ),
                            evaluate_oracle_direction(
                                "topology_tangent_projected",
                                tangent_direction,
                                direction_source=(
                                    "raw_schur_metric_topology_tangent_projection"
                                ),
                            ),
                            evaluate_oracle_direction(
                                "topology_normal_corrected",
                                normal_direction,
                                direction_source=(
                                    "raw_schur_metric_topology_normal_correction"
                                ),
                            ),
                            evaluate_oracle_direction(
                                "rank_boundary_budgeted_projection",
                                rank_boundary_direction,
                                direction_source=(
                                    "rank_boundary_budgeted_slack_aware_"
                                    "halfspace_projection"
                                ),
                                extra_fields={
                                    "rank_boundary_projection": (
                                        rank_boundary_diagnostics
                                    )
                                },
                            ),
                        ],
                    }
                    oracle = ensure_frozen_step_oracle()
                    oracle["valid"] = True
                    oracle["checkpoints"].append(checkpoint_record)
                    oracle["oracle_extra_residual_evaluations"] = int(
                        schur_oracle_extra_residual_evaluations
                    )
                    oracle["oracle_extra_jacobian_evaluations"] = int(
                        schur_oracle_extra_jacobian_evaluations
                    )

                def try_one_active_set_restoration_from_first_reject(
                    *,
                    outer_iteration: int,
                    start: np.ndarray,
                    full_cost_before: float,
                    current_alpha_for_refresh: float,
                ) -> tuple[bool, np.ndarray, Any | None, float]:
                    nonlocal schur_active_set_restoration_applied
                    nonlocal schur_active_set_restoration_diagnostics
                    nonlocal schur_trial_residual_evaluations
                    if (
                        not cfg.joint_schur_one_active_set_restoration_at_first_reject
                    ):
                        return False, start, None, current_alpha_for_refresh
                    if schur_active_set_restoration_diagnostics is not None:
                        return False, start, None, current_alpha_for_refresh
                    diagnostics: dict[str, Any] = {
                        "schema": (
                            "hierarchical_joint_schur_one_active_set_"
                            "restoration_at_first_reject_v1"
                        ),
                        "enabled": True,
                        "trigger_checkpoint": (
                            "first_production_no_accepted_alpha"
                        ),
                        "trigger_outer_iteration": int(outer_iteration),
                        "candidate_source": (
                            "frozen_oracle_rank_boundary_budgeted_projection"
                        ),
                        "solution_mutated": False,
                        "applied": False,
                        "selected_alpha": None,
                        "selected_backtrack": None,
                        "reason": None,
                        "oracle_extra_residual_evaluations_at_trigger": int(
                            schur_oracle_extra_residual_evaluations
                        ),
                        "oracle_extra_jacobian_evaluations_at_trigger": int(
                            schur_oracle_extra_jacobian_evaluations
                        ),
                        "restoration_function_evaluations": 0,
                        "fresh_relinearization": {
                            "performed": False,
                            "nfev": 0,
                            "njev": 0,
                        },
                        "refresh_performed": False,
                        "refresh_accepted": None,
                        "refresh_stop_reason": None,
                    }
                    oracle = ensure_frozen_step_oracle()
                    checkpoint_record = (
                        oracle["checkpoints"][-1]
                        if oracle.get("checkpoints")
                        else None
                    )
                    if (
                        not isinstance(checkpoint_record, dict)
                        or checkpoint_record.get("checkpoint")
                        != "first_production_no_accepted_alpha"
                    ):
                        diagnostics["reason"] = "first_reject_checkpoint_unavailable"
                        schur_active_set_restoration_diagnostics = diagnostics
                        return False, start, None, current_alpha_for_refresh
                    if checkpoint_record.get("valid") is not True:
                        diagnostics["reason"] = "first_reject_checkpoint_invalid"
                        schur_active_set_restoration_diagnostics = diagnostics
                        return False, start, None, current_alpha_for_refresh
                    rank_direction = schur_active_set_restoration_rank_direction
                    if rank_direction is None:
                        diagnostics["reason"] = "rank_boundary_direction_unavailable"
                        schur_active_set_restoration_diagnostics = diagnostics
                        return False, start, None, current_alpha_for_refresh
                    rank_record = next(
                        (
                            item
                            for item in checkpoint_record.get("directions", [])
                            if item.get("name")
                            == "rank_boundary_budgeted_projection"
                        ),
                        None,
                    )
                    if not isinstance(rank_record, dict):
                        diagnostics["reason"] = (
                            "rank_boundary_direction_record_unavailable"
                        )
                        schur_active_set_restoration_diagnostics = diagnostics
                        return False, start, None, current_alpha_for_refresh
                    projection = rank_record.get("rank_boundary_projection")
                    projection_ready, projection_reason = (
                        _joint_schur_rank_boundary_restoration_readiness(
                            projection
                        )
                    )
                    diagnostics["rank_boundary_projection_valid"] = (
                        isinstance(projection, dict)
                        and projection.get("valid") is True
                    )
                    diagnostics["rank_boundary_active_constraint_count"] = (
                        int(projection.get("active_constraint_count") or 0)
                        if isinstance(projection, dict)
                        else None
                    )
                    diagnostics["rank_boundary_metric_correction_l2"] = (
                        float(projection.get("metric_correction_l2") or 0.0)
                        if isinstance(projection, dict)
                        else None
                    )
                    if not projection_ready:
                        diagnostics["reason"] = projection_reason
                        schur_active_set_restoration_diagnostics = diagnostics
                        return False, start, None, current_alpha_for_refresh
                    passed_trials = [
                        trial
                        for trial in rank_record.get("trials", [])
                        if trial.get("production_equivalent_gate_passed") is True
                    ]
                    diagnostics["production_equivalent_pass_alphas"] = [
                        float(trial["alpha"]) for trial in passed_trials
                    ]
                    selected_trial = _joint_schur_select_restoration_trial(
                        rank_record.get("trials", [])
                    )
                    if selected_trial is None:
                        diagnostics["reason"] = (
                            "rank_boundary_has_no_production_equivalent_alpha"
                        )
                        schur_active_set_restoration_diagnostics = diagnostics
                        return False, start, None, current_alpha_for_refresh
                    selected_alpha = float(selected_trial["alpha"])
                    if not _joint_schur_restoration_has_refresh_budget(
                        outer_iteration,
                        requested_outer_iterations,
                    ):
                        diagnostics["reason"] = (
                            "refresh_outer_budget_unavailable_after_first_reject"
                        )
                        diagnostics["selected_alpha"] = selected_alpha
                        diagnostics["selected_backtrack"] = int(
                            selected_trial.get("backtrack", -1)
                        )
                        schur_active_set_restoration_diagnostics = diagnostics
                        return False, start, None, current_alpha_for_refresh
                    proposed = start + selected_alpha * np.asarray(
                        rank_direction, dtype=np.float64
                    )
                    candidate = np.clip(proposed, lower, upper)
                    if not np.all(np.isfinite(candidate)):
                        diagnostics["reason"] = "candidate_nonfinite"
                        schur_active_set_restoration_diagnostics = diagnostics
                        return False, start, None, current_alpha_for_refresh
                    candidate_residual = residual(candidate)
                    schur_trial_residual_evaluations += 1
                    diagnostics["restoration_function_evaluations"] = 1
                    full_cost_after = _scipy_compatible_robust_cost(
                        candidate_residual,
                        loss=cfg.robust_loss,
                        f_scale=cfg.f_scale,
                    )
                    if (
                        not np.all(np.isfinite(candidate_residual))
                        or not np.isfinite(full_cost_after)
                    ):
                        diagnostics["reason"] = "candidate_residual_nonfinite"
                        schur_active_set_restoration_diagnostics = diagnostics
                        return False, start, None, current_alpha_for_refresh
                    actual_full_reduction = full_cost_before - full_cost_after
                    reduction_ratio = selected_trial.get("full_reduction_ratio")
                    reduction_ratio = (
                        None
                        if reduction_ratio is None
                        else float(reduction_ratio)
                    )
                    record_schur_final_accepted_step(
                        candidate=candidate,
                        candidate_residual=candidate_residual,
                        full_cost_after=full_cost_after,
                        step_start=start,
                        full_cost_before=full_cost_before,
                        actual_full_reduction=actual_full_reduction,
                        reduction_ratio=reduction_ratio,
                        alpha=selected_alpha,
                    )
                    linearization_count_before = len(schur_linearization_results)
                    fresh_linearization = linearize_full(candidate)
                    fresh_nfev = sum(
                        int(getattr(item, "nfev", 0) or 0)
                        for item in schur_linearization_results[
                            linearization_count_before:
                        ]
                    )
                    fresh_njev_values = [
                        getattr(item, "njev", None)
                        for item in schur_linearization_results[
                            linearization_count_before:
                        ]
                    ]
                    fresh_njev = (
                        None
                        if any(value is None for value in fresh_njev_values)
                        else sum(int(value or 0) for value in fresh_njev_values)
                    )
                    next_alpha = (
                        min(1.0, 2.0 * selected_alpha)
                        if adaptive_outer
                        and reduction_ratio is not None
                        and reduction_ratio >= 0.75
                        else selected_alpha
                    )
                    diagnostics.update(
                        {
                            "applied": True,
                            "solution_mutated": True,
                            "reason": None,
                            "selected_alpha": selected_alpha,
                            "selected_backtrack": int(
                                selected_trial.get("backtrack", -1)
                            ),
                            "selected_trial": {
                                "alpha": selected_alpha,
                                "backtrack": int(
                                    selected_trial.get("backtrack", -1)
                                ),
                                "production_equivalent_gate_passed": True,
                                "full_reduction_ratio": reduction_ratio,
                                "full_quadratic_predicted_reduction": (
                                    selected_trial.get(
                                        "full_quadratic_predicted_reduction"
                                    )
                                ),
                                "objective_gate_passed": selected_trial.get(
                                    "objective_gate_passed"
                                ),
                                "topology_gate_passed": selected_trial.get(
                                    "topology_gate_passed"
                                ),
                            },
                            "restoration": {
                                "full_robust_cost_before": float(
                                    full_cost_before
                                ),
                                "full_robust_cost_after": float(
                                    full_cost_after
                                ),
                                "full_robust_actual_reduction": float(
                                    actual_full_reduction
                                ),
                                "full_reduction_ratio": reduction_ratio,
                                "coarse_graph_squared_residual": (
                                    selected_trial.get(
                                        "coarse_graph_squared_residual"
                                    )
                                ),
                                "coarse_graph_cost_ratio": selected_trial.get(
                                    "coarse_graph_cost_ratio"
                                ),
                                "cumulative_coarse_graph_cost_ratio": (
                                    selected_trial.get(
                                        "cumulative_coarse_graph_cost_ratio"
                                    )
                                ),
                                "aggregate_graph_gate_passed": (
                                    selected_trial.get(
                                        "aggregate_graph_gate_passed"
                                    )
                                ),
                                "cumulative_aggregate_graph_gate_passed": (
                                    selected_trial.get(
                                        "cumulative_aggregate_graph_gate_passed"
                                    )
                                ),
                                "cumulative_edge_family_gate_passed": (
                                    selected_trial.get(
                                        "cumulative_edge_family_gate_passed"
                                    )
                                ),
                                "minimum_alpha_gate_passed": selected_trial.get(
                                    "minimum_alpha_gate_passed"
                                ),
                                "direction_model_gate_passed": (
                                    selected_trial.get(
                                        "direction_model_gate_passed"
                                    )
                                ),
                            },
                            "fresh_relinearization": {
                                "performed": True,
                                "nfev": int(fresh_nfev),
                                "njev": fresh_njev,
                                "linearization_source": getattr(
                                    fresh_linearization,
                                    "linearization_source",
                                    None,
                                ),
                            },
                            "next_refresh_initial_alpha": float(next_alpha),
                            "refresh_performed": False,
                            "refresh_accepted": None,
                            "refresh_stop_reason": None,
                            "stop_policy": (
                                "apply_once_then_allow_one_standard_schur_"
                                "outer_refresh"
                            ),
                        }
                    )
                    schur_active_set_restoration_diagnostics = diagnostics
                    schur_active_set_restoration_applied = True
                    return True, candidate, fresh_linearization, float(next_alpha)

                restoration_refresh_iteration: int | None = None
                for outer_iteration in range(requested_outer_iterations):
                    freeze_coarse_scales_this_outer = bool(
                        cfg.joint_schur_step_freeze_coarse_scales
                    )
                    linearization_leak_inf = float(
                        np.linalg.norm(
                            current_frame_result.x - joint_start,
                            ord=np.inf,
                        )
                    )
                    linearization_leak_limit = float(
                        1e-10
                        * (
                            1.0
                            + np.linalg.norm(joint_start, ord=np.inf)
                        )
                    )
                    linearization_gate_passed = bool(
                        np.all(np.isfinite(current_frame_result.x))
                        and np.isfinite(current_frame_result.cost)
                        and np.all(np.isfinite(current_frame_result.fun))
                        and np.all(np.isfinite(current_frame_result.grad))
                        and linearization_leak_inf <= linearization_leak_limit
                    )
                    if not linearization_gate_passed:
                        stop_reason = "joint_linearization_gate_failed"
                        if (
                            restoration_refresh_iteration is not None
                            and outer_iteration >= restoration_refresh_iteration
                        ):
                            assert (
                                schur_active_set_restoration_diagnostics
                                is not None
                            )
                            _joint_schur_record_restoration_refresh(
                                schur_active_set_restoration_diagnostics,
                                schur_outer_diagnostics,
                                outer_iteration=outer_iteration,
                                accepted=False,
                                trial_count=0,
                                refresh_stop_reason=stop_reason,
                            )
                        break
                    if (
                        outer_iteration == 0
                        and (
                            cfg.joint_schur_step_max_initial_diagonal_low_frequency_efficiency
                            is not None
                            or cfg.joint_schur_step_max_initial_diagonal_reduction_fraction
                            is not None
                        )
                    ):
                        initial_diagonal_low_frequency_gate = (
                            _joint_diagonal_low_frequency_efficiency_from_solver_result(
                                current_frame_result,
                                frame_base=frame_base,
                                num_variables=num_variables,
                                damping=cfg.joint_schur_damping,
                                aggregate_size=(
                                    cfg.joint_schur_step_frame_low_frequency_trust_aggregate_size
                                ),
                                freeze_coarse_scales=(
                                    freeze_coarse_scales_this_outer
                                ),
                            )
                        )
                        diagonal_efficiency = (
                            initial_diagonal_low_frequency_gate["efficiency"]
                        )
                        diagonal_reduction_fraction = (
                            initial_diagonal_low_frequency_gate[
                                "predicted_quadratic_reduction_fraction"
                            ]
                        )
                        diagonal_efficiency_gate_passed = bool(
                            cfg.joint_schur_step_max_initial_diagonal_low_frequency_efficiency
                            is None
                            or (
                                diagonal_efficiency is not None
                                and diagonal_efficiency
                                <= cfg.joint_schur_step_max_initial_diagonal_low_frequency_efficiency
                            )
                        )
                        diagonal_reduction_gate_passed = bool(
                            cfg.joint_schur_step_max_initial_diagonal_reduction_fraction
                            is None
                            or (
                                diagonal_reduction_fraction is not None
                                and diagonal_reduction_fraction
                                <= cfg.joint_schur_step_max_initial_diagonal_reduction_fraction
                            )
                        )
                        diagonal_gate_passed = bool(
                            diagonal_efficiency_gate_passed
                            and diagonal_reduction_gate_passed
                        )
                        initial_diagonal_low_frequency_gate.update(
                            {
                                "maximum_efficiency": (
                                    cfg.joint_schur_step_max_initial_diagonal_low_frequency_efficiency
                                ),
                                "maximum_reduction_fraction": (
                                    cfg.joint_schur_step_max_initial_diagonal_reduction_fraction
                                ),
                                "efficiency_gate_passed": (
                                    diagonal_efficiency_gate_passed
                                ),
                                "reduction_fraction_gate_passed": (
                                    diagonal_reduction_gate_passed
                                ),
                                "passed": diagonal_gate_passed,
                                "decision": (
                                    "continue_to_schur_factorization"
                                    if diagonal_gate_passed
                                    else "skip_schur_and_use_public_joint"
                                ),
                            }
                        )
                        if not diagonal_gate_passed:
                            stop_reason = "initial_diagonal_gate_rejected"
                            if (
                                restoration_refresh_iteration is not None
                                and outer_iteration
                                >= restoration_refresh_iteration
                            ):
                                assert (
                                    schur_active_set_restoration_diagnostics
                                    is not None
                                )
                                _joint_schur_record_restoration_refresh(
                                    schur_active_set_restoration_diagnostics,
                                    schur_outer_diagnostics,
                                    outer_iteration=outer_iteration,
                                    accepted=False,
                                    trial_count=0,
                                    refresh_stop_reason=stop_reason,
                                )
                            break
                    try:
                        (
                            schur_diagnostics_raw,
                            schur_direction,
                            schur_frame_direction,
                            next_coarse_grid_cache,
                        ) = _joint_schur_diagnostics_from_solver_result(
                            current_frame_result,
                            frame_base=frame_base,
                            num_variables=num_variables,
                            damping=cfg.joint_schur_damping,
                            cg_maxiter=cfg.joint_schur_cg_maxiter,
                            cg_rtol=cfg.joint_schur_cg_rtol,
                            two_level_aggregate_size=(
                                cfg.joint_schur_two_level_aggregate_size
                            ),
                            two_level_cache=coarse_grid_cache,
                            prefer_two_level_direction=adaptive_outer,
                            freeze_coarse_scales=freeze_coarse_scales_this_outer,
                            full_space_block_pcg_probe=(
                                cfg.joint_schur_full_space_block_pcg_probe
                            ),
                        )
                        reused_grid = bool(
                            schur_diagnostics_raw.get("two_level", {}).get(
                                "coarse_grid_reused", False
                            )
                        )
                        if (
                            adaptive_outer
                            and reused_grid
                            and not schur_solve_usable(schur_diagnostics_raw)
                        ):
                            (
                                schur_diagnostics_raw,
                                schur_direction,
                                schur_frame_direction,
                                next_coarse_grid_cache,
                            ) = _joint_schur_diagnostics_from_solver_result(
                                current_frame_result,
                                frame_base=frame_base,
                                num_variables=num_variables,
                                damping=cfg.joint_schur_damping,
                                cg_maxiter=cfg.joint_schur_cg_maxiter,
                                cg_rtol=cfg.joint_schur_cg_rtol,
                                two_level_aggregate_size=(
                                    cfg.joint_schur_two_level_aggregate_size
                                ),
                                two_level_cache=None,
                                prefer_two_level_direction=True,
                                freeze_coarse_scales=freeze_coarse_scales_this_outer,
                                full_space_block_pcg_probe=(
                                    cfg.joint_schur_full_space_block_pcg_probe
                                ),
                            )
                            schur_diagnostics_raw[
                                "stale_coarse_grid_rebuilt_after_cg_failure"
                            ] = True
                        if (
                            adaptive_outer
                            and cfg.joint_schur_two_level_aggregate_size is not None
                            and not schur_solve_usable(schur_diagnostics_raw)
                        ):
                            failed_two_level = {
                                "linear_solver": schur_diagnostics_raw.get(
                                    "linear_solver"
                                ),
                                "cg_info": schur_diagnostics_raw.get("cg_info"),
                                "cg_relative_residual": schur_diagnostics_raw.get(
                                    "cg_relative_residual"
                                ),
                            }
                            (
                                schur_diagnostics_raw,
                                schur_direction,
                                schur_frame_direction,
                                next_coarse_grid_cache,
                            ) = _joint_schur_diagnostics_from_solver_result(
                                current_frame_result,
                                frame_base=frame_base,
                                num_variables=num_variables,
                                damping=cfg.joint_schur_damping,
                                cg_maxiter=cfg.joint_schur_cg_maxiter,
                                cg_rtol=cfg.joint_schur_cg_rtol,
                                two_level_aggregate_size=None,
                                prefer_two_level_direction=False,
                                freeze_coarse_scales=freeze_coarse_scales_this_outer,
                                full_space_block_pcg_probe=(
                                    cfg.joint_schur_full_space_block_pcg_probe
                                ),
                            )
                            schur_diagnostics_raw["two_level_fallback"] = {
                                **failed_two_level,
                                "fallback_linear_solver": (
                                    "local_preconditioned_cg"
                                ),
                            }
                    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                        schur_direction = None
                        schur_frame_direction = None
                        next_coarse_grid_cache = None
                        schur_diagnostics_raw = {
                            "schema": "hierarchical_joint_schur_diagnostics_v1",
                            "linearization": (
                                "scipy_least_squares_result_linearization"
                            ),
                            "solution_mutated": False,
                            "status": "failed",
                            "reason": f"{type(exc).__name__}:{exc}",
                        }
                        if (
                            adaptive_outer
                            and cfg.joint_schur_two_level_aggregate_size is not None
                        ):
                            try:
                                (
                                    schur_diagnostics_raw,
                                    schur_direction,
                                    schur_frame_direction,
                                    next_coarse_grid_cache,
                                ) = _joint_schur_diagnostics_from_solver_result(
                                    current_frame_result,
                                    frame_base=frame_base,
                                    num_variables=num_variables,
                                    damping=cfg.joint_schur_damping,
                                    cg_maxiter=cfg.joint_schur_cg_maxiter,
                                    cg_rtol=cfg.joint_schur_cg_rtol,
                                    two_level_aggregate_size=None,
                                    prefer_two_level_direction=False,
                                    freeze_coarse_scales=freeze_coarse_scales_this_outer,
                                    full_space_block_pcg_probe=(
                                        cfg.joint_schur_full_space_block_pcg_probe
                                    ),
                                )
                                schur_diagnostics_raw["two_level_fallback"] = {
                                    "reason": f"{type(exc).__name__}:{exc}",
                                    "fallback_linear_solver": (
                                        "local_preconditioned_cg"
                                    ),
                                }
                            except (
                                RuntimeError,
                                ValueError,
                                np.linalg.LinAlgError,
                            ):
                                pass
                    if not cfg.joint_schur_step:
                        break
                    if (
                        schur_direction is None
                        or schur_frame_direction is None
                        or not schur_solve_usable(schur_diagnostics_raw)
                    ):
                        stop_reason = (
                            "schur_direction_unavailable_or_cg_not_converged"
                        )
                        if (
                            restoration_refresh_iteration is not None
                            and outer_iteration >= restoration_refresh_iteration
                        ):
                            assert (
                                schur_active_set_restoration_diagnostics
                                is not None
                            )
                            _joint_schur_record_restoration_refresh(
                                schur_active_set_restoration_diagnostics,
                                schur_outer_diagnostics,
                                outer_iteration=outer_iteration,
                                accepted=False,
                                trial_count=0,
                                refresh_stop_reason=stop_reason,
                            )
                        break
                    o0_coarse_row_stop = 7 * len(edges)
                    o0_coarse_before = np.asarray(
                        current_frame_result.fun,
                        dtype=np.float64,
                    )[:o0_coarse_row_stop]
                    o0_coarse_cost_before = float(
                        o0_coarse_before @ o0_coarse_before
                    )
                    o0_coarse_statistics_before = (
                        _coarse_edge_residual_statistics(
                            o0_coarse_before,
                            edges,
                        )
                    )
                    if initial_coarse_cost is None:
                        initial_coarse_cost = o0_coarse_cost_before
                        initial_coarse_statistics = o0_coarse_statistics_before
                    o0_production_frame_direction = schur_frame_direction
                    o0_direction_model_source = "raw_schur_backsubstitution"
                    if (
                        cfg.joint_schur_step_frame_low_frequency_trust_ratio
                        is not None
                    ):
                        o0_production_frame_direction, _ = (
                            _trust_frame_low_frequency_direction(
                                schur_direction,
                                schur_frame_direction,
                                aggregate_size=(
                                    cfg.joint_schur_step_frame_low_frequency_trust_aggregate_size
                                ),
                                maximum_component_block_rms_ratio=(
                                    cfg.joint_schur_step_frame_low_frequency_trust_ratio
                                ),
                            )
                        )
                        o0_direction_model_source = (
                            "recomputed_after_frame_low_frequency_projection"
                        )
                    record_frozen_constrained_step_oracle(
                        "O0_initial",
                        outer_iteration=outer_iteration,
                        linearization=current_frame_result,
                        start=joint_start,
                        coarse_before=o0_coarse_before,
                        coarse_statistics_before=o0_coarse_statistics_before,
                        full_cost_before=float(current_frame_result.cost),
                        coarse_direction=schur_direction,
                        frame_direction=schur_frame_direction,
                        production_frame_direction=o0_production_frame_direction,
                        current_alpha_for_checkpoint=current_alpha,
                        direction_model_source=o0_direction_model_source,
                        stop_reason_at_checkpoint=stop_reason,
                    )
                    (
                        current_efficiency_aggregate_size,
                        current_low_frequency_efficiency,
                    ) = selected_low_frequency_efficiency(
                        schur_diagnostics_raw,
                        metric="euclidean_reduction_per_low_frequency_energy",
                    )
                    (
                        _,
                        current_jacobi_low_frequency_efficiency,
                    ) = selected_low_frequency_efficiency(
                        schur_diagnostics_raw,
                        metric="jacobi_reduction_per_low_frequency_energy",
                    )
                    if outer_iteration == 0:
                        initial_low_frequency_efficiency = (
                            current_low_frequency_efficiency
                        )
                        low_frequency_efficiency_aggregate_size = (
                            current_efficiency_aggregate_size
                        )
                        initial_jacobi_low_frequency_efficiency = (
                            current_jacobi_low_frequency_efficiency
                        )
                    initial_jacobi_efficiency_gate_passed = bool(
                        cfg.joint_schur_step_max_initial_jacobi_low_frequency_efficiency
                        is None
                        or (
                            initial_jacobi_low_frequency_efficiency is not None
                            and initial_jacobi_low_frequency_efficiency
                            <= cfg.joint_schur_step_max_initial_jacobi_low_frequency_efficiency
                        )
                    )
                    if not initial_jacobi_efficiency_gate_passed:
                        schur_outer_diagnostics.append(
                            {
                                "outer_iteration": int(outer_iteration),
                                "freeze_coarse_scales": (
                                    freeze_coarse_scales_this_outer
                                ),
                                "accepted": False,
                                "linearization_gate_passed": (
                                    linearization_gate_passed
                                ),
                                "linearization_leak_inf": linearization_leak_inf,
                                "linearization_leak_limit": (
                                    linearization_leak_limit
                                ),
                                "coarse_graph_squared_residual_before": None,
                                "coarse_edge_statistics_before": None,
                                "full_robust_cost_before": float(
                                    current_frame_result.cost
                                ),
                                "low_frequency_efficiency": (
                                    current_low_frequency_efficiency
                                ),
                                "jacobi_low_frequency_efficiency": (
                                    current_jacobi_low_frequency_efficiency
                                ),
                                "initial_jacobi_efficiency_gate_passed": False,
                                "low_frequency_efficiency_ratio": None,
                                "low_frequency_efficiency_gate_passed": True,
                                "frame_low_frequency_trust": None,
                                "schur_diagnostics": schur_diagnostics_raw,
                                "trials": [],
                            }
                        )
                        stop_reason = (
                            "initial_jacobi_low_frequency_efficiency_above_threshold"
                            if initial_jacobi_low_frequency_efficiency is not None
                            else "initial_jacobi_low_frequency_efficiency_unavailable"
                        )
                        if (
                            restoration_refresh_iteration is not None
                            and outer_iteration >= restoration_refresh_iteration
                        ):
                            assert (
                                schur_active_set_restoration_diagnostics
                                is not None
                            )
                            _joint_schur_record_restoration_refresh(
                                schur_active_set_restoration_diagnostics,
                                schur_outer_diagnostics,
                                outer_iteration=outer_iteration,
                                accepted=False,
                                trial_count=0,
                                refresh_stop_reason=stop_reason,
                            )
                        break
                    low_frequency_efficiency_ratio = (
                        float(
                            current_low_frequency_efficiency
                            / initial_low_frequency_efficiency
                        )
                        if current_low_frequency_efficiency is not None
                        and initial_low_frequency_efficiency is not None
                        and initial_low_frequency_efficiency > 0.0
                        else None
                    )
                    low_frequency_efficiency_gate_passed = bool(
                        cfg.joint_schur_step_min_low_frequency_efficiency_ratio
                        is None
                        or outer_iteration == 0
                        or (
                            low_frequency_efficiency_ratio is not None
                            and low_frequency_efficiency_ratio
                            >= cfg.joint_schur_step_min_low_frequency_efficiency_ratio
                        )
                    )
                    if not low_frequency_efficiency_gate_passed:
                        schur_outer_diagnostics.append(
                            {
                                "outer_iteration": int(outer_iteration),
                                "freeze_coarse_scales": (
                                    freeze_coarse_scales_this_outer
                                ),
                                "accepted": False,
                                "linearization_gate_passed": (
                                    linearization_gate_passed
                                ),
                                "linearization_leak_inf": linearization_leak_inf,
                                "linearization_leak_limit": (
                                    linearization_leak_limit
                                ),
                                "coarse_graph_squared_residual_before": None,
                                "coarse_edge_statistics_before": None,
                                "full_robust_cost_before": float(
                                    current_frame_result.cost
                                ),
                                "low_frequency_efficiency": (
                                    current_low_frequency_efficiency
                                ),
                                "low_frequency_efficiency_ratio": (
                                    low_frequency_efficiency_ratio
                                ),
                                "low_frequency_efficiency_gate_passed": False,
                                "frame_low_frequency_trust": None,
                                "schur_diagnostics": schur_diagnostics_raw,
                                "trials": [],
                            }
                        )
                        stop_reason = (
                            "low_frequency_model_efficiency_ratio_below_threshold"
                        )
                        if (
                            restoration_refresh_iteration is not None
                            and outer_iteration >= restoration_refresh_iteration
                        ):
                            assert (
                                schur_active_set_restoration_diagnostics
                                is not None
                            )
                            _joint_schur_record_restoration_refresh(
                                schur_active_set_restoration_diagnostics,
                                schur_outer_diagnostics,
                                outer_iteration=outer_iteration,
                                accepted=False,
                                trial_count=0,
                                refresh_stop_reason=stop_reason,
                            )
                        break
                    coarse_row_stop = 7 * len(edges)
                    coarse_before = np.asarray(
                        current_frame_result.fun,
                        dtype=np.float64,
                    )[:coarse_row_stop]
                    coarse_cost_before = float(coarse_before @ coarse_before)
                    coarse_statistics_before = _coarse_edge_residual_statistics(
                        coarse_before, edges
                    )
                    full_cost_before = float(current_frame_result.cost)
                    if initial_coarse_cost is None:
                        initial_coarse_cost = coarse_cost_before
                        initial_coarse_statistics = coarse_statistics_before
                    outer_trials = []
                    outer_accepted = False
                    trusted_frame_direction = schur_frame_direction
                    frame_low_frequency_trust = None
                    full_directional_gradient = float(
                        schur_diagnostics_raw["full_directional_gradient"]
                    )
                    full_directional_curvature = float(
                        schur_diagnostics_raw["full_directional_curvature"]
                    )
                    direction_model_source = "raw_schur_backsubstitution"
                    if (
                        cfg.joint_schur_step_frame_low_frequency_trust_ratio
                        is not None
                    ):
                        (
                            trusted_frame_direction,
                            frame_low_frequency_trust,
                        ) = _trust_frame_low_frequency_direction(
                            schur_direction,
                            schur_frame_direction,
                            aggregate_size=(
                                cfg.joint_schur_step_frame_low_frequency_trust_aggregate_size
                            ),
                            maximum_component_block_rms_ratio=(
                                cfg.joint_schur_step_frame_low_frequency_trust_ratio
                            ),
                        )
                        (
                            full_directional_gradient,
                            full_directional_curvature,
                        ) = full_direction_quadratic_model(
                            current_frame_result,
                            schur_direction,
                            trusted_frame_direction,
                        )
                        direction_model_source = (
                            "recomputed_after_frame_low_frequency_projection"
                        )
                    full_direction = np.concatenate(
                        (schur_direction, trusted_frame_direction)
                    )
                    for backtrack in range(cfg.joint_schur_step_backtracks):
                        alpha = float(current_alpha * (0.5**backtrack))
                        proposed = joint_start + alpha * full_direction
                        candidate = np.clip(proposed, lower, upper)
                        clipped_coarse_variable_count = int(
                            np.count_nonzero(
                                ~np.isclose(
                                    candidate[:frame_base],
                                    proposed[:frame_base],
                                    rtol=0.0,
                                    atol=1e-12,
                                )
                            )
                        )
                        clipped_frame_variable_count = int(
                            np.count_nonzero(
                                ~np.isclose(
                                    candidate[frame_base:],
                                    proposed[frame_base:],
                                    rtol=0.0,
                                    atol=1e-12,
                                )
                            )
                        )
                        candidate_residual = residual(candidate)
                        if cfg.joint_schur_step_final_solver:
                            schur_trial_residual_evaluations += 1
                        coarse_after = candidate_residual[:coarse_row_stop]
                        coarse_cost_after = float(coarse_after @ coarse_after)
                        coarse_statistics_after = _coarse_edge_residual_statistics(
                            coarse_after, edges
                        )
                        coarse_statistics_ratios = _coarse_edge_statistics_ratios(
                            coarse_statistics_before, coarse_statistics_after
                        )
                        assert initial_coarse_statistics is not None
                        cumulative_coarse_statistics_ratios = (
                            _coarse_edge_statistics_ratios(
                                initial_coarse_statistics,
                                coarse_statistics_after,
                            )
                        )
                        coarse_cost_ratio = _nonnegative_cost_ratio(
                            coarse_cost_after, coarse_cost_before
                        )
                        assert initial_coarse_cost is not None
                        cumulative_coarse_cost_ratio = _nonnegative_cost_ratio(
                            coarse_cost_after, initial_coarse_cost
                        )
                        predicted_full_reduction = float(
                            -(
                                alpha
                                * full_directional_gradient
                                + 0.5
                                * alpha**2
                                * full_directional_curvature
                            )
                        )
                        aggregate_gate_passed = bool(
                            coarse_cost_ratio
                            <= cfg.joint_schur_step_max_graph_cost_ratio
                        )
                        cumulative_aggregate_gate_passed = bool(
                            cumulative_coarse_cost_ratio
                            <= cfg.joint_schur_step_max_graph_cost_ratio
                        )
                        family_gate_checks = (
                            (
                                cfg.joint_schur_step_max_adjacent_p95_ratio,
                                coarse_statistics_ratios["adjacent"]["p95"],
                            ),
                            (
                                effective_loop_p95_ratio,
                                coarse_statistics_ratios["loop"]["p95"],
                            ),
                            (
                                effective_loop_max_ratio,
                                coarse_statistics_ratios["loop"]["max"],
                            ),
                        )
                        edge_family_gate_passed = all(
                            threshold is None
                            or observed is None
                            or observed <= threshold
                            for threshold, observed in family_gate_checks
                        )
                        cumulative_family_gate_checks = (
                            (
                                cfg.joint_schur_step_max_adjacent_p95_ratio,
                                cumulative_coarse_statistics_ratios["adjacent"][
                                    "p95"
                                ],
                            ),
                            (
                                effective_loop_p95_ratio,
                                cumulative_coarse_statistics_ratios["loop"]["p95"],
                            ),
                            (
                                effective_loop_max_ratio,
                                cumulative_coarse_statistics_ratios["loop"]["max"],
                            ),
                        )
                        cumulative_edge_family_gate_passed = all(
                            threshold is None
                            or observed is None
                            or observed <= threshold
                            for threshold, observed in cumulative_family_gate_checks
                        )
                        minimum_alpha_gate_passed = bool(
                            cfg.joint_schur_step_min_accepted_alpha is None
                            or alpha >= cfg.joint_schur_step_min_accepted_alpha
                        )
                        direction_model_gate_passed = bool(
                            clipped_coarse_variable_count == 0
                            and clipped_frame_variable_count == 0
                            and predicted_full_reduction > 0.0
                            and minimum_alpha_gate_passed
                        )
                        full_cost_after = _scipy_compatible_robust_cost(
                            candidate_residual,
                            loss=cfg.robust_loss,
                            f_scale=cfg.f_scale,
                        )
                        actual_full_reduction = full_cost_before - full_cost_after
                        reduction_ratio = (
                            float(actual_full_reduction / predicted_full_reduction)
                            if predicted_full_reduction > 0.0
                            else None
                        )
                        robust_cost_nonincreasing = bool(
                            full_cost_after
                            <= full_cost_before
                            + max(1e-12, 1e-10 * abs(full_cost_before))
                        )
                        hierarchy_gate_passed = bool(
                            aggregate_gate_passed
                            and cumulative_aggregate_gate_passed
                            and cumulative_edge_family_gate_passed
                            and direction_model_gate_passed
                        )
                        accepted = bool(
                            hierarchy_gate_passed
                            and np.all(np.isfinite(candidate))
                            and robust_cost_nonincreasing
                            and reduction_ratio is not None
                            and reduction_ratio >= 0.25
                        )
                        trial = {
                            "outer_iteration": int(outer_iteration),
                            "freeze_coarse_scales": freeze_coarse_scales_this_outer,
                            "backtrack": int(backtrack),
                            "alpha": alpha,
                            "coarse_step_l2": float(
                                np.linalg.norm(
                                    candidate[:frame_base]
                                    - joint_start[:frame_base]
                                )
                            ),
                            "frame_step_l2": float(
                                np.linalg.norm(
                                    candidate[frame_base:]
                                    - joint_start[frame_base:]
                                )
                            ),
                            "frame_step_block_rms": float(
                                np.linalg.norm(
                                    candidate[frame_base:]
                                    - joint_start[frame_base:]
                                )
                                / np.sqrt((num_variables - frame_base) // 6)
                            ),
                            "full_step_l2": float(
                                np.linalg.norm(candidate - joint_start)
                            ),
                            "coarse_graph_squared_residual": coarse_cost_after,
                            "coarse_graph_cost_ratio": float(coarse_cost_ratio),
                            "cumulative_coarse_graph_cost_ratio": float(
                                cumulative_coarse_cost_ratio
                            ),
                            "full_quadratic_predicted_reduction": (
                                predicted_full_reduction
                            ),
                            "full_directional_gradient": (
                                full_directional_gradient
                            ),
                            "full_directional_curvature": (
                                full_directional_curvature
                            ),
                            "direction_model_source": direction_model_source,
                            "full_quadratic_predicted_reduction_fraction": (
                                float(predicted_full_reduction / full_cost_before)
                                if full_cost_before > 0.0
                                else None
                            ),
                            "coarse_edge_statistics": coarse_statistics_after,
                            "coarse_edge_statistics_ratios": coarse_statistics_ratios,
                            "cumulative_coarse_edge_statistics_ratios": (
                                cumulative_coarse_statistics_ratios
                            ),
                            "aggregate_graph_gate_passed": aggregate_gate_passed,
                            "edge_family_gate_passed": edge_family_gate_passed,
                            "edge_family_gate_role": "advisory_step_diagnostic",
                            "cumulative_aggregate_graph_gate_passed": (
                                cumulative_aggregate_gate_passed
                            ),
                            "cumulative_edge_family_gate_passed": (
                                cumulative_edge_family_gate_passed
                            ),
                            "direction_model_gate_passed": (
                                direction_model_gate_passed
                            ),
                            "minimum_alpha_gate_passed": (
                                minimum_alpha_gate_passed
                            ),
                            "coarse_graph_gate_passed": bool(
                                hierarchy_gate_passed
                            ),
                            "clipped_coarse_variable_count": (
                                clipped_coarse_variable_count
                            ),
                            "clipped_frame_variable_count": (
                                clipped_frame_variable_count
                            ),
                            "active_coarse_bound_count": int(
                                np.count_nonzero(
                                    np.isclose(
                                        candidate[:frame_base],
                                        lower[:frame_base],
                                        rtol=0.0,
                                        atol=1e-12,
                                    )
                                    | np.isclose(
                                        candidate[:frame_base],
                                        upper[:frame_base],
                                        rtol=0.0,
                                        atol=1e-12,
                                    )
                                )
                            ),
                            "full_robust_cost": full_cost_after,
                            "full_robust_actual_reduction": (
                                actual_full_reduction
                            ),
                            "full_reduction_ratio": reduction_ratio,
                            "full_robust_cost_nonincreasing": (
                                robust_cost_nonincreasing
                            ),
                            "accepted": accepted,
                        }
                        outer_trials.append(trial)
                        schur_step_trials.append(trial)
                        if accepted:
                            outer_accepted = True
                            accepted_outer_iterations += 1
                            if cfg.joint_schur_step_final_solver:
                                record_schur_final_accepted_step(
                                    candidate=candidate,
                                    candidate_residual=candidate_residual,
                                    full_cost_after=full_cost_after,
                                    step_start=joint_start,
                                    full_cost_before=full_cost_before,
                                    actual_full_reduction=actual_full_reduction,
                                    reduction_ratio=reduction_ratio,
                                    alpha=alpha,
                                )
                            joint_start = candidate
                            accepted_warm_cost = full_cost_after
                            coarse_grid_cache = next_coarse_grid_cache
                            current_alpha = (
                                min(1.0, 2.0 * alpha)
                                if adaptive_outer
                                and reduction_ratio is not None
                                and reduction_ratio >= 0.75
                                else alpha
                            )
                            if (
                                outer_iteration + 1 < requested_outer_iterations
                                and not (
                                    restoration_refresh_iteration is not None
                                    and outer_iteration
                                    >= restoration_refresh_iteration
                                )
                            ):
                                current_frame_result = linearize_full(joint_start)
                            break
                    if not outer_accepted:
                        record_frozen_constrained_step_oracle(
                            "first_production_no_accepted_alpha",
                            outer_iteration=outer_iteration,
                            linearization=current_frame_result,
                            start=joint_start,
                            coarse_before=coarse_before,
                            coarse_statistics_before=coarse_statistics_before,
                            full_cost_before=full_cost_before,
                            coarse_direction=schur_direction,
                            frame_direction=schur_frame_direction,
                            production_frame_direction=trusted_frame_direction,
                            current_alpha_for_checkpoint=current_alpha,
                            direction_model_source=direction_model_source,
                            stop_reason_at_checkpoint=(
                                "no_acceptable_backtracking_step"
                            ),
                        )
                        if (
                            cfg.joint_schur_one_active_set_restoration_at_first_reject
                            and not schur_active_set_restoration_applied
                        ):
                            (
                                restoration_applied_this_outer,
                                restoration_candidate,
                                restoration_linearization,
                                restoration_next_alpha,
                            ) = (
                                try_one_active_set_restoration_from_first_reject(
                                    outer_iteration=outer_iteration,
                                    start=joint_start,
                                    full_cost_before=full_cost_before,
                                    current_alpha_for_refresh=current_alpha,
                                )
                            )
                            if restoration_applied_this_outer:
                                joint_start = restoration_candidate
                                accepted_warm_cost = float(
                                    schur_final_accepted_cost
                                )
                                current_frame_result = restoration_linearization
                                current_alpha = restoration_next_alpha
                                coarse_grid_cache = None
                                restoration_refresh_iteration = outer_iteration + 1
                    schur_outer_diagnostics.append(
                        {
                            "outer_iteration": int(outer_iteration),
                            "freeze_coarse_scales": freeze_coarse_scales_this_outer,
                            "accepted": outer_accepted,
                            "linearization_gate_passed": (
                                linearization_gate_passed
                            ),
                            "linearization_leak_inf": linearization_leak_inf,
                            "linearization_leak_limit": (
                                linearization_leak_limit
                            ),
                            "coarse_graph_squared_residual_before": (
                                coarse_cost_before
                            ),
                            "coarse_edge_statistics_before": (
                                coarse_statistics_before
                            ),
                            "full_robust_cost_before": full_cost_before,
                            "low_frequency_efficiency": (
                                current_low_frequency_efficiency
                            ),
                            "jacobi_low_frequency_efficiency": (
                                current_jacobi_low_frequency_efficiency
                            ),
                            "initial_jacobi_efficiency_gate_passed": (
                                initial_jacobi_efficiency_gate_passed
                            ),
                            "low_frequency_efficiency_ratio": (
                                low_frequency_efficiency_ratio
                            ),
                            "low_frequency_efficiency_gate_passed": (
                                low_frequency_efficiency_gate_passed
                            ),
                            "full_directional_gradient": (
                                full_directional_gradient
                            ),
                            "full_directional_curvature": (
                                full_directional_curvature
                            ),
                            "direction_model_source": direction_model_source,
                            "frame_low_frequency_trust": (
                                frame_low_frequency_trust
                            ),
                            "schur_diagnostics": schur_diagnostics_raw,
                            "trials": outer_trials,
                            "one_active_set_restoration": (
                                schur_active_set_restoration_diagnostics
                                if (
                                    cfg.joint_schur_one_active_set_restoration_at_first_reject
                                    and not outer_accepted
                                    and schur_active_set_restoration_diagnostics
                                    is not None
                                    and schur_active_set_restoration_diagnostics.get(
                                        "trigger_outer_iteration"
                                    )
                                    == int(outer_iteration)
                                )
                                else None
                            ),
                        }
                    )
                    if (
                        restoration_refresh_iteration is not None
                        and outer_iteration >= restoration_refresh_iteration
                    ):
                        assert (
                            schur_active_set_restoration_diagnostics
                            is not None
                        )
                        refresh_stop_reason = (
                            "accepted_standard_schur_refresh"
                            if outer_accepted
                            else "no_acceptable_backtracking_step"
                        )
                        _joint_schur_record_restoration_refresh(
                            schur_active_set_restoration_diagnostics,
                            schur_outer_diagnostics,
                            outer_iteration=outer_iteration,
                            accepted=outer_accepted,
                            trial_count=len(outer_trials),
                            refresh_stop_reason=refresh_stop_reason,
                        )
                        stop_reason = (
                            "one_active_set_restoration_refresh_completed"
                            if outer_accepted
                            else "one_active_set_restoration_refresh_no_accepted_step"
                        )
                        break
                    if not outer_accepted:
                        if (
                            schur_active_set_restoration_applied
                            and restoration_refresh_iteration
                            == outer_iteration + 1
                        ):
                            stop_reason = "active_set_restoration_applied"
                            continue
                        stop_reason = "no_acceptable_backtracking_step"
                        break
                if (
                    cfg.joint_schur_one_active_set_restoration_at_first_reject
                    and schur_active_set_restoration_diagnostics is None
                ):
                    schur_active_set_restoration_diagnostics = {
                        "schema": (
                            "hierarchical_joint_schur_one_active_set_"
                            "restoration_at_first_reject_v1"
                        ),
                        "enabled": True,
                        "trigger_checkpoint": (
                            "first_production_no_accepted_alpha"
                        ),
                        "solution_mutated": False,
                        "applied": False,
                        "reason": "first_reject_checkpoint_not_reached",
                        "oracle_extra_residual_evaluations_at_trigger": int(
                            schur_oracle_extra_residual_evaluations
                        ),
                        "oracle_extra_jacobian_evaluations_at_trigger": int(
                            schur_oracle_extra_jacobian_evaluations
                        ),
                        "restoration_function_evaluations": 0,
                        "fresh_relinearization": {
                            "performed": False,
                            "nfev": 0,
                            "njev": 0,
                        },
                        "refresh_performed": False,
                        "refresh_accepted": None,
                        "refresh_stop_reason": None,
                    }
                if cfg.joint_schur_step:
                    schur_step_diagnostics = {
                        "schema": "hierarchical_joint_full_schur_step_v2",
                        "step_model": (
                            "coarse_schur_solve_plus_frame_backsubstitution_"
                            "with_low_frequency_trust"
                            if cfg.joint_schur_step_frame_low_frequency_trust_ratio
                            is not None
                            else "coarse_schur_solve_plus_frame_backsubstitution"
                        ),
                        "trial_objective": "full_robust_joint_objective",
                        "topology_filter": (
                            "step_and_cumulative_aggregate_plus_cumulative_edge_family"
                        ),
                        "accepted": bool(
                            accepted_outer_iterations > 0
                            or schur_active_set_restoration_applied
                        ),
                        "accepted_outer_iterations": int(
                            accepted_outer_iterations
                        ),
                        "requested_outer_iterations": int(
                            requested_outer_iterations
                        ),
                        "adaptive_trust_region": adaptive_outer,
                        "minimum_reduction_ratio": 0.25,
                        "minimum_accepted_alpha": (
                            cfg.joint_schur_step_min_accepted_alpha
                        ),
                        "frame_low_frequency_trust_ratio": (
                            cfg.joint_schur_step_frame_low_frequency_trust_ratio
                        ),
                        "frame_low_frequency_trust_aggregate_size": int(
                            cfg.joint_schur_step_frame_low_frequency_trust_aggregate_size
                        ),
                        "freeze_coarse_scales": bool(
                            cfg.joint_schur_step_freeze_coarse_scales
                        ),
                        "minimum_low_frequency_efficiency_ratio": (
                            cfg.joint_schur_step_min_low_frequency_efficiency_ratio
                        ),
                        "maximum_initial_jacobi_low_frequency_efficiency": (
                            cfg.joint_schur_step_max_initial_jacobi_low_frequency_efficiency
                        ),
                        "initial_diagonal_low_frequency_gate": (
                            initial_diagonal_low_frequency_gate
                        ),
                        "initial_jacobi_low_frequency_efficiency": (
                            initial_jacobi_low_frequency_efficiency
                        ),
                        "initial_low_frequency_efficiency": (
                            initial_low_frequency_efficiency
                        ),
                        "low_frequency_efficiency_aggregate_size": (
                            low_frequency_efficiency_aggregate_size
                        ),
                        "coarse_graph_squared_residual_before": (
                            initial_coarse_cost
                        ),
                        "coarse_edge_statistics_before": (
                            initial_coarse_statistics
                        ),
                        "full_robust_cost_before": initial_full_cost,
                        "max_coarse_graph_cost_ratio": float(
                            cfg.joint_schur_step_max_graph_cost_ratio
                        ),
                        "max_adjacent_p95_ratio": (
                            cfg.joint_schur_step_max_adjacent_p95_ratio
                        ),
                        "max_loop_p95_ratio": (
                            effective_loop_p95_ratio
                        ),
                        "max_loop_max_ratio": (
                            effective_loop_max_ratio
                        ),
                        "stop_reason": stop_reason,
                        "outer_iterations": schur_outer_diagnostics,
                        "trials": schur_step_trials,
                    }
                    if cfg.joint_schur_one_active_set_restoration_at_first_reject:
                        schur_step_diagnostics[
                            "one_active_set_restoration_at_first_reject"
                        ] = schur_active_set_restoration_diagnostics
                    if cfg.joint_schur_frozen_constrained_step_oracle:
                        oracle = ensure_frozen_step_oracle()
                        if not oracle["checkpoints"]:
                            oracle["checkpoints"].append(
                                {
                                    "checkpoint": "O0_initial",
                                    "outer_iteration": None,
                                    "valid": False,
                                    "reason": "schur_direction_not_constructed",
                                    "solution_mutated": False,
                                }
                            )
                        oracle["oracle_extra_residual_evaluations"] = int(
                            schur_oracle_extra_residual_evaluations
                        )
                        oracle["oracle_extra_jacobian_evaluations"] = int(
                            schur_oracle_extra_jacobian_evaluations
                        )
                        schur_step_diagnostics[
                            "frozen_constrained_step_oracle"
                        ] = oracle
                    if cfg.joint_schur_step_final_solver:
                        if schur_final_accepted_x is not None:
                            final_candidate_x = schur_final_accepted_x
                            assert schur_final_accepted_fun is not None
                            assert schur_final_accepted_cost is not None
                            final_candidate_fun = schur_final_accepted_fun
                            final_candidate_cost = schur_final_accepted_cost
                        else:
                            final_candidate_x = np.asarray(
                                joint_start, dtype=np.float64
                            ).copy()
                            if current_frame_result is not None and np.array_equal(
                                current_frame_result.x, final_candidate_x
                            ):
                                final_candidate_fun = np.asarray(
                                    current_frame_result.fun, dtype=np.float64
                                ).copy()
                                final_candidate_cost = float(
                                    current_frame_result.cost
                                )
                            else:
                                final_candidate_fun = residual(final_candidate_x)
                                final_candidate_cost = _scipy_compatible_robust_cost(
                                    final_candidate_fun,
                                    loss=cfg.robust_loss,
                                    f_scale=cfg.f_scale,
                                )
                        (
                            schur_final_solver_result,
                            schur_final_solver_diagnostics,
                        ) = build_schur_final_solver_result(
                            final_candidate_x,
                            final_candidate_fun,
                            final_candidate_cost,
                            accepted=bool(
                                accepted_outer_iterations > 0
                                or schur_active_set_restoration_applied
                            ),
                            stop_reason=stop_reason,
                        )
                        if (
                            cfg.joint_schur_one_active_set_restoration_at_first_reject
                            and schur_active_set_restoration_diagnostics
                            is not None
                        ):
                            schur_final_solver_diagnostics[
                                "one_active_set_restoration_at_first_reject"
                            ] = schur_active_set_restoration_diagnostics
                        if (
                            cfg.joint_schur_one_active_set_restoration_at_first_reject
                            and schur_active_set_restoration_applied
                        ):
                            schur_final_solver_diagnostics["mode"] = (
                                "joint_schur_final_solver_one_active_set_"
                                "restoration_canary"
                                if schur_final_solver_diagnostics["success"]
                                else "joint_schur_final_solver_one_active_set_"
                                "restoration_canary_not_converged"
                            )
                        result = schur_final_solver_result
                        schur_step_diagnostics["final_polish_mode"] = "none"
                        if schur_active_set_restoration_applied:
                            schur_step_diagnostics["selected_solution"] = (
                                "schur_final_solver_one_active_set_"
                                "restoration_canary"
                                if result.success
                                else "schur_final_solver_one_active_set_"
                                "restoration_canary_not_converged"
                            )
                        else:
                            schur_step_diagnostics["selected_solution"] = (
                                "schur_final_solver"
                                if result.success
                                else "schur_final_solver_not_converged"
                            )
                        schur_step_diagnostics["final_solver"] = (
                            schur_final_solver_diagnostics
                        )
                    elif accepted_outer_iterations == 0:
                        joint_start = (
                            public_pilot_result.x.copy()
                            if public_pilot_result is not None
                            else initial_vector
                        )
                        schur_step_diagnostics["fallback_start"] = (
                            "public_pilot_continuation"
                            if public_pilot_result is not None
                            else "public_joint_initialization"
                        )
                        if (
                            public_pilot_result is None
                            and hasattr(current_frame_result, "raw_fun")
                            and np.array_equal(
                                current_frame_result.x, joint_start
                            )
                        ):
                            final_solve_initial_cache = current_frame_result
                else:
                    joint_start = initial_vector
        if (
            result is None
            and joint_subspace_diagnostics is not None
            and joint_subspace_diagnostics["accepted"]
            and cfg.joint_subspace_conditional_frame_polish
            and (
                cfg.joint_subspace_stage_b_rotation_trust_ratio is not None
                or cfg.joint_subspace_stage_b_p64_gauge_fix
                or cfg.joint_subspace_stage_b_p64_sparse_basis
            )
        ):
            initial_fine_statistics = joint_subspace_diagnostics[
                "initial_local_residual_statistics"
            ]["families"]["fine"]
            initial_fine_rotation_p95 = initial_fine_statistics[
                "rotation_radians"
            ]["p95"]
            initial_fine_translation_p95 = initial_fine_statistics[
                "translation_scene_units"
            ]["p95"]
            if (
                initial_fine_rotation_p95 is None
                or initial_fine_translation_p95 is None
                or not np.isfinite(initial_fine_rotation_p95)
                or not np.isfinite(initial_fine_translation_p95)
            ):
                joint_subspace_diagnostics.update(
                    {
                        "accepted": False,
                        "decision": "reject_stage_b_trust_scale_unavailable",
                        "fallback": "public_joint_initialization",
                        "stage_b_frame_trust": {
                            "schema": (
                                "hierarchical_joint_stage_b_frame_trust_v4"
                                if cfg.joint_subspace_stage_b_p64_sparse_basis
                                else
                                "hierarchical_joint_stage_b_frame_trust_v3"
                                if cfg.joint_subspace_stage_b_p64_gauge_fix
                                else "hierarchical_joint_stage_b_frame_trust_v2"
                            ),
                            "enabled": True,
                            "valid": False,
                            "reason": "initial_fine_p95_unavailable",
                        },
                    }
                )
                joint_start = initial_vector
            else:
                rotation_scale = max(
                    float(initial_fine_rotation_p95), 1e-4
                )
                translation_scale = max(
                    float(initial_fine_translation_p95), 1e-6
                )
                if cfg.joint_subspace_stage_b_p64_sparse_basis:
                    joint_subspace_stage_b_trust = {
                        "schema": "hierarchical_joint_stage_b_frame_trust_v4",
                        "enabled": True,
                        "valid": True,
                        "mode": "p64_sparse_exact_hierarchical_basis",
                        "parameterization": "delta_frame=[P64,N_sparse]_c",
                        "role": (
                            "unbiased_nonredundant_coordinate_transform_"
                            "not_regularization"
                        ),
                        "interpretation": (
                            "stage_b_optimizes_all_frame_deltas_in_a_square_"
                            "sparse_hierarchical_basis_without_extra_residuals"
                        ),
                        "unbiased_physical_optimum": True,
                        "initial_fine_rotation_p95": float(
                            initial_fine_rotation_p95
                        ),
                        "initial_fine_translation_p95": float(
                            initial_fine_translation_p95
                        ),
                    }
                elif cfg.joint_subspace_stage_b_p64_gauge_fix:
                    joint_subspace_stage_b_trust = {
                        "schema": "hierarchical_joint_stage_b_frame_trust_v3",
                        "enabled": True,
                        "valid": True,
                        "mode": "p64_unbiased_low_frequency_gauge_fix",
                        "parameterization": "delta_frame=P64_z+q",
                        "role": (
                            "unbiased_gauge_fixing_not_high_frequency_regularization"
                        ),
                        "interpretation": (
                            "stage_b_chooses_a_p64_decomposition_gauge_"
                            "without_penalizing_any_physical_frame_delta"
                        ),
                        "unbiased_physical_optimum": True,
                        "initial_fine_rotation_p95": float(
                            initial_fine_rotation_p95
                        ),
                        "initial_fine_translation_p95": float(
                            initial_fine_translation_p95
                        ),
                        "rotation_scale_floor": 1e-4,
                        "translation_scale_floor": 1e-6,
                        "rotation_scale": rotation_scale,
                        "translation_scale": translation_scale,
                    }
                else:
                    rotation_trust_radius = float(
                        cfg.joint_subspace_stage_b_rotation_trust_ratio
                        * rotation_scale
                    )
                    translation_trust_radius = float(
                        cfg.joint_subspace_stage_b_translation_trust_ratio
                        * translation_scale
                    )
                    joint_subspace_stage_b_trust = {
                        "schema": "hierarchical_joint_stage_b_frame_trust_v2",
                        "enabled": True,
                        "valid": True,
                        "mode": "p64_direct_z_plus_q_proximal",
                        "parameterization": "delta_frame=P64_z+q",
                        "interpretation": (
                            "stage_b_penalizes_only_the_frame_correction_"
                            "component_outside_the_p64_low_frequency_column_"
                            "space"
                        ),
                        "exact_p64_complement": True,
                        "rotation_trust_ratio": float(
                            cfg.joint_subspace_stage_b_rotation_trust_ratio
                        ),
                        "translation_trust_ratio": float(
                            cfg.joint_subspace_stage_b_translation_trust_ratio
                        ),
                        "initial_fine_rotation_p95": float(
                            initial_fine_rotation_p95
                        ),
                        "initial_fine_translation_p95": float(
                            initial_fine_translation_p95
                        ),
                        "rotation_scale_floor": 1e-4,
                        "translation_scale_floor": 1e-6,
                        "rotation_trust_radius": rotation_trust_radius,
                        "translation_trust_radius": translation_trust_radius,
                    }
                joint_subspace_diagnostics["stage_b_frame_trust"] = (
                    joint_subspace_stage_b_trust
                )
        if result is None:
            if (
                joint_subspace_diagnostics is not None
                and joint_subspace_diagnostics["accepted"]
                and cfg.joint_subspace_conditional_frame_polish
            ):
                if joint_subspace_stage_b_trust is None:
                    result = relax_frames(
                        joint_start,
                        max_nfev=int(final_solve_kwargs["max_nfev"]),
                    )
                else:
                    if (
                        joint_subspace_stage_b_trust["mode"]
                        == "p64_sparse_exact_hierarchical_basis"
                    ):
                        if active_analytic_jacobian is None:
                            result = relax_frames(
                                joint_start,
                                max_nfev=int(final_solve_kwargs["max_nfev"]),
                            )
                            stage_b_diagnostics = {
                                "schema": (
                                    "hierarchical_joint_stage_b_frame_trust_v4"
                                ),
                                "enabled": True,
                                "valid": False,
                                "reason": "analytic_jacobian_unavailable",
                                "mode": "fixed_coarse_frame_polish_fallback",
                                "requested_mode": (
                                    "p64_sparse_exact_basis_outer_cycle_coarse_backtracking"
                                    if cfg.joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking
                                    else "p64_sparse_exact_basis_outer_cycle"
                                    if cfg.joint_subspace_stage_b_p64_sparse_basis_outer_cycle
                                    else "p64_sparse_exact_hierarchical_basis"
                                ),
                                "fallback_mode": (
                                    "full_frame_variables_with_coarse_fixed"
                                ),
                                "fallback": "fixed_coarse_relax_frames",
                                "outer_cycle_enabled": bool(
                                    cfg.joint_subspace_stage_b_p64_sparse_basis_outer_cycle
                                ),
                                "outer_cycle_selected_stage": (
                                    "analytic_jacobian_unavailable_fallback"
                                    if cfg.joint_subspace_stage_b_p64_sparse_basis_outer_cycle
                                    else None
                                ),
                                "selected_sparse_stage": None,
                            }
                        elif (
                            cfg.joint_subspace_stage_b_p64_sparse_basis_outer_cycle
                        ):
                            (
                                result,
                                stage_b_diagnostics,
                                stage_b_discarded_results,
                            ) = relax_sparse_basis_outer_cycle(
                                joint_start,
                                max_nfev=int(final_solve_kwargs["max_nfev"]),
                            )
                            discarded_final_joint_results.extend(
                                stage_b_discarded_results
                            )
                            outer_cycle_diagnostics = stage_b_diagnostics
                            raw_outer_selected_stage = (
                                outer_cycle_diagnostics.get("selected_stage")
                            )
                            selected_sparse_stage = (
                                raw_outer_selected_stage
                                if raw_outer_selected_stage in {"B1", "B2"}
                                else None
                            )
                            selected_sparse_diagnostics = (
                                outer_cycle_diagnostics["stages"]
                                .get(selected_sparse_stage, {})
                                .get("sparse_basis", {})
                                if selected_sparse_stage is not None
                                else {}
                            )
                            stage_b_diagnostics = {
                                **selected_sparse_diagnostics,
                                "outer_cycle_enabled": True,
                                "outer_cycle_selected_stage": (
                                    raw_outer_selected_stage
                                ),
                                "selected_sparse_stage": selected_sparse_stage,
                                "outer_cycle": outer_cycle_diagnostics,
                            }
                            if (
                                cfg.joint_subspace_stage_b_p64_sparse_basis_outer_cycle_coarse_backtracking
                            ):
                                stage_b_diagnostics.update(
                                    {
                                        "outer_cycle_coarse_backtracking_enabled": True,
                                        "outer_cycle_coarse_backtracking_selected": bool(
                                            outer_cycle_diagnostics.get(
                                                "coarse_backtracking_selected",
                                                False,
                                            )
                                        ),
                                        "outer_cycle_coarse_backtracking_selected_alpha": (
                                            outer_cycle_diagnostics.get(
                                                "coarse_backtracking_selected_alpha"
                                            )
                                        ),
                                    }
                                )
                        else:
                            result, stage_b_diagnostics = (
                                relax_frames_p64_sparse_basis(
                                    joint_start,
                                    max_nfev=int(
                                        final_solve_kwargs["max_nfev"]
                                    ),
                                )
                            )
                    elif (
                        joint_subspace_stage_b_trust["mode"]
                        == "p64_unbiased_low_frequency_gauge_fix"
                    ):
                        result, stage_b_diagnostics = (
                            relax_frames_p64_gauge_fix(
                                joint_start,
                                max_nfev=int(final_solve_kwargs["max_nfev"]),
                                rotation_scale=joint_subspace_stage_b_trust[
                                    "rotation_scale"
                                ],
                                translation_scale=(
                                    joint_subspace_stage_b_trust[
                                        "translation_scale"
                                    ]
                                ),
                            )
                        )
                    else:
                        result, stage_b_diagnostics = (
                            relax_frames_p64_proximal(
                                joint_start,
                                max_nfev=int(final_solve_kwargs["max_nfev"]),
                                rotation_trust_radius=(
                                    joint_subspace_stage_b_trust[
                                        "rotation_trust_radius"
                                    ]
                                ),
                                translation_trust_radius=(
                                    joint_subspace_stage_b_trust[
                                        "translation_trust_radius"
                                    ]
                                ),
                            )
                        )
                    joint_subspace_stage_b_trust.update(
                        stage_b_diagnostics
                    )
                    stage_b_cost_before = float(
                        joint_subspace_diagnostics[
                            "candidate_full_robust_cost"
                        ]
                    )
                    joint_subspace_stage_b_trust.update(
                        {
                            "stage_b_physical_cost_before": (
                                stage_b_cost_before
                            ),
                            "stage_b_augmented_cost_before": (
                                stage_b_cost_before
                            ),
                        }
                    )
                frame_polish_step = np.asarray(
                    result.x[frame_base:] - joint_start[frame_base:],
                    dtype=np.float64,
                )
                joint_subspace_frame_polish_mode_energy = (
                    _sequential_pose_mode_energy_diagnostics(
                        frame_polish_step,
                        block_dimension=6,
                        aggregate_sizes=(
                            cfg.joint_subspace_control_stride,
                        ),
                        component_groups=(
                            ("rotation", 0, 3),
                            ("translation", 3, 6),
                        ),
                    )
                )
                stage_b_mode = (
                    None
                    if joint_subspace_stage_b_trust is None
                    else joint_subspace_stage_b_trust.get("mode")
                )
                stage_b_valid = bool(
                    joint_subspace_stage_b_trust is not None
                    and joint_subspace_stage_b_trust.get("valid", True)
                )
                outer_cycle_selected_stage = (
                    None
                    if joint_subspace_stage_b_trust is None
                    else joint_subspace_stage_b_trust.get(
                        "outer_cycle_selected_stage"
                    )
                )
                outer_cycle_coarse_backtracking_selected = bool(
                    joint_subspace_stage_b_trust is not None
                    and joint_subspace_stage_b_trust.get(
                        "outer_cycle_coarse_backtracking_selected", False
                    )
                )
                sparse_outer_cycle_active = bool(
                    stage_b_valid
                    and stage_b_mode == "p64_sparse_exact_hierarchical_basis"
                    and cfg.joint_subspace_stage_b_p64_sparse_basis_outer_cycle
                )
                joint_subspace_diagnostics["final_polish_mode"] = (
                    "p64_sparse_exact_basis_outer_cycle_coarse_backtracking_selected_B2"
                    if sparse_outer_cycle_active
                    and outer_cycle_selected_stage == "B2"
                    and outer_cycle_coarse_backtracking_selected
                    else "p64_sparse_exact_basis_outer_cycle_selected_B2"
                    if sparse_outer_cycle_active
                    and outer_cycle_selected_stage == "B2"
                    else "p64_sparse_exact_basis_outer_cycle_selected_B1"
                    if sparse_outer_cycle_active
                    and outer_cycle_selected_stage == "B1"
                    else "p64_sparse_exact_basis_outer_cycle_rejected_public_fallback"
                    if sparse_outer_cycle_active
                    else "p64_sparse_exact_basis_frame_variables_with_coarse_fixed"
                    if stage_b_valid
                    and stage_b_mode == "p64_sparse_exact_hierarchical_basis"
                    else "p64_gauge_fixed_frame_variables_with_coarse_fixed"
                    if stage_b_valid
                    and stage_b_mode == "p64_unbiased_low_frequency_gauge_fix"
                    else "p64_proximal_frame_variables_with_coarse_fixed"
                    if stage_b_valid
                    and stage_b_mode == "p64_direct_z_plus_q_proximal"
                    else "full_frame_variables_with_coarse_fixed"
                )
                outer_cycle_diagnostics_for_polish = (
                    {}
                    if joint_subspace_stage_b_trust is None
                    else joint_subspace_stage_b_trust.get("outer_cycle", {})
                )
                joint_subspace_diagnostics["fixed_coarse_frame_polish"] = {
                    "num_function_evaluations": int(result.nfev),
                    "num_jacobian_evaluations": (
                        None
                        if getattr(result, "njev", None) is None
                        else int(result.njev)
                    ),
                    "success": bool(result.success),
                    "termination_status": int(result.status),
                    "message": str(result.message),
                    "scipy_cost": float(result.cost),
                    "stage_b_frame_trust": joint_subspace_stage_b_trust,
                    "outer_cycle_total_nfev": (
                        outer_cycle_diagnostics_for_polish.get(
                            "candidate_total_nfev"
                        )
                    ),
                    "outer_cycle_total_njev": (
                        outer_cycle_diagnostics_for_polish.get(
                            "candidate_total_njev"
                        )
                    ),
                }
                if cfg.joint_subspace_unrestricted_repair_max_nfev > 0:
                    used_nfev = int(joint_subspace_result.nfev) + int(
                        result.nfev
                    )
                    remaining_nfev = max(0, int(cfg.max_nfev) - used_nfev)
                    repair_nfev = min(
                        int(cfg.joint_subspace_unrestricted_repair_max_nfev),
                        remaining_nfev,
                    )
                    if repair_nfev > 0:
                        fixed_coarse_result = result
                        repair_started = perf_counter()
                        result = solve_once(
                            fixed_coarse_result.x,
                            {
                                **least_squares_kwargs,
                                "max_nfev": repair_nfev,
                            },
                        )
                        discarded_final_joint_results.append(
                            fixed_coarse_result
                        )
                        joint_subspace_unrestricted_repair = {
                            "schema": (
                                "hierarchical_joint_bounded_unrestricted_"
                                "repair_v1"
                            ),
                            "interpretation": (
                                "release_coarse_and_frame_variables_for_a_"
                                "bounded_post_polish_repair"
                            ),
                            "requested_max_nfev": int(
                                cfg.joint_subspace_unrestricted_repair_max_nfev
                            ),
                            "effective_max_nfev": int(repair_nfev),
                            "num_function_evaluations": int(result.nfev),
                            "num_jacobian_evaluations": (
                                None
                                if getattr(result, "njev", None) is None
                                else int(result.njev)
                            ),
                            "success": bool(result.success),
                            "termination_status": int(result.status),
                            "message": str(result.message),
                            "wall_seconds": float(
                                perf_counter() - repair_started
                            ),
                        }
                        joint_subspace_diagnostics["final_polish_mode"] = (
                            "fixed_coarse_frame_then_bounded_unrestricted_joint"
                        )
            else:
                result = solve_once(
                    joint_start,
                    final_solve_kwargs,
                    initial_cache=final_solve_initial_cache,
                )
                final_solve_initial_cache_reused = bool(
                    final_solve_initial_cache is not None
                )
                if joint_subspace_diagnostics is not None:
                    joint_subspace_diagnostics["final_polish_mode"] = (
                        "unrestricted_full_joint"
                    )
        if (
            joint_subspace_diagnostics is not None
            and joint_subspace_diagnostics["accepted"]
        ):
            final_subspace_residual = residual(result.x)
            coarse_row_stop = 7 * len(edges)
            initial_subspace_residual = initial_residual
            initial_subspace_coarse = initial_subspace_residual[:coarse_row_stop]
            final_subspace_coarse = final_subspace_residual[:coarse_row_stop]
            initial_subspace_coarse_cost = float(
                initial_subspace_coarse @ initial_subspace_coarse
            )
            final_subspace_coarse_cost = float(
                final_subspace_coarse @ final_subspace_coarse
            )
            final_subspace_coarse_ratio = (
                final_subspace_coarse_cost / initial_subspace_coarse_cost
                if initial_subspace_coarse_cost > 0.0
                else 1.0
                if final_subspace_coarse_cost == 0.0
                else np.inf
            )
            final_subspace_statistics_ratio = _coarse_edge_statistics_ratios(
                _coarse_edge_residual_statistics(initial_subspace_coarse, edges),
                _coarse_edge_residual_statistics(final_subspace_coarse, edges),
            )
            final_subspace_local_statistics = (
                _joint_local_residual_statistics(
                    final_subspace_residual,
                    num_coarse_edges=len(edges),
                    num_fine_edges=len(fine_edges),
                    num_frame_priors=len(prior_records),
                    fine_edge_weight=fine_edge_weight,
                    coarse_prior_weight=coarse_prior_weight,
                    translation_scale=cfg.translation_scale,
                )
            )
            final_subspace_local_statistics_ratios = (
                _joint_local_residual_statistics_ratios(
                    joint_subspace_diagnostics[
                        "initial_local_residual_statistics"
                    ],
                    final_subspace_local_statistics,
                )
            )
            frame_polish_mode_energy = (
                joint_subspace_frame_polish_mode_energy
                if joint_subspace_frame_polish_mode_energy is not None
                else _sequential_pose_mode_energy_diagnostics(
                    np.asarray(
                        result.x[frame_base:] - joint_start[frame_base:],
                        dtype=np.float64,
                    ),
                    block_dimension=6,
                    aggregate_sizes=(cfg.joint_subspace_control_stride,),
                    component_groups=(
                        ("rotation", 0, 3),
                        ("translation", 3, 6),
                    ),
                )
            )
            final_subspace_family_checks = (
                (
                    cfg.joint_schur_step_max_adjacent_p95_ratio,
                    final_subspace_statistics_ratio["adjacent"]["p95"],
                ),
                (
                    cfg.joint_schur_step_max_loop_p95_ratio,
                    final_subspace_statistics_ratio["loop"]["p95"],
                ),
                (
                    cfg.joint_schur_step_max_loop_max_ratio,
                    final_subspace_statistics_ratio["loop"]["max"],
                ),
            )
            final_subspace_family_gate = all(
                threshold is None
                or observed is None
                or observed <= threshold
                for threshold, observed in final_subspace_family_checks
            )
            subspace_polish_cost_before = float(
                joint_subspace_diagnostics["candidate_full_robust_cost"]
            )
            final_subspace_cost = _scipy_compatible_robust_cost(
                final_subspace_residual,
                loss=cfg.robust_loss,
                f_scale=cfg.f_scale,
            )
            subspace_solver_success_gate = bool(
                joint_subspace_stage_b_trust is None
                or not joint_subspace_stage_b_trust.get("valid", True)
                or result.success
            )
            subspace_final_gate_passed = bool(
                subspace_solver_success_gate
                and np.all(np.isfinite(result.x))
                and np.all(np.isfinite(final_subspace_residual))
                and np.isfinite(final_subspace_cost)
                and final_subspace_cost
                <= subspace_polish_cost_before
                + max(1e-12, 1e-10 * abs(subspace_polish_cost_before))
                and final_subspace_coarse_ratio
                <= cfg.joint_schur_step_max_graph_cost_ratio
                and final_subspace_family_gate
            )
            stage_b_gate_mode = (
                None
                if joint_subspace_stage_b_trust is None
                else joint_subspace_stage_b_trust.get("mode")
            )
            stage_b_gate_valid = bool(
                joint_subspace_stage_b_trust is not None
                and joint_subspace_stage_b_trust.get("valid", True)
            )
            outer_cycle_gate_selected_stage = (
                None
                if joint_subspace_stage_b_trust is None
                else joint_subspace_stage_b_trust.get(
                    "outer_cycle_selected_stage"
                )
            )
            outer_cycle_gate_coarse_backtracking_selected = bool(
                joint_subspace_stage_b_trust is not None
                and joint_subspace_stage_b_trust.get(
                    "outer_cycle_coarse_backtracking_selected", False
                )
            )
            sparse_outer_cycle_gate_active = bool(
                stage_b_gate_valid
                and stage_b_gate_mode == "p64_sparse_exact_hierarchical_basis"
                and cfg.joint_subspace_stage_b_p64_sparse_basis_outer_cycle
            )
            joint_subspace_diagnostics["final_joint_gate"] = {
                "passed": subspace_final_gate_passed,
                "policy": (
                    "strict_solver_success_plus_full_objective_nonincrease_"
                    "plus_cumulative_coarse_topology"
                    if stage_b_gate_valid
                    else "full_objective_nonincrease_plus_cumulative_coarse_"
                    "topology"
                ),
                "success": bool(result.success),
                "solver_success_gate_passed": subspace_solver_success_gate,
                "robust_cost_before": subspace_polish_cost_before,
                "robust_cost_after": float(final_subspace_cost),
                "coarse_graph_cost_ratio": float(
                    final_subspace_coarse_ratio
                ),
                "coarse_edge_statistics_ratios": (
                    final_subspace_statistics_ratio
                ),
                "local_motion_gate_role": (
                    "diagnostic_only_round15_hierarchical_discovery"
                    if joint_subspace_stage_b_trust is not None
                    else "diagnostic_only_round14_discovery"
                ),
                "local_residual_statistics": (
                    final_subspace_local_statistics
                ),
                "local_residual_statistics_ratios": (
                    final_subspace_local_statistics_ratios
                ),
                "frame_polish_step_mode_energy": frame_polish_mode_energy,
                "stage_b_frame_trust": joint_subspace_stage_b_trust,
                "unrestricted_repair": joint_subspace_unrestricted_repair,
                "public_fallback_outside_stage_budget": False,
                "aggregate_graph_gate_passed": bool(
                    final_subspace_coarse_ratio
                    <= cfg.joint_schur_step_max_graph_cost_ratio
                ),
                "edge_family_gate_passed": final_subspace_family_gate,
                "selected_solution": (
                    "subspace_fixed_coarse_then_bounded_unrestricted_repair"
                    if subspace_final_gate_passed
                    and joint_subspace_unrestricted_repair is not None
                    else "subspace_then_p64_sparse_exact_basis_outer_cycle_coarse_backtracking_selected_B2"
                    if subspace_final_gate_passed
                    and sparse_outer_cycle_gate_active
                    and outer_cycle_gate_selected_stage == "B2"
                    and outer_cycle_gate_coarse_backtracking_selected
                    else "subspace_then_p64_sparse_exact_basis_outer_cycle_selected_B2"
                    if subspace_final_gate_passed
                    and sparse_outer_cycle_gate_active
                    and outer_cycle_gate_selected_stage == "B2"
                    else "subspace_then_p64_sparse_exact_basis_outer_cycle_selected_B1"
                    if subspace_final_gate_passed
                    and sparse_outer_cycle_gate_active
                    and outer_cycle_gate_selected_stage == "B1"
                    else "subspace_then_p64_sparse_exact_basis_frame_polish"
                    if subspace_final_gate_passed
                    and stage_b_gate_valid
                    and stage_b_gate_mode == "p64_sparse_exact_hierarchical_basis"
                    else "subspace_then_p64_gauge_fixed_frame_polish"
                    if subspace_final_gate_passed
                    and stage_b_gate_valid
                    and stage_b_gate_mode == "p64_unbiased_low_frequency_gauge_fix"
                    else "subspace_then_p64_proximal_frame_polish"
                    if subspace_final_gate_passed
                    and stage_b_gate_valid
                    and stage_b_gate_mode == "p64_direct_z_plus_q_proximal"
                    else "subspace_then_fixed_coarse_frame_polish"
                    if subspace_final_gate_passed
                    and cfg.joint_subspace_conditional_frame_polish
                    else "subspace_warm_started_full_joint"
                    if subspace_final_gate_passed
                    else "public_full_joint_fallback"
                ),
            }
            if not subspace_final_gate_passed:
                discarded_final_joint_results.append(result)
                result = solve_once(initial_vector, least_squares_kwargs)
                joint_subspace_diagnostics["final_joint_gate"]["fallback"] = (
                    "public_joint_initialization"
                )
                joint_subspace_diagnostics["final_joint_gate"][
                    "public_fallback_outside_stage_budget"
                ] = True
        if schur_step_diagnostics is not None:
            schur_step_diagnostics["final_polish_mode"] = (
                "none"
                if cfg.joint_schur_step_final_solver
                else "full_joint"
            )
            schur_step_diagnostics["public_joint_initial_linearization_reused"] = (
                final_solve_initial_cache_reused
            )
        if (
            cfg.joint_schur_step
            and not cfg.joint_schur_step_final_solver
            and schur_step_diagnostics is not None
            and schur_step_diagnostics["accepted"]
        ):
            coarse_row_stop = 7 * len(edges)
            assert initial_coarse_cost is not None
            assert initial_coarse_statistics is not None

            def evaluate_final_joint_gate(
                candidate_result: Any,
                *,
                robust_cost_before: float,
            ) -> dict[str, Any]:
                final_coarse = residual(candidate_result.x)[:coarse_row_stop]
                final_coarse_statistics = _coarse_edge_residual_statistics(
                    final_coarse, edges
                )
                final_coarse_ratios = _coarse_edge_statistics_ratios(
                    initial_coarse_statistics, final_coarse_statistics
                )
                final_coarse_cost = float(final_coarse @ final_coarse)
                final_coarse_cost_ratio = (
                    final_coarse_cost / initial_coarse_cost
                    if initial_coarse_cost > 0.0
                    else 1.0
                    if final_coarse_cost == 0.0
                    else np.inf
                )
                passed = bool(
                    np.all(np.isfinite(candidate_result.x))
                    and np.all(np.isfinite(candidate_result.fun))
                    and np.isfinite(candidate_result.cost)
                    and candidate_result.cost
                    <= robust_cost_before
                    + max(1e-12, 1e-10 * abs(robust_cost_before))
                )
                return {
                    "passed": passed,
                    "policy": (
                        "full_objective_nonincrease_after_guarded_full_schur_start"
                    ),
                    "success": bool(candidate_result.success),
                    "coarse_graph_cost_ratio": float(final_coarse_cost_ratio),
                    "coarse_edge_statistics_ratios": final_coarse_ratios,
                    "robust_cost_before": float(robust_cost_before),
                    "robust_cost_after": float(candidate_result.cost),
                    "candidate_solver_mode": "full_joint",
                }

            warm_start_gate = evaluate_final_joint_gate(
                result,
                robust_cost_before=accepted_warm_cost,
            )
            schur_step_diagnostics["final_joint_gate"] = {
                **warm_start_gate,
                "reference": "accepted_full_coupled_schur_start",
                "selected_solution": "warm_started_full_joint",
            }
            if not warm_start_gate["passed"]:
                discarded_final_joint_results.append(result)
                public_fallback_start = (
                    public_pilot_result.x
                    if public_pilot_result is not None
                    else initial_vector
                )
                public_result = solve_once(
                    public_fallback_start, final_solve_kwargs
                )
                public_gate = evaluate_final_joint_gate(
                    public_result,
                    robust_cost_before=initial_full_cost,
                )
                schur_step_diagnostics["final_joint_gate"][
                    "public_initialization_candidate"
                ] = public_gate
                result = public_result
                schur_step_diagnostics["final_joint_gate"]["fallback"] = (
                    "public_joint_initialization"
                )
                schur_step_diagnostics["final_joint_gate"][
                    "selected_solution"
                ] = "public_full_joint"
    finally:
        solver_wall_seconds = perf_counter() - solver_wall_start
        with residual_profile_lock:
            residual_profile_solver_active = False
        if active_analytic_jacobian is not None:
            active_analytic_jacobian.restore_threads()
    if joint_full_sparse_exact_basis_diagnostics is not None:
        final_residual = np.asarray(result.fun, dtype=np.float64)
    else:
        final_residual = residual(result.x)
    final_optimizer_residual = optimizer_residual_view(final_residual)
    final_unpack_started = perf_counter()
    final_frame_deltas = None
    if factorized_frame_chart_residual_v4:
        coarse, frames, final_frame_deltas = (
            unpack_factorized_frame_chart_residual_v4(result.x)
        )
    else:
        coarse, frames = unpack(result.x)
    final_unpack_wall_seconds = perf_counter() - final_unpack_started
    factorized_frame_chart_residual_v4_diagnostics = None
    if factorized_frame_chart_residual_v4:
        def residual_component_max(values: np.ndarray) -> dict[str, float]:
            coarse_values = np.asarray(
                values[: 7 * len(edges)], dtype=np.float64
            ).reshape(len(edges), 7)
            local_values = np.asarray(
                values[7 * len(edges) :], dtype=np.float64
            ).reshape(len(fine_edges) + len(prior_records), 6)
            scale_values = (
                coarse_values[:, 6] if len(edges) else np.zeros(0, dtype=np.float64)
            )
            rotation_parts = []
            translation_parts = []
            if len(edges):
                rotation_parts.append(coarse_values[:, :3])
                translation_parts.append(coarse_values[:, 3:6])
            if len(local_values):
                rotation_parts.append(local_values[:, :3])
                translation_parts.append(local_values[:, 3:6])
            rotations = (
                np.concatenate(rotation_parts, axis=0)
                if rotation_parts
                else np.zeros((0, 3), dtype=np.float64)
            )
            translations = (
                np.concatenate(translation_parts, axis=0)
                if translation_parts
                else np.zeros((0, 3), dtype=np.float64)
            )
            return {
                "scale_log": float(
                    np.max(np.abs(scale_values), initial=0.0)
                ),
                "rotation_rad": float(
                    np.max(np.linalg.norm(rotations, axis=1), initial=0.0)
                ),
                "translation": float(
                    np.max(np.linalg.norm(translations, axis=1), initial=0.0)
                ),
            }

        identity_component_max = residual_component_max(initial_residual)
        identity_local_gate_pass = bool(
            identity_component_max["scale_log"] < 1e-9
            and identity_component_max["rotation_rad"] < 1e-8
            and identity_component_max["translation"] < 1e-8
        )
        absolute_compose_error_diagnostic: dict[str, Any]
        try:
            absolute_residual = _weighted_edge_residuals_batch_exact(
                edges, coarse, cfg.translation_scale
            )
            absolute_compose_error_diagnostic = {
                "valid": True,
                "component_max": residual_component_max(
                    np.concatenate(
                        [
                            absolute_residual,
                            np.zeros(
                                6 * (len(fine_edges) + len(prior_records)),
                                dtype=np.float64,
                            ),
                        ]
                    )
                ),
            }
        except ValueError as exc:
            absolute_compose_error_diagnostic = {
                "valid": False,
                "reason": str(exc),
            }
        final_pose_finite = _sim3_sequence_finite([*coarse, *frames])
        delta_finite = (
            final_frame_deltas is not None
            and _sim3_sequence_finite(final_frame_deltas)
        )
        export_materialization_safety = (
            _factorized_v4_export_materialization_safety(
                initial_frame_poses,
                final_frame_deltas,
                translation_scale=cfg.translation_scale,
            )
        )
        constants_valid = bool(
            factorized_constants_diagnostics is not None
            and factorized_constants_diagnostics.get(
                "coarse_path_identity_gate_pass"
            )
            is True
        )
        factorized_frame_chart_residual_v4_diagnostics = {
            "schema": "factorized_frame_chart_residual_v4",
            "mode": "frozen_coarse_factorized_frame_chart_residual",
            "enabled": True,
            "valid": bool(
                constants_valid
                and final_pose_finite
                and delta_finite
                and export_materialization_safety["gate_pass"]
            ),
            "reason": (
                None
                if constants_valid
                and final_pose_finite
                and delta_finite
                and export_materialization_safety["gate_pass"]
                else "factorized_constants_final_finite_or_export_safety_gate_failed"
            ),
            "activation_scope": "freeze_coarse_alignments_adjacent_chain_keyframe_or_frozen",
            "coarse_residual_source": "adjacent_edge_path_identity_constant_zero_when_coarse_frozen",
            "fine_residual_source": "M_inverse_Db_inverse_stable_base_Da",
            "prior_residual_source": "stable_chart_error_Q_times_Df",
            "fallback_to_absolute_residual": False,
            "initialization": factorized_v4_initialization_diagnostics,
            "identity_delta_component_max": identity_component_max,
            "identity_delta_local_residual_gate_pass": identity_local_gate_pass,
            "identity_delta_local_residual_gate_thresholds": {
                "scale_log": 1e-9,
                "rotation_rad": 1e-8,
                "translation": 1e-8,
            },
            "identity_delta_residual_vector": (
                initial_residual.tolist() if row_count <= 512 else None
            ),
            "final_residual_vector": (
                final_residual.tolist() if row_count <= 512 else None
            ),
            "absolute_compose_error_diagnostic_only": (
                absolute_compose_error_diagnostic
            ),
            "final_materialized_pose_finite": final_pose_finite,
            "final_delta_finite": bool(delta_finite),
            "export_materialization_safety": export_materialization_safety,
            **(
                factorized_constants_diagnostics
                if factorized_constants_diagnostics is not None
                else {}
            ),
        }
        if not export_materialization_safety["gate_pass"]:
            raise ValueError(
                "factorized_frame_chart_residual_v4 export materialization "
                "unsafe: "
                f"{export_materialization_safety}"
            )
    result_njev = getattr(result, "njev", None)
    num_jacobian_evaluations = (
        None if result_njev is None else int(result_njev)
    )
    with residual_profile_lock:
        residual_profile = {
            "residual_calls_total": int(residual_calls_total),
            "residual_calls_solver": int(residual_calls_solver),
            "residual_wall_seconds_total": float(residual_wall_seconds_total),
            "residual_wall_seconds_solver": float(residual_wall_seconds_solver),
            "solver_wall_seconds": float(solver_wall_seconds),
        }
    incremental_diagnostics = (
        incremental_cache.diagnostics()
        if incremental_cache is not None
        else {
            "incremental_frozen_residual_enabled": False,
            "incremental_frozen_residual_mode": "disabled",
            "incremental_frozen_residual_fallback_reason": incremental_fallback_reason,
            "incremental_frozen_residual_batches": 0,
            "incremental_frozen_residual_vectors": 0,
            "incremental_frozen_residual_hits": 0,
            "incremental_frozen_residual_full_fallbacks": 0,
            "incremental_frozen_residual_baseline_updates": 0,
            "incremental_frozen_residual_local_rows_recomputed_total": 0,
            "incremental_frozen_residual_local_rows_recomputed_max": 0,
            "incremental_frozen_residual_local_rotation_rows_recomputed_total": 0,
            "incremental_frozen_residual_phase_seconds": {
                "dependency_selection": 0.0,
                "frame_update": 0.0,
                "result_copy_write": 0.0,
                "fine_rows": 0.0,
                "prior_rows": 0.0,
            },
        }
    )
    analytic_diagnostics = {
        "analytic_frozen_jacobian_enabled": analytic_jacobian is not None,
        "analytic_frozen_jacobian_fallback_reason": analytic_fallback_reason,
        "analytic_frozen_jacobian_calls": (
            0 if analytic_jacobian is None else int(analytic_jacobian.calls)
        ),
        "analytic_frozen_jacobian_wall_seconds": (
            0.0 if analytic_jacobian is None else float(analytic_jacobian.wall_seconds)
        ),
        "analytic_frozen_jacobian_preparation_wall_seconds": (
            0.0
            if analytic_jacobian is None
            else float(analytic_jacobian.preparation_wall_seconds)
        ),
        "analytic_frozen_jacobian_derivative_wall_seconds": (
            0.0
            if analytic_jacobian is None
            else float(analytic_jacobian.derivative_wall_seconds)
        ),
        "analytic_frozen_jacobian_assembly_wall_seconds": (
            0.0
            if analytic_jacobian is None
            else float(analytic_jacobian.assembly_wall_seconds)
        ),
        "analytic_frozen_jacobian_threads_requested": (
            int(cfg.analytic_jacobian_threads)
        ),
        "analytic_frozen_jacobian_threads_actual": (
            0 if analytic_jacobian is None else int(analytic_jacobian.threads_actual)
        ),
        "analytic_frozen_jacobian_backend": (
            None if analytic_jacobian is None else analytic_jacobian.backend
        ),
        "analytic_frozen_jacobian_csr_topology_precomputed": (
            analytic_jacobian is not None
        ),
        "analytic_closed_form_frame_derivatives_reused": bool(
            analytic_jacobian is not None
            and analytic_jacobian.backend == "closed_form"
        ),
        "analytic_closed_form_nonzero_blocks_only": bool(
            analytic_jacobian is not None
            and analytic_jacobian.backend == "closed_form"
        ),
        "analytic_closed_form_log_inverse_intermediates_reused": bool(
            analytic_jacobian is not None
            and analytic_jacobian.backend == "closed_form"
        ),
        "final_frame_rotation_conversion": "batched_scipy_from_rotvec",
        "final_frame_sim3_validation": (
            "batched_finite_then_immutable_construct"
        ),
        "final_solution_unpack_wall_seconds": float(final_unpack_wall_seconds),
        "frozen_relative_constraint_intermediate_sim3_elided": bool(
            frozen_vectorized is not None
        ),
        "aligned_pose_sim3_validation": "batched_finite_then_immutable_construct",
        "aligned_pose_sim3_validation_scope": "all_chunks_single_batch",
        "frozen_relative_sim3_validation": "batched_finite_then_immutable_construct",
        "frozen_relative_component_computation": "batched_explicit_exact",
        "frozen_coarse_constant_residual": "batched_exact",
        "frozen_vectorized_sim3_materialization": (
            "components_only" if frozen_vectorized is not None else "sim3_objects"
        ),
        "initial_frame_parameterization": "batched_scipy_from_matrix_as_rotvec",
        "initial_parameterization_wall_seconds": float(
            initial_parameterization_wall_seconds
        ),
        "residual_rotation_matrix_product": "explicit_three_term_exact",
        "residual_rotation_matrix_product_backend": (
            "numexpr_exact" if _numexpr is not None else "numpy_exact_fallback"
        ),
        "residual_rotation_matrix_product_threads": _NUMEXPR_EXACT_THREADS,
        "so3_three_component_reductions": "explicit_exact_shared_cosine",
        "robust_loss_implementation": (
            "exact_sparse_outlier_huber"
            if exact_huber_loss is not None
            else "scipy_builtin_or_callable"
        ),
        "robust_loss_row_count": int(optimizer_row_count),
        "robust_loss_scope": "optimizer_residual_rows",
        "robust_loss_calls": (
            0 if exact_huber_loss is None else int(exact_huber_loss.calls)
        ),
        "robust_loss_outliers_total": (
            0 if exact_huber_loss is None else int(exact_huber_loss.outliers_total)
        ),
        "robust_loss_outliers_max": (
            0 if exact_huber_loss is None else int(exact_huber_loss.outliers_max)
        ),
        "analytic_fast_residual_logmap_enabled": bool(
            analytic_jacobian is not None and cfg.analytic_fast_residual_logmap
        ),
        "analytic_residual_rotation_conversion": (
            "numpy_closed_form_scipy_near_pi_fallback"
            if analytic_jacobian is not None and cfg.analytic_fast_residual_logmap
            else "scipy_from_matrix_as_rotvec"
        ),
    }
    dynamic_analytic_diagnostics = (
        dynamic_analytic_jacobian.diagnostics()
        if dynamic_analytic_jacobian is not None
        else {
            "analytic_dynamic_jacobian_enabled": False,
            "analytic_dynamic_jacobian_fallback_reason": (
                dynamic_analytic_fallback_reason
            ),
            "analytic_dynamic_jacobian_mode": "disabled",
            "analytic_dynamic_jacobian_calls": 0,
            "analytic_dynamic_jacobian_wall_seconds": 0.0,
            "analytic_dynamic_jacobian_preparation_wall_seconds": 0.0,
            "analytic_dynamic_jacobian_derivative_wall_seconds": 0.0,
            "analytic_dynamic_jacobian_assembly_wall_seconds": 0.0,
            "analytic_dynamic_jacobian_threads_requested": int(
                cfg.analytic_jacobian_threads
            ),
            "analytic_dynamic_jacobian_threads_actual": 0,
            "analytic_dynamic_jacobian_backend": (
                cfg.analytic_dynamic_jacobian_backend
            ),
            "analytic_dynamic_jacobian_local_rotation_projection": (
                cfg.analytic_dynamic_jacobian_local_rotation_projection
            ),
            "analytic_dynamic_jacobian_csr_topology_precomputed": False,
            "analytic_dynamic_jacobian_root_columns_omitted": False,
            "analytic_dynamic_jacobian_local_rotation_requirement": None,
        }
    )
    active_jacobian_nnz = (
        int(active_analytic_jacobian.last_nnz)
        if active_analytic_jacobian is not None
        else int(optimizer_sparsity.nnz)
        if optimizer_sparsity is not None
        else None
    )
    active_jacobian_density = (
        float(active_jacobian_nnz / (optimizer_row_count * num_variables))
        if active_jacobian_nnz is not None
        and optimizer_row_count
        and num_variables
        else None
    )
    joint_family_diagnostics = (
        _joint_family_diagnostics_from_solver_result(
            result,
            num_coarse_edges=len(edges),
            num_fine_edges=len(fine_edges),
            num_frame_priors=len(prior_records),
            frame_base=frame_base,
            num_variables=num_variables,
        )
        if cfg.joint_family_diagnostics and not cfg.freeze_coarse_alignments
        else None
    )
    def frame_relaxation_diagnostics(
        stage_result: Any,
        start: np.ndarray,
        *,
        max_nfev: int,
        interpretation: str,
    ) -> dict[str, Any]:
        frame_gradient = np.asarray(stage_result.grad[frame_base:], dtype=np.float64)
        return {
            "schema": "hierarchical_joint_frame_relaxation_v1",
            "interpretation": interpretation,
            "max_nfev": int(max_nfev),
            "success": bool(stage_result.success),
            "termination_status": int(stage_result.status),
            "num_function_evaluations": int(stage_result.nfev),
            "num_jacobian_evaluations": (
                None
                if getattr(stage_result, "njev", None) is None
                else int(stage_result.njev)
            ),
            "scipy_cost": float(stage_result.cost),
            "optimality": float(stage_result.optimality),
            "frame_gradient_inf": (
                float(np.linalg.norm(frame_gradient, ord=np.inf))
                if frame_gradient.size
                else 0.0
            ),
            "coarse_step_inf": float(
                np.linalg.norm(
                    stage_result.x[:frame_base] - start[:frame_base],
                    ord=np.inf,
                )
            ),
            "frame_step_l2": float(
                np.linalg.norm(stage_result.x[frame_base:] - start[frame_base:])
            ),
        }

    joint_frame_relaxation = (
        {
            **frame_relaxation_diagnostics(
                frame_relaxation_result,
                initial_vector,
                max_nfev=cfg.joint_frame_relaxation_max_nfev,
                interpretation="fixed_coarse_conditional_frame_warm_start",
            ),
        }
        if frame_relaxation_result is not None
        else None
    )
    joint_public_pilot = (
        {
            "schema": "hierarchical_joint_public_pilot_v1",
            "requested_max_nfev": int(
                cfg.joint_schur_step_public_pilot_max_nfev
            ),
            "num_function_evaluations": int(public_pilot_result.nfev),
            "num_jacobian_evaluations": (
                None
                if getattr(public_pilot_result, "njev", None) is None
                else int(public_pilot_result.njev)
            ),
            "success": bool(public_pilot_result.success),
            "termination_status": int(public_pilot_result.status),
            "message": str(public_pilot_result.message),
            "scipy_cost": float(public_pilot_result.cost),
            "optimality": float(public_pilot_result.optimality),
            "remaining_public_nfev": max(
                0,
                int(cfg.max_nfev) - int(public_pilot_result.nfev),
            ),
            "decision": (
                "return_converged_public_pilot"
                if public_pilot_result.success
                else "activate_schur_then_continue_with_remaining_budget"
            ),
        }
        if public_pilot_result is not None
        else None
    )
    joint_schur_diagnostics = schur_diagnostics_raw
    hierarchy_stage_results = tuple(
        stage_result
        for stage_result in (
            joint_subspace_result,
            public_pilot_result,
            frame_relaxation_result,
            *(
                ()
                if cfg.joint_schur_step_final_solver
                else schur_linearization_results
            ),
            *discarded_final_joint_results,
        )
        if stage_result is not None
        and stage_result is not result
        and not (
            final_solve_initial_cache_reused
            and stage_result is final_solve_initial_cache
        )
    )
    hierarchy_stage_njev_complete = all(
        getattr(stage_result, "njev", None) is not None
        for stage_result in hierarchy_stage_results
    )
    warm_start_nfev = (
        0
        if hierarchical_warm_start_diagnostics is None
        else int(hierarchical_warm_start_diagnostics["num_function_evaluations"])
    )
    warm_start_njev = (
        0
        if hierarchical_warm_start_diagnostics is None
        else hierarchical_warm_start_diagnostics["num_jacobian_evaluations"]
    )
    final_gate_selected_solution = (
        None
        if joint_subspace_diagnostics is None
        else joint_subspace_diagnostics.get("final_joint_gate", {}).get(
            "selected_solution"
        )
    )
    stage_b_solver_mode = None
    if joint_subspace_stage_b_trust is not None:
        stage_b_solver_trust_mode = joint_subspace_stage_b_trust.get("mode")
        stage_b_solver_trust_valid = bool(
            joint_subspace_stage_b_trust.get("valid", True)
        )
        stage_b_solver_outer_stage = joint_subspace_stage_b_trust.get(
            "outer_cycle_selected_stage"
        )
        stage_b_solver_coarse_backtracking_selected = bool(
            joint_subspace_stage_b_trust.get(
                "outer_cycle_coarse_backtracking_selected", False
            )
        )
        if (
            stage_b_solver_trust_valid
            and stage_b_solver_trust_mode
            == "p64_sparse_exact_hierarchical_basis"
            and cfg.joint_subspace_stage_b_p64_sparse_basis_outer_cycle
        ):
            if (
                final_gate_selected_solution == "public_full_joint_fallback"
                or stage_b_solver_outer_stage not in {"B1", "B2"}
            ):
                stage_b_solver_mode = (
                    "joint_subspace_then_p64_sparse_exact_basis_outer_cycle_"
                    "rejected_public_fallback"
                )
            elif stage_b_solver_outer_stage == "B2":
                if stage_b_solver_coarse_backtracking_selected:
                    stage_b_solver_mode = (
                        "joint_subspace_then_p64_sparse_exact_basis_outer_"
                        "cycle_coarse_backtracking_selected_B2"
                    )
                else:
                    stage_b_solver_mode = (
                        "joint_subspace_then_p64_sparse_exact_basis_outer_"
                        "cycle_selected_B2"
                    )
            else:
                stage_b_solver_mode = (
                    "joint_subspace_then_p64_sparse_exact_basis_outer_cycle_"
                    "selected_B1"
                )
        elif (
            stage_b_solver_trust_valid
            and stage_b_solver_trust_mode
            == "p64_sparse_exact_hierarchical_basis"
        ):
            stage_b_solver_mode = (
                "joint_subspace_then_p64_sparse_exact_basis_frame_polish"
            )
        elif (
            stage_b_solver_trust_valid
            and stage_b_solver_trust_mode
            == "p64_unbiased_low_frequency_gauge_fix"
        ):
            stage_b_solver_mode = "joint_subspace_then_p64_gauge_fixed_frame_polish"
        elif (
            stage_b_solver_trust_valid
            and stage_b_solver_trust_mode == "p64_direct_z_plus_q_proximal"
        ):
            stage_b_solver_mode = "joint_subspace_then_p64_proximal_frame_polish"
    active_bound_count = int(np.count_nonzero(result.active_mask))
    diagnostics = {
        "status": "ok" if result.success else "not_converged",
        "optimizer_used": True,
        "success": bool(result.success),
        "termination_status": int(result.status),
        "message": str(result.message),
        "num_function_evaluations": int(result.nfev),
        "num_function_evaluations_total": int(result.nfev)
        + sum(int(stage_result.nfev) for stage_result in hierarchy_stage_results)
        + warm_start_nfev,
        "num_jacobian_evaluations": num_jacobian_evaluations,
        "num_jacobian_evaluations_total": (
            None
            if num_jacobian_evaluations is None
            or not hierarchy_stage_njev_complete
            or warm_start_njev is None
            else num_jacobian_evaluations
            + sum(
                int(stage_result.njev)
                for stage_result in hierarchy_stage_results
            )
            + int(warm_start_njev)
        ),
        "max_nfev": int(cfg.max_nfev),
        "solver_ftol": float(cfg.solver_ftol),
        "solver_xtol": float(cfg.solver_xtol),
        "solver_gtol": float(cfg.solver_gtol),
        "joint_tr_solver": (
            "lsmr" if cfg.joint_lsmr_tolerance is not None else "auto"
        ),
        "joint_lsmr_tolerance": (
            None
            if cfg.joint_lsmr_tolerance is None
            else float(cfg.joint_lsmr_tolerance)
        ),
        "num_variables": int(num_variables),
        "full_residual_row_count": int(row_count),
        "optimizer_row_count": int(optimizer_row_count),
        "optimizer_residual_row_start": int(optimizer_residual_row_start),
        "optimizer_residual_scope": (
            "variable_fine_prior_rows"
            if frozen_coarse_rows_excluded_from_optimizer
            else "full_residual_rows"
        ),
        "frozen_coarse_rows_excluded_from_optimizer": bool(
            frozen_coarse_rows_excluded_from_optimizer
        ),
        "num_chunks": int(num_chunks),
        "num_frames": int(num_frames),
        "solver_mode": (
            "keyframe_then_full"
            if hierarchical_warm_start_diagnostics is not None
            else joint_full_sparse_exact_basis_diagnostics.get("mode")
            if joint_full_sparse_exact_basis_diagnostics is not None
            else "joint_subspace_early_skip_public"
            if joint_subspace_diagnostics is not None
            and joint_subspace_diagnostics.get("decision")
            == "early_skip_near_zero_coarse_denominator"
            else "joint_subspace_then_bounded_unrestricted_repair"
            if joint_subspace_unrestricted_repair is not None
            else stage_b_solver_mode
            if stage_b_solver_mode is not None
            else "joint_subspace_then_fixed_coarse_frame_polish"
            if joint_subspace_diagnostics is not None
            and cfg.joint_subspace_conditional_frame_polish
            else "joint_subspace_then_full"
            if joint_subspace_diagnostics is not None
            else schur_final_solver_diagnostics.get("mode")
            if schur_final_solver_diagnostics is not None
            else None
        ),
        "long_sequence_mode": cfg.long_sequence_mode,
        "max_frames_per_solve": int(cfg.max_frames_per_solve),
        "num_keyframes": (
            None
            if hierarchical_warm_start_diagnostics is None
            else int(hierarchical_warm_start_diagnostics["num_keyframes"])
        ),
        "num_coarse_edges": int(len(edges)),
        "num_fine_edges": int(len(fine_edges)),
        "num_frame_priors": int(len(prior_records)),
        "frame_stride": int(cfg.frame_stride),
        "residual_family_normalization": bool(cfg.normalize_residual_families),
        "configured_fine_edge_weight": float(cfg.fine_edge_weight),
        "configured_coarse_prior_weight": float(cfg.coarse_prior_weight),
        "fine_edge_weight": fine_edge_weight,
        "coarse_prior_weight": coarse_prior_weight,
        "translation_scale": float(cfg.translation_scale),
        "max_log_scale_delta": float(cfg.max_log_scale_delta),
        "chunk_scale_trust_mode": cfg.chunk_scale_trust_mode,
        "chunk_scale_trust_radii": (
            None
            if scale_trust_radii is None
            else {
                "min": float(np.min(scale_trust_radii[1:]))
                if len(scale_trust_radii) > 1
                else None,
                "median": float(np.median(scale_trust_radii[1:]))
                if len(scale_trust_radii) > 1
                else None,
                "max": float(np.max(scale_trust_radii[1:]))
                if len(scale_trust_radii) > 1
                else None,
            }
        ),
        "coarse_alignments_frozen": bool(cfg.freeze_coarse_alignments),
        "dynamic_residual_vectorized": bool(
            cfg.vectorize_dynamic_residual and dynamic_analytic_jacobian is not None
        ),
        "frozen_residual_constants_precomputed": bool(
            frozen_coarse is not None and cfg.precompute_frozen_residual_constants
        ),
        "frozen_residual_vectorized": bool(frozen_vectorized is not None),
        "fine_build_phase_seconds": fine_build_phase_seconds,
        "frozen_aligned_local_poses_reused": bool(
            frozen_aligned_local_components is not None
        ),
        "aligned_pose_projection_mode": (
            "per_chunk_batched_svd"
            if frozen_aligned_local_components is not None
            else "per_pose_svd"
        ),
        "aligned_pose_storage": (
            "components_then_unique_global_frame_sim3"
            if frozen_aligned_local_components is not None
            else "per_pose_sim3"
        ),
        "point_corrections_applied": True,
        "jacobian_mode": (
            (
                "sparse_dynamic_closed_form_block_analytic"
                if dynamic_analytic_jacobian.backend == "closed_form"
                else "sparse_dynamic_torch_block_analytic"
            )
            if dynamic_analytic_jacobian is not None
            else
            "sparse_closed_form_block_analytic"
            if analytic_jacobian is not None
            and analytic_jacobian.backend == "closed_form"
            else "sparse_torch_block_analytic"
            if analytic_jacobian is not None
            else "sparse_finite_difference"
            if cfg.use_jac_sparsity
            else "dense_finite_difference"
        ),
        "jacobian_structure": (
            "dynamic_joint_cf_local_blocks"
            if dynamic_analytic_jacobian is not None
            else
            "component_exact_frame_blocks"
            if sparsity is not None
            and (incremental_cache is not None or analytic_jacobian is not None)
            else "block_conservative"
            if sparsity is not None
            else None
        ),
        "solver_worker_backend": (
            "analytic_callable"
            if active_analytic_jacobian is not None
            else "thread_pool"
            if cfg.solver_workers > 1
            else "serial"
        ),
        "solver_workers_requested": int(cfg.solver_workers),
        "solver_workers_actual": (
            1 if active_analytic_jacobian is not None else int(cfg.solver_workers)
        ),
        "solver_workers_runtime_fallback_reason": (
            solver_workers_runtime_fallback_reason
        ),
        "jacobian_nnz": active_jacobian_nnz,
        "jacobian_density": active_jacobian_density,
        "jacobian_sparsity_build_seconds": float(sparsity_build_seconds),
        **({"residual_components": {
            "scope": "full_residual_rows_after_all_residual_weights",
            "coarse_rows_constant": bool(cfg.freeze_coarse_alignments),
            "initial": _residual_component_diagnostics(
                initial_residual, len(edges), len(fine_edges), len(prior_records),
                loss=cfg.robust_loss, f_scale=cfg.f_scale,
            ),
            "final": _residual_component_diagnostics(
                final_residual, len(edges), len(fine_edges), len(prior_records),
                loss=cfg.robust_loss, f_scale=cfg.f_scale,
            ),
        }} if cfg.record_residual_components else {}),
        "initial_cost": float(np.mean(initial_residual**2)) if initial_residual.size else 0.0,
        "initial_robust_cost": _scipy_compatible_robust_cost(
            initial_residual,
            loss=cfg.robust_loss,
            f_scale=cfg.f_scale,
        ),
        "initial_optimizer_robust_cost": _scipy_compatible_robust_cost(
            initial_optimizer_residual,
            loss=cfg.robust_loss,
            f_scale=cfg.f_scale,
        ),
        "final_cost": float(np.mean(final_residual**2)) if final_residual.size else 0.0,
        "final_robust_cost": _scipy_compatible_robust_cost(
            final_residual,
            loss=cfg.robust_loss,
            f_scale=cfg.f_scale,
        ),
        "final_optimizer_robust_cost": _scipy_compatible_robust_cost(
            final_optimizer_residual,
            loss=cfg.robust_loss,
            f_scale=cfg.f_scale,
        ),
        "scipy_cost": float(result.cost),
        "scipy_cost_scope": (
            "variable_fine_prior_rows_only"
            if frozen_coarse_rows_excluded_from_optimizer
            else "full_residual_rows"
        ),
        "optimality": float(result.optimality),
        "gradient_inf_norm": float(
            np.linalg.norm(np.asarray(result.grad, dtype=np.float64), ord=np.inf)
        ),
        "solution_step_norm": float(np.linalg.norm(result.x - initial_vector)),
        "active_bound_count": active_bound_count,
        "solution_finite": bool(np.all(np.isfinite(result.x))),
        "residual_finite": bool(np.all(np.isfinite(final_residual))),
        "cost_finite": bool(
            np.isfinite(result.cost)
            and np.all(np.isfinite(final_residual))
        ),
        **(
            {"joint_family_diagnostics": joint_family_diagnostics}
            if joint_family_diagnostics is not None
            else {}
        ),
        **(
            {"joint_public_pilot": joint_public_pilot}
            if joint_public_pilot is not None
            else {}
        ),
        **(
            {"joint_frame_relaxation": joint_frame_relaxation}
            if joint_frame_relaxation is not None
            else {}
        ),
        **(
            {"joint_full_sparse_exact_basis": joint_full_sparse_exact_basis_diagnostics}
            if joint_full_sparse_exact_basis_diagnostics is not None
            else {}
        ),
        **(
            {"joint_subspace_continuation": None}
            if joint_full_sparse_exact_basis_diagnostics is not None
            else {}
        ),
        **(
            {"joint_subspace_continuation": joint_subspace_diagnostics}
            if joint_subspace_diagnostics is not None
            and joint_full_sparse_exact_basis_diagnostics is None
            else {}
        ),
        **(
            {"joint_schur_diagnostics": joint_schur_diagnostics}
            if joint_schur_diagnostics is not None
            else {}
        ),
        **(
            {"joint_schur_final_solver": schur_final_solver_diagnostics}
            if schur_final_solver_diagnostics is not None
            else {}
        ),
        **(
            {"joint_schur_step": schur_step_diagnostics}
            if schur_step_diagnostics is not None
            else {}
        ),
        **(
            {
                "factorized_frame_chart_residual_v4": (
                    factorized_frame_chart_residual_v4_diagnostics
                )
            }
            if factorized_frame_chart_residual_v4_diagnostics is not None
            else {}
        ),
        **(
            {"hierarchical_warm_start": hierarchical_warm_start_diagnostics}
            if hierarchical_warm_start_diagnostics is not None
            else {}
        ),
        "solver_wall_seconds_total_including_hierarchical_warm_start": float(
            solver_wall_seconds
            + (
                0.0
                if hierarchical_warm_start_diagnostics is None
                else hierarchical_warm_start_diagnostics["wall_seconds"]
            )
        ),
        **residual_profile,
        **incremental_diagnostics,
        **analytic_diagnostics,
        **dynamic_analytic_diagnostics,
    }
    return HierarchicalAlphaBasis(coarse, frames, True, diagnostics)


def apply_hierarchical_alpha(
    chunks: Sequence[Any],
    chunk_ranges: Sequence[Any],
    basis: HierarchicalAlphaBasis,
    *,
    alpha: float,
    apply_frame_corrections: bool = True,
) -> list[Any]:
    """Apply a precomputed hierarchical basis with a specific export alpha."""

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    chunk_list = list(chunks)
    starts_ends = [_range_start_end(item) for item in chunk_ranges]
    if len(chunk_list) != len(starts_ends):
        raise ValueError("chunk and range counts differ")
    if len(basis.coarse_alignments) != len(chunk_list):
        raise ValueError("basis coarse alignment count does not match chunks")
    num_frames = max((end for _, end in starts_ends), default=0)
    if len(basis.frame_poses) != num_frames:
        raise ValueError("basis frame pose count does not match chunk ranges")
    if not chunk_list:
        return []
    _validate_ranges(starts_ends, num_frames)
    effective_alpha = float(alpha) if apply_frame_corrections else 0.0
    if basis.apply_point_corrections:
        return _apply_frame_solution(
            chunk_list,
            starts_ends,
            basis.coarse_alignments,
            basis.frame_poses,
            apply_frame_corrections=True,
            frame_correction_alpha=effective_alpha,
        )

    local_poses = [_chunk_c2w(chunk) for chunk in chunk_list]
    initial_frame_poses = _initial_frame_poses(
        local_poses,
        starts_ends,
        basis.coarse_alignments,
        num_frames,
    )
    return _apply_trajectory_only_solution(
        chunk_list,
        starts_ends,
        basis.frame_poses,
        initial_frame_poses,
        alpha=effective_alpha,
    )


def _range_start_end(item: Any) -> tuple[int, int]:
    if hasattr(item, "start") and hasattr(item, "end"):
        return int(item.start), int(item.end)
    if isinstance(item, dict):
        return int(item["start"]), int(item["end"])
    if len(item) != 2:
        raise ValueError(f"chunk range must contain start/end, got {item!r}")
    return int(item[0]), int(item[1])


def _validate_ranges(ranges: Sequence[tuple[int, int]], num_frames: int) -> None:
    covered = np.zeros(num_frames, dtype=bool)
    for start, end in ranges:
        if start < 0 or end <= start or end > num_frames:
            raise ValueError(f"invalid chunk range {(start, end)}")
        covered[start:end] = True
    if not bool(covered.all()):
        raise ValueError("chunk ranges do not cover all global frames")


def _validate_edges(num_chunks: int, edges: Sequence[Sim3Edge]) -> None:
    for edge in edges:
        if edge.source == edge.target:
            raise ValueError("self edges are not allowed")
        if not (0 <= edge.source < num_chunks and 0 <= edge.target < num_chunks):
            raise ValueError(f"edge index out of range: {edge.source}->{edge.target}")


def _sim3_sequence_finite(transforms: Sequence[Sim3]) -> bool:
    return bool(
        all(
            np.isfinite(transform.scale)
            and np.all(np.isfinite(transform.rotation))
            and np.all(np.isfinite(transform.translation))
            for transform in transforms
        )
    )


def _factorized_v4_export_materialization_safety(
    base_frame_poses: Sequence[Sim3],
    frame_deltas: Sequence[Sim3] | None,
    *,
    translation_scale: float,
) -> dict[str, Any]:
    residual_translation_tolerance = 1e-8 / max(
        abs(float(translation_scale)), 1e-12
    )
    if frame_deltas is None:
        return {
            "gate_pass": False,
            "reason": "delta_unavailable",
            "translation_residual_tolerance": float(residual_translation_tolerance),
            "max_base_translation_ulp": 0.0,
            "max_delta_translation_contribution": 0.0,
            "max_ulp_over_delta_contribution": 0.0,
            "relative_ulp_threshold": 1e-2,
            "unsafe_frames": [],
            "unsafe_frame_count": 0,
        }
    if len(base_frame_poses) != len(frame_deltas):
        return {
            "gate_pass": False,
            "reason": "frame_count_mismatch",
            "base_frame_count": int(len(base_frame_poses)),
            "delta_frame_count": int(len(frame_deltas)),
            "translation_residual_tolerance": float(residual_translation_tolerance),
            "max_base_translation_ulp": 0.0,
            "max_delta_translation_contribution": 0.0,
            "max_ulp_over_delta_contribution": 0.0,
            "relative_ulp_threshold": 1e-2,
            "unsafe_frames": [],
            "unsafe_frame_count": 0,
        }

    export_spacing_max = 0.0
    export_contribution_max = 0.0
    export_spacing_over_contribution_max = 0.0
    export_unsafe_frames: list[int] = []
    for frame_index, (base_pose, delta_pose) in enumerate(
        zip(base_frame_poses, frame_deltas)
    ):
        contribution = base_pose.rotation @ delta_pose.translation
        spacing = np.abs(np.spacing(base_pose.translation))
        contribution_abs = np.abs(contribution)
        nontrivial = contribution_abs > residual_translation_tolerance
        ratio = spacing / np.maximum(contribution_abs, 1e-300)
        unsafe_components = nontrivial & (
            (contribution_abs <= 2.0 * spacing)
            | ((spacing > residual_translation_tolerance) & (ratio > 1e-2))
        )
        export_spacing_max = max(
            export_spacing_max, float(np.max(spacing, initial=0.0))
        )
        export_contribution_max = max(
            export_contribution_max,
            float(np.max(contribution_abs, initial=0.0)),
        )
        export_spacing_over_contribution_max = max(
            export_spacing_over_contribution_max,
            float(np.max(ratio[nontrivial], initial=0.0))
            if np.any(nontrivial)
            else 0.0,
        )
        if np.any(unsafe_components):
            export_unsafe_frames.append(int(frame_index))
    return {
        "gate_pass": not export_unsafe_frames,
        "translation_residual_tolerance": float(residual_translation_tolerance),
        "max_base_translation_ulp": float(export_spacing_max),
        "max_delta_translation_contribution": float(
            export_contribution_max
        ),
        "max_ulp_over_delta_contribution": float(
            export_spacing_over_contribution_max
        ),
        "relative_ulp_threshold": 1e-2,
        "unsafe_frames": export_unsafe_frames[:16],
        "unsafe_frame_count": int(len(export_unsafe_frames)),
    }


def _factorized_v4_left_compose_export_materialization_safety(
    base_frame_poses: Sequence[Sim3],
    left_corrections: Sequence[Sim3] | None,
    *,
    translation_scale: float,
) -> dict[str, Any]:
    residual_translation_tolerance = 1e-8 / max(
        abs(float(translation_scale)), 1e-12
    )
    if left_corrections is None:
        return {
            "gate_pass": False,
            "reason": "left_correction_unavailable",
            "composition": "left_compose_C_then_B",
            "translation_residual_tolerance": float(residual_translation_tolerance),
            "max_base_translation_ulp": 0.0,
            "max_left_correction_translation_contribution": 0.0,
            "max_ulp_over_left_correction_contribution": 0.0,
            "relative_ulp_threshold": 1e-2,
            "unsafe_frames": [],
            "unsafe_frame_count": 0,
        }
    if len(base_frame_poses) != len(left_corrections):
        return {
            "gate_pass": False,
            "reason": "frame_count_mismatch",
            "composition": "left_compose_C_then_B",
            "base_frame_count": int(len(base_frame_poses)),
            "left_correction_count": int(len(left_corrections)),
            "translation_residual_tolerance": float(residual_translation_tolerance),
            "max_base_translation_ulp": 0.0,
            "max_left_correction_translation_contribution": 0.0,
            "max_ulp_over_left_correction_contribution": 0.0,
            "relative_ulp_threshold": 1e-2,
            "unsafe_frames": [],
            "unsafe_frame_count": 0,
        }

    export_spacing_max = 0.0
    export_contribution_max = 0.0
    export_spacing_over_contribution_max = 0.0
    export_unsafe_frames: list[int] = []
    for frame_index, (base_pose, correction) in enumerate(
        zip(base_frame_poses, left_corrections)
    ):
        base_term = correction.rotation @ base_pose.translation
        contribution = correction.translation
        spacing = np.abs(np.spacing(base_term))
        contribution_abs = np.abs(contribution)
        nontrivial = contribution_abs > residual_translation_tolerance
        ratio = spacing / np.maximum(contribution_abs, 1e-300)
        unsafe_components = nontrivial & (
            (contribution_abs <= 2.0 * spacing)
            | ((spacing > residual_translation_tolerance) & (ratio > 1e-2))
        )
        export_spacing_max = max(
            export_spacing_max, float(np.max(spacing, initial=0.0))
        )
        export_contribution_max = max(
            export_contribution_max,
            float(np.max(contribution_abs, initial=0.0)),
        )
        export_spacing_over_contribution_max = max(
            export_spacing_over_contribution_max,
            float(np.max(ratio[nontrivial], initial=0.0))
            if np.any(nontrivial)
            else 0.0,
        )
        if np.any(unsafe_components):
            export_unsafe_frames.append(int(frame_index))
    return {
        "gate_pass": not export_unsafe_frames,
        "composition": "left_compose_C_then_B",
        "translation_residual_tolerance": float(residual_translation_tolerance),
        "max_base_translation_ulp": float(export_spacing_max),
        "max_left_correction_translation_contribution": float(
            export_contribution_max
        ),
        "max_ulp_over_left_correction_contribution": float(
            export_spacing_over_contribution_max
        ),
        "relative_ulp_threshold": 1e-2,
        "unsafe_frames": export_unsafe_frames[:16],
        "unsafe_frame_count": int(len(export_unsafe_frames)),
    }


def _chunk_c2w(chunk: Any) -> np.ndarray:
    _, value = _get_field(chunk, C2W_FIELDS)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"chunk c2w must have shape [S,4,4], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("chunk c2w contains non-finite values")
    return array


def _chunk_points(chunk: Any) -> tuple[str, np.ndarray]:
    name, value = _get_field(chunk, POINT_FIELDS)
    points = np.asarray(value, dtype=np.float64)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError(f"chunk points must have shape (...,3), got {points.shape}")
    return name, points


def _c2w_to_sim3(pose: np.ndarray) -> Sim3:
    rotation = np.asarray(pose[:3, :3], dtype=np.float64)
    u, _, vt = np.linalg.svd(rotation)
    projected = u @ vt
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    return Sim3(rotation=projected, translation=np.asarray(pose[:3, 3], dtype=np.float64))


def _c2w_batch_to_sim3(poses: np.ndarray) -> list[Sim3]:
    """Project one chunk of C2W poses with one batched SVD call."""

    projected, translations = _project_c2w_batch(poses)
    return _validated_sim3_batch(
        np.ones(len(projected), dtype=np.float64), projected, translations
    )


def _project_c2w_batch(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project one C2W batch to SO(3), returning validated-shape components."""

    poses_array = np.asarray(poses, dtype=np.float64)
    if poses_array.ndim != 3 or poses_array.shape[1:] not in {(3, 4), (4, 4)}:
        raise ValueError(
            "batched c2w poses must have shape (N, 3, 4) or (N, 4, 4), "
            f"got {poses_array.shape}"
        )
    rotations = poses_array[:, :3, :3]
    u, _, vt = np.linalg.svd(rotations)
    projected = u @ vt
    reflected = np.linalg.det(projected) < 0.0
    if np.any(reflected):
        u = u.copy()
        u[reflected, :, -1] *= -1.0
        projected = u @ vt
    translations = poses_array[:, :3, 3]
    return projected, translations


def _validated_sim3_batch(
    scales: np.ndarray, rotations: np.ndarray, translations: np.ndarray
) -> list[Sim3]:
    scales, rotations, translations = _validate_sim3_batch_components(
        scales, rotations, translations
    )
    result = []
    for scale, rotation, translation in zip(scales, rotations, translations):
        transform = object.__new__(Sim3)
        object.__setattr__(transform, "scale", float(scale))
        object.__setattr__(transform, "rotation", rotation)
        object.__setattr__(transform, "translation", translation)
        result.append(transform)
    return result


def _validate_sim3_batch_components(
    scales: np.ndarray, rotations: np.ndarray, translations: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scales = np.asarray(scales, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    count = len(scales)
    if scales.shape != (count,):
        raise ValueError("batched scales must have shape (N,)")
    if rotations.shape != (count, 3, 3):
        raise ValueError("batched rotations must have shape (N, 3, 3)")
    if translations.shape != (count, 3):
        raise ValueError("batched translations must have shape (N, 3)")
    if not np.all(np.isfinite(scales)) or np.any(np.abs(scales) < 1e-12):
        raise ValueError("batched scales contain invalid values")
    if not np.all(np.isfinite(rotations)):
        raise ValueError("batched projected rotations contain non-finite values")
    if not np.all(np.isfinite(translations)):
        raise ValueError("batched translations contain non-finite values")
    return scales, rotations, translations


def _sim3_to_c2w(pose: Sim3) -> np.ndarray:
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = pose.rotation
    output[:3, 3] = pose.translation
    return output


def _initial_frame_poses(
    local_poses: Sequence[np.ndarray],
    ranges: Sequence[tuple[int, int]],
    alignments: Sequence[Sim3],
    num_frames: int,
) -> list[Sim3]:
    result: list[Sim3 | None] = [None] * num_frames
    for chunk_index, poses in enumerate(local_poses):
        start, end = ranges[chunk_index]
        for local_index, global_index in enumerate(range(start, end)):
            candidate = _c2w_to_sim3(
                alignments[chunk_index].apply_c2w(poses[local_index])
            )
            if result[global_index] is None:
                result[global_index] = candidate
    if any(item is None for item in result):
        raise ValueError("could not initialize every global frame pose")
    return [item for item in result if item is not None]


def _aligned_local_frame_pose_components(
    local_poses: Sequence[np.ndarray], alignments: Sequence[Sim3]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Transform frozen-coarse poses once without materializing duplicate Sim3s."""

    if len(local_poses) != len(alignments):
        raise ValueError("local pose/alignment counts do not match")
    lengths: list[int] = []
    projected_rotations: list[np.ndarray] = []
    projected_translations: list[np.ndarray] = []
    for poses, alignment in zip(local_poses, alignments):
        rotation, translation = _project_c2w_batch(
            alignment.apply_c2w(np.asarray(poses))
        )
        lengths.append(len(rotation))
        projected_rotations.append(rotation)
        projected_translations.append(translation)
    if not lengths:
        return []
    rotations = np.concatenate(projected_rotations, axis=0)
    translations = np.concatenate(projected_translations, axis=0)
    _, rotations, translations = _validate_sim3_batch_components(
        np.ones(len(rotations), dtype=np.float64), rotations, translations
    )
    result: list[tuple[np.ndarray, np.ndarray]] = []
    offset = 0
    for length in lengths:
        result.append(
            (
                rotations[offset : offset + length],
                translations[offset : offset + length],
            )
        )
        offset += length
    return result


def _initial_frame_poses_from_aligned_components(
    aligned_local_components: Sequence[tuple[np.ndarray, np.ndarray]],
    ranges: Sequence[tuple[int, int]],
    num_frames: int,
) -> list[Sim3]:
    rotations = np.empty((num_frames, 3, 3), dtype=np.float64)
    translations = np.empty((num_frames, 3), dtype=np.float64)
    assigned = np.zeros(num_frames, dtype=bool)
    for (local_rotations, local_translations), (start, end) in zip(
        aligned_local_components, ranges
    ):
        if len(local_rotations) != end - start or len(local_translations) != end - start:
            raise ValueError("aligned local pose count does not match chunk range")
        global_indices = np.arange(start, end, dtype=np.int64)
        take = ~assigned[global_indices]
        selected = global_indices[take]
        rotations[selected] = local_rotations[take]
        translations[selected] = local_translations[take]
        assigned[selected] = True
    if not np.all(assigned):
        raise ValueError("could not initialize every global frame pose")
    return _validated_sim3_batch(
        np.ones(num_frames, dtype=np.float64), rotations, translations
    )


def _consensus_frame_poses(
    local_poses: Sequence[np.ndarray],
    ranges: Sequence[tuple[int, int]],
    alignments: Sequence[Sim3],
    num_frames: int,
    *,
    selected_frames: set[int] | None = None,
    fallback_frames: Sequence[Sim3] | None = None,
) -> list[Sim3]:
    if selected_frames is not None:
        if fallback_frames is None or len(fallback_frames) != num_frames:
            raise ValueError("sparse consensus requires one fallback pose per frame")
        if any(frame < 0 or frame >= num_frames for frame in selected_frames):
            raise ValueError("selected consensus frame is out of bounds")
    occurrences: list[list[Sim3]] = [[] for _ in range(num_frames)]
    for chunk_index, (poses, (start, end)) in enumerate(zip(local_poses, ranges)):
        for local_index, frame in enumerate(range(start, end)):
            if selected_frames is not None and frame not in selected_frames:
                continue
            occurrences[frame].append(
                _c2w_to_sim3(alignments[chunk_index].apply_c2w(poses[local_index]))
            )
    result = []
    for frame, candidates in enumerate(occurrences):
        if not candidates:
            if selected_frames is not None:
                result.append(fallback_frames[frame])
                continue
            raise ValueError(f"frame {frame} has no chunk occurrence")
        if len(candidates) == 1:
            result.append(candidates[0])
            continue
        rotations = np.stack([item.rotation for item in candidates], axis=0)
        translations = np.stack([item.translation for item in candidates], axis=0)
        result.append(
            Sim3(
                rotation=Rotation.from_matrix(rotations).mean().as_matrix(),
                translation=np.mean(translations, axis=0),
            )
        )
    return result


def _sparse_keyframe_indices(num_frames: int, stride: int) -> list[int]:
    if stride < 1:
        raise ValueError("stride must be positive")
    if num_frames < 1:
        return []
    keyframes = list(range(0, num_frames, stride))
    if keyframes[-1] != num_frames - 1:
        keyframes.append(num_frames - 1)
    return keyframes


def _joint_frame_subspace_prolongation(
    num_frames: int,
    starts_ends: Sequence[tuple[int, int]],
    *,
    stride: int,
) -> tuple[list[int], Any, np.ndarray]:
    """Build a column-normalized piecewise-linear frame correction basis.

    The fixed frame 0 is present only as a zero-correction interpolation
    endpoint.  Every other control frame owns one six-dimensional SE(3)
    tangent block.  Chunk endpoints are retained so the reduced stage can
    express corrections at every coarse/fine interface while still evaluating
    the unchanged full residual graph.
    """

    from scipy import sparse

    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if stride < 1:
        raise ValueError("stride must be positive")
    selected = set(_sparse_keyframe_indices(num_frames, stride))
    for start, end in starts_ends:
        selected.add(int(start))
        selected.add(int(end) - 1)
    selected.add(0)
    selected.add(num_frames - 1)
    control_frames = sorted(selected)
    if control_frames[0] != 0 or control_frames[-1] != num_frames - 1:
        raise RuntimeError("frame subspace controls must cover both endpoints")

    free_controls = control_frames[1:]
    control_position = {
        frame: position for position, frame in enumerate(free_controls)
    }
    row_indices: list[int] = []
    column_indices: list[int] = []
    weights: list[float] = []
    interval = 0
    for frame in range(1, num_frames):
        while control_frames[interval + 1] < frame:
            interval += 1
        left = control_frames[interval]
        right = control_frames[interval + 1]
        ratio = float(frame - left) / float(right - left)
        if left != 0 and ratio < 1.0:
            row_indices.append(frame - 1)
            column_indices.append(control_position[left])
            weights.append(1.0 - ratio)
        row_indices.append(frame - 1)
        column_indices.append(control_position[right])
        weights.append(ratio)

    scalar = sparse.csr_matrix(
        (weights, (row_indices, column_indices)),
        shape=(num_frames - 1, len(free_controls)),
        dtype=np.float64,
    )
    scalar_column_norms = np.sqrt(
        np.asarray(scalar.multiply(scalar).sum(axis=0), dtype=np.float64).reshape(-1)
    )
    if np.any(~np.isfinite(scalar_column_norms)) or np.any(
        scalar_column_norms <= 0.0
    ):
        raise RuntimeError("frame subspace contains an empty or nonfinite control")
    normalized_scalar = scalar @ sparse.diags(1.0 / scalar_column_norms)
    prolongation = sparse.kron(
        normalized_scalar,
        sparse.eye(6, format="csr", dtype=np.float64),
        format="csr",
    )
    return free_controls, prolongation, scalar_column_norms


def _joint_frame_sparse_exact_basis(
    num_frames: int,
    starts_ends: Sequence[tuple[int, int]],
    *,
    stride: int,
) -> dict[str, Any]:
    """Build a square sparse frame basis with P64 controls first."""

    from scipy import sparse

    control_frames, frame_prolongation, _ = _joint_frame_subspace_prolongation(
        num_frames,
        starts_ends,
        stride=stride,
    )
    scalar_prolongation = frame_prolongation[0::6, 0::6].tocsr()
    scalar_rows, scalar_controls = scalar_prolongation.shape
    if scalar_rows != num_frames - 1:
        raise RuntimeError("sparse exact basis scalar row mismatch")
    if scalar_controls != len(control_frames):
        raise RuntimeError("sparse exact basis control count mismatch")
    pivot_rows = np.asarray(
        [int(frame) - 1 for frame in control_frames], dtype=np.int64
    )
    if (
        pivot_rows.size != scalar_controls
        or np.any(pivot_rows < 0)
        or np.any(pivot_rows >= scalar_rows)
    ):
        raise RuntimeError("sparse exact basis pivot row mismatch")
    pivot_values = np.asarray(
        scalar_prolongation[pivot_rows, np.arange(scalar_controls)]
    ).reshape(-1)
    if np.any(np.abs(pivot_values) <= 1e-12):
        raise RuntimeError("sparse exact basis pivot is too small")

    pivot_set = {int(row) for row in pivot_rows}
    detail_rows = [
        row for row in range(scalar_rows) if row not in pivot_set
    ]
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for detail_column, detail_row in enumerate(detail_rows):
        row_indices.append(detail_row)
        column_indices.append(detail_column)
        values.append(1.0)
        row_start = scalar_prolongation.indptr[detail_row]
        row_stop = scalar_prolongation.indptr[detail_row + 1]
        for control_column, weight in zip(
            scalar_prolongation.indices[row_start:row_stop],
            scalar_prolongation.data[row_start:row_stop],
            strict=True,
        ):
            pivot_row = int(pivot_rows[control_column])
            pivot_value = float(pivot_values[control_column])
            row_indices.append(pivot_row)
            column_indices.append(detail_column)
            values.append(-float(weight) / pivot_value)
    scalar_detail_unnormalized = sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(scalar_rows, len(detail_rows)),
        dtype=np.float64,
    )
    if scalar_detail_unnormalized.shape[1] > 0:
        detail_column_norms = np.sqrt(
            np.asarray(
                scalar_detail_unnormalized.multiply(
                    scalar_detail_unnormalized
                ).sum(axis=0),
                dtype=np.float64,
            ).reshape(-1)
        )
        if np.any(~np.isfinite(detail_column_norms)) or np.any(
            detail_column_norms <= 0.0
        ):
            raise RuntimeError("sparse exact basis detail norm invalid")
        scalar_detail = (
            scalar_detail_unnormalized
            @ sparse.diags(1.0 / detail_column_norms)
        ).tocsr()
        detail_nnz = np.diff(scalar_detail_unnormalized.tocsc().indptr)
    else:
        detail_column_norms = np.zeros(0, dtype=np.float64)
        scalar_detail = scalar_detail_unnormalized
        detail_nnz = np.zeros(0, dtype=np.int64)

    scalar_transform = sparse.hstack(
        (scalar_prolongation, scalar_detail), format="csr"
    )
    if scalar_transform.shape != (scalar_rows, scalar_rows):
        raise RuntimeError("sparse exact basis transform must be square")
    orthogonality = (scalar_prolongation.T @ scalar_detail).tocoo()
    orthogonality_inf = (
        float(np.max(np.abs(orthogonality.data)))
        if orthogonality.nnz
        else 0.0
    )
    reference_inf = max(
        float(
            np.max(np.abs((scalar_prolongation.T @ scalar_prolongation).data))
        ),
        1e-12,
    )
    orthogonality_relative_inf = orthogonality_inf / reference_inf
    if orthogonality_relative_inf > 1e-10:
        raise RuntimeError("sparse exact basis P^T B orthogonality failed")
    frame_transform = sparse.kron(
        scalar_transform,
        sparse.eye(6, format="csr", dtype=np.float64),
        format="csr",
    )
    dense_scalar_condition = (
        float(np.linalg.cond(scalar_transform.toarray()))
        if scalar_rows <= 128
        else None
    )

    def _finite_summary(values: np.ndarray) -> dict[str, float | None]:
        if values.size == 0:
            return {"min": None, "median": None, "p95": None, "max": None}
        return {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }

    return {
        "control_frames": control_frames,
        "detail_rows": detail_rows,
        "pivot_rows": pivot_rows.tolist(),
        "pivot_values": pivot_values,
        "scalar_prolongation": scalar_prolongation,
        "scalar_detail": scalar_detail,
        "scalar_transform": scalar_transform,
        "frame_transform": frame_transform,
        "orthogonality_inf": orthogonality_inf,
        "orthogonality_relative_inf": float(orthogonality_relative_inf),
        "detail_column_norms_summary": _finite_summary(detail_column_norms),
        "detail_nnz_summary": _finite_summary(detail_nnz.astype(np.float64)),
        "dense_scalar_condition": dense_scalar_condition,
    }


def _interpolate_sparse_frame_corrections(
    initial_frames: Sequence[Sim3],
    consensus_frames: Sequence[Sim3],
    *,
    stride: int,
) -> list[Sim3]:
    if len(initial_frames) != len(consensus_frames):
        raise ValueError("initial and consensus frame counts must match")
    if stride < 1:
        raise ValueError("stride must be positive")
    num_frames = len(initial_frames)
    if num_frames <= 1 or stride == 1:
        return list(consensus_frames)

    keyframes = _sparse_keyframe_indices(num_frames, stride)
    corrections = [
        consensus_frames[index].compose(initial_frames[index].inverse())
        for index in keyframes
    ]
    result: list[Sim3 | None] = [None] * num_frames
    for interval in range(len(keyframes) - 1):
        start, end = keyframes[interval], keyframes[interval + 1]
        start_correction, end_correction = corrections[interval], corrections[interval + 1]
        slerp = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(
                np.stack([start_correction.rotation, end_correction.rotation], axis=0)
            ),
        )
        for frame in range(start, end + 1):
            ratio = 0.0 if end == start else float(frame - start) / float(end - start)
            correction = Sim3(
                rotation=slerp([ratio]).as_matrix()[0],
                translation=(
                    (1.0 - ratio) * start_correction.translation
                    + ratio * end_correction.translation
                ),
            )
            result[frame] = correction.compose(initial_frames[frame])
    if any(item is None for item in result):
        raise RuntimeError("sparse correction interpolation did not cover every frame")
    return [item for item in result if item is not None]


def _chunk_scale_trust_radii(
    num_chunks: int,
    edges: Sequence[Sim3Edge],
    alignments: Sequence[Sim3],
    *,
    cfg: HierarchicalPoseGraphConfig,
) -> np.ndarray:
    """Per-chunk log-scale trust radii from coarse edge consistency.

    For every Sim(3) edge the warm-start scale residual is
    ``|log(s_source / (s_edge * s_target))|`` (the same convention as the
    coarse residual). A chunk whose incident edges disagree more keeps a
    wider optimizer bound; a chunk whose edges already agree is protected
    with a tighter bound. Radii are clipped to
    ``[min_ratio, max_ratio] * max_log_scale_delta``. No ground truth enters.
    """

    residuals_per_chunk: list[list[float]] = [[] for _ in range(num_chunks)]
    for edge in edges:
        residual = abs(
            np.log(alignments[edge.source].scale)
            - np.log(edge.transform.scale)
            - np.log(alignments[edge.target].scale)
        )
        residuals_per_chunk[edge.source].append(residual)
        residuals_per_chunk[edge.target].append(residual)
    base = float(cfg.max_log_scale_delta)
    lower = cfg.chunk_scale_trust_min_ratio * base
    upper = cfg.chunk_scale_trust_max_ratio * base
    radii = np.full(num_chunks, base, dtype=np.float64)
    for chunk, residuals in enumerate(residuals_per_chunk):
        if residuals:
            radii[chunk] = float(
                np.clip(
                    cfg.chunk_scale_trust_gain * float(np.median(residuals)),
                    lower,
                    upper,
                )
            )
    return radii


def _select_keyframes(
    initial_frame_poses: Sequence[Sim3],
    starts_ends: Sequence[tuple[int, int]],
    num_frames: int,
    *,
    cfg: HierarchicalPoseGraphConfig,
) -> list[int]:
    """Pick fine-graph keyframes for a long sequence.

    Selection is either uniform in frame index or equalized in accumulated
    motion of the coarse-aligned trajectory (a non-GT signal). Chunk endpoint
    and overlap coverage is then enforced so every chunk contributes at least
    one fine edge and adjacent chunks keep shared prior frames.
    """

    target = _sparse_keyframe_indices(num_frames, cfg.keyframe_target_stride)
    if cfg.keyframe_selection == "uniform":
        selected = set(target)
    else:
        rotations = np.stack(
            [pose.rotation for pose in initial_frame_poses], axis=0
        )
        translations = np.stack(
            [pose.translation for pose in initial_frame_poses], axis=0
        )
        relative = np.einsum("nij,nik->njk", rotations[:-1], rotations[1:])
        angles = np.linalg.norm(
            Rotation.from_matrix(relative).as_rotvec(), axis=1
        )
        offsets = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        angle_scale = float(np.mean(angles)) or 1.0
        offset_scale = float(np.mean(offsets)) or 1.0
        steps = angles / angle_scale + offsets / offset_scale
        cumulative = np.concatenate(([0.0], np.cumsum(steps)))
        quantiles = np.linspace(0.0, cumulative[-1], len(target))
        picks = np.searchsorted(cumulative, quantiles, side="left")
        selected = set(int(index) for index in np.clip(picks, 0, num_frames - 1))
    selected.add(0)
    selected.add(num_frames - 1)

    for start, end in starts_ends:
        selected.add(start)
        selected.add(end - 1)
    ordered_ranges = sorted(starts_ends)
    for (_, end_a), (start_b, _) in zip(ordered_ranges, ordered_ranges[1:]):
        overlap = range(start_b, end_a)
        if len(overlap) == 0:
            continue
        required = min(cfg.keyframe_min_overlap_shared, len(overlap))
        if sum(1 for frame in overlap if frame in selected) >= required:
            continue
        positions = np.linspace(0, len(overlap) - 1, required).astype(int)
        for position in positions:
            selected.add(overlap[position])
    return sorted(selected)


def _interpolate_keyframe_corrections(
    initial_frames: Sequence[Sim3],
    keyframes: Sequence[int],
    solved_keyframe_poses: Sequence[Sim3],
) -> list[Sim3]:
    """Propagate solved keyframe corrections to every frame.

    Mirrors :func:`_interpolate_sparse_frame_corrections` but supports
    arbitrary (non-uniform) keyframe indices.
    """

    frame_poses, _ = _interpolate_keyframe_corrections_with_left_corrections(
        initial_frames,
        keyframes,
        solved_keyframe_poses,
    )
    return frame_poses


def _interpolate_keyframe_corrections_with_left_corrections(
    initial_frames: Sequence[Sim3],
    keyframes: Sequence[int],
    solved_keyframe_poses: Sequence[Sim3],
) -> tuple[list[Sim3], list[Sim3]]:
    """Return materialized frames and the pre-materialization left corrections."""

    if len(keyframes) != len(solved_keyframe_poses):
        raise ValueError("keyframe and solved pose counts must match")
    if not keyframes or keyframes[0] != 0 or keyframes[-1] != len(initial_frames) - 1:
        raise ValueError("keyframes must start at 0 and end at the last frame")
    corrections = [
        solved.compose(initial_frames[index].inverse())
        for index, solved in zip(keyframes, solved_keyframe_poses)
    ]
    num_frames = len(initial_frames)
    result: list[Sim3 | None] = [None] * num_frames
    interpolated_corrections: list[Sim3 | None] = [None] * num_frames
    for interval in range(len(keyframes) - 1):
        start, end = keyframes[interval], keyframes[interval + 1]
        start_correction = corrections[interval]
        end_correction = corrections[interval + 1]
        slerp = Slerp(
            [0.0, 1.0],
            Rotation.from_matrix(
                np.stack(
                    [start_correction.rotation, end_correction.rotation], axis=0
                )
            ),
        )
        for frame in range(start, end + 1):
            ratio = 0.0 if end == start else float(frame - start) / float(end - start)
            correction = Sim3(
                rotation=slerp([ratio]).as_matrix()[0],
                translation=(
                    (1.0 - ratio) * start_correction.translation
                    + ratio * end_correction.translation
                ),
            )
            interpolated_corrections[frame] = correction
            result[frame] = correction.compose(initial_frames[frame])
    if any(item is None for item in result) or any(
        item is None for item in interpolated_corrections
    ):
        raise RuntimeError("keyframe correction interpolation missed a frame")
    return (
        [item for item in result if item is not None],
        [item for item in interpolated_corrections if item is not None],
    )


def _factorized_v4_keyframe_continuation_budgets(
    max_nfev: int,
    budget_fractions: Sequence[float]
    | None = None,
) -> list[int]:
    fractions = (
        _FACTORIZED_V4_KEYFRAME_CONTINUATION_BUDGET_FRACTIONS
        if budget_fractions is None
        else tuple(float(item) for item in budget_fractions)
    )
    if not fractions:
        raise RuntimeError("factorized v4 keyframe continuation schedule mismatch")
    total = int(max_nfev)
    if total < len(fractions):
        raise ValueError("max_nfev is too small for keyframe continuation")
    budgets: list[int] = []
    remaining = total
    for index, fraction in enumerate(fractions):
        if index == len(fractions) - 1:
            budget = remaining
        else:
            levels_left = len(fractions) - index - 1
            proposed = max(1, int(np.floor(total * float(fraction))))
            budget = min(proposed, remaining - levels_left)
        budgets.append(int(budget))
        remaining -= int(budget)
    return budgets


def _factorized_v4_keyframe_level_nonincrease(
    initial_cost: float, final_cost: float
) -> bool:
    if not np.isfinite(initial_cost) or not np.isfinite(final_cost):
        return False
    return bool(final_cost <= initial_cost + max(1e-12, 1e-10 * abs(initial_cost)))


def _factorized_v4_left_correction_to_right_delta(
    base_pose: Sim3, correction: Sim3
) -> Sim3:
    if not np.isclose(correction.scale, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(
            "factorized_frame_chart_residual_v4 keyframe warm-start "
            "requires SE3 left corrections"
        )
    if not np.isclose(base_pose.scale, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(
            "factorized_frame_chart_residual_v4 keyframe warm-start "
            "requires SE3 base frame charts"
        )
    delta_rotation = base_pose.rotation.T @ correction.rotation @ base_pose.rotation
    delta_translation = base_pose.rotation.T @ (
        (correction.rotation - np.eye(3, dtype=np.float64))
        @ base_pose.translation
        + correction.translation
    )
    return Sim3(
        scale=1.0,
        rotation=delta_rotation,
        translation=delta_translation,
    )


def _factorized_v4_initial_delta_parameters(
    initial_deltas: Sequence[Sim3],
    num_frames: int,
) -> np.ndarray:
    deltas = list(initial_deltas)
    if len(deltas) != num_frames:
        raise ValueError(
            "factorized_frame_chart_residual_v4_initial_frame_deltas length "
            "must match num_frames"
        )
    first_delta = deltas[0]
    if (
        not np.isclose(first_delta.scale, 1.0, rtol=0.0, atol=1e-12)
        or not np.allclose(
            first_delta.rotation, np.eye(3), rtol=0.0, atol=1e-12
        )
        or not np.allclose(
            first_delta.translation, np.zeros(3), rtol=0.0, atol=1e-12
        )
    ):
        raise ValueError(
            "factorized_frame_chart_residual_v4 initial frame0 delta "
            "must be identity"
        )
    if any(
        not np.isclose(delta.scale, 1.0, rtol=0.0, atol=1e-10)
        for delta in deltas
    ):
        raise ValueError(
            "factorized_frame_chart_residual_v4 initial frame deltas "
            "must be SE3"
        )
    if num_frames <= 1:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate([_se3_to_parameters(delta) for delta in deltas[1:]])


def _factorized_v4_keyframe_initial_deltas_from_left_corrections(
    base_frames: Sequence[Sim3],
    keyframes: Sequence[int],
    left_corrections: Sequence[Sim3] | None,
) -> list[Sim3] | None:
    if left_corrections is None:
        return None
    if len(left_corrections) != len(base_frames):
        raise ValueError("left correction count must match full frame count")
    deltas = [Sim3.identity()]
    for global_frame in keyframes[1:]:
        deltas.append(
            _factorized_v4_left_correction_to_right_delta(
                base_frames[global_frame],
                left_corrections[global_frame],
            )
        )
    return deltas


def _solve_keyframe_fine_graph_level(
    local_poses: Sequence[np.ndarray],
    starts_ends: Sequence[tuple[int, int]],
    edges: Sequence[Sim3Edge],
    alignments: list[Sim3],
    initial_frame_poses: Sequence[Sim3],
    num_frames: int,
    *,
    cfg: HierarchicalPoseGraphConfig,
    target_stride: int,
    max_nfev: int,
    warm_left_corrections: Sequence[Sim3] | None = None,
    objective_projection_alpha_grid: Sequence[float] | None = None,
    level_index: int = 0,
) -> dict[str, Any]:
    level_cfg_for_selection = _dataclass_replace(
        cfg,
        keyframe_target_stride=int(target_stride),
    )
    keyframes = _select_keyframes(
        initial_frame_poses, starts_ends, num_frames, cfg=level_cfg_for_selection
    )
    position_of = {frame: position for position, frame in enumerate(keyframes)}
    reduced_chunks: list[dict[str, np.ndarray]] = []
    reduced_ranges: list[tuple[int, int]] = []
    for poses, (start, end) in zip(local_poses, starts_ends):
        chunk_keyframes = [
            frame for frame in keyframes if start <= frame < end
        ]
        if len(chunk_keyframes) < 2:
            raise RuntimeError(
                "keyframe selection left a chunk with fewer than two keyframes"
            )
        reduced_chunks.append(
            {
                "extrinsic_c2w": np.stack(
                    [poses[frame - start] for frame in chunk_keyframes], axis=0
                )
            }
        )
        first = position_of[chunk_keyframes[0]]
        last = position_of[chunk_keyframes[-1]]
        if last - first + 1 != len(chunk_keyframes):
            raise RuntimeError("chunk keyframes are not contiguous in keyframe order")
        reduced_ranges.append((first, last + 1))

    warm_initial_deltas = None
    warm_initial_delta_candidates = None
    if cfg.factorized_frame_chart_residual_v4 and warm_left_corrections is not None:
        if objective_projection_alpha_grid is None:
            warm_initial_deltas = (
                _factorized_v4_keyframe_initial_deltas_from_left_corrections(
                    initial_frame_poses,
                    keyframes,
                    warm_left_corrections,
                )
            )
        else:
            warm_initial_delta_candidates = []
            for alpha in objective_projection_alpha_grid:
                scaled_corrections = [
                    _scaled_se3_correction(correction, float(alpha))
                    for correction in warm_left_corrections
                ]
                deltas = (
                    _factorized_v4_keyframe_initial_deltas_from_left_corrections(
                        initial_frame_poses,
                        keyframes,
                        scaled_corrections,
                    )
                )
                if deltas is None:
                    raise RuntimeError("scaled warm-start deltas are unavailable")
                warm_initial_delta_candidates.append((float(alpha), tuple(deltas)))
    reduced_config = _dataclass_replace(
        cfg,
        long_sequence_mode="overlap_consensus",
        max_frames_per_solve=max(cfg.max_frames_per_solve, len(keyframes)),
        keyframe_target_stride=int(target_stride),
        max_nfev=int(max_nfev),
        factorized_frame_chart_residual_v4_keyframe_continuation=False,
        factorized_frame_chart_residual_v4_keyframe_continuation_objective_projection=False,
        factorized_frame_chart_residual_v4_initial_frame_deltas=(
            None if warm_initial_deltas is None else tuple(warm_initial_deltas)
        ),
        factorized_frame_chart_residual_v4_initial_frame_delta_candidates=(
            None
            if warm_initial_delta_candidates is None
            else tuple(warm_initial_delta_candidates)
        ),
    )
    reduced = build_hierarchical_basis(
        reduced_chunks,
        reduced_ranges,
        edges,
        alignments,
        config=reduced_config,
    )
    coarse = list(
        alignments if cfg.freeze_coarse_alignments else reduced.coarse_alignments
    )
    interpolation_initial = (
        list(initial_frame_poses)
        if cfg.freeze_coarse_alignments
        else _initial_frame_poses(local_poses, starts_ends, coarse, num_frames)
    )
    frame_poses, left_corrections = (
        _interpolate_keyframe_corrections_with_left_corrections(
            interpolation_initial, keyframes, reduced.frame_poses
        )
    )
    initial_cost = float(reduced.diagnostics.get("initial_cost", np.inf))
    final_cost = float(reduced.diagnostics.get("final_cost", np.inf))
    reduced_v4 = reduced.diagnostics.get("factorized_frame_chart_residual_v4", {})
    initialization = (
        reduced_v4.get("initialization")
        if isinstance(reduced_v4, dict)
        else None
    )
    public_zero_cost = (
        float(initialization["public_zero_cost"])
        if isinstance(initialization, dict)
        and "public_zero_cost" in initialization
        else initial_cost
    )
    public_zero_robust_cost = (
        float(initialization["public_zero_robust_cost"])
        if isinstance(initialization, dict)
        and "public_zero_robust_cost" in initialization
        else float(reduced.diagnostics.get("initial_robust_cost", np.inf))
    )
    warm_initial_cost = (
        float(initialization["warm_initial_cost"])
        if isinstance(initialization, dict)
        and "warm_initial_cost" in initialization
        else initial_cost
    )
    warm_initial_robust_cost = (
        float(initialization["warm_initial_robust_cost"])
        if isinstance(initialization, dict)
        and "warm_initial_robust_cost" in initialization
        else public_zero_robust_cost
    )
    warm_start_accepted = (
        bool(initialization["warm_start_accepted"])
        if isinstance(initialization, dict)
        and "warm_start_accepted" in initialization
        else warm_left_corrections is None
    )
    nonincrease = _factorized_v4_keyframe_level_nonincrease(
        public_zero_cost, final_cost
    )
    final_robust_cost = float(
        reduced.diagnostics.get(
            "final_robust_cost",
            reduced.diagnostics.get("scipy_cost", np.inf),
        )
    )
    robust_nonincrease = _factorized_v4_keyframe_level_nonincrease(
        public_zero_robust_cost,
        final_robust_cost,
    )
    num_jacobian_evaluations = reduced.diagnostics.get(
        "num_jacobian_evaluations"
    )
    termination_status = reduced.diagnostics.get("termination_status")
    strict_success = bool(
        reduced.diagnostics.get("success", False)
        and termination_status is not None
        and int(termination_status) in {1, 2, 3, 4}
    )
    diagnostics = {
        "level_index": int(level_index),
        "target_stride": int(target_stride),
        "max_nfev": int(max_nfev),
        "num_keyframes": int(len(keyframes)),
        "keyframes": [int(frame) for frame in keyframes],
        "warm_start_source": (
            "accepted_previous_left_corrections"
            if warm_left_corrections is not None
            else "zero_identity"
        ),
        "warm_start_frame0_identity": True,
        "public_zero_cost": public_zero_cost,
        "public_zero_robust_cost": public_zero_robust_cost,
        "warm_initial_cost": warm_initial_cost,
        "warm_initial_robust_cost": warm_initial_robust_cost,
        "warm_start_accepted": warm_start_accepted,
        "warm_rejected_reason": (
            initialization.get("warm_rejected_reason")
            if isinstance(initialization, dict)
            else None
        ),
        "objective_projection": (
            initialization.get("objective_projection")
            if isinstance(initialization, dict)
            else None
        ),
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "final_robust_cost": final_robust_cost,
        "final_cost_nonincrease_vs_public_zero": nonincrease,
        "final_robust_cost_nonincrease_vs_public_zero": robust_nonincrease,
        "objective_nonincrease": nonincrease,
        "success": bool(reduced.diagnostics.get("success", False)),
        "strict_success_status": strict_success,
        "status": reduced.diagnostics.get("status"),
        "termination_status": termination_status,
        "num_function_evaluations": int(
            reduced.diagnostics.get("num_function_evaluations", 0)
        ),
        "num_jacobian_evaluations": (
            None
            if num_jacobian_evaluations is None
            else int(num_jacobian_evaluations)
        ),
    }
    return {
        "keyframes": keyframes,
        "reduced": reduced,
        "coarse": coarse,
        "interpolation_initial": interpolation_initial,
        "frame_poses": frame_poses,
        "left_corrections": left_corrections,
        "diagnostics": diagnostics,
    }


def _solve_keyframe_fine_graph(
    local_poses: Sequence[np.ndarray],
    starts_ends: Sequence[tuple[int, int]],
    edges: Sequence[Sim3Edge],
    alignments: list[Sim3],
    initial_frame_poses: Sequence[Sim3],
    num_frames: int,
    *,
    cfg: HierarchicalPoseGraphConfig,
) -> HierarchicalAlphaBasis:
    """Solve the hierarchical graph on a keyframe subproblem, then interpolate.

    Frozen path: replaces unoptimized ``overlap_consensus`` for long sequences.
    Joint path: keeps chunk Sim(3) free while reducing frame variables; non-
    keyframe corrections are interpolated against the optimized coarse layout.
    """

    continuation_diagnostics = None
    if cfg.factorized_frame_chart_residual_v4_keyframe_continuation:
        objective_projection_enabled = bool(
            cfg.factorized_frame_chart_residual_v4_keyframe_continuation_objective_projection
        )
        stride_multipliers = (
            _FACTORIZED_V4_KEYFRAME_CONTINUATION_V2_STRIDE_MULTIPLIERS
            if objective_projection_enabled
            else _FACTORIZED_V4_KEYFRAME_CONTINUATION_STRIDE_MULTIPLIERS
        )
        budget_fractions = (
            _FACTORIZED_V4_KEYFRAME_CONTINUATION_V2_BUDGET_FRACTIONS
            if objective_projection_enabled
            else _FACTORIZED_V4_KEYFRAME_CONTINUATION_BUDGET_FRACTIONS
        )
        budgets = _factorized_v4_keyframe_continuation_budgets(
            cfg.max_nfev,
            budget_fractions,
        )
        warm_left_corrections: Sequence[Sim3] | None = None
        levels: list[dict[str, Any]] = []
        final_level: dict[str, Any] | None = None
        final_level_selected = False
        for level_index, (multiplier, budget) in enumerate(
            zip(stride_multipliers, budgets)
        ):
            target_stride = int(cfg.keyframe_target_stride) * int(multiplier)
            level = _solve_keyframe_fine_graph_level(
                local_poses,
                starts_ends,
                edges,
                alignments,
                initial_frame_poses,
                num_frames,
                cfg=cfg,
                target_stride=target_stride,
                max_nfev=budget,
                warm_left_corrections=warm_left_corrections,
                objective_projection_alpha_grid=(
                    _FACTORIZED_V4_KEYFRAME_CONTINUATION_V2_ALPHA_GRID
                    if objective_projection_enabled
                    and warm_left_corrections is not None
                    else None
                ),
                level_index=level_index,
            )
            level_diagnostics = dict(level["diagnostics"])
            accepted_for_warm_start = bool(
                level_diagnostics["final_cost_nonincrease_vs_public_zero"]
                and level_diagnostics[
                    "final_robust_cost_nonincrease_vs_public_zero"
                ]
            )
            is_final = level_index == len(budgets) - 1
            final_success = bool(
                level_diagnostics["strict_success_status"]
                if objective_projection_enabled
                else level_diagnostics["success"]
            )
            selected = bool(
                accepted_for_warm_start and (final_success if is_final else True)
            )
            level_diagnostics["accepted_for_next_warm_start"] = (
                accepted_for_warm_start if not is_final else False
            )
            level_diagnostics["selected_final_level"] = bool(is_final and selected)
            level_diagnostics["original_target_stride"] = int(
                cfg.keyframe_target_stride
            )
            levels.append(level_diagnostics)
            if accepted_for_warm_start and not is_final:
                warm_left_corrections = level["left_corrections"]
            elif not is_final:
                warm_left_corrections = None
            if is_final:
                final_level = level
                final_level_selected = selected
        if final_level is None:
            raise RuntimeError("keyframe continuation did not run a final level")
        final_reduced = final_level["reduced"]
        reduced = final_reduced
        coarse = list(final_level["coarse"])
        interpolation_initial = list(final_level["interpolation_initial"])
        if final_level_selected:
            frame_poses = list(final_level["frame_poses"])
            outer_left_corrections = list(final_level["left_corrections"])
        else:
            frame_poses = list(interpolation_initial)
            outer_left_corrections = [Sim3.identity() for _ in range(num_frames)]
        continuation_diagnostics = {
            "schema": (
                "factorized_frame_chart_residual_v4_keyframe_continuation_v2"
                if objective_projection_enabled
                else "factorized_frame_chart_residual_v4_keyframe_continuation_v1"
            ),
            "enabled": True,
            "objective_projection_enabled": objective_projection_enabled,
            "stride_multipliers": [
                int(item) for item in stride_multipliers
            ],
            "budget_fractions": [
                float(item) for item in budget_fractions
            ],
            "alpha_grid": (
                [
                    float(item)
                    for item in _FACTORIZED_V4_KEYFRAME_CONTINUATION_V2_ALPHA_GRID
                ]
                if objective_projection_enabled
                else None
            ),
            "level_budgets": [int(item) for item in budgets],
            "budget_total": int(sum(budgets)),
            "max_nfev": int(cfg.max_nfev),
            "total_nfev": int(
                sum(level["num_function_evaluations"] for level in levels)
            ),
            "total_njev": (
                None
                if any(
                    level["num_jacobian_evaluations"] is None for level in levels
                )
                else int(
                    sum(
                        int(level["num_jacobian_evaluations"])
                        for level in levels
                    )
                )
            ),
            "levels": levels,
            "final_level_index": int(len(levels) - 1),
            "final_target_stride": int(levels[-1]["target_stride"]),
            "original_target_stride": int(cfg.keyframe_target_stride),
            "final_keyframes": list(levels[-1]["keyframes"]),
            "final_success": bool(levels[-1]["success"]),
            "final_objective_nonincrease": bool(
                levels[-1]["objective_nonincrease"]
            ),
            "final_selected": bool(final_level_selected),
            "fallback_policy": "zero_identity_when_level_objective_increases",
        }
    else:
        level = _solve_keyframe_fine_graph_level(
            local_poses,
            starts_ends,
            edges,
            alignments,
            initial_frame_poses,
            num_frames,
            cfg=cfg,
            target_stride=cfg.keyframe_target_stride,
            max_nfev=cfg.max_nfev,
            warm_left_corrections=None,
            level_index=0,
        )
        reduced = level["reduced"]
        coarse = list(level["coarse"])
        interpolation_initial = list(level["interpolation_initial"])
        frame_poses = list(level["frame_poses"])
        outer_left_corrections = list(level["left_corrections"])
        keyframes = list(level["keyframes"])

    if continuation_diagnostics is not None:
        keyframes = list(continuation_diagnostics["final_keyframes"])
    reduced_num_keyframes = len(keyframes)
    factorized_keyframe_diagnostics = None
    if cfg.factorized_frame_chart_residual_v4:
        inner_v4 = reduced.diagnostics.get("factorized_frame_chart_residual_v4")
        if not isinstance(inner_v4, dict) or inner_v4.get("enabled") is not True:
            raise ValueError(
                "factorized_frame_chart_residual_v4 keyframe reduced solver did not "
                "activate"
            )
        if inner_v4.get("fallback_to_absolute_residual") is not False:
            raise ValueError(
                "factorized_frame_chart_residual_v4 keyframe reduced solver attempted "
                "absolute residual fallback"
            )
        full_pose_finite = _sim3_sequence_finite([*coarse, *frame_poses])
        outer_left_correction_finite = _sim3_sequence_finite(
            outer_left_corrections
        )
        outer_export_safety = _factorized_v4_left_compose_export_materialization_safety(
            interpolation_initial,
            outer_left_corrections,
            translation_scale=cfg.translation_scale,
        )
        continuation_valid = (
            True
            if continuation_diagnostics is None
            else bool(continuation_diagnostics["final_selected"])
        )
        outer_valid = bool(
            inner_v4.get("valid", False)
            and full_pose_finite
            and outer_left_correction_finite
            and outer_export_safety["gate_pass"]
            and continuation_valid
        )
        factorized_keyframe_diagnostics = {
            **inner_v4,
            "mode": "keyframe_reduced_frozen_coarse_factorized_frame_chart_residual",
            "activation_route": "keyframe_reduced_solver",
            "outer_solver_mode": "keyframe_fine_graph",
            "inner_solver_mode": reduced.diagnostics.get("solver_mode"),
            "outer_num_frames": int(num_frames),
            "reduced_num_frames": int(reduced_num_keyframes),
            "num_keyframes": int(reduced_num_keyframes),
            "keyframe_target_stride": int(cfg.keyframe_target_stride),
            "stride_reduction_preserved": bool(reduced_num_keyframes < num_frames),
            "outer_final_materialized_pose_finite": full_pose_finite,
            "outer_left_correction_finite": outer_left_correction_finite,
            "outer_export_materialization_safety": outer_export_safety,
            "valid": outer_valid,
            "reason": (
                inner_v4.get("reason")
                if outer_valid
                else "keyframe_outer_final_finite_gate_failed"
            ),
        }
        if continuation_diagnostics is not None:
            factorized_keyframe_diagnostics["keyframe_continuation"] = (
                continuation_diagnostics
            )
        if not outer_export_safety["gate_pass"]:
            raise ValueError(
                "factorized_frame_chart_residual_v4 keyframe export "
                "materialization unsafe: "
                f"{outer_export_safety}"
            )
    diagnostics = {
        **reduced.diagnostics,
        "solver_mode": "keyframe_fine_graph",
        "inner_solver_mode": reduced.diagnostics.get("solver_mode"),
        "keyframe_selection": cfg.keyframe_selection,
        "keyframe_target_stride": int(cfg.keyframe_target_stride),
        "num_keyframes": int(reduced_num_keyframes),
        "num_frames": int(num_frames),
        "max_frames_per_solve": int(cfg.max_frames_per_solve),
        "coarse_alignments_frozen": bool(cfg.freeze_coarse_alignments),
        "point_corrections_applied": bool(reduced.apply_point_corrections),
    }
    if continuation_diagnostics is not None:
        diagnostics["num_function_evaluations"] = int(
            continuation_diagnostics["total_nfev"]
        )
        diagnostics["num_function_evaluations_total"] = int(
            continuation_diagnostics["total_nfev"]
        )
        diagnostics["num_jacobian_evaluations"] = (
            None
            if continuation_diagnostics["total_njev"] is None
            else int(continuation_diagnostics["total_njev"])
        )
        diagnostics["num_jacobian_evaluations_total"] = (
            None
            if continuation_diagnostics["total_njev"] is None
            else int(continuation_diagnostics["total_njev"])
        )
        diagnostics["max_nfev"] = int(cfg.max_nfev)
        diagnostics["success"] = bool(continuation_diagnostics["final_selected"])
        diagnostics["status"] = (
            "ok"
            if continuation_diagnostics["final_selected"]
            else "not_converged"
        )
    if factorized_keyframe_diagnostics is not None:
        diagnostics["factorized_frame_chart_residual_v4"] = (
            factorized_keyframe_diagnostics
        )
    return HierarchicalAlphaBasis(
        coarse_alignments=coarse,
        frame_poses=frame_poses,
        apply_point_corrections=reduced.apply_point_corrections,
        diagnostics=diagnostics,
    )


def _apply_trajectory_only_solution(
    chunks: Sequence[Any],
    ranges: Sequence[tuple[int, int]],
    frame_poses: Sequence[Sim3],
    initial_frame_poses: Sequence[Sim3],
    *,
    alpha: float,
) -> list[Any]:
    output = []
    for chunk, (start, end) in zip(chunks, ranges):
        c2w_name, c2w = _get_field(chunk, C2W_FIELDS)
        poses = np.asarray(c2w)
        pose_dtype = poses.dtype if np.issubdtype(poses.dtype, np.floating) else np.float32
        aligned_poses = np.empty_like(poses, dtype=pose_dtype)
        for local_index, frame in enumerate(range(start, end)):
            correction = frame_poses[frame].compose(initial_frame_poses[frame].inverse())
            bounded = _scaled_se3_correction(correction, alpha)
            final_pose = bounded.compose(initial_frame_poses[frame])
            aligned_poses[local_index] = _sim3_to_c2w(final_pose)
        output.append(_replace_chunk_fields(chunk, {c2w_name: aligned_poses}))
    return output


def _build_fine_edges(
    local_poses: Sequence[np.ndarray],
    ranges: Sequence[tuple[int, int]],
    *,
    frame_stride: int,
) -> list[tuple[int, int, int, np.ndarray, np.ndarray]]:
    records: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
    for chunk_index, (poses, (start, end)) in enumerate(zip(local_poses, ranges)):
        length = end - start
        last = length - frame_stride - 1
        offsets = list(range(0, last + 1, frame_stride)) if last >= 0 else []
        if last >= 0 and last not in offsets:
            offsets.append(last)
        for local_a in offsets:
            local_b = local_a + frame_stride
            records.append(
                (
                    chunk_index,
                    start + local_a,
                    start + local_b,
                    poses[local_a],
                    poses[local_b],
                )
            )
    return records


def _build_prior_records(
    local_poses: Sequence[np.ndarray],
    ranges: Sequence[tuple[int, int]],
) -> list[tuple[int, int, np.ndarray]]:
    records: list[tuple[int, int, np.ndarray]] = []
    for chunk_index, (poses, (start, end)) in enumerate(zip(local_poses, ranges)):
        for local_index, frame in enumerate(range(start, end)):
            records.append((chunk_index, frame, poses[local_index]))
    return records


def _sim3_to_parameters(transform: Sim3) -> np.ndarray:
    return np.concatenate(
        [
            Rotation.from_matrix(transform.rotation).as_rotvec(),
            transform.translation,
            np.array([np.log(abs(transform.scale))], dtype=np.float64),
        ]
    )


def _se3_to_parameters(transform: Sim3) -> np.ndarray:
    return np.concatenate(
        [
            Rotation.from_matrix(transform.rotation).as_rotvec(),
            transform.translation,
        ]
    )


def _parameters_to_sim3(parameters: np.ndarray) -> Sim3:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (7,):
        raise ValueError(f"Sim3 parameters must have shape (7,), got {values.shape}")
    return Sim3(
        scale=float(np.exp(values[6])),
        rotation=Rotation.from_rotvec(values[:3]).as_matrix(),
        translation=values[3:6],
    )


def _parameters_to_se3(parameters: np.ndarray) -> Sim3:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError(f"SE3 parameters must have shape (6,), got {values.shape}")
    return Sim3(
        rotation=Rotation.from_rotvec(values[:3]).as_matrix(),
        translation=values[3:6],
    )


def _inverse_compose_without_intermediate(transform_b: Sim3, transform_a: Sim3) -> Sim3:
    """Return ``b.inverse().compose(a)`` without constructing inverse(b)."""

    scale, rotation, translation = _inverse_compose_components_without_intermediate(
        transform_b, transform_a
    )
    return Sim3(scale=scale, rotation=rotation, translation=translation)


def _inverse_compose_components_without_intermediate(
    transform_b: Sim3, transform_a: Sim3
) -> tuple[float, np.ndarray, np.ndarray]:
    inverse_rotation = transform_b.rotation.T
    inverse_scale = 1.0 / transform_b.scale
    inverse_translation = -(
        inverse_scale * (inverse_rotation @ transform_b.translation)
    )
    return (
        inverse_scale * transform_a.scale,
        inverse_rotation @ transform_a.rotation,
        (
            inverse_scale * (inverse_rotation @ transform_a.translation)
            + inverse_translation
        ),
    )


def _inverse_compose_components_batch_exact(
    transforms_b: Sequence[Sim3], transforms_a: Sequence[Sim3]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch ``b.inverse().compose(a)`` while preserving per-item arithmetic."""

    if len(transforms_a) != len(transforms_b):
        raise ValueError("batched inverse/compose inputs must have equal length")
    scales_a = np.asarray([item.scale for item in transforms_a], dtype=np.float64)
    scales_b = np.asarray([item.scale for item in transforms_b], dtype=np.float64)
    rotations_a = np.asarray(
        [item.rotation for item in transforms_a], dtype=np.float64
    ).reshape(-1, 3, 3)
    rotations_b = np.asarray(
        [item.rotation for item in transforms_b], dtype=np.float64
    ).reshape(-1, 3, 3)
    translations_a = np.asarray(
        [item.translation for item in transforms_a], dtype=np.float64
    ).reshape(-1, 3)
    translations_b = np.asarray(
        [item.translation for item in transforms_b], dtype=np.float64
    ).reshape(-1, 3)
    return _inverse_compose_component_arrays_exact(
        scales_b,
        rotations_b,
        translations_b,
        scales_a,
        rotations_a,
        translations_a,
    )


def _inverse_compose_component_arrays_exact(
    scales_b: np.ndarray,
    rotations_b: np.ndarray,
    translations_b: np.ndarray,
    scales_a: np.ndarray,
    rotations_a: np.ndarray,
    translations_a: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch component arrays with the exact arithmetic order of inverse/compose."""

    scales_a = np.asarray(scales_a, dtype=np.float64).reshape(-1)
    scales_b = np.asarray(scales_b, dtype=np.float64).reshape(-1)
    rotations_a = np.asarray(rotations_a, dtype=np.float64).reshape(-1, 3, 3)
    rotations_b = np.asarray(rotations_b, dtype=np.float64).reshape(-1, 3, 3)
    translations_a = np.asarray(translations_a, dtype=np.float64).reshape(-1, 3)
    translations_b = np.asarray(translations_b, dtype=np.float64).reshape(-1, 3)
    counts = {
        len(scales_a),
        len(scales_b),
        len(rotations_a),
        len(rotations_b),
        len(translations_a),
        len(translations_b),
    }
    if len(counts) != 1:
        raise ValueError("batched inverse/compose component counts must match")
    inverse_rotation = np.swapaxes(rotations_b, -1, -2)
    inverse_scale = 1.0 / scales_b
    inverse_translation = -(
        inverse_scale[:, None]
        * np.matmul(inverse_rotation, translations_b[..., None])[..., 0]
    )
    return (
        inverse_scale * scales_a,
        np.matmul(inverse_rotation, rotations_a),
        (
            inverse_scale[:, None]
            * np.matmul(inverse_rotation, translations_a[..., None])[..., 0]
            + inverse_translation
        ),
    )


def _residual_component_diagnostics(
    residual: np.ndarray,
    num_coarse: int,
    num_fine: int,
    num_prior: int,
    *,
    loss: str,
    f_scale: float,
) -> dict[str, Any]:
    """Describe the exact weighted rows, without interpreting units as metres."""
    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    expected = 7 * num_coarse + 6 * (num_fine + num_prior)
    if values.size != expected:
        raise ValueError(f"residual diagnostic row mismatch: {values.size} != {expected}")
    output: dict[str, Any] = {}
    offset = 0
    for family, count, width in (("coarse", num_coarse, 7),
                                 ("fine", num_fine, 6),
                                 ("prior", num_prior, 6)):
        rows = values[offset:offset + count * width].reshape(count, width)
        offset += count * width
        components = {"rotation": rows[:, :3], "translation": rows[:, 3:6]}
        if width == 7:
            components["log_scale"] = rows[:, 6:7]
        output[family] = {}
        for name, component in components.items():
            flat = component.reshape(-1)
            finite = bool(np.all(np.isfinite(flat)))
            absolute = np.abs(flat)
            output[family][name] = {
                "count": int(flat.size), "finite": finite,
                "abs_p50_p90_p95_p99_max": (
                    np.percentile(absolute, [50, 90, 95, 99, 100]).tolist()
                    if flat.size and finite else None
                ),
                "rms": float(np.sqrt(np.mean(flat ** 2))) if flat.size and finite else None,
                "huber_outlier_count": (
                    int(np.count_nonzero(absolute > f_scale)) if loss == "huber" else None
                ),
                "huber_outlier_fraction": (
                    float(np.mean(absolute > f_scale))
                    if loss == "huber" and flat.size and finite else None
                ),
                "robust_cost": (
                    _scipy_compatible_robust_cost(flat, loss=loss, f_scale=f_scale)
                    if finite else None
                ),
            }
    return output


def _sim3_residual(error: Sim3, translation_scale: float) -> np.ndarray:
    return np.concatenate(
        [
            Rotation.from_matrix(error.rotation).as_rotvec(),
            error.translation * float(translation_scale),
            np.array([np.log(abs(error.scale))], dtype=np.float64),
        ]
    )


def _se3_residual(error: Sim3, translation_scale: float) -> np.ndarray:
    return np.concatenate(
        [
            Rotation.from_matrix(error.rotation).as_rotvec(),
            error.translation * float(translation_scale),
        ]
    )


def _weighted_edge_residual(
    edge: Sim3Edge,
    alignments: Sequence[Sim3],
    translation_scale: float,
) -> np.ndarray:
    predicted = alignments[edge.target].inverse().compose(alignments[edge.source])
    error = edge.transform.inverse().compose(predicted)
    weight = float(edge.diagnostics.get("weight", 1.0))
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(f"edge weight must be finite and non-negative, got {weight}")
    return np.sqrt(weight) * _sim3_residual(error, translation_scale)


def _weighted_edge_residuals_batch_exact(
    edges: Sequence[Sim3Edge],
    alignments: Sequence[Sim3],
    translation_scale: float,
) -> np.ndarray:
    """Evaluate fixed coarse residuals in edge order without temporary Sim3 churn."""

    if not edges:
        return np.zeros(0, dtype=np.float64)
    targets = [alignments[edge.target] for edge in edges]
    sources = [alignments[edge.source] for edge in edges]
    predicted_components = _inverse_compose_components_batch_exact(targets, sources)
    predicted = _validated_sim3_batch(*predicted_components)
    measured = [edge.transform for edge in edges]
    error_scale, error_rotation, error_translation = (
        _inverse_compose_components_batch_exact(measured, predicted)
    )
    weights = np.asarray(
        [float(edge.diagnostics.get("weight", 1.0)) for edge in edges],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("edge weights must be finite and non-negative")
    residuals = np.concatenate(
        [
            Rotation.from_matrix(error_rotation).as_rotvec(),
            error_translation * float(translation_scale),
            np.log(np.abs(error_scale))[:, None],
        ],
        axis=1,
    )
    return (np.sqrt(weights)[:, None] * residuals).reshape(-1)


def _build_jacobian_sparsity(
    num_variables: int,
    coarse_edges: Sequence[Sim3Edge],
    fine_edges: Sequence[tuple[int, int, int, np.ndarray, np.ndarray]],
    prior_records: Sequence[tuple[int, int, np.ndarray]],
    coarse_offsets: dict[int, int],
    frame_offsets: dict[int, int],
    *,
    translation_scale: float,
    component_exact_frame_blocks: bool = False,
) -> lil_matrix:
    from scipy.sparse import lil_matrix

    del translation_scale
    rows = 7 * len(coarse_edges) + 6 * (len(fine_edges) + len(prior_records))
    if component_exact_frame_blocks and not coarse_offsets:
        return _build_frozen_component_exact_jacobian_sparsity(
            num_variables,
            num_coarse_edges=len(coarse_edges),
            fine_edges=fine_edges,
            prior_records=prior_records,
            frame_offsets=frame_offsets,
        )
    sparsity = lil_matrix((rows, num_variables), dtype=np.int8)
    row = 0

    def mark(row_start: int, row_width: int, variables: Sequence[tuple[int | None, int]]) -> None:
        for offset, width in variables:
            if offset is not None:
                sparsity[row_start : row_start + row_width, offset : offset + width] = 1

    for edge in coarse_edges:
        mark(
            row,
            7,
            [
                (coarse_offsets.get(edge.source), 7),
                (coarse_offsets.get(edge.target), 7),
            ],
        )
        row += 7
    for chunk_index, frame_a, frame_b, _, _ in fine_edges:
        if component_exact_frame_blocks:
            coarse_offset = coarse_offsets.get(chunk_index)
            frame_a_offset = frame_offsets.get(frame_a)
            frame_b_offset = frame_offsets.get(frame_b)
            mark(
                row,
                3,
                [
                    (coarse_offset, 7),
                    (frame_a_offset, 3),
                    (frame_b_offset, 3),
                ],
            )
            mark(
                row + 3,
                3,
                [
                    (coarse_offset, 7),
                    (frame_b_offset, 3),
                    (
                        None if frame_a_offset is None else frame_a_offset + 3,
                        3,
                    ),
                    (
                        None if frame_b_offset is None else frame_b_offset + 3,
                        3,
                    ),
                ],
            )
        else:
            mark(
                row,
                6,
                [
                    (coarse_offsets.get(chunk_index), 7),
                    (frame_offsets.get(frame_a), 6),
                    (frame_offsets.get(frame_b), 6),
                ],
            )
        row += 6
    for chunk_index, frame, _ in prior_records:
        if component_exact_frame_blocks:
            coarse_offset = coarse_offsets.get(chunk_index)
            frame_offset = frame_offsets.get(frame)
            mark(row, 3, [(coarse_offset, 7), (frame_offset, 3)])
            mark(
                row + 3,
                3,
                [
                    (coarse_offset, 7),
                    (None if frame_offset is None else frame_offset + 3, 3),
                ],
            )
        else:
            mark(
                row,
                6,
                [
                    (coarse_offsets.get(chunk_index), 7),
                    (frame_offsets.get(frame), 6),
                ],
            )
        row += 6
    return sparsity


def _build_frozen_component_exact_jacobian_sparsity(
    num_variables: int,
    *,
    num_coarse_edges: int,
    fine_edges: Sequence[tuple[int, int, int, np.ndarray, np.ndarray]],
    prior_records: Sequence[tuple[int, int, np.ndarray]],
    frame_offsets: dict[int, int],
):
    from scipy.sparse import coo_matrix

    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    block_width = 3
    block_index = np.arange(block_width, dtype=np.int64)

    def append_blocks(row_starts: np.ndarray, column_starts: np.ndarray) -> None:
        valid = column_starts >= 0
        if not bool(np.any(valid)):
            return
        row_values = row_starts[valid, None, None] + block_index[None, :, None]
        column_values = (
            column_starts[valid, None, None] + block_index[None, None, :]
        )
        shape = (int(valid.sum()), block_width, block_width)
        row_parts.append(np.broadcast_to(row_values, shape).reshape(-1))
        column_parts.append(np.broadcast_to(column_values, shape).reshape(-1))

    fine_count = len(fine_edges)
    fine_row_starts = 7 * num_coarse_edges + 6 * np.arange(
        fine_count, dtype=np.int64
    )
    fine_a_offsets = np.asarray(
        [frame_offsets.get(item[1], -1) for item in fine_edges], dtype=np.int64
    )
    fine_b_offsets = np.asarray(
        [frame_offsets.get(item[2], -1) for item in fine_edges], dtype=np.int64
    )
    append_blocks(fine_row_starts, fine_a_offsets)
    append_blocks(fine_row_starts, fine_b_offsets)
    append_blocks(fine_row_starts + 3, fine_b_offsets)
    append_blocks(fine_row_starts + 3, np.where(fine_a_offsets >= 0, fine_a_offsets + 3, -1))
    append_blocks(fine_row_starts + 3, np.where(fine_b_offsets >= 0, fine_b_offsets + 3, -1))

    prior_count = len(prior_records)
    prior_row_starts = (
        7 * num_coarse_edges
        + 6 * fine_count
        + 6 * np.arange(prior_count, dtype=np.int64)
    )
    prior_offsets = np.asarray(
        [frame_offsets.get(item[1], -1) for item in prior_records], dtype=np.int64
    )
    append_blocks(prior_row_starts, prior_offsets)
    append_blocks(
        prior_row_starts + 3,
        np.where(prior_offsets >= 0, prior_offsets + 3, -1),
    )

    num_rows = 7 * num_coarse_edges + 6 * (fine_count + prior_count)
    if not row_parts:
        return coo_matrix((num_rows, num_variables), dtype=np.int8).tocsr()
    row_indices = np.concatenate(row_parts)
    column_indices = np.concatenate(column_parts)
    values = np.ones(row_indices.size, dtype=np.int8)
    return coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(num_rows, num_variables),
        dtype=np.int8,
    ).tocsr()


def _apply_frame_solution(
    chunks: Sequence[Any],
    ranges: Sequence[tuple[int, int]],
    coarse_alignments: Sequence[Sim3],
    frame_poses: Sequence[Sim3],
    *,
    apply_frame_corrections: bool,
    frame_correction_alpha: float,
) -> list[Any]:
    if apply_frame_corrections and float(frame_correction_alpha) > 0.0:
        return _apply_positive_frame_solution_component_validated(
            chunks,
            ranges,
            coarse_alignments,
            frame_poses,
            frame_correction_alpha=float(frame_correction_alpha),
        )
    output: list[Any] = []
    for chunk_index, (chunk, (start, end)) in enumerate(zip(chunks, ranges)):
        point_name, points = _chunk_points(chunk)
        c2w_name, c2w = _get_field(chunk, C2W_FIELDS)
        poses = np.asarray(c2w, dtype=np.float64)
        point_dtype = points.dtype if np.issubdtype(points.dtype, np.floating) else np.float32
        pose_dtype = poses.dtype if np.issubdtype(poses.dtype, np.floating) else np.float32
        aligned_points = np.empty_like(points, dtype=point_dtype)
        aligned_poses = np.empty_like(poses, dtype=pose_dtype)
        coarse = coarse_alignments[chunk_index]
        for local_index, frame in enumerate(range(start, end)):
            initial_global_pose = _c2w_to_sim3(coarse.apply_c2w(poses[local_index]))
            base_points = coarse.apply_points(points[local_index])
            if apply_frame_corrections:
                correction = frame_poses[frame].compose(initial_global_pose.inverse())
                bounded = _scaled_se3_correction(correction, frame_correction_alpha)
                aligned_points[local_index] = bounded.apply_points(base_points)
                aligned_poses[local_index] = bounded.apply_c2w(
                    coarse.apply_c2w(poses[local_index])
                )
            else:
                aligned_points[local_index] = base_points
                aligned_poses[local_index] = coarse.apply_c2w(poses[local_index])
        output.append(
            _replace_chunk_fields(
                chunk,
                {point_name: aligned_points, c2w_name: aligned_poses},
            )
        )
    return output


def _apply_positive_frame_solution_component_validated(
    chunks: Sequence[Any],
    ranges: Sequence[tuple[int, int]],
    coarse_alignments: Sequence[Sim3],
    frame_poses: Sequence[Sim3],
    *,
    frame_correction_alpha: float = 1.0,
) -> list[Any]:
    """Apply positive alpha via validated component kernels without temporary Sim3."""

    if frame_correction_alpha < 1.0:
        return _apply_partial_frame_solution_globally_batched(
            chunks,
            ranges,
            coarse_alignments,
            frame_poses,
            frame_correction_alpha=frame_correction_alpha,
        )

    output: list[Any] = []
    for chunk_index, (chunk, (start, end)) in enumerate(zip(chunks, ranges)):
        point_name, points = _chunk_points(chunk)
        c2w_name, c2w = _get_field(chunk, C2W_FIELDS)
        poses = np.asarray(c2w, dtype=np.float64)
        point_dtype = points.dtype if np.issubdtype(points.dtype, np.floating) else np.float32
        pose_dtype = poses.dtype if np.issubdtype(poses.dtype, np.floating) else np.float32
        coarse = coarse_alignments[chunk_index]
        length = end - start
        aligned_points = np.empty_like(points, dtype=point_dtype)
        aligned_poses = np.empty_like(poses, dtype=pose_dtype)
        correction_scales = np.empty(length, dtype=np.float64)
        correction_rotations = np.empty((length, 3, 3), dtype=np.float64)
        correction_translations = np.empty((length, 3), dtype=np.float64)
        for local_index, frame in enumerate(range(start, end)):
            local_pose = poses[local_index]
            initial_c2w = local_pose.copy()
            initial_c2w[:3, :3] = np.einsum(
                "ij,...jk->...ik", coarse.rotation, local_pose[:3, :3]
            )
            initial_c2w[:3, 3] = (
                coarse.scale
                * np.einsum("ij,...j->...i", coarse.rotation, local_pose[:3, 3])
                + coarse.translation
            )
            initial_u, _, initial_vt = np.linalg.svd(initial_c2w[:3, :3])
            initial_rotation = initial_u @ initial_vt
            projected_det = (
                initial_rotation[0, 0]
                * (
                    initial_rotation[1, 1] * initial_rotation[2, 2]
                    - initial_rotation[1, 2] * initial_rotation[2, 1]
                )
                - initial_rotation[0, 1]
                * (
                    initial_rotation[1, 0] * initial_rotation[2, 2]
                    - initial_rotation[1, 2] * initial_rotation[2, 0]
                )
                + initial_rotation[0, 2]
                * (
                    initial_rotation[1, 0] * initial_rotation[2, 1]
                    - initial_rotation[1, 1] * initial_rotation[2, 0]
                )
            )
            if projected_det < 0.0:
                initial_u[:, -1] *= -1.0
                initial_rotation = initial_u @ initial_vt
            initial_translation = initial_c2w[:3, 3]
            inverse_rotation = initial_rotation.T
            inverse_translation = -(inverse_rotation @ initial_translation)
            target = frame_poses[frame]
            correction_scale = target.scale
            correction_rotation = target.rotation @ inverse_rotation
            correction_translation = (
                target.scale * (target.rotation @ inverse_translation)
                + target.translation
            )
            correction_scales[local_index] = correction_scale
            correction_rotations[local_index] = correction_rotation
            correction_translations[local_index] = correction_translation
            base_points = (
                coarse.scale
                * np.einsum(
                    "ij,...j->...i", coarse.rotation, points[local_index]
                )
                + coarse.translation
            )
            aligned_points[local_index] = (
                correction_scale
                * np.einsum(
                    "ij,...j->...i", correction_rotation, base_points
                )
                + correction_translation
            )
            aligned_pose = initial_c2w.copy()
            aligned_pose[:3, :3] = np.einsum(
                "ij,...jk->...ik", correction_rotation, initial_c2w[:3, :3]
            )
            aligned_pose[:3, 3] = (
                correction_scale
                * np.einsum(
                    "ij,...j->...i", correction_rotation, initial_c2w[:3, 3]
                )
                + correction_translation
            )
            aligned_poses[local_index] = aligned_pose
        _validate_sim3_batch_components(
            correction_scales, correction_rotations, correction_translations
        )
        output.append(
            _replace_chunk_fields(
                chunk,
                {
                    point_name: aligned_points,
                    c2w_name: aligned_poses,
                },
            )
        )
    return output


def _apply_partial_frame_solution_globally_batched(
    chunks: Sequence[Any],
    ranges: Sequence[tuple[int, int]],
    coarse_alignments: Sequence[Sim3],
    frame_poses: Sequence[Sim3],
    *,
    frame_correction_alpha: float,
) -> list[Any]:
    """Batch the element-wise SO(3) interpolation across all chunk occurrences."""

    prepared: list[tuple[Any, str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    correction_scales: list[np.ndarray] = []
    correction_rotations: list[np.ndarray] = []
    correction_translations: list[np.ndarray] = []
    for chunk_index, (chunk, (start, end)) in enumerate(zip(chunks, ranges)):
        point_name, points = _chunk_points(chunk)
        c2w_name, c2w = _get_field(chunk, C2W_FIELDS)
        poses = np.asarray(c2w, dtype=np.float64)
        point_dtype = points.dtype if np.issubdtype(points.dtype, np.floating) else np.float32
        coarse = coarse_alignments[chunk_index]
        initial_c2ws = poses.copy()
        initial_c2ws[:, :3, :3] = np.einsum(
            "ij,njk->nik", coarse.rotation, poses[:, :3, :3]
        )
        initial_c2ws[:, :3, 3] = (
            coarse.scale
            * np.einsum("ij,nj->ni", coarse.rotation, poses[:, :3, 3])
            + coarse.translation
        )
        coarse_points = np.empty_like(points, dtype=point_dtype)
        coarse_points[:] = (
            coarse.scale
            * np.einsum("ij,n...j->n...i", coarse.rotation, points)
            + coarse.translation
        )
        initial_u, _, initial_vt = np.linalg.svd(initial_c2ws[:, :3, :3])
        initial_rotations = initial_u @ initial_vt
        reflected = np.linalg.det(initial_rotations) < 0.0
        if np.any(reflected):
            initial_u = initial_u.copy()
            initial_u[reflected, :, -1] *= -1.0
            initial_rotations = initial_u @ initial_vt
        inverse_rotations = np.swapaxes(initial_rotations, 1, 2)
        inverse_translations = -np.einsum(
            "nij,nj->ni", inverse_rotations, initial_c2ws[:, :3, 3]
        )
        targets = frame_poses[start:end]
        target_scales = np.asarray([target.scale for target in targets], dtype=np.float64)
        target_rotations = np.asarray(
            [target.rotation for target in targets], dtype=np.float64
        )
        target_translations = np.asarray(
            [target.translation for target in targets], dtype=np.float64
        )
        rotations = np.einsum(
            "nij,njk->nik", target_rotations, inverse_rotations
        )
        translations = (
            target_scales[:, None]
            * np.einsum("nij,nj->ni", target_rotations, inverse_translations)
            + target_translations
        )
        _validate_sim3_batch_components(target_scales, rotations, translations)
        prepared.append(
            (chunk, point_name, c2w_name, initial_c2ws, coarse_points, poses)
        )
        correction_scales.append(target_scales)
        correction_rotations.append(rotations)
        correction_translations.append(translations)

    if not correction_rotations:
        return []
    occurrence_counts = [len(item) for item in correction_translations]
    all_scales = np.concatenate(correction_scales, axis=0)
    if np.any(all_scales <= 0.0):
        raise ValueError("partial frame correction requires positive Sim3 scales")
    bounded_scales = np.exp(frame_correction_alpha * np.log(all_scales))
    all_rotations = np.concatenate(correction_rotations, axis=0)
    bounded_rotations = _numpy_rotvec_to_matrix(
        frame_correction_alpha * _numpy_matrix_to_rotvec_hybrid(all_rotations)
    )
    output: list[Any] = []
    offset = 0
    for prepared_item, translations, count in zip(
        prepared, correction_translations, occurrence_counts
    ):
        chunk, point_name, c2w_name, initial_c2ws, coarse_points, poses = prepared_item
        bounded = bounded_rotations[offset : offset + count]
        bounded_scale = bounded_scales[offset : offset + count]
        offset += count
        bounded_translations = frame_correction_alpha * translations
        aligned_points = np.empty_like(coarse_points)
        aligned_points[:] = (
            bounded_scale[:, None, None, None]
            * np.einsum("nij,n...j->n...i", bounded, coarse_points)
            + bounded_translations[:, None, None, :]
        )
        pose_dtype = poses.dtype if np.issubdtype(poses.dtype, np.floating) else np.float32
        aligned_poses = np.empty_like(poses, dtype=pose_dtype)
        aligned_poses[:] = initial_c2ws
        aligned_poses[:, :3, :3] = np.einsum(
            "nij,njk->nik", bounded, initial_c2ws[:, :3, :3]
        )
        aligned_poses[:, :3, 3] = (
            bounded_scale[:, None]
            * np.einsum("nij,nj->ni", bounded, initial_c2ws[:, :3, 3])
            + bounded_translations
        )
        output.append(
            _replace_chunk_fields(
                chunk,
                {point_name: aligned_points, c2w_name: aligned_poses},
            )
        )
    return output


def _scaled_se3_correction(correction: Sim3, alpha: float) -> Sim3:
    """Interpolate an SE3 correction from identity for bounded hybrid output."""

    if alpha <= 0.0:
        return Sim3.identity()
    if alpha >= 1.0:
        return correction
    rotvec = Rotation.from_matrix(correction.rotation).as_rotvec()
    return Sim3(
        rotation=Rotation.from_rotvec(float(alpha) * rotvec).as_matrix(),
        translation=float(alpha) * correction.translation,
    )


__all__ = [
    "HierarchicalAlphaBasis",
    "HierarchicalPoseGraphConfig",
    "HierarchicalPoseGraphResult",
    "apply_hierarchical_alpha",
    "build_hierarchical_basis",
    "optimize_hierarchical_pose_graph",
]
