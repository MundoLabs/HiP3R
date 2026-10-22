"""Reconstruction presets for keyframe, frozen-anchor, and joint optimization."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from long_reconstruction.runner import (
    DEFAULT_HIERARCHICAL_COARSE_MAX_NFEV,
    ReconstructionConfig,
)


METHOD_SETTINGS = ("frozen", "joint", "keyframe")
_FACTORIZED_V4_KEYFRAME_CONTINUATION_SCHEDULE_VERSION = (
    "factorized_frame_chart_residual_v4_keyframe_continuation_v1"
)
_FACTORIZED_V4_KEYFRAME_CONTINUATION_STRIDE_MULTIPLIERS = (4, 2, 1)
_FACTORIZED_V4_KEYFRAME_CONTINUATION_BUDGET_FRACTIONS = (0.15, 0.25, 0.60)
_FACTORIZED_V4_KEYFRAME_CONTINUATION_V2_SCHEDULE_VERSION = (
    "factorized_frame_chart_residual_v4_keyframe_continuation_v2"
)
_FACTORIZED_V4_KEYFRAME_CONTINUATION_V2_STRIDE_MULTIPLIERS = (2, 1)
_FACTORIZED_V4_KEYFRAME_CONTINUATION_V2_BUDGET_FRACTIONS = (0.125, 0.875)
_FACTORIZED_V4_KEYFRAME_CONTINUATION_V2_ALPHA_GRID = (
    1.0,
    0.5,
    0.25,
    0.125,
    0.0,
)

_OPERATIONAL_OVERRIDES = frozenset(
    {
        "chunk_size",
        "overlap",
        "loop_window_size",
        "max_alignment_points",
        "chain_global_scale_gauge_centering_v1",
        "chain_global_scale_gauge_centering_span_cap",
        "hierarchical_max_nfev",
        "hierarchical_coarse_max_nfev",
        "hierarchical_coarse_loop_continuation_stages",
        "hierarchical_coarse_adaptive_loop_continuation",
        "hierarchical_coarse_continuation_max_nfev",
        "hierarchical_coarse_adaptive_budget_fallback_max_nfev",
        "hierarchical_solver_ftol",
        "hierarchical_solver_xtol",
        "hierarchical_solver_gtol",
        "hierarchical_require_solver_convergence",
        "hierarchical_robust_loss",
        "hierarchical_f_scale",
        "max_sim3_iterations",
        "hierarchical_coarse_accept_max_nfev_optimality",
        "hierarchical_max_frames_per_solve",
        "hierarchical_factorized_v4_keyframe_continuation",
        "hierarchical_factorized_v4_keyframe_continuation_objective_projection",
        "hierarchical_long_sequence_mode",
        "hierarchical_keyframe_selection",
        "hierarchical_keyframe_target_stride",
        "hierarchical_keyframe_min_overlap_shared",
        "save_chunk_predictions",
        "local_chunk_workers",
        "local_chunk_gpu_ids",
        "postprocess_loop_workers",
        "artifact_load_workers",
        "parallel_result_root",
    }
)


def _shared_config() -> ReconstructionConfig:
    return ReconstructionConfig(
        chunk_size=60,
        overlap=30,
        loop_window_size=20,
        postprocess="hierarchical_pose_graph",
        confidence_threshold=0.0,
        max_alignment_points=200_000,
        max_sim3_iterations=10,
        residual_edge_weighting=True,
        residual_edge_quality_quantile="p95_else_p90",
        use_jac_sparsity=True,
        save_chunk_predictions=False,
        hierarchical_fine_edge_weight=1.0,
        hierarchical_coarse_prior_weight=0.25,
        hierarchical_frame_stride=1,
        hierarchical_max_nfev=100,
        hierarchical_solver_ftol=1e-8,
        hierarchical_solver_xtol=1e-8,
        hierarchical_solver_gtol=1e-8,
        hierarchical_coarse_max_nfev=DEFAULT_HIERARCHICAL_COARSE_MAX_NFEV,
        hierarchical_robust_loss="huber",
        hierarchical_f_scale=1.0,
        hierarchical_translation_scale=1.0,
        hierarchical_apply_frame_corrections=True,
        hierarchical_frame_correction_alpha=1.0,
        hierarchical_max_frames_per_solve=2_000,
        hierarchical_consensus_keyframe_stride=10,
        hierarchical_solver_workers=1,
        hierarchical_incremental_frozen_residual=True,
        hierarchical_analytic_jacobian_backend="closed_form",
        hierarchical_analytic_fast_residual_logmap=True,
        hierarchical_coarse_analytic_jacobian=True,
        hierarchical_require_solver_convergence=True,
        hierarchical_coarse_x_scale="jac",
        hierarchical_coarse_lsmr_tolerance=1e-8,
        hierarchical_coarse_loop_continuation_stages=4,
        hierarchical_coarse_adaptive_loop_continuation=True,
        hierarchical_coarse_adaptive_full_max_nfev=500,
        hierarchical_coarse_adaptive_budget_fallback_max_nfev=(
            DEFAULT_HIERARCHICAL_COARSE_MAX_NFEV
        ),
        hierarchical_coarse_prefer_boundary_restart_for_active_bounds=True,
        hierarchical_coarse_continuation_max_nfev=100,
        hierarchical_coarse_boundary_restart_passes=2,
        hierarchical_coarse_boundary_restart_max_nfev=1000,
        local_chunk_workers=1,
        postprocess_loop_workers=4,
        artifact_load_workers=4,
    )


def _frozen_config(shared: ReconstructionConfig) -> ReconstructionConfig:
    """Solve the chunk Sim(3) graph, freeze it, refine frame SE(3) only."""
    return replace(
        shared,
        hierarchical_freeze_coarse_alignments=True,
        hierarchical_joint_coarse_warm_start=False,
        hierarchical_normalize_residual_families=False,
        hierarchical_analytic_frozen_jacobian=True,
        hierarchical_analytic_dynamic_jacobian=False,
        hierarchical_vectorize_dynamic_residual=False,
    )


def _joint_config(shared: ReconstructionConfig) -> ReconstructionConfig:
    """Warm-started joint chunk/frame solve with balanced residual families."""
    return replace(
        shared,
        hierarchical_freeze_coarse_alignments=False,
        hierarchical_joint_coarse_warm_start=True,
        hierarchical_normalize_residual_families=True,
        # Keep this bound constant: rounding log(1.05) can affect convergence.
        hierarchical_max_log_scale_delta=0.048790164169432,
        hierarchical_solver_ftol=1e-6,
        hierarchical_solver_xtol=1e-6,
        hierarchical_solver_gtol=1e-6,
        hierarchical_analytic_frozen_jacobian=False,
        hierarchical_analytic_dynamic_jacobian=True,
        hierarchical_analytic_dynamic_jacobian_backend="closed_form",
        hierarchical_analytic_dynamic_jacobian_local_rotation_projection="scalar_svd",
        hierarchical_vectorize_dynamic_residual=True,
    )


def _keyframe_config(shared: ReconstructionConfig) -> ReconstructionConfig:
    """Frozen anchors with the validated long-sequence keyframe fine graph."""
    return replace(
        _frozen_config(shared),
        hierarchical_long_sequence_mode="keyframe_fine_graph",
        hierarchical_keyframe_selection="uniform",
        hierarchical_keyframe_target_stride=5,
        hierarchical_keyframe_min_overlap_shared=3,
    )


def method_config(setting: str, **overrides: Any) -> ReconstructionConfig:
    """Resolve one public setting and reject every unknown override."""
    if setting not in METHOD_SETTINGS:
        raise ValueError(f"setting must be one of {METHOD_SETTINGS}, got {setting!r}")
    unknown = sorted(set(overrides) - _OPERATIONAL_OVERRIDES)
    if unknown:
        raise TypeError(
            "unknown or identity-changing method option(s): " + ", ".join(unknown)
        )
    if (
        setting == "joint"
        and bool(overrides.get("chain_global_scale_gauge_centering_v1", False))
    ):
        raise ValueError(
            "chain_global_scale_gauge_centering_v1/factorized chart v4 is only "
            "supported for frozen or keyframe settings"
        )

    shared = _shared_config()
    if setting == "frozen":
        resolved = _frozen_config(shared)
    elif setting == "joint":
        resolved = _joint_config(shared)
    else:
        resolved = _keyframe_config(shared)
    return replace(resolved, **overrides)


def activation_expectation(
    setting: str, config: ReconstructionConfig | None = None
) -> dict[str, Any]:
    config = method_config(setting) if config is None else config
    frozen_like = setting != "joint"
    return {
        "setting": setting,
        "postprocess": "hierarchical_pose_graph",
        "use_jac_sparsity": True,
        "solver_workers": 1,
        "fine_max_nfev": config.hierarchical_max_nfev,
        "fine_ftol": config.hierarchical_solver_ftol,
        "fine_xtol": config.hierarchical_solver_xtol,
        "fine_gtol": config.hierarchical_solver_gtol,
        "coarse_max_nfev": config.hierarchical_coarse_max_nfev,
        "max_frames_per_solve": config.hierarchical_max_frames_per_solve,
        "long_sequence_mode": config.hierarchical_long_sequence_mode,
        "coarse_jacobian_mode": "sparse_torch_block_analytic",
        "coarse_x_scale": "jac",
        "coarse_alignments_frozen": frozen_like,
        "joint_coarse_warm_start": setting == "joint",
        "residual_family_normalization": setting == "joint",
        "jacobian_mode": (
            "sparse_finite_difference"
            if frozen_like and config.chain_global_scale_gauge_centering_v1
            else (
                "sparse_closed_form_block_analytic"
                if frozen_like
                else "sparse_dynamic_closed_form_block_analytic"
            )
        ),
        "dynamic_residual_vectorized": setting == "joint",
    }


def preset_dict(setting: str) -> dict[str, Any]:
    return asdict(method_config(setting))


def _actual_diagnostics(
    diagnostics: dict[str, Any],
    hierarchical: dict[str, Any],
    coarse: dict[str, Any],
) -> dict[str, Any]:
    """Flatten the runtime diagnostics fields the activation audit compares."""
    return {
        "postprocess": diagnostics.get("postprocess"),
        "coarse_alignments_frozen": hierarchical.get("coarse_alignments_frozen"),
        "joint_coarse_warm_start": hierarchical.get("joint_coarse_warm_start"),
        "residual_family_normalization": hierarchical.get(
            "residual_family_normalization"
        ),
        "jacobian_mode": hierarchical.get("jacobian_mode"),
        "dynamic_residual_vectorized": hierarchical.get(
            "dynamic_residual_vectorized"
        ),
        "solver_workers_requested": hierarchical.get("solver_workers_requested"),
        "solver_workers_actual": hierarchical.get("solver_workers_actual"),
        "fine_max_nfev": hierarchical.get("max_nfev"),
        "fine_ftol": hierarchical.get("solver_ftol"),
        "fine_xtol": hierarchical.get("solver_xtol"),
        "fine_gtol": hierarchical.get("solver_gtol"),
        "fine_nfev": hierarchical.get("num_function_evaluations"),
        "coarse_jacobian_mode": coarse.get("jacobian_mode"),
        "coarse_x_scale": coarse.get("x_scale"),
        "coarse_nfev": coarse.get("num_function_evaluations"),
        "num_variables": hierarchical.get("num_variables"),
        "num_chunks": hierarchical.get("num_chunks"),
        "num_frames": hierarchical.get("num_frames"),
        "num_coarse_edges": hierarchical.get("num_coarse_edges"),
        "num_fine_edges": hierarchical.get("num_fine_edges"),
        "num_frame_priors": hierarchical.get("num_frame_priors"),
        "fine_edge_weight": hierarchical.get("fine_edge_weight"),
        "coarse_prior_weight": hierarchical.get("coarse_prior_weight"),
        "solver_mode": hierarchical.get("solver_mode"),
        "max_frames_per_solve": hierarchical.get("max_frames_per_solve"),
        "num_keyframes": hierarchical.get("num_keyframes"),
    }


def _is_overlap_consensus(
    setting: str, expected: dict[str, Any], actual: dict[str, Any]
) -> bool:
    """Frozen sequences above the frame threshold legally skip the fine solver."""
    return (
        setting == "frozen"
        and actual["solver_mode"] == "overlap_consensus"
        and actual["max_frames_per_solve"] == expected["max_frames_per_solve"]
    )


def _long_sequence_mode_proven(
    setting: str,
    config: ReconstructionConfig,
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> bool:
    """Fail closed on documented reduced and continuation long-sequence paths.

    When ``hierarchical_long_sequence_mode='keyframe_fine_graph'`` was
    requested and the scene exceeds the frame threshold, the runtime must
    prove the keyframe solve (mode string plus a sane keyframe count) for
    both Frozen and Joint. Conversely a keyframe solver mode is only legal
    when it was requested.
    """
    requested_mode = config.hierarchical_long_sequence_mode
    reduced_modes = {"keyframe_fine_graph", "keyframe_then_full"}
    actual_mode = actual["solver_mode"]
    if actual_mode in reduced_modes:
        return (
            actual_mode == requested_mode
            and isinstance(actual["num_keyframes"], int)
            and isinstance(actual["num_frames"], int)
            and 2 <= actual["num_keyframes"] < actual["num_frames"]
            and actual["num_frames"] > expected["max_frames_per_solve"]
        )
    if requested_mode in reduced_modes:
        long_sequence = (
            isinstance(actual["num_frames"], int)
            and actual["num_frames"] > expected["max_frames_per_solve"]
        )
        return not long_sequence
    return True


def _has_no_coarse_edges(actual: dict[str, Any], coarse: dict[str, Any]) -> bool:
    """Single-chunk scenes legally report an explicit no-edge coarse stage."""
    return (
        actual["num_coarse_edges"] == 0
        and coarse.get("status") == "no_edges"
        and coarse.get("optimizer_used") is False
    )


def _factorized_v4_coarse_skip(
    config: ReconstructionConfig, coarse: dict[str, Any]
) -> bool:
    return (
        config.chain_global_scale_gauge_centering_v1
        and coarse.get("solver_mode")
        == "factorized_adjacent_chain_no_absolute_coarse_solve"
        and coarse.get("optimizer_used") is False
    )


def _factorized_v4_activation_proven(
    config: ReconstructionConfig,
    hierarchical: dict[str, Any],
    actual: dict[str, Any],
) -> bool:
    if not config.chain_global_scale_gauge_centering_v1:
        return True
    v4 = hierarchical.get("factorized_frame_chart_residual_v4")
    if not isinstance(v4, dict):
        return False
    if not (
        v4.get("enabled") is True
        and v4.get("valid") is True
        and v4.get("fallback_to_absolute_residual") is False
    ):
        return False
    if actual["solver_mode"] == "keyframe_fine_graph":
        return (
            v4.get("activation_route") == "keyframe_reduced_solver"
            and v4.get("outer_solver_mode") == "keyframe_fine_graph"
        )
    return True


def _factorized_v4_keyframe_continuation_proven(
    config: ReconstructionConfig,
    hierarchical: dict[str, Any],
) -> bool:
    if not config.hierarchical_factorized_v4_keyframe_continuation:
        return True
    v4 = hierarchical.get("factorized_frame_chart_residual_v4")
    if not isinstance(v4, dict):
        return False
    continuation = v4.get("keyframe_continuation")
    if not isinstance(continuation, dict):
        return False
    levels = continuation.get("levels")
    budgets = continuation.get("level_budgets")
    total_nfev = continuation.get("total_nfev")
    budget_total = continuation.get("budget_total")
    if not isinstance(levels, list) or not levels:
        return False
    if not isinstance(budgets, list) or len(budgets) != len(levels):
        return False
    try:
        budget_total_int = int(budget_total)
        total_nfev_int = int(total_nfev)
        level_budget_sum = sum(int(item) for item in budgets)
    except (TypeError, ValueError):
        return False
    objective_projection = bool(
        config.hierarchical_factorized_v4_keyframe_continuation_objective_projection
    )
    expected_schema = (
        _FACTORIZED_V4_KEYFRAME_CONTINUATION_V2_SCHEDULE_VERSION
        if objective_projection
        else _FACTORIZED_V4_KEYFRAME_CONTINUATION_SCHEDULE_VERSION
    )
    expected_multipliers = (
        _FACTORIZED_V4_KEYFRAME_CONTINUATION_V2_STRIDE_MULTIPLIERS
        if objective_projection
        else _FACTORIZED_V4_KEYFRAME_CONTINUATION_STRIDE_MULTIPLIERS
    )
    expected_fractions = (
        _FACTORIZED_V4_KEYFRAME_CONTINUATION_V2_BUDGET_FRACTIONS
        if objective_projection
        else _FACTORIZED_V4_KEYFRAME_CONTINUATION_BUDGET_FRACTIONS
    )
    level_fields_present = all(
        isinstance(level, dict)
        and all(
            field in level
            for field in (
                "target_stride",
                "max_nfev",
                "status",
                "success",
                "warm_start_source",
                "final_cost",
            )
        )
        and (
            not objective_projection
            or all(
                field in level
                for field in (
                    "strict_success_status",
                    "termination_status",
                )
            )
        )
        for level in levels
    )
    projection_proven = True
    if objective_projection:
        final_level = levels[-1] if isinstance(levels[-1], dict) else {}
        projected_levels = [
            level
            for level in levels
            if isinstance(level, dict)
            and level.get("warm_start_source")
            == "accepted_previous_left_corrections"
        ]
        projection_proven = bool(
            continuation.get("objective_projection_enabled") is True
            and continuation.get("alpha_grid")
            == list(_FACTORIZED_V4_KEYFRAME_CONTINUATION_V2_ALPHA_GRID)
            and continuation.get("final_success") is True
            and final_level.get("strict_success_status") is True
            and final_level.get("selected_final_level") is True
            and projected_levels
        )
        for level in projected_levels:
            projection = level.get("objective_projection")
            candidates = (
                projection.get("candidates")
                if isinstance(projection, dict)
                else None
            )
            selected_alpha = (
                projection.get("selected_alpha")
                if isinstance(projection, dict)
                else None
            )
            if not (
                isinstance(projection, dict)
                and projection.get("schema")
                == "factorized_frame_chart_residual_v4_objective_projection_v2"
                and selected_alpha
                in list(_FACTORIZED_V4_KEYFRAME_CONTINUATION_V2_ALPHA_GRID)
                and "selected_raw_cost" in projection
                and "selected_robust_cost" in projection
                and "alpha_zero_fail_closed" in projection
                and isinstance(candidates, list)
                and candidates
                and all(
                    isinstance(candidate, dict)
                    and all(
                        field in candidate
                        for field in (
                            "alpha",
                            "raw_cost",
                            "robust_cost",
                            "raw_pass",
                            "robust_pass",
                            "passes",
                        )
                    )
                    for candidate in candidates
                )
            ):
                projection_proven = False
                break
    return bool(
        continuation.get("schema")
        == expected_schema
        and continuation.get("enabled") is True
        and continuation.get("final_selected") is True
        and continuation.get("stride_multipliers")
        == list(expected_multipliers)
        and continuation.get("budget_fractions")
        == list(expected_fractions)
        and continuation.get("objective_projection_enabled")
        is objective_projection
        and budget_total_int == level_budget_sum
        and budget_total_int <= int(config.hierarchical_max_nfev)
        and 0 <= total_nfev_int <= int(config.hierarchical_max_nfev)
        and level_fields_present
        and projection_proven
    )


def _graph_counts_present(
    actual: dict[str, Any], expected: dict[str, Any], overlap_consensus: bool
) -> bool:
    if overlap_consensus:
        return (
            isinstance(actual["num_chunks"], int)
            and actual["num_chunks"] > 0
            and isinstance(actual["num_frames"], int)
            and actual["num_frames"] > expected["max_frames_per_solve"]
        )
    return (
        all(
            isinstance(actual[name], int) and actual[name] > 0
            for name in ("num_variables", "num_fine_edges", "num_frame_priors")
        )
        and isinstance(actual["num_coarse_edges"], int)
        and actual["num_coarse_edges"] >= 0
    )


def _residual_weights_match(
    setting: str,
    config: ReconstructionConfig,
    actual: dict[str, Any],
    graph_counts_present: bool,
) -> bool:
    if setting == "joint" and graph_counts_present:
        normalization_mass = max(actual["num_coarse_edges"], 1)
        return (
            actual["fine_edge_weight"]
            == normalization_mass / actual["num_fine_edges"]
            and actual["coarse_prior_weight"]
            == normalization_mass / actual["num_frame_priors"]
        )
    return (
        actual["fine_edge_weight"] == config.hierarchical_fine_edge_weight
        and actual["coarse_prior_weight"] == config.hierarchical_coarse_prior_weight
    )


def verify_activation(
    setting: str,
    diagnostics: dict[str, Any],
    config: ReconstructionConfig | None = None,
) -> dict[str, Any]:
    """Fail closed when runtime diagnostics do not prove the selected path."""
    hierarchical = diagnostics.get("hierarchical")
    if not isinstance(hierarchical, dict):
        raise RuntimeError("missing hierarchical activation diagnostics")
    coarse = hierarchical.get("coarse_anchor")
    if not isinstance(coarse, dict):
        raise RuntimeError("missing coarse-anchor activation diagnostics")

    config = method_config(setting) if config is None else config
    expected = activation_expectation(setting, config)
    actual = _actual_diagnostics(diagnostics, hierarchical, coarse)
    overlap_consensus = _is_overlap_consensus(setting, expected, actual)
    no_coarse_edges = _has_no_coarse_edges(actual, coarse)
    factorized_v4_coarse_skip = _factorized_v4_coarse_skip(config, coarse)
    graph_counts_present = _graph_counts_present(actual, expected, overlap_consensus)
    residual_weights_match = overlap_consensus or _residual_weights_match(
        setting, config, actual, graph_counts_present
    )

    checks = {
        "postprocess": actual["postprocess"] == expected["postprocess"],
        "long_sequence_mode": _long_sequence_mode_proven(
            setting, config, expected, actual
        ),
        "coarse_alignments_frozen": (
            actual["coarse_alignments_frozen"]
            == expected["coarse_alignments_frozen"]
        ),
        "joint_coarse_warm_start": (
            actual["joint_coarse_warm_start"]
            == expected["joint_coarse_warm_start"]
        ),
        "residual_family_normalization": overlap_consensus or (
            actual["residual_family_normalization"]
            == expected["residual_family_normalization"]
        ),
        "jacobian_mode": overlap_consensus
        or actual["jacobian_mode"] == expected["jacobian_mode"],
        "dynamic_residual_vectorized": overlap_consensus or (
            actual["dynamic_residual_vectorized"]
            == expected["dynamic_residual_vectorized"]
        ),
        "solver_workers": overlap_consensus or actual["solver_workers_actual"] == 1,
        "fine_max_nfev": overlap_consensus
        or actual["fine_max_nfev"] == expected["fine_max_nfev"],
        "fine_tolerances": overlap_consensus or (
            actual["fine_ftol"] == expected["fine_ftol"]
            and actual["fine_xtol"] == expected["fine_xtol"]
            and actual["fine_gtol"] == expected["fine_gtol"]
        ),
        "fine_budget_respected": overlap_consensus or (
            isinstance(actual["fine_nfev"], int)
            and actual["fine_nfev"] <= expected["fine_max_nfev"]
        ),
        "coarse_jacobian_mode": (
            no_coarse_edges
            or factorized_v4_coarse_skip
            or actual["coarse_jacobian_mode"] == expected["coarse_jacobian_mode"]
        ),
        "coarse_x_scale": no_coarse_edges
        or factorized_v4_coarse_skip
        or actual["coarse_x_scale"] == expected["coarse_x_scale"],
        "coarse_budget_respected": (
            no_coarse_edges
            or factorized_v4_coarse_skip
            or (
                isinstance(actual["coarse_nfev"], int)
                and actual["coarse_nfev"] <= expected["coarse_max_nfev"]
            )
        ),
        "graph_counts_present": graph_counts_present,
        "residual_weights": residual_weights_match,
        "factorized_frame_chart_residual_v4": (
            _factorized_v4_activation_proven(config, hierarchical, actual)
        ),
        "factorized_v4_keyframe_continuation": (
            _factorized_v4_keyframe_continuation_proven(config, hierarchical)
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    audit = {"expected": expected, "actual": actual, "checks": checks, "passed": not failed}
    if failed:
        raise RuntimeError(f"method activation audit failed: {', '.join(failed)}; {audit}")
    return audit


__all__ = [
    "METHOD_SETTINGS",
    "activation_expectation",
    "method_config",
    "preset_dict",
    "verify_activation",
]
