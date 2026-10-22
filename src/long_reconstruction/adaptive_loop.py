"""GT-free scoring and safety gates for matched no-loop/loop solutions.

The two solutions must come from one staged forward protocol identity.  Both
are evaluated against the same serialized adjacent and loop measurements; their
solver-specific objective values are deliberately ignored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .sim3 import Sim3


@dataclass(frozen=True)
class AdaptiveLoopPolicy:
    """Frozen, inference-only thresholds for a scene-level decision."""

    min_loop_rmse_reduction_fraction: float
    max_trusted_rmse_ratio: float
    max_trusted_p95_ratio: float
    max_coarse_log_scale_delta: float
    max_coarse_rotation_delta_degrees: float
    max_coarse_translation_delta_fraction: float
    require_convergence: bool = True
    require_no_active_bounds: bool = True
    require_finite: bool = True

    def __post_init__(self) -> None:
        for name in (
            "min_loop_rmse_reduction_fraction",
            "max_trusted_rmse_ratio",
            "max_trusted_p95_ratio",
            "max_coarse_log_scale_delta",
            "max_coarse_rotation_delta_degrees",
            "max_coarse_translation_delta_fraction",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_trusted_rmse_ratio < 1.0:
            raise ValueError("max_trusted_rmse_ratio must be at least 1")
        if self.max_trusted_p95_ratio < 1.0:
            raise ValueError("max_trusted_p95_ratio must be at least 1")

    @property
    def protocol_metadata(self) -> dict[str, Any]:
        return {
            "schema": "hip3r_adaptive_loop_policy_v1",
            "policy": asdict(self),
            "score_schema": "hip3r_common_measurement_score_v1",
        }


@dataclass(frozen=True)
class AdaptiveLoopPolicyV2:
    """Stable two-route policy for small corrections and supported closures.

    Version 1 divided by the adjacent-only residual.  That residual is nearly
    zero by construction for a chained solution, so the resulting ratios are
    ill-conditioned.  Version 2 instead measures the adjacent cost paid by the
    loop solution relative to the *pre-correction loop inconsistency*.  Large
    corrections are accepted only through a multi-loop consensus route.
    """

    min_loop_rmse_reduction_fraction: float
    max_trusted_cost_fraction: float
    max_small_log_scale_delta: float
    max_small_rotation_delta_degrees: float
    max_small_translation_delta_fraction: float
    min_consensus_loop_measurements: int
    min_consensus_loop_rmse_reduction_fraction: float
    max_consensus_log_scale_delta: float
    max_consensus_rotation_delta_degrees: float
    max_consensus_translation_delta_fraction: float
    require_convergence: bool = True
    require_no_active_bounds: bool = True
    require_finite: bool = True

    def __post_init__(self) -> None:
        for name in (
            "min_loop_rmse_reduction_fraction",
            "max_trusted_cost_fraction",
            "max_small_log_scale_delta",
            "max_small_rotation_delta_degrees",
            "max_small_translation_delta_fraction",
            "min_consensus_loop_rmse_reduction_fraction",
            "max_consensus_log_scale_delta",
            "max_consensus_rotation_delta_degrees",
            "max_consensus_translation_delta_fraction",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not isinstance(self.min_consensus_loop_measurements, int):
            raise ValueError("min_consensus_loop_measurements must be an integer")
        if self.min_consensus_loop_measurements < 2:
            raise ValueError("min_consensus_loop_measurements must be at least 2")
        for name in (
            "min_loop_rmse_reduction_fraction",
            "min_consensus_loop_rmse_reduction_fraction",
        ):
            if float(getattr(self, name)) > 1.0:
                raise ValueError(f"{name} must be at most 1")
        if (
            self.min_consensus_loop_rmse_reduction_fraction
            < self.min_loop_rmse_reduction_fraction
        ):
            raise ValueError(
                "consensus loop reduction cannot be weaker than the global minimum"
            )
        for suffix in (
            "log_scale_delta",
            "rotation_delta_degrees",
            "translation_delta_fraction",
        ):
            if float(getattr(self, f"max_consensus_{suffix}")) < float(
                getattr(self, f"max_small_{suffix}")
            ):
                raise ValueError(
                    f"max_consensus_{suffix} cannot be below max_small_{suffix}"
                )

    @property
    def protocol_metadata(self) -> dict[str, Any]:
        return {
            "schema": "hip3r_adaptive_loop_policy_v2",
            "policy": asdict(self),
            "score_schema": "hip3r_common_measurement_score_v1",
            "derived_feature_schema": "hip3r_adaptive_loop_stable_features_v2",
        }


def score_common_measurement_sets(
    no_loop_manifest: Mapping[str, Any],
    loop_manifest: Mapping[str, Any],
    *,
    translation_scale: float = 1.0,
) -> dict[str, Any]:
    """Evaluate both basis solutions on identical adjacent and loop edges."""

    if not np.isfinite(translation_scale) or translation_scale <= 0.0:
        raise ValueError("translation_scale must be positive finite")
    no_loop_forward = no_loop_manifest.get("forward_protocol_id")
    loop_forward = loop_manifest.get("forward_protocol_id")
    if no_loop_forward is None or loop_forward is None:
        raise ValueError("dual solutions are missing forward protocol identity")
    if no_loop_forward != loop_forward:
        raise ValueError("dual solutions do not share one forward protocol identity")

    no_loop_edges = _edge_records(no_loop_manifest)
    loop_edges = _edge_records(loop_manifest)
    no_loop_adjacent = [edge for edge in no_loop_edges if edge["kind"] == "adjacent"]
    common_adjacent = [edge for edge in loop_edges if edge["kind"] == "adjacent"]
    common_loops = [edge for edge in loop_edges if edge["kind"] == "loop"]
    no_loop_adjacent_records = _measurement_records(no_loop_adjacent)
    common_adjacent_records = _measurement_records(common_adjacent)
    common_loop_records = _measurement_records(common_loops)
    if no_loop_adjacent_records != common_adjacent_records:
        raise ValueError("adjacent measurement sets differ between dual branches")

    alignments = {
        "no_loop": _coarse_alignments(no_loop_manifest),
        "loop": _coarse_alignments(loop_manifest),
    }
    if len(alignments["no_loop"]) != len(alignments["loop"]):
        raise ValueError("dual solutions have different coarse alignment counts")
    required_chunks = max(
        1,
        1
        + max(
            (
                max(edge["source"], edge["target"])
                for edge in [*common_adjacent, *common_loops]
            ),
            default=-1,
        ),
    )
    if len(alignments["loop"]) < required_chunks:
        raise ValueError("measurement endpoint exceeds alignment count")

    scores: dict[str, Any] = {}
    for solution, values in alignments.items():
        scores[solution] = {
            "trusted_common": _score_edges(
                common_adjacent, values, translation_scale=translation_scale
            ),
            "loop_common": _score_edges(
                common_loops, values, translation_scale=translation_scale
            ),
        }
    no_loop_score = scores["no_loop"]
    loop_score = scores["loop"]
    loop_reduction = (
        _relative_reduction(
            no_loop_score["loop_common"]["rmse"],
            loop_score["loop_common"]["rmse"],
        )
        if common_loops
        else None
    )
    trusted_ratios = {
        "rmse": _safe_ratio(
            loop_score["trusted_common"]["rmse"],
            no_loop_score["trusted_common"]["rmse"],
        ),
        "p95_edge_norm": _safe_ratio(
            loop_score["trusted_common"]["p95_edge_norm"],
            no_loop_score["trusted_common"]["p95_edge_norm"],
        ),
    }
    correction = _coarse_correction(alignments["no_loop"], alignments["loop"])
    return {
        "schema": "hip3r_common_measurement_score_v1",
        "evidence_scope": "inference_only_no_gt",
        "forward_protocol_id": no_loop_forward,
        "translation_scale": float(translation_scale),
        "measurement_sets": {
            "trusted_common": {
                "definition": "all serialized adjacent edges; common-set, not held-out",
                "count": len(common_adjacent),
                "edge_identity": _measurement_identity(common_adjacent_records),
            },
            "loop_common": {
                "definition": "all admitted loop edges from the loop branch",
                "count": len(common_loops),
                "edge_identity": _measurement_identity(common_loop_records),
            },
        },
        "solutions": scores,
        "comparisons": {
            "loop_rmse_reduction_fraction": loop_reduction,
            "trusted_loop_over_no_loop": trusted_ratios,
            "coarse_correction": correction,
        },
        "solver": {
            "no_loop": _solver_status(no_loop_manifest),
            "loop": _solver_status(loop_manifest),
        },
    }


def decide_adaptive_loop(
    score: Mapping[str, Any], policy: AdaptiveLoopPolicy
) -> dict[str, Any]:
    """Apply a frozen policy to a GT-free common-measurement score."""

    if score.get("schema") != "hip3r_common_measurement_score_v1":
        raise ValueError("unsupported common-measurement score schema")
    comparisons = _mapping(score.get("comparisons"))
    trusted = _mapping(comparisons.get("trusted_loop_over_no_loop"))
    correction = _mapping(comparisons.get("coarse_correction"))
    solver = _mapping(score.get("solver"))
    reasons: list[str] = []
    loop_measurements = _finite_int(
        _mapping(_mapping(score.get("measurement_sets")).get("loop_common")).get(
            "count"
        )
    )
    candidate_available = loop_measurements is not None and loop_measurements > 0
    if loop_measurements is None:
        reasons.append("loop_measurement_count_missing")
    elif loop_measurements <= 0:
        reasons.append("no_admitted_loop_measurements")

    if candidate_available:
        _minimum_gate(
            comparisons.get("loop_rmse_reduction_fraction"),
            policy.min_loop_rmse_reduction_fraction,
            "loop_common_improvement_below_threshold",
            reasons,
        )
        _maximum_gate(
            trusted.get("rmse"),
            policy.max_trusted_rmse_ratio,
            "trusted_common_rmse_regression",
            reasons,
        )
        _maximum_gate(
            trusted.get("p95_edge_norm"),
            policy.max_trusted_p95_ratio,
            "trusted_common_p95_regression",
            reasons,
        )
        _maximum_gate(
            correction.get("max_log_scale_delta"),
            policy.max_coarse_log_scale_delta,
            "coarse_scale_correction_too_large",
            reasons,
        )
        _maximum_gate(
            correction.get("max_rotation_delta_degrees"),
            policy.max_coarse_rotation_delta_degrees,
            "coarse_rotation_correction_too_large",
            reasons,
        )
        _maximum_gate(
            correction.get("max_translation_delta_fraction"),
            policy.max_coarse_translation_delta_fraction,
            "coarse_translation_correction_too_large",
            reasons,
        )
        if policy.require_convergence:
            for branch in ("no_loop", "loop"):
                if _mapping(solver.get(branch)).get("convergence_valid") is not True:
                    reasons.append(f"{branch}_solver_not_converged")
        if policy.require_no_active_bounds:
            for branch in ("no_loop", "loop"):
                active = _finite_int(
                    _mapping(solver.get(branch)).get("active_bound_count")
                )
                if active is None:
                    reasons.append(f"{branch}_active_bounds_missing")
                elif active != 0:
                    reasons.append(f"{branch}_active_bounds_present")
        if policy.require_finite:
            for branch in ("no_loop", "loop"):
                status = _mapping(solver.get(branch))
                if status.get("solution_finite") is not True:
                    reasons.append(f"{branch}_solution_nonfinite_or_missing")
                if status.get("residual_finite") is not True:
                    reasons.append(f"{branch}_residual_nonfinite_or_missing")

    enabled = not reasons
    return {
        "schema": "hip3r_adaptive_loop_decision_v1",
        "evidence_scope": "inference_only_no_gt",
        "loop_enabled": enabled,
        "selected_source": "loop" if enabled else "no_loop",
        "reasons": reasons if reasons else ["all_frozen_gates_passed"],
        "policy": asdict(policy),
        "policy_protocol": policy.protocol_metadata,
        "forward_protocol_id": score.get("forward_protocol_id"),
        "limitations": [
            "trusted checks are a frozen common set but are part of both solve objectives",
            "this policy is valid only after thresholds are frozen on a declared development split",
        ],
    }


def stable_adaptive_loop_features(score: Mapping[str, Any]) -> dict[str, Any]:
    """Derive numerically stable v2 features from a v1 common-set score.

    The adjacent-only solution is a chain and therefore has an adjacent RMSE
    close to machine precision.  It must not be used as a ratio denominator.
    The trusted-cost fraction instead normalizes the adjacent residual paid by
    the loop solution by the loop inconsistency present before correction.
    """

    if score.get("schema") != "hip3r_common_measurement_score_v1":
        raise ValueError("unsupported common-measurement score schema")
    solutions = _mapping(score.get("solutions"))
    no_loop = _mapping(solutions.get("no_loop"))
    loop = _mapping(solutions.get("loop"))
    no_loop_loop_rmse = _mapping(no_loop.get("loop_common")).get("rmse")
    loop_trusted_rmse = _mapping(loop.get("trusted_common")).get("rmse")
    trusted_cost_fraction = _safe_ratio(loop_trusted_rmse, no_loop_loop_rmse)
    return {
        "schema": "hip3r_adaptive_loop_stable_features_v2",
        "trusted_cost_fraction": trusted_cost_fraction,
        "trusted_cost_numerator": loop_trusted_rmse,
        "trusted_cost_denominator": no_loop_loop_rmse,
        "denominator_definition": "no_loop solution RMSE on admitted loop edges",
        "numerator_definition": "loop solution RMSE on common adjacent edges",
    }


def decide_adaptive_loop_v2(
    score: Mapping[str, Any], policy: AdaptiveLoopPolicyV2
) -> dict[str, Any]:
    """Apply the stable inference-only v2 policy to completed dual solutions."""

    if score.get("schema") != "hip3r_common_measurement_score_v1":
        raise ValueError("unsupported common-measurement score schema")
    comparisons = _mapping(score.get("comparisons"))
    correction = _mapping(comparisons.get("coarse_correction"))
    solver = _mapping(score.get("solver"))
    stable = stable_adaptive_loop_features(score)
    reasons: list[str] = []
    loop_measurements = _finite_int(
        _mapping(_mapping(score.get("measurement_sets")).get("loop_common")).get(
            "count"
        )
    )
    candidate_available = loop_measurements is not None and loop_measurements > 0
    if loop_measurements is None:
        reasons.append("loop_measurement_count_missing")
    elif loop_measurements <= 0:
        reasons.append("no_admitted_loop_measurements")

    route_results = {
        "small_correction": {"passed": False, "failed_gates": ["not_evaluated"]},
        "multi_loop_consensus": {
            "passed": False,
            "failed_gates": ["not_evaluated"],
        },
    }
    if candidate_available:
        _minimum_gate(
            comparisons.get("loop_rmse_reduction_fraction"),
            policy.min_loop_rmse_reduction_fraction,
            "loop_common_improvement_below_threshold",
            reasons,
        )
        _maximum_gate(
            stable.get("trusted_cost_fraction"),
            policy.max_trusted_cost_fraction,
            "trusted_common_cost_too_large",
            reasons,
        )
        small_failures: list[str] = []
        _maximum_gate(
            correction.get("max_log_scale_delta"),
            policy.max_small_log_scale_delta,
            "small_path_scale_correction_too_large",
            small_failures,
        )
        _maximum_gate(
            correction.get("max_rotation_delta_degrees"),
            policy.max_small_rotation_delta_degrees,
            "small_path_rotation_correction_too_large",
            small_failures,
        )
        _maximum_gate(
            correction.get("max_translation_delta_fraction"),
            policy.max_small_translation_delta_fraction,
            "small_path_translation_correction_too_large",
            small_failures,
        )
        consensus_failures: list[str] = []
        if loop_measurements < policy.min_consensus_loop_measurements:
            consensus_failures.append("consensus_path_insufficient_loop_measurements")
        _minimum_gate(
            comparisons.get("loop_rmse_reduction_fraction"),
            policy.min_consensus_loop_rmse_reduction_fraction,
            "consensus_path_loop_improvement_below_threshold",
            consensus_failures,
        )
        _maximum_gate(
            correction.get("max_log_scale_delta"),
            policy.max_consensus_log_scale_delta,
            "consensus_path_scale_correction_too_large",
            consensus_failures,
        )
        _maximum_gate(
            correction.get("max_rotation_delta_degrees"),
            policy.max_consensus_rotation_delta_degrees,
            "consensus_path_rotation_correction_too_large",
            consensus_failures,
        )
        _maximum_gate(
            correction.get("max_translation_delta_fraction"),
            policy.max_consensus_translation_delta_fraction,
            "consensus_path_translation_correction_too_large",
            consensus_failures,
        )
        route_results = {
            "small_correction": {
                "passed": not small_failures,
                "failed_gates": small_failures,
            },
            "multi_loop_consensus": {
                "passed": not consensus_failures,
                "failed_gates": consensus_failures,
            },
        }
        if small_failures and consensus_failures:
            reasons.append("no_adaptive_acceptance_route_passed")
        if policy.require_convergence:
            for branch in ("no_loop", "loop"):
                if _mapping(solver.get(branch)).get("convergence_valid") is not True:
                    reasons.append(f"{branch}_solver_not_converged")
        if policy.require_no_active_bounds:
            for branch in ("no_loop", "loop"):
                active = _finite_int(
                    _mapping(solver.get(branch)).get("active_bound_count")
                )
                if active is None:
                    reasons.append(f"{branch}_active_bounds_missing")
                elif active != 0:
                    reasons.append(f"{branch}_active_bounds_present")
        if policy.require_finite:
            for branch in ("no_loop", "loop"):
                status = _mapping(solver.get(branch))
                if status.get("solution_finite") is not True:
                    reasons.append(f"{branch}_solution_nonfinite_or_missing")
                if status.get("residual_finite") is not True:
                    reasons.append(f"{branch}_residual_nonfinite_or_missing")

    enabled = not reasons
    selected_route = None
    if enabled:
        selected_route = (
            "small_correction"
            if route_results["small_correction"]["passed"]
            else "multi_loop_consensus"
        )
    return {
        "schema": "hip3r_adaptive_loop_decision_v2",
        "evidence_scope": "inference_only_no_gt",
        "loop_enabled": enabled,
        "selected_source": "loop" if enabled else "no_loop",
        "selected_route": selected_route,
        "reasons": reasons if reasons else ["all_frozen_gates_passed"],
        "route_results": route_results,
        "stable_features": stable,
        "policy": asdict(policy),
        "policy_protocol": policy.protocol_metadata,
        "forward_protocol_id": score.get("forward_protocol_id"),
        "limitations": [
            "trusted checks are a frozen common set but are part of both solve objectives",
            "multi-loop count is support evidence, not a proof that every loop is correct",
            "thresholds require a new unseen-benchmark audit after protocol development",
        ],
    }


def decide_adaptive_loop_for_policy(
    score: Mapping[str, Any], policy: AdaptiveLoopPolicy | AdaptiveLoopPolicyV2
) -> dict[str, Any]:
    """Dispatch a frozen adaptive-loop policy without weakening schema checks."""

    if isinstance(policy, AdaptiveLoopPolicyV2):
        return decide_adaptive_loop_v2(score, policy)
    if isinstance(policy, AdaptiveLoopPolicy):
        return decide_adaptive_loop(score, policy)
    raise TypeError(f"unsupported adaptive-loop policy type: {type(policy)!r}")


def frozen_policy_protocol_metadata(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the scientific fields of a frozen calibration manifest."""

    schema = manifest.get("schema")
    if schema == "hip3r_frozen_adaptive_loop_policy_v2":
        keys = (
            "schema",
            "frozen",
            "implementation_commit",
            "development_split",
            "threshold_origin",
            "calibration_artifact",
            "calibration_protocol",
            "policy",
            "policy_protocol",
        )
    else:
        keys = (
            "schema",
            "frozen",
            "development_split",
            "calibration_artifact",
            "calibration_protocol",
            "policy",
            "policy_protocol",
        )
    return {key: manifest.get(key) for key in keys}


def _coarse_alignments(manifest: Mapping[str, Any]) -> list[Sim3]:
    basis = _mapping(manifest.get("basis"))
    values = basis.get("coarse_alignments")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("basis coarse_alignments are missing")
    return [_sim3_from_record(item) for item in values]


def _edge_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = manifest.get("edges")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("basis edges are missing")
    records = []
    for value in values:
        item = _mapping(value)
        kind = str(item.get("kind"))
        if kind not in {"adjacent", "loop"}:
            raise ValueError(f"unsupported coarse edge kind: {kind!r}")
        diagnostics = dict(_mapping(item.get("diagnostics")))
        weight = _finite_float(diagnostics.get("weight", 1.0))
        if weight is None or weight < 0.0:
            raise ValueError("edge weight must be finite and nonnegative")
        records.append(
            {
                "source": int(item["source"]),
                "target": int(item["target"]),
                "kind": kind,
                "transform": _sim3_from_record(item.get("transform")),
                "weight": weight,
            }
        )
    return records


def _score_edges(
    edges: Sequence[Mapping[str, Any]],
    alignments: Sequence[Sim3],
    *,
    translation_scale: float,
) -> dict[str, Any]:
    if not edges:
        return {
            "count": 0,
            "rmse": None,
            "median_edge_norm": None,
            "p95_edge_norm": None,
            "max_edge_norm": None,
        }
    residuals = []
    norms = []
    for edge in edges:
        predicted = (
            alignments[int(edge["target"])]
            .inverse()
            .compose(alignments[int(edge["source"])])
        )
        measured = edge["transform"]
        assert isinstance(measured, Sim3)
        error = measured.inverse().compose(predicted)
        vector = np.concatenate(
            (
                Rotation.from_matrix(error.rotation).as_rotvec(),
                error.translation * float(translation_scale),
                np.asarray([np.log(abs(error.scale))], dtype=np.float64),
            )
        ) * np.sqrt(float(edge["weight"]))
        if not np.all(np.isfinite(vector)):
            raise ValueError("common measurement produced non-finite residual")
        residuals.append(vector)
        norms.append(float(np.linalg.norm(vector)))
    matrix = np.asarray(residuals, dtype=np.float64).reshape(len(edges), 7)
    edge_norms = np.asarray(norms, dtype=np.float64)
    return {
        "count": len(edges),
        "rmse": float(np.sqrt(np.mean(np.square(matrix)))),
        "median_edge_norm": float(np.median(edge_norms)),
        "p95_edge_norm": float(np.quantile(edge_norms, 0.95)),
        "max_edge_norm": float(np.max(edge_norms)),
    }


def _sim3_from_record(value: Any) -> Sim3:
    record = _mapping(value)
    matrix = np.asarray(record.get("matrix"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("serialized Sim3 matrix must be finite 4x4")
    linear = matrix[:3, :3]
    determinant = float(np.linalg.det(linear))
    if not np.isfinite(determinant) or determinant <= 0.0:
        raise ValueError("serialized Sim3 must have positive determinant")
    scale = float(np.cbrt(determinant))
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=1e-6):
        raise ValueError("serialized Sim3 rotation is not orthonormal")
    return Sim3(scale=scale, rotation=rotation, translation=matrix[:3, 3])


def _solver_status(manifest: Mapping[str, Any]) -> dict[str, Any]:
    basis = _mapping(_mapping(manifest.get("basis")).get("diagnostics"))
    convergence = _mapping(basis.get("convergence_gate"))
    coarse = _mapping(basis.get("coarse_anchor"))
    return {
        "convergence_valid": convergence.get("valid"),
        "solution_finite": basis.get("solution_finite"),
        "residual_finite": basis.get("residual_finite"),
        "active_bound_count": coarse.get("active_bound_count"),
        "basis_protocol_id": manifest.get("basis_protocol_id"),
    }


def _measurement_records(edges: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": int(edge["source"]),
            "target": int(edge["target"]),
            "kind": str(edge["kind"]),
            "matrix": edge["transform"].as_matrix().tolist(),
            "weight": float(edge["weight"]),
        }
        for edge in edges
    ]


def _measurement_identity(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": int(record["source"]),
            "target": int(record["target"]),
            "kind": str(record["kind"]),
            "weight": float(record["weight"]),
        }
        for record in records
    ]


def _coarse_correction(a: Sequence[Sim3], b: Sequence[Sim3]) -> dict[str, float | None]:
    scale_values = [
        abs(np.log(abs(right.scale / left.scale))) for left, right in zip(a, b)
    ]
    rotation_values = [
        float(
            np.degrees(
                np.linalg.norm(
                    Rotation.from_matrix(right.rotation @ left.rotation.T).as_rotvec()
                )
            )
        )
        for left, right in zip(a, b)
    ]
    translation_values = [
        float(np.linalg.norm(right.translation - left.translation))
        for left, right in zip(a, b)
    ]
    if a:
        root = a[0].translation
        baseline_span = max(
            float(np.linalg.norm(item.translation - root)) for item in a
        )
    else:
        baseline_span = 0.0
    maximum_translation = max(translation_values, default=0.0)
    normalized_translation = (
        float(maximum_translation / baseline_span)
        if baseline_span > 0.0
        else (0.0 if maximum_translation <= 0.0 else None)
    )
    return {
        "max_log_scale_delta": float(max(scale_values, default=0.0)),
        "max_rotation_delta_degrees": float(max(rotation_values, default=0.0)),
        "max_translation_delta": float(maximum_translation),
        "baseline_translation_span": float(baseline_span),
        "max_translation_delta_fraction": normalized_translation,
    }


def _relative_reduction(baseline: float, candidate: float) -> float | None:
    if baseline <= 0.0:
        return 0.0 if candidate <= 0.0 else None
    return float((baseline - candidate) / baseline)


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    resolved_numerator = _finite_float(numerator)
    resolved_denominator = _finite_float(denominator)
    if resolved_numerator is None or resolved_denominator is None:
        return None
    if resolved_denominator > 0.0:
        return float(resolved_numerator / resolved_denominator)
    return 1.0 if resolved_numerator <= 0.0 else None


def _minimum_gate(
    value: Any, threshold: float, reason: str, reasons: list[str]
) -> None:
    resolved = _finite_float(value)
    if resolved is None:
        reasons.append(f"missing_{reason}")
    elif resolved < threshold:
        reasons.append(reason)


def _maximum_gate(
    value: Any, threshold: float, reason: str, reasons: list[str]
) -> None:
    resolved = _finite_float(value)
    if resolved is None:
        reasons.append(f"missing_{reason}")
    elif resolved > threshold:
        reasons.append(reason)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _finite_int(value: Any) -> int | None:
    number = _finite_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


__all__ = [
    "AdaptiveLoopPolicy",
    "AdaptiveLoopPolicyV2",
    "decide_adaptive_loop",
    "decide_adaptive_loop_for_policy",
    "decide_adaptive_loop_v2",
    "frozen_policy_protocol_metadata",
    "score_common_measurement_sets",
    "stable_adaptive_loop_features",
]
