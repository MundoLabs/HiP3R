from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import dataclasses
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from long_reconstruction.sim3 import (
    Sim3,
    _EPS as SIM3_EPS,
    _estimate_sim3_finite_validated_paired_confidence,
    estimate_sim3,
)


POINT_FIELDS = ("world_points", "points", "pts3d", "point_map")
CONFIDENCE_FIELDS = ("world_points_conf", "confidence", "conf", "points_conf", "depth_conf")
C2W_FIELDS = ("extrinsic_c2w", "c2w", "c2ws", "camera_poses", "camera_pose", "poses")
_ADJACENT_RANGE_PAIRING_POLICY = "global_frame_intersection_v1"
_ADJACENT_LEGACY_PAIRING_POLICY = "legacy_local_tail_head_v1"


@dataclass(frozen=True)
class Sim3Edge:
    source: int
    target: int
    transform: Sim3
    kind: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PostProcessResult:
    chunks: list[Any]
    alignments: list[Sim3]
    edges: list[Sim3Edge]
    diagnostics: dict[str, Any]
    frame_poses: list[Sim3] | None = None


def _coerce_adjacent_range(
    name: str,
    value: Any,
    *,
    num_frames: int,
) -> tuple[int, int]:
    if isinstance(value, Mapping):
        start = int(value["start"])
        end = int(value["end"])
    elif hasattr(value, "start") and hasattr(value, "end"):
        start = int(value.start)
        end = int(value.end)
    else:
        values = list(value)
        if len(values) != 2:
            raise ValueError(f"{name} range must have start and end")
        start = int(values[0])
        end = int(values[1])
    if start < 0 or end <= start:
        raise ValueError(f"{name} range must satisfy 0 <= start < end")
    if end - start != int(num_frames):
        raise ValueError(
            f"{name} range length {end - start} does not match chunk frame "
            f"count {num_frames}"
        )
    return start, end


def _adjacent_pairing_slices(
    *,
    overlap: int,
    previous_num_frames: int,
    next_num_frames: int,
    previous_range: Any | None,
    next_range: Any | None,
) -> tuple[slice, slice, int, dict[str, Any]]:
    if (previous_range is None) != (next_range is None):
        raise ValueError("previous_range and next_range must be provided together")

    if previous_range is None:
        if overlap > previous_num_frames or overlap > next_num_frames:
            raise ValueError(
                "overlap cannot exceed either adjacent chunk length: "
                f"overlap={overlap}, previous={previous_num_frames}, next={next_num_frames}"
            )
        previous_start = max(previous_num_frames - max(overlap, 0), 0)
        next_end = max(overlap, 0)
        return (
            slice(-overlap, None) if overlap > 0 else slice(0, 0),
            slice(0, next_end),
            int(overlap),
            {
                "pairing_policy": _ADJACENT_LEGACY_PAIRING_POLICY,
                "requested_overlap": int(overlap),
                "previous_local_start": int(previous_start),
                "previous_local_end": int(previous_num_frames),
                "next_local_start": 0,
                "next_local_end": int(next_end),
            },
        )

    if overlap <= 0:
        raise ValueError("range-aware adjacent pairing requires positive overlap")

    previous_global_start, previous_global_end = _coerce_adjacent_range(
        "previous",
        previous_range,
        num_frames=previous_num_frames,
    )
    next_global_start, next_global_end = _coerce_adjacent_range(
        "next",
        next_range,
        num_frames=next_num_frames,
    )
    intersection_start = max(previous_global_start, next_global_start)
    intersection_end = min(previous_global_end, next_global_end)
    intersection_length = intersection_end - intersection_start
    if intersection_length <= 0:
        raise ValueError(
            "adjacent chunk ranges do not have a shared global-frame intersection: "
            f"previous={previous_global_start}:{previous_global_end}, "
            f"next={next_global_start}:{next_global_end}"
        )

    effective_overlap = min(int(overlap), int(intersection_length))
    global_start = intersection_end - effective_overlap
    global_end = intersection_end
    previous_local_start = global_start - previous_global_start
    previous_local_end = global_end - previous_global_start
    next_local_start = global_start - next_global_start
    next_local_end = global_end - next_global_start
    if (
        previous_local_start < 0
        or previous_local_end > previous_num_frames
        or next_local_start < 0
        or next_local_end > next_num_frames
        or previous_local_end - previous_local_start != effective_overlap
        or next_local_end - next_local_start != effective_overlap
    ):
        raise ValueError("adjacent global-frame pairing produced invalid local slices")

    diagnostics = {
        "pairing_policy": _ADJACENT_RANGE_PAIRING_POLICY,
        "requested_overlap": int(overlap),
        "actual_intersection": int(intersection_length),
        "global_start": int(global_start),
        "global_end": int(global_end),
        "previous_global_start": int(previous_global_start),
        "previous_global_end": int(previous_global_end),
        "next_global_start": int(next_global_start),
        "next_global_end": int(next_global_end),
        "previous_local_start": int(previous_local_start),
        "previous_local_end": int(previous_local_end),
        "next_local_start": int(next_local_start),
        "next_local_end": int(next_local_end),
    }
    return (
        slice(previous_local_start, previous_local_end),
        slice(next_local_start, next_local_end),
        int(effective_overlap),
        diagnostics,
    )


def estimate_adjacent_edge(
    previous_chunk: Any,
    next_chunk: Any,
    *,
    overlap: int,
    previous_range: Any | None = None,
    next_range: Any | None = None,
    previous_index: int = 0,
    next_index: int = 1,
    use_scale: bool = True,
    confidence_threshold: float | None = 0.0,
    max_points: int | None = None,
    max_iterations: int = 10,
    huber_delta: float | None = None,
    stability_strata: int = 0,
    stability_frame_indices: Sequence[int] | None = None,
) -> Sim3Edge:
    """Estimate an adjacent edge with ``next_chunk -> previous_chunk`` direction."""

    previous_num_frames = _get_points(previous_chunk).shape[0]
    next_num_frames = _get_points(next_chunk).shape[0]
    previous_slice, next_slice, effective_overlap, pairing_diagnostics = (
        _adjacent_pairing_slices(
            overlap=int(overlap),
            previous_num_frames=previous_num_frames,
            next_num_frames=next_num_frames,
            previous_range=previous_range,
            next_range=next_range,
        )
    )
    if overlap <= 0:
        return Sim3Edge(
            source=next_index,
            target=previous_index,
            transform=Sim3.identity(),
            kind="adjacent",
            diagnostics={
                "status": "no_overlap",
                "overlap": int(overlap),
                "direction": "next_to_previous",
                **pairing_diagnostics,
            },
        )

    previous_points = _get_points(previous_chunk)[previous_slice]
    next_points = _get_points(next_chunk)[next_slice]
    next_confidence = _get_optional_confidence(next_chunk, frame_slice=next_slice)
    previous_confidence = _get_optional_confidence(
        previous_chunk, frame_slice=previous_slice
    )
    finite_validated = _has_finite_validated_fields(
        previous_chunk, "world_points", "world_points_conf"
    ) and _has_finite_validated_fields(
        next_chunk, "world_points", "world_points_conf"
    )
    if finite_validated and next_confidence is not None and previous_confidence is not None:
        transform, diagnostics = _estimate_sim3_finite_validated_paired_confidence(
            next_points,
            previous_points,
            next_confidence,
            previous_confidence,
            use_scale=use_scale,
            confidence_threshold=confidence_threshold,
            max_points=max_points,
            max_iterations=max_iterations,
            huber_delta=huber_delta,
        )
    else:
        transform, diagnostics = estimate_sim3(
            next_points,
            previous_points,
            _combine_confidence(next_confidence, previous_confidence),
            use_scale=use_scale,
            confidence_threshold=confidence_threshold,
            max_points=max_points,
            max_iterations=max_iterations,
            huber_delta=huber_delta,
        )
    if stability_strata:
        frame_indices = (
            list(range(int(pairing_diagnostics.get("global_start", 0)),
                       int(pairing_diagnostics.get("global_end", effective_overlap))))
            if stability_frame_indices is None
            else list(stability_frame_indices)
        )
        stability, _ = _sim3_stability_diagnostics(
            next_points,
            previous_points,
            next_confidence,
            previous_confidence,
            full_transform=transform,
            frame_indices=frame_indices,
            num_strata=stability_strata,
            finite_validated=finite_validated,
            use_scale=use_scale,
            confidence_threshold=confidence_threshold,
            max_points=max_points,
            max_iterations=max_iterations,
            huber_delta=huber_delta,
        )
        diagnostics["estimator_stability"] = stability
    diagnostics.update(
        {
            "overlap": int(effective_overlap),
            "direction": "next_to_previous",
            **pairing_diagnostics,
        }
    )
    return Sim3Edge(
        source=next_index,
        target=previous_index,
        transform=transform,
        kind="adjacent",
        diagnostics=diagnostics,
    )


def estimate_loop_edge(
    joint_prediction: Any,
    endpoint_a_slice: Any,
    endpoint_b_slice: Any,
    *,
    joint_a_slice: Any | None = None,
    joint_b_slice: Any | None = None,
    endpoint_a_index: int = 0,
    endpoint_b_index: int = 1,
    use_scale: bool = True,
    confidence_threshold: float | None = 0.0,
    max_points: int | None = None,
    max_iterations: int = 10,
    huber_delta: float | None = None,
    endpoint_workers: int = 1,
    stability_strata: int = 0,
    endpoint_a_frame_indices: Sequence[int] | None = None,
    endpoint_b_frame_indices: Sequence[int] | None = None,
) -> Sim3Edge:
    """Estimate a loop edge as ``endpoint A -> endpoint B``.

    ``joint_prediction`` is in the joint-forward coordinate frame. The endpoint
    slices are the corresponding local predictions for endpoint A and endpoint B.
    If joint slices are omitted, the joint prediction is split as A frames followed
    by B frames.
    """

    joint_points = _get_points(joint_prediction)
    a_points = _get_points(endpoint_a_slice)
    b_points = _get_points(endpoint_b_slice)
    if joint_a_slice is None or joint_b_slice is None:
        num_a = a_points.shape[0]
        num_b = b_points.shape[0]
        joint_a_slice = slice(0, num_a)
        joint_b_slice = slice(num_a, num_a + num_b)

    joint_a_points = joint_points[joint_a_slice]
    joint_b_points = joint_points[joint_b_slice]
    if endpoint_workers < 1:
        raise ValueError("endpoint_workers must be positive")
    endpoint_tasks = (
        (
            joint_a_points,
            a_points,
            _get_optional_confidence(joint_prediction, frame_slice=joint_a_slice),
            _get_optional_confidence(endpoint_a_slice),
            _has_finite_validated_fields(joint_prediction, "world_points", "world_points_conf")
            and _has_finite_validated_fields(endpoint_a_slice, "world_points", "world_points_conf"),
        ),
        (
            joint_b_points,
            b_points,
            _get_optional_confidence(joint_prediction, frame_slice=joint_b_slice),
            _get_optional_confidence(endpoint_b_slice),
            _has_finite_validated_fields(joint_prediction, "world_points", "world_points_conf")
            and _has_finite_validated_fields(endpoint_b_slice, "world_points", "world_points_conf"),
        ),
    )

    def estimate_endpoint(task):
        source, target, source_confidence, target_confidence, finite_validated = task
        if finite_validated and source_confidence is not None and target_confidence is not None:
            return _estimate_sim3_finite_validated_paired_confidence(
                source,
                target,
                source_confidence,
                target_confidence,
                use_scale=use_scale,
                confidence_threshold=confidence_threshold,
                max_points=max_points,
                max_iterations=max_iterations,
                huber_delta=huber_delta,
            )
        return estimate_sim3(
            source,
            target,
            _combine_confidence(source_confidence, target_confidence),
            use_scale=use_scale,
            confidence_threshold=confidence_threshold,
            max_points=max_points,
            max_iterations=max_iterations,
            huber_delta=huber_delta,
        )

    if endpoint_workers > 1:
        with ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="hip3r-loop-endpoint",
        ) as executor:
            endpoint_results = list(executor.map(estimate_endpoint, endpoint_tasks))
    else:
        endpoint_results = [estimate_endpoint(task) for task in endpoint_tasks]
    (joint_to_a, diag_a), (joint_to_b, diag_b) = endpoint_results

    a_to_b = joint_to_b.compose(joint_to_a.inverse())
    loop_stability = None
    if stability_strata:
        stability_a, transforms_a = _sim3_stability_diagnostics(
            joint_a_points,
            a_points,
            endpoint_tasks[0][2],
            endpoint_tasks[0][3],
            full_transform=joint_to_a,
            frame_indices=(
                list(range(joint_a_points.shape[0]))
                if endpoint_a_frame_indices is None
                else list(endpoint_a_frame_indices)
            ),
            num_strata=stability_strata,
            finite_validated=endpoint_tasks[0][4],
            use_scale=use_scale,
            confidence_threshold=confidence_threshold,
            max_points=max_points,
            max_iterations=max_iterations,
            huber_delta=huber_delta,
        )
        stability_b, transforms_b = _sim3_stability_diagnostics(
            joint_b_points,
            b_points,
            endpoint_tasks[1][2],
            endpoint_tasks[1][3],
            full_transform=joint_to_b,
            frame_indices=(
                list(range(joint_b_points.shape[0]))
                if endpoint_b_frame_indices is None
                else list(endpoint_b_frame_indices)
            ),
            num_strata=stability_strata,
            finite_validated=endpoint_tasks[1][4],
            use_scale=use_scale,
            confidence_threshold=confidence_threshold,
            max_points=max_points,
            max_iterations=max_iterations,
            huber_delta=huber_delta,
        )
        propagated = []
        for index, (fit_a, fit_b) in enumerate(zip(transforms_a, transforms_b)):
            if fit_a is None or fit_b is None:
                propagated.append({"stratum": index, "status": "subedge_failed"})
                continue
            propagated.append(_sim3_disagreement(
                fit_b.compose(fit_a.inverse()), a_to_b, index,
                float(stability_b["target_radius"]),
            ))
        ok = [item for item in propagated if item["status"] == "ok"]
        loop_stability = {
            "schema": "sim3_estimator_stability/v1",
            "num_strata": int(stability_strata),
            "successful_strata": len(ok),
            "status": "ok" if len(ok) == stability_strata else "failed",
            "uncertainty": (
                float(np.median([item["dimensionless_disagreement"] for item in ok]))
                if len(ok) == stability_strata else None
            ),
            "target_radius": float(stability_b["target_radius"]),
            "strata": propagated,
            "endpoint_a": stability_a,
            "endpoint_b": stability_b,
        }
    direction_error = _direction_error(joint_points, joint_to_a, joint_to_b, a_to_b)
    residual_quality_quantile = next(
        (
            key
            for key in ("residual_p95", "residual_p90")
            if all(diagnostics.get(key) is not None for diagnostics in (diag_a, diag_b))
        ),
        None,
    )
    endpoint_residuals = (
        [
            float(diagnostics[residual_quality_quantile])
            for diagnostics in (diag_a, diag_b)
        ]
        if residual_quality_quantile is not None
        else []
    )
    diagnostics = {
        "status": "ok" if diag_a.get("status") == "ok" and diag_b.get("status") == "ok" else "subedge_failed",
        "direction": "endpoint_a_to_endpoint_b",
        "joint_to_a": diag_a,
        "joint_to_b": diag_b,
        "direction_error": direction_error,
        "residual_p95": (
            max(endpoint_residuals)
            if residual_quality_quantile == "residual_p95"
            else None
        ),
        "residual_quality": max(endpoint_residuals) if endpoint_residuals else None,
        "residual_quality_quantile": residual_quality_quantile,
        "endpoint_workers": 2 if endpoint_workers > 1 else 1,
    }
    if loop_stability is not None:
        diagnostics["estimator_stability"] = loop_stability
    return Sim3Edge(
        source=endpoint_a_index,
        target=endpoint_b_index,
        transform=a_to_b,
        kind="loop",
        diagnostics=diagnostics,
    )


def _stratum_ids(points: np.ndarray, frame_indices: Sequence[int], num_strata: int) -> np.ndarray:
    """Assign every dense correspondence to a deterministic spatial-temporal stratum."""
    shape = np.asarray(points).shape
    if num_strata < 2:
        raise ValueError("stability strata must be at least two")
    if len(shape) < 2 or shape[-1] != 3:
        raise ValueError(f"points must end in xyz, got {shape}")
    if len(frame_indices) != shape[0]:
        raise ValueError("stability frame index count differs from point frames")
    spatial_shape = shape[1:-1]
    spatial_size = int(np.prod(spatial_shape, dtype=np.int64))
    frame = np.repeat(np.asarray(frame_indices, dtype=np.int64), spatial_size)
    if len(spatial_shape) >= 2:
        y, x = np.indices(spatial_shape[-2:], dtype=np.int64)
        spatial = (y * 19349663 + x * 83492791).reshape(-1)
        if len(spatial_shape) > 2:
            prefix = int(np.prod(spatial_shape[:-2], dtype=np.int64))
            spatial = np.tile(spatial, prefix)
    else:
        spatial = np.arange(spatial_size, dtype=np.int64) * 83492791
    spatial = np.tile(spatial, shape[0])
    return np.mod(frame * 73856093 + spatial, num_strata)


def _robust_target_radius(target: np.ndarray, max_points: int | None) -> float:
    flat = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(flat).all(axis=1)
    flat = flat[finite]
    if flat.shape[0] == 0:
        return 1.0
    limit = flat.shape[0] if max_points is None else min(flat.shape[0], int(max_points))
    if flat.shape[0] > limit:
        flat = flat[np.linspace(0, flat.shape[0] - 1, limit, dtype=np.int64)]
    center = np.median(flat, axis=0)
    radius = float(np.median(np.linalg.norm(flat - center, axis=1)))
    return max(radius, SIM3_EPS)


def _sim3_disagreement(
    estimate: Sim3,
    reference: Sim3,
    stratum: int,
    target_radius: float,
) -> dict[str, Any]:
    relative_rotation = estimate.rotation @ reference.rotation.T
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    components = {
        "log_scale": abs(float(np.log(abs(estimate.scale / reference.scale)))),
        "rotation_radians": abs(float(np.arccos(cosine))),
        "translation_over_target_radius": (
            float(np.linalg.norm(estimate.translation - reference.translation))
            / max(float(target_radius), SIM3_EPS)
        ),
    }
    return {
        "stratum": int(stratum),
        "status": "ok",
        **components,
        "dimensionless_disagreement": float(np.median(list(components.values()))),
    }


def _sim3_stability_diagnostics(
    source: np.ndarray,
    target: np.ndarray,
    source_confidence: np.ndarray | None,
    target_confidence: np.ndarray | None,
    *,
    full_transform: Sim3,
    frame_indices: Sequence[int],
    num_strata: int,
    finite_validated: bool,
    use_scale: bool,
    confidence_threshold: float | None,
    max_points: int | None,
    max_iterations: int,
    huber_delta: float | None,
) -> tuple[dict[str, Any], list[Sim3 | None]]:
    ids = _stratum_ids(source, frame_indices, num_strata)
    source_flat = np.asarray(source).reshape(-1, 3)
    target_flat = np.asarray(target).reshape(-1, 3)
    source_conf_flat = None if source_confidence is None else np.asarray(source_confidence).reshape(-1)
    target_conf_flat = None if target_confidence is None else np.asarray(target_confidence).reshape(-1)
    target_radius = _robust_target_radius(target, max_points)
    rows: list[dict[str, Any]] = []
    transforms: list[Sim3 | None] = []
    for stratum in range(num_strata):
        selected = ids == stratum
        if finite_validated and source_conf_flat is not None and target_conf_flat is not None:
            fitted, fit_diag = _estimate_sim3_finite_validated_paired_confidence(
                source_flat[selected], target_flat[selected],
                source_conf_flat[selected], target_conf_flat[selected],
                use_scale=use_scale, confidence_threshold=confidence_threshold,
                max_points=max_points, max_iterations=max_iterations,
                huber_delta=huber_delta,
            )
        else:
            confidence = _combine_confidence(
                None if source_conf_flat is None else source_conf_flat[selected],
                None if target_conf_flat is None else target_conf_flat[selected],
            )
            fitted, fit_diag = estimate_sim3(
                source_flat[selected], target_flat[selected], confidence,
                use_scale=use_scale, confidence_threshold=confidence_threshold,
                max_points=max_points, max_iterations=max_iterations,
                huber_delta=huber_delta,
            )
        if fit_diag.get("status") != "ok":
            rows.append({"stratum": stratum, "status": str(fit_diag.get("status")),
                         "num_input": int(selected.sum())})
            transforms.append(None)
            continue
        rows.append({**_sim3_disagreement(fitted, full_transform, stratum, target_radius),
                     "num_input": int(selected.sum()),
                     "num_used": int(fit_diag.get("num_used", selected.sum()))})
        transforms.append(fitted)
    ok = [row for row in rows if row["status"] == "ok"]
    payload = {
        "schema": "sim3_estimator_stability/v1",
        "partition": "global_frame_xy_hash_mod_v1",
        "aggregation": "median_components_then_median_strata_v1",
        "num_strata": int(num_strata),
        "successful_strata": len(ok),
        "status": "ok" if len(ok) == num_strata else "failed",
        "uncertainty": (
            float(np.median([row["dimensionless_disagreement"] for row in ok]))
            if len(ok) == num_strata else None
        ),
        "target_radius": target_radius,
        "strata": rows,
    }
    return payload, transforms


def residual_edge_weight(
    diagnostics: dict[str, Any],
    *,
    residual_key: str = "residual_p95",
    residual_scale: float = 0.05,
    min_weight: float = 0.05,
    max_weight: float = 1.0,
) -> float:
    """Map an hip3r endpoint residual to a bounded scalar pose-graph weight."""
    residual = diagnostics.get(residual_key)
    if residual is None:
        return 1.0
    scale = max(float(residual_scale), 1e-8)
    weight = 1.0 / (1.0 + max(float(residual), 0.0) / scale)
    return float(np.clip(weight, min_weight, max_weight))


@dataclass
class ReconstructionPostProcessor:
    overlap: int
    use_scale: bool = True
    confidence_threshold: float | None = 0.0
    max_points: int | None = 20000
    max_iterations: int = 10
    huber_delta: float | None = None
    stability_strata: int = 0
    loop_constraints: Sequence[Sim3Edge] | None = None
    pose_graph_optimizer: Callable[..., Sequence[Sim3] | dict[str, Any]] | None = None
    use_jac_sparsity: bool = False
    chain_global_scale_gauge_centering_v1: bool = False
    chain_global_scale_gauge_centering_span_cap: float = 50.0
    chain_global_scale_gauge_centering_defer_translation_gate_to_factorized_v4: (
        bool
    ) = False

    def __call__(
        self,
        chunks: Sequence[Any],
        *,
        chunk_ranges: Sequence[Any] | None = None,
        loop_constraints: Sequence[Sim3Edge] | None = None,
        pose_graph_optimizer: Callable[..., Sequence[Sim3] | dict[str, Any]] | None = None,
    ) -> PostProcessResult:
        return self.process(
            chunks,
            chunk_ranges=chunk_ranges,
            loop_constraints=loop_constraints,
            pose_graph_optimizer=pose_graph_optimizer,
        )

    def process(
        self,
        chunks: Sequence[Any],
        *,
        chunk_ranges: Sequence[Any] | None = None,
        adjacent_constraints: Sequence[Sim3Edge] | None = None,
        loop_constraints: Sequence[Sim3Edge] | None = None,
        pose_graph_optimizer: Callable[..., Sequence[Sim3] | dict[str, Any]] | None = None,
    ) -> PostProcessResult:
        chunks_list = list(chunks)
        if not chunks_list:
            return PostProcessResult(
                chunks=[],
                alignments=[],
                edges=[],
                diagnostics={"status": "empty", "optimizer_used": False},
            )

        adjacent_edges = (
            list(adjacent_constraints)
            if adjacent_constraints is not None
            else self.build_adjacent_edges(chunks_list, chunk_ranges=chunk_ranges)
        )
        chain_gauge_diagnostics: dict[str, Any] | None = None
        if self.chain_global_scale_gauge_centering_v1:
            initial_alignments, chain_gauge_diagnostics = (
                self._chain_adjacent_alignments_global_scale_gauge_centered(
                    len(chunks_list),
                    adjacent_edges,
                    span_cap=self.chain_global_scale_gauge_centering_span_cap,
                    defer_translation_residual_to_factorized_frame_chart=(
                        self
                        .chain_global_scale_gauge_centering_defer_translation_gate_to_factorized_v4
                    ),
                )
            )
        else:
            initial_alignments = self._chain_adjacent_alignments(len(chunks_list), adjacent_edges)
        loops = list(loop_constraints if loop_constraints is not None else self.loop_constraints or [])
        optimizer = pose_graph_optimizer if pose_graph_optimizer is not None else self.pose_graph_optimizer

        if optimizer is None:
            alignments = initial_alignments
            optimizer_used = False
        else:
            alignments = _coerce_alignments(
                _call_optimizer(
                    optimizer,
                    num_chunks=len(chunks_list),
                    edges=adjacent_edges + loops,
                    initial_alignments=initial_alignments,
                    use_jac_sparsity=self.use_jac_sparsity,
                ),
                len(chunks_list),
            )
            optimizer_used = True

        aligned_chunks = [_apply_alignment(chunk, alignment) for chunk, alignment in zip(chunks_list, alignments)]
        diagnostics = {
            "status": "ok",
            "num_chunks": len(chunks_list),
            "num_adjacent_edges": len(adjacent_edges),
            "num_loop_edges": len(loops),
            "optimizer_used": optimizer_used,
            "loop_edges_ignored": bool(loops and not optimizer_used),
        }
        if chain_gauge_diagnostics is not None:
            diagnostics["chain_global_scale_gauge_centering_v1"] = chain_gauge_diagnostics
        return PostProcessResult(
            chunks=aligned_chunks,
            alignments=alignments,
            edges=adjacent_edges + (loops if optimizer_used else []),
            diagnostics=diagnostics,
        )

    def build_adjacent_edges(
        self,
        chunks: Sequence[Any],
        *,
        chunk_ranges: Sequence[Any] | None = None,
        workers: int = 1,
    ) -> list[Sim3Edge]:
        if workers < 1:
            raise ValueError("workers must be positive")
        if chunk_ranges is not None and len(chunk_ranges) != len(chunks):
            raise ValueError(
                f"chunk_ranges count {len(chunk_ranges)} != chunks count {len(chunks)}"
            )

        def build_one(index: int) -> Sim3Edge:
            return estimate_adjacent_edge(
                chunks[index - 1],
                chunks[index],
                overlap=self.overlap,
                previous_range=(
                    None if chunk_ranges is None else chunk_ranges[index - 1]
                ),
                next_range=None if chunk_ranges is None else chunk_ranges[index],
                previous_index=index - 1,
                next_index=index,
                use_scale=self.use_scale,
                confidence_threshold=self.confidence_threshold,
                max_points=self.max_points,
                max_iterations=self.max_iterations,
                huber_delta=self.huber_delta,
                stability_strata=self.stability_strata,
            )

        indices = range(1, len(chunks))
        actual_workers = min(workers, max(len(chunks) - 1, 0))
        if actual_workers > 1:
            with ThreadPoolExecutor(
                max_workers=actual_workers,
                thread_name_prefix="hip3r-adjacent-edge",
            ) as executor:
                return list(executor.map(build_one, indices))
        return [build_one(index) for index in indices]

    @staticmethod
    def _chain_adjacent_alignments(num_chunks: int, adjacent_edges: Sequence[Sim3Edge]) -> list[Sim3]:
        alignments = [Sim3.identity() for _ in range(num_chunks)]
        for edge in adjacent_edges:
            if edge.source == edge.target:
                raise ValueError(f"self edge is not allowed: {edge}")
            if not (0 <= edge.source < num_chunks and 0 <= edge.target < num_chunks):
                raise ValueError(f"edge indices out of range: {edge}")
            alignments[edge.source] = alignments[edge.target].compose(edge.transform)
        return alignments

    @staticmethod
    def _chain_adjacent_alignments_global_scale_gauge_centered(
        num_chunks: int,
        adjacent_edges: Sequence[Sim3Edge],
        *,
        span_cap: float = 50.0,
        defer_translation_residual_to_factorized_frame_chart: bool = False,
    ) -> tuple[list[Sim3], dict[str, Any]]:
        if not np.isfinite(span_cap) or float(span_cap) <= 0.0:
            raise ValueError(
                "chain_global_scale_gauge_centering_v1 span cap must be positive finite"
            )
        if num_chunks < 1:
            return [], {
                "schema": "chain_global_scale_gauge_centering_v2",
                "policy": "minimal_shift_with_component_residual_gates",
                "enabled": True,
                "gate_pass": True,
                "span_cap": float(span_cap),
                "num_chunks": int(num_chunks),
            }
        log_scales = np.zeros(num_chunks, dtype=np.float64)
        rotations = [np.eye(3, dtype=np.float64) for _ in range(num_chunks)]
        translations = [np.zeros(3, dtype=np.float64) for _ in range(num_chunks)]
        assigned = [False] * num_chunks
        assigned[0] = True
        first_floor_crossing_edge: dict[str, Any] | None = None
        sim3_log_floor = float(np.log(SIM3_EPS))

        for edge_index, edge in enumerate(adjacent_edges):
            if edge.source == edge.target:
                raise ValueError(f"self edge is not allowed: {edge}")
            if not (0 <= edge.source < num_chunks and 0 <= edge.target < num_chunks):
                raise ValueError(f"edge indices out of range: {edge}")
            edge_scale = float(edge.transform.scale)
            if not np.isfinite(edge_scale) or edge_scale <= 0.0:
                raise ValueError(
                    "chain_global_scale_gauge_centering_v1 requires positive finite "
                    f"edge scale, got {edge_scale} at edge {edge_index}"
                )
            if not assigned[edge.target]:
                raise ValueError(
                    "chain_global_scale_gauge_centering_v1 requires chain-ordered "
                    f"adjacent edges; target {edge.target} is not assigned"
                )
            target_log = float(log_scales[edge.target])
            target_rotation = rotations[edge.target]
            target_translation = translations[edge.target]
            source_log = target_log + float(np.log(edge_scale))
            source_rotation = target_rotation @ edge.transform.rotation
            source_translation = (
                np.exp(target_log) * (target_rotation @ edge.transform.translation)
                + target_translation
            )
            if not np.isfinite(source_log):
                raise ValueError(
                    "chain_global_scale_gauge_centering_v1 produced non-finite raw log scale"
                )
            if not np.all(np.isfinite(source_rotation)):
                raise ValueError(
                    "chain_global_scale_gauge_centering_v1 produced non-finite raw rotation"
                )
            if not np.all(np.isfinite(source_translation)):
                raise ValueError(
                    "chain_global_scale_gauge_centering_v1 produced non-finite raw translation"
                )
            log_scales[edge.source] = source_log
            rotations[edge.source] = source_rotation
            translations[edge.source] = source_translation
            assigned[edge.source] = True
            if first_floor_crossing_edge is None and source_log < sim3_log_floor:
                first_floor_crossing_edge = {
                    "edge_index": edge_index,
                    "source": int(edge.source),
                    "target": int(edge.target),
                    "raw_log_scale": source_log,
                    "sim3_log_floor": sim3_log_floor,
                }

        if not all(assigned):
            missing = [index for index, item in enumerate(assigned) if not item]
            raise ValueError(
                "chain_global_scale_gauge_centering_v1 requires connected chain; "
                f"missing chunks={missing}"
            )
        raw_translations = np.asarray(translations, dtype=np.float64)
        if not np.all(np.isfinite(raw_translations)):
            raise ValueError(
                "chain_global_scale_gauge_centering_v1 raw translations are non-finite"
            )
        pre_min = float(np.min(log_scales)) if len(log_scales) else 0.0
        pre_max = float(np.max(log_scales)) if len(log_scales) else 0.0
        pre_span = pre_max - pre_min
        if not np.isfinite(pre_span):
            raise ValueError(
                "chain_global_scale_gauge_centering_v1 log-scale span is non-finite"
            )
        if pre_span > float(span_cap):
            raise ValueError(
                "chain_global_scale_gauge_centering_v1 log-scale span cap exceeded: "
                f"span={pre_span} cap={float(span_cap)}"
            )

        margin = 2.0
        feasible_lo = sim3_log_floor + margin
        feasible_hi = -feasible_lo
        feasible_span = feasible_hi - feasible_lo
        if pre_span > feasible_span:
            raise ValueError(
                "chain_global_scale_gauge_centering_v2 feasible log-scale span exceeded: "
                f"span={pre_span} feasible_span={feasible_span}"
            )
        delta_lower = feasible_lo - pre_min
        delta_upper = feasible_hi - pre_max
        if delta_lower <= 0.0 <= delta_upper:
            delta_log = 0.0
        elif abs(delta_lower) <= abs(delta_upper):
            delta_log = float(delta_lower)
        else:
            delta_log = float(delta_upper)

        post_log_scales = log_scales + delta_log
        post_scales = np.exp(post_log_scales)
        post_translations = np.exp(delta_log) * raw_translations
        if not np.all(np.isfinite(post_translations)):
            raise ValueError(
                "chain_global_scale_gauge_centering_v1 post translations are non-finite"
            )
        if not np.all(np.isfinite(post_scales)) or np.any(post_scales < SIM3_EPS):
            raise ValueError(
                "chain_global_scale_gauge_centering_v1 post scale fell below Sim3 floor"
            )

        alignments = [
            Sim3(
                scale=float(scale),
                rotation=rotation,
                translation=translation,
            )
            for scale, rotation, translation in zip(post_scales, rotations, post_translations)
        ]
        translation_scale = 1.0
        scale_tolerance = 1e-9
        rotation_tolerance = 1e-8
        max_scale_error = 0.0
        max_rotation_error = 0.0
        max_translation_error = 0.0
        max_translation_tolerance = 0.0
        max_translation_over_tolerance = 0.0
        max_translation_relative_error = 0.0
        failed_components: list[str] = []
        for edge in adjacent_edges:
            pred = alignments[edge.target].inverse().compose(alignments[edge.source])
            residual = edge.transform.inverse().compose(pred)
            edge_scale = float(edge.transform.scale)
            scale_error = abs(float(np.log(residual.scale)))
            rotation_error = float(
                np.linalg.norm(Rotation.from_matrix(residual.rotation).as_rotvec())
            )
            translation_error = float(np.linalg.norm(residual.translation))
            edge_t_ref = max(
                1.0,
                float(np.linalg.norm(edge.transform.translation))
                / max(abs(edge_scale), SIM3_EPS),
            )
            translation_tolerance = min(
                1e-6 / translation_scale,
                max(1e-8 / translation_scale, 1e-10 * edge_t_ref),
            )
            max_scale_error = max(max_scale_error, scale_error)
            max_rotation_error = max(max_rotation_error, rotation_error)
            max_translation_error = max(max_translation_error, translation_error)
            max_translation_tolerance = max(max_translation_tolerance, translation_tolerance)
            max_translation_over_tolerance = max(
                max_translation_over_tolerance,
                translation_error / max(translation_tolerance, SIM3_EPS),
            )
            max_translation_relative_error = max(
                max_translation_relative_error,
                translation_error / max(edge_t_ref, SIM3_EPS),
            )
            if scale_error > scale_tolerance and "scale" not in failed_components:
                failed_components.append("scale")
            if rotation_error > rotation_tolerance and "rotation" not in failed_components:
                failed_components.append("rotation")
            if (
                translation_error > translation_tolerance
                and "translation" not in failed_components
            ):
                failed_components.append("translation")
        translation_only_deferred = bool(
            defer_translation_residual_to_factorized_frame_chart
            and failed_components == ["translation"]
        )
        if failed_components and not translation_only_deferred:
            raise ValueError(
                "chain_global_scale_gauge_centering_v2 component residual gate failed: "
                f"failed_components={failed_components} "
                f"scale={max_scale_error}/{scale_tolerance} "
                f"rotation={max_rotation_error}/{rotation_tolerance} "
                f"translation={max_translation_error}/{max_translation_tolerance}"
            )

        raw_translation_norms = np.linalg.norm(raw_translations, axis=1)
        post_translation_norms = np.linalg.norm(post_translations, axis=1)
        diagnostics = {
            "schema": "chain_global_scale_gauge_centering_v2",
            "policy": "minimal_shift_with_component_residual_gates",
            "enabled": True,
            "gate_pass": not bool(failed_components),
            "translation_residual_deferred_to_factorized_frame_chart": (
                translation_only_deferred
            ),
            "span_cap": float(span_cap),
            "minimal_shift_margin_log": margin,
            "feasible_log_scale_lo": feasible_lo,
            "feasible_log_scale_hi": feasible_hi,
            "feasible_delta_lower": float(delta_lower),
            "feasible_delta_upper": float(delta_upper),
            "minimal_shift_delta_log": float(delta_log),
            "sim3_scale_floor": float(SIM3_EPS),
            "first_floor_crossing_edge": first_floor_crossing_edge,
            "pre_log_scale_min": pre_min,
            "pre_log_scale_max": pre_max,
            "pre_log_scale_span": pre_span,
            "post_log_scale_min": float(np.min(post_log_scales)) if len(post_log_scales) else 0.0,
            "post_log_scale_max": float(np.max(post_log_scales)) if len(post_log_scales) else 0.0,
            "post_log_scale_span": (
                float(np.max(post_log_scales) - np.min(post_log_scales))
                if len(post_log_scales)
                else 0.0
            ),
            "post_scale_min": float(np.min(post_scales)) if len(post_scales) else 1.0,
            "post_scale_max": float(np.max(post_scales)) if len(post_scales) else 1.0,
            "raw_translation_norm_min": float(np.min(raw_translation_norms)) if len(raw_translation_norms) else 0.0,
            "raw_translation_norm_max": float(np.max(raw_translation_norms)) if len(raw_translation_norms) else 0.0,
            "post_translation_norm_min": float(np.min(post_translation_norms)) if len(post_translation_norms) else 0.0,
            "post_translation_norm_max": float(np.max(post_translation_norms)) if len(post_translation_norms) else 0.0,
            "component_error_max": {
                "scale_log": float(max_scale_error),
                "rotation_rad": float(max_rotation_error),
                "translation": float(max_translation_error),
            },
            "component_relative_error_max": {
                "scale_log_over_tolerance": float(max_scale_error / scale_tolerance),
                "rotation_rad_over_tolerance": float(max_rotation_error / rotation_tolerance),
                "translation_over_tolerance": float(max_translation_over_tolerance),
                "translation_over_edge_reference": float(max_translation_relative_error),
            },
            "component_tolerances": {
                "scale_log": scale_tolerance,
                "rotation_rad": rotation_tolerance,
                "translation_scale": translation_scale,
                "translation_max": float(max_translation_tolerance),
                "translation_rule": (
                    "min(1e-6/translation_scale, "
                    "max(1e-8/translation_scale, 1e-10*edge_t_ref))"
                ),
            },
            "failed_component": (
                None if not failed_components else failed_components[0]
            ),
            "failed_components": failed_components,
        }
        return alignments, diagnostics


def _get_field(chunk: Any, names: Iterable[str]) -> tuple[str, Any]:
    for name in names:
        if isinstance(chunk, dict) and name in chunk:
            return name, chunk[name]
        if hasattr(chunk, name):
            return name, getattr(chunk, name)
    raise AttributeError(f"chunk has none of the expected fields: {', '.join(names)}")


def _get_optional_field(chunk: Any, names: Iterable[str]) -> tuple[str, Any] | tuple[None, None]:
    try:
        return _get_field(chunk, names)
    except AttributeError:
        return None, None


def _get_points(chunk: Any) -> np.ndarray:
    _, points = _get_field(chunk, POINT_FIELDS)
    arr = np.asarray(points)
    if arr.ndim < 2 or arr.shape[-1] != 3:
        raise ValueError(f"points must have shape (..., 3), got {arr.shape}")
    return arr


def _has_finite_validated_fields(chunk: Any, *names: str) -> bool:
    checker = getattr(chunk, "_has_finite_validated_fields", None)
    return bool(checker is not None and checker(*names))


def _get_optional_confidence(chunk: Any, frame_slice: Any | None = None) -> np.ndarray | None:
    _, confidence = _get_optional_field(chunk, CONFIDENCE_FIELDS)
    if confidence is None:
        return None
    conf = np.asarray(confidence)
    if frame_slice is not None and conf.ndim >= 1:
        conf = conf[frame_slice]
    return conf


def _combine_confidence(a_conf: np.ndarray | None, b_conf: np.ndarray | None) -> np.ndarray | None:
    if a_conf is None and b_conf is None:
        return None
    if a_conf is None:
        return np.asarray(b_conf, dtype=np.float64)
    if b_conf is None:
        return np.asarray(a_conf, dtype=np.float64)
    a = np.maximum(np.asarray(a_conf, dtype=np.float64), 0.0)
    b = np.maximum(np.asarray(b_conf, dtype=np.float64), 0.0)
    return np.sqrt(a * b)


def _direction_error(
    joint_points: np.ndarray,
    joint_to_a: Sim3,
    joint_to_b: Sim3,
    a_to_b: Sim3,
) -> float:
    flat = np.asarray(joint_points, dtype=np.float64).reshape(-1, 3)
    if flat.shape[0] > 128:
        pick = np.linspace(0, flat.shape[0] - 1, 128, dtype=np.int64)
        flat = flat[pick]
    composed = a_to_b.apply_points(joint_to_a.apply_points(flat))
    direct = joint_to_b.apply_points(flat)
    return float(np.sqrt(np.mean(np.sum((composed - direct) ** 2, axis=1))))


def _apply_alignment(chunk: Any, alignment: Sim3) -> Any:
    point_name, points = _get_field(chunk, POINT_FIELDS)
    updates: dict[str, np.ndarray] = {point_name: alignment.apply_points(points)}
    c2w_name, c2w = _get_optional_field(chunk, C2W_FIELDS)
    if c2w is not None:
        updates[c2w_name] = alignment.apply_c2w(c2w)
    return _replace_chunk_fields(chunk, updates)


def _replace_chunk_fields(chunk: Any, updates: dict[str, np.ndarray]) -> Any:
    if dataclasses.is_dataclass(chunk):
        return dataclasses.replace(chunk, **updates)
    if isinstance(chunk, dict):
        updated = dict(chunk)
        updated.update(updates)
        return updated
    if hasattr(chunk, "_replace"):
        try:
            return chunk._replace(**updates)
        except TypeError:
            pass
    updated = copy.copy(chunk)
    for name, value in updates.items():
        setattr(updated, name, value)
    return updated


def _call_optimizer(
    optimizer: Callable[..., Sequence[Sim3] | dict[str, Any]],
    *,
    num_chunks: int,
    edges: Sequence[Sim3Edge],
    initial_alignments: Sequence[Sim3],
    use_jac_sparsity: bool = False,
) -> Sequence[Sim3] | dict[str, Any]:
    signature = inspect.signature(optimizer)
    parameters = signature.parameters
    accepts_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    accepts_named_inputs = accepts_keywords or all(
        name in parameters for name in ("num_chunks", "edges", "initial_alignments")
    )
    accepts_sparse = accepts_keywords or "use_jac_sparsity" in parameters
    if use_jac_sparsity and not accepts_sparse:
        raise TypeError(
            "pose graph optimizer does not accept use_jac_sparsity; "
            "refusing to silently fall back to the dense solver"
        )

    kwargs = {
        "num_chunks": num_chunks,
        "edges": edges,
        "initial_alignments": list(initial_alignments),
    }
    if use_jac_sparsity:
        kwargs["use_jac_sparsity"] = True
    if accepts_named_inputs:
        return optimizer(**kwargs)
    return optimizer(num_chunks, edges, list(initial_alignments))


def _coerce_alignments(result: Sequence[Sim3] | dict[str, Any], expected_count: int) -> list[Sim3]:
    alignments = result.get("alignments") if isinstance(result, dict) else result
    alignments_list = list(alignments)
    if len(alignments_list) != expected_count:
        raise ValueError(f"optimizer returned {len(alignments_list)} alignments, expected {expected_count}")
    if not all(isinstance(alignment, Sim3) for alignment in alignments_list):
        raise TypeError("optimizer must return Sim3 alignments")
    return alignments_list


__all__ = [
    "ReconstructionPostProcessor",
    "PostProcessResult",
    "Sim3Edge",
    "estimate_adjacent_edge",
    "estimate_loop_edge",
    "residual_edge_weight",
]
