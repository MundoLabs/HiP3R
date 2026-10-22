"""Self-contained Sim3 pose-graph optimization for the HiP3R post-processing stage."""

from __future__ import annotations

import threading
import time
from typing import Any, Sequence

import numpy as np
from scipy import __version__ as _SCIPY_VERSION
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, lil_matrix
from scipy.spatial.transform import Rotation

from long_reconstruction.postprocess import Sim3Edge
from long_reconstruction.sim3 import Sim3


_MAX_LOG_SCALE_DELTA = float(np.log(100.0))
_TORCH_POSE_GRAPH_RUNTIME: tuple[Any, Any, Any] | None = None
_TORCH_POSE_GRAPH_RUNTIME_LOCK = threading.Lock()


def prepare_torch_pose_graph_analytic_runtime() -> tuple[Any, Any, Any]:
    """Load the local Torch Jacobian runtime once, safe for background prewarm."""

    global _TORCH_POSE_GRAPH_RUNTIME
    if _TORCH_POSE_GRAPH_RUNTIME is not None:
        return _TORCH_POSE_GRAPH_RUNTIME
    with _TORCH_POSE_GRAPH_RUNTIME_LOCK:
        if _TORCH_POSE_GRAPH_RUNTIME is None:
            import torch

            from long_reconstruction.hierarchical_pose_graph import (
                _torch_matrix_to_rotvec,
                _torch_rotvec_to_matrix,
            )

            # The first jacrev/vmap call lazily imports Torch's functional
            # transform stack.  Trigger it on one scalar while artifact
            # validation is running so the real coarse solve does not pay the
            # several-second one-time import cost.
            def _prewarm_function(value: Any) -> Any:
                return value * value

            torch.func.vmap(torch.func.jacrev(_prewarm_function))(
                torch.zeros((1, 1), dtype=torch.float64)
            )

            _TORCH_POSE_GRAPH_RUNTIME = (
                torch,
                _torch_matrix_to_rotvec,
                _torch_rotvec_to_matrix,
            )
    return _TORCH_POSE_GRAPH_RUNTIME


def _sim3_to_parameters(transform: Sim3) -> np.ndarray:
    return np.concatenate(
        [
            Rotation.from_matrix(transform.rotation).as_rotvec(),
            transform.translation,
            np.array([np.log(transform.scale)], dtype=np.float64),
        ]
    )


def _parameters_to_sim3(parameters: np.ndarray) -> Sim3:
    parameters = np.asarray(parameters, dtype=np.float64)
    return Sim3(
        scale=float(np.exp(parameters[6])),
        rotation=Rotation.from_rotvec(parameters[:3]).as_matrix(),
        translation=parameters[3:6],
    )


def _component_weights(values: Sequence[float]) -> np.ndarray:
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape != (3,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("residual_component_weights must be three finite positive values")
    return np.repeat(weights, (3, 3, 1))


def _edge_residual(edge: Sim3Edge, alignments: Sequence[Sim3],
                   residual_component_weights: Sequence[float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    predicted_source_to_target = alignments[edge.target].inverse().compose(alignments[edge.source])
    error = edge.transform.inverse().compose(predicted_source_to_target)
    weight = float(edge.diagnostics.get("weight", 1.0))
    if not np.isfinite(weight) or weight < 0:
        raise ValueError(f"edge weight must be finite and non-negative, got {weight}")
    result = np.sqrt(weight) * np.concatenate(
        [
            Rotation.from_matrix(error.rotation).as_rotvec(),
            error.translation,
            np.array([np.log(error.scale)], dtype=np.float64),
        ]
    )
    if tuple(residual_component_weights) != (1.0, 1.0, 1.0):
        result *= _component_weights(residual_component_weights)
    return result


def _scalar_edge_residuals(edges: Sequence[Sim3Edge], alignments: Sequence[Sim3],
                          residual_component_weights: Sequence[float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    if not edges:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate([_edge_residual(edge, alignments, residual_component_weights) for edge in edges])


class _PoseGraphResidualBatch:
    """Precomputed residual evaluator that preserves scalar Sim3 operation order."""

    def __init__(self, num_chunks: int, edges: Sequence[Sim3Edge], fixed_root: Sim3,
                 residual_component_weights: Sequence[float] = (1.0, 1.0, 1.0)) -> None:
        self.component_weights = _component_weights(residual_component_weights)
        self.num_chunks = int(num_chunks)
        self.root_scale = fixed_root.scale
        self.root_rotation = fixed_root.rotation
        self.root_translation = fixed_root.translation

        self.edge_count = len(edges)
        self.sources = []
        self.targets = []
        self.measured_inv_scales = []
        self.measured_inv_rotations = []
        self.measured_inv_translations = []
        self.sqrt_weights = []
        for edge in edges:
            weight = float(edge.diagnostics.get("weight", 1.0))
            if not np.isfinite(weight) or weight < 0:
                raise ValueError(f"edge weight must be finite and non-negative, got {weight}")

            inv_rotation = edge.transform.rotation.T
            inv_scale = 1.0 / edge.transform.scale
            inv_translation = -(inv_scale * (inv_rotation @ edge.transform.translation))

            self.sources.append(edge.source)
            self.targets.append(edge.target)
            self.measured_inv_scales.append(inv_scale)
            self.measured_inv_rotations.append(inv_rotation)
            self.measured_inv_translations.append(inv_translation)
            self.sqrt_weights.append(np.sqrt(weight))

    def residual(self, vector: np.ndarray) -> np.ndarray:
        if not self.sources:
            return np.zeros(0, dtype=np.float64)

        (
            scales,
            rotations,
            translations,
            inverse_scales,
            inverse_rotations,
            inverse_translations,
        ) = self._unpack_arrays(vector)
        residuals = np.empty(7 * self.edge_count, dtype=np.float64)
        error_rotations = np.empty((self.edge_count, 3, 3), dtype=np.float64)
        error_translations = np.empty((self.edge_count, 3), dtype=np.float64)
        log_error_scales = np.empty(self.edge_count, dtype=np.float64)
        for edge_index in range(self.edge_count):
            source = self.sources[edge_index]
            target = self.targets[edge_index]
            predicted_scale = inverse_scales[target] * scales[source]
            predicted_rotation = inverse_rotations[target] @ rotations[source]
            predicted_translation = (
                inverse_scales[target] * (inverse_rotations[target] @ translations[source])
                + inverse_translations[target]
            )

            error_scale = self.measured_inv_scales[edge_index] * predicted_scale
            error_rotation = self.measured_inv_rotations[edge_index] @ predicted_rotation
            error_translation = (
                self.measured_inv_scales[edge_index]
                * (self.measured_inv_rotations[edge_index] @ predicted_translation)
                + self.measured_inv_translations[edge_index]
            )

            error_rotations[edge_index] = error_rotation
            error_translations[edge_index] = error_translation
            log_error_scales[edge_index] = np.log(error_scale)

        error_rotvecs = Rotation.from_matrix(error_rotations).as_rotvec()
        for edge_index in range(self.edge_count):
            row = 7 * edge_index
            sqrt_weight = self.sqrt_weights[edge_index]
            residuals[row : row + 3] = sqrt_weight * error_rotvecs[edge_index]
            residuals[row + 3 : row + 6] = sqrt_weight * error_translations[edge_index]
            residuals[row + 6] = sqrt_weight * log_error_scales[edge_index]
        if not np.all(self.component_weights == 1.0):
            residuals.reshape(-1, 7)[:] *= self.component_weights
        return residuals

    def _unpack_arrays(
        self, vector: np.ndarray
    ) -> tuple[
        list[float],
        list[np.ndarray],
        list[np.ndarray],
        list[float],
        list[np.ndarray],
        list[np.ndarray],
    ]:
        parameters = np.asarray(vector, dtype=np.float64)
        scales = [self.root_scale]
        rotations = [self.root_rotation]
        translations = [self.root_translation]
        free_count = self.num_chunks - 1
        if free_count:
            free_parameters = parameters.reshape(free_count, 7)
            free_rotations = Rotation.from_rotvec(free_parameters[:, :3]).as_matrix()
        else:
            free_rotations = np.empty((0, 3, 3), dtype=np.float64)
        for index in range(free_count):
            item = parameters[index * 7 : (index + 1) * 7]
            scales.append(float(np.exp(item[6])))
            rotations.append(free_rotations[index])
            translations.append(item[3:6])

        inverse_scales = []
        inverse_rotations = []
        inverse_translations = []
        for index in range(self.num_chunks):
            inverse_rotation = rotations[index].T
            inverse_scale = 1.0 / scales[index]
            inverse_translation = -(inverse_scale * (inverse_rotation @ translations[index]))
            inverse_scales.append(inverse_scale)
            inverse_rotations.append(inverse_rotation)
            inverse_translations.append(inverse_translation)
        return (
            scales,
            rotations,
            translations,
            inverse_scales,
            inverse_rotations,
            inverse_translations,
        )


def _residual_gate_diagnostics(
    exact_residual: np.ndarray,
    scalar_residual: np.ndarray,
) -> dict[str, Any]:
    if exact_residual.shape != scalar_residual.shape:
        raise AssertionError(
            "exact pose graph residual shape mismatch: "
            f"{exact_residual.shape} != {scalar_residual.shape}"
        )
    abs_delta = np.abs(exact_residual - scalar_residual)
    max_abs = float(np.max(abs_delta)) if abs_delta.size else 0.0
    max_rel = (
        float(np.max(abs_delta / np.maximum(np.abs(scalar_residual), np.finfo(np.float64).tiny)))
        if abs_delta.size
        else 0.0
    )
    array_equal = bool(np.array_equal(exact_residual, scalar_residual))
    if not array_equal:
        raise AssertionError(
            "exact pose graph residual failed scalar reference gate: "
            f"max_abs={max_abs}, max_rel={max_rel}"
        )
    return {
        "residual_mode": "exact_precomputed_sim3_loop",
        "residual_exact_implementation": (
            "raw_loop_batched_node_rotvec_node_inverse_precompute_batched_error_rotvec"
        ),
        "residual_rotation_conversion": (
            "batched_node_from_rotvec_as_matrix_and_error_from_matrix_as_rotvec"
        ),
        "residual_node_rotation_conversion": "batched_scipy_rotation_from_rotvec_as_matrix",
        "residual_error_rotation_conversion": "batched_scipy_rotation_from_matrix_as_rotvec",
        "residual_exact_gate": "array_equal",
        "residual_exact_gate_max_abs": max_abs,
        "residual_exact_gate_max_rel": max_rel,
    }


def _jacobian_sparsity(num_chunks: int, edges: Sequence[Sim3Edge]):
    """Return the block sparsity of 7D edge residuals against free Sim3 nodes."""
    matrix = lil_matrix((7 * len(edges), 7 * max(num_chunks - 1, 0)), dtype=np.int8)
    for edge_index, edge in enumerate(edges):
        row = slice(7 * edge_index, 7 * (edge_index + 1))
        for node in (edge.source, edge.target):
            if node == 0:
                continue
            column = slice(7 * (node - 1), 7 * node)
            matrix[row, column] = 1
    return matrix.tocsr()


class _TorchPoseGraphJacobian:
    """Batched local Sim3 Jacobians assembled into a sparse global matrix."""

    def __init__(
        self, num_chunks: int, edges: Sequence[Sim3Edge], fixed_root: Sim3,
        residual_component_weights: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        self.component_weights = _component_weights(residual_component_weights)
        torch, _torch_matrix_to_rotvec, _torch_rotvec_to_matrix = (
            prepare_torch_pose_graph_analytic_runtime()
        )

        self._torch = torch
        self._matrix_to_rotvec = _torch_matrix_to_rotvec
        self._rotvec_to_matrix = _torch_rotvec_to_matrix
        self._num_chunks = int(num_chunks)
        self._shape = (7 * len(edges), 7 * max(num_chunks - 1, 0))
        self._sources = np.asarray([edge.source for edge in edges], dtype=np.int64)
        self._targets = np.asarray([edge.target for edge in edges], dtype=np.int64)
        measured_inv = [edge.transform.inverse() for edge in edges]
        self._measured_inv_rotations = torch.as_tensor(
            np.stack([item.rotation for item in measured_inv]), dtype=torch.float64
        )
        self._measured_inv_translations = torch.as_tensor(
            np.stack([item.translation for item in measured_inv]), dtype=torch.float64
        )
        self._measured_inv_log_scales = torch.as_tensor(
            np.log([item.scale for item in measured_inv]), dtype=torch.float64
        )
        self._sqrt_weights = torch.as_tensor(
            np.sqrt([float(edge.diagnostics.get("weight", 1.0)) for edge in edges]),
            dtype=torch.float64,
        )
        self._root_parameters = torch.as_tensor(
            _sim3_to_parameters(fixed_root), dtype=torch.float64
        )

        edge_indices: list[int] = []
        row_indices: list[int] = []
        column_indices: list[int] = []
        local_rows: list[int] = []
        local_columns: list[int] = []
        for edge_index, edge in enumerate(edges):
            for node, offset in ((edge.source, 0), (edge.target, 7)):
                if node == 0:
                    continue
                for local_row in range(7):
                    for node_column in range(7):
                        edge_indices.append(edge_index)
                        row_indices.append(7 * edge_index + local_row)
                        column_indices.append(7 * (node - 1) + node_column)
                        local_rows.append(local_row)
                        local_columns.append(offset + node_column)
        self._edge_indices = np.asarray(edge_indices, dtype=np.int64)
        self._row_indices = np.asarray(row_indices, dtype=np.int64)
        self._column_indices = np.asarray(column_indices, dtype=np.int64)
        self._local_rows = np.asarray(local_rows, dtype=np.int64)
        self._local_columns = np.asarray(local_columns, dtype=np.int64)
        self.calls = 0
        self.wall_seconds = 0.0

        def edge_residual(
            parameters: Any,
            measured_rotation: Any,
            measured_translation: Any,
            measured_log_scale: Any,
            sqrt_weight: Any,
        ) -> Any:
            source = parameters[:7]
            target = parameters[7:]
            source_rotation = self._rotvec_to_matrix(source[:3])
            target_rotation = self._rotvec_to_matrix(target[:3])
            target_rotation_inverse = target_rotation.transpose(-1, -2)
            predicted_rotation = target_rotation_inverse @ source_rotation
            predicted_translation = torch.exp(-target[6]) * (
                target_rotation_inverse @ (source[3:6] - target[3:6])
            )
            error_rotation = measured_rotation @ predicted_rotation
            error_translation = (
                torch.exp(measured_log_scale)
                * (measured_rotation @ predicted_translation)
                + measured_translation
            )
            return sqrt_weight * torch.cat(
                (
                    self._matrix_to_rotvec(error_rotation),
                    error_translation,
                    (measured_log_scale + source[6] - target[6]).reshape(1),
                )
            )

        self._batched_jacobian = torch.func.vmap(
            torch.func.jacrev(edge_residual, argnums=0)
        )

    def __call__(self, vector: np.ndarray):
        started = time.perf_counter()
        torch = self._torch
        free = torch.as_tensor(
            np.asarray(vector, dtype=np.float64).reshape(self._num_chunks - 1, 7)
        )
        nodes = torch.cat((self._root_parameters.reshape(1, 7), free), dim=0)
        parameters = torch.cat(
            (nodes[self._sources], nodes[self._targets]), dim=1
        )
        local = self._batched_jacobian(
            parameters,
            self._measured_inv_rotations,
            self._measured_inv_translations,
            self._measured_inv_log_scales,
            self._sqrt_weights,
        ).detach().numpy()
        data = local[
            self._edge_indices, self._local_rows, self._local_columns
        ]
        result = coo_matrix(
            (data, (self._row_indices, self._column_indices)), shape=self._shape
        ).tocsr()
        if not np.all(self.component_weights == 1.0):
            result = result.multiply(np.tile(self.component_weights, self._shape[0] // 7)[:, None]).tocsr()
        self.calls += 1
        self.wall_seconds += time.perf_counter() - started
        return result


def optimize_sim3_pose_graph(
    num_chunks: int,
    edges: Sequence[Sim3Edge],
    initial_alignments: Sequence[Sim3],
    *,
    max_nfev: int = 100,
    robust_loss: str = "huber",
    f_scale: float = 1.0,
    use_jac_sparsity: bool = False,
    analytic_jacobian: bool = False,
    x_scale: str = "unit",
    lsmr_tolerance: float | None = None,
    residual_component_weights: Sequence[float] = (1.0, 1.0, 1.0),
    record_residual_components: bool = False,
) -> dict[str, Any]:
    """Optimize chunk-to-global Sim3 poses while fixing chunk zero as the gauge."""
    _component_weights(residual_component_weights)
    if num_chunks <= 0:
        return {"alignments": [], "diagnostics": {"status": "empty"}}
    if len(initial_alignments) != num_chunks:
        raise ValueError(f"expected {num_chunks} initial alignments, got {len(initial_alignments)}")
    if x_scale not in {"unit", "jac"}:
        raise ValueError("x_scale must be 'unit' or 'jac'")
    if lsmr_tolerance is not None and lsmr_tolerance <= 0.0:
        raise ValueError("lsmr_tolerance must be positive")
    for edge in edges:
        if edge.source == edge.target:
            raise ValueError("self edges are not allowed")
        if not (0 <= edge.source < num_chunks and 0 <= edge.target < num_chunks):
            raise ValueError(f"edge index out of range: {edge.source}->{edge.target}")

    if num_chunks == 1 or not edges:
        return {
            "alignments": list(initial_alignments),
            "diagnostics": {
                "status": "no_edges",
                "optimizer_used": False,
                "initial_cost": 0.0,
                "final_cost": 0.0,
            },
        }

    fixed_root = initial_alignments[0]
    initial_vector = np.concatenate([_sim3_to_parameters(item) for item in initial_alignments[1:]])
    residual_batch = _PoseGraphResidualBatch(num_chunks, edges, fixed_root, residual_component_weights)

    def unpack(vector: np.ndarray) -> list[Sim3]:
        alignments = [fixed_root]
        for index in range(num_chunks - 1):
            alignments.append(_parameters_to_sim3(vector[index * 7 : (index + 1) * 7]))
        return alignments

    def residual(vector: np.ndarray) -> np.ndarray:
        nonlocal residual_calls_solver
        nonlocal residual_calls_total
        nonlocal residual_wall_seconds_solver
        nonlocal residual_wall_seconds_total

        with residual_profile_lock:
            solver_call = residual_profile_solver_active
        started = time.perf_counter()
        try:
            if not edges:
                return np.zeros(0, dtype=np.float64)
            return residual_batch.residual(vector)
        finally:
            elapsed = time.perf_counter() - started
            with residual_profile_lock:
                residual_calls_total += 1
                residual_wall_seconds_total += elapsed
                if solver_call:
                    residual_calls_solver += 1
                    residual_wall_seconds_solver += elapsed

    residual_profile_lock = threading.Lock()
    residual_calls_total = 0
    residual_calls_solver = 0
    residual_wall_seconds_total = 0.0
    residual_wall_seconds_solver = 0.0
    residual_profile_solver_active = False
    initial_residual = residual(initial_vector)
    residual_gate = _residual_gate_diagnostics(
        initial_residual,
        _scalar_edge_residuals(edges, unpack(initial_vector), residual_component_weights),
    )
    lower = np.full_like(initial_vector, -np.inf)
    upper = np.full_like(initial_vector, np.inf)
    scale_indices = np.arange(6, initial_vector.size, 7)
    lower[scale_indices] = initial_vector[scale_indices] - _MAX_LOG_SCALE_DELTA
    upper[scale_indices] = initial_vector[scale_indices] + _MAX_LOG_SCALE_DELTA
    jac_sparsity = _jacobian_sparsity(num_chunks, edges) if use_jac_sparsity else None
    analytic_jacobian_callable = (
        _TorchPoseGraphJacobian(num_chunks, edges, fixed_root, residual_component_weights)
        if analytic_jacobian and use_jac_sparsity
        else None
    )
    with residual_profile_lock:
        residual_profile_solver_active = True
    solver_wall_start = time.perf_counter()
    try:
        solver_options: dict[str, Any] = {}
        if x_scale == "jac":
            solver_options["x_scale"] = "jac"
        if lsmr_tolerance is not None:
            solver_options["tr_solver"] = "lsmr"
            solver_options["tr_options"] = {
                "atol": float(lsmr_tolerance),
                "btol": float(lsmr_tolerance),
            }
        result = least_squares(
            residual,
            initial_vector,
            bounds=(lower, upper),
            loss=robust_loss,
            f_scale=float(f_scale),
            max_nfev=int(max_nfev),
            jac=(analytic_jacobian_callable if analytic_jacobian_callable is not None else "2-point"),
            jac_sparsity=(None if analytic_jacobian_callable is not None else jac_sparsity),
            **solver_options,
        )
    finally:
        solver_wall_seconds = time.perf_counter() - solver_wall_start
        with residual_profile_lock:
            residual_profile_solver_active = False
    final_residual = residual(result.x)
    active_mask = np.asarray(result.active_mask, dtype=np.int8)
    parameter_components = (
        "rotation_x",
        "rotation_y",
        "rotation_z",
        "translation_x",
        "translation_y",
        "translation_z",
        "log_scale",
    )
    active_bounds = [
        {
            "chunk_index": int(parameter_index // 7 + 1),
            "component": parameter_components[parameter_index % 7],
            "side": "lower" if active_mask[parameter_index] < 0 else "upper",
            "initial_value": float(initial_vector[parameter_index]),
            "final_value": float(result.x[parameter_index]),
            "bound_value": float(
                lower[parameter_index]
                if active_mask[parameter_index] < 0
                else upper[parameter_index]
            ),
        }
        for parameter_index in np.flatnonzero(active_mask)
    ]
    result_njev = getattr(result, "njev", None)
    num_jacobian_evaluations = None if result_njev is None else int(result_njev)
    with residual_profile_lock:
        residual_profile = {
            "residual_calls_total": int(residual_calls_total),
            "residual_calls_solver": int(residual_calls_solver),
            "residual_wall_seconds_total": float(residual_wall_seconds_total),
            "residual_wall_seconds_solver": float(residual_wall_seconds_solver),
            "solver_wall_seconds": float(solver_wall_seconds),
        }
    component_diagnostics = {}
    if record_residual_components:
        from long_reconstruction.hierarchical_pose_graph import _residual_component_diagnostics
        component_diagnostics = {"residual_components": {
            "scope": "actual_solver_rows_after_edge_and_component_weights",
            "initial": _residual_component_diagnostics(
                initial_residual, len(edges), 0, 0, loss=robust_loss, f_scale=f_scale),
            "final": _residual_component_diagnostics(
                final_residual, len(edges), 0, 0, loss=robust_loss, f_scale=f_scale),
        }}
    return {
        "alignments": unpack(result.x),
        "diagnostics": {
            **component_diagnostics,
            "residual_component_weights": list(map(float, residual_component_weights)),
            "status": "ok" if result.success else "not_converged",
            "optimizer_used": True,
            "success": bool(result.success),
            "termination_status": int(result.status),
            "message": str(result.message),
            "num_edges": len(edges),
            "num_function_evaluations": int(result.nfev),
            "num_jacobian_evaluations": num_jacobian_evaluations,
            "num_variables": int(initial_vector.size),
            "num_residuals": int(initial_residual.size),
            "scipy_version": str(_SCIPY_VERSION),
            "method": "trf",
            "x_scale": x_scale,
            "tr_solver": "lsmr" if lsmr_tolerance is not None else "auto",
            "lsmr_tolerance": lsmr_tolerance,
            **residual_profile,
            **residual_gate,
            "jacobian_mode": (
                "sparse_torch_block_analytic"
                if analytic_jacobian_callable is not None
                else "sparse_finite_difference"
                if use_jac_sparsity
                else "dense_finite_difference"
            ),
            "analytic_jacobian_calls": (
                analytic_jacobian_callable.calls
                if analytic_jacobian_callable is not None
                else 0
            ),
            "analytic_jacobian_wall_seconds": (
                analytic_jacobian_callable.wall_seconds
                if analytic_jacobian_callable is not None
                else 0.0
            ),
            "jacobian_nnz": int(jac_sparsity.nnz) if jac_sparsity is not None else None,
            "jacobian_density": (
                float(jac_sparsity.nnz / np.prod(jac_sparsity.shape))
                if jac_sparsity is not None and np.prod(jac_sparsity.shape)
                else None
            ),
            "max_log_scale_delta": _MAX_LOG_SCALE_DELTA,
            "initial_cost": float(np.mean(initial_residual**2)) if initial_residual.size else 0.0,
            "final_cost": float(np.mean(final_residual**2)) if final_residual.size else 0.0,
            "scipy_cost": float(result.cost),
            "optimality": float(result.optimality),
            "gradient_inf_norm": float(
                np.linalg.norm(np.asarray(result.grad, dtype=np.float64), ord=np.inf)
            ),
            "solution_step_norm": float(np.linalg.norm(result.x - initial_vector)),
            "active_bound_count": len(active_bounds),
            "active_bounds": active_bounds,
            "solution_finite": bool(np.all(np.isfinite(result.x))),
            "residual_finite": bool(np.all(np.isfinite(final_residual))),
            "cost_finite": bool(
                np.isfinite(result.cost)
                and np.all(np.isfinite(final_residual))
            ),
        },
    }


__all__ = ["optimize_sim3_pose_graph"]
