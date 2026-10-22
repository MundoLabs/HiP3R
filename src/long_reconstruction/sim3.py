from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


_EPS = 1e-12


def _eye3() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def _zeros3() -> np.ndarray:
    return np.zeros(3, dtype=np.float64)


@dataclass(frozen=True)
class Sim3:
    """Similarity transform with source-to-target semantics.

    A transform maps points as ``target = scale * rotation @ source + translation``.
    ``a_to_c = b_to_c.compose(a_to_b)`` applies ``a_to_b`` first, then ``b_to_c``.
    """

    scale: float = 1.0
    rotation: np.ndarray = field(default_factory=_eye3)
    translation: np.ndarray = field(default_factory=_zeros3)

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
        if translation.shape != (3,):
            raise ValueError(f"translation must have shape (3,), got {translation.shape}")
        scale = float(self.scale)
        if not np.isfinite(scale) or abs(scale) < _EPS:
            raise ValueError(f"scale must be finite and non-zero, got {scale}")
        if not np.all(np.isfinite(rotation)):
            raise ValueError("rotation contains non-finite values")
        if not np.all(np.isfinite(translation)):
            raise ValueError("translation contains non-finite values")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @classmethod
    def identity(cls) -> "Sim3":
        return cls()

    def compose(self, other: "Sim3") -> "Sim3":
        """Return ``self(other(x))``."""

        return Sim3(
            scale=self.scale * other.scale,
            rotation=self.rotation @ other.rotation,
            translation=self.scale * (self.rotation @ other.translation) + self.translation,
        )

    def inverse(self) -> "Sim3":
        inv_rotation = self.rotation.T
        inv_scale = 1.0 / self.scale
        return Sim3(
            scale=inv_scale,
            rotation=inv_rotation,
            translation=-(inv_scale * (inv_rotation @ self.translation)),
        )

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        points_arr = np.asarray(points, dtype=np.float64)
        if points_arr.shape[-1] != 3:
            raise ValueError(f"points must end with dimension 3, got {points_arr.shape}")
        return self.scale * np.einsum("ij,...j->...i", self.rotation, points_arr) + self.translation

    def apply_c2w(self, c2w: np.ndarray) -> np.ndarray:
        """Apply this world-space Sim3 to camera-to-world pose matrices.

        The camera center is transformed by the full Sim3. The orientation is rotated
        by ``rotation`` and remains a rigid 3x3 camera basis.
        """

        pose = np.asarray(c2w, dtype=np.float64)
        if pose.shape[-2:] not in ((3, 4), (4, 4)):
            raise ValueError(f"c2w must end with shape (3, 4) or (4, 4), got {pose.shape}")
        aligned = pose.copy()
        aligned[..., :3, :3] = np.einsum("ij,...jk->...ik", self.rotation, pose[..., :3, :3])
        aligned[..., :3, 3] = self.apply_points(pose[..., :3, 3])
        return aligned

    def as_matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix


def estimate_sim3(
    source: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray | None = None,
    *,
    use_scale: bool = True,
    confidence_threshold: float | None = 0.0,
    max_points: int | None = None,
    max_iterations: int = 10,
    huber_delta: float | None = None,
    convergence_tol: float = 1e-6,
    min_points: int = 3,
) -> tuple[Sim3, dict[str, Any]]:
    """Estimate a confidence-weighted robust Sim3 from source to target.

    The estimator first filters non-finite correspondences and low-confidence points,
    optionally takes a deterministic evenly spaced subset, then runs weighted Umeyama
    inside a Huber IRLS loop.
    """

    return _estimate_sim3_impl(
        source,
        target,
        confidence,
        use_scale=use_scale,
        confidence_threshold=confidence_threshold,
        max_points=max_points,
        max_iterations=max_iterations,
        huber_delta=huber_delta,
        convergence_tol=convergence_tol,
        min_points=min_points,
        finite_validated_points=False,
    )


def _estimate_sim3_finite_validated(
    source: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray | None = None,
    *,
    use_scale: bool = True,
    confidence_threshold: float | None = 0.0,
    max_points: int | None = None,
    max_iterations: int = 10,
    huber_delta: float | None = None,
    convergence_tol: float = 1e-6,
    min_points: int = 3,
) -> tuple[Sim3, dict[str, Any]]:
    return _estimate_sim3_impl(
        source,
        target,
        confidence,
        use_scale=use_scale,
        confidence_threshold=confidence_threshold,
        max_points=max_points,
        max_iterations=max_iterations,
        huber_delta=huber_delta,
        convergence_tol=convergence_tol,
        min_points=min_points,
        finite_validated_points=True,
    )


def _estimate_sim3_finite_validated_paired_confidence(
    source: np.ndarray,
    target: np.ndarray,
    confidence_a: np.ndarray,
    confidence_b: np.ndarray,
    *,
    use_scale: bool = True,
    confidence_threshold: float | None = 0.0,
    max_points: int | None = None,
    max_iterations: int = 10,
    huber_delta: float | None = None,
    convergence_tol: float = 1e-6,
    min_points: int = 3,
) -> tuple[Sim3, dict[str, Any]]:
    source_flat = _flatten_points(source, "source")
    target_flat = _flatten_points(target, "target")
    if source_flat.shape != target_flat.shape:
        raise ValueError(
            f"source and target shapes must match, got {source_flat.shape} and "
            f"{target_flat.shape}"
        )
    points_shape = np.asarray(source).shape
    confidence_a_flat = _flatten_confidence_native(
        confidence_a, source_flat.shape[0], points_shape
    )
    confidence_b_flat = _flatten_confidence_native(
        confidence_b, source_flat.shape[0], points_shape
    )

    fallback_reason: str | None = None
    max_points_int: int | None = None
    if confidence_threshold != 0.0:
        fallback_reason = "confidence_threshold_not_zero"
    elif max_points is None:
        fallback_reason = "max_points_missing"
    else:
        max_points_int = int(max_points)
        if max_points_int <= 0:
            fallback_reason = "max_points_not_positive"
        elif source_flat.shape[0] <= max_points_int:
            fallback_reason = "sampling_not_required"
        elif not bool(np.isfinite(confidence_a_flat).all()) or not bool(
            np.isfinite(confidence_b_flat).all()
        ):
            fallback_reason = "confidence_nonfinite"
        elif float(confidence_a_flat.min()) <= 0.0 or float(
            confidence_b_flat.min()
        ) <= 0.0:
            fallback_reason = "confidence_nonpositive"

    if fallback_reason is not None:
        confidence_a_float64 = confidence_a_flat.astype(np.float64, copy=False)
        confidence_b_float64 = confidence_b_flat.astype(np.float64, copy=False)
        combined = np.sqrt(
            np.maximum(confidence_a_float64, 0.0)
            * np.maximum(confidence_b_float64, 0.0)
        )
        transform, diagnostics = _estimate_sim3_finite_validated(
            source,
            target,
            combined,
            use_scale=use_scale,
            confidence_threshold=confidence_threshold,
            max_points=max_points,
            max_iterations=max_iterations,
            huber_delta=huber_delta,
            convergence_tol=convergence_tol,
            min_points=min_points,
        )
        diagnostics["paired_confidence_fallback_reason"] = fallback_reason
        return transform, diagnostics

    assert max_points_int is not None
    num_input = int(source_flat.shape[0])
    indices = np.linspace(0, num_input - 1, max_points_int, dtype=np.int64)
    sampled_confidence_a = confidence_a_flat[indices].astype(np.float64, copy=False)
    sampled_confidence_b = confidence_b_flat[indices].astype(np.float64, copy=False)
    sampled_confidence = np.sqrt(
        np.maximum(sampled_confidence_a, 0.0)
        * np.maximum(sampled_confidence_b, 0.0)
    )
    transform, diagnostics = _estimate_sim3_impl(
        source_flat[indices],
        target_flat[indices],
        sampled_confidence,
        use_scale=use_scale,
        confidence_threshold=0.0,
        max_points=None,
        max_iterations=max_iterations,
        huber_delta=huber_delta,
        convergence_tol=convergence_tol,
        min_points=min_points,
        finite_validated_points=False,
    )
    diagnostics.update(
        {
            "num_input": num_input,
            "num_valid": num_input,
            "num_used": max_points_int,
            "confidence_threshold": 0.0,
            "max_points": max_points_int,
            "correspondence_filter_mode": (
                "finite_validated_paired_confidence_linear_sample"
            ),
            "correspondence_filter_fallback_reason": None,
            "paired_confidence_fallback_reason": None,
        }
    )
    return transform, diagnostics


def _estimate_sim3_impl(
    source: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray | None,
    *,
    use_scale: bool,
    confidence_threshold: float | None,
    max_points: int | None,
    max_iterations: int,
    huber_delta: float | None,
    convergence_tol: float,
    min_points: int,
    finite_validated_points: bool,
) -> tuple[Sim3, dict[str, Any]]:
    source_flat = _flatten_points(source, "source")
    target_flat = _flatten_points(target, "target")
    if source_flat.shape != target_flat.shape:
        raise ValueError(f"source and target shapes must match, got {source_flat.shape} and {target_flat.shape}")

    conf_flat = _flatten_confidence(confidence, source_flat.shape[0], np.asarray(source).shape)
    if finite_validated_points:
        src, tgt, conf, filter_diag = _filter_correspondences_finite_validated_sample(
            source_flat,
            target_flat,
            conf_flat,
            confidence_threshold=confidence_threshold,
            max_points=max_points,
        )
        if src is None:
            src, tgt, conf, filter_diag = _filter_correspondences(
                source_flat,
                target_flat,
                conf_flat,
                confidence_threshold=confidence_threshold,
                max_points=max_points,
                correspondence_filter_mode="full_filter_fallback",
                fallback_reason=filter_diag["correspondence_filter_fallback_reason"],
            )
    else:
        src, tgt, conf, filter_diag = _filter_correspondences(
            source_flat,
            target_flat,
            conf_flat,
            confidence_threshold=confidence_threshold,
            max_points=max_points,
            correspondence_filter_mode="full_filter",
            fallback_reason=None,
        )
    diagnostics: dict[str, Any] = {
        **filter_diag,
        "use_scale": bool(use_scale),
        "max_iterations": int(max_iterations),
        "huber_delta": None if huber_delta is None else float(huber_delta),
        "converged": False,
        "iterations": 0,
    }

    if src.shape[0] < min_points:
        diagnostics.update({"status": "insufficient_correspondences"})
        return Sim3.identity(), diagnostics

    weights = np.maximum(conf.astype(np.float64), 0.0)
    if float(weights.sum()) <= _EPS:
        diagnostics.update({"status": "zero_confidence"})
        return Sim3.identity(), diagnostics

    sim3 = Sim3.identity()
    prev_norm_weights: np.ndarray | None = None
    status = "ok"
    for iteration in range(max(1, int(max_iterations))):
        sim3, status = _weighted_umeyama(src, tgt, weights, use_scale=use_scale)
        if status != "ok":
            diagnostics.update({"status": status, "iterations": iteration + 1})
            return Sim3.identity(), diagnostics

        residuals = _residuals(sim3, src, tgt)
        delta = _resolve_huber_delta(residuals, weights, huber_delta)
        robust_weights = np.ones_like(residuals)
        large = residuals > delta
        robust_weights[large] = delta / np.maximum(residuals[large], _EPS)
        new_weights = np.maximum(conf, 0.0) * robust_weights
        if float(new_weights.sum()) <= _EPS:
            diagnostics.update({"status": "zero_irls_weight", "iterations": iteration + 1})
            return Sim3.identity(), diagnostics

        norm_weights = new_weights / new_weights.sum()
        if prev_norm_weights is not None:
            delta_weights = float(np.abs(norm_weights - prev_norm_weights).sum())
            if delta_weights < convergence_tol:
                diagnostics["converged"] = True
                weights = new_weights
                diagnostics["iterations"] = iteration + 1
                break
        prev_norm_weights = norm_weights
        weights = new_weights
        diagnostics["iterations"] = iteration + 1

    sim3, status = _weighted_umeyama(src, tgt, weights, use_scale=use_scale)
    if status != "ok":
        diagnostics.update({"status": status})
        return Sim3.identity(), diagnostics

    final_residuals = _residuals(sim3, src, tgt)
    diagnostics.update(_residual_diagnostics(final_residuals, weights))
    diagnostics.update(
        {
            "status": "ok",
            "scale": float(sim3.scale),
            "rotation_det": float(np.linalg.det(sim3.rotation)),
            "translation_norm": float(np.linalg.norm(sim3.translation)),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
        }
    )
    return sim3, diagnostics


def _filter_correspondences_finite_validated_sample(
    source: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    *,
    confidence_threshold: float | None,
    max_points: int | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    num_input = int(source.shape[0])
    fallback_reason: str | None = None
    max_points_int: int | None = None
    if confidence_threshold != 0.0:
        fallback_reason = "confidence_threshold_not_zero"
    elif max_points is None:
        fallback_reason = "max_points_missing"
    else:
        max_points_int = int(max_points)
        if max_points_int <= 0:
            fallback_reason = "max_points_not_positive"
        elif num_input <= max_points_int:
            fallback_reason = "sampling_not_required"
        elif not bool(np.isfinite(confidence).all()):
            fallback_reason = "confidence_nonfinite"
        elif float(confidence.min()) <= 0.0:
            fallback_reason = "confidence_nonpositive"

    if fallback_reason is not None:
        return (
            None,
            None,
            None,
            {
                "num_input": num_input,
                "num_valid": None,
                "num_used": None,
                "confidence_threshold": (
                    None if confidence_threshold is None else float(confidence_threshold)
                ),
                "max_points": None if max_points is None else int(max_points),
                "correspondence_filter_mode": "full_filter_fallback",
                "correspondence_filter_fallback_reason": fallback_reason,
            },
        )

    assert max_points_int is not None
    indices = np.linspace(0, num_input - 1, max_points_int, dtype=np.int64)
    diagnostics = {
        "num_input": num_input,
        "num_valid": num_input,
        "num_used": max_points_int,
        "confidence_threshold": 0.0,
        "max_points": max_points_int,
        "correspondence_filter_mode": "finite_validated_linear_sample",
        "correspondence_filter_fallback_reason": None,
    }
    return (
        source[indices].astype(np.float64, copy=False),
        target[indices].astype(np.float64, copy=False),
        confidence[indices].astype(np.float64, copy=False),
        diagnostics,
    )


def _flatten_points(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points)
    if arr.ndim < 2 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3), got {arr.shape}")
    return arr.reshape(-1, 3)


def _flatten_confidence(
    confidence: np.ndarray | None,
    num_points: int,
    points_shape: tuple[int, ...],
) -> np.ndarray:
    if confidence is None:
        return np.ones(num_points, dtype=np.float64)
    conf = np.asarray(confidence, dtype=np.float64)
    if conf.shape == points_shape[:-1]:
        return conf.reshape(-1)
    if conf.shape == points_shape[:-1] + (1,):
        return conf.reshape(-1)
    if conf.shape == points_shape:
        return np.nanmean(conf, axis=-1).reshape(-1)
    if conf.size == num_points:
        return conf.reshape(-1)
    raise ValueError(f"confidence shape {conf.shape} is not compatible with points shape {points_shape}")


def _flatten_confidence_native(
    confidence: np.ndarray,
    num_points: int,
    points_shape: tuple[int, ...],
) -> np.ndarray:
    """Flatten scalar confidence without eagerly widening the full array."""
    conf = np.asarray(confidence)
    if conf.shape == points_shape[:-1] or conf.shape == points_shape[:-1] + (1,):
        return conf.reshape(-1)
    if conf.shape == points_shape:
        # Preserve the generic helper's float64 nanmean semantics for vector confidence.
        return np.nanmean(conf.astype(np.float64, copy=False), axis=-1).reshape(-1)
    if conf.size == num_points:
        return conf.reshape(-1)
    raise ValueError(f"confidence shape {conf.shape} is not compatible with points shape {points_shape}")


def _filter_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    *,
    confidence_threshold: float | None,
    max_points: int | None,
    correspondence_filter_mode: str = "full_filter",
    fallback_reason: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1) & np.isfinite(confidence)
    if confidence_threshold is not None:
        valid &= confidence >= float(confidence_threshold)
    valid &= confidence > 0.0
    indices = np.flatnonzero(valid)
    num_after_threshold = int(indices.size)

    if max_points is not None:
        max_points_int = int(max_points)
        if max_points_int <= 0:
            raise ValueError(f"max_points must be positive, got {max_points}")
        if indices.size > max_points_int:
            pick = np.linspace(0, indices.size - 1, max_points_int, dtype=np.int64)
            indices = indices[pick]

    diagnostics = {
        "num_input": int(source.shape[0]),
        "num_valid": num_after_threshold,
        "num_used": int(indices.size),
        "confidence_threshold": None if confidence_threshold is None else float(confidence_threshold),
        "max_points": None if max_points is None else int(max_points),
        "correspondence_filter_mode": correspondence_filter_mode,
        "correspondence_filter_fallback_reason": fallback_reason,
    }
    return (
        source[indices].astype(np.float64, copy=False),
        target[indices].astype(np.float64, copy=False),
        confidence[indices].astype(np.float64, copy=False),
        diagnostics,
    )


def _weighted_umeyama(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    use_scale: bool,
) -> tuple[Sim3, str]:
    weight_sum = float(weights.sum())
    if source.shape[0] < 3:
        return Sim3.identity(), "insufficient_correspondences"
    if weight_sum <= _EPS:
        return Sim3.identity(), "zero_weight"

    weights_norm = weights / weight_sum
    source_mean = np.sum(source * weights_norm[:, None], axis=0)
    target_mean = np.sum(target * weights_norm[:, None], axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_var = float(np.sum(weights_norm * np.sum(source_centered * source_centered, axis=1)))
    if source_var <= _EPS:
        return Sim3.identity(), "degenerate_source"

    covariance = (target_centered * weights_norm[:, None]).T @ source_centered
    try:
        u, singular_values, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return Sim3.identity(), "svd_failed"

    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        sign[-1] = -1.0
    rotation = u @ np.diag(sign) @ vt
    scale = 1.0
    if use_scale:
        scale = float(np.sum(singular_values * sign) / source_var)
        if not np.isfinite(scale) or abs(scale) <= _EPS:
            return Sim3.identity(), "degenerate_scale"
    translation = target_mean - scale * (rotation @ source_mean)
    return Sim3(scale=scale, rotation=rotation, translation=translation), "ok"


def _residuals(sim3: Sim3, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(sim3.apply_points(source) - target, axis=1)


def _resolve_huber_delta(residuals: np.ndarray, weights: np.ndarray, huber_delta: float | None) -> float:
    if huber_delta is not None:
        return max(float(huber_delta), _EPS)
    median = _weighted_quantile(residuals, weights, 0.5)
    mad = _weighted_quantile(np.abs(residuals - median), weights, 0.5)
    sigma = 1.4826 * mad
    return max(float(median + 1.345 * sigma), _EPS)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values_sorted = values[order]
    weights_sorted = weights[order]
    cumulative = np.cumsum(weights_sorted)
    if cumulative[-1] <= _EPS:
        return float(np.quantile(values, quantile))
    cutoff = quantile * cumulative[-1]
    return float(values_sorted[np.searchsorted(cumulative, cutoff, side="left")])


def _residual_diagnostics(residuals: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    weights_norm = weights / max(float(weights.sum()), _EPS)
    rmse = np.sqrt(np.sum(weights_norm * residuals * residuals))
    median = _weighted_quantile(residuals, weights, 0.5)
    p90 = _weighted_quantile(residuals, weights, 0.9)
    cutoff = max(median + 2.0 * (p90 - median), _EPS)
    return {
        "residual_mean": float(np.sum(weights_norm * residuals)),
        "residual_median": float(median),
        "residual_rmse": float(rmse),
        "residual_p90": float(p90),
        "inlier_ratio": float(np.mean(residuals <= cutoff)),
    }


__all__ = ["Sim3", "estimate_sim3"]
