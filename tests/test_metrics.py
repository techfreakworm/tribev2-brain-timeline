"""Tests for ROI reduction -- pure numpy, synthetic atlas, no model."""

from __future__ import annotations

import numpy as np
import pytest

from tribescore.metrics import (
    DEFAULT_METRICS,
    Atlas,
    MetricSpec,
    reduce_to_metrics,
    roi_means,
)


def _synthetic_atlas(n_vertices: int = 12) -> Atlas:
    """Round-robin assign vertices to the ROI names used by DEFAULT_METRICS."""
    rois = sorted(
        {roi for spec in DEFAULT_METRICS for roi in spec.roi_weights}
    )
    labels = np.array([rois[i % len(rois)] for i in range(n_vertices)])
    return Atlas(labels=labels)


def test_roi_means_shape_and_values():
    atlas = Atlas(labels=np.array(["a", "a", "b", "b"]))
    timeline = np.array(
        [
            [1.0, 3.0, 10.0, 20.0],  # t=0: a->2, b->15
            [2.0, 4.0, 30.0, 40.0],  # t=1: a->3, b->35
        ]
    )
    means = roi_means(timeline, atlas)
    assert set(means) == {"a", "b"}
    assert means["a"].shape == (2,)
    assert np.allclose(means["a"], [2.0, 3.0])
    assert np.allclose(means["b"], [15.0, 35.0])


def test_roi_means_rejects_length_mismatch():
    atlas = Atlas(labels=np.array(["a", "b", "c"]))
    with pytest.raises(ValueError):
        roi_means(np.zeros((5, 4)), atlas)  # 4 vertices != 3 labels


def test_reduce_to_metrics_returns_all_metrics_with_right_shape():
    n_t, n_vertices = 25, 12
    rng = np.random.default_rng(0)
    timeline = rng.standard_normal((n_t, n_vertices))
    atlas = _synthetic_atlas(n_vertices)

    curves = reduce_to_metrics(timeline, atlas, rescale_0_1=True)

    assert set(curves) == {m.name for m in DEFAULT_METRICS}
    for name, curve in curves.items():
        assert curve.shape == (n_t,), name
        assert np.all(np.isfinite(curve)), name
        # Rescaled curves live in [0, 1].
        assert curve.min() >= -1e-9 and curve.max() <= 1.0 + 1e-9, name


def test_reduce_to_metrics_smoothing_preserves_length():
    n_t, n_vertices = 30, 12
    timeline = np.linspace(0, 1, n_t)[:, None] * np.ones((1, n_vertices))
    atlas = _synthetic_atlas(n_vertices)
    curves = reduce_to_metrics(
        timeline, atlas, smooth_window=5, rescale_0_1=False
    )
    for curve in curves.values():
        assert curve.shape == (n_t,)


def test_custom_metric_weighting_is_respected():
    # Two ROIs; a metric that subtracts one from the other.
    atlas = Atlas(labels=np.array(["x", "y"]))
    timeline = np.array([[2.0, 0.0], [4.0, 0.0]])  # x rises, y flat at 0
    spec = MetricSpec(name="diff", roi_weights={"x": 1.0, "y": -1.0})
    curves = reduce_to_metrics(
        timeline, atlas, metrics=[spec], rescale_0_1=False
    )
    # (1*x + -1*y) / (|1|+|-1|) = x/2  -> [1.0, 2.0]
    assert np.allclose(curves["diff"], [1.0, 2.0])
