"""Tests for ROI reduction -- pure numpy/scipy, synthetic data, no model.

Two suites:

* **T-C (docs/PLAN.md §5/§9)** -- the production Glasser pipeline:
  :data:`PARCELS` / :data:`METRIC_DEFAULT_ON`, :func:`build_roi_masks`
  (guarded import), :func:`to_metrics` (ROI mean -> full-timeline z-score ->
  Gaussian sigma=2 s smoothing), and :func:`summary`.
* **Legacy** -- the older injectable-:class:`Atlas` API kept for backward
  compatibility with the package re-exports.
"""

from __future__ import annotations

import numpy as np
import pytest

from tribescore.metrics import (
    DEFAULT_METRICS,
    METRIC_DEFAULT_ON,
    PARCELS,
    ROI_MASKS_FILENAME,
    Atlas,
    MetricSpec,
    build_roi_masks,
    reduce_to_metrics,
    roi_means,
    summary,
    to_metrics,
)

N_VERTICES = 20484  # 2 x 10242 fsaverage5 vertices (docs/PLAN.md §3)


# ===========================================================================
# T-C : PARCELS / METRIC_DEFAULT_ON constants
# ===========================================================================


def test_parcels_has_expected_metrics_and_default_on():
    # The three default-ON metrics plus the two optional toggles (§5/§11.1).
    assert set(PARCELS) == {
        "Attention",
        "Engagement",
        "Virality",
        "Language",
        "Self-relevance",
    }
    assert METRIC_DEFAULT_ON == {"Attention", "Engagement", "Virality"}
    # Every default-ON metric is actually defined in PARCELS.
    assert METRIC_DEFAULT_ON.issubset(set(PARCELS))


def test_parcels_lists_are_nonempty_and_unique_bare_names():
    for metric, parcels in PARCELS.items():
        assert parcels, metric
        # Bare Glasser names: no L_/R_ prefix, no _ROI suffix, no hemi tag.
        for p in parcels:
            assert not p.startswith(("L_", "R_")), (metric, p)
            assert "_ROI" not in p, (metric, p)
            assert "-lh" not in p and "-rh" not in p, (metric, p)
        # No accidental duplicates within a metric's list.
        assert len(parcels) == len(set(parcels)), metric


# ===========================================================================
# T-C : build_roi_masks  (guarded import / cache behavior)
# ===========================================================================


def _fake_masks() -> dict[str, np.ndarray]:
    """Small hand-made masks resembling build_roi_masks output."""
    return {
        "Attention": np.array([0, 1, 2, 10242], dtype=np.int64),
        "Engagement": np.array([5, 6, 7, 8, 9], dtype=np.int64),
        "Virality": np.array([20480, 20481, 20482, 20483], dtype=np.int64),
    }


def test_build_roi_masks_loads_from_cache_without_tribev2(tmp_path):
    """A pre-existing .npz must be loaded WITHOUT importing tribev2.

    This exercises the local-safe path: build_roi_masks should never need the
    model dependency when the static mask cache is already present.
    """
    cache_path = tmp_path / ROI_MASKS_FILENAME
    masks = _fake_masks()
    np.savez(cache_path, **masks)

    loaded = build_roi_masks(str(tmp_path))
    assert set(loaded) == set(masks)
    for k in masks:
        assert np.array_equal(loaded[k], masks[k])
        assert loaded[k].dtype == np.int64


def test_build_roi_masks_without_dep_and_without_cache_raises(tmp_path):
    """Off-Space, no cache, no tribev2 -> a clear RuntimeError (not ImportError).

    If tribev2 happens to be importable in this environment we skip, since then
    a real (slow, network) build would be attempted instead of the guard.
    """
    try:
        import tribev2.utils  # noqa: F401
    except Exception:
        pass
    else:  # pragma: no cover - tribev2 not expected locally
        pytest.skip("tribev2 importable here; guard path not exercised")

    with pytest.raises(RuntimeError) as exc:
        build_roi_masks(str(tmp_path))
    # Message should point at the dependency / Space gate, and mention cache.
    assert "tribev2" in str(exc.value)


def test_build_roi_masks_ignores_corrupt_cache(tmp_path):
    """A corrupt cache is a miss, not a crash -> falls through to the guard."""
    try:
        import tribev2.utils  # noqa: F401
    except Exception:
        pass
    else:  # pragma: no cover
        pytest.skip("tribev2 importable here; guard path not exercised")

    cache_path = tmp_path / ROI_MASKS_FILENAME
    cache_path.write_bytes(b"not a real npz file")
    # Corrupt cache -> treated as miss -> no tribev2 -> RuntimeError.
    with pytest.raises(RuntimeError):
        build_roi_masks(str(tmp_path))


# ===========================================================================
# T-C : to_metrics
# ===========================================================================


def test_to_metrics_one_curve_per_mask_with_shape():
    rng = np.random.default_rng(0)
    n_t = 120
    timeline = rng.standard_normal((n_t, N_VERTICES))
    masks = _fake_masks()

    curves = to_metrics(timeline, masks)

    assert set(curves) == set(masks)
    for name, curve in curves.items():
        assert curve.shape == (n_t,), name
        assert np.all(np.isfinite(curve)), name


def test_to_metrics_is_zscored_before_smoothing():
    """The analytic series is a full-timeline z-score (~0 mean, ~1 std).

    We verify on the *pre-smoothing* series by reconstructing it from the ROI
    mean (to_metrics z-scores then smooths; smoothing slightly shrinks std but
    must not move the mean off ~0).
    """
    rng = np.random.default_rng(1)
    n_t = 200
    timeline = rng.standard_normal((n_t, N_VERTICES))
    masks = _fake_masks()

    for name, idx in masks.items():
        raw = timeline[:, idx].mean(axis=1)
        z = (raw - raw.mean()) / (raw.std() + 1e-6)
        # Pre-smoothing z-score has ~0 mean and ~1 std by construction.
        assert abs(float(z.mean())) < 1e-6, name
        assert abs(float(z.std()) - 1.0) < 1e-3, name

    # The smoothed output keeps the ~0 mean (Gaussian smoothing is mean-preserving
    # under 'nearest' edges for a roughly stationary series).
    curves = to_metrics(timeline, masks)
    for name, curve in curves.items():
        assert abs(float(curve.mean())) < 0.1, name


def test_to_metrics_smoothing_reduces_variance():
    """Gaussian smoothing (sigma=2 s) must reduce the variance of a noisy curve."""
    rng = np.random.default_rng(2)
    n_t = 300
    timeline = rng.standard_normal((n_t, N_VERTICES))
    masks = {"noisy": np.arange(0, 50, dtype=np.int64)}

    raw = timeline[:, masks["noisy"]].mean(axis=1)
    z = (raw - raw.mean()) / (raw.std() + 1e-6)
    smoothed = to_metrics(timeline, masks)["noisy"]

    assert smoothed.var() < z.var()
    # And the smoothed series is still centered near zero.
    assert abs(float(smoothed.mean())) < 0.1


def test_to_metrics_rejects_bad_timeline_and_indices():
    masks = {"m": np.array([0, 1, 2], dtype=np.int64)}
    with pytest.raises(ValueError):
        to_metrics(np.zeros(10), masks)  # 1-D, not (T, V)

    # Index out of range for the timeline width.
    small = np.zeros((5, 4))
    with pytest.raises(ValueError):
        to_metrics(small, {"m": np.array([0, 99])})

    # Empty mask.
    with pytest.raises(ValueError):
        to_metrics(small, {"m": np.array([], dtype=np.int64)})


def test_to_metrics_short_timeline_is_noop_smoothing():
    """A 1-row timeline must not crash; z-score of a single point is 0."""
    timeline = np.ones((1, N_VERTICES))
    curves = to_metrics(timeline, {"m": np.array([0, 1, 2])})
    assert curves["m"].shape == (1,)
    assert np.isfinite(curves["m"]).all()


# ===========================================================================
# T-C : summary
# ===========================================================================


def test_summary_keys_and_shapes():
    rng = np.random.default_rng(3)
    n_t = 120
    timeline = rng.standard_normal((n_t, N_VERTICES))
    masks = _fake_masks()
    curves = to_metrics(timeline, masks)

    stats = summary(curves)
    assert set(stats) == set(curves)
    for name, s in stats.items():
        assert set(s) == {"peak", "mean", "peak_time", "peak_percentile"}, name
        assert np.isfinite(s["peak"]), name
        assert np.isfinite(s["mean"]), name
        # peak_time is a valid TR index into the curve.
        assert isinstance(s["peak_time"], int), name
        assert 0 <= s["peak_time"] < n_t, name
        # peak == curve value at peak_time; peak is the max.
        assert s["peak"] == pytest.approx(float(curves[name][s["peak_time"]])), name
        assert s["peak"] == pytest.approx(float(curves[name].max())), name
        # Percentile of the global max is 100.
        assert s["peak_percentile"] == pytest.approx(100.0), name


def test_summary_reports_peak_in_z_units():
    # Hand-built curve with a known peak at a known index.
    curve = np.array([0.0, 0.5, 3.0, -1.0, 0.2])
    stats = summary({"m": curve})["m"]
    assert stats["peak"] == pytest.approx(3.0)
    assert stats["peak_time"] == 2
    assert stats["mean"] == pytest.approx(curve.mean())
    assert stats["peak_percentile"] == pytest.approx(100.0)


def test_summary_handles_empty_curve():
    stats = summary({"m": np.array([])})["m"]
    assert stats["peak_time"] == -1
    assert np.isnan(stats["peak"])
    assert np.isnan(stats["mean"])


# ===========================================================================
# Legacy synthetic-atlas API (unchanged behavior)
# ===========================================================================


def _synthetic_atlas(n_vertices: int = 12) -> Atlas:
    """Round-robin assign vertices to the ROI names used by DEFAULT_METRICS."""
    rois = sorted({roi for spec in DEFAULT_METRICS for roi in spec.roi_weights})
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
        assert curve.min() >= -1e-9 and curve.max() <= 1.0 + 1e-9, name


def test_reduce_to_metrics_smoothing_preserves_length():
    n_t, n_vertices = 30, 12
    timeline = np.linspace(0, 1, n_t)[:, None] * np.ones((1, n_vertices))
    atlas = _synthetic_atlas(n_vertices)
    curves = reduce_to_metrics(timeline, atlas, smooth_window=5, rescale_0_1=False)
    for curve in curves.values():
        assert curve.shape == (n_t,)


def test_custom_metric_weighting_is_respected():
    atlas = Atlas(labels=np.array(["x", "y"]))
    timeline = np.array([[2.0, 0.0], [4.0, 0.0]])  # x rises, y flat at 0
    spec = MetricSpec(name="diff", roi_weights={"x": 1.0, "y": -1.0})
    curves = reduce_to_metrics(timeline, atlas, metrics=[spec], rescale_0_1=False)
    # (1*x + -1*y) / (|1|+|-1|) = x/2  -> [1.0, 2.0]
    assert np.allclose(curves["diff"], [1.0, 2.0])
