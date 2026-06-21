"""Tests for the windowed inference engine -- pure numpy, no model.

These run anywhere ``numpy`` + ``pytest`` are importable. They never touch
``torch`` or ``tribev2``: the per-window inference is replaced by a synthetic
``infer_fn`` that returns deterministic numpy arrays. This is exactly the
injection seam :func:`tribescore.windowing.run_windowed` is designed around.
"""

from __future__ import annotations

import numpy as np
import pytest

from tribescore.windowing import plan_windows, run_windowed


# ---------------------------------------------------------------------------
# Synthetic infer_fn factories
# ---------------------------------------------------------------------------


def make_infer_fn(n_vertices: int = 8, tr_s: float = 1.0):
    """Build a synthetic per-window ``infer_fn``.

    For a window ``[start_s, end_s)`` it emits one row per ``tr_s`` seconds.
    Each row's value encodes its *absolute* time, so a correctly stitched
    timeline is easy to reason about: vertex ``v`` at absolute time ``t`` is
    ``t + v``.
    """

    def infer_fn(events, start_s: float, end_s: float):
        offsets = np.arange(0.0, max(end_s - start_s, tr_s), tr_s)
        abs_times = start_s + offsets
        # (t, n_vertices): broadcast absolute time across vertices, + vertex id.
        activity = abs_times[:, None] + np.arange(n_vertices)[None, :]
        return activity, offsets

    return infer_fn


def empty_infer_fn(events, start_s: float, end_s: float):
    """An ``infer_fn`` that always returns zero rows (no events in window)."""
    return np.zeros((0, 4)), np.zeros((0,))


# ---------------------------------------------------------------------------
# plan_windows
# ---------------------------------------------------------------------------


def test_plan_windows_covers_timeline_with_overlap():
    bounds = plan_windows(total_duration_s=10.0, window_s=4.0, hop_s=2.0)
    assert bounds.ndim == 2 and bounds.shape[1] == 2
    # Starts advance by hop; first starts at 0; last window ends exactly at total.
    assert bounds[0, 0] == 0.0
    assert np.allclose(np.diff(bounds[:, 0]), 2.0)
    assert bounds[-1, 1] == pytest.approx(10.0)
    # Every window is clamped within the timeline.
    assert np.all(bounds[:, 1] <= 10.0 + 1e-9)


def test_plan_windows_rejects_gap():
    # hop_s > window_s would leave the timeline partially unscored.
    with pytest.raises(ValueError):
        plan_windows(total_duration_s=10.0, window_s=2.0, hop_s=5.0)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_plan_windows_rejects_nonpositive(bad):
    with pytest.raises(ValueError):
        plan_windows(total_duration_s=bad, window_s=2.0, hop_s=1.0)


# ---------------------------------------------------------------------------
# run_windowed -- the core contract
# ---------------------------------------------------------------------------


def test_run_windowed_shape_and_axis_monotonicity():
    n_vertices = 8
    total = 30.0
    timeline, time_axis = run_windowed(
        events=object(),  # opaque sentinel -- never inspected
        infer_fn=make_infer_fn(n_vertices=n_vertices),
        window_s=10.0,
        hop_s=5.0,
        total_duration_s=total,
        target_hz=1.0,
        normalize=False,
    )

    # Shape: (T, n_vertices) with T = floor(total*hz)+1.
    expected_T = int(np.floor(total * 1.0)) + 1
    assert timeline.shape == (expected_T, n_vertices)

    # Time axis: right length, starts at 0, ends at total, strictly increasing.
    assert time_axis.shape == (expected_T,)
    assert time_axis[0] == pytest.approx(0.0)
    assert time_axis[-1] == pytest.approx(total)
    assert np.all(np.diff(time_axis) > 0), "time_axis must be strictly monotonic"

    # No NaNs/Infs anywhere in the stitched output.
    assert np.all(np.isfinite(timeline))


def test_run_windowed_recovers_known_signal():
    """With the time-encoding synthetic fn, stitched values ~= t + vertex_id."""
    n_vertices = 4
    total = 20.0
    timeline, time_axis = run_windowed(
        events=object(),
        infer_fn=make_infer_fn(n_vertices=n_vertices, tr_s=1.0),
        window_s=10.0,
        hop_s=5.0,
        total_duration_s=total,
        target_hz=1.0,
        normalize=False,
        taper=False,  # plain averaging makes the expected value exact
    )
    expected = time_axis[:, None] + np.arange(n_vertices)[None, :]
    # Nearest-neighbour resampling can be off by < 1 TR at the seams; allow it.
    assert np.allclose(timeline, expected, atol=1.0)


def test_run_windowed_normalize_produces_zero_mean_unit_var():
    timeline, _ = run_windowed(
        events=object(),
        infer_fn=make_infer_fn(n_vertices=6),
        window_s=10.0,
        hop_s=5.0,
        total_duration_s=40.0,
        normalize=True,
    )
    # Each (non-constant) vertex column should be ~standardized.
    assert np.allclose(timeline.mean(axis=0), 0.0, atol=1e-6)
    std = timeline.std(axis=0)
    nonconst = std > 1e-6
    assert np.allclose(std[nonconst], 1.0, atol=1e-6)


def test_run_windowed_non_overlapping_hop_equals_window():
    # hop_s == window_s is the degenerate non-overlapping case; must still work.
    timeline, time_axis = run_windowed(
        events=object(),
        infer_fn=make_infer_fn(n_vertices=3),
        window_s=10.0,
        hop_s=10.0,
        total_duration_s=30.0,
        normalize=False,
    )
    assert timeline.shape[0] == time_axis.shape[0]
    assert np.all(np.diff(time_axis) > 0)
    assert np.all(np.isfinite(timeline))


def test_run_windowed_raises_when_all_windows_empty():
    with pytest.raises(ValueError):
        run_windowed(
            events=object(),
            infer_fn=empty_infer_fn,
            window_s=10.0,
            hop_s=5.0,
            total_duration_s=20.0,
        )


def test_run_windowed_requires_known_duration():
    # No total_duration_s and a sentinel events object -> cannot infer length.
    with pytest.raises(ValueError):
        run_windowed(
            events=object(),
            infer_fn=make_infer_fn(),
            window_s=10.0,
            hop_s=5.0,
            total_duration_s=None,
        )


def test_run_windowed_rejects_malformed_window_result():
    def bad_infer_fn(events, start_s, end_s):
        # 1-D activity is invalid; must be (t, n_vertices).
        return np.arange(5.0), np.arange(5.0)

    with pytest.raises(ValueError):
        run_windowed(
            events=object(),
            infer_fn=bad_infer_fn,
            window_s=10.0,
            hop_s=5.0,
            total_duration_s=20.0,
        )
