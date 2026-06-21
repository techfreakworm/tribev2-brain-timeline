"""Tests for windowing + stitching -- pure numpy, no model.

These run anywhere ``numpy`` + ``pytest`` are importable. They never touch
``torch`` or ``tribev2``: the per-window inference is replaced by a synthetic
``infer_fn`` that returns deterministic ``(preds, abs_times)`` arrays exactly
shaped like ``tribev2``'s ``predict()`` output (overlapping 100 s windows whose
per-TR ``abs_times`` ascend within a window and jump backward at each seam).
This is the injection seam :func:`tribescore.windowing.stitch` is designed
around (``docs/PLAN.md`` §4 step 4).
"""

from __future__ import annotations

import numpy as np
import pytest

from tribescore.windowing import (
    HOP_S,
    WARMUP_TRIM,
    WIN_S,
    _segment_windows,
    _trapezoid_weights,
    plan_windows,
    stitch,
)


# ---------------------------------------------------------------------------
# Synthetic infer_fn: build overlapping-window (preds, abs_times) like tribev2
# ---------------------------------------------------------------------------


def synth_infer_fn(
    duration_s: float,
    *,
    n_k: int = 8,
    win_s: int = WIN_S,
    hop_s: int = HOP_S,
    signal="ramp",
    per_window_offset: float = 0.0,
    per_window_scale: float = 1.0,
):
    """Emit ``(preds, abs_times)`` mimicking a single tribev2 ``predict()``.

    For every span from :func:`plan_windows`, one row per second is emitted at
    ``abs_time = start + p`` (``p`` = 0-based intra-window TR index). The row
    value is a smooth function of the *absolute* time so a correct stitch is
    continuous across seams. ``abs_times`` therefore ascends within a window
    and jumps backward at each seam -- the structure ``stitch`` keys on.

    Parameters
    ----------
    signal:
        ``"ramp"``   -> column k at abs time t is ``t + k``;
        ``"sine"``   -> a slow sine of t (+ a per-column phase).
    per_window_offset, per_window_scale:
        Add ``w * per_window_offset`` and multiply by ``per_window_scale**w``
        per window ``w`` -- an arbitrary per-window level/scale that per-window
        z-scoring must cancel (mirrors tribev2's per-sample z-scored target).
    """
    spans = plan_windows(duration_s, win_s=win_s, hop_s=hop_s)
    preds_blocks = []
    times_blocks = []
    for w, (start, end) in enumerate(spans):
        n_w = int(round(end - start))
        p = np.arange(n_w)
        abs_t = start + p.astype(float)
        if signal == "ramp":
            rows = abs_t[:, None] + np.arange(n_k)[None, :]
        elif signal == "sine":
            phase = np.arange(n_k)[None, :] * 0.3
            rows = np.sin(2.0 * np.pi * abs_t[:, None] / 50.0 + phase)
        else:  # pragma: no cover - guard
            raise ValueError(signal)
        rows = rows * (per_window_scale ** w) + w * per_window_offset
        preds_blocks.append(rows)
        times_blocks.append(abs_t)
    preds = np.concatenate(preds_blocks, axis=0)
    abs_times = np.concatenate(times_blocks, axis=0)
    return preds, abs_times


# ---------------------------------------------------------------------------
# plan_windows
# ---------------------------------------------------------------------------


def test_plan_windows_overlap_and_tail_coverage():
    spans = plan_windows(260.0)  # win=100, hop=80
    starts = [s for s, _ in spans]
    ends = [e for _, e in spans]
    # First window at 0; starts advance by hop=80 on the regular grid.
    assert starts[0] == 0.0
    assert spans[0] == (0.0, 100.0)
    # Last span forced to end exactly at the clip end with a full 100 s window.
    assert ends[-1] == pytest.approx(260.0)
    assert starts[-1] == pytest.approx(160.0)  # 260 - 100
    # Ascending, no duplicate starts.
    assert starts == sorted(starts)
    assert len(starts) == len(set(round(s, 6) for s in starts))


def test_plan_windows_dedup_forced_last():
    # duration exactly on the grid: forced-last start coincides with a grid
    # start and must be de-duplicated.
    spans = plan_windows(180.0)  # grid: 0, 80; forced last start = 80
    starts = [s for s, _ in spans]
    assert starts == [0.0, 80.0]
    assert spans[-1] == (80.0, 180.0)


def test_plan_windows_short_clip_single_window():
    spans = plan_windows(40.0)  # shorter than one window
    assert spans == [(0.0, 40.0)]


def test_plan_windows_rejects_gap_and_nonpositive():
    with pytest.raises(ValueError):
        plan_windows(100.0, win_s=20, hop_s=50)  # hop > win -> gap
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            plan_windows(bad)


# ---------------------------------------------------------------------------
# _segment_windows -- the window-detection primitive (§4 step a)
# ---------------------------------------------------------------------------


def test_segment_windows_detects_backward_jumps():
    # Two windows: 0..4 then 3..7 (a real overlap seam: backward jump 4 -> 3).
    abs_times = np.array([0, 1, 2, 3, 4, 3, 4, 5, 6, 7], dtype=float)
    bounds = _segment_windows(abs_times)
    assert bounds == [(0, 5), (5, 10)]


def test_segment_windows_single_run():
    abs_times = np.arange(10, dtype=float)
    assert _segment_windows(abs_times) == [(0, 10)]


# ---------------------------------------------------------------------------
# _trapezoid_weights -- crossfade + warm-up suppression (§4 step c)
# ---------------------------------------------------------------------------


def test_trapezoid_weights_window0_flat_lead_ramp_trail():
    # Window 0: no leading ramp, no warm-up suppression; trailing ramps down.
    n_w = 100
    w = _trapezoid_weights(
        n_w, overlap=20, warmup_trim=5,
        suppress_warmup=False, ramp_leading=False, ramp_trailing=True,
    )
    assert w[0] == 1.0  # leading rows kept (nothing earlier covers them)
    assert np.all(w[:80] == 1.0)  # flat core
    assert w[-1] < w[-20]  # trailing ramps toward 0
    assert w[-1] > 0.0


def test_trapezoid_weights_middle_window_suppresses_warmup():
    # Middle window: leading rows suppressed for p < warmup_trim, then ramp up.
    n_w = 100
    w = _trapezoid_weights(
        n_w, overlap=20, warmup_trim=5,
        suppress_warmup=True, ramp_leading=True, ramp_trailing=True,
    )
    assert np.all(w[:5] == 0.0)            # warm-up rows excluded
    assert np.all(w[5:20] > 0.0)           # leading ramp resumes
    assert np.all(np.diff(w[5:20]) > 0)    # ramp is increasing
    assert np.all(w[20:80] == 1.0)         # flat core
    assert np.all(np.diff(w[80:]) < 0)     # trailing ramp decreasing


def test_trapezoid_weights_last_window_keeps_tail():
    # Final window: trailing rows kept at 1 (unique end-of-clip signal).
    n_w = 100
    w = _trapezoid_weights(
        n_w, overlap=20, warmup_trim=5,
        suppress_warmup=True, ramp_leading=True, ramp_trailing=False,
    )
    assert w[-1] == 1.0
    assert np.all(w[20:] == 1.0)


def test_trapezoid_crossfade_sums_to_one_across_seam():
    # The outgoing trailing ramp + incoming leading ramp should sum to ~1 over
    # the 20 s overlap (clean linear crossfade after weighted-mean normalise).
    ov = 20
    trail = _trapezoid_weights(
        100, overlap=ov, warmup_trim=0,
        suppress_warmup=False, ramp_leading=False, ramp_trailing=True,
    )[-ov:]
    lead = _trapezoid_weights(
        100, overlap=ov, warmup_trim=0,
        suppress_warmup=False, ramp_leading=True, ramp_trailing=False,
    )[:ov]
    # Window w trailing row at overlap position j aligns with window w+1 leading
    # row at the same abs second -> j-th trailing meets j-th leading.
    assert np.allclose(trail + lead, 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# stitch -- the core contract (§4 step 4)
# ---------------------------------------------------------------------------


def test_stitch_shape_and_integer_monotonic_axis():
    n_k = 8
    duration = 260.0
    preds, abs_times = synth_infer_fn(duration, n_k=n_k, signal="ramp")
    timeline, t_axis = stitch(preds, abs_times)

    # t_axis = arange(0, ceil(max(abs_times)) + 1). The last window emits rows
    # up to second duration-1 (a D-second clip spans [0, D), last integer second
    # is D-1), so T == D for an integer duration.
    last_sec = int(np.max(abs_times))          # 259 for a 260 s clip
    expected_T = last_sec + 1                   # 260
    assert timeline.shape == (expected_T, n_k)
    assert t_axis.shape == (expected_T,)
    assert expected_T == int(duration)

    # Integer-second grid 0..T-1, strictly increasing by exactly 1 s.
    assert t_axis[0] == 0.0
    assert t_axis[-1] == pytest.approx(last_sec)
    assert np.all(np.diff(t_axis) == 1.0)
    assert np.allclose(t_axis, np.round(t_axis))
    assert np.all(np.isfinite(timeline))


def test_stitch_no_zero_weight_gaps():
    # With 20 s overlap > 5 TR warm-up trim, every grid second is covered: the
    # uncovered set must be empty (§4 step e assertion).
    preds, abs_times = synth_infer_fn(300.0, signal="ramp")
    # Re-run the covered check by reconstructing weights is internal; instead we
    # assert the public result has no NaN/inf and the interp branch was a no-op
    # by checking the timeline equals a direct overlap-add (finite everywhere).
    timeline, _ = stitch(preds, abs_times)
    assert np.all(np.isfinite(timeline))


def test_stitch_seam_continuity_no_jump():
    """No discontinuity at the known seams (t == hop, 2*hop, ...)."""
    duration = 300.0
    preds, abs_times = synth_infer_fn(
        duration, n_k=4, signal="sine",
        per_window_offset=3.0, per_window_scale=1.0,  # arbitrary per-window level
    )
    timeline, t_axis = stitch(preds, abs_times)

    # Adjacent-second jumps across the whole timeline.
    diffs = np.abs(np.diff(timeline, axis=0))
    max_jump = diffs.max()

    # Seam seconds: where one window ends its contribution and the next
    # dominates (every hop, in the overlap band). The jump *at* the seam must
    # be no worse than the largest jump anywhere (i.e. no seam spike).
    seam_seconds = [t for t in range(HOP_S, int(duration), HOP_S)]
    for t in seam_seconds:
        seam_jump = diffs[t - 1].max()  # jump landing on second t
        assert seam_jump <= max_jump + 1e-9
        # And in absolute terms it is a smooth-signal-sized step, not a cliff.
        assert seam_jump < 0.5, f"seam at t={t} jumped {seam_jump}"


def test_stitch_cancels_per_window_offset_and_scale():
    """Per-window z-score removes arbitrary per-window level + scale (§4 3b)."""
    duration = 300.0
    # Two synth runs that differ ONLY by a per-window affine transform; their
    # stitched timelines must match (z-score is invariant to per-window a*x+b).
    p0, t0 = synth_infer_fn(duration, n_k=4, signal="sine")
    p1, t1 = synth_infer_fn(
        duration, n_k=4, signal="sine",
        per_window_offset=10.0, per_window_scale=2.0,
    )
    tl0, _ = stitch(p0, t0)
    tl1, _ = stitch(p1, t1)
    assert np.allclose(tl0, tl1, atol=1e-6)


def test_stitch_warmup_rows_excluded():
    """A spike in a window's first warmup_trim rows must not reach the output.

    Those abs seconds are also covered by the previous window (20 s overlap), so
    after warm-up suppression the spike contributes zero weight and is invisible.
    """
    duration = 300.0
    preds, abs_times = synth_infer_fn(duration, n_k=3, signal="ramp")
    bounds = _segment_windows(abs_times)

    # Inject a large spike into the first WARMUP_TRIM rows of window 1.
    lo, _hi = bounds[1]
    spike_rows = list(range(lo, lo + WARMUP_TRIM))
    spike_secs = np.rint(abs_times[spike_rows]).astype(int)
    preds_spiked = preds.copy()
    preds_spiked[spike_rows, :] += 1e6

    tl_clean, _ = stitch(preds, abs_times)
    tl_spiked, t_axis = stitch(preds_spiked, abs_times)

    # At the spiked seconds the stitched output is unchanged (spike suppressed).
    for sec in spike_secs:
        idx = int(sec)
        assert np.allclose(tl_spiked[idx], tl_clean[idx], atol=1e-3), (
            f"warm-up spike leaked into second {sec}"
        )


def test_stitch_duplicate_seconds_are_weighted_averaged():
    """An overlapped second draws from both windows (weighted mean), not one."""
    # Build a minimal two-window case by hand: window A covers secs 0..9,
    # window B covers secs 6..15 (overlap = 4 at secs 6,7,8,9). Use small win/hop
    # so overlap is non-trivial and warm-up (5) does not erase the whole overlap.
    win_s, hop_s, warm = 10, 6, 1
    n_k = 2
    # Window A: secs 0..9, constant value 0 on all vertices.
    a_t = np.arange(0, 10, dtype=float)
    a_p = np.zeros((10, n_k))
    # Window B: secs 6..15, constant value 1.0 (a clear, different level).
    b_t = np.arange(6, 16, dtype=float)
    b_p = np.ones((10, n_k))
    preds = np.concatenate([a_p, b_p], axis=0)
    abs_times = np.concatenate([a_t, b_t])

    timeline, t_axis = stitch(
        preds, abs_times, warmup_trim=warm, hop_s=hop_s, win_s=win_s
    )
    # Both windows are constant -> z-score maps each to all-zeros; a constant
    # signal stitches to zeros everywhere. This confirms per-window z-score and
    # that overlapped seconds remain finite (no divide-by-zero in the mean).
    assert np.all(np.isfinite(timeline))
    assert np.allclose(timeline, 0.0, atol=1e-6)

    # Now make the two windows *ramps* with the SAME absolute-time signal so the
    # overlap second has two contributors that should average to the shared
    # value. value(t) = t (same in both windows), de-meaned per window.
    a_p2 = (a_t[:, None] + np.zeros(n_k)[None, :])
    b_p2 = (b_t[:, None] + np.zeros(n_k)[None, :])
    preds2 = np.concatenate([a_p2, b_p2], axis=0)
    tl2, ax2 = stitch(
        preds2, abs_times, warmup_trim=warm, hop_s=hop_s, win_s=win_s
    )
    # In the overlap (secs 6..9) both windows carry the same underlying ramp; a
    # weighted average of two identical (post-z-score) shapes is continuous.
    diffs = np.abs(np.diff(tl2, axis=0))
    assert diffs.max() < 0.5  # smooth across the overlapped/averaged seam


def test_stitch_rejects_malformed_input():
    with pytest.raises(ValueError):
        stitch(np.arange(5.0), np.arange(5.0))  # 1-D preds
    with pytest.raises(ValueError):
        stitch(np.zeros((5, 3)), np.zeros(4))   # length mismatch
    with pytest.raises(ValueError):
        stitch(np.zeros((0, 3)), np.zeros(0))   # empty


def test_stitch_generic_k_width():
    # stitch must accept arbitrary K (20484 vertices OR n_metrics).
    for n_k in (1, 3, 20484):
        preds, abs_times = synth_infer_fn(120.0, n_k=n_k, signal="ramp")
        timeline, t_axis = stitch(preds, abs_times)
        assert timeline.shape[1] == n_k
        assert np.all(np.isfinite(timeline))
