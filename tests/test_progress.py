"""Unit tests for the determinate per-clip encode progress bar (Option X).

Covers tribe-brain's must-fixes: monotonic full-band mapping across P windows
(no reset), throttle (final clip always emits), and the never-raise sink guard.
"""
import pytest

from tribescore.progress import ENCODE_HI, ENCODE_LO, encode_frac, make_clip_sink


def _raise(*_a):
    raise RuntimeError("boom")


def _recorder():
    calls = []

    def progress(frac, desc=None):
        calls.append((frac, desc))

    return calls, progress


# --- encode_frac ------------------------------------------------------------
def test_encode_frac_endpoints_and_bounds():
    assert encode_frac(0, 0, 100, 2) == pytest.approx(ENCODE_LO)      # first clip
    assert encode_frac(1, 100, 100, 2) == pytest.approx(ENCODE_HI)    # last clip, last window
    for p in range(4):
        for d in (0, 1, 50, 100, 250):
            assert ENCODE_LO <= encode_frac(p, d, 100, 2) <= ENCODE_HI


def test_encode_frac_monotonic_across_windows():
    P = 3
    seq = [encode_frac(p, d, 100, P) for p in range(P) for d in range(0, 101, 10)]
    assert seq == sorted(seq)  # non-decreasing across every window seam


def test_encode_frac_degrades_to_single_sweep():
    assert encode_frac(0, 0, 50, 1) == pytest.approx(ENCODE_LO)
    assert encode_frac(0, 50, 50, 1) == pytest.approx(ENCODE_HI)


# --- make_clip_sink ---------------------------------------------------------
def test_sink_single_window_labels_and_monotonic():
    calls, progress = _recorder()
    ticks = iter(range(1000))
    sink = make_clip_sink(progress, n_pass=1, clock=lambda: next(ticks))
    for d in range(1, 6):
        sink(d, 5)
    fracs = [c[0] for c in calls]
    assert fracs == sorted(fracs)
    assert fracs[-1] == pytest.approx(ENCODE_HI)
    assert "window 1/1" in calls[-1][1] and "clip 5/5" in calls[-1][1]


def test_sink_detects_second_window_and_stays_monotonic():
    calls, progress = _recorder()
    ticks = iter(range(1000))
    sink = make_clip_sink(progress, n_pass=2, clock=lambda: next(ticks))
    for d in range(1, 4):       # window 1
        sink(d, 3)
    for d in range(1, 4):       # window 2 — counter restarts
        sink(d, 3)
    labels = [c[1] for c in calls]
    assert any("window 1/2" in s for s in labels)
    assert any("window 2/2" in s for s in labels)
    fracs = [c[0] for c in calls]
    assert fracs == sorted(fracs)               # no reset at the seam
    assert fracs[-1] == pytest.approx(ENCODE_HI)


def test_sink_throttles_but_always_emits_final():
    calls, progress = _recorder()
    sink = make_clip_sink(progress, n_pass=1, clock=lambda: 0.0, min_interval=0.4)
    sink(1, 5)   # first emit always (t seeded far in the past)
    sink(2, 5)   # throttled (Δ=0 < 0.4, not final)
    sink(3, 5)   # throttled
    sink(5, 5)   # final clip of the window -> always emits
    assert len(calls) == 2
    assert "clip 1/5" in calls[0][1] and "clip 5/5" in calls[1][1]


# --- fast_encode sink plumbing ---------------------------------------------
def test_fast_encode_sink_set_emit_clear_and_never_raises():
    from tribescore import fast_encode

    seen = []
    fast_encode.set_progress_sink(lambda d, t: seen.append((d, t)))
    fast_encode._emit_progress(3, 10)
    assert seen == [(3, 10)]

    fast_encode.set_progress_sink(_raise)        # a raising sink must be swallowed
    fast_encode._emit_progress(1, 1)             # must NOT raise

    fast_encode.clear_progress_sink()
    fast_encode._emit_progress(9, 9)             # cleared -> no-op
    assert seen == [(3, 10)]
