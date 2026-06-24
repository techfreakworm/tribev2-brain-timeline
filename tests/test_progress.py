"""Unit tests for the in-card determinate progress bar state (Option T).

Covers tribe-brain's must-fixes: monotonic full-band mapping across P windows
(no reset, via the per-window counter-restart detection), the active flag, and
the never-raise fast_encode sink plumbing.
"""
import pytest

from tribescore import progress as P


def test_encode_frac_endpoints_and_bounds():
    P.begin(2)
    assert P.encode_frac() == pytest.approx(0.0)        # nothing encoded yet
    P.sink(1, 10)                                        # window 1, clip 1/10
    assert 0.0 < P.encode_frac() < 0.5
    P.sink(10, 10)                                       # window 1 complete
    assert P.encode_frac() == pytest.approx(0.5)         # 1 of 2 windows
    P.sink(1, 10)                                        # window 2 starts (restart)
    P.sink(10, 10)                                       # window 2 complete
    assert P.encode_frac() == pytest.approx(1.0)


def test_sink_detects_window_pass_and_stays_monotonic():
    P.begin(2)
    fracs = []
    for d in range(1, 6):        # window 1 of 2
        P.sink(d, 5)
        fracs.append(P.encode_frac())
    assert P.snapshot()["pass_idx"] == 0
    for d in range(1, 6):        # window 2 — counter restarts
        P.sink(d, 5)
        fracs.append(P.encode_frac())
    assert P.snapshot()["pass_idx"] == 1
    assert fracs == sorted(fracs)                        # no reset at the seam
    assert fracs[-1] == pytest.approx(1.0)


def test_single_window_degrades_cleanly():
    P.begin(1)
    P.sink(1, 4)
    assert P.encode_frac() == pytest.approx(0.25)
    P.sink(4, 4)
    assert P.encode_frac() == pytest.approx(1.0)


def test_begin_end_active_flag_and_snapshot_is_copy():
    P.begin(3)
    snap = P.snapshot()
    assert snap["active"] is True and snap["n_pass"] == 3
    snap["done"] = 999                                   # snapshot is a copy
    assert P.snapshot()["done"] == 0
    P.end()
    assert P.snapshot()["active"] is False


def test_fast_encode_sink_plumbing_never_raises():
    from tribescore import fast_encode

    seen = []
    fast_encode.set_progress_sink(lambda d, t: seen.append((d, t)))
    fast_encode._emit_progress(2, 9)
    assert seen == [(2, 9)]

    def _boom(*_a):
        raise RuntimeError("x")

    fast_encode.set_progress_sink(_boom)
    fast_encode._emit_progress(1, 1)                     # must NOT raise
    fast_encode.clear_progress_sink()
    fast_encode._emit_progress(5, 5)                     # cleared -> no-op
    assert seen == [(2, 9)]
