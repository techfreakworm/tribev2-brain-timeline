"""Pure helpers for the determinate per-clip encode progress bar (Option X).

The Gradio app (``app.py``) builds a per-clip *sink* here and registers it on
:mod:`tribescore.fast_encode`; the V-JEPA2 encode loop then calls it once per
clip with ``(done, total)``. The sink maps that into the overall progress bar
and drives the native ``gr.Progress`` widget.

These helpers are pure (the ``progress`` callback and the clock are injected),
so they unit-test without importing the heavy Gradio app. Design + rationale:
``docs/superpowers/specs/2026-06-24-progress-bar-design.md``.

LOCAL-ONLY: the sink is registered only on the in-process (Apple-silicon MPS)
path. On the HF Space ``_gpu_infer`` runs in a forked ``@spaces.GPU`` subprocess
that cannot see a parent-set sink, so the coarse stage progress is the Space
fallback there (and :func:`fast_encode._emit_progress` no-ops when unset).
"""
from __future__ import annotations

import time as _time

# The V-JEPA2 encode occupies the 0.15->0.70 slice of the overall progress bar;
# the stitch/metrics/render stages keep their existing 0.75/0.92/1.0 marks.
ENCODE_LO: float = 0.15
ENCODE_HI: float = 0.70


def encode_frac(pass_idx: int, done: int, total: int, n_pass: int) -> float:
    """Monotonic map of (window pass, within-window clip) into the encode band.

    With the dedup output-cache OFF, tribev2 re-runs the encode loop once per
    window, so the bar advances across ``n_pass`` windows without resetting:
    ``frac = LO + (HI-LO) * (pass_idx + done/total) / n_pass``.

    ``pass_idx`` is the 0-based window index; ``done``/``total`` are clips within
    the current window; ``n_pass`` is the window count from ``plan_windows()``.
    Clamped to ``[ENCODE_LO, ENCODE_HI]`` so an under-/over-estimated ``n_pass``
    can never overshoot into the stitch band. Degrades cleanly to a single
    0->100% sweep when ``n_pass == 1`` (e.g. once the output cache lands).
    """
    n_pass = max(1, n_pass)
    within = (done / total) if total else 0.0
    frac = ENCODE_LO + (ENCODE_HI - ENCODE_LO) * ((pass_idx + within) / n_pass)
    return min(ENCODE_HI, max(ENCODE_LO, frac))


def make_clip_sink(progress, n_pass: int, *, clock=None, min_interval: float = 0.4):
    """Build a per-clip ``sink(done, total)`` for ``fast_encode.set_progress_sink``.

    Tracks window passes by the per-window counter restart (``done`` returning to
    a value <= the last seen), maps monotonically via :func:`encode_frac`, and
    throttles UI updates to ``min_interval`` seconds — except the final clip of a
    window (``done == total``), which always emits. ``progress`` is the
    ``gr.Progress`` callable; ``clock`` is injectable for tests.
    """
    clock = clock or _time.monotonic
    st = {"pass": 0, "last": 0, "t": -1e9}

    def _sink(done: int, total: int) -> None:
        # New extraction pass (next window) when the per-window counter restarts.
        if done <= st["last"]:
            st["pass"] += 1
        st["last"] = done
        now = clock()
        if now - st["t"] < min_interval and done != total:
            return  # throttle the UI update only; pass tracking already advanced
        st["t"] = now
        window = min(st["pass"] + 1, max(1, n_pass))
        progress(
            encode_frac(st["pass"], done, total, n_pass),
            desc=f"Encoding video · clip {done}/{total} · window {window}/{max(1, n_pass)}",
        )

    return _sink
