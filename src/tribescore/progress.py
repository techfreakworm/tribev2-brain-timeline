"""Live per-clip progress state for the determinate in-card bar (Option T).

The native ``gr.Progress`` bar does NOT render in this app's loading-card layout
(smoke-test confirmed: the sink fires but Gradio's status-tracker stays empty).
So we drive the card's OWN HTML instead: the V-JEPA2 encode sink writes the
per-clip counters into a module-global dict here, and a ``gr.Timer`` in app.py
reads it ~3x/s and re-renders the loading card. tribe-brain-reviewed (Option T
over the worker-thread generator: no manual MPS thread, no generator into the
Gradio-6.11 visibility chain). Design doc:
``docs/superpowers/specs/2026-06-24-progress-bar-design.md``.

LOCAL-ONLY: ``begin()`` is called only off-Space, so on the HF Space ``active``
stays False, the Timer reads inactive → ``gr.skip()``, and the coarse stage
``progress()`` calls remain the Space fallback. Plain int writes are GIL-atomic,
so no lock is needed for the single-writer (encode) / single-reader (Timer) case.
"""
from __future__ import annotations

#: Live state. ``active`` gates the Timer; ``done/total`` are clips within the
#: current window; ``pass_idx`` is the 0-based window; ``n_pass`` the window
#: count from ``plan_windows()``; ``_last`` tracks the per-window counter restart.
_STATE = {"active": False, "done": 0, "total": 0, "pass_idx": 0, "n_pass": 1, "_last": 0}


def begin(n_pass: int) -> None:
    """Mark a video encode as starting and reset counters (call before _gpu_infer)."""
    _STATE.update(active=True, done=0, total=0, pass_idx=0, n_pass=max(1, int(n_pass)), _last=0)


def end() -> None:
    """Mark the encode finished (call in a ``finally``)."""
    _STATE["active"] = False


def sink(done: int, total: int) -> None:
    """Per-clip sink registered on ``fast_encode``. Detects a new window pass by
    the per-window counter restart (``done`` returning to <= the last seen), so
    the bar advances monotonically across windows with no reset. No throttle here
    — the 0.3 s Timer is the throttle (it just samples whatever the last write was)."""
    if done <= _STATE["_last"]:
        _STATE["pass_idx"] += 1
    _STATE["_last"] = int(done)
    _STATE["done"] = int(done)
    _STATE["total"] = int(total)


def snapshot() -> dict:
    """Return a shallow copy of the live state (for the Timer handler)."""
    return dict(_STATE)


def encode_frac(snap: dict | None = None) -> float:
    """0..1 fraction of the whole multi-window encode from a snapshot.

    ``frac = (pass_idx + done/total) / n_pass`` — monotonic across windows,
    clamped to [0, 1]. Degrades to a single 0→1 sweep when ``n_pass == 1``.
    """
    s = snap if snap is not None else _STATE
    n = max(1, int(s.get("n_pass", 1) or 1))
    total = int(s.get("total", 0) or 0)
    within = (int(s.get("done", 0) or 0) / total) if total else 0.0
    return min(1.0, max(0.0, (int(s.get("pass_idx", 0) or 0) + within) / n))
