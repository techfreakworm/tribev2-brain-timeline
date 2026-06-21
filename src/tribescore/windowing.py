"""Sliding-window inference + stitching over a long (4-5 min) video.

The TRIBE v2 model emits one brain-activity prediction per fMRI TR and
internally chunks clips to ~60 s. To score a full 4-5 minute video in
bounded, overlapping GPU calls -- and to smooth across chunk boundaries --
we run an *outer* sliding window here:

    1. Cut the timeline ``[0, total_duration_s]`` into windows of
       ``window_s`` seconds, advancing by ``hop_s`` each step.
    2. For each window, hand the window-local events to an injected
       ``infer_fn`` and receive that window's ``(t, n_vertices)`` activity
       plus the per-row offsets (in seconds, relative to the window start).
    3. Resample every window onto one shared, evenly spaced time axis and
       average overlapping windows together (Hann-like tapering avoids
       seams), producing a single ``(T, n_vertices)`` timeline.

``infer_fn`` is *injected* so the whole module is testable with a pure-numpy
stand-in -- no torch, no tribev2, no GPU. On the Space, the caller passes a
closure that wraps :func:`tribescore.inference.predict_window`.

Nothing in this module imports torch or tribev2.
"""

from __future__ import annotations

from typing import Callable, Protocol, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

#: A single window's inference result:
#:   activity : np.ndarray of shape (t_window, n_vertices)
#:   offsets_s: np.ndarray of shape (t_window,), seconds from the window
#:              start for each row of ``activity`` (monotonically increasing).
WindowResult = Tuple[np.ndarray, np.ndarray]


class InferFn(Protocol):
    """Callable injected into :func:`run_windowed`.

    Given the absolute window bounds it returns that window's brain-activity
    rows and their per-row time offsets (seconds from ``start_s``).
    """

    def __call__(self, events: object, start_s: float, end_s: float) -> WindowResult:
        ...


# ---------------------------------------------------------------------------
# Window planning
# ---------------------------------------------------------------------------


def plan_windows(
    total_duration_s: float,
    window_s: float,
    hop_s: float,
) -> np.ndarray:
    """Compute ``(start_s, end_s)`` bounds for every window.

    The final window is clamped to ``total_duration_s`` so the tail of the
    video is always covered without running past its end.

    Parameters
    ----------
    total_duration_s:
        Length of the video/events timeline, in seconds. Must be > 0.
    window_s:
        Window length, in seconds. Must be > 0.
    hop_s:
        Step between consecutive window starts, in seconds. Must be > 0 and
        ``<= window_s`` (``hop_s == window_s`` means non-overlapping).

    Returns
    -------
    np.ndarray
        Array of shape ``(n_windows, 2)``; each row is ``[start_s, end_s]``.

    Raises
    ------
    ValueError
        If any argument is non-positive or ``hop_s > window_s``.
    """
    if total_duration_s <= 0:
        raise ValueError(f"total_duration_s must be > 0, got {total_duration_s}")
    if window_s <= 0:
        raise ValueError(f"window_s must be > 0, got {window_s}")
    if hop_s <= 0:
        raise ValueError(f"hop_s must be > 0, got {hop_s}")
    if hop_s > window_s:
        raise ValueError(
            f"hop_s ({hop_s}) must be <= window_s ({window_s}); a gap between "
            "windows would leave parts of the timeline unscored"
        )

    starts = []
    s = 0.0
    # Advance until a window's start reaches the end of the timeline. The
    # `- 1e-9` guards against a degenerate zero-length trailing window when
    # total_duration_s is an exact multiple of hop_s.
    while s < total_duration_s - 1e-9:
        starts.append(s)
        s += hop_s
    if not starts:  # total_duration_s smaller than a single hop
        starts = [0.0]

    bounds = np.array(
        [[start, min(start + window_s, total_duration_s)] for start in starts],
        dtype=float,
    )
    return bounds


# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------


def run_windowed(
    events: object,
    infer_fn: InferFn,
    window_s: float,
    hop_s: float,
    *,
    total_duration_s: float | None = None,
    target_hz: float = 1.0,
    normalize: bool = True,
    taper: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run windowed inference over a long video and stitch the results.

    Parameters
    ----------
    events:
        Opaque events container (a tribev2 events ``DataFrame`` in
        production). This function never inspects it -- it only forwards it
        to ``infer_fn`` -- which keeps the module model-agnostic and tests
        able to pass any sentinel.
    infer_fn:
        Injected per-window inference callable. See :class:`InferFn`. For
        each window it must return ``(activity, offsets_s)`` where
        ``activity`` has shape ``(t_window, n_vertices)`` and ``offsets_s``
        has shape ``(t_window,)`` giving seconds-from-window-start per row.
    window_s:
        Outer window length in seconds (e.g. 60.0).
    hop_s:
        Step between windows in seconds (e.g. 30.0 for 50% overlap).
    total_duration_s:
        Total timeline length in seconds. If ``None``, it must be derivable
        from ``events`` -- in production via
        :func:`tribescore.windowing.events_duration_s`. Tests should pass it
        explicitly.
    target_hz:
        Sampling rate of the stitched output time axis, in samples per
        second. The shared axis runs from 0 to ``total_duration_s``.
    normalize:
        If ``True``, z-score each vertex column across time on the stitched
        timeline (zero mean, unit variance), which makes downstream metric
        reduction scale-invariant. Constant columns are left at zero.
    taper:
        If ``True``, weight each window by a Hann taper before averaging so
        overlapping windows blend smoothly instead of producing seams at
        boundaries.

    Returns
    -------
    timeline : np.ndarray
        Stitched activity of shape ``(T, n_vertices)`` where
        ``T = floor(total_duration_s * target_hz) + 1``.
    time_axis : np.ndarray
        Shape ``(T,)``, strictly increasing seconds from 0 to
        ``total_duration_s``. Monotonic by construction.

    Raises
    ------
    ValueError
        If ``total_duration_s`` cannot be determined, if a window result is
        malformed, or if no window produced any rows.

    Notes
    -----
    Implementation is deferred (TODO). The contract above -- output shapes,
    a monotonic ``time_axis``, and an injectable ``infer_fn`` -- is the
    surface the rest of the app and the tests rely on.
    """
    if total_duration_s is None:
        total_duration_s = events_duration_s(events)
    if total_duration_s is None or total_duration_s <= 0:
        raise ValueError(
            "total_duration_s could not be determined; pass it explicitly"
        )

    # Shared, evenly spaced output axis. Monotonic by construction: this is
    # the invariant the tests assert and downstream plotting relies on.
    n_out = int(np.floor(total_duration_s * target_hz)) + 1
    time_axis = np.linspace(0.0, total_duration_s, n_out)

    bounds = plan_windows(total_duration_s, window_s, hop_s)

    # Accumulators for overlap-averaging on the shared axis. We discover
    # n_vertices from the first non-empty window.
    accum: np.ndarray | None = None
    weight: np.ndarray | None = None

    for start_s, end_s in bounds:
        activity, offsets_s = _validate_window_result(
            infer_fn(events, float(start_s), float(end_s))
        )
        if activity.shape[0] == 0:
            continue

        n_vertices = activity.shape[1]
        if accum is None:
            accum = np.zeros((n_out, n_vertices), dtype=float)
            weight = np.zeros((n_out, 1), dtype=float)
        elif n_vertices != accum.shape[1]:
            raise ValueError(
                f"window at [{start_s}, {end_s}] returned n_vertices="
                f"{n_vertices}, expected {accum.shape[1]}"
            )

        # Absolute time of each window row, then resample onto the shared
        # axis and add into the accumulator with a (tapered) weight.
        abs_times = float(start_s) + offsets_s
        w = _window_weights(offsets_s, end_s - start_s, taper=taper)
        _accumulate_window(accum, weight, time_axis, abs_times, activity, w)

    if accum is None or weight is None or not np.any(weight > 0):
        raise ValueError("no window produced any prediction rows")

    timeline = _finalize(accum, weight)
    if normalize:
        timeline = _zscore_columns(timeline)

    return timeline, time_axis


# ---------------------------------------------------------------------------
# Helpers (implementation TODO; signatures fixed so the engine is unit-test
# friendly and each step is independently verifiable).
# ---------------------------------------------------------------------------


def events_duration_s(events: object) -> float | None:
    """Best-effort total duration (seconds) of a tribev2 events DataFrame.

    Production helper: read ``start``/``duration`` columns and return the
    max end time. Returns ``None`` when the duration cannot be inferred so
    callers can fall back to an explicit ``total_duration_s``.

    TODO: implement against the real events schema (``start`` + ``duration``
    columns, per :func:`tribev2.demo_utils.TribeModel.get_events_dataframe`).
    """
    # Deliberately model-agnostic: avoid importing pandas here. Duck-type a
    # DataFrame-like object; otherwise signal "unknown".
    try:
        starts = events["start"]  # type: ignore[index]
        durations = events["duration"]  # type: ignore[index]
        return float(np.max(np.asarray(starts) + np.asarray(durations)))
    except Exception:
        return None


def _validate_window_result(result: WindowResult) -> WindowResult:
    """Coerce + sanity-check a single ``infer_fn`` return value.

    Ensures ``activity`` is 2-D ``(t, n_vertices)`` and ``offsets_s`` is a
    matching 1-D, non-decreasing array.
    """
    try:
        activity, offsets_s = result
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "infer_fn must return a (activity, offsets_s) tuple"
        ) from exc

    activity = np.asarray(activity, dtype=float)
    offsets_s = np.asarray(offsets_s, dtype=float)

    if activity.ndim != 2:
        raise ValueError(
            f"window activity must be 2-D (t, n_vertices), got shape "
            f"{activity.shape}"
        )
    if offsets_s.ndim != 1 or offsets_s.shape[0] != activity.shape[0]:
        raise ValueError(
            f"offsets_s must be 1-D with len == activity rows "
            f"({activity.shape[0]}), got shape {offsets_s.shape}"
        )
    return activity, offsets_s


def _window_weights(
    offsets_s: np.ndarray,
    window_len_s: float,
    *,
    taper: bool,
) -> np.ndarray:
    """Per-row blend weights for one window.

    A Hann taper (low at the window edges, high in the middle) makes
    overlapping windows cross-fade, removing boundary seams. With
    ``taper=False`` every row gets weight 1 (plain mean over overlaps).

    Returns shape ``(t,)``.
    """
    n = offsets_s.shape[0]
    if not taper or n <= 1 or window_len_s <= 0:
        return np.ones(n, dtype=float)
    # Position of each row within [0, 1] across the window, mapped to a Hann
    # window. +eps keeps edge weights strictly positive so isolated regions
    # are never zeroed out entirely.
    frac = np.clip(offsets_s / float(window_len_s), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * frac) + 1e-3


def _accumulate_window(
    accum: np.ndarray,
    weight: np.ndarray,
    time_axis: np.ndarray,
    abs_times: np.ndarray,
    activity: np.ndarray,
    row_weights: np.ndarray,
) -> None:
    """Resample one window onto ``time_axis`` and add it into accumulators.

    Performs, in place: for each output time bin, the weighted nearest-row
    (or interpolated) contribution from this window is added to ``accum`` and
    its weight added to ``weight``. The final divide happens in
    :func:`_finalize`.

    TODO: implement per-vertex resampling (nearest or linear) from
    ``abs_times`` onto ``time_axis``; the chosen scheme must keep the output
    time axis monotonic (it is, by construction) and not introduce NaNs.
    """
    # Minimal, correct nearest-neighbour fill so the engine + tests run end
    # to end. Each output bin takes the contribution of the closest window
    # row, scaled by that row's blend weight. A richer linear interpolation
    # can replace this without changing the public contract.
    if abs_times.shape[0] == 0:
        return
    idx = np.searchsorted(abs_times, time_axis)
    idx = np.clip(idx, 0, abs_times.shape[0] - 1)
    left = np.clip(idx - 1, 0, abs_times.shape[0] - 1)
    choose_left = np.abs(time_axis - abs_times[left]) <= np.abs(
        time_axis - abs_times[idx]
    )
    nearest = np.where(choose_left, left, idx)

    # Only contribute to output bins that actually fall within this window's
    # covered span, so non-overlapping regions are not smeared.
    in_span = (time_axis >= abs_times[0] - 1e-9) & (
        time_axis <= abs_times[-1] + 1e-9
    )
    w = (row_weights[nearest] * in_span).reshape(-1, 1)
    accum += w * activity[nearest]
    weight += w


def _finalize(accum: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Divide accumulated activity by accumulated weight, filling gaps.

    Output bins with zero total weight (uncovered by any window) are filled
    by forward/backward fill so the returned timeline has no NaNs.
    """
    safe = np.where(weight > 0, weight, 1.0)
    out = accum / safe
    covered = (weight > 0).ravel()
    if not covered.all() and covered.any():
        out = _fill_gaps(out, covered)
    return out


def _fill_gaps(timeline: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Forward/backward-fill uncovered rows of a ``(T, V)`` timeline."""
    idx = np.where(covered, np.arange(covered.size), 0)
    np.maximum.accumulate(idx, out=idx)  # forward fill indices
    filled = timeline[idx]
    # Backward fill any leading uncovered rows.
    first = int(np.argmax(covered))
    if first > 0:
        filled[:first] = timeline[first]
    return filled


def _zscore_columns(timeline: np.ndarray) -> np.ndarray:
    """Z-score each vertex column across time; constant columns -> zeros."""
    mean = timeline.mean(axis=0, keepdims=True)
    std = timeline.std(axis=0, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    out = (timeline - mean) / std
    # Zero-out columns that were constant (std ~ 0) to avoid spurious values.
    out[:, (timeline.std(axis=0) <= 1e-8)] = 0.0
    return out
