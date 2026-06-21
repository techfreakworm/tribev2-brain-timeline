"""Windowing + stitching for a long (4-5 min) clip's brain-activity timeline.

TRIBE v2 emits one prediction per fMRI TR (TR = 1.0 s ⇒ 1 Hz). To score a full
4-5 minute clip we configure **overlap into tribev2's own segmenter** (a single
``predict()`` per Run; ``data.overlap_trs_train = 20`` ⇒ overlapping 100 s
windows at an 80 s stride) and then stitch the per-TR rows back into one
continuous, seam-smooth ``(T, K)`` timeline here. This module is the pure-numpy,
fully unit-testable core of that pipeline (``docs/PLAN.md`` §4):

  * :func:`plan_windows` -- the window spans, for the progress UI and the
    per-window ``ffmpeg`` *fallback*. Pure function, no model.
  * :func:`stitch` -- the real work: detect each window from ``abs_times``,
    per-window per-vertex z-score, trapezoidal-crossfade overlap-add onto an
    integer-second grid, leading warm-up suppression, and zero-weight-gap
    interpolation. Pure numpy.

The heavy ``predict()`` is injected as ``(preds, abs_times)`` (the seam tests
exercise with synthetic ramps/sines), so nothing here imports torch or tribev2.

``stitch`` is generic over the row width ``K``: ``K = 20484`` (raw fsaverage5
vertices) in production, or ``K = n_metrics`` if metric reduction is applied
before stitching. The algorithm never assumes a particular ``K``.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Locked parameters (``docs/PLAN.md`` §4). Window length is *forced* to 100 s by
# the checkpoint pooler (§3); hop = 80 s ⇒ 20 s / 20 % overlap (HRF-length
# left-context). warmup_trim = 5 TRs (leading-only).
# ---------------------------------------------------------------------------

WIN_S: int = 100
HOP_S: int = 80
WARMUP_TRIM: int = 5

#: Small epsilon for the per-window per-vertex z-score denominator (§4 step 3b).
_ZSCORE_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Window planning
# ---------------------------------------------------------------------------


def plan_windows(
    duration_s: float,
    win_s: int = WIN_S,
    hop_s: int = HOP_S,
) -> List[Tuple[float, float]]:
    """Plan the overlapping window spans covering ``[0, duration_s]``.

    Implements ``docs/PLAN.md`` §4 step 1. Returns ascending ``(start, end)``
    spans advancing by ``hop_s`` (e.g. ``[(0,100),(80,180),…]``). The **last
    span is forced to** ``(max(0.0, duration_s - 100), duration_s)`` so the
    tail of the clip is always covered by a full-length window, and spans whose
    ``start`` repeats are de-duplicated (keeping the first occurrence, ascending
    order preserved).

    This is used only for the "Window k/N" progress label and the per-window
    ``ffmpeg`` *fallback* (§4 Fallback) -- the real path lets tribev2's own
    segmenter tile the clip. Pure function; touches no model.

    Parameters
    ----------
    duration_s:
        Clip length in seconds. Must be > 0.
    win_s:
        Window length in seconds (fixed at 100 by the pooler lock, §3).
    hop_s:
        Stride between window starts in seconds (80 ⇒ 20 s overlap).

    Returns
    -------
    list[tuple[float, float]]
        Ascending, de-duplicated ``(start, end)`` spans. The final span ends
        exactly at ``duration_s``.

    Raises
    ------
    ValueError
        If ``duration_s <= 0``, ``win_s <= 0``, ``hop_s <= 0``, or
        ``hop_s > win_s`` (a gap would leave the timeline partially uncovered).
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    if win_s <= 0:
        raise ValueError(f"win_s must be > 0, got {win_s}")
    if hop_s <= 0:
        raise ValueError(f"hop_s must be > 0, got {hop_s}")
    if hop_s > win_s:
        raise ValueError(
            f"hop_s ({hop_s}) must be <= win_s ({win_s}); a gap between "
            "windows would leave parts of the timeline unscored"
        )

    win_s_f = float(win_s)
    hop_s_f = float(hop_s)

    # Regular grid of starts, advancing by hop until the next window would start
    # at/after the end of the clip.
    starts: List[float] = []
    s = 0.0
    while s < duration_s - 1e-9:
        starts.append(s)
        s += hop_s_f
    if not starts:  # clip shorter than a single hop
        starts.append(0.0)

    # Force the LAST span to a full-length window ending exactly at the clip end
    # (§4 step 1): overwrite the final grid start -- which would otherwise yield
    # a short tail window clamped to ``duration_s`` -- with ``duration_s -
    # win_s``. Dedup then collapses it against an earlier grid start when they
    # coincide (e.g. ``duration_s`` an exact multiple of ``hop_s``).
    starts[-1] = max(0.0, duration_s - win_s_f)

    # De-dup repeated starts, keep ascending order + first occurrence. Round the
    # key so float wobble (e.g. two starts ~equal) collapses cleanly.
    spans: List[Tuple[float, float]] = []
    seen: set[float] = set()
    for start in sorted(starts):
        key = round(start, 6)
        if key in seen:
            continue
        seen.add(key)
        end = min(start + win_s_f, duration_s)
        spans.append((start, end))
    return spans


# ---------------------------------------------------------------------------
# Stitching (the real work)
# ---------------------------------------------------------------------------


def stitch(
    preds: np.ndarray,
    abs_times: np.ndarray,
    *,
    warmup_trim: int = WARMUP_TRIM,
    hop_s: int = HOP_S,
    win_s: int = WIN_S,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stitch overlapping per-TR windows into one continuous 1 Hz timeline.

    Implements the 6-step algorithm in ``docs/PLAN.md`` §4 step 3, exactly:

    a. **Segment into windows** by detecting ``abs_times`` resets: a new window
       begins wherever ``abs_times`` does not strictly increase (the backward
       jump at a seam, e.g. ``…, 98, 99, 80, 81, …``). Each window gets an index
       ``w`` and each row an intra-window position ``p`` (0-based).
    b. **Per-window per-vertex z-score** over that window's rows:
       ``(x - mean_t) / (std_t + 1e-6)``.
    c. **Trapezoidal crossfade weights** (overlap ``= win_s - hop_s``): the
       leading edge ramps ``0→1`` over the first ``overlap`` rows, the trailing
       edge ramps ``1→0`` over the last ``overlap`` rows, ``=1`` between -- so
       overlapping windows blend with ``Σ weight ≈ 1`` (no low-SNR seam band).
       Then **warm-up suppression**: for ``w > 0``, weight ``= 0`` for
       ``p < warmup_trim`` (the leading ramp effectively starts at
       ``p = warmup_trim``). Window 0 keeps its leading rows (nothing earlier
       covers them); the final window keeps its trailing rows at weight 1 (its
       tail is the unique end-of-clip signal -- not ramped down).
    d. **Overlap-add** onto the integer-second grid
       ``t_axis = arange(0, ceil(max(abs_times)) + 1)``:
       ``timeline[t] = Σ_w weight·zrow / Σ_w weight`` over all rows whose
       ``round(abs_time) == t`` (the weighted mean *is* the crossfade).
    e. **Zero-weight grid seconds** (all-suppressed gaps -- should not occur
       with 20 s overlap > 5 TR trim) are linearly interpolated from neighbours.
    f. Returns ``(timeline (T, K), t_axis (T,))``.

    Parameters
    ----------
    preds:
        Per-TR rows, shape ``(R, K)``. ``K = 20484`` raw vertices in production
        or ``K = n_metrics`` -- the algorithm is generic over ``K``.
    abs_times:
        Shape ``(R,)`` -- absolute second on the input clock for each row
        (``round(segment.start)``; §3). Carries the per-window ascent +
        backward-jump-at-seam structure step (a) keys on.
    warmup_trim:
        Number of leading TRs to suppress in every window except window 0.
    hop_s, win_s:
        Stride / window length in seconds; ``overlap = win_s - hop_s``.

    Returns
    -------
    timeline : np.ndarray
        Shape ``(T, K)`` -- the stitched, seam-smooth 1 Hz timeline.
    t_axis : np.ndarray
        Shape ``(T,)``, dtype float -- the integer-second grid
        ``0, 1, …, ceil(max(abs_times))``. Strictly increasing by construction.

    Raises
    ------
    ValueError
        If ``preds``/``abs_times`` are malformed (wrong ndim, length mismatch,
        or empty), or if -- after stitching -- any grid second still has zero
        weight *and* cannot be interpolated (no covered neighbours).
    """
    preds = np.asarray(preds, dtype=float)
    abs_times = np.asarray(abs_times, dtype=float)

    if preds.ndim != 2:
        raise ValueError(
            f"preds must be 2-D (R, K), got shape {preds.shape}"
        )
    if abs_times.ndim != 1 or abs_times.shape[0] != preds.shape[0]:
        raise ValueError(
            f"abs_times must be 1-D with len == preds rows ({preds.shape[0]}), "
            f"got shape {abs_times.shape}"
        )
    if preds.shape[0] == 0:
        raise ValueError("stitch received zero prediction rows")

    overlap = int(win_s) - int(hop_s)
    n_rows, n_k = preds.shape

    # --- (a) segment into windows by detecting non-increasing abs_times ------
    # window_starts_idx[i] = row index where window i begins.
    window_bounds = _segment_windows(abs_times)
    n_windows = len(window_bounds)

    # --- (d) integer-second output grid --------------------------------------
    t_max = int(np.ceil(float(np.max(abs_times))))
    t_axis = np.arange(0, t_max + 1, dtype=float)
    n_out = t_axis.shape[0]

    accum = np.zeros((n_out, n_k), dtype=float)
    weight_sum = np.zeros(n_out, dtype=float)

    for w, (lo, hi) in enumerate(window_bounds):
        rows = slice(lo, hi)
        win_preds = preds[rows]              # (n_w, K)
        win_times = abs_times[rows]          # (n_w,)
        n_w = win_preds.shape[0]
        if n_w == 0:
            continue

        # (b) per-window per-vertex z-score over time.
        zrows = _zscore_over_time(win_preds)

        # (c) trapezoidal crossfade weights + warm-up suppression.
        is_first = w == 0
        is_last = w == n_windows - 1
        wts = _trapezoid_weights(
            n_w,
            overlap=overlap,
            warmup_trim=warmup_trim,
            suppress_warmup=not is_first,
            ramp_leading=not is_first,
            ramp_trailing=not is_last,
        )

        # (d) overlap-add onto the integer-second grid.
        bins = np.rint(win_times).astype(int)
        valid = (bins >= 0) & (bins < n_out) & (wts > 0)
        if not np.any(valid):
            continue
        b = bins[valid]
        wv = wts[valid]
        # Scatter-add weighted rows and weights into the grid.
        np.add.at(accum, b, zrows[valid] * wv[:, None])
        np.add.at(weight_sum, b, wv)

    # (d) normalise the weighted sum → weighted mean (the crossfade). Bins with
    # zero weight divide by a safe 1.0 here and are fixed up in step (e).
    covered = weight_sum > 0
    safe = np.where(covered, weight_sum, 1.0)
    timeline = accum / safe[:, None]

    # (e) linearly interpolate any zero-weight grid seconds from neighbours.
    if not covered.all():
        timeline = _interp_uncovered(timeline, covered, t_axis)

    return timeline, t_axis


# ---------------------------------------------------------------------------
# Stitch helpers (pure numpy; each independently testable)
# ---------------------------------------------------------------------------


def _segment_windows(abs_times: np.ndarray) -> List[Tuple[int, int]]:
    """Split row indices into per-window ``[lo, hi)`` runs (``§4`` step a).

    A new window begins at row ``i`` whenever ``abs_times[i] <= abs_times[i-1]``
    -- the backward jump (or repeat) that marks a tribev2 window boundary after
    an ascending run. Returns a list of half-open ``(lo, hi)`` index ranges, in
    order.
    """
    n = abs_times.shape[0]
    if n == 0:
        return []
    # Boundaries where the sequence does not strictly increase.
    breaks = np.nonzero(abs_times[1:] <= abs_times[:-1])[0] + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [n]))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _zscore_over_time(win_preds: np.ndarray) -> np.ndarray:
    """Per-vertex z-score over time for one window (``§4`` step b).

    ``(x - mean_t) / (std_t + 1e-6)`` along the time (row) axis. The ``+eps``
    keeps a constant vertex finite (it maps to ~0).
    """
    mean_t = win_preds.mean(axis=0, keepdims=True)
    std_t = win_preds.std(axis=0, keepdims=True)
    return (win_preds - mean_t) / (std_t + _ZSCORE_EPS)


def _trapezoid_weights(
    n_w: int,
    *,
    overlap: int,
    warmup_trim: int,
    suppress_warmup: bool,
    ramp_leading: bool,
    ramp_trailing: bool,
) -> np.ndarray:
    """Trapezoidal crossfade weights for one window (``§4`` step c).

    Flat ``= 1`` core with linear ramps over the ``overlap`` edge rows:

      * ``ramp_leading``  -- leading edge ramps ``0→1`` over the first
        ``overlap`` rows (else flat 1 from the start -- window 0).
      * ``ramp_trailing`` -- trailing edge ramps ``1→0`` over the last
        ``overlap`` rows (else flat 1 to the end -- the final window).
      * ``suppress_warmup`` -- force weight ``= 0`` for ``p < warmup_trim``
        (every window but the first).

    The leading/trailing ramps are reflections of one another, so at a seam the
    outgoing window's trailing ramp and the incoming window's leading ramp sum
    to ≈ 1 across the overlap (a clean linear crossfade after the weighted-mean
    normalisation in :func:`stitch`).

    Returns shape ``(n_w,)``.
    """
    wts = np.ones(n_w, dtype=float)
    ov = max(0, int(overlap))

    if ov > 0:
        # Linear ramp values for the overlap region, strictly within (0, 1) and
        # symmetric: position j in [0, ov-1] → (j + 1) / (ov + 1). The mirror of
        # this ramp (used on the trailing edge) is its reverse, so a leading
        # ramp r and a trailing ramp (1 - r at the aligned row) crossfade.
        ramp = (np.arange(ov, dtype=float) + 1.0) / (ov + 1.0)

        if ramp_leading:
            k = min(ov, n_w)
            wts[:k] = ramp[:k]
        if ramp_trailing:
            k = min(ov, n_w)
            # Trailing rows ramp 1→0: reverse of the leading ramp.
            wts[n_w - k:] = np.minimum(wts[n_w - k:], ramp[::-1][ov - k:])

    if suppress_warmup and warmup_trim > 0:
        k = min(int(warmup_trim), n_w)
        wts[:k] = 0.0

    return wts


def _interp_uncovered(
    timeline: np.ndarray,
    covered: np.ndarray,
    t_axis: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate zero-weight grid rows from covered neighbours.

    Implements ``§4`` step e. Per vertex column, uncovered seconds are filled by
    ``np.interp`` over the covered seconds. Should be a no-op with 20 s overlap
    (the tests assert the uncovered set is empty); included for robustness.

    Raises
    ------
    ValueError
        If there are no covered rows at all to interpolate from.
    """
    if not covered.any():
        raise ValueError(
            "stitch produced an all-zero-weight timeline (no rows survived "
            "warm-up suppression); check abs_times / window structure"
        )
    out = timeline.copy()
    xp = t_axis[covered]
    uncovered = ~covered
    xq = t_axis[uncovered]
    for j in range(out.shape[1]):
        out[uncovered, j] = np.interp(xq, xp, timeline[covered, j])
    return out


# ---------------------------------------------------------------------------
# Deprecated compatibility shim
# ---------------------------------------------------------------------------


def run_windowed(
    events: object,
    infer_fn,
    window_s: float = WIN_S,
    hop_s: float = HOP_S,
    *,
    total_duration_s: float | None = None,
    **_legacy_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deprecated thin adapter kept so importers do not break.

    .. deprecated::
        The locked design (``docs/PLAN.md`` §4) is :func:`plan_windows` +
        :func:`stitch`. The real path runs **one** ``predict()`` over the whole
        clip (tribev2's own segmenter does the windowing) and feeds the
        resulting ``(preds, abs_times)`` straight into :func:`stitch`; there is
        no per-window outer loop. This wrapper only exists for any caller still
        importing ``run_windowed`` -- prefer ``stitch`` directly.

    Drives the injected per-window ``infer_fn`` (legacy contract:
    ``infer_fn(events, start_s, end_s) -> (activity (t,K), offsets_s (t,))``)
    over :func:`plan_windows` spans, assembles the concatenated
    ``(preds, abs_times)`` those windows imply (``abs_time = start + offset``),
    and returns ``stitch``'s ``(timeline, t_axis)``.

    Parameters
    ----------
    events:
        Opaque container forwarded untouched to ``infer_fn``.
    infer_fn:
        Per-window callable with the legacy signature above.
    window_s, hop_s:
        Window length / stride (default to the locked 100 / 80).
    total_duration_s:
        Clip length; required (the legacy ``events``-introspection path is
        gone). Raises ``ValueError`` if missing.

    Returns
    -------
    (timeline, t_axis):
        As :func:`stitch`.
    """
    if total_duration_s is None or total_duration_s <= 0:
        raise ValueError(
            "run_windowed is deprecated; pass total_duration_s explicitly (or "
            "migrate to stitch())"
        )

    spans = plan_windows(total_duration_s, win_s=int(window_s), hop_s=int(hop_s))
    preds_blocks: List[np.ndarray] = []
    times_blocks: List[np.ndarray] = []
    for start_s, end_s in spans:
        activity, offsets_s = infer_fn(events, float(start_s), float(end_s))
        activity = np.asarray(activity, dtype=float)
        offsets_s = np.asarray(offsets_s, dtype=float)
        if activity.ndim != 2:
            raise ValueError(
                f"infer_fn activity must be 2-D (t, K), got {activity.shape}"
            )
        if activity.shape[0] == 0:
            continue
        preds_blocks.append(activity)
        times_blocks.append(float(start_s) + offsets_s)

    if not preds_blocks:
        raise ValueError("no window produced any prediction rows")

    preds = np.concatenate(preds_blocks, axis=0)
    abs_times = np.concatenate(times_blocks, axis=0)
    return stitch(preds, abs_times, hop_s=int(hop_s), win_s=int(window_s))
