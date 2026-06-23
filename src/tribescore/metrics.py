"""Reduce a brain-activity timeline to named "brain-metric" curves.

TRIBE v2 predicts fMRI-like activity over the cortical surface: a tensor of
shape ``(T, 20484)`` where ``20484 = 2 x 10242`` ``fsaverage5`` vertices
(left hemisphere ``[0:10242]`` then right ``[10242:20484]``). This module
collapses that high-dimensional signal into a handful of interpretable curves
over time -- our derived "brain metrics".

This is the implementation of task **T-C** in ``docs/PLAN.md`` (see §5 for the
atlas, the exact Glasser parcel lists, the normalization + smoothing recipe,
and the honesty caveats; see §9 T-C for the exact signatures).

Atlas
-----
We use **HCP-MMP1 (Glasser 2016)** via tribev2's own shipped helper
(:mod:`tribev2.utils`), *not* nilearn Yeo/Schaefer. ``get_hcp_roi_indices``
returns vertex indices **already in the 20484 output index space** (the right
hemisphere ``+10242`` offset is applied for us), strips the ``L_``/``R_``
prefix and ``_ROI`` suffix (so keys are bare Glasser names like ``FEF``,
``LIPv``, ``TPOJ1``, ``p32``, ``10r``), pools both hemispheres under one key,
and supports ``*`` wildcards. This guarantees index agreement with ``preds``
and removes the entire nilearn atlas-fetch + manual vertex-alignment surface.

Honesty caveat (§5): "Virality" is a **research proxy** from cortical value-
region activity, not a guarantee of going viral. ``facebook/tribev2`` is
cortical-only (no ventral striatum / NAcc -- the strongest neuroforecasting
node), so the vmPFC/mPFC signal is the validated *complement*, not a
substitute. Because the training target was per-sample z-scored + detrended,
**only relative temporal dynamics are interpretable -- absolute "scores" are
meaningless.** Summaries therefore report *relative* peaks (z-units /
percentile).

Local-import safety
-------------------
Importing this module never touches ``tribev2``/``torch``/``mne``: the only
function that needs them is :func:`build_roi_masks`, whose import of
``tribev2.utils`` is guarded and deferred. ``to_metrics`` / :func:`summary`
are pure numpy + scipy and fully unit-testable locally. The legacy
synthetic-atlas API (:class:`Atlas`, :func:`reduce_to_metrics`, ...) is kept
for backward compatibility with the package re-exports and existing callers.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# T-C : metric -> Glasser (HCP-MMP1) parcels  (docs/PLAN.md §5)
# ---------------------------------------------------------------------------
#
# Bare ``combine=False`` Glasser names (no ``L_``/``R_`` prefix, no ``_ROI``
# suffix) -- exactly as returned by ``tribev2.utils.get_hcp_labels(...,
# combine=False)`` and consumed by ``get_hcp_roi_indices``. The metric value
# per TR is the mean activity over the *union* of these parcels' vertices
# across both hemispheres.
#
# These lists are transcribed verbatim from the §5 tables. Do NOT silently
# "fix" a name here -- ``build_roi_masks`` validates every key against the live
# atlas and logs + drops any that the installed ``mne`` version does not know,
# so naming drift surfaces loudly rather than corrupting the masks.

PARCELS: Dict[str, list[str]] = {
    # --- default ON ---------------------------------------------------------
    # Dorsal attention (FEF + IPS complex) + Ventral attention
    # (TPOJ / PGi / PGs / PFm + IFJ) + Frontoparietal control (DLPFC).
    "Attention": [
        "FEF", "LIPv", "LIPd", "VIP", "MIP", "AIP", "IP0", "IP1", "IP2",
        "TPOJ1", "TPOJ2", "PGi", "PGs", "PFm",
        "IFJa", "IFJp",
        "p9-46v", "a9-46v", "9-46d", "46", "8C", "i6-8", "s6-8",
    ],
    # Sensory drive (early visual + motion + early/assoc. auditory) +
    # associative integration (STS).
    "Engagement": [
        "V1", "V2", "V3", "V4", "V3A", "V3B", "V6", "V6A", "MT", "MST",
        "A1", "LBelt", "MBelt", "PBelt", "A4", "A5",
        "STSdp", "STSvp", "STSda", "STSva", "TE1p", "TE2p",
    ],
    # vmPFC / mOFC / pgACC / mPFC cortical *value* signal (virality proxy).
    "Virality": [
        "10r", "10v", "10d", "10pp", "p32", "s32", "a24", "d32", "25",
        "OFC", "pOFC", "11l", "13l", "9m",
    ],
    # --- default OFF (toggleable) -------------------------------------------
    # Core language network (text + audio driven).
    "Language": [
        "44", "45", "IFSa", "STSdp", "STSvp", "STGa", "TE1a", "A5",
        "PSL", "SFL", "55b",
    ],
    # Default-mode / self & social relevance -- a secondary sharing cue.
    "Self-relevance": [
        "7m", "POS2", "v23ab", "d23ab", "31pv", "31pd", "RSC", "PCV",
        "9m", "10r", "PGs", "PGi",
    ],
}

#: Metrics shown ON by default in the UI (§5 / §11.1). The remaining keys of
#: :data:`PARCELS` ("Language", "Self-relevance") are toggleable, default OFF.
METRIC_DEFAULT_ON: frozenset[str] = frozenset({"Attention", "Engagement", "Virality"})

#: Filename of the persisted ROI-mask cache (§5 / §7). Keyed by atlas + mesh so
#: a future mesh change cannot silently load stale indices.
ROI_MASKS_FILENAME = "roi_masks_hcpmmp1_fsaverage5.npz"

#: z-score floor (§5): ``(x - mean) / (std + EPS)`` -- guards against a flat
#: (zero-variance) curve producing NaNs/inf.
_EPS = 1e-6

#: Gaussian temporal smoothing width in seconds (§5). TR = 1 s so sigma is in
#: samples too. ``truncate=3`` -> ~13-tap kernel; matches BOLD sluggishness
#: while preserving peaks for click-to-seek (§6).
_SMOOTH_SIGMA_S = 2.0
_SMOOTH_TRUNCATE = 3.0


# ---------------------------------------------------------------------------
# T-C : build_roi_masks  (docs/PLAN.md §5, §9 T-C)
# ---------------------------------------------------------------------------


def build_roi_masks(cache_dir: str, mesh: str = "fsaverage5") -> Dict[str, np.ndarray]:
    """Build (or load) per-metric vertex-index masks over the cortical mesh.

    For every metric in :data:`PARCELS`, gather the vertex indices of its
    Glasser parcels in the **20484 output index space** and de-duplicate them::

        valid = set(get_hcp_labels(mesh=mesh, combine=False, hemi="both"))
        mask[metric] = np.unique(np.concatenate([
            get_hcp_roi_indices(p, hemi="both", mesh=mesh)
            for p in PARCELS[metric] if p in valid
        ]))

    Any parcel name not present in the live atlas (naming can vary by ``mne``
    version) is dropped *and logged* -- we never silently substitute wrong
    indices. The result is **static** (input-independent), so it is persisted
    to ``cache_dir/roi_masks_hcpmmp1_fsaverage5.npz`` and reloaded thereafter.

    This is the only function in the module that needs ``tribev2`` (and,
    transitively, ``mne`` + a one-time ~1.5 GB atlas download -- §5/§7). The
    import is **guarded and deferred**: importing :mod:`tribescore.metrics`
    works without ``tribev2`` installed; only *calling* this function off the
    Space without the dependency raises.

    Parameters
    ----------
    cache_dir:
        Directory for the persisted ``.npz`` mask cache. Created if missing.
    mesh:
        Surface mesh name. Must be ``"fsaverage5"`` -- the checkpoint's output
        space (§3). Exposed only so the cache key and atlas calls stay in sync.

    Returns
    -------
    dict
        ``{metric_name: np.ndarray}`` -- a sorted, unique 1-D int array of
        vertex indices per metric, ready to index columns of a
        ``(T, 20484)`` timeline.

    Raises
    ------
    RuntimeError
        If ``tribev2`` (or its atlas backend) is unavailable when a fresh
        build is required -- i.e. called off-Space without the dependency and
        no cache present. The message points at the gate in §0/§5.
    ValueError
        If, after validation, a metric has zero usable parcels (atlas
        mismatch severe enough that the masks would be meaningless).
    """
    cache_path = os.path.join(cache_dir, ROI_MASKS_FILENAME)

    # --- fast path: load from the persisted cache --------------------------
    cached = _load_roi_masks(cache_path)
    if cached is not None:
        return cached

    # --- slow path: build from the live atlas (needs tribev2 + mne) --------
    try:  # guarded, deferred import -- keeps this module import-safe locally
        from tribev2.utils import get_hcp_labels, get_hcp_roi_indices
    except Exception as exc:  # ImportError or a backend (mne) import failure
        raise RuntimeError(
            "build_roi_masks needs `tribev2` (which pulls in `mne` + the "
            "HCP-MMP1 atlas) to compute ROI vertex indices. Per docs/PLAN.md "
            "§0/§5 this runs on the HF Space, not locally. No mask cache was "
            f"found at {cache_path!r} either, so the masks cannot be built "
            f"here. Original error: {exc!r}"
        ) from exc

    # The live set of valid bare Glasser names for this mesh (§5 startup guard).
    valid = set(get_hcp_labels(mesh=mesh, combine=False, hemi="both").keys())

    masks: Dict[str, np.ndarray] = {}
    for metric, parcels in PARCELS.items():
        present = [p for p in parcels if p in valid]
        missing = [p for p in parcels if p not in valid]
        if missing:
            logger.warning(
                "build_roi_masks[%s]: %d/%d parcels not in the %s HCP-MMP1 "
                "atlas, dropping them: %s",
                metric, len(missing), len(parcels), mesh, missing,
            )
        if not present:
            raise ValueError(
                f"metric {metric!r} has no valid Glasser parcels against the "
                f"{mesh} HCP-MMP1 atlas (wanted {parcels}); cannot build a "
                "mask. Check the installed `mne` atlas naming."
            )
        idx = np.unique(
            np.concatenate(
                [
                    np.asarray(get_hcp_roi_indices(p, hemi="both", mesh=mesh))
                    for p in present
                ]
            )
        ).astype(np.int64)
        masks[metric] = idx
        logger.info(
            "build_roi_masks[%s]: %d parcels -> %d unique vertices",
            metric, len(present), idx.size,
        )

    _save_roi_masks(cache_path, masks)
    return masks


def _load_roi_masks(cache_path: str) -> Dict[str, np.ndarray] | None:
    """Load persisted masks from ``cache_path``; return ``None`` if absent.

    A corrupt/unreadable cache is treated as a miss (logged) rather than a
    hard error, so a bad file self-heals on the next build.
    """
    if not os.path.exists(cache_path):
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as npz:
            masks = {k: np.asarray(npz[k], dtype=np.int64) for k in npz.files}
        if not masks:
            return None
        logger.info("build_roi_masks: loaded %d masks from %s", len(masks), cache_path)
        return masks
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "build_roi_masks: could not read mask cache %s (%r); rebuilding.",
            cache_path, exc,
        )
        return None


def _save_roi_masks(cache_path: str, masks: Mapping[str, np.ndarray]) -> None:
    """Persist masks to ``cache_path`` as a ``.npz`` (best effort)."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    try:
        np.savez(cache_path, **{k: np.asarray(v) for k, v in masks.items()})
        logger.info("build_roi_masks: cached %d masks to %s", len(masks), cache_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "build_roi_masks: could not write mask cache %s (%r); continuing "
            "without persistence.",
            cache_path, exc,
        )


# ---------------------------------------------------------------------------
# T-C : to_metrics  (docs/PLAN.md §5, §9 T-C)
# ---------------------------------------------------------------------------


def to_metrics(
    timeline: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Reduce a ``(T, 20484)`` activity timeline to named metric curves.

    For each metric mask, the pipeline (§5) is, per TR:

    1. **ROI mean** -- average activity over that metric's vertices ->
       ``raw`` of shape ``(T,)``.
    2. **Full-timeline z-score** -- ``(raw - raw.mean()) / (raw.std() + 1e-6)``
       (the canonical, plotted series; absolute level is not meaningful).
    3. **Gaussian temporal smoothing** -- ``gaussian_filter1d`` with
       ``sigma = 2`` s (TR = 1 s), ``truncate = 3`` (~13-tap), matching BOLD
       sluggishness while preserving peaks for click-to-seek.

    Pure numpy + scipy -- no model, fully unit-testable.

    Parameters
    ----------
    timeline:
        Activity of shape ``(T, 20484)`` (the stitched, globally-z-scored
        output of :func:`tribescore.windowing.stitch`). Columns are
        ``fsaverage5`` vertices in the order the model emits them.
    masks:
        ``{metric_name: vertex_indices}`` from :func:`build_roi_masks` (or, in
        tests, hand-made index arrays).

    Returns
    -------
    dict
        ``{metric_name: curve}`` with one z-scored, smoothed ``(T,)`` curve
        per mask, in the same order as ``masks``.

    Raises
    ------
    ValueError
        If ``timeline`` is not 2-D, or a mask references a vertex index
        outside ``[0, n_vertices)``.
    """
    timeline = np.asarray(timeline, dtype=float)
    if timeline.ndim != 2:
        raise ValueError(f"timeline must be 2-D (T, V), got {timeline.shape}")
    n_t, n_v = timeline.shape

    out: Dict[str, np.ndarray] = {}
    for name, idx in masks.items():
        idx = np.asarray(idx)
        if idx.size == 0:
            raise ValueError(f"mask {name!r} is empty")
        if idx.min() < 0 or idx.max() >= n_v:
            raise ValueError(
                f"mask {name!r} indexes vertices outside [0, {n_v}) "
                f"(min={int(idx.min())}, max={int(idx.max())})"
            )
        raw = timeline[:, idx].mean(axis=1)               # (T,) ROI mean
        z = _zscore(raw)                                   # full-timeline z
        out[name] = _smooth(z, n_t)                        # Gaussian sigma=2 s
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    """Full-series z-score with the §5 floor: ``(x - mean) / (std + 1e-6)``."""
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + _EPS)


def _smooth(x: np.ndarray, n_t: int) -> np.ndarray:
    """Gaussian temporal smoothing (sigma=2 s, truncate=3); length-preserving.

    Degenerate short series (``T < 2``) are returned unchanged -- there is
    nothing to smooth and ``gaussian_filter1d`` on a single sample is a no-op.
    """
    if n_t < 2:
        return x
    return gaussian_filter1d(
        x, sigma=_SMOOTH_SIGMA_S, truncate=_SMOOTH_TRUNCATE, mode="nearest"
    )


# ---------------------------------------------------------------------------
# T-C : summary  (docs/PLAN.md §5, §9 T-C)
# ---------------------------------------------------------------------------


def summary(curves: Mapping[str, np.ndarray]) -> Dict[str, dict]:
    """Per-metric headline statistics for the summary strip (§5/§6).

    Everything is reported in **relative** terms only (z-units / percentile)
    -- absolute "scores" are meaningless for this model (§5 caveat).

    Parameters
    ----------
    curves:
        ``{metric_name: curve}`` as returned by :func:`to_metrics` -- each a
        z-scored, smoothed ``(T,)`` array.

    Returns
    -------
    dict
        ``{metric_name: stats}`` where ``stats`` has:

        * ``peak``           -- max value, in z-units.
        * ``mean``           -- mean value, in z-units (~0 for a z-scored
          curve; nonzero after smoothing/asymmetry).
        * ``peak_time``      -- integer TR index (= second, TR = 1 s) of the
          peak; ``-1`` for an empty curve.
        * ``peak_percentile``-- the peak's percentile rank within its own
          curve (100.0 for a strict global max).
    """
    out: Dict[str, dict] = {}
    for name, curve in curves.items():
        curve = np.asarray(curve, dtype=float)
        if curve.size == 0:
            out[name] = {
                "peak": float("nan"),
                "mean": float("nan"),
                "peak_time": -1,
                "peak_percentile": float("nan"),
            }
            continue
        peak_idx = int(np.argmax(curve))
        peak_val = float(curve[peak_idx])
        # Percentile rank of the peak within the curve (fraction <= peak).
        peak_pct = float((curve <= peak_val).mean() * 100.0)
        out[name] = {
            "peak": peak_val,
            "mean": float(curve.mean()),
            "peak_time": peak_idx,
            "peak_percentile": peak_pct,
        }
    return out


# ===========================================================================
# Legacy synthetic-atlas API (backward compatibility)
# ===========================================================================
#
# The package re-exports ``DEFAULT_METRICS`` and ``reduce_to_metrics`` from
# :mod:`tribescore` (see ``tribescore/__init__.py``), and earlier code/tests
# built curves from an injectable :class:`Atlas`. That surface is kept intact
# and unchanged below so nothing that imports it breaks. New code should use
# the T-C API above (:data:`PARCELS`, :func:`build_roi_masks`,
# :func:`to_metrics`, :func:`summary`), which is the real, Glasser-grounded
# pipeline from docs/PLAN.md §5.


@dataclass(frozen=True)
class MetricSpec:
    """Definition of one *legacy* synthetic-atlas metric.

    Attributes
    ----------
    name:
        Display name of the metric (e.g. ``"attention"``).
    roi_weights:
        Mapping from ROI label -> signed weight. The metric at each time
        point is the weighted average of those ROIs' mean activity.
    description:
        Short human-readable rationale.
    """

    name: str
    roi_weights: Mapping[str, float]
    description: str = ""


#: Legacy illustrative metric suite (synthetic ROI *names*, not Glasser
#: parcels). Retained only for backward compatibility / existing re-exports;
#: the canonical, cited mapping is :data:`PARCELS`.
DEFAULT_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="attention",
        roi_weights={
            "dorsal_attention": 1.0,
            "frontoparietal": 0.7,
            "visual": 0.5,
            "default_mode": -0.4,  # DMN deactivates under focused attention
        },
        description=(
            "Fronto-parietal / dorsal-attention engagement minus "
            "default-mode activity."
        ),
    ),
    MetricSpec(
        name="virality",
        roi_weights={
            "limbic": 1.0,  # reward / affective salience
            "ventral_attention": 0.6,  # salience / orienting
            "visual": 0.3,
        },
        description=(
            "Reward/affective salience and orienting response -- a heuristic "
            "proxy for how 'shareable' a moment looks."
        ),
    ),
    MetricSpec(
        name="engagement",
        roi_weights={
            "visual": 0.5,
            "somatomotor": 0.5,
            "dorsal_attention": 0.5,
            "frontoparietal": 0.5,
            "limbic": 0.5,
            "ventral_attention": 0.5,
            "default_mode": 0.5,
        },
        description="Overall cortical involvement (global activity).",
    ),
)


@dataclass
class Atlas:
    """Vertex -> ROI assignment for a fixed mesh (legacy synthetic API).

    Attributes
    ----------
    labels:
        Array of shape ``(n_vertices,)`` of ROI label strings, one per
        vertex, aligned to the columns of the activity timeline.
    roi_names:
        Sorted unique ROI names (cached for convenience).
    """

    labels: np.ndarray
    roi_names: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.labels.ndim != 1:
            raise ValueError(
                f"atlas labels must be 1-D (n_vertices,), got {self.labels.shape}"
            )
        if not self.roi_names:
            self.roi_names = tuple(sorted(set(self.labels.tolist())))


def load_fsaverage5_atlas() -> Atlas:
    """Legacy hook (deferred). Use :func:`build_roi_masks` instead.

    Raises
    ------
    NotImplementedError
        Always -- the real ROI grounding is now :func:`build_roi_masks`
        (HCP-MMP1 / Glasser via ``tribev2.utils``). For tests, construct an
        :class:`Atlas` with synthetic labels directly.
    """
    raise NotImplementedError(
        "Synthetic fsaverage5 atlas loading is superseded by build_roi_masks "
        "(HCP-MMP1 / Glasser). For tests, construct an Atlas with synthetic "
        "labels directly."
    )


def roi_means(timeline: np.ndarray, atlas: Atlas) -> Dict[str, np.ndarray]:
    """Average activity within each ROI at every time point (legacy API).

    Parameters
    ----------
    timeline:
        Activity of shape ``(T, n_vertices)``.
    atlas:
        Vertex-to-ROI assignment; ``atlas.labels`` must have length
        ``n_vertices``.

    Returns
    -------
    dict
        ``{roi_name: curve}`` where each ``curve`` has shape ``(T,)``.

    Raises
    ------
    ValueError
        If the atlas length does not match the number of vertices.
    """
    timeline = np.asarray(timeline, dtype=float)
    if timeline.ndim != 2:
        raise ValueError(f"timeline must be 2-D (T, V), got {timeline.shape}")
    if atlas.labels.shape[0] != timeline.shape[1]:
        raise ValueError(
            f"atlas has {atlas.labels.shape[0]} labels but timeline has "
            f"{timeline.shape[1]} vertices"
        )

    out: Dict[str, np.ndarray] = {}
    for roi in atlas.roi_names:
        mask = atlas.labels == roi
        if not np.any(mask):
            continue
        out[roi] = timeline[:, mask].mean(axis=1)
    return out


def reduce_to_metrics(
    timeline: np.ndarray,
    atlas: Atlas,
    metrics: Sequence[MetricSpec] = DEFAULT_METRICS,
    *,
    smooth_window: int = 1,
    rescale_0_1: bool = True,
) -> Dict[str, np.ndarray]:
    """Collapse a ``(T, n_vertices)`` timeline into named curves (legacy API).

    For each :class:`MetricSpec`, take the weighted combination of its ROIs'
    mean activity, optionally smooth it over time, and optionally rescale to
    ``[0, 1]`` for display. Superseded by :func:`to_metrics` for production.

    Parameters
    ----------
    timeline:
        Activity of shape ``(T, n_vertices)``.
    atlas:
        Vertex-to-ROI assignment.
    metrics:
        Metric specifications to compute. Defaults to :data:`DEFAULT_METRICS`.
    smooth_window:
        Moving-average window length (in TRs/samples). ``1`` disables smoothing.
    rescale_0_1:
        If ``True``, min-max rescale each curve to ``[0, 1]`` for plotting.

    Returns
    -------
    dict
        ``{metric_name: curve}`` where each ``curve`` has shape ``(T,)``.
    """
    per_roi = roi_means(timeline, atlas)
    n_t = timeline.shape[0]

    out: Dict[str, np.ndarray] = {}
    for spec in metrics:
        curve = np.zeros(n_t, dtype=float)
        total_w = 0.0
        for roi, weight in spec.roi_weights.items():
            if roi in per_roi:
                curve = curve + weight * per_roi[roi]
                total_w += abs(weight)
        if total_w > 0:
            curve = curve / total_w
        if smooth_window > 1:
            curve = _moving_average(curve, smooth_window)
        if rescale_0_1:
            curve = _minmax(curve)
        out[spec.name] = curve
    return out


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge-padding to preserve length (legacy)."""
    if window <= 1 or x.size == 0:
        return x
    window = min(window, x.size)
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad : pad + x.size]


def _minmax(x: np.ndarray) -> np.ndarray:
    """Rescale to ``[0, 1]``; flat signals map to all-zeros (legacy)."""
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
