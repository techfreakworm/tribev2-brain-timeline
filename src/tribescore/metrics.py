"""Reduce a brain-activity timeline to named "brain-metric" curves.

TRIBE v2 predicts fMRI-like activity over the cortical surface: a tensor of
shape ``(T, n_vertices)`` where each vertex is a location on a standard mesh
(``fsaverage5``). This module collapses that high-dimensional signal into a
handful of interpretable curves over time -- our derived "brain metrics":

    * **attention**  -- engagement of fronto-parietal / dorsal-attention
      regions and primary sensory cortex.
    * **virality**   -- a heuristic combining reward/affective salience with
      the breadth of cortical co-activation (how "shareable" a moment looks).
    * **engagement** -- overall cortical involvement (global activity), a
      catch-all for how much the brain is "doing".

IMPORTANT: these are DERIVED, heuristic interpretations, not validated
neuroscientific or commercial measurements. The mapping from anatomical ROIs
to product-flavoured metric names is an editorial choice of this demo; see
README.md and NOTICE.

The ROI grouping is provided by an atlas that labels each vertex with a
region. In production we resolve a parcellation onto ``fsaverage5`` (e.g. via
``nilearn``); here we keep the math model-free and atlas-injectable so it is
unit-testable with synthetic labels.

Nothing in this module imports torch or tribev2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """Definition of one derived metric.

    Attributes
    ----------
    name:
        Display name of the metric (e.g. ``"attention"``).
    roi_weights:
        Mapping from ROI label -> signed weight. The metric at each time
        point is the weighted average of those ROIs' mean activity. Positive
        weights add, negative weights subtract (e.g. suppressed regions).
    description:
        Short human-readable rationale, surfaced in the UI/tooltip.
    """

    name: str
    roi_weights: Mapping[str, float]
    description: str = ""


#: The default metric suite. ROI *names* here are placeholders for a real
#: parcellation's region labels; the production atlas loader
#: (:func:`load_fsaverage5_atlas`) must map vertices to compatible labels.
#:
#: TODO: replace these illustrative ROI groupings with a concrete, cited
#: parcellation (e.g. Yeo-7/17 networks or Schaefer-400) and tune weights.
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


# ---------------------------------------------------------------------------
# Atlas
# ---------------------------------------------------------------------------


@dataclass
class Atlas:
    """Vertex -> ROI assignment for a fixed mesh.

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
    """Load a parcellation mapped onto the ``fsaverage5`` surface.

    Production helper: fetch a parcellation (e.g. Yeo networks or Schaefer)
    via ``nilearn`` and project it to per-vertex labels matching the order of
    the TRIBE v2 prediction columns.

    Returns
    -------
    Atlas
        Per-vertex ROI labels for ``fsaverage5``.

    Notes
    -----
    Deferred (TODO). This is the only place an atlas resource is fetched, and
    it is called lazily so importing this module stays dependency-free.
    """
    raise NotImplementedError(
        "fsaverage5 atlas loading is implemented on the Space (nilearn). "
        "For tests, construct an Atlas with synthetic labels directly."
    )


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------


def roi_means(timeline: np.ndarray, atlas: Atlas) -> Dict[str, np.ndarray]:
    """Average activity within each ROI at every time point.

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
    """Collapse a ``(T, n_vertices)`` timeline into named metric curves.

    For each :class:`MetricSpec`, take the weighted combination of its ROIs'
    mean activity, optionally smooth it over time, and optionally rescale to
    ``[0, 1]`` for display.

    Parameters
    ----------
    timeline:
        Activity of shape ``(T, n_vertices)`` (typically already z-scored by
        :func:`tribescore.windowing.run_windowed`).
    atlas:
        Vertex-to-ROI assignment.
    metrics:
        Metric specifications to compute. Defaults to :data:`DEFAULT_METRICS`.
    smooth_window:
        Moving-average window length (in TRs/samples) applied to each curve.
        ``1`` disables smoothing.
    rescale_0_1:
        If ``True``, min-max rescale each curve to ``[0, 1]`` for plotting.

    Returns
    -------
    dict
        ``{metric_name: curve}`` where each ``curve`` has shape ``(T,)``.

    Notes
    -----
    The numerics here are intentionally simple and fully implemented so the
    reduction is unit-testable end to end with a synthetic atlas. The
    editorial ROI groupings in :data:`DEFAULT_METRICS` are the placeholder
    part (TODO: replace with a cited parcellation).
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


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge-padding to preserve length."""
    if window <= 1 or x.size == 0:
        return x
    window = min(window, x.size)
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad : pad + x.size]


def _minmax(x: np.ndarray) -> np.ndarray:
    """Rescale to ``[0, 1]``; flat signals map to all-zeros."""
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo <= 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
