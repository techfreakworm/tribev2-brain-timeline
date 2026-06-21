"""TRIBE v2 Video Brain-Score: derive brain-metric curves from a video.

This package wraps the ``facebook/tribev2`` model (a model that predicts
fMRI-like brain activity from video, audio, and text) and turns its
per-timepoint predictions into human-readable "brain-metric" curves
(attention, virality, engagement) plotted over a full 4-5 minute timeline.

Module map
----------
``inference``
    Thin, import-safe wrapper around the TRIBE v2 model: load the
    checkpoint and run per-window prediction. Heavy imports (torch,
    tribev2) are deferred so this module imports cleanly on a CPU-only
    box without the model installed.
``windowing``
    Sliding-window orchestration over a long video. Slices events into
    overlapping windows, calls an injected ``infer_fn`` per window, and
    stitches the per-window outputs into a single ``(T, n_vertices)``
    activity timeline plus a monotonic time axis. Pure-numpy and fully
    testable without the model.
``metrics``
    Reduces a ``(T, n_vertices)`` activity timeline to named metric
    curves by aggregating brain activity over regions of interest (ROIs).
``plotting``
    Builds the timeline figure (metric curves vs. time) for the UI.

Design notes
------------
* TRIBE v2 already chunks long clips internally (~60s segments) and emits
  one prediction row per fMRI TR. The :mod:`windowing` layer here is an
  *outer* window: it lets us process arbitrarily long videos in bounded
  GPU calls and smooth across window boundaries via overlap-averaging.
* Nothing in this package loads a model at import time. The only place a
  model is materialised is inside :func:`tribescore.inference.load_model`,
  which is called lazily (and, on the Space, behind an ``@spaces.GPU``
  decorated function).

Model execution runs on a Hugging Face ZeroGPU Space, never locally.
"""

from __future__ import annotations

__all__ = [
    "__version__",
    "run_windowed",
    "reduce_to_metrics",
    "DEFAULT_METRICS",
]

__version__ = "0.1.0"

# Re-export the light, model-free public surface. These imports are safe:
# windowing and metrics depend only on numpy, never on torch/tribev2.
from tribescore.metrics import DEFAULT_METRICS, reduce_to_metrics
from tribescore.windowing import run_windowed
