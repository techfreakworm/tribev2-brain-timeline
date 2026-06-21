"""Thin, import-safe wrapper around the TRIBE v2 model.

This module is the *only* place the real model is touched. Everything heavy --
``torch``, ``tribev2`` -- is imported lazily inside functions so that

    import tribescore.inference

succeeds on a CPU-only machine with none of those installed. The actual model
load and forward pass happen exclusively on the Hugging Face ZeroGPU Space,
behind an ``@spaces.GPU``-decorated entrypoint in ``app.py``.

Reference API (from ``tribev2.demo_utils.TribeModel``)::

    model  = TribeModel.from_pretrained("facebook/tribev2")
    events = model.get_events_dataframe(video_path="clip.mp4")
    preds, segments = model.predict(events)   # preds: (T, n_vertices)

Each row of ``preds`` is one fMRI TR; each ``segment`` carries an ``offset``
and ``duration`` that locate that row on the video timeline -- which is what
:mod:`tribescore.windowing` needs to build a shared time axis.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

import numpy as np

# Type-only imports: evaluated by type checkers, never at runtime, so they
# cannot drag torch/tribev2/pandas into a plain ``import``.
if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

#: Default Hugging Face Hub repo id for the TRIBE v2 checkpoint.
DEFAULT_MODEL_ID = "facebook/tribev2"

#: Default cortical mesh used by the model / plotter (matches the reference
#: Space's ``PlotBrain(mesh="fsaverage5")``).
DEFAULT_MESH = "fsaverage5"


# ---------------------------------------------------------------------------
# Environment guards
# ---------------------------------------------------------------------------


def on_spaces() -> bool:
    """Return ``True`` when running inside a Hugging Face Space.

    Used to gate model loading: heavy initialisation should only happen in
    the Space runtime, never during a local import or a CI syntax check.
    Hugging Face sets ``SPACE_ID`` (and related ``SPACE_*``) env vars on
    every Space.
    """
    return bool(os.environ.get("SPACE_ID") or os.environ.get("SPACE_HOST"))


def torch_available() -> bool:
    """Return ``True`` if ``torch`` can be imported in this environment."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def tribev2_available() -> bool:
    """Return ``True`` if the ``tribev2`` package can be imported."""
    import importlib.util

    return importlib.util.find_spec("tribev2") is not None


def assert_model_runtime() -> None:
    """Raise a clear error if the heavy model stack is unavailable.

    Call this at the top of any function that is about to load or run the
    model, so failures are explicit instead of surfacing as opaque
    ``ImportError``/CUDA errors deep in a call stack.
    """
    missing = []
    if not torch_available():
        missing.append("torch")
    if not tribev2_available():
        missing.append("tribev2")
    if missing:
        raise RuntimeError(
            "TRIBE v2 model runtime is unavailable (missing: "
            f"{', '.join(missing)}). Model execution runs on the Hugging "
            "Face ZeroGPU Space, not locally."
        )


# ---------------------------------------------------------------------------
# Model load + windowed prediction
# ---------------------------------------------------------------------------


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    cache_folder: str | Path = "cache",
    device: str = "auto",
) -> Any:
    """Load the TRIBE v2 model from the Hub (Space-only).

    Lazily imports ``tribev2`` and returns a ready ``TribeModel`` in eval
    mode. This is expensive (downloads weights, allocates GPU memory) and
    must only run on the Space.

    Parameters
    ----------
    model_id:
        Hugging Face Hub repo id of the checkpoint.
    cache_folder:
        Directory for cached features/weights.
    device:
        Torch device string; ``"auto"`` selects CUDA when available.

    Returns
    -------
    tribev2.demo_utils.TribeModel
        Loaded model instance.

    Notes
    -----
    Deferred (TODO). Mirror the reference Space::

        from tribev2.demo_utils import TribeModel
        return TribeModel.from_pretrained(model_id,
                                          cache_folder=str(cache_folder),
                                          device=device)
    """
    assert_model_runtime()
    raise NotImplementedError(
        "load_model is wired up on the Space. See the docstring for the "
        "two-line TribeModel.from_pretrained body."
    )


def build_events(model: Any, video_path: str | Path) -> "pd.DataFrame":
    """Build the events DataFrame for a video (Space-only).

    Thin pass-through to ``TribeModel.get_events_dataframe(video_path=...)``,
    which extracts audio, transcribes words, and attaches sentence/context
    annotations.

    TODO (on Space): ``return model.get_events_dataframe(video_path=str(video_path))``
    """
    assert_model_runtime()
    raise NotImplementedError("build_events is wired up on the Space.")


def predict_window(
    model: Any,
    events: "pd.DataFrame",
    start_s: float,
    end_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict brain activity for one time window -- the injected ``infer_fn``.

    This is the concrete function that :func:`tribescore.windowing.run_windowed`
    calls per window (wrapped in a closure binding ``model`` and ``events``).
    It slices ``events`` to ``[start_s, end_s)``, runs ``model.predict``, and
    returns the window's activity plus per-row offsets *relative to*
    ``start_s`` -- exactly the :class:`tribescore.windowing.WindowResult`
    contract.

    Parameters
    ----------
    model:
        A loaded ``TribeModel``.
    events:
        Full-video events DataFrame from :func:`build_events`.
    start_s, end_s:
        Absolute window bounds in seconds.

    Returns
    -------
    activity : np.ndarray
        Shape ``(t_window, n_vertices)``.
    offsets_s : np.ndarray
        Shape ``(t_window,)``; seconds from ``start_s`` for each row,
        derived from each returned segment's ``offset``.

    Notes
    -----
    Deferred (TODO). Sketch::

        window_events = slice_events(events, start_s, end_s)
        preds, segments = model.predict(window_events)
        offsets = np.array([seg.offset - start_s for seg in segments])
        return preds, offsets

    ``preds`` already has shape ``(n_segments, n_vertices)`` and each segment
    exposes ``.offset`` / ``.duration`` (see ``TribeModel.predict``).
    """
    assert_model_runtime()
    raise NotImplementedError("predict_window is wired up on the Space.")


def slice_events(
    events: "pd.DataFrame", start_s: float, end_s: float
) -> "pd.DataFrame":
    """Return the subset of events overlapping ``[start_s, end_s)``.

    Keeps any event whose ``[start, start + duration)`` span intersects the
    window, then rebases ``start`` so the slice is window-local.

    TODO: implement against the real events schema. Pure pandas, no model.
    """
    raise NotImplementedError("slice_events is implemented alongside the wrapper.")
