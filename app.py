"""Gradio entrypoint for the TRIBE v2 Video Brain-Score Space.

Score a 4-5 minute video with ``facebook/tribev2`` and plot derived
brain-metric curves (attention, virality, engagement) over the full timeline.

IMPORTANT -- import safety
--------------------------
This module must ``import`` cleanly on a CPU-only machine WITHOUT ``torch``,
``tribev2``, or ``spaces`` installed (CI runs ``ast.parse`` / a plain import
as a smoke test). Therefore:

  * The model is NEVER loaded at import time. It is loaded lazily, once, the
    first time inference runs -- and only inside the Space (guarded by
    :func:`tribescore.inference.on_spaces`).
  * ``spaces`` is imported behind a try/except with a no-op ``GPU`` fallback,
    so the decorator is always defined even when the package is absent.
  * Heavy work lives in :mod:`tribescore`; this file is just UI + wiring.

Model execution runs on the Hugging Face ZeroGPU Space. See README.md.
"""

from __future__ import annotations

import os

import gradio as gr

from tribescore import __version__
from tribescore.inference import DEFAULT_MODEL_ID, on_spaces

# whisperx (pulled in by tribev2) runs through an old huggingface_hub; disable
# hf_transfer to match the proven reference Space.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# ZeroGPU budget (seconds) for one full-video scoring call. The reference
# demo uses 480s (8 min); a 4-5 min video plus feature extraction fits there.
GPU_DURATION_S = 480

# Demo input: a short, freely-licensed clip so the Space works out of the box.
DEFAULT_VIDEO_URL = (
    "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4"
)


# ---------------------------------------------------------------------------
# spaces.GPU shim: real decorator on the Space, transparent no-op elsewhere so
# this file imports without the `spaces` package present.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only where `spaces` is installed
    import spaces

    gpu = spaces.GPU(duration=GPU_DURATION_S)
except Exception:  # ModuleNotFoundError locally, or any spaces init failure

    def gpu(fn):  # type: ignore[misc]
        """No-op stand-in for ``spaces.GPU`` when the package is unavailable."""
        return fn


# ---------------------------------------------------------------------------
# Lazy, singleton model handle. Populated on first use, on the Space only.
# ---------------------------------------------------------------------------
_MODEL = None


def _get_model():
    """Load the TRIBE v2 model once, lazily, guarded to the Space runtime.

    Never called at import time. Raises a clear error if invoked off-Space or
    without the heavy stack, so a local run fails loudly instead of silently
    attempting a multi-GB download.
    """
    global _MODEL
    if _MODEL is None:
        if not on_spaces():
            raise RuntimeError(
                "Model loading is only supported on the Hugging Face ZeroGPU "
                "Space. Run this app there; locally it stays a UI shell."
            )
        # Deferred import: heavy stack touched only here, only on the Space.
        from tribescore.inference import load_model

        _MODEL = load_model(DEFAULT_MODEL_ID)
    return _MODEL


# ---------------------------------------------------------------------------
# Inference callback (UI -> windowed prediction -> metric curves -> figure).
# ---------------------------------------------------------------------------
@gpu
def score_video(video_url: str):
    """Score a video and return a brain-metric timeline figure.

    End-to-end flow (assembled from the :mod:`tribescore` building blocks):

        1. download the clip                       (inference helpers)
        2. build the events DataFrame              (inference.build_events)
        3. windowed prediction over the timeline   (windowing.run_windowed,
           with ``infer_fn = partial(predict_window, model, events)``)
        4. ROI reduction to metric curves          (metrics.reduce_to_metrics)
        5. render the timeline figure              (plotting.plot_metric_timeline)

    Returns
    -------
    plotly.graph_objects.Figure
        The metric-vs-time plot for ``gr.Plot``.

    Notes
    -----
    Body is deferred (TODO). It is intentionally thin -- it only orchestrates
    already-specified functions. Kept minimal here so this module imports
    without the model present.
    """
    raise NotImplementedError(
        "score_video is wired up on the Space. The building blocks live in "
        "tribescore.{inference,windowing,metrics,plotting}."
    )


# ---------------------------------------------------------------------------
# UI shell
# ---------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    """Build the Gradio Blocks UI (no model touched here)."""
    with gr.Blocks(title="TRIBE v2 Video Brain-Score") as demo:
        gr.Markdown(
            f"""
            # TRIBE v2 Video Brain-Score

            Score a 4-5 minute video with **`{DEFAULT_MODEL_ID}`** and plot
            derived **brain-metric** curves -- *attention*, *virality*,
            *engagement* -- across the full timeline.

            > **Note:** these curves are *derived, heuristic* interpretations
            > of predicted fMRI-like brain activity, mapped from cortical
            > regions of interest. They are exploratory, not validated
            > measurements. See the README / NOTICE.

            > **TODO:** wire `score_video` to the `tribescore` pipeline. Model
            > execution runs on **ZeroGPU** (this UI shell imports without the
            > model present).
            """
        )

        with gr.Row():
            video_url = gr.Textbox(
                label="Video URL",
                value=DEFAULT_VIDEO_URL,
                placeholder="https://.../clip.mp4  (4-5 min, .mp4/.mkv/.mov/.webm)",
            )
        run_btn = gr.Button("Score video", variant="primary")
        plot = gr.Plot(label="Derived brain metrics over time")

        # Wiring is in place; the callback itself is a guarded TODO until the
        # Space-side pipeline lands.
        run_btn.click(fn=score_video, inputs=[video_url], outputs=[plot])

        gr.Markdown(f"<sub>tribescore v{__version__}</sub>")

    return demo


# Module-level handle so `gradio app.py` / Spaces autodetection can find it,
# without launching at import (guarded by the __main__ check below).
demo = build_demo()


if __name__ == "__main__":
    demo.launch()
