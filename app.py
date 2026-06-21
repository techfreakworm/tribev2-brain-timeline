"""Gradio entrypoint — TRIBE v2 Video Brain-Score (Cortical Observatory).

Score a video / audio / text input with ``facebook/tribev2`` and plot derived
brain-metric curves (attention, engagement, virality, …) over the full
timeline. Three input modes (PLAN.md §2) all feed the same pipeline:

    input -> get_events_dataframe -> @spaces.GPU predict (approach B, §4)
          -> windowing.stitch -> metrics.to_metrics -> plotting.timeline_figure

Import safety: the heavy stack (torch / tribev2 / mne) is touched only on the
Space, inside lazily-imported functions. ``gradio`` / ``theme`` / ``ui`` /
``plotly`` are required to import this module (it is the Gradio app), so it is
validated by ``ast.parse`` locally and run for real only on the Space.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

# src-layout: the local ``tribescore`` package lives under ``src/``. HF Spaces
# runs this file from the repo root with no editable install, so make the
# package importable by putting ``<repo>/src`` on the path before importing it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import gradio as gr

import theme
import ui
from tribescore import __version__
from tribescore.inference import (
    DEFAULT_MESH,
    DEFAULT_MODEL_ID,
    load_model,
    on_spaces,
    run_inference,
)
from tribescore.metrics import build_roi_masks, summary, to_metrics
from tribescore.plotting import seek_js, timeline_figure
from tribescore.windowing import stitch

logger = logging.getLogger("tribescore.app")
logging.basicConfig(level=logging.INFO)

# whisperx (pulled in by tribev2) runs via `uvx` in an isolated env that has NO
# hf_transfer. The Space pre-sets HF_HUB_ENABLE_HF_TRANSFER=1, which the whisperx
# subprocess inherits -> its hub download raises "hf_transfer not available".
# FORCE it off (hard set, not setdefault) so the subprocess inherits 0 — exactly
# as the proven reference Space (cbensimon/tribe-v2-demo) does. Affects every
# ASR path (text mode + non-audio_only video/audio).
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# --- constants --------------------------------------------------------------
GPU_DURATION_S = 480          # ZeroGPU reservation per Run (§7; reference uses 480)
MAX_DURATION_S = 300          # 5 min hard cap (§11.3)
MIN_USEFUL_S = 10             # advisory: clips shorter than this are degenerate
MAX_TEXT_CHARS = 5000         # cap TTS length so text mode can't blow GPU time
SAMPLE_VIDEO_URL = (
    "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4"
)

# UI metric display name  <->  metrics.PARCELS key (the two sides differ).
NAME_TO_KEY = {
    "Attention": "Attention",
    "Engagement / arousal": "Engagement",
    "Virality (proxy)": "Virality",
    "Language / semantic load": "Language",
    "Self-relevance / DMN": "Self-relevance",
}
KEY_TO_NAME = {v: k for k, v in NAME_TO_KEY.items()}


# --- cache dir (persistent /data if available, else ./cache; §7) ------------
def _resolve_cache_dir() -> str:
    for cand in ("/data", os.path.join(os.getcwd(), "cache")):
        try:
            os.makedirs(cand, exist_ok=True)
            if os.access(cand, os.W_OK):
                return cand
        except Exception:
            continue
    return tempfile.mkdtemp(prefix="tribescore-")


CACHE_DIR = _resolve_cache_dir()
# Point MNE's data dir at the cache so the HCP-MMP atlas download is reused.
# mne requires MNE_DATA to be an EXISTING directory, so create it up front.
os.environ.setdefault("MNE_DATA", os.path.join(CACHE_DIR, "mne_data"))
os.makedirs(os.environ["MNE_DATA"], exist_ok=True)


# --- spaces.GPU shim: real decorator on the Space, no-op locally ------------
try:  # pragma: no cover - exercised only where `spaces` is installed
    import spaces

    gpu = spaces.GPU(duration=GPU_DURATION_S)
except Exception:

    def gpu(fn):  # type: ignore[misc]
        return fn


# --- lazy singletons --------------------------------------------------------
_MASKS = None


def _get_masks():
    """ROI masks (HCP-MMP1), built once and cached (§5). CPU; needs tribev2+mne."""
    global _MASKS
    if _MASKS is None:
        _MASKS = build_roi_masks(CACHE_DIR, mesh=DEFAULT_MESH)
    return _MASKS


# Eager model load at startup on the Space so the `spaces` runtime registers the
# CUDA allocations during the supported startup phase (§7; mirrors cbensimon /
# qwen). Building the model only downloads the 708 MB ckpt — backbones load
# lazily inside predict() — so this is safe even if a backbone gate were closed.
if on_spaces():
    try:
        load_model(CACHE_DIR)
        logger.info("eager model load complete at startup")
    except Exception as exc:  # non-fatal: first Run retries + surfaces the error
        logger.warning("eager model load failed (will retry on first run): %r", exc)


# --- GPU-timed inference (only the model forward is on the GPU clock) --------
@gpu
def _gpu_infer(mode: str, src_path: str, audio_only: bool):
    """Run one whole-clip predict() on ZeroGPU. Everything else is CPU."""
    model = load_model(CACHE_DIR)  # singleton: instant after eager startup load
    _torch = None
    try:  # TF32 fast matmul/conv (no result change) + peak-VRAM telemetry (logs)
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.backends.cuda.matmul.allow_tf32 = True
            _torch.backends.cudnn.allow_tf32 = True
            _torch.set_float32_matmul_precision("high")
            _torch.cuda.reset_peak_memory_stats()
    except Exception:
        _torch = None
    # NOTE: a blanket bf16 autocast here sped the encode ~2.3x BUT made the model
    # OUTPUT bf16, which broke tribev2's internal `.numpy()` (numpy has no
    # bfloat16) -> "unsupported ScalarType BFloat16". bf16 must live INSIDE the
    # encode loop (neuralset fork, Phase 2), casting features back to fp32 before
    # they're converted to numpy/cached. Plain fp32 here (TF32 still on).
    out = run_inference(model, mode, src_path, audio_only=audio_only)
    try:
        if _torch is not None and _torch.cuda.is_available():
            logger.info("PEAK_VRAM_GB=%.2f", _torch.cuda.max_memory_allocated() / 1e9)
    except Exception:
        pass
    return out


# --- helpers ----------------------------------------------------------------
def _probe_duration_s(path: str) -> float | None:
    """Media duration in seconds via ffprobe; ``None`` if it can't be read."""
    import subprocess

    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1", str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def _text_to_tmp(text: str) -> str:
    """Write text-mode input to a temp ``.txt`` for get_events_dataframe."""
    fd, path = tempfile.mkstemp(suffix=".txt", dir=CACHE_DIR)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _media_html(mode: str, src_path: str) -> str:
    """Custom ``id='tm-video'`` media element the timeline click seeks (§6).

    Uses Gradio's file route to serve the uploaded file. Best-effort: if the
    route differs, the timeline + metrics still render; only seek is affected.
    """
    url = "/gradio_api/file=" + os.path.abspath(src_path)
    if mode == "audio":
        return (
            '<div class="co-video-wrap">'
            f'<audio id="tm-video" controls preload="metadata" src="{url}">'
            "Your browser can't play this audio.</audio></div>"
        )
    if mode == "video":
        suffix = Path(src_path).suffix.lower().lstrip(".") or "mp4"
        return ui.video_html(url, mime=f"video/{suffix}")
    # text mode: synthesized speech is internal; no preview element.
    return (
        '<div class="co-video-wrap"><div class="co-video-empty">'
        "synthesized speech (no preview) — timeline below</div></div>"
    )


def _enter_loading():
    """First click step: reveal the loading state, hide the others."""
    return (
        gr.update(visible=False),  # empty
        gr.update(visible=True),   # loading
        gr.update(visible=False),  # error
        gr.update(visible=False),  # result_grp
    )


def _ok(media, fig, summary_html_str):
    """Success update tuple for [empty, loading, error, result_grp, media, timeline, summary]."""
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=media),
        gr.update(value=fig),
        gr.update(value=summary_html_str),
    )


def _fail(message: str):
    """Error update tuple (same 7 outputs)."""
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value=ui.error_html(message), visible=True),
        gr.update(visible=False),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def _score_impl(mode, src_path, selected_names, audio_only, progress):
    """Shared pipeline: validate -> GPU predict -> stitch -> metrics -> figure.

    Returns the 7-output update tuple for the results column.
    """
    try:
        if not src_path:
            return _fail("Add an input first, then press Score.")

        # Default to the three ON metrics if the user cleared the selection.
        names = list(selected_names) if selected_names else list(ui.METRICS_DEFAULT_ON)
        keys = [NAME_TO_KEY[n] for n in names if n in NAME_TO_KEY]
        if not keys:
            return _fail("Pick at least one brain metric to plot.")

        # Duration guard (media modes only; §11.3).
        if mode in ("video", "audio"):
            dur = _probe_duration_s(src_path)
            if dur is not None and dur > MAX_DURATION_S + 1:
                mins = dur / 60.0
                return _fail(
                    f"This clip is {mins:.1f} min; the max is "
                    f"{MAX_DURATION_S // 60} min. Trim it and try again."
                )

        progress(0.05, desc="Preparing input…")
        if mode == "text":
            if not str(src_path).strip():
                return _fail("Enter some text to score.")

        progress(0.15, desc="Running tribev2 on ZeroGPU (feature extraction + prediction)…")
        preds, abs_times = _gpu_infer(mode, src_path, bool(audio_only))

        if preds.shape[0] == 0:
            return _fail(
                "The model returned no predictions for this input. If it's very "
                "short or silent, try a longer clip with speech."
            )

        progress(0.75, desc="Stitching windows + reducing to metrics…")
        timeline, t_axis = stitch(preds, abs_times)
        masks = _get_masks()
        curves_by_key = to_metrics(timeline, masks)  # keyed by PARCELS key

        # Re-key to UI display names; keep only the selected metrics.
        curves = {
            KEY_TO_NAME[k]: v for k, v in curves_by_key.items() if k in KEY_TO_NAME
        }
        selected_display = [KEY_TO_NAME[k] for k in keys]

        progress(0.92, desc="Rendering timeline…")
        fig = timeline_figure(t_axis, curves, selected=selected_display)

        # Summary for the selected metrics; adapt summary()'s peak_time -> peak_t.
        sel_curves = {n: curves[n] for n in selected_display if n in curves}
        stats = summary(sel_curves)
        stats_html = {
            n: {"peak": s["peak"], "mean": s["mean"], "peak_t": s["peak_time"]}
            for n, s in stats.items()
        }
        summary_str = ui.summary_html(stats_html)

        progress(1.0, desc="Done.")
        media = _media_html(mode, src_path)
        return _ok(media, fig, summary_str)

    except Exception as exc:  # surface a clean, actionable message
        logger.exception("scoring failed")
        msg = str(exc).strip() or exc.__class__.__name__
        # The gated-Llama case has a recognisable signature.
        if "gated" in msg.lower() or "403" in msg or "awaiting" in msg.lower():
            msg = (
                "The text backbone (Llama-3.2-3B) isn't accessible yet. Try the "
                "<strong>Audio-only (debug)</strong> toggle, which skips it."
            )
        return _fail(msg)


# --- app --------------------------------------------------------------------
_RESULT_OUTPUTS_KEYS = (
    "empty", "loading", "error", "result_grp", "media_html", "timeline", "summary",
)


def build_demo() -> gr.Blocks:
    with gr.Blocks(theme=theme.build_theme(), css=theme.CSS, title="TRIBE v2 Video Brain-Score") as demo:
        gr.HTML(
            '<div class="co-masthead">'
            '<div class="co-wordmark">TRIBE&nbsp;v2 '
            '<span class="co-wordmark-sub">· Video Brain-Score</span></div>'
            '<div class="co-status-dot">in-silico neuroscience</div>'
            "</div>"
        )
        if on_spaces():
            gr.HTML(
                '<div class="co-quota">⚡ Runs on <strong>ZeroGPU</strong> — each '
                "Score reserves a GPU slot from <strong>your</strong> daily "
                "allowance. Heaviest: Video · lightest: Text.</div>"
            )

        with gr.Row(equal_height=False):
            # Left rail: the mode-switcher (Video / Audio / Text).
            with gr.Column(scale=4, elem_classes=["co-rail"]):
                with gr.Tabs():
                    with gr.Tab("🎬 Video"):
                        v = ui.build_video_tab()
                    with gr.Tab("🔊 Audio"):
                        a = ui.build_audio_tab()
                    with gr.Tab("📝 Text"):
                        t = ui.build_text_tab()

            # Right hero: the synchronized readout (shared across modes).
            with gr.Column(scale=6):
                r = ui.build_results()

        gr.HTML(
            f'<div class="co-footer">tribescore v{__version__} · model '
            f'<code>{DEFAULT_MODEL_ID}</code> (CC-BY-NC-4.0, non-commercial '
            "research demo) · curves are a derived research proxy, not validated "
            "measurements</div>"
        )

        result_outputs = [r[k] for k in _RESULT_OUTPUTS_KEYS]
        loading_outputs = [r["empty"], r["loading"], r["error"], r["result_grp"]]
        js_seek = seek_js()

        # Sample clip → fill the Video component.
        v["sample_btn"].click(fn=lambda: SAMPLE_VIDEO_URL, inputs=None, outputs=[v["video"]])

        # --- Video ---
        v["run_btn"].click(_enter_loading, None, loading_outputs).then(
            lambda vid, metrics, ao, pr=gr.Progress(): _score_impl("video", vid, metrics, ao, pr),
            inputs=[v["video"], v["metrics"], v["audio_only"]],
            outputs=result_outputs,
        ).then(fn=None, js=js_seek)

        # --- Audio ---
        a["run_btn"].click(_enter_loading, None, loading_outputs).then(
            lambda aud, metrics, ao, pr=gr.Progress(): _score_impl("audio", aud, metrics, ao, pr),
            inputs=[a["audio"], a["metrics"], a["audio_only"]],
            outputs=result_outputs,
        ).then(fn=None, js=js_seek)

        # --- Text ---
        t["run_btn"].click(_enter_loading, None, loading_outputs).then(
            lambda txt, metrics, pr=gr.Progress(): _score_impl(
                "text", _text_to_tmp(txt[:MAX_TEXT_CHARS]) if txt else "", metrics, False, pr
            ),
            inputs=[t["text"], t["metrics"]],
            outputs=result_outputs,
        ).then(fn=None, js=js_seek)

    return demo


demo = build_demo()


if __name__ == "__main__":
    # concurrency 1: one heavy ZeroGPU task at a time. ssr_mode False + show_error
    # so prediction-function exceptions surface in the UI + logs (§7).
    demo.queue(default_concurrency_limit=1).launch(
        show_error=True,
        ssr_mode=False,
        allowed_paths=[CACHE_DIR, tempfile.gettempdir()],
    )
