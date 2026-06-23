"""Thin, import-safe wrapper around the TRIBE v2 model.

This module is the *only* place the real model is touched. Everything heavy --
``torch``, ``tribev2`` -- is imported lazily inside functions so that

    import tribescore.inference

succeeds on a CPU-only machine with none of those installed. The actual model
load and forward pass happen exclusively on the Hugging Face ZeroGPU Space,
behind an ``@spaces.GPU``-decorated entrypoint in ``app.py``.

Reference API (from ``tribev2.demo_utils.TribeModel`` -- verified against the
upstream source, ``docs/PLAN.md`` §3)::

    from tribev2 import TribeModel
    model  = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=CACHE_DIR,
        device="auto",
        config_update={"data.overlap_trs_train": 20},  # 20 s overlap, §4
    )
    events = model.get_events_dataframe(video_path="clip.mp4")
    preds, segments = model.predict(events)   # preds: (R, 20484)

Each row of ``preds`` is one fMRI TR (TR = 1.0 s, so one row per second);
each ``segment`` carries an absolute ``.start`` (seconds on the input clock)
that locates that row on the media timeline -- which is what
:mod:`tribescore.windowing` needs to build a shared time axis. The window
length is **fixed at 100 s** by the checkpoint pooler (``n_output_timesteps``
is baked into the ckpt; §3); the only loader knob we set is
``data.overlap_trs_train = 20`` to get overlapping 100 s windows at an 80 s
stride.

**This file never decorates anything with ``@spaces.GPU``** so it stays
importable without the ``spaces`` package. The caller in ``app.py`` wraps
:func:`run_inference` in ``@spaces.GPU(duration=480)`` (§7).
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

#: Loader-time config override applied at ``from_pretrained``. 20 TR (= 20 s,
#: TR = 1.0 s) overlap on the ``"all"`` split's segmenter ⇒ overlapping 100 s
#: windows at an 80 s stride (``docs/PLAN.md`` §3/§4). ``data.duration_trs`` is
#: deliberately NOT touched -- the checkpoint pooler locks the window at 100 s
#: and changing it crashes ``predict()`` (§3, R8).
#: ``data.num_workers = 0``: ZeroGPU runs the GPU function in a *daemonic*
#: subprocess, and a daemon process cannot spawn children -- so a DataLoader
#: with ``num_workers > 0`` (the shipped config uses 20) raises
#: ``AssertionError: daemonic processes are not allowed to have children``.
#: Force single-process data loading inside the ``@spaces.GPU`` worker.
CONFIG_UPDATE: dict[str, int] = {
    "data.overlap_trs_train": 20,
    "data.num_workers": 0,
    # Perf: the dominant cost is per-modality backbone feature extraction
    # (V-JEPA2 -> W2V-BERT -> LLaMA, loaded->extract->freed one at a time) at
    # small batch sizes, which underuses ZeroGPU's ~48 GB (RTX Pro 6000
    # Blackwell). Raise the heavy extractors' batch sizes to exploit the idle
    # VRAM. video_feature.image = V-JEPA2-ViT-g (heaviest; #1 win, was 8);
    # text_feature = LLaMA-3.2-3B (was 4); data.batch_size = the small head
    # (near-free). audio_feature has no batch knob; image_feature/DINOv2 is
    # inactive (features_to_use = [text, audio, video]). Throughput-only --
    # outputs/stitch/normalization are unchanged. Ramp empirically if VRAM allows.
    "data.video_feature.image.batch_size": 16,
    "data.text_feature.batch_size": 16,
    "data.batch_size": 16,
}

#: Local Apple-silicon (MPS) loader override. The raised extractor batch sizes
#: above are CUDA/ZeroGPU-tuned and inert-to-counterproductive on MPS (the
#: V-JEPA2 encode loop is hardcoded ``batch=1`` unless the CUDA-gated batched
#: patch applies — it does not on MPS — and the text batch is moot in Fast
#: mode), so locally we keep ONLY the two load-bearing knobs: ``overlap_trs_train
#: = 20`` (locks the overlapping 100 s / 80 s windowing the stitch keystone
#: depends on) and ``num_workers = 0`` (single-process loading; avoids macOS
#: fork/spawn DataLoader quirks). ``data.batch_size`` falls back to the shipped
#: default. Batch size cannot perturb segment/abs-time structure, so this is
#: keystone-safe.
LOCAL_CONFIG_UPDATE: dict[str, int] = {
    "data.overlap_trs_train": 20,
    "data.num_workers": 0,
}

#: Valid ``mode`` values accepted by :func:`run_inference`. These map onto the
#: ``{mode}_path`` keyword of ``TribeModel.get_events_dataframe``.
VALID_MODES: tuple[str, ...] = ("video", "audio", "text")

#: Modes for which the interim ``audio_only`` path (skip ASR + Llama) is valid.
#: Text mode synthesises speech from text and *requires* the full pipeline, so
#: ``audio_only`` is rejected there.
_AUDIO_ONLY_MODES: tuple[str, ...] = ("video", "audio")

#: ``type`` column value for the one-row events DataFrame, per ``mode``. Mirrors
#: ``TribeModel.get_events_dataframe`` (``"Audio"`` for audio, ``"Video"`` for
#: video). Used only by the ``audio_only`` branch.
_EVENT_TYPE_BY_MODE: dict[str, str] = {"video": "Video", "audio": "Audio"}


# ---------------------------------------------------------------------------
# Module-level model singleton
# ---------------------------------------------------------------------------

#: Cached, loaded model. Populated on the first :func:`load_model` call so the
#: 708 MB checkpoint is downloaded/built once and reused across every
#: ``@spaces.GPU`` invocation (the ``spaces`` runtime keeps the module alive
#: between GPU sessions; §7).
_MODEL: Any | None = None


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
    ``ImportError``/CUDA errors deep in a call stack. This is the guard that
    keeps the module importable off-Space while making an *actual* model
    invocation off-Space fail loudly (``docs/PLAN.md`` §0, T-A).
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
# Model load (module-level singleton)
# ---------------------------------------------------------------------------


def load_model(
    cache_dir: str, *, model_id: str = DEFAULT_MODEL_ID, device: str | None = None
) -> Any:
    """Load the TRIBE v2 model from the Hub once and cache it (Space-only).

    Implements ``docs/PLAN.md`` §9 T-A. Lazily imports ``tribev2`` and returns
    a ready ``TribeModel`` in eval mode, built with the locked
    :data:`CONFIG_UPDATE` (``overlap_trs_train = 20``). The result is memoised
    in the module-level :data:`_MODEL` singleton so the 708 MB checkpoint is
    downloaded/built only on the first call and reused across every
    ``@spaces.GPU`` invocation.

    Building the model downloads ``config.yaml`` + ``best.ckpt`` only; the
    modality backbones (incl. gated Llama) load lazily inside ``predict()`` and
    are freed afterwards, so this call succeeds even before Meta approves Llama
    access -- only live inference of the text path would 403 (§3).

    Parameters
    ----------
    cache_dir:
        Directory for cached weights/features (the persistent ``/data`` mount
        on the Space when available; ``docs/PLAN.md`` §7). Created by
        ``from_pretrained`` if it does not exist.
    model_id:
        Hugging Face Hub repo id of the checkpoint. Defaults to
        :data:`DEFAULT_MODEL_ID` (``"facebook/tribev2"``).

    Returns
    -------
    tribev2.TribeModel
        Loaded model instance, ready for ``get_events_dataframe`` / ``predict``.

    Raises
    ------
    RuntimeError
        If invoked off-Space where the model runtime is unavailable (see
        :func:`assert_model_runtime`).
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    assert_model_runtime()

    # Local Apple-silicon: route the HF backbones (V-JEPA2 / W2V-BERT / Llama)
    # and the brain head to MPS. ``enable_mps`` monkeypatches neuralset's
    # ``device="auto"`` resolution (cpu -> mps) and ``head_device()`` returns
    # ``"mps"``; BOTH are no-ops off Apple-silicon (head_device() -> "auto",
    # enable_mps() returns False), so the ZeroGPU Space path is unchanged.
    from tribescore.mps import enable_mps, head_device, mps_available

    local = enable_mps()  # True on Apple-silicon (or TRIBE_FORCE_CPU); no-op on Space
    if device is None:
        device = head_device()
    # On Apple-silicon use the lean local loader config (D4); the Space keeps the
    # CUDA-tuned batch sizes. ``local`` also covers the TRIBE_FORCE_CPU debug path.
    config = LOCAL_CONFIG_UPDATE if (local or mps_available()) else CONFIG_UPDATE

    # Deferred import: the heavy stack is touched only here, only on the Space.
    from tribev2 import TribeModel

    _MODEL = TribeModel.from_pretrained(
        model_id,
        cache_folder=str(cache_dir),
        device=device,
        config_update=dict(config),
    )
    return _MODEL


# ---------------------------------------------------------------------------
# Inference (the whole-video predict, approach B; §4)
# ---------------------------------------------------------------------------


def run_inference(
    model: Any,
    mode: str,
    src_path: str,
    *,
    audio_only: bool = False,
    out_info: dict | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run one whole-clip ``predict()`` and return per-TR activity + abs times.

    Implements ``docs/PLAN.md`` §9 T-A (approach B, §4): a single
    ``get_events_dataframe`` + single ``predict`` per Run. With
    ``overlap_trs_train = 20`` set at load time, tribev2's own segmenter tiles
    the clip into overlapping 100 s windows (80 s stride) internally, so this
    one call covers the entire video; :func:`tribescore.windowing.stitch` then
    re-assembles a continuous 1 Hz timeline from the returned ``abs_times``.

    .. note::
        **The caller decorates this with** ``@spaces.GPU(duration=480)`` (§7).
        It is intentionally left undecorated here so the module imports without
        the ``spaces`` package.

    Parameters
    ----------
    model:
        A loaded ``TribeModel`` (from :func:`load_model`).
    mode:
        One of :data:`VALID_MODES` (``"video"``, ``"audio"``, ``"text"``).
        Selects the ``{mode}_path`` keyword of ``get_events_dataframe``.
    src_path:
        Path to the input media (or ``.txt`` for ``mode == "text"``).
    audio_only:
        Interim fast path (``video``/``audio`` modes only) that **skips ASR +
        Llama** by building the events via
        ``get_audio_and_text_events(df, audio_only=True)``. Used to validate
        the heavy GPU pipeline (V-JEPA2 + DINOv2 + W2V-BERT) + windowing +
        metrics on-Space before Meta approves Llama (§12 step 4). The model
        tolerates the missing text modality (``modality_dropout``; R4).

    Returns
    -------
    preds : np.ndarray
        Shape ``(R, 20484)`` -- per-TR predicted cortical activity on the
        fsaverage5 mesh (LH ``[0:10242]`` then RH ``[10242:20484]``; §3).
    abs_times : np.ndarray
        Shape ``(R,)``, dtype float -- ``round(segment.start)`` for each row,
        i.e. the absolute second on the input clock that the row predicts.
        These carry the within-window ascent + backward-jump-at-seam structure
        that :func:`tribescore.windowing.stitch` keys on (§4 step 3a).

    Raises
    ------
    ValueError
        If ``mode`` is not in :data:`VALID_MODES`, or ``audio_only`` is set for
        ``mode == "text"`` (text requires the full TTS + ASR + Llama path).
    RuntimeError
        If invoked off-Space where the model runtime is unavailable.
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"mode must be one of {VALID_MODES}, got {mode!r}"
        )
    if audio_only and mode not in _AUDIO_ONLY_MODES:
        raise ValueError(
            "audio_only is only valid for video/audio modes (text mode "
            "synthesises speech and needs the full ASR + Llama pipeline)"
        )

    assert_model_runtime()

    if audio_only:
        events = _build_audio_only_events(mode, src_path)
    else:
        events = model.get_events_dataframe(**{f"{mode}_path": src_path})

    # Surface the synthesized-speech file (text mode) so the UI can preview the
    # actual audio the model scored — the text path TTS-synthesises speech inside
    # get_events_dataframe, and its "Audio" event's filepath is that file (written
    # under the infra cache dir -> servable by the app's file route). Best-effort.
    if out_info is not None and mode == "text":
        try:
            arows = events[events["type"] == "Audio"]
            if len(arows):
                out_info["media_path"] = str(arows.iloc[0]["filepath"])
        except Exception:
            pass

    preds, segments = model.predict(events)
    preds = np.asarray(preds, dtype=float)
    abs_times = np.array([round(seg.start) for seg in segments], dtype=float)
    return preds, abs_times


def _build_audio_only_events(mode: str, src_path: str) -> "pd.DataFrame":
    """Build the one-row events DataFrame for the ``audio_only`` path.

    Reproduces the event row that ``TribeModel.get_events_dataframe`` builds
    for audio/video, then routes it through
    ``tribev2.demo_utils.get_audio_and_text_events(df, audio_only=True)`` so
    the ASR (whisperx) + Llama-context transforms are skipped (§9 T-A).

    Deferred import keeps the module CPU-importable.
    """
    import pandas as pd

    from tribev2.demo_utils import get_audio_and_text_events

    event = {
        "type": _EVENT_TYPE_BY_MODE[mode],
        "filepath": str(src_path),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }
    df = pd.DataFrame([event])
    return get_audio_and_text_events(df, audio_only=True)


# ---------------------------------------------------------------------------
# Fallback: per-window ffmpeg trim + predict (§4 "Fallback"; only if approach
# B's overlap-config / .start realignment misbehaves on-Space)
# ---------------------------------------------------------------------------


def infer_window(
    model: Any,
    clip_path: str,
    window_start: float,
    *,
    win_s: int = 100,
    mode: str = "video",
    audio_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict one explicit 100 s window via an ``ffmpeg`` trim (FALLBACK).

    Implements the ``docs/PLAN.md`` §4 *Fallback*: trim ``clip_path`` to
    ``[window_start, window_start + win_s]`` with ``ffmpeg``, run
    ``get_events_dataframe`` + ``predict`` on the trimmed clip, and tag each
    returned row with an absolute time of ``window_start + p`` (``p`` = the
    0-based intra-window TR index). This is deterministic by construction (no
    reliance on ``list_segments`` / ``.start`` realignment or overlap dedup),
    at the cost of N backbone load/free cycles. The caller drives one window
    per ``plan_windows`` span and feeds every ``(preds, abs_times)`` pair into
    the same :func:`tribescore.windowing.stitch`.

    .. note::
        The caller decorates this with ``@spaces.GPU(duration≈200)`` -- one
        bounded GPU session per window (§7). Undecorated here for importability.
        Shells out to ``ffmpeg`` only on the Space.

    Parameters
    ----------
    model:
        A loaded ``TribeModel``.
    clip_path:
        Path to the full-length source media.
    window_start:
        Absolute start time of this window, in seconds (a ``plan_windows``
        span's left edge).
    win_s:
        Window length in seconds. Fixed at 100 (the checkpoint pooler lock;
        §3) -- do not change.
    mode:
        ``"video"`` or ``"audio"`` (the fallback covers media modes; text uses
        the whole-clip path).
    audio_only:
        Skip ASR + Llama, as in :func:`run_inference`.

    Returns
    -------
    preds : np.ndarray
        Shape ``(R, 20484)`` for this window.
    abs_times : np.ndarray
        Shape ``(R,)``, dtype float -- ``window_start + p`` for each row.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"video"``/``"audio"``.
    RuntimeError
        If invoked off-Space, or if the ``ffmpeg`` trim fails.
    """
    if mode not in _AUDIO_ONLY_MODES:
        raise ValueError(
            f"infer_window supports video/audio modes only, got {mode!r}"
        )

    assert_model_runtime()

    import subprocess
    import tempfile

    suffix = Path(clip_path).suffix or (".mp4" if mode == "video" else ".wav")
    with tempfile.TemporaryDirectory() as tmp:
        clip = str(Path(tmp) / f"window{suffix}")
        # Trim [window_start, window_start + win_s]; re-encode so the cut is
        # frame-accurate (input seeking with -c copy lands on keyframes).
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(float(window_start)),
            "-i",
            str(clip_path),
            "-t",
            str(float(win_s)),
            clip,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg trim failed for window at {window_start}s: "
                f"{proc.stderr.strip()[-500:]}"
            )

        if audio_only:
            events = _build_audio_only_events(mode, clip)
        else:
            events = model.get_events_dataframe(**{f"{mode}_path": clip})
        preds, segments = model.predict(events)

    preds = np.asarray(preds, dtype=float)
    # Deterministic x-axis: the p-th retained row sits at window_start + p.
    abs_times = window_start + np.arange(preds.shape[0], dtype=float)
    return preds, abs_times
