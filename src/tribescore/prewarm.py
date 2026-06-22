"""Startup pre-warm — cache everything OUTSIDE ``@spaces.GPU`` so no Score downloads.

ZeroGPU bills the ``@spaces.GPU`` function's **wall-time**, and today the gated
Llama / uvx-whisperx / spacy assets download INSIDE the GPU-timed ``predict`` —
i.e. you pay GPU quota for a network download, and the first user eats the
latency. Pre-warming at **container startup** (module import, where ZeroGPU runs
a no-real-GPU CUDA-emulation mode → **not billed**) makes every Score find them
cached. The non-gated HF models are build-baked via ``preload_from_hub`` in the
README; this module covers what canNOT be baked: the MNE/HCP atlas (not on HF),
gated Llama (no build-time auth), the uvx whisperx tool (runtime-installed),
spacy (cli download) and the wav2vec align model (torchaudio).

Robust: each step is independently ``try``/``except`` — a failed pre-warm just
means that one asset downloads on first use (logged), never crashes startup. The
heavy Quality-mode set (Llama + whisperx + spacy, ~10 GB + a uvx install) is
**gated by ``PREWARM_QUALITY``** so a Fast-mode / perf rebuild isn't bogged down.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("tribescore.prewarm")


def prewarm_masks(cache_dir: str, mesh: str = "fsaverage5") -> None:
    """Download the MNE/HCP-MMP atlas + build the ROI masks (both Fast & Quality)."""
    from tribescore.metrics import build_roi_masks

    build_roi_masks(cache_dir, mesh=mesh)
    logger.info("prewarm: ROI masks + HCP-MMP atlas cached")


def prewarm_video_model(tribe_model) -> None:
    """Build + cache the V-JEPA2 backbone on CPU at startup (un-bills per-Score build).

    Today the ~7.6 GB V-JEPA2 ViT-g ``from_pretrained`` runs INSIDE the GPU-timed
    ``predict`` (the dedup encode builds it per Score) — i.e. billed GPU wall-time
    for a disk load. Building it here, in the un-billed parent process, lets every
    ``@spaces.GPU`` fork inherit it copy-on-write and pay only the ``.to(cuda)``
    transfer. No-op unless the model exposes a vjepa2 ``video_feature``.
    """
    vf = getattr(getattr(tribe_model, "data", None), "video_feature", None)
    if vf is None or "vjepa2" not in getattr(getattr(vf, "image", None), "model_name", ""):
        logger.info("prewarm: no vjepa2 video_feature -> skipping backbone preload")
        return
    from tribescore.fast_encode import build_video_model

    build_video_model(vf.image.model_name, vf.image.pretrained, vf.layer_type, vf.num_frames)
    logger.info("prewarm: V-JEPA2 backbone built + cached on CPU")


def prewarm_llama() -> None:
    """Cache the gated Llama-3.2-3B (needs HF_TOKEN; skipped otherwise)."""
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        logger.warning("prewarm: HF_TOKEN absent -> skipping gated Llama-3.2-3B")
        return
    from huggingface_hub import snapshot_download

    snapshot_download(
        "meta-llama/Llama-3.2-3B",
        token=tok,
        allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],
    )
    logger.info("prewarm: Llama-3.2-3B cached")


def prewarm_spacy() -> None:
    """Ensure the spacy English model neuralset's text path uses is present."""
    import spacy

    try:
        spacy.load("en_core_web_lg")
    except Exception:
        from spacy.cli import download

        download("en_core_web_lg")
    logger.info("prewarm: spacy en_core_web_lg ready")


def prewarm_whisperx() -> None:
    """Pre-install the uvx whisperx tool + download large-v3 + the align model.

    Runs whisperx on a 0.5 s silent clip on CPU (no GPU needed just to populate
    the caches). The faster-whisper-large-v3 weights are already build-baked via
    preload_from_hub; this covers the uvx tool install + the torchaudio align model.
    """
    import subprocess
    import tempfile

    wav = "/tmp/prewarm_silence.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
         "-t", "0.5", "-acodec", "pcm_s16le", wav],
        capture_output=True, check=True,
    )
    env = dict(os.environ)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    with tempfile.TemporaryDirectory() as od:
        subprocess.run(
            ["uvx", "whisperx", wav, "--model", "large-v3", "--language", "en",
             "--device", "cpu", "--compute_type", "int8",
             "--align_model", "WAV2VEC2_ASR_LARGE_LV60K_960H",
             "--output_dir", od, "--output_format", "json"],
            capture_output=True, env=env, timeout=1200,
        )
    logger.info("prewarm: whisperx (uvx + large-v3 + align) cached")


def prewarm_all(cache_dir: str, model=None) -> None:
    """Pre-warm everything: masks always; V-JEPA2 backbone if ``model`` given;
    Quality set (Llama/whisperx/spacy) gated by ``PREWARM_QUALITY``."""
    try:
        prewarm_masks(cache_dir)
    except Exception as exc:
        logger.warning("prewarm masks failed (will download on first Score): %r", exc)

    if model is not None:
        try:
            prewarm_video_model(model)
        except Exception as exc:
            logger.warning("prewarm video backbone failed (will build in-fork): %r", exc)

    if os.environ.get("PREWARM_QUALITY"):
        for name, fn in (("llama", prewarm_llama), ("spacy", prewarm_spacy), ("whisperx", prewarm_whisperx)):
            try:
                fn()
            except Exception as exc:
                logger.warning("prewarm %s failed (will download on first use): %r", name, exc)
    else:
        logger.info(
            "prewarm: PREWARM_QUALITY unset -> skipping Llama/whisperx/spacy "
            "(Fast-mode only; set PREWARM_QUALITY=1 to pre-cache the Quality stack)"
        )
