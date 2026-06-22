"""Local Apple-silicon (MPS) enablement — applied OFF the Space only.

The TRIBE v2 stack pins device per feature-extractor and for the brain head.
The shipped ``facebook/tribev2`` ``config.yaml`` hardcodes ``device: cuda`` for
every extractor (video / video.image / audio / text), and neuralset's
``HuggingFaceMixin.model_post_init`` (``extractors/base.py``) only re-resolves
the value when it is the string ``"auto"`` — so on an Apple-silicon box the
config's literal ``"cuda"`` flows straight through to ``.to("cuda")`` and raises
*"Torch not compiled with CUDA enabled"*. tribev2's ``TribeModel.from_pretrained``
(``demo_utils.py``) has the same ``auto -> cuda/cpu`` logic for the head.

This module monkeypatches the neuralset resolution so that, on Apple silicon,
the resolved device for every HF backbone is **coerced to MPS** (covering
V-JEPA 2 via ``HuggingFaceVideo.image``, Wav2Vec2-BERT via ``HuggingFaceAudio``
and Llama via ``HuggingFaceText`` through the single ``HuggingFaceMixin`` hook),
and the head is steered via :func:`head_device`. Setting ``TRIBE_FORCE_CPU=1``
coerces everything to CPU instead (an explicit-CPU debug path).

It is import-safe and a **no-op** when torch / neuralset are absent or neither
MPS nor the CPU-override applies (e.g. on the ZeroGPU Space) — so the Space path
is never affected, mirroring the existing ``tribescore.patches`` (bf16) pattern.

.. note::
    ``PYTORCH_ENABLE_MPS_FALLBACK=1`` must be set **before the first
    ``import torch``** in the process for it to take effect, so the real entry
    points (``app.py`` / local launchers) set it at module top. :func:`enable_mps`
    also sets it defensively.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("tribescore.mps")


def _force_cpu() -> bool:
    return bool(os.environ.get("TRIBE_FORCE_CPU"))


def mps_available() -> bool:
    """``True`` iff a usable Metal (MPS) backend is present in this torch build."""
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _target_device() -> str | None:
    """The local device to coerce backbones onto, or ``None`` for a no-op.

    ``cpu`` when ``TRIBE_FORCE_CPU`` is set, else ``mps`` when available, else
    ``None`` (leave the upstream cuda/cpu resolution untouched — e.g. on Space).
    """
    if _force_cpu():
        return "cpu"
    return "mps" if mps_available() else None


def head_device() -> str:
    """Device string for ``TribeModel.from_pretrained`` (the brain head).

    Kept in lock-step with the extractor coercion in :func:`enable_mps` so the
    head and the backbones never resolve to different devices (which would crash
    on a cross-device matmul). ``"auto"`` off Apple-silicon preserves upstream.
    """
    return _target_device() or "auto"


def _wrap_post_init(cls, target: str) -> None:
    """Wrap ``cls.model_post_init`` to coerce the resolved ``device`` to *target*.

    Idempotent. Uses ``object.__setattr__`` to bypass the pydantic ``Literal``
    validation on ``device`` (which only admits auto/cpu/cuda) — safe because
    neuralset excludes ``device`` from both the class- and cache-UID, so the
    non-literal value never reaches serialisation/caching.
    """
    if getattr(cls, "_tribescore_mps", False):
        return
    _orig = cls.model_post_init

    def _patched(self, log__):  # noqa: ANN001 - mirror upstream signature
        _orig(self, log__)
        if getattr(self, "device", None) != target:
            object.__setattr__(self, "device", target)

    cls.model_post_init = _patched
    cls._tribescore_mps = True


def enable_mps(*, verbose: bool = True) -> bool:
    """Coerce neuralset HF backbones onto the local device (MPS, or CPU override).

    Returns ``True`` if a coercion patch was applied, ``False`` if it was a
    no-op (no torch/neuralset, or neither MPS nor ``TRIBE_FORCE_CPU`` applies).
    Call once before the first ``TribeModel.from_pretrained`` / inference.
    """
    target = _target_device()
    if target is None:
        return False

    if target == "mps":
        # Any op without an MPS kernel falls back to CPU instead of raising.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    try:
        from neuralset.extractors import base as nsbase
    except Exception as exc:  # neuralset absent (e.g. on a bare CPU box) -> skip
        logger.warning("MPS device patch skipped (neuralset unavailable): %r", exc)
        return False

    # One hook covers V-JEPA2 (.image), Wav2Vec2-BERT and Llama.
    _wrap_post_init(nsbase.HuggingFaceMixin, target)

    # OpticalFlow keeps its own device field (inactive in this pipeline) — patch
    # defensively so nothing in the video module can sneak back onto cuda/cpu.
    try:
        from neuralset.extractors import video as nsv

        if hasattr(nsv, "OpticalFlow"):
            _wrap_post_init(nsv.OpticalFlow, target)
    except Exception:
        pass

    if verbose:
        logger.info("device patch applied (HF backbones -> %s)", target)
    return True
