"""Runtime monkeypatches applied ON THE SPACE to speed up inference.

These touch ``neuralset``/``torch`` internals, so they're imported lazily inside
the functions (never at module import) — this file stays importable on a plain
CPU box with neither installed. Apply them from inside the ``@spaces.GPU`` worker
(idempotent) so they're active in the forked GPU process.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("tribescore.patches")


def apply_bf16_video_encode() -> bool:
    """Run the V-JEPA2 video encode (the dominant cost) in bf16, return fp32.

    The "Encoding video" loop (neuralset ``HuggingFaceVideo._get_data``) calls
    ``_HFVideoModel.predict_hidden_states`` once per timepoint (hardcoded
    ``batch=1``). The per-clip ViT-g forward is the bottleneck (~2 s/it, ~104
    iters). Wrapping just that forward in bf16 autocast roughly halves it; we then
    cast the hidden states back to **fp32** so the downstream
    ``_aggregate_tokens(...).cpu().numpy()`` keeps working (numpy has no
    bfloat16 — a blanket autocast around ``predict()`` broke exactly there).

    Scope: V-JEPA2 only. The audio (W2V-BERT), text (LLaMA) and the TRIBE head
    stay fp32. Correctness: ViT-g is LayerNorm-based (no cross-sample mixing) and
    every metric is z-scored over time, so bf16 only adds float-order noise.

    Idempotent; a no-op (returns ``False``) if ``neuralset``/``torch``/CUDA are
    unavailable.
    """
    try:
        import torch
        from neuralset.extractors import video as nsv
    except Exception as exc:  # neuralset/torch absent (e.g. local) -> skip
        logger.warning("bf16 video-encode patch skipped (deps unavailable): %r", exc)
        return False

    model_cls = nsv._HFVideoModel
    if getattr(model_cls, "_tribescore_bf16", False):
        return True

    _orig = model_cls.predict_hidden_states

    def _patched(self, images, audio=None):  # noqa: ANN001 - mirror upstream sig
        if torch.cuda.is_available():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = _orig(self, images, audio)
            # Cast hidden states back to fp32 for downstream numpy/aggregation.
            try:
                return out.float()
            except AttributeError:
                return out
        return _orig(self, images, audio)

    model_cls.predict_hidden_states = _patched
    model_cls._tribescore_bf16 = True
    logger.info("applied bf16 V-JEPA2 encode patch (fp32 output)")
    return True
