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


def apply_batched_video_encode(batch_size: int = 8) -> bool:
    """Batch the V-JEPA2 "Encoding video" loop (vjepa2 only) for ~3-8x.

    neuralset ``HuggingFaceVideo._get_data`` runs one clip/timepoint (hardcoded
    ``batch=1``). This replaces it with a version that forwards B clips per
    processor call in bf16, with byte-identical post-processing
    (``_aggregate_tokens``/``_aggregate_layers`` per clip) so stitch/metrics are
    unchanged. vjepa2-only; falls back to the original for any other model or on
    any error. Idempotent. Bypasses exca disk caching for the vjepa2 path (fine
    for one-shot inference). The OOM guard auto-halves the batch.
    """
    try:
        from contextlib import nullcontext

        import numpy as np
        import torch
        from neuralset import base as nsbase
        from neuralset.extractors import video as nsv
        from neuralset.extractors.image import _fix_pixel_values
        from neuralset.extractors.video import _HFVideoModel, _VideoImage
    except Exception as exc:  # neuralset/torch absent -> skip
        logger.warning("batched video-encode patch skipped (deps unavailable): %r", exc)
        return False

    cls = nsv.HuggingFaceVideo
    if getattr(cls, "_tribescore_batched", False):
        return True
    _orig = cls._get_data

    def _batched_get_data(self, events):
        # Only handle the native-video (vjepa2) path; defer everything else.
        if not any(z in self.image.model_name for z in _HFVideoModel.MODELS):
            yield from _orig(self, events)
            return
        if "vjepa2" not in self.image.model_name:
            yield from _orig(self, events)
            return
        try:
            model = _HFVideoModel(
                model_name=self.image.model_name,
                pretrained=self.image.pretrained,
                layer_type=self.layer_type,
                num_frames=self.num_frames,
            )
            if model.model.device.type == "cpu":
                model.model.to(self.image.device)
            freq0 = events[0].frequency if self.frequency == "native" else self.frequency
            T = 1 / freq0 if self.clip_duration is None else self.clip_duration
            subtimes = list(
                k / model.num_frames * T for k in reversed(range(model.num_frames))
            )
            B = max(1, int(batch_size))

            for event in events:
                video = event.read()
                freq = self.frequency if self.frequency != "native" else event.frequency
                expect_frames = nsbase.Frequency(freq).to_ind(event.duration)
                times = np.linspace(0, video.duration, expect_frames + 1)[1:]
                # Build each clip's frame stack exactly as the original loop.
                clips = []
                for t in times:
                    ims = [_VideoImage(video=video, time=max(0, t - t2)) for t2 in subtimes]
                    pil = [i.read() for i in ims]
                    if pil and self.max_imsize is not None:
                        factor = max(pil[0].size) / self.max_imsize
                        if factor > 1:
                            size = tuple(int(s / factor) for s in pil[0].size)
                            pil = [p.resize(size) for p in pil]
                    clips.append(np.array([np.array(p) for p in pil]))  # (num_frames,H,W,3)
                video.close()

                output = np.array([])
                k = 0
                i = 0
                while i < len(clips):
                    bs = B
                    while True:
                        chunk = clips[i : i + bs]
                        try:
                            inputs = model.processor(videos=list(chunk), return_tensors="pt")
                            _fix_pixel_values(inputs)
                            inputs = inputs.to(model.model.device)
                            ctx = (
                                torch.autocast("cuda", dtype=torch.bfloat16)
                                if torch.cuda.is_available()
                                else nullcontext()
                            )
                            with torch.inference_mode(), ctx:
                                pred = model.model(**inputs)
                            states = pred.hidden_states  # tuple(L) of (bs,tokens,feat)
                            out = torch.cat(
                                [x.unsqueeze(1) for x in states], axis=1
                            ).float()  # (bs,L,tokens,feat)
                            break
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            if bs == 1:
                                raise
                            bs = max(1, bs // 2)
                    if out.shape[0] != len(chunk):
                        raise RuntimeError(
                            f"batched processor returned {out.shape[0]} for "
                            f"{len(chunk)} clips; not batching as expected"
                        )
                    for b in range(out.shape[0]):
                        embd = self.image._aggregate_tokens(out[b]).cpu().numpy()
                        if not self.image.cache_all_layers and self.image.cache_n_layers is None:
                            embd = self.image._aggregate_layers(embd)
                        if not output.size:
                            output = np.zeros((len(times),) + embd.shape)
                        output[k] = embd
                        k += 1
                    i += out.shape[0]
                assert k == len(times)
                output = output.transpose(list(range(1, output.ndim)) + [0])
                yield nsbase.TimedArray(
                    data=output.astype(np.float32),
                    frequency=freq,
                    start=nsbase._UNSET_START,
                    duration=event.duration,
                )
        except Exception as exc:  # never break inference -> fall back to the slow, correct path
            logger.warning("batched encode failed (%r); falling back to original", exc)
            yield from _orig(self, events)

    cls._get_data = _batched_get_data
    cls._tribescore_batched = True
    logger.info("applied batched V-JEPA2 encode patch (B=%d)", batch_size)
    return True
