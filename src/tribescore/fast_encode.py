"""Frame-dedup V-JEPA 2 encode — exact, keystone-safe latency + quota win.

The encoder samples a 64-frame window per output TR, and adjacent windows OVERLAP
heavily, so the same physical frame is **decoded AND processed many times**
(measured ~8× redundancy: 1536 frame-reads / 193 unique on a 12 s clip). This
reimplements ``HuggingFaceVideo._get_data`` for the vjepa2 path to decode +
process each UNIQUE frame **once** and assemble each clip by indexing.

Why this is numerically EXACT (verified against transformers' VJEPA2 video
processor): the processor is purely per-frame spatial — resize → rescale →
normalize (ImageNet mean/std, crop 256), with **no temporal op / frame-count
enforcement** — so a frame's processed tensor is identical regardless of which
clip it sits in. The V-JEPA 2 forward, ``_aggregate_tokens`` / ``_aggregate_layers``,
and the emitted ``TimedArray`` are reproduced byte-for-byte from the original
loop, so the downstream stitch / metrics / ``abs_times`` (the keystone) are
unchanged. vjepa2-only; falls back to the original ``_get_data`` on ANY error.
Idempotent. Applies bf16 autocast itself (it bypasses ``predict_hidden_states``,
so the separate ``patches.apply_bf16_video_encode`` does not cover this path).

QUOTA NOTE: deduping cuts the decode + processor wall-time inside the
``@spaces.GPU`` reservation (~44 % of per-clip time, ~8× redundant → ~5.5 %),
so it reduces both latency AND billed GPU-seconds. Moving the (now-deduped) prep
fully OUTSIDE ``@spaces.GPU`` is a further increment (see docs/PLAN.md).

⚠ IMPLEMENTED OFFLINE, NOT YET GPU-VALIDATED. Before trusting it, run the
parity check (``profiler.profile_validate_dedup``): same clip, dedup off vs on,
assert max-abs preds diff is float-noise. Until then it is OFF by default
(enable via ``apply_frame_dedup_encode()`` / the ``TRIBE_DEDUP`` env in app.py).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("tribescore.fast_encode")

#: Populated by the last dedup encode run (diagnostics for the validator).
LAST_TIMING: dict = {}

#: Cache of built ``_HFVideoModel`` backbones, keyed by model identity. Built
#: ONCE on CPU in the parent process at container startup (see
#: ``prewarm.prewarm_video_model``), then inherited by every ``@spaces.GPU``
#: fork copy-on-write so each Score pays only the ``.to(cuda)`` transfer — never
#: the ~7.6 GB V-JEPA2 ``from_pretrained`` build, which today runs INSIDE the
#: GPU-timed reservation and is therefore billed. CUDA state cannot be forked,
#: so the cache is deliberately kept on CPU and moved per-fork.
_VIDEO_MODEL_CACHE: dict = {}

#: Diagnostics from the last ``build_video_model`` call — the key validation
#: signal for backbone-reuse. ``hit=True`` inside a Score proves the fork
#: inherited the startup-built backbone (the fork-COW mechanism prep-outside
#: relies on); ``build_s`` is the (un-billed) build time when it was a miss.
LAST_BACKBONE: dict = {"hit": None, "build_s": 0.0}

#: Per-fork cache of the dedup's MATERIALIZED output (one TimedArray per event),
#: keyed by event identity. The DataLoader's per-window __call__ -> _get_timed_arrays
#: -> _get_data re-invokes this; without the cache it RE-EXTRACTS every window (the
#: dedup bypasses exca's cache), which on a 60s/1080p clip was ~59s of redundant
#: billed work (~half the Score). With it, prepare extracts ONCE and the loop
#: retrieves. Exact (returns the identical TimedArray, no recompute). Bounded +
#: naturally per-Score since each @spaces.GPU fork is fresh.
_DEDUP_OUT_CACHE: dict = {}


def _register_layer_empty_cache_hooks(hf_model, every: int) -> int:
    """Register a forward-hook on each V-JEPA2 ENCODER layer that frees the
    per-layer attention transients during the forward, bounding the per-clip peak.

    Targets ONLY `hf_model.encoder.layer` (NOT the predictor / pooler `.layer`
    lists). Fires `torch.mps.synchronize()` + `torch.mps.empty_cache()` every
    `every`-th layer. Idempotent via a flag set ON THE MODEL OBJECT (the model is
    cached in `_VIDEO_MODEL_CACHE` across Score runs, so a module-level flag would
    let hooks stack each run). Returns the number of hooks registered (0 if already
    done or no encoder layers found). MPS-only; callers gate on MPS.
    """
    import torch as _torch

    if getattr(hf_model, "_tribescore_layerhook", False):
        return 0
    enc = getattr(hf_model, "encoder", None)
    layers = getattr(enc, "layer", None)
    if layers is None:
        logger.warning("layer-hook: no encoder.layer ModuleList found; relying on per-clip empty_cache")
        return 0

    counter = {"i": 0}

    def _hook(_module, _inp, _out):
        counter["i"] += 1
        if counter["i"] % every == 0 and hasattr(_torch.mps, "empty_cache"):
            _torch.mps.synchronize()
            _torch.mps.empty_cache()

    n = 0
    for layer in layers:
        layer.register_forward_hook(_hook)
        n += 1
    hf_model._tribescore_layerhook = True
    logger.info("layer-hook: registered per-layer empty_cache on %d encoder layers (every=%d)", n, every)
    return n


def build_video_model(model_name: str, pretrained, layer_type, num_frames):
    """Get-or-build a cached (CPU) ``_HFVideoModel`` backbone. Idempotent.

    Call at startup (parent process) to un-bill the per-Score build; the dedup
    encode path then reuses this instance and only moves it to CUDA inside the
    fork. Safe to call repeatedly — a cache hit is a dict lookup.
    """
    import time as _time

    from neuralset.extractors.video import _HFVideoModel

    key = (model_name, bool(pretrained), layer_type, num_frames)
    model = _VIDEO_MODEL_CACHE.get(key)
    if model is not None:
        LAST_BACKBONE.update(hit=True, build_s=0.0)
        return model
    _t = _time.perf_counter()
    # MPS-ONLY: force EAGER attention for V-JEPA2. The default SDPA falls back to the
    # O(N^2) MPSGraph path and, at 8192 tokens/clip, the unsupported op materializes
    # on CPU (PYTORCH_ENABLE_MPS_FALLBACK) -> ~90GB/clip -> OS OOM. Eager attention is
    # plain matmul+softmax (all MPS-supported, stays on-device, numerically identical
    # to SDPA), so it stays in the Metal pool where the watermark can bound it. CUDA
    # (the Space) keeps fast SDPA. neuralset's _HFVideoModel doesn't expose the kwarg,
    # so we transiently inject it into AutoModel.from_pretrained for this build only.
    import torch as _torch

    _eager = _torch.backends.mps.is_available() and "vjepa2" in model_name
    if _eager:
        import transformers as _tf

        _AM = _tf.AutoModel
        _orig_fp = _AM.from_pretrained

        def _eager_from_pretrained(name, *a, **k):
            k.setdefault("attn_implementation", "eager")
            return _orig_fp(name, *a, **k)

        _AM.from_pretrained = _eager_from_pretrained
    try:
        model = _HFVideoModel(
            model_name=model_name,
            pretrained=pretrained,
            layer_type=layer_type,
            num_frames=num_frames,
        )
        _ = model.model  # force the from_pretrained build now (so it's timed here)
    finally:
        if _eager:
            _AM.from_pretrained = _orig_fp
    # Bound the per-clip MPS peak: free the O(N²) attention transients per-layer
    # during the forward (numerically identical — frees only unreferenced memory).
    # MPS-only; on CUDA the per-clip empty_cache + VRAM headroom suffice.
    if _torch.backends.mps.is_available() and "vjepa2" in model_name:
        every = int(os.environ.get("TRIBE_MPS_EMPTY_EVERY", "4"))
        _register_layer_empty_cache_hooks(model.model, every=every)
    _VIDEO_MODEL_CACHE[key] = model
    LAST_BACKBONE.update(hit=False, build_s=round(_time.perf_counter() - _t, 2))
    logger.info(
        "built + cached V-JEPA2 backbone %s on %s in %.1fs (%d cached)",
        model_name, model.model.device, LAST_BACKBONE["build_s"], len(_VIDEO_MODEL_CACHE),
    )
    return model


def apply_frame_dedup_encode() -> bool:
    """Monkeypatch ``HuggingFaceVideo._get_data`` (vjepa2) to dedup frame prep.

    Returns ``True`` if applied (or already applied), ``False`` if deps absent.
    """
    try:
        import time as _time
        from contextlib import nullcontext

        import numpy as np
        import torch
        from neuralset import base as nsbase
        from neuralset.extractors import video as nsv
        from neuralset.extractors.image import _fix_pixel_values
        from neuralset.extractors.video import _HFVideoModel, _VideoImage
    except Exception as exc:  # neuralset/torch absent (e.g. local import) -> skip
        logger.warning("frame-dedup patch skipped (deps unavailable): %r", exc)
        return False

    cls = nsv.HuggingFaceVideo
    if getattr(cls, "_tribescore_dedup", False):
        return True
    _orig = cls._get_data

    def _key(ts: float) -> float:
        # microsecond rounding: merges float-noise-equal timestamps (same physical
        # frame; structural overlap is exact-equal) but never two different frames
        # (frame period ~tens of ms >> 1e-6 s).
        return round(float(ts), 6)

    def _dedup_get_data(self, events):
        # Only the native vjepa2 path; defer everything else unchanged.
        if "vjepa2" not in self.image.model_name:
            yield from _orig.__get__(self, type(self))(events)
            return
        # Output cache (opt-in via TRIBE_DEDUP_CACHE; default OFF until the
        # multi-window parity gate passes). Key = filepath + extractor CONFIG, NO
        # duration: the loader's per-window __call__ passes a WINDOWED event (its
        # duration is the window, not the full clip), so a duration key would miss;
        # one file+config = one full extraction, sliced downstream per window.
        cache_on = bool(os.environ.get("TRIBE_DEDUP_CACHE"))
        ckeys = []
        for _ev in events:
            if not cache_on:
                ckeys.append(None)
                continue
            try:
                ckeys.append((
                    getattr(_ev, "filepath", None) or repr(_ev),
                    self.image.model_name, self.layer_type,
                    str(self.frequency), self.num_frames, self.max_imsize,
                ))
            except Exception:
                ckeys.append(None)
        # Fast path: all events already extracted -> retrieve without building anything.
        if ckeys and all(k is not None and k in _DEDUP_OUT_CACHE for k in ckeys):
            for k in ckeys:
                yield _DEDUP_OUT_CACHE[k]
            return
        try:
            # Reuse the startup-built CPU backbone (un-bills the per-Score
            # from_pretrained); each fork inherits it COW and only pays .to(cuda).
            model = build_video_model(
                self.image.model_name,
                self.image.pretrained,
                self.layer_type,
                self.num_frames,
            )
            if model.model.device.type == "cpu":
                model.model.to(self.image.device)
            dev = model.model.device

            freq0 = events[0].frequency if self.frequency == "native" else self.frequency
            T = 1 / freq0 if self.clip_duration is None else self.clip_duration
            subtimes = list(
                k / model.num_frames * T for k in reversed(range(model.num_frames))
            )
            # bf16 autocast: ~2x on CUDA (the Space). On MPS it ALSO halves the
            # per-layer attention buffers so the eager forward fits under the Metal
            # watermark (fp32 eager still peaked ~57GB/clip). bf16-on-MPS matches the
            # bf16 Space output better than fp32 did, so it's a parity improvement.
            if torch.cuda.is_available():
                ctx = torch.autocast("cuda", dtype=torch.bfloat16)
            elif torch.backends.mps.is_available():
                ctx = torch.autocast("mps", dtype=torch.bfloat16)
            else:
                ctx = nullcontext()

            for event, ckey in zip(events, ckeys):
                if ckey is not None and ckey in _DEDUP_OUT_CACHE:
                    yield _DEDUP_OUT_CACHE[ckey]  # already extracted this Score
                    continue
                video = event.read()
                freq = self.frequency if self.frequency != "native" else event.frequency
                expect_frames = nsbase.Frequency(freq).to_ind(event.duration)
                times = np.linspace(0, video.duration, expect_frames + 1)[1:]

                # Per-clip frame timestamps (identical formula to the original loop).
                clip_ts = [[max(0.0, t - t2) for t2 in subtimes] for t in times]

                # Unique frames with a representative timestamp.
                uniq: dict[float, float] = {}
                for ts_list in clip_ts:
                    for ts in ts_list:
                        k = _key(ts)
                        if k not in uniq:
                            uniq[k] = ts
                # Decode in SORTED time order so moviepy reads sequentially (forward)
                # instead of seeking per frame — that's what makes deduping a win.
                sorted_keys = sorted(uniq, key=lambda k: uniq[k])
                idx_of = {k: i for i, k in enumerate(sorted_keys)}

                _td = _time.perf_counter()
                uniq_frames = []
                for k in sorted_keys:
                    pil = _VideoImage(video=video, time=uniq[k]).read()
                    if self.max_imsize is not None:
                        factor = max(pil.size) / self.max_imsize
                        if factor > 1:
                            pil = pil.resize(tuple(int(s / factor) for s in pil.size))
                    uniq_frames.append(np.array(pil))
                video.close()
                t_decode = _time.perf_counter() - _td

                # Process ALL unique frames in ONE call as a single U-frame "video"
                # (per-frame spatial only -> identical to per-clip processing).
                _tp = _time.perf_counter()
                inputs = model.processor(videos=[np.array(uniq_frames)], return_tensors="pt")
                _fix_pixel_values(inputs)
                uniq_pv = inputs["pixel_values_videos"][0].to(dev)  # (U, 3, H, W)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_proc = _time.perf_counter() - _tp

                # Assemble + forward each clip by indexing the unique prepped frames.
                _tf = _time.perf_counter()
                _mps = torch.backends.mps.is_available()
                output = np.array([])
                for k, ts_list in enumerate(clip_ts):
                    sel = [idx_of[_key(ts)] for ts in ts_list]
                    clip_pv = uniq_pv[sel].unsqueeze(0)  # (1, num_frames, 3, H, W)
                    with torch.inference_mode(), ctx:
                        pred = model.model(pixel_values_videos=clip_pv)
                    states = pred.hidden_states  # tuple(L) of (1, tokens, feat)
                    out = torch.cat([x.unsqueeze(1) for x in states], axis=1).float()
                    embd = self.image._aggregate_tokens(out[0]).cpu().numpy()
                    if not self.image.cache_all_layers and self.image.cache_n_layers is None:
                        embd = self.image._aggregate_layers(embd)
                    if not output.size:
                        output = np.zeros((len(times),) + embd.shape)
                    output[k] = embd
                    # MPS-ONLY: the Metal caching allocator does NOT return per-clip
                    # intermediates (40-layer hidden_states + attention) to the OS, so
                    # RSS grows ~linearly per TR and OOMs the machine (125 GB on a 15 s
                    # clip). Free + reclaim each iteration -> bounds peak to ~1 clip.
                    # CUDA (the Space) is skipped so it stays fast; it has VRAM headroom.
                    if _mps:
                        del pred, states, out, embd, clip_pv
                        if hasattr(torch.mps, "empty_cache"):
                            torch.mps.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_fwd = _time.perf_counter() - _tf
                LAST_TIMING.clear()
                LAST_TIMING.update(
                    unique=len(sorted_keys),
                    total_reads=sum(len(x) for x in clip_ts),
                    decode_s=round(t_decode, 2),
                    proc_s=round(t_proc, 2),
                    fwd_s=round(t_fwd, 2),
                    backbone_hit=LAST_BACKBONE["hit"],
                    backbone_build_s=LAST_BACKBONE["build_s"],
                )

                output = output.transpose(list(range(1, output.ndim)) + [0])
                ta = nsbase.TimedArray(
                    data=output.astype(np.float32),
                    frequency=freq,
                    start=nsbase._UNSET_START,
                    duration=event.duration,
                )
                if ckey is not None:
                    if len(_DEDUP_OUT_CACHE) > 6:  # safety bound (fork is per-Score anyway)
                        _DEDUP_OUT_CACHE.clear()
                    _DEDUP_OUT_CACHE[ckey] = ta
                yield ta
        except Exception as exc:  # never break inference -> fall back to the correct path
            # TRAP A: on MPS an out-of-memory / HIGH-watermark error must surface
            # CLEANLY, not fall back — the original path decodes+processes MORE
            # (no dedup, no empty_cache) and would re-OOM harder, crashing the OS.
            _m = str(exc).lower()
            if torch.backends.mps.is_available() and (
                isinstance(exc, MemoryError)
                or "out of memory" in _m or "watermark" in _m or "mps backend out" in _m
            ):
                logger.error("frame-dedup hit MPS memory limit (%r); surfacing, NOT falling back", exc)
                raise
            logger.warning("frame-dedup encode failed (%r); falling back to original", exc)
            yield from _orig.__get__(self, type(self))(events)

    cls._tribescore_dedup_orig = _orig  # stash for remove_frame_dedup_encode (A/B test)
    cls._get_data = _dedup_get_data
    cls._tribescore_dedup = True
    logger.info("applied frame-dedup V-JEPA2 encode patch")
    return True


def remove_frame_dedup_encode() -> bool:
    """Restore the original ``_get_data`` (for A/B parity tests). Idempotent."""
    try:
        from neuralset.extractors import video as nsv
    except Exception:
        return False
    cls = nsv.HuggingFaceVideo
    orig = getattr(cls, "_tribescore_dedup_orig", None)
    if orig is not None:
        cls._get_data = orig
        cls._tribescore_dedup = False
        return True
    return False
