"""ZeroGPU step-1 profiling harness — is the V-JEPA2 ViT-g forward compute-bound,
and does the `xlarge` size give a free compute speedup (vs only more VRAM)?

Loads V-JEPA 2 the way the pipeline does (default attention, bf16, output_hidden_states)
and times ONE 64-frame clip forward — GPU-kernel time vs wall — on whichever ZeroGPU
size the decorator requests. Two decorated entrypoints (`profile_large` /
`profile_xlarge`) let us A/B the same clip across tiers in a single deploy.

This is a temporary measurement harness wired into the Space UI; it does not touch
the real inference path.
"""

from __future__ import annotations

import time

VJEPA2 = "facebook/vjepa2-vitg-fpc64-256"


def _profile_core(size_label: str) -> str:
    import torch
    from transformers import AutoModel

    out = [f"### ZeroGPU PROFILE — requested size = {size_label}"]

    if not torch.cuda.is_available():
        return "\n".join(out + ["CUDA not available (not on ZeroGPU?) — no measurement."])

    p = torch.cuda.get_device_properties(0)
    out.append(
        f"GPU = {torch.cuda.get_device_name(0)} | VRAM = {p.total_memory/1e9:.1f} GB "
        f"| SMs = {p.multi_processor_count} | compute = sm_{p.major}{p.minor} "
        f"| torch = {torch.__version__}"
    )

    # Load V-JEPA2 ViT-g exactly as neuralset does: default attn (no override),
    # output_hidden_states, bf16 (the Space's encode precision).
    t0 = time.perf_counter()
    model = (
        AutoModel.from_pretrained(VJEPA2, output_hidden_states=True, torch_dtype=torch.bfloat16)
        .to("cuda")
        .eval()
    )
    load_s = time.perf_counter() - t0
    attn = getattr(model.config, "_attn_implementation", "?")
    out.append(f"load = {load_s:.1f} s | resolved attn_impl = {attn!r} | dtype = bf16")

    # One clip = (B=1, frames=64, C=3, H=256, W=256), as the encoder consumes.
    x = torch.randn(1, 64, 3, 256, 256, dtype=torch.bfloat16, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(3):  # warmup: CUDA kernel autotune / cudnn benchmark
            model(pixel_values_videos=x)
        torch.cuda.synchronize()

        N = 12
        # GPU-active time: cuda events bracket the whole N-forward run (includes any
        # inter-kernel idle gaps -> if wall >> this, there's launch/CPU overhead).
        ev_s = torch.cuda.Event(enable_timing=True)
        ev_e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        ev_s.record()
        for _ in range(N):
            model(pixel_values_videos=x)
        ev_e.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - w0) / N * 1000.0
        gpu_ms = ev_s.elapsed_time(ev_e) / N

        # Kernel breakdown (top-5 self-CUDA) to see if attention dominates (-> flash win).
        top = ""
        try:
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(3):
                    model(pixel_values_videos=x)
                torch.cuda.synchronize()
            rows = prof.key_averages()
            rows = sorted(rows, key=lambda r: r.self_cuda_time_total, reverse=True)[:5]
            tot = sum(r.self_cuda_time_total for r in prof.key_averages()) or 1
            top = " | ".join(
                f"{r.key[:24]} {100*r.self_cuda_time_total/tot:.0f}%" for r in rows
            )
        except Exception as exc:  # profiler optional
            top = f"(profiler unavailable: {type(exc).__name__})"

    peak = torch.cuda.max_memory_allocated() / 1e9
    out.append(
        f"per-forward: GPU-active = {gpu_ms:.1f} ms | wall = {wall_ms:.1f} ms "
        f"| GPU/wall = {gpu_ms/wall_ms:.2f}"
    )
    out.append(
        "=> COMPUTE-BOUND (GPU≈wall: batching/prefetch won't help)"
        if gpu_ms / wall_ms > 0.90
        else "=> launch/CPU overhead present (wall >> GPU: prefetch/batching may help)"
    )
    out.append(f"peak VRAM during forward = {peak:.1f} GB")
    out.append(f"top CUDA kernels: {top}")
    return "\n".join(out)


# --- ZeroGPU entrypoints: same body, different requested size literal -----------
try:  # real decorator on the Space; identity no-op locally
    import spaces

    _gpu_large = spaces.GPU(duration=150, size="large")
    _gpu_xlarge = spaces.GPU(duration=150, size="xlarge")
except Exception:  # pragma: no cover

    def _gpu_large(fn):
        return fn

    def _gpu_xlarge(fn):
        return fn


@_gpu_large
def profile_large() -> str:
    return _profile_core("large")


@_gpu_xlarge
def profile_xlarge() -> str:
    return _profile_core("xlarge")


def _pipeline_core() -> str:
    """Break a REAL Fast-mode video clip's per-clip wall into stages.

    Monkeypatch-times the actual neuralset path (not a reimplementation), so the
    decode backend / processor / device copies are exactly the pipeline's:
      decode  = _VideoImage.read (64 frame reads/clip, moviepy)
      gpu_call= _HFVideoModel.predict_hidden_states (processor + H2D + forward + cat)
      aggregate = BaseExtractor._aggregate_tokens (+ the trailing .cpu().numpy())
    Reports STEADY-STATE per-clip (skips the first 5 warmup clips). gpu_call vs the
    known ~288ms isolated large forward exposes the processor/H2D (CPU-prep) share.
    """
    import os
    import subprocess
    import time
    import urllib.request

    import torch
    from neuralset.extractors import base as nsbase
    from neuralset.extractors import video as nsv

    # Replicate app._gpu_infer's GPU setup so the forward matches the REAL app:
    # TF32 + the bf16 V-JEPA2 encode patch. Without this, run_inference() runs the
    # encoder in fp32 (~6x slower), which would mis-attribute the time to "processor".
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    try:
        from tribescore.patches import apply_bf16_video_encode

        apply_bf16_video_encode()  # patches predict_hidden_states with bf16 autocast
    except Exception:
        pass

    from neuralset.extractors import audio as nsaudio

    decode_fr: list[float] = []      # per-frame decode times
    frame_times: list[float] = []    # frame timestamps (unique-vs-total redundancy)
    gpu_clip: list[float] = []       # predict_hidden_states per clip (proc+H2D+fwd+cat)
    fwd_clip: list[float] = []       # the V-JEPA2 model forward ALONE per clip
    agg_clip: list[float] = []
    audio_t: list[float] = []        # _process_wav (one-time-ish)

    _orig_read = nsv._VideoImage.read
    _orig_phs = nsv._HFVideoModel.predict_hidden_states  # OUTERMOST (post-bf16-patch)
    _orig_agg = nsbase.HuggingFaceMixin._aggregate_tokens
    _orig_wav = nsaudio.HuggingFaceAudio._process_wav

    def _read(self):
        t = time.perf_counter()
        o = _orig_read(self)
        decode_fr.append(time.perf_counter() - t)
        frame_times.append(round(float(getattr(self, "time", 0.0)), 4))
        return o

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _phs(self, images, audio=None):
        # Wrap the actual nn.Module forward ONCE to time the bf16 forward alone.
        m = self.model
        if not getattr(m, "_prof_fwd", False):
            _mf = m.forward

            def _mfw(*a, **k):
                _sync()
                t = time.perf_counter()
                r = _mf(*a, **k)
                _sync()
                fwd_clip.append(time.perf_counter() - t)
                return r

            m.forward = _mfw
            m._prof_fwd = True
        _sync()
        t = time.perf_counter()
        o = _orig_phs(self, images, audio)  # bf16-patched outermost
        _sync()
        gpu_clip.append(time.perf_counter() - t)
        return o

    def _agg(self, latents):
        t = time.perf_counter()
        o = _orig_agg(self, latents)
        agg_clip.append(time.perf_counter() - t)
        return o

    def _wav(self, wav):
        _sync()
        t = time.perf_counter()
        o = _orig_wav(self, wav)
        _sync()
        audio_t.append(time.perf_counter() - t)
        return o

    nsv._VideoImage.read = _read
    nsv._HFVideoModel.predict_hidden_states = _phs
    nsbase.HuggingFaceMixin._aggregate_tokens = _agg
    nsaudio.HuggingFaceAudio._process_wav = _wav

    import traceback

    try:
        # Synthetic 12s 480p clip (testsrc video + silent audio) — no network needed.
        # Decode cost is dominated by moviepy per-frame seek+decode, which is
        # content-independent, so this is representative for the attribution.
        clip = "/tmp/prof_clip.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=854x480:rate=24",
             "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "12",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", clip],
            capture_output=True, check=True,
        )

        from tribescore.inference import load_model, run_inference

        cache = "/tmp/profcache"
        os.makedirs(cache, exist_ok=True)
        t0 = time.perf_counter()
        model = load_model(cache)
        t_load = time.perf_counter() - t0

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        preds, abs_times = run_inference(model, "video", clip, audio_only=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_total = time.perf_counter() - t1
    except Exception:
        return "PIPELINE PROFILE FAILED:\n" + traceback.format_exc()
    finally:
        nsv._VideoImage.read = _orig_read
        nsv._HFVideoModel.predict_hidden_states = _orig_phs
        nsbase.HuggingFaceMixin._aggregate_tokens = _orig_agg
        nsaudio.HuggingFaceAudio._process_wav = _orig_wav

    def steady(xs):
        s = xs[5:] if len(xs) > 6 else xs  # drop warmup clips
        return (sum(s) / len(s) * 1000.0) if s else 0.0

    n_clip = len(gpu_clip)
    decode_ms = (sum(decode_fr) / max(n_clip, 1)) * 1000.0  # 64 frame-reads/clip lumped
    fwd_ms = steady(fwd_clip)                                # bf16 model forward ALONE
    gpu_ms = steady(gpu_clip)                                # proc + H2D + fwd + cat
    proc_cat_ms = max(gpu_ms - fwd_ms, 0.0)                  # processor + H2D + cat
    agg_ms = steady(agg_clip)
    audio_total = sum(audio_t)                               # one-time W2V-BERT
    per_clip_wall = 1000.0 * t_total / max(n_clip, 1)

    # frame redundancy (cache sizing): how many unique frame timestamps vs total reads
    uniq = len(set(frame_times))
    tot_fr = len(frame_times)

    # close the books: video per-clip buckets x n_clip + one-time audio vs total wall
    video_s = (sum(decode_fr) + sum(gpu_clip) + sum(agg_clip))
    gap_s = t_total - video_s - audio_total

    fwd_check = "PASS (bf16 took)" if fwd_ms < 500 else f"FAIL — looks fp32 ({fwd_ms:.0f}ms, expected ~288)"

    out = [
        "### PIPELINE PROFILE v2 (hardened: bf16+TF32, forward measured directly)",
        f"GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | "
        f"model_load={t_load:.1f}s | run_inference TOTAL={t_total:.1f}s | "
        f"clips={n_clip} | TRs={len(abs_times)} | per-clip wall={per_clip_wall:.0f}ms",
        f"SELF-CHECK forward={fwd_ms:.0f}ms -> {fwd_check}",
        "--- steady-state per clip (warmup 5 dropped) ---",
        f"GPU forward (bf16, alone)        : {fwd_ms:7.1f} ms/clip  ({100*fwd_ms/per_clip_wall:.0f}%)",
        f"processor + H2D + cat            : {proc_cat_ms:7.1f} ms/clip  ({100*proc_cat_ms/per_clip_wall:.0f}%)",
        f"decode (64 frames/clip, moviepy) : {decode_ms:7.1f} ms/clip  ({100*decode_ms/per_clip_wall:.0f}%)",
        f"aggregate (_aggregate_tokens)    : {agg_ms:7.1f} ms/clip  ({100*agg_ms/per_clip_wall:.0f}%)",
        "--- one-time ---",
        f"audio W2V-BERT (_process_wav x{len(audio_t)}) : {audio_total:.2f}s total "
        f"({100*audio_total/t_total:.0f}% of run)",
        "--- frame redundancy (cache sizing) ---",
        f"frames: {tot_fr} read / {uniq} unique -> redundancy {tot_fr/max(uniq,1):.1f}x "
        f"(>1 => a frame-prep cache cuts that much decode+resize)",
        "--- close the books ---",
        f"video buckets {video_s:.1f}s + audio {audio_total:.1f}s + GAP {gap_s:.1f}s "
        f"= TOTAL {t_total:.1f}s   (GAP = to-numpy + head + loop + warmup)",
    ]
    return "\n".join(out)


try:
    _gpu_pipe = spaces.GPU(duration=240, size="large")
except Exception:  # pragma: no cover

    def _gpu_pipe(fn):
        return fn


@_gpu_pipe
def profile_pipeline() -> str:
    return _pipeline_core()


try:
    _gpu_probe = spaces.GPU(duration=60, size="large")
except Exception:  # pragma: no cover

    def _gpu_probe(fn):
        return fn


@_gpu_probe
def billing_probe(secs: float = 30.0) -> str:
    """Step-0 sleep test: hold the GPU idle for `secs`, do ZERO GPU ops.

    Run it, then compare your ZeroGPU quota meter before vs after:
      ~secs deducted -> billing = reservation WALL-TIME (idle counts) -> prep-outside WINS.
      ~0s deducted   -> compute-metered -> prep-outside is unnecessary.
    """
    import time

    import torch

    t0 = time.perf_counter()
    time.sleep(float(secs))  # CPU idle, NO GPU op
    held = time.perf_counter() - t0
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return (
        f"billing_probe: held the GPU ({gpu}) for {held:.1f}s with ZERO GPU ops.\n"
        f"Compare your ZeroGPU quota BEFORE vs AFTER this call:\n"
        f"  ~{held:.0f}s deducted -> billed on WALL-TIME (idle counts) -> prep-outside saves quota.\n"
        f"  ~0s deducted -> compute-metered -> prep-outside unnecessary."
    )


@_gpu_pipe  # duration=240, size large
def profile_validate_dedup() -> str:
    """A/B the frame-dedup encode vs baseline on one clip: parity + speedup."""
    import os
    import subprocess
    import time
    import traceback

    import numpy as np
    import torch

    try:
        clip = "/tmp/prof_clip.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=854x480:rate=24",
             "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "12",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", clip],
            capture_output=True, check=True,
        )
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        from tribescore.patches import apply_bf16_video_encode

        apply_bf16_video_encode()
        from tribescore import fast_encode
        from tribescore.inference import load_model, run_inference

        cache = "/tmp/profcache"
        os.makedirs(cache, exist_ok=True)
        model = load_model(cache)

        def _run():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.perf_counter()
            preds, at = run_inference(model, "video", clip, audio_only=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return preds, at, time.perf_counter() - t

        fast_encode.remove_frame_dedup_encode()  # baseline
        p0, a0, dt0 = _run()
        fast_encode.apply_frame_dedup_encode()  # dedup
        p1, a1, dt1 = _run()
        tim = dict(fast_encode.LAST_TIMING)
        fast_encode.remove_frame_dedup_encode()

        diff = float(np.abs(p0 - p1).max())
        scale = float(np.abs(p0).max()) + 1e-9
        at_eq = bool(np.array_equal(a0, a1))
        ok = "PARITY OK (float-noise)" if diff / scale < 1e-3 else "★ MISMATCH — investigate"
        return (
            "### DEDUP PARITY + SPEEDUP\n"
            f"baseline {dt0:.1f}s | dedup {dt1:.1f}s | speedup {dt0/max(dt1,1e-6):.2f}x\n"
            f"preds {p0.shape} vs {p1.shape}\n"
            f"max|Δpreds| = {diff:.3e} (rel {diff/scale:.2e}) -> {ok}\n"
            f"abs_times identical = {at_eq} (keystone)\n"
            f"dedup internals: {tim}  "
            f"(redundancy {tim.get('total_reads',0)}/{tim.get('unique',1)} = "
            f"{tim.get('total_reads',0)/max(tim.get('unique',1),1):.1f}x)"
        )
    except Exception:
        return "VALIDATE FAILED:\n" + traceback.format_exc()


@_gpu_pipe  # duration=240, size large
def profile_full() -> str:
    """ONE real Fast video Score, broken into EVERY stage to find the bottleneck.

    Hooks the real neuralset path: decode (_VideoImage.read), video forward+proc
    (predict_hidden_states), aggregate, audio (_process_wav); times load_model and
    the whole run_inference; the remainder = head forward + event/WAV build +
    to-numpy + orchestration. Tags CPU (movable outside @spaces.GPU -> quota) vs
    GPU (must stay inside -> billed).
    """
    import os
    import subprocess
    import time
    import traceback

    import torch

    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        from tribescore.patches import apply_bf16_video_encode

        apply_bf16_video_encode()

        from neuralset.extractors import audio as nsaudio
        from neuralset.extractors import base as nsbase
        from neuralset.extractors import video as nsv

        T: dict = {}
        saved = []

        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        def hook(cls, name, key):
            orig = getattr(cls, name)

            def w(self, *a, **k):
                _sync()
                t = time.perf_counter()
                r = orig(self, *a, **k)
                _sync()
                T[key] = T.get(key, 0.0) + (time.perf_counter() - t)
                return r

            setattr(cls, name, w)
            saved.append((cls, name, orig))

        hook(nsv._VideoImage, "read", "decode")
        hook(nsv._HFVideoModel, "predict_hidden_states", "v_forward")  # proc+H2D+fwd+cat
        hook(nsbase.HuggingFaceMixin, "_aggregate_tokens", "aggregate")
        hook(nsaudio.HuggingFaceAudio, "_process_wav", "audio")

        clip = "/tmp/prof_clip.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=854x480:rate=24",
             "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "12",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", clip],
            capture_output=True, check=True,
        )
        from tribescore.inference import load_model, run_inference

        cache = "/tmp/profcache"
        os.makedirs(cache, exist_ok=True)
        _sync()
        t = time.perf_counter()
        model = load_model(cache)
        _sync()
        T["load_model"] = time.perf_counter() - t

        _sync()
        t = time.perf_counter()
        preds, at = run_inference(model, "video", clip, audio_only=True)
        _sync()
        total = time.perf_counter() - t

        for cls, name, orig in saved:
            setattr(cls, name, orig)

        within = sum(T.get(s, 0) for s in ("decode", "v_forward", "aggregate", "audio"))
        remainder = total - within
        n_tr = len(at)

        def pct(v):
            return f"{100*v/max(total,1e-6):.0f}%"

        out = [
            "### FULL PIPELINE BREAKDOWN (Fast video, 12s, large)",
            f"GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} | "
            f"run_inference TOTAL={total:.1f}s | load_model(head)={T.get('load_model',0):.1f}s | TRs={n_tr}",
            "--- stages within run_inference ---",
            f"decode      (CPU, movable): {T.get('decode',0):6.1f}s ({pct(T.get('decode',0))})",
            f"v_forward   (proc CPU + fwd GPU): {T.get('v_forward',0):6.1f}s ({pct(T.get('v_forward',0))})",
            f"audio       (W2V-BERT, GPU): {T.get('audio',0):6.1f}s ({pct(T.get('audio',0))})",
            f"aggregate   : {T.get('aggregate',0):6.1f}s ({pct(T.get('aggregate',0))})",
            f"REMAINDER   : {remainder:6.1f}s ({pct(remainder)})  "
            f"<- head forward + event/WAV build + V-JEPA2 load + to-numpy + orchestration",
            "--- interpretation ---",
            "CPU/movable-outside-@gpu (quota-savable): decode + the processor part of "
            "v_forward + event/WAV build (in REMAINDER). GPU/must-stay-inside: the V-JEPA2 "
            "& W2V-BERT & head forwards.  If REMAINDER is large -> the bottleneck is the head "
            "or event-build, NOT decode (consistent with the dedup giving no speedup).",
        ]
        return "\n".join(out)
    except Exception:
        return "PROFILE_FULL FAILED:\n" + traceback.format_exc()


@_gpu_pipe  # duration=240, size large
def profile_breakdown(video) -> str:
    """CLEAN coarse-timer breakdown of run_inference on an UPLOADED real clip.

    The fix for profile_full's lie: NO per-frame hooks (those inflated decode via
    1536 perf_counter+sync calls). Hooks only once-per-call boundaries —
    _build_audio_only_events, Data.get_loaders, each extractor's prepare(), the head
    FmriEncoder.forward — so the '11s other' decomposes cleanly. Trims to the first
    45s + adds a silent track (the real clip is silent 1080p HEVC) via -c:v copy so
    the prep is ~free, not a billed re-encode.
    """
    import os
    import subprocess
    import time
    import traceback

    import torch

    try:
        if not video:
            return "no clip uploaded — use the file picker / pass a path via the API"
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        from tribescore.patches import apply_bf16_video_encode

        apply_bf16_video_encode()
        from tribescore import fast_encode
        from tribescore import inference as tsi

        fast_encode.apply_frame_dedup_encode()
        from neuralset.extractors.audio import HuggingFaceAudio
        from neuralset.extractors.video import HuggingFaceVideo
        from tribescore.inference import load_model, run_inference

        clip = "/tmp/break_clip.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-i", video, "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
             "-map", "0:v:0", "-map", "1:a:0", "-t", "45", "-c:v", "copy", "-c:a", "aac",
             "-shortest", clip],
            capture_output=True, check=True,
        )
        meta = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,nb_frames,duration", "-of", "csv=p=0", clip],
            capture_output=True, text=True,
        ).stdout.strip()

        cache = "/tmp/profcache"
        os.makedirs(cache, exist_ok=True)
        model = load_model(cache)

        T: dict = {}
        saved = []

        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        def hook(cls, name, key):
            orig = getattr(cls, name)

            def w(self, *a, **k):
                _sync()
                t0 = time.perf_counter()
                r = orig(self, *a, **k)
                _sync()
                T[key] = T.get(key, 0.0) + (time.perf_counter() - t0)
                return r

            setattr(cls, name, w)
            saved.append((cls, name, orig))

        # module-function hook for the audio_only event build
        _orig_ev = tsi._build_audio_only_events

        def _w_ev(*a, **k):
            t0 = time.perf_counter()
            r = _orig_ev(*a, **k)
            T["events"] = T.get("events", 0.0) + (time.perf_counter() - t0)
            return r

        tsi._build_audio_only_events = _w_ev

        hook(type(model.data), "get_loaders", "get_loaders")
        hook(HuggingFaceVideo, "prepare", "video_extract")
        hook(HuggingFaceAudio, "prepare", "audio_extract")
        hook(type(model._model), "forward", "head")

        _sync()
        t = time.perf_counter()
        preds, at = run_inference(model, "video", clip, audio_only=True)
        _sync()
        total = time.perf_counter() - t

        tsi._build_audio_only_events = _orig_ev
        for cls, name, orig in saved:
            setattr(cls, name, orig)

        vt = dict(fast_encode.LAST_TIMING)
        gl = T.get("get_loaders", 0.0)
        ve = T.get("video_extract", 0.0)
        ae = T.get("audio_extract", 0.0)
        ev = T.get("events", 0.0)
        hd = T.get("head", 0.0)
        assembly = gl - ve - ae
        remainder = total - ev - gl - hd

        def pct(v):
            return f"{100*v/max(total,1e-6):4.0f}%"

        return "\n".join([
            "### CLEAN BREAKDOWN (uploaded real clip, first 45s, large, coarse timers)",
            f"clip[w,h,frames,dur]={meta} | TRs={len(at)} | TOTAL run_inference={total:.1f}s",
            "--- top-level stages of run_inference ---",
            f"events  (_build_audio_only_events, CPU): {ev:7.1f}s ({pct(ev)})",
            f"get_loaders (extract + assemble)       : {gl:7.1f}s ({pct(gl)})",
            f"   |- video_extract (V-JEPA2)          : {ve:7.1f}s ({pct(ve)})",
            f"   |- audio_extract (W2V-BERT)         : {ae:7.1f}s ({pct(ae)})",
            f"   |- assembly + model loads           : {assembly:7.1f}s ({pct(assembly)})",
            f"head loop (FmriEncoder.forward, GPU)   : {hd:7.1f}s ({pct(hd)})",
            f"remainder (abs_times+numpy+orch)       : {remainder:7.1f}s ({pct(remainder)})",
            f"--- video_extract internals (decode/proc/fwd) ---",
            f"   {vt}",
        ])
    except Exception:
        return "BREAKDOWN FAILED:\n" + traceback.format_exc()


def build_profiler():
    """Wire a small profiler panel into the Space UI (temporary step-1 harness)."""
    import gradio as gr

    with gr.Accordion("🔧 ZeroGPU profiler (step-1: compute-bound + large vs xlarge)", open=True):
        out = gr.Textbox(label="profile output", lines=14, interactive=False)
        with gr.Row():
            b_large = gr.Button("Profile · large")
            b_xlarge = gr.Button("Profile · xlarge")
            b_pipe = gr.Button("Profile · pipeline breakdown")
        with gr.Row():
            b_bill = gr.Button("Step-0 · billing sleep-test (idle 30s)")
            b_dedup = gr.Button("Validate · frame-dedup parity+speedup")
            b_full = gr.Button("★ Profile · FULL pipeline breakdown")
        with gr.Row():
            clip_in = gr.File(label="real clip for CLEAN breakdown", type="filepath")
            b_break = gr.Button("★★ Profile · CLEAN breakdown (upload real clip)")
        b_large.click(profile_large, inputs=None, outputs=out, api_name="profile_large")
        b_xlarge.click(profile_xlarge, inputs=None, outputs=out, api_name="profile_xlarge")
        b_pipe.click(profile_pipeline, inputs=None, outputs=out, api_name="profile_pipeline")
        b_bill.click(billing_probe, inputs=None, outputs=out, api_name="billing_probe")
        b_dedup.click(profile_validate_dedup, inputs=None, outputs=out, api_name="validate_dedup")
        b_full.click(profile_full, inputs=None, outputs=out, api_name="profile_full")
        b_break.click(profile_breakdown, inputs=clip_in, outputs=out, api_name="profile_breakdown")
    return out
