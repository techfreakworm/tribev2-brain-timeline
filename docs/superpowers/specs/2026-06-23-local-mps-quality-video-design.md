# Local long-video Quality scoring on Apple-silicon (MPS) — Design

**Date:** 2026-06-23
**Status:** Approved (brainstorm) — pending tribe-brain spec review + operator spec review
**Authors:** Mayank Gupta (operator), with tribe-brain (review/brainstorm)

## 1. Problem & goal

The HF ZeroGPU Space cannot score long videos: the `@spaces.GPU` reservation
(480 s requested / 720 s scheduled) is exceeded by the full-multimodal V-JEPA2
forward on a 2–3 min clip, and the task is aborted ("GPU task aborted"). There is
**no HF path for long videos at all**, so the local M5 Max (128 GB unified
memory, MPS) is the *only* way to process them.

The blocker is the **V-JEPA2 ViT-g (`facebook/vjepa2-vitg-fpc64-256`) OOM on MPS**:
64 frames/clip → 8192 tokens; the per-layer attention score/softmax buffers
(`16×8192×8192`, ~4.3 GB each in fp32) accumulate across the 40-layer forward
because the Metal allocator holds them, peaking at ~57 GB fp32 for **one clip** —
which, stacked against a 128 GB machine, crashes the OS near 125 GB.

**Goal:** score a 2–3 min video in **Quality mode** (V-JEPA2 + audio + ASR + Llama)
**locally** via the Gradio UI, with system memory staying **< 90 GB** throughout
(hard ceiling 105 GB), and the timeline rendering **correctly across window seams**.

**Success criteria:**
- A 2–3 min clip scores end-to-end in Quality mode locally; timeline + summary render.
- `vm_stat` system-memory peak stays **< 90 GB** for the whole run.
- The timeline has **no ~22× seam discontinuity** on multi-window (>150 s) clips.
- Local bf16 output A/Bs against the HF/CUDA result within float-noise on the
  z-scored metrics.

## 2. Key reframe (verified against code)

**The OOM is per-clip, not per-video.** The live MPS path
(`src/tribescore/fast_encode.py::_dedup_get_data`) processes **one ~1.5 s TR-clip at
a time** under `torch.inference_mode()`, with `B=1` (no clip batching), no autograd
graph, flushing each clip's output to a small numpy row (~82 KB; a 3-min timeline
≈ ~10 MB). Peak memory is therefore set by a **single 64-frame forward** and is
**independent of total video length**. A 3-min video is ~120 serial clips — slower,
but not more memory.

**One length-dependent caveat:** `uniq_pv` (`fast_encode.py:253`) holds *all* unique
prepped frames for the event on-device through the clip loop — ~2.3 GB fp32 /
~1.1 GB bf16 at 3 min (~16 unique frames/s). Fine to ~3 min under an 80 GB budget;
would need chunking only at ~10 min+. **Flagged, not blocking.**

## 3. Approaches considered

| Approach | Per-clip peak | Parity vs CUDA | Impl risk | Verdict |
|---|---|---|---|---|
| **A+B — per-layer `empty_cache` hook + bf16 autocast** | ~20–30 GB | High (bf16 matmul, fp32 softmax/LN; metrics are z-scored) | Low — hook registration, no math change | **PRIMARY** |
| C — blocked/streaming attention (fp32) | ~10–20 GB | Identical (fp32) *iff* no per-pair attn bias | Medium-high — rewrites attention | **Fallback** |
| B alone — bf16 autocast (current) | ~30–50 GB, watermark-dependent | High | Lowest, but unverified under the gate / fragile | Superseded by A+B |
| D — reduce frames/spatial | lowest | **Unacceptable — feature drift** (head trained on fpc64-256) | — | **Rejected** |

**Why A+B:** bf16 autocast and a per-*clip* `empty_cache` are **already live** on the
MPS path (`fast_encode.py:208`, `:280-283`); eager attention is already injected
(`:92`). A+B is the minimal delta — it adds per-**layer** frees so the O(N²)
attention transients stop accumulating across the 40-layer depth — and is
**numerically identical** (frees only unreferenced buffers; touches no math). bf16
also matches how the model runs on the HF Space, so long local videos stay
*consistent* with existing short-video Space results.

**A's memory floor (~20–30 GB, not lower):** the dedup needs
`output_hidden_states=True` (it concatenates all 40 layers), so the hidden-state
stack (~1.8 GB fp32 / ~0.9 GB bf16) is *referenced* and unfreeable; `empty_cache`
only reclaims the big attention score/softmax transients — exactly what we want.

**Why C is fallback, not primary:** it bounds memory *structurally* and keeps fp32
parity, but rewrites VJEPA2 attention and carries a verification obligation
(confirm the encoder has **no additive/per-pair attention bias** — true for a
standard ViT, but must be checked). Trigger to adopt C: A+B grazes the watermark on
the largest clips, or byte-exact fp32 is required.

## 4. The "dynamic" decision (operator request) — resolved

Operator proposed scaling parallelism to free memory ("50 GB free → parallelise →
faster; 25 GB → less"). **Resolution: keep the adaptive *mechanism*, reject the
*throughput* premise.**

- V-JEPA2 ViT-g is **GPU-compute-bound**; one 8192-token clip **over-saturates** the
  40-core M5 Max GPU (tiles into ~65k threadgroups vs 40 cores). Batched parallel
  forwards add memory with **no throughput gain** (measured on CUDA: B=4 ≈ 231 s vs
  B=1 bf16 ≈ 173 s). → **`B=1` hard-coded**, with a comment so it isn't "optimized."
- Frame-dedup already cut decode 44%→5.5%, so prefetch overlap buys only ~5–10%.
- **Honest envelope:** a 3-min video ≈ **3–6 min of irreducible GPU forward**,
  independent of free RAM. The adaptive policy recovers ~**1.3–1.5×** between
  safest/slowest and roomy/fastest — **not** linear in memory.

So free RAM is used to **stay safe** (the governor) and **shave overhead** (cadence
+ shallow prefetch), never to cut matmul time.

## 5. Component design (phased)

Phases are ordered by dependency: each needs the prior to validate.

### Phase 1 — Runs safely (the enabler)
- **Per-layer `empty_cache` hook:** register a `forward_hook` on each of the 40
  VJEPA2 encoder layers (in `_dedup_get_data`, after `build_video_model` returns the
  model) that calls `torch.mps.synchronize(); torch.mps.empty_cache()`. MPS-only
  (gated on `_mps`); idempotent (guard attr like existing patches); no-op on CUDA.
- **Fixed cadence** `TRIBE_MPS_EMPTY_EVERY` (default **4**) — fire every K-th layer
  to cap sync overhead. K=1 is the safety setting. Keep the existing per-clip free.
- **Keep bf16 autocast** (`fast_encode.py:208`) — do **not** switch to native
  `.to(bf16)` (worse parity; the hook already secures the memory native bf16 saves).
- **Tighten watermark:** `PYTORCH_MPS_HIGH_WATERMARK_RATIO` 0.6 → **0.45** so an
  over-budget clip raises a catchable `RuntimeError` (handled by Trap-A,
  `fast_encode.py:311-320`) far below the system gate. Leave `LOW=0.3`.
- **`B=1` hard-coded** in the dedup path + rationale comment.
- **`vm_stat` safety governor** (new module, e.g. `src/tribescore/memguard.py`):
  parse `active+wired+compressed` (NEVER psutil RSS — Metal *wired* memory is
  invisible to RSS; it reclaims ~25 s after process death). Expose:
  - `headroom_gb()`,
  - `require_headroom(min_gb=25)` — **refuse to start** with a clear message if below,
  - `check_or_abort(min_gb=...)` — call **every ~8 clips** inside the forward loop;
    abort the Run cleanly at ~90 GB system (the OS-crash guard).
- **Target:** ~20–30 GB/clip peak; system peak < 80 GB on a 3-min clip.

### Phase 2 — Quality path on MPS
- whisperx ASR (already int8-patched for CPU/MPS via `mps.py`) on the 2–3 min audio;
  **Llama-3.2-3B** coerced to MPS (existing `mps.py` device coercion). Both wired;
  main work is **verifying the combined memory budget** holds with V-JEPA2 +
  W2V-BERT + Llama resident (they run at different phases, but Llama stays loaded).
- **Fix the audio→ASR-for-video chain.** On HF, a video/quality run showed
  `Extract audio from video events: 1/1` then `Extracting words from audio: 0it` →
  `No transcripts found` → **both audio AND text extractors dropped** → video-only
  scoring despite Quality. If this is general (not video-specific), local Quality
  would also silently drop to video-only. **Investigate the video→audio-chunking→
  ASR chain so Word events reach Llama.** (Audio-*mode* Quality already works
  locally, so the bug is likely in the video→audio extraction/chunking step.)

### Phase 3 — WIN-1 seam fix (correctness for >150 s)
- Normalize `windowing.stitch` crossfade weights over the overlap band so multi-
  window timelines don't show the ~22× seam discontinuity. **2–3 min videos cross
  this every time** — required for a correct long-video timeline. Independent of
  memory; bounding memory makes it *run*, this makes it *correct*.

### Phase 4 — Adaptive speed tiering (optional, deferred)
- Only after Phase 1 runs and is parity-validated. The operator's "use free RAM"
  speed layer, honestly ~1.3–1.5×:
  - **Headroom-tiered cadence `E`:** `H≥40 → off` (rely on watermark, fastest);
    `25–40 → every 8`; `15–25 → every 4`; `<15 → every layer + drop HIGH→0.4`.
  - **Prefetch depth `P` = clamp(H/2GB, 0, 2)** cross-window (decode window N+1 while
    forwarding N). Small win, free.
  - **Mid-run feedback:** every ~8 clips re-read `vm_stat`; if a tier dropped
    (another app grabbed RAM) → tighten `E`, set `P=0`, `empty_cache` now; if below
    the per-clip floor → sleep ~3 s then re-check, else abort cleanly.

## 6. Memory budget (Quality, 3-min clip)

| Item | Resident / peak | Notes |
|---|---|---|
| Baseline system (OS + app + other) | ~20–25 GB | machine-state-dependent; the governor measures it |
| V-JEPA2 per-clip (A+B, bf16) | ~20–30 GB peak | the dominant transient; per-clip, length-independent |
| `uniq_pv` (per-video) | ~1.1 GB bf16 | resident under every clip; grows with length |
| Llama-3.2-3B (MPS, bf16) | ~6–8 GB | resident once loaded |
| W2V-BERT audio | ~few GB | transient |
| whisperx (uvx subprocess, int8) | ~few GB | separate process, ASR phase only |
| **Worst-case concurrent peak** | **~55–65 GB system** | under the 90 GB target with margin |

The `vm_stat` governor is the authority that protects the OS; the watermark only
bounds the MPS pool. Keep both.

## 7. Verification plan

All from `~/Projects/tests` (scripts + venv), **never** the repo root.
1. **Staged memory:** score 15–30 s (single window) → 60 s → 2–3 min on MPS with the
   hook on; record `vm_stat` peak + s/clip at each. Assert peak < 80 GB.
2. **Parity A/B:** same short clip, local bf16 vs the HF/CUDA result; expect
   `max|Δ|` ≈ float-noise on z-scored metrics. (Also empirically closes the bf16-
   parity question + the MPS-autocast-softmax-policy nuance.)
3. **Quality chain:** confirm Word events reach Llama (transcript non-empty, text
   extractor not dropped) on a real speech video.
4. **Seam correctness:** a >150 s clip shows a continuous timeline (no ~22× step).
5. **Gate:** all green → flip `KNOWN_ISSUES.md` Video·Local ⛔→✅.

## 8. Risks & out of scope

- **A+B parity** depends on MPS autocast's softmax/LayerNorm policy (may differ from
  CUDA). Either way it fits; z-scored metrics absorb it. Verify empirically (test 2).
- **Per-layer hook overhead** (~+30–40% worst-case at K=1) — mitigated by cadence;
  this is why Phase 4 tiering exists. Tolerable for "runs at all."
- **C's hidden obligation** (if adopted): verify no additive/per-pair attention bias
  before trusting block-equivalence.
- **`uniq_pv` chunking** deferred past ~10 min.
- **Batched-parallel forwards** ruled out (compute-bound) — do not revisit.
- **Llama license** already accepted (HF_TOKEN present); local uses the cached model.

## 9. Open items for implementation

- Locate the VJEPA2 encoder layer `ModuleList` on the `transformers` model object
  for hook registration (encoder `.layer` / `.blocks`).
- Confirm MPS autocast softmax policy (parity nuance).
- Determine whether the audio→ASR `0it` drop is video-specific or general before
  scoping the Phase-2 fix.
