# Known Issues, Gaps & Roadmap

Status of `tribev2-brain-timeline` (TRIBE v2 brain-score Gradio app) as of 2026-06-23.
Tracks open bugs, current limitations, and planned improvements. For architecture
see [`docs/PLAN.md`](docs/PLAN.md); for deploy setup see [`README.md`](README.md).

## Current status

| Mode | Local (Apple-silicon / MPS) | HF ZeroGPU Space |
| --- | --- | --- |
| Audio · Fast | ✅ | ✅ |
| Audio · Quality | ✅ | ✅ |
| Video · Fast | ⛔ (HF-only) | ✅ |
| Video · Quality | ⛔ (HF-only) | ✅ |
| Text | ✅ | ✅ |

All Space modes are billed-verified end-to-end (timeline renders, clean `@gpu`
release). Local audio/text run fully on MPS; **local video is HF-only** (see Gaps).

---

## Known bugs

1. **WIN-1 — multi-window seam step (long clips).** On clips longer than ~150 s,
   `windowing.stitch`'s crossfade weights don't sum to 1 across the warm-up band,
   producing a ~22× discontinuity at window seams. Single-window clips (≲100 s) are
   unaffected. *Fix:* normalize the crossfade weights over the overlap region.
2. **No-input 422 → blank panel.** Clicking *Score* with no input 422s at the Gradio
   queue-join **before** `_score_impl` runs, so the friendly "Add an input first"
   message never fires and the result panel stays blank. *Preferred fix:* make the
   input component nullable so `None` routes through the existing server-side guard
   (not a JS guard).
3. **Too-short clip → degenerate timeline.** `MIN_USEFUL_S = 10` exists but is not
   enforced: a sub-10 s clip scores to a near-flat, meaningless timeline instead of a
   friendly warning. *Fix:* enforce the lower bound for video/audio modes.
4. **Dead metric-guard branch (cosmetic).** `if not keys: _fail("Pick at least one
   brain metric")` is unreachable — an empty metric selection silently defaults to the
   three ON metrics (`METRICS_DEFAULT_ON`). Harmless; the message implies a guard that
   never triggers.

---

## Gaps & limitations

- **Local video is HF-only.** V-JEPA 2 ViT-g produces 8192 tokens/clip; in fp32 on
  MPS the Metal allocator holds every layer's attention buffer through the 40-layer
  forward (~85 GB for one clip) → OOMs a 128 GB Mac. Audio/text are light (no V-JEPA2)
  and run locally; video scoring is only available on the Space. A bounded local fix
  is scoped under *Future → Local MPS video*.
- **`PREWARM_QUALITY` is a Space variable, not in the repo.** Without it, the first
  Quality Score downloads Llama-3.2-3B + whisperx **inside** the billed `@gpu` call.
  Set `PREWARM_QUALITY=1` (Settings → Variables) so the Quality stack pre-caches during
  the un-billed container startup. See README step 5.
- **~14 GB hub copy at every Space cold start.** The writable-cache fix copies the
  baked HF cache to a writable dir before `import gradio` (so gated runtime downloads
  don't `EACCES`). Free, but adds ~1–2 min to cold starts. A build-time alternative is
  under Future.
- **Frame-dedup multi-window parity not re-validated (long video).** Frame-dedup
  (`TRIBE_DEDUP`, **ON** by default — opt out with `TRIBE_NO_DEDUP=1`) is GPU-validated
  numerically exact (max|Δ|=0) and confirmed on the video/quality billed run — but on a
  single-window (15 s) clip. A dedicated **≥150 s multi-window A/B parity** check
  (`profile_validate_dedup`, removed with the profiler harness) has not been re-run, so
  the long-video case isn't formally closed. The dedup-output-cache (`TRIBE_DEDUP_CACHE`)
  is **OFF** by default and stays off pending that check. For guaranteed native-path
  correctness on long videos, set `TRIBE_NO_DEDUP=1` (slower / more quota, no dedup
  indexing risk).
- **Absolute scores are not interpretable.** The model target was per-sample z-scored,
  so only **relative temporal dynamics** are meaningful (already stated in the UI).
- **Virality is a cortical-only research proxy** (no ventral striatum / NAcc) — a
  vmPFC/mPFC complement, not a guarantee.
- **Text mode needs network (gTTS).** Text is synthesized to speech via Google TTS;
  offline/air-gapped use would fail the text path.

---

## Future improvements

### Performance (HF / ZeroGPU)
- **Prep-outside-decorator (biggest win, not yet done).** ZeroGPU bills the entire
  `@spaces.GPU` wall-time, and the GPU sits idle ~44% of a per-clip during CPU prep
  (video decode, whisperx ASR). Moving that CPU work outside the decorator is estimated
  at ~3.3× more Scores/day.
- **Build-time `HF_HOME`.** Set `HF_HOME=/data/.huggingface` at build (needs persistent
  storage) so `preload_from_hub` bakes into a writable location — eliminates the ~14 GB
  startup copy entirely.
- **`large` vs `xlarge` tier.** `xlarge` (4g.96gb, 188 SMs) is ~1.8× faster than `large`
  (2g.48gb, 94 SMs) but costs 2× quota; under prep-outside it becomes ~quota-neutral.
  Peak VRAM is only ~13 GB, so the forward is compute-bound, not VRAM-bound.
- **AOT compile / fp8 (Blackwell sm_120).** Parked; potential further encode speedup.

### Local MPS video — "attempt E" (needs operator go-ahead; RAM-gated)
- Register a `forward_hook` on each V-JEPA 2 encoder layer that calls
  `torch.mps.synchronize()` + `torch.mps.empty_cache()` to free unreferenced attention
  buffers **during** the forward, bounding peak to ~48 GB. Numerically identical (frees
  only unreferenced memory). Would make short clips (15–30 s) scorable locally for
  offline dev/debug **without** burning ZeroGPU quota; long clips stay slow → use the
  Space. **Not run yet** — requires executing the local MPS video pipeline, which is
  gated off by current RAM constraints.
- Secondary: lower the MPS HIGH watermark `0.6 → ~0.4` once memory is reliably on-MPS,
  so an over-budget clip raises a catchable `RuntimeError` instead of stalling. Low
  urgency (attempt E would already bound memory).

### UX / polish
- Enforce `MIN_USEFUL_S` with a friendly sub-10 s warning (bug #3).
- Make inputs nullable to route empty Scores through the server-side guard (bug #2).
- Normalize WIN-1 seam crossfade for long multi-window clips (bug #1).
- Add a "Built with Llama" attribution (Llama license requirement).
- Swap the animated Sintel sample for a more representative clip.
- Remove the now-unwired `profiler.py` (dead code after the UI harness removal) or keep
  it as an internal dev tool.
- Cross-metric z-axis framing consistency in the timeline plot.

---

## Resolved this cycle (for context)

- **Quality-on-HF EACCES.** Gated Llama / whisperx-xet couldn't create new repo dirs in
  the read-only baked hub. Fixed by redirecting `HF_HOME` to a writable copy of the
  baked cache **before `import gradio`** (which freezes the HF cache constants at import).
- **Gradio 6.11 container-visibility drop.** A container's `visible=True` was dropped
  when the same response updated its children's values; fixed via a standalone `.then()`
  reveal step gated on a success `gr.State`.
- **whisperx float16-on-CPU crash (local).** CTranslate2 has no efficient fp16 on a
  CPU/MPS host; rewritten to `int8` via a subprocess-level monkeypatch (`mps.py`).
- **Quality stack downloaded inside the billed `@gpu` call.** Now pre-warmed at the
  un-billed container startup via `PREWARM_QUALITY=1`.
- **Text mode had no media preview.** Now renders the synthesized gTTS speech the model
  scored as a seekable audio player (`id="tm-video"`), like Audio mode.
- **Profiler harness removed** from the UI (`/profile_*` endpoints + accordion) before
  the Space goes public.
