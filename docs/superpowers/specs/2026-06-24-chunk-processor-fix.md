# Design: chunk the frame `processor()` call to bound the memory transient

**Branch:** `feat/local-mps-quality-video`  ·  **Status:** for tribe-brain feasibility review

## Problem (measured)

Long/HD clips breach the 105 GB gate even cache-off + fast mode. `neeraj_reel_2.mp4` (71 s, 720×1280,
~4000 unique frames) peaked **~96 GB** and the watchdog had to kill it; the in-process memguard never fired
(it checks between clips; this is a within-clip transient). sintel (lower-res) was fine at ~56–61 GB.

**Root cause** — `src/tribescore/fast_encode.py`:
- L400-407: `uniq_frames` = a Python list of ALL unique decoded frames as numpy arrays (at source/`max_imsize`
  resolution). For ~4000 HD frames that's ~11 GB held on CPU.
- **L411-416 (the killer):** `inputs = model.processor(videos=[np.array(uniq_frames)])` processes **ALL unique
  frames in ONE call**. The resize/normalize intermediate scales with `#frames × source_resolution`
  (~4000 × 720×1280×3 fp32 ≈ tens of GB). That single transient + the held `uniq_frames` + the model baseline
  is the ~96 GB peak. (The per-clip forward at L433-436 is already bounded by Approach C + bf16, ~10-20 GB.)
- The comment at L411-412 states this is **per-frame spatial only → identical to per-clip processing** — i.e.
  NO temporal/cross-frame dependency. If true, batching is bit-identical.

## Proposed fix — process `uniq_frames` in CHUNKS, concat into `uniq_pv`

Replace the single processor call (L414-416) with a batched loop:

```python
CHUNK = int(os.environ.get("TRIBE_PROC_CHUNK", "256"))
parts = []
for i in range(0, len(uniq_frames), CHUNK):
    inp = model.processor(videos=[np.array(uniq_frames[i:i + CHUNK])], return_tensors="pt")
    _fix_pixel_values(inp)
    parts.append(inp["pixel_values_videos"][0])   # (chunk, 3, 256, 256)
    del inp
uniq_pv = torch.cat(parts, dim=0).to(dev)          # (U, 3, 256, 256) — same as before
del parts
uniq_frames.clear()                                 # free the ~11 GB CPU list before the forward loop
```

- Bounds the processor transient to `CHUNK × source_res` (~tens of GB → ~GB-scale).
- Frees `uniq_frames` (the ~11 GB held list) before the per-clip forward loop.
- `uniq_pv` (final) is unchanged (~1.5-3 GB), so the forward loop (L430+) is untouched.
- Expected peak: from ~96 GB → ~55-60 GB (the forward-phase floor), so full-res `neeraj_reel_2` fits under 90.
- Opt-out / tunable via `TRIBE_PROC_CHUNK`. MPS + CUDA both benefit; Space unaffected functionally.

## Questions for tribe-brain
1. **Correctness:** is the V-JEPA2 `processor` for this path truly **per-frame spatial** (resize+normalize, no
   cross-frame temporal op, no running stats / batch-norm-over-frames)? i.e. does
   `cat([processor(batch_i)])` == `processor(all)` exactly? The comment asserts it — verify against the
   installed transformers V-JEPA2 image/video processor before we rely on it.
2. **Chunk size:** 256 a good default (memory vs per-call overhead)? Should it adapt to resolution?
3. **`uniq_frames.clear()`** before the forward loop — safe? Anything downstream still references it?
4. **Other held buffers:** after this, is the peak the per-clip forward (~10-20 GB) + `uniq_pv` (~3 GB) +
   baseline (~25 GB)? Any other O(clip-length) buffer I'm missing (the `output`/`ta` array is CPU + small)?
5. **Is chunking the right fix**, or also cap source resolution (a `max_imsize` that the V-JEPA2 256 resize
   makes lossless anyway)? Chunk seems strictly better (no quality change). Agree?
6. Does this interact with the dedup cache (the cached `ta` is post-loop, unchanged) or Option-T progress?

Goal: let the operator score full-res long reels (neeraj_reel_2, 71 s 720×1280) under the 90 GB watchdog
without downscaling. Reason deeply (sequential-thinking); give a decisive feasibility verdict + must-fixes.
