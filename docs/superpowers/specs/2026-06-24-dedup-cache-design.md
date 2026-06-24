# Design: enable the dedup OUTPUT cache (TRIBE_DEDUP_CACHE) — long-video speedup

**Branch:** `feat/local-mps-quality-video`  ·  **Status:** ✅ REVIEWED — tribe-brain ruling below; ready to implement (parity gate first).

## ✅ FINAL PLAN (tribe-brain ruling — Theory A proven, safe by construction)

- **Theory A PROVEN** from `neuralset/extractors/video.py:237-244`: `_get_timed_arrays` calls `_get_data(events)`
  with the **full-clip event**; per-window slicing is `ta.with_start(...).overlap(start, duration)` *downstream*.
  Theory B (windowed events into `_get_data`) is **false** → the cache key (file+config, no duration) is correct.
- **★ HARDEN THE KEY (must-fix):** add `event.offset` + `event.duration` to the dedup key (mirror exca's
  `item_uid = f"{path}_{offset:.2f}_{duration:.2f}"`). Theory A → still hits (full event = fixed offset/dur); if
  the pattern ever became windowed → keys differ → cache **miss → re-extract → still correct**, never a silent
  wrong slice. Converts "safe because Theory A" into "safe by construction".
- **CLEAR PER-SCORE (must-fix):** clear `_DEDUP_OUT_CACHE` at Score start (local path). The whole speedup is
  *intra-Score* (one extraction reused across that Score's per-window calls); cross-Score reuse is marginal and
  risks stale hits (Gradio recycles temp upload paths → same path / different file). Reject mtime/size keying.
  The existing `len>6: clear()` is a weak net, not a fix.
- **`n_pass=1` when cache on (must-fix):** cache-on ⟹ the sink fires during the ONE prepare extraction; every
  per-window call is a cache **hit** (sink silent) ⟹ one sweep ⟹ `n_pass=1` ⟹ clean 0→100%. Retires the P-pass
  mapping (keep only as a gated cache-OFF fallback if that path stays).
- **Memory: NOT a 105 GB risk** — the cached `TimedArray` is the reduced per-TR embeddings (post
  `_aggregate_tokens`/`_aggregate_layers`), ~MB–tens-of-MB, not raw frames or 40-layer hidden states.
- **Scope: LOCAL-ONLY now**; Space is a separate follow-up (also a real quota win — each per-window
  re-extraction is billed GPU time — but honor the operator's gate and validate Space-side separately).

### Parity gate (must pass before enabling)
Harness `~/Projects/tests/parity_dedup.py`: a ≥2-window clip (≥150 s); run cache-OFF then cache-ON **in one
process, clearing `_DEDUP_OUT_CACHE` between**; compare **raw pre-stitch preds** (the global z-score + crossfade
in stitch can MASK a small diff via normalization); assert **`max|Δ| == 0` exactly** (NOT `< tol` — a tiny
nonzero = MPS bf16 extraction non-determinism, a *separate* finding; flag it, don't widen tolerance); **log
#`_dedup_get_data` calls + cache hit/miss** (hits ≥ P−1 within a Score proves the speedup is *realized*). Run
from `~/Projects/tests` under the 105 GB gate. **Abort if Δ≠0** — do not enable.

### Implementation order
1. Harden the key (offset+duration) + clear-per-Score (local). 2. Build the parity harness. 3. Run the gate
(cache-OFF baseline is slow — use the smallest ≥2-window clip to bound it). 4. If `Δ==0`: enable local
(`TRIBE_DEDUP_CACHE=1` in run_local) + set `n_pass=1` cache-on. 5. Re-verify: long run faster + still
memory-bounded + bar fills 0→100% cleanly.

---


## Goal

Enable `TRIBE_DEDUP_CACHE` (default OFF) so the encode runs **once per Score instead of twice** — killing the
double-extract that made the 156 s long-fast run take ~2.4 hr. Expected ~2× (or more) on long video. **Bonus:**
it also fixes the Option-T progress-bar saturation wart (a single encode pass → a clean 0→100% fill).

**Hard constraint (operator standing rule):** do NOT enable the cache without a multi-window parity gate
(max|Δ| = 0). A wrong cache key would produce *silently wrong* long-video scores.

## How the cache works (verified, `src/tribescore/fast_encode.py`)

`apply_frame_dedup_encode` monkeypatches `HuggingFaceVideo._get_data`. The cache `_DEDUP_OUT_CACHE` (module
dict) stores the extraction output (a `TimedArray`) per event.
- **Key** (line 308-312): `(filepath, model_name, layer_type, str(frequency), num_frames, max_imsize)` — **NO
  duration**. Comment's rationale (296-300): the loader's per-window `__call__` passes a *windowed* event whose
  duration is the window, so a duration key "would miss; one file+config = one full extraction, sliced
  downstream per window."
- **Read** (316-319 fast-path; 357-358 per-event): if key present, `yield _DEDUP_OUT_CACHE[k]`.
- **Write** (462-464): `if len(_DEDUP_OUT_CACHE) > 6: clear()` then `_DEDUP_OUT_CACHE[ckey] = ta`.

## ⚠️ The crux — an unresolved correctness question

`_dedup_get_data` extracts from the event it's given: `video = event.read()`,
`times = np.linspace(0, video.duration, expect_frames+1)[1:]` (line 360-363) — i.e. it extracts **that event's
time range**. So everything hinges on the call pattern:

- **Theory A — called ONCE with the full-clip event** (windows sliced downstream in `stitch`): then key =
  file+config correctly identifies the one full extraction; the double-extract (fit+predict) reuses it. Cache
  is **SAFE**; enabling it just removes the 2nd encode.
- **Theory B — called PER-WINDOW with windowed events**: then window-2's call has the **same key** as
  window-1 (duration omitted) → it HITS window-1's cached output → **window 2 gets window 1's frames →
  SILENTLY WRONG**.

The code comment asserts Theory A, but it is **not proven** — and it determines both correctness AND the
progress-bar pass count (below). The parity gate must settle it empirically.

## Progress-bar interaction (Option T `n_pass`)

The Option-T bar's `n_pass` = `len(plan_windows(dur))` (window count), and `pass_idx` increments once per
`_dedup_get_data` call (per encode-loop restart). So the bar is only calibrated if **#calls == #windows**:
- Cache OFF, Theory A: 2 calls (fit+predict) over the full clip → for a 2-window clip `n_pass=2` *coincidentally*
  matches → bar fills 0→100%. For a 1-window clip: 2 calls but `n_pass=1` → bar saturates at 100% mid-run (the
  observed wart).
- Cache ON, Theory A: **1 call** (predict cached) → `pass_idx` stays 0 → frac maxes at `1/n_pass` = **50% for a
  2-window clip**. ⚠️ NEW BUG: enabling the cache would make the bar stop at 50%.

So **enabling the cache REQUIRES recalibrating `n_pass`**. Cleanest: `n_pass` should equal the actual number of
encode passes, not the window count. Cache ON, Theory A → 1 pass total over all clips → `n_pass=1` → clean
0→100%. (This also retro-fixes the 1-window wart.) The parity test will report the real call count to pin this.

## Plan

1. **Parity harness** (`~/Projects/tests/parity_dedup.py`): run the real inference path on a multi-window clip
   (sintel_long 156 s, 2 windows — or a faster 110 s) TWICE in one process: `TRIBE_DEDUP_CACHE` unset (baseline)
   then set, clearing `_DEDUP_OUT_CACHE` between. Dump the **raw `preds`** (pre-stitch) each time; assert
   `max|preds_on - preds_off| == 0`. Also log the number of `_dedup_get_data` calls + sink restarts per run
   (settles Theory A vs B and the real pass count). Run from `~/Projects/tests`, under the 105 GB gate.
2. **If exact (Theory A confirmed):** enable the cache for the LOCAL path (set `TRIBE_DEDUP_CACHE=1` in
   run_local.py / a local gate — NOT globally on the Space without its own check) and recalibrate `n_pass` to the
   real pass count (likely 1 with cache on).
3. **If not exact (Theory B / key collision):** do NOT enable. Add window-identity to the key (e.g. event start
   time or window index) so per-window extractions don't collide, then re-run parity.
4. Re-verify the long-fast run is faster + still memory-bounded + the bar fills 0→100% cleanly.

## Concerns to weigh

- **Persistence/staleness:** locally the process is long-lived (not a per-Score fork like the Space), so
  `_DEDUP_OUT_CACHE` persists across Scores. Key includes filepath but NOT content — re-scoring a *different*
  file at the *same path* would return a stale hit. Clear per-Score locally, or key on mtime/size?
- **Memory:** caching the full extraction `TimedArray` for a long clip (4456 frames × feat dims) holds a large
  array resident alongside the working copy — does this threaten the 105 GB gate on the longest clips?
- **Space path:** the Space forks per Score (cache fresh), and the parity gate is validated locally; do we
  enable on the Space too, or keep it local-only until separately validated there?

## Questions for tribe-brain
1. **Theory A vs B** — from the neuralset loader/`__call__` flow, is `_get_data` called once (full clip) or
   per-window (windowed events)? If you can't tell from source, the parity harness is the gate — but your read
   would save a cycle.
2. **`n_pass` recalibration** — agree the cache flips the pass count (2→1) and `n_pass` must track the real
   encode-pass count rather than the window count? Best way to derive it (env-aware: 1 if cache on, else 2)?
3. **Parity signal** — compare raw `preds` (pre-stitch) vs the final timeline? Pre-stitch seems strictly more
   sensitive; agree?
4. **Persistence/staleness + memory** — clear the local cache per-Score (lose cross-Score reuse but safe), or
   key on file mtime/size? And is the big cached array a real 105 GB risk on long clips?
5. **Scope** — enable local-only first, Space later; or both once parity passes?
