# Design: real per-clip progress bar (replace the indeterminate "Scoring…" loader)

**Branch:** `feat/local-mps-quality-video`  ·  **Status:** DECIDED → implemented; pending live smoke-test.

## DECISION (tribe-brain-reviewed, operator-confirmed)

**Option X** — native `gr.Progress` driven by a module-global per-clip sink, CSS-themed amber; card
kept as a themed header. tribe-brain ruled X decisively for the one-shot goal (Y stacks worker-thread×MPS
+ a generator into the Gradio-6.11 visibility-drop chain + an output-arity ripple); operator confirmed X
over the literal in-card mockup. Implemented per tribe-brain's must-fix list:

1. Sink registered LOCAL-only (`mode=="video" and not on_spaces()`); coarse stage calls stay as the Space fallback.
2. `clear_progress_sink()` in a `finally`.
3. `_emit_progress` wrapped try/except, never raises into the encode.
4. ~0.4 s throttle (final clip of each window always emits).
5. Sink subdivides the **0.15→0.70** band only; existing 0.75/0.92/1.0 stitch/metrics/render marks unchanged.
6. Fake indeterminate `co-loading-bar` dropped; native bar CSS-themed amber.

**Double-extract** handled as P-pass monotonic mapping (`P = len(plan_windows(dur))`), not hardcoded 2 — no
reset between windows, degrades to a single sweep if P=1. **SEPARATE next task (NOT here):** the P×
re-extraction is a real 2-3× speed waste; the dedup *output* cache fixes it but is parity-gated
(`TRIBE_DEDUP_CACHE`, ≥150 s multi-window) — do not couple it into this UI change.

**Files:** `src/tribescore/progress.py` (new, pure helpers + tests), `src/tribescore/fast_encode.py`
(sink + per-clip emit), `app.py` (sink wiring in `_score_impl`), `ui.py` (drop fake bar),
`theme.py` (native-progress amber CSS). Tests: `tests/test_progress.py` (7, green).

**Pending:** a single ~5-10 s short-clip live smoke-test on the next app restart (after the long-video
validation runs) to confirm the native bar renders where expected and the amber theming hits Gradio
6.11's actual progress DOM classes — tune CSS selectors only if needed.

---

**Original review brief below (kept for the record):**

## Goal & approved decision

Replace the indeterminate "Scoring…" loader with a **determinate, per-clip progress bar** that
advances smoothly with a live clip counter (operator approved the per-clip mockup, keeping the
existing card styling):

```
        🔬
      Scoring…
  Encoding video · clip 340 / 720 · 47%
  ▰▰▰▰▰▰▰▰▱▱▱▱▱▱▱▱  47%
```

## Root cause (why it looks indeterminate today)

`app.py:_score_impl` already wires `gr.Progress`, but it only fires at coarse boundaries:
`progress(0.05)` → `progress(0.15, "feature extraction + prediction")` → `progress(0.75, "stitching")`.
The **entire ~20-min video encode happens inside the single `0.15 → 0.75` jump** (`_gpu_infer`,
`app.py:420`), so the native bar **sits frozen at 15%** the whole run. On top, `_enter_loading`
reveals a custom card (`ui.py:238-251`) whose `<div class="co-loading-bar">` is a CSS **indeterminate**
animation — visually busy but meaningless. The two combine to read as a spinner.

## Integration points (verified anchors)

- **Per-clip loop** — `src/tribescore/fast_encode.py:378` `for k, ts_list in enumerate(clip_ts):`
  - `k` = clips done; `len(clip_ts)` = total forward passes (= the heavy unit).
  - `LAST_TIMING` already records `unique=len(sorted_keys)`, `total_reads=sum(len(x) for x in clip_ts)` (fast_encode.py:409-417).
  - memguard `check_or_abort()` already runs here every `k % 8` (line 393) — proven safe injection site.
- **`_gpu_infer`** — `app.py:217` runs the whole predict. **Local (MPS): in-process. Space: forked `@spaces.GPU` subprocess** (comment at app.py:259 "the (forked) GPU worker").
- **Loading card** — `ui.py:238-251`, inner `gr.HTML` is currently anonymous; body div has `id="co-progress"`.
- **Result helpers** — `app.py:_enter_loading`(337), `_ok`(347), `_fail`(375), `_reveal_result`(368);
  `_RESULT_OUTPUTS_KEYS`(480) = (empty, loading, error, result_grp, media_html, timeline, summary);
  `score_outputs = result_outputs + [ok_state]` (8 outputs). Note the existing **Gradio 6.11 visibility-drop
  workaround**: result group is revealed in a separate `.then(_reveal_result)` step because Gradio drops a
  container `visible=True` when the same response updates the container's children's values.

## The architecture fork (the decision for tribe-brain)

### Option X — native `gr.Progress`, no threading  *(simplest / most one-shot-reliable)*
- Thread a **module-global progress proxy** into the encode loop: `fast_encode.set_progress_sink(fn)` /
  `clear_progress_sink()`; in the loop call `fn(k+1, len(clip_ts))` (best-effort, never raises).
- In `_score_impl`, the sink is `lambda done,total: progress(0.15 + 0.55*done/total, desc=f"Encoding video · clip {done}/{total}")`.
- Gradio's **native** determinate bar now advances per clip. CSS-theme it to the orange card aesthetic;
  keep the card as a static header (drop the fake `co-loading-bar`).
- **Pros:** no threading, no generator, no output-arity change; uses the documented Gradio mechanism →
  lowest one-shot risk. **Cons:** the moving bar is Gradio's native widget, not literally the card's
  inner `<div>` (visually close with CSS, not identical to the mockup).

### Option Y — generator + worker thread driving the custom card  *(truest to approved mockup)*
- Same module-global sink in fast_encode (updates a thread-safe counter).
- `_score_impl` becomes a **generator**: spawn `_gpu_infer` in a `threading.Thread`; poll the counter;
  `yield` updated `ui.loading_html(frac, desc)` into a **named `loading_body`** gr.HTML every ~0.5 s;
  on thread join, continue to stitch/metrics/figure and yield the final `_ok`/`_fail` tuple.
- ui.py: extract the card's inner HTML into a returned `loading_body` component + add pure
  `loading_html(frac, desc)` builder (determinate fill) + CSS `.co-loading-fill`.
- app.py: add `loading_body` to the outputs; ripple the +1 arity through `_ok`/`_fail`/`_enter_loading`/`_RESULT_OUTPUTS_KEYS`.
- **Pros:** the card's own bar moves exactly as mocked. **Cons:** threading × MPS, a generator feeding the
  same `.then(_reveal_result)` chain that already has a 6.11 visibility quirk, and an output-arity ripple —
  more surfaces to get wrong in one shot.

## Cross-cutting constraints (both options)

1. **Space fork**: the module-global sink updates are invisible across the `@spaces.GPU` fork, so on the
   Space the per-clip bar silently no-ops and we **keep the coarse `progress()` stage calls as the fallback**.
   Per-clip smooth progress is a **local-MPS-only** enhancement. Must NOT break or slow the Space/CUDA path
   (sink is a cheap best-effort callable; no-op when unset).
2. **Double-extract (cache OFF)**: `_dedup_get_data` runs **twice** per Score, so the loop (and the bar)
   fills 0→100% **twice**. Proposed handling: a small pass counter so desc reads "pass 1/2"; OR map both
   passes into the 0.15–0.70 band (monotonic, no reset). Need a ruling.
3. **Never raise into the encode**: sink call wrapped in try/except; a progress failure must not abort scoring.
4. **No app restart during validation**: editing files is safe (running process won't reload); the change is
   verified live only after the long-fast + long-quality validation runs finish and I restart.
5. **Commit**: sole-author Mayank Gupta, no Claude footer.

## My recommendation

Lean **Option X** for a reliable one-shot (it's the mechanism Gradio is built for, zero new failure
surfaces), CSS-themed to match the card — UNLESS tribe-brain judges the visual fidelity of the literal
card bar worth Option Y's added surface area. Either way: keep coarse fallback for the Space, guard the
sink, and pick the double-extract handling.

## Questions for tribe-brain

1. **X vs Y** given the explicit one-shot goal — is Option Y's threading+generator+arity ripple worth the
   exact-card fidelity, or is CSS-themed native (X) the right call?
2. **Double-extract**: pass-aware desc ("pass 1/2") vs monotonic single-fill mapping — which is less confusing?
3. If Y: any Gradio-6.11 hazard in a **generator** that streams `loading_body` updates and then hands a final
   tuple to the existing `.then(_reveal_result)` visibility-workaround chain? (They already fight a 6.11 drop.)
4. If X: will `gr.Progress(desc=...)` called ~hundreds–thousands of times (per clip) over 20 min flood the
   queue / cause perceptible overhead? Should I throttle to every N clips or ~0.5 s?
5. Anything in tribev2's globals/threadlocal state that makes running `_gpu_infer` in a worker thread (Y) unsafe?
