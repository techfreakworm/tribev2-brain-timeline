# Design: lock Score button while scoring + persist/download curves

**Branch:** `feat/local-mps-quality-video`  ·  **Status:** for tribe-brain quick review

Two small UX fixes for friction the operator hit: a phantom queued re-run (double-tap on Score, button
not disabled, `concurrency_limit=1` ran the 2nd request after the 1st), and no way to retrieve a finished
run's curves (nothing persisted → had to re-score).

## Feature A — disable the Score button while a Score runs

**Cause:** the run button stays clickable during processing; a 2nd click queues a 2nd full Score that fires
when the 1st finishes. **Fix:** disable on start, re-enable at the end, per tab.

- `_enter_loading` returns one extra update `gr.update(interactive=False)`; each tab wires its own button as
  the extra output: `loading_outputs = [empty, loading, error, result_grp, tab["run_btn"]]`.
- A final `.then(_enable_btn, None, [tab["run_btn"]])` returns `gr.update(interactive=True)`.
- Safe re-enable: `_score_impl` never raises (wraps everything in try/except → `_fail`), so the chain always
  reaches the re-enable `.then`. The 3 tabs (video/audio/text) each get the same wiring.

## Feature B — persist + offer download of the curves

On a successful Score, write the curves so a finished run isn't lost.

- In `_score_impl` after `curves`/`t_axis` are built: write a **CSV** (columns: `t_s`, then one per selected
  metric) + a small **JSON** (metric → values + the summary peaks) to an allowed path. Filename
  `tribescore_<clipstem>_<HHMMSS>.csv` (Python `datetime` is fine here — this is app.py, not a workflow).
- Surface a download **link appended to the existing `summary_html`** (a gr.HTML we already set) — NO new
  component / no output-arity change: `<a href="...">⬇ Download curves (CSV)</a>`.
- Path: write under `tempfile.gettempdir()` (already in `allowed_paths` per run_local) or `CACHE_DIR`.

## Questions for tribe-brain
1. **A:** is `interactive=False/True` toggling via `_enter_loading`+final `.then` the right idiom, or does
   Gradio 6.11 have a cleaner built-in (e.g. `trigger_mode`, or auto-disable during a running event)? Any
   interaction with the existing `.then(_reveal_result)` visibility-drop workaround?
2. **B (the key one):** to make `<a href>` download work, what's the correct Gradio-6.11 file-serving URL
   for a file under `allowed_paths` — `/gradio_api/file=<abs_path>` vs `/file=<abs_path>` vs needing a
   `gr.DownloadButton`/`gr.File` component? Which is most robust (and won't require an output-arity change)?
3. **B:** best dir + cleanup so curve files don't accumulate unbounded (tempdir auto-clears on reboot; do we
   prune, or write to a known `scored_curves/` dir for the operator's cross-reel study)?
4. Any interaction of either change with the `gr.Timer` (Option T) or the dedup cache?
