# Design: timeline playhead follows video playback

**Branch:** `feat/local-mps-quality-video`  ·  **Status:** for tribe-brain review (target: one-shot implement)

## Goal

When the user clicks a spike, the video seeks there (works today) and a vertical cyan line is drawn at
that second. **Problem:** the line is static — it stays at the clicked second while the video plays on.
**Want:** the line should *follow* `video.currentTime` during playback (a moving playhead), so you can
read which curve values correspond to the frame currently playing.

## Root cause (verified)

`src/tribescore/plotting.py:seek_js` (206–261) binds `plotly_click` → it (a) sets
`video.currentTime = ev.points[0].x` + plays, and (b) draws the playhead **once** via
`Plotly.relayout(gd, {shapes:[{type:'line', xref:'x', yref:'paper', x0:t, x1:t, y0:0, y1:1,
line:{color:'#36D1C4', width:1.5}}]})` (lines 243–249). Nothing updates that shape afterward, so it is
frozen at the clicked second. There is **no `timeupdate`/rAF listener** on the video.

Verified context:
- The playhead is the **only layout shape** — the zeroline is an axis property (`yaxis.zeroline`), and the
  amber hover line is Plotly's built-in **spikeline** (axis `showspikes`), neither is a `layout.shapes`
  entry. So `shapes[0]` unambiguously *is* the playhead → a partial `'shapes[0].x0'` update can't clobber
  anything else.
- `seek_js` is re-run on every Score via `app.py` `.then(fn=None, js=js_seek)` (the click chain, ~line
  540). Gradio re-renders the media HTML each Score, so the `tm-video` **element is new each run**.
- The same JS runs on the Space too (harmless — pure DOM/Plotly, no backend coupling).

## Approach — pure JS, extend `seek_js` (no Python change)

Add a `bindPlayhead()` alongside the existing click bind: attach to the `tm-video` element a loop that
moves `shapes[0]` to `video.currentTime` via a **partial** relayout (`{'shapes[0].x0': t, 'shapes[0].x1': t}`
— shape-only, no trace redraw). The click→seek path is untouched; clicking just sets the starting point the
line then tracks from.

Sketch:
```js
const bindPlayhead = () => {
  const v = document.getElementById(VIDEO_ID);
  const gd = (document.getElementById(PLOT_ID)||{}).querySelector?.('.js-plotly-plot');
  if (!v || !gd || !window.Plotly) return false;
  if (v.dataset.coPlayheadBound === '1') return true;   // idempotent per (fresh) element
  v.dataset.coPlayheadBound = '1';
  let rafId = null, lastX = -1;
  const ensureShape = () => {                            // create shapes[0] if a play precedes any click
    if (!((gd.layout && gd.layout.shapes) || []).length)
      window.Plotly.relayout(gd, {shapes:[{type:'line',xref:'x',yref:'paper',x0:0,x1:0,y0:0,y1:1,
        line:{color:'#36D1C4',width:1.5}}]});
  };
  const move = () => {
    const t = v.currentTime;
    if (Number.isFinite(t) && Math.abs(t - lastX) >= 0.03) {   // ~30Hz cap
      lastX = t; window.Plotly.relayout(gd, {'shapes[0].x0': t, 'shapes[0].x1': t});
    }
  };
  const loop = () => { move(); if (!v.paused && !v.ended) rafId = requestAnimationFrame(loop); };
  v.addEventListener('play',  () => { ensureShape(); cancelAnimationFrame(rafId); rafId = requestAnimationFrame(loop); });
  v.addEventListener('pause', () => { cancelAnimationFrame(rafId); move(); });
  v.addEventListener('seeked', move);                   // scrub-while-paused
  v.addEventListener('ended', () => cancelAnimationFrame(rafId));
  return true;
};
// call bindPlayhead() in the same retry/bind flow as the click bind
```

## Design decisions (my recommendations)

| Decision | Options | Recommend |
|---|---|---|
| Smoothness | native `timeupdate` (~4 Hz, choppy, free) vs rAF loop | **rAF**, gated to relayout only when `currentTime` moved ≥0.03 s (~30 Hz cap) |
| Appearance | only after a click vs auto-appear on play | **Auto on play** (`ensureShape`) so it tracks even with no prior click |
| Update method | full `{shapes:[...]}` replace vs partial `'shapes[0].x0'` | **Partial** (cheap, can't clobber) |
| Lifecycle | — | stop rAF on `pause`/`ended`; one `seeked` update; idempotent per fresh video element |

## Constraints
- Pure JS in `plotting.py:seek_js`; no Python/data change; must NOT break the existing click→seek.
- Re-binds cleanly each Score (new `tm-video` element); no listener stacking / stale rAF loops.
- Harmless on the Space; sole-author commit (Mayank Gupta), no footer; branch `feat/local-mps-quality-video`.

## Questions for tribe-brain
1. **rAF vs timeupdate**: is a ~30 Hz `Plotly.relayout` (shape-only) on this timeline plot smooth and cheap,
   or is there jank/CPU risk that argues for native `timeupdate` (~4 Hz) or a coarser cap?
2. **Partial-update robustness**: with Gradio re-rendering the `gr.Plot` each Score, is `'shapes[0].x0'`
   reliable, and is the `gd` (`.js-plotly-plot`) reference stable across the run? Any need to re-resolve `gd`
   inside the loop?
3. **Race**: any hazard between the click handler's full `{shapes:[...]}` replace and the rAF partial
   `shapes[0].x0` update interleaving (e.g. click during playback)?
4. **Lifecycle/cleanup**: best guard against listener stacking / leaked rAF loops given the per-Score element
   churn — is the `dataset` guard on the fresh element sufficient, or should old loops be explicitly torn down?
5. **Plotly/Gradio gotchas**: does a 30 Hz relayout interfere with hover tooltips, zoom/pan, or fire a
   `plotly_relayout` feedback storm? Does the shape survive user zoom/pan (xref:'x' data coords)?
