---
title: TRIBE v2 Video Brain-Score
emoji: 🧠
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.11.0
python_version: '3.12'
app_file: app.py
pinned: false
license: apache-2.0
---

# TRIBE v2 Video Brain-Score 🧠

Score a **4-5 minute video** with [`facebook/tribev2`](https://huggingface.co/facebook/tribev2)
and plot derived **brain-metric** curves — *attention*, *virality*,
*engagement* — across the full timeline.

TRIBE v2 predicts **fMRI-like brain activity** (cortical-surface responses)
from a video's vision, audio, and text streams. This app runs that model over
a long clip and reduces the high-dimensional brain signal into a few
human-readable curves you can read at a glance.

> ⚠️ **These curves are derived, heuristic interpretations** of predicted
> brain activity — exploratory, *not* validated neuroscientific or commercial
> measurements. See [Methodology](#methodology) and [`NOTICE`](./NOTICE).

---

## How it works

```
video URL
   │
   ▼
events DataFrame            tribev2: extract audio, transcribe words,
(vision + audio + text)     attach sentence/context  (get_events_dataframe)
   │
   ▼
windowed inference          src/tribescore/windowing.py
(sliding window + stitch)   run_windowed(events, infer_fn, window_s, hop_s)
   │                        → (timeline: (T, n_vertices), time_axis: (T,))
   ▼
ROI reduction               src/tribescore/metrics.py
(brain regions → metrics)   reduce_to_metrics(timeline, atlas)
   │                        → {attention, virality, engagement}
   ▼
timeline figure             src/tribescore/plotting.py
(metric curves vs time)     plot_metric_timeline(time_axis, curves)
```

The model already chunks clips internally (~60 s) and emits one prediction per
fMRI **TR**. The `windowing` layer is an *outer* sliding window: it processes a
long video in bounded, overlapping GPU calls and stitches the per-window
outputs (overlap-averaged, Hann-tapered) onto a single shared time axis.

## Methodology

> **Status: placeholder mapping.** The ROI→metric definitions below are an
> editorial starting point and are marked `TODO` in the code; they will be
> replaced with a concrete, cited parcellation and tuned weights.

`facebook/tribev2` outputs brain activity over the cortical surface as a
`(T, n_vertices)` array on a standard mesh (`fsaverage5`). We map that to
metrics by aggregating activity within **regions of interest (ROIs)** from a
parcellation, then taking a signed weighted combination per metric:

| Metric         | Intuition (placeholder ROI grouping)                                   |
| -------------- | ---------------------------------------------------------------------- |
| **attention**  | fronto-parietal / dorsal-attention + sensory cortex, minus default-mode |
| **virality**   | limbic reward / affective salience + ventral-attention orienting        |
| **engagement** | overall cortical involvement (global activity)                          |

The reduction math (ROI means → weighted combine → smooth → rescale) is fully
implemented and unit-tested with a synthetic atlas; only the *anatomical*
groupings are placeholders.

## Model execution runs on ZeroGPU

**The model only runs on the Hugging Face ZeroGPU Space.** Inference is
performed inside a function decorated with `@spaces.GPU(duration=480)`, and the
weights are downloaded at runtime on the Space — never vendored in this repo.

This repository is structured so that **everything imports without the heavy
stack present**: `app.py` and the `tribescore` package import cleanly with just
`numpy` (no `torch`, `tribev2`, or `spaces`), the model is loaded lazily behind
an [`on_spaces()`](./src/tribescore/inference.py) guard, and the windowing /
metrics logic is tested with a synthetic, pure-numpy `infer_fn`. Do **not**
attempt to run the model on a CPU box.

## Project layout

```
.
├── app.py                       # Gradio Blocks entrypoint (model loaded lazily, Space-only)
├── requirements.txt             # Space runtime deps (tribev2, torch==2.6.0, gradio, ...)
├── pyproject.toml               # package metadata + pytest config (src/ layout)
├── src/tribescore/
│   ├── __init__.py
│   ├── inference.py             # thin tribev2 wrapper (import-safe, GPU-guarded)
│   ├── windowing.py             # sliding-window inference + stitch/normalize
│   ├── metrics.py               # ROI reduction → named metric curves
│   └── plotting.py              # timeline figure (plotly)
├── tests/
│   ├── test_windowing.py        # exercises run_windowed with a synthetic infer_fn
│   └── test_metrics.py          # ROI reduction with a synthetic atlas
├── LICENSE                      # Apache-2.0 (our code)
└── NOTICE                       # third-party model licenses + non-commercial notice
```

## Development

The library + tests run anywhere with just `numpy` and `pytest` — **no model,
no GPU**:

```bash
python -m pytest -q
```

The full app (model + UI) is meant to run on the Space. To install the runtime
stack there, see `requirements.txt`.

## Attribution & License

The **source code** in this repository (the Gradio app, the `tribescore`
package, tests, and configuration) is licensed under the
[Apache License 2.0](./LICENSE) — © 2026 *TRIBE v2 Video Brain-Score
contributors*.

This app **wraps `facebook/tribev2`** and, transitively, several foundation
models, each under its own license. The TRIBE v2 weights are **CC-BY-NC-4.0
(non-commercial)**, so **this app is a non-commercial research demo**. See
[`NOTICE`](./NOTICE) for the full list (TRIBE v2, LLaMA 3.2, V-JEPA 2,
Wav2Vec2-BERT, DINOv2) and authoritative links.

- **Model:** [`facebook/tribev2`](https://huggingface.co/facebook/tribev2) ·
  [source](https://github.com/facebookresearch/tribev2)
- **Reference Space:** [`cbensimon/tribe-v2-demo`](https://huggingface.co/spaces/cbensimon/tribe-v2-demo)
