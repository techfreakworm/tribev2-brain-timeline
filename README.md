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
suggested_hardware: zero-a10g
hf_oauth: false
preload_from_hub:
  - facebook/tribev2 best.ckpt,config.yaml
  - facebook/vjepa2-vitg-fpc64-256 *.safetensors,*.json
  - facebook/w2v-bert-2.0 *.safetensors,*.json
---

# TRIBE v2 Video Brain-Score 🧠

Score a **video, audio clip, or text passage** with
[`facebook/tribev2`](https://huggingface.co/facebook/tribev2) and plot derived
**brain-metric** curves — *attention*, *engagement*, *virality* — across the
full timeline.

TRIBE v2 predicts **fMRI-like brain activity** (cortical-surface responses) from
a clip's vision, audio, and text streams. This app runs that model over a long
input via a sliding window, then reduces the high-dimensional brain signal into a
few human-readable curves you can read at a glance and scrub against the media.

> ⚠️ **These curves are derived, heuristic interpretations** of predicted brain
> activity — exploratory, *not* validated neuroscientific or commercial
> measurements. See [Methodology](#methodology) and [`NOTICE`](./NOTICE).

- **Live Space:** Hugging Face ZeroGPU (private) — model execution happens only there.
- **Source:** <https://github.com/techfreakworm/tribev2-brain-timeline> (this repo).

---

## The three modes

A left-rail switcher exposes one mode per genuinely-supported tribev2 modality;
the same windowed pipeline and synchronized readout serve all three.

| Mode | Input | Heavy backbone(s) used |
| --- | --- | --- |
| **🎬 Video** | upload a clip or paste a URL | V-JEPA 2 (vision) + Wav2Vec2-BERT (audio) |
| **🔊 Audio** | upload an audio file | Wav2Vec2-BERT (audio) |
| **📝 Text** | paste text | language/audio path (gated **Llama-3.2-3B**) |

A **⚡ Fast mode** toggle (on by default for video/audio) skips the whisperx
**speech-to-text → Llama text** path, which is the slow, gated part. Turn it off
for a full multimodal run (requires the gated Llama-3.2-3B + an `HF_TOKEN` — see
[Deploying](#deploying-as-a-zerogpu-space)).

> tribev2 also bundles a DINOv2 image backbone, but it is **inactive** in this
> pipeline — there is deliberately no standalone "image" mode.

## How it works

```
input (video URL/file · audio file · text)
   │
   ▼
get_events_dataframe            tribev2: decode, extract audio, optional whisperx
(vision + audio + text rows)    ASR → words / sentences / context
   │
   ▼
windowed @spaces.GPU inference  src/tribescore/{windowing,inference}.py
(100 s windows, 80 s hop)       plan_windows() → per-window predict() on ZeroGPU
   │                            → preds (R, 20484) on fsaverage5, TR = 1 s
   ▼
stitch                          src/tribescore/windowing.py
(per-window z-score + crossfade) stitch(preds, abs_times) → timeline (T, 20484)
   │
   ▼
ROI reduction                   src/tribescore/metrics.py
(Glasser parcels → metrics)     build_roi_masks() + to_metrics(timeline, masks)
   │                            → {Attention, Engagement, Virality, …}
   ▼
timeline figure                 src/tribescore/plotting.py
(metric curves vs time)         timeline_figure(t_axis, curves)
```

TRIBE v2 chunks each clip internally and emits one prediction per fMRI **TR**.
The `windowing` layer is an *outer* sliding window: it processes a long input in
bounded, overlapping GPU calls (100 s windows, 80 s hop) and stitches the
per-window outputs (per-window z-score + trapezoidal crossfade, with a leading
warm-up trim) onto a single shared time axis.

## Methodology

`facebook/tribev2` outputs predicted fMRI activity over the cortical surface as a
`(T, 20484)` array on the **fsaverage5** mesh (`20484 = 2 × 10242` vertices, left
then right hemisphere), one row per fMRI **TR** (TR = 1 s ⇒ 1 Hz). We reduce that
to named curves by averaging activity within **HCP-MMP1 (Glasser 2016)** cortical
parcels — via tribev2's own shipped atlas helper
(`tribev2.utils.get_hcp_roi_indices`), so vertex indices are already aligned to
the model's output space.

| Metric | Cortical parcels (HCP-MMP1) | Basis |
| --- | --- | --- |
| **Attention** | dorsal attention (FEF, IPS: LIP/VIP/MIP/AIP/IP0–2) + ventral attention (TPOJ, PGi/PGs/PFm, IFJ) + frontoparietal control (DLPFC) | attention-network fMRI |
| **Engagement / arousal** | early visual + motion (V1–V4, MT/MST) + auditory (A1, belts, A4/A5) + STS | sensory + associative drive |
| **Virality (proxy)** | vmPFC / mOFC / pgACC / mPFC value regions (10r/10v/10d, p32/s32, a24/d32, OFC, 9m, …) | neuroforecasting value signal — Genevsky & Knutson 2015; Genevsky 2017; Scholz et al. 2017 (PNAS); Doré et al. 2019 |
| *Language / semantic load* (optional) | core language network (44/45, IFSa, STG/STS, PSL, SFL, 55b) | text + audio driven |
| *Self-relevance / DMN* (optional) | default-mode (7m, v23ab/d23ab, 31pv/pd, RSC, PCV, 9m, …) | self / social relevance |

Per TR we take the mean over each metric's parcel vertices, z-score over the full
timeline, and Gaussian-smooth (σ ≈ 2 s). The reduction is implemented in
`src/tribescore/metrics.py` and unit-tested with synthetic masks.

> ⚠️ **"Virality" is a research proxy**, not a guarantee. tribev2 is
> cortical-only (no ventral striatum / NAcc — the strongest neuroforecasting
> node), so this is the validated vmPFC/mPFC *complement*. Because the model's
> training target was per-sample z-scored + detrended, **absolute scores are
> meaningless — only relative temporal dynamics are valid.**

## Performance

The **V-JEPA 2 ViT-g** video encoder dominates runtime (one forward per ~1 s of
video, and it is compute-bound). Two optimisations are applied on the Space, both
**V-JEPA 2-only** and both correctness-safe (every metric is z-scored over time,
so they add only float-order noise — validated to match the fp32 baseline):

- **bf16 autocast** on the encode forward (≈ 2× over fp32; outputs cast back to
  fp32 so downstream numpy/aggregation is unaffected).
- **TF32** matmul/conv.

End-to-end a ~1-minute clip scores in **~140–175 s** on ZeroGPU in Fast mode.

**Ahead-of-time compilation (experimental, currently disabled).** `torch.compile`
/ AOTInductor (the HF *zerogpu-aoti* path) was explored to push the compute-bound
encoder further. On **torch 2.8** it hits a TorchDynamo tracer bug in V-JEPA 2's
`get_position_ids` (`AssertionError: Cannot construct ConstantVariable for value
of type torch.device`) that `fullgraph=False` cannot catch — it raises at forward
time. It is therefore **not enabled** in this build; **bf16 + TF32 is the stable
ceiling**. A future `torch==2.9.1` bump may unblock it (unverified). The next
config-only speed lever is reducing V-JEPA 2 `num_frames` 64→32 (~2×), which
trades temporal fidelity and needs validation.

## Model execution runs on ZeroGPU only

**The model only runs on the Hugging Face ZeroGPU Space.** Inference is performed
inside a function decorated with `@spaces.GPU`, and the weights are downloaded at
runtime on the Space — never vendored in this repo.

The repository is structured so that **everything imports without the heavy stack
present**: `app.py` and the `tribescore` package import cleanly with just `numpy`
(no `torch`, `tribev2`, or `spaces`), the model is loaded lazily behind an
[`on_spaces()`](./src/tribescore/inference.py) guard, and the windowing / metrics
logic is tested with a synthetic, pure-numpy inference function. Do **not**
attempt to run the model on a CPU box.

## Running locally (tests only)

The library + tests run anywhere with just `numpy` and `pytest` — **no model, no
GPU**:

```bash
python -m pytest -q
```

This exercises the windowing stitch and the ROI→metric reduction with a synthetic
inference function and a synthetic atlas. The full app (model + UI) is meant to
run on the Space; see [Deploying](#deploying-as-a-zerogpu-space).

## Deploying as a ZeroGPU Space

This repo **is** a Hugging Face Space — the YAML front-matter at the top of this
README configures it. To deploy your own:

1. Create a **Gradio** Space on **ZeroGPU** (`zero-a10g`) hardware (or *Duplicate*
   the Space).
2. Push this repo's contents. `requirements.txt` installs the runtime stack and
   `packages.txt` adds `ffmpeg`.
3. The **non-gated** backbones (TRIBE v2 checkpoint, V-JEPA 2, Wav2Vec2-BERT) are
   **preloaded at build time** via the `preload_from_hub:` front-matter, for fast
   cold starts.
4. **The text path / full multimodal run needs the gated
   [`meta-llama/Llama-3.2-3B`](https://huggingface.co/meta-llama/Llama-3.2-3B).**
   Accept its licence on the Hub, then add an **`HF_TOKEN`** secret to the Space
   (a token from an account with access). Gated repos cannot be preloaded at build
   (no build-time auth), so they download at runtime into a writable cache.
   Without the token, use **Fast mode** / Video+Audio, which skip the text path.
5. All model weights download at runtime on the Space; nothing is vendored here.

### Requirements

Pinned in [`requirements.txt`](./requirements.txt). Notable constraints:

- **`tribev2[plotting]`** is installed from a
  [fork](https://github.com/techfreakworm/tribev2) that only relaxes upstream's
  `torch`/`torchvision` upper caps so the model installs alongside the
  ZeroGPU-required torch (otherwise functionally identical to upstream).
- **`torch==2.8.0` / `torchvision==0.23.0`** — ZeroGPU requires one of
  `{2.8.0, 2.9.1, 2.10.0, 2.11.0}`; 2.8.0 is closest to upstream's tested 2.6.
- `gradio==6.11.0` (matches the front-matter `sdk_version`), `spaces`, `nilearn`,
  `plotly`, `numpy`, `mne` (HCP-MMP1 atlas), `scipy`, `uv` (`uvx` runs whisperx ASR).

## Project layout

```
.
├── app.py                  # Gradio Blocks entrypoint (3 mode tabs; model lazy, Space-only)
├── theme.py                # "Cortical Observatory" Gradio theme + CSS
├── ui.py                   # per-mode tab builders + results panel
├── requirements.txt        # Space runtime deps (tribev2 fork, torch==2.8.0, gradio, ...)
├── packages.txt            # apt deps (ffmpeg)
├── pyproject.toml          # package metadata + pytest config (src/ layout)
├── src/tribescore/
│   ├── inference.py        # tribev2 wrapper: load_model / run_inference / config (import-safe, GPU-guarded)
│   ├── windowing.py        # plan_windows + stitch (sliding-window inference, pure numpy)
│   ├── metrics.py          # HCP-MMP1 ROI masks → named metric curves
│   ├── plotting.py         # plotly timeline figure + click-to-seek JS
│   └── patches.py          # Space-only speed monkeypatch (bf16 V-JEPA 2 encode)
├── tests/                  # pure-numpy unit tests (windowing, metrics) — no GPU
├── docs/PLAN.md            # implementation plan
├── LICENSE                 # Apache-2.0 (our code)
└── NOTICE                  # third-party model licences + non-commercial notice
```

## Attribution & License

The **source code** in this repository (the Gradio app, the `tribescore` package,
tests, and configuration) is licensed under the [Apache License 2.0](./LICENSE) —
© 2026 *TRIBE v2 Video Brain-Score contributors*.

This app **wraps `facebook/tribev2`** and, transitively, several foundation
models, each under its own license. The TRIBE v2 weights are **CC-BY-NC-4.0
(non-commercial)**, so **this app is a non-commercial research demo**. See
[`NOTICE`](./NOTICE) for the full list (TRIBE v2, Llama 3.2, V-JEPA 2,
Wav2Vec2-BERT, DINOv2) and authoritative links.

- **Model:** [`facebook/tribev2`](https://huggingface.co/facebook/tribev2) ·
  [source](https://github.com/facebookresearch/tribev2)
- **Reference Space:** [`cbensimon/tribe-v2-demo`](https://huggingface.co/spaces/cbensimon/tribe-v2-demo)
