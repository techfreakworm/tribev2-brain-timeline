# TRIBE v2 Video Brain-Score — Implementation Plan & Design

**Status:** Plan for operator review. Build is GATED on (a) operator approval of this plan and (b) Meta's approval of the gated LLaMA-3.2-3B access (currently *awaiting review*). The repo is scaffolded; **no model has been or will be run locally**.

**Owners:** `tribe-manager` (lead) + `brain` (decision proxy, max-effort + sequential-thinking) + coding agents (ultra/high).

---

## 0. Hard gates & guardrails (non-negotiable)

1. **No local model execution.** This VPS has no GPU and cannot run tribev2. ALL model execution + testing happens **only on the ZeroGPU HF Space**. Locally we only: write code, run syntax/synthetic-data unit tests, and fetch config/metadata. The inference engine is built around an **injected `infer_fn`** so it unit-tests with pure-numpy stubs.
2. **Exactly ONE Hugging Face Space, ever.** If a 2nd Space, extra GPU, paid tier, or any additional HF resource is ever needed → **STOP and emit a `NEEDS_INPUT`** to the operator first.
3. **Open source.** Public GitHub repo; OUR code under Apache-2.0; a `NOTICE` documents the non-commercial CC-BY-NC model + Llama license.
4. **No Anthropic API keys, anywhere.**

---

## 1. Model identity (CONFIRMED — no repo link needed)

- **Repo:** [`facebook/tribev2`](https://huggingface.co/facebook/tribev2) — official `facebook/` org; 601 likes / 86k downloads; `best.ckpt` (708 MB) + `config.yaml`. License **CC-BY-NC-4.0 (non-commercial)**.
- **Paper:** d'Ascoli, Rapin, Benchetrit, Brookes, Begany, Raugel, Banville, King (2026), *"A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience"* (Meta FAIR / Brain & AI).
- **Code:** `github.com/facebookresearch/tribev2` (pip-installable). **Reference Space:** `cbensimon/tribe-v2-demo` (cbensimon = HF's ZeroGPU engineer — gold-standard pattern).
- **What it is:** a multimodal **fMRI brain encoder**. Inputs = naturalistic stimuli (video/audio/text). Output = **predicted cortical brain activity** of the *average subject* on the **fsaverage5** mesh.
- **Backbones (from shipped `config.yaml`):**
  | Modality | Repo | Gated? |
  |---|---|---|
  | Text | `meta-llama/Llama-3.2-3B` (base) | **YES — `manual`, awaiting Meta review** |
  | Video | `facebook/vjepa2-vitg-fpc64-256` | no |
  | Audio | `facebook/w2v-bert-2.0` | no |
  | Image | `facebook/dinov2-large` (on video frames) | no |
- **Variant note:** `facebook/tribev2-subcortical` exists but adds a second model → **excluded** (single-Space rule; cortical model only).

---

## 2. Supported input MODES (exhaustive — only what the model genuinely supports)

From `demo_utils.get_events_dataframe()` (`text_path` | `audio_path` | `video_path`, with `VALID_SUFFIXES`), the model card ("video, audio, text"), and the paper ("vision, audition, and language"), tribev2 has **exactly three input entry points**:

| Mode | Input | Internal feature extractors engaged | Timeline output |
|---|---|---|---|
| 🎬 **Video** *(primary)* | `.mp4 .mov .mkv .avi .webm` | V-JEPA2 (video) + DINOv2 (frames) + W2V-BERT (extracted audio) + LLaMA (ASR word context) | ✅ full multimodal |
| 🔊 **Audio** | `.wav .mp3 .flac .ogg` | W2V-BERT + LLaMA (ASR words) | ✅ |
| 📝 **Text** | textbox / `.txt` | gTTS → W2V-BERT + LLaMA (synthesized speech) | ✅ (over synthesized speech length) |

**There is NO standalone Image mode** — DINOv2 image features are derived *internally from video frames*; the public API exposes no `image_path`. We will **not fake** an image mode. The UI exposes a **3-tab mode-switcher (Video / Audio / Text)**; all three feed the *same* metric-timeline pipeline (`predict()` is modality-agnostic on the events DataFrame).

---

## 3. Inference API surface + exact tensors (item 1)

```python
from tribev2 import TribeModel
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache",
                                   device="auto", config_update={...})  # average_subjects forced True
events = model.get_events_dataframe(video_path=... | audio_path=... | text_path=...)
preds, segments = model.predict(events, verbose=True)
```

**Verified facts (from `model.py`, `main.py`, `demo_utils.py`, `config.yaml`):**
- `from_pretrained` downloads `config.yaml` + `best.ckpt` via `hf_hub_download`, builds `FmriEncoderModel`, loads `state_dict` (strict), `.to(device).eval()`. Sets `average_subjects=True` → single "average subject" (`n_subjects=0`). Accepts `config_update` dict (our hook for overlap/window tuning).
- **Backbones load lazily** inside `extractor.prepare()` *during `predict()`*, then are **freed** (`_free_extractor_model`) → ⇒ **building the model at startup does NOT touch gated Llama; only `predict()` does.** (So the Space can deploy + boot before Meta approval; only live inference 403s.)
- **`predict()` returns `(preds, all_segments)`**:
  - `preds`: `np.ndarray` shape **`(n_kept_TRs, 20484)`** — per-TR predicted activity. `20484 = 2 × 10242` fsaverage5 vertices, **left hemisphere [0:10242] then right [10242:20484]** (`utils_fmri.apply` `np.vstack([left, right])`).
  - `all_segments`: list aligned to `preds` rows; each carries `.offset` / `.duration` → **absolute time on the video clock**.
- **TR = 1.0 s** (`neuro.frequency=1.0`, `Data.TR = 1/frequency`) → **1 prediction per second**.
- **Native segmentation:** `Data.get_loaders` tiles the timeline via `ns.segments.list_segments(stride=(duration_trs−overlap_trs)·TR, duration=duration_trs·TR)`. Shipped: `duration_trs=100`, `overlap_trs=0`, `batch_size=8`, `max_seq_len=1024`, `hidden=1152`, `depth=8`. ⇒ default = **non-overlapping 100 s windows**; head supports up to **1024 TRs (~17 min)** — so the "short clip" limit is a **GPU memory/time** constraint, not a model limit.
- **Forward:** features `(B,L,D,T)` → per-modality `projectors` → concat → transformer (`time_pos_embed`, `max_seq_len=1024`) → `SubjectLayers` predictor → `(B, 20484, T)` → `AdaptiveAvgPool1d(duration_trs)`. `predict()` then expands each segment into per-TR rows and drops empty TRs.

---

## 4. Long-video windowing + stitching (item 2 — the core feature)

**Goal:** score a 4–5 min (240–300 s) input and emit a continuous `(T_seconds, 20484)` timeline at 1 Hz, smooth across window seams, within ZeroGPU budget.

**Chosen approach = (B) configure overlap INTO tribev2's own segmenter + absolute-time stitch.** (Rejected: (A) default non-overlapping tiling → hard 100 s boundary artifacts; (C) manual per-window `predict()` → reloads backbones each window = prohibitive — kept only as the GPU-budget fallback, see §7.)

**Algorithm (`src/tribescore/windowing.py`):**
1. `plan_windows(duration_s, win_s=100, hop_s=80)` → list of `(start,end)` (defaults: **100 s window = matches training `duration_trs` so the head stays in-distribution; 20 s / 20 % overlap**). Final partial window kept (`stride_drop_incomplete=False`).
2. Run inference (via injected `infer_fn`, real impl wraps `predict()`); collect `(preds, abs_times)` per window.
3. `stitch(window_results)` →
   - place each per-TR row at its **absolute time** (`segment.offset + intra-TR index`);
   - **warm-up trim**: drop first/last ~5 TRs of each interior window (transformer edge);
   - **per-window normalization**: z-score each window's per-vertex timecourse (remove window-level offset) — *or* linear detrend; pick via brain spec;
   - **crossfade**: linear taper-blend overlapping TRs;
   - resample/bin to a uniform 1 Hz `t_axis`.
   - Returns `(timeline: (T_total, 20484), t_axis: (T_total,))`.
4. **Fully unit-testable** with a synthetic `infer_fn` (numpy ramps/sines) — asserts shape, monotone `t_axis`, seam continuity. No model.

> **Pending brain-spec refinement:** exact `win_s`/`hop_s`, normalization (z-score vs detrend), and crossfade shape will be finalized by the `brain` teammate's windowing spec (HRF/temporal-receptive-field reasoning) and validated on-Space. Defaults above are implementation-ready.

---

## 5. Brain-activity → metric curves (item 3)

tribev2 outputs **brain activity, not metrics** — we derive named curves by ROI reduction over fsaverage5, grounded in the neuroforecasting literature. The **official demo notebook references Yeo networks + ROI analysis**, corroborating this approach.

**Pipeline (`src/tribescore/metrics.py`):**
- `roi_masks(mesh="fsaverage5")`: fetch a cortical atlas via **nilearn** (Yeo-7/17 networks; optionally Schaefer-400 for finer ROIs), cached. Map vertex indices → networks for both hemispheres (respecting the left-then-right ordering).
- `to_metrics(timeline)` → `dict[name → curve(T,)]`: per TR, **mean over the ROI's vertices**, then normalize (per-curve min-max or z-score to a 0–100 display scale) and smooth (rolling mean / Gaussian, ~3–5 s).

**v1 metric → ROI mapping (to be finalized + caveated by brain spec):**
| Metric | ROI basis | Literature |
|---|---|---|
| **Attention** | Dorsal + Ventral Attention + Frontoparietal Control networks (Yeo) | attention-network fMRI |
| **Engagement / arousal** | Visual + SomMotor + global sensory/associative drive | sensory-response magnitude |
| **Virality (proxy)** | Default-network medial PFC / vmPFC value regions | Genevsky & Knutson 2015; Falk/Scholz/Baek "brain-as-predictor" of population sharing |

> **Honesty caveat (must surface in UI + README):** "Virality" is a **research proxy** from cortical value-region activity — *not* a guarantee of going viral. `facebook/tribev2` is **cortical-only** (no ventral striatum/NAcc), so we use the vmPFC/mPFC cortical proxy. Labeled as such everywhere.

---

## 6. Gradio UI + frontend-design (item 4)

Applying the **frontend-design** skill: a distinctive, subject-grounded identity — **not** default Gradio Blocks, and **not** one of the three AI-default looks.

**Design direction — "Cortical Observatory":** a calm, clinical-but-warm scientific instrument that reads a video and shows how the average brain would respond over time.

**Token system:**
- **Color (deep slate + BOLD-colormap accents):** `bg #0E1116`, `panel #161B22`, `ink #E8EDF2`, `dim #8A93A0`, `border #232A33`; **accent (activation hot) `#FFB454` amber**, **counter-accent (cool) `#36D1C4` cyan** — the two ends of an fMRI activation colormap, used with restraint. Metric curves use a colorblind-safe scientific ramp (viridis/cividis sample).
- **Type (3 roles):** display = **Space Grotesk** (wordmark + section headers, sparingly); body = **Inter** (Gradio default chain); **data/timestamps = a tabular-figure mono (JetBrains Mono / IBM Plex Mono)** — *justified*: a timeline needs aligned tabular numerals.
- **Layout:** two zones. Left rail = the **mode-switcher (Video / Audio / Text tabs)** + per-mode input widget + metric toggles + Run + quota banner (Space-only). Right hero = the synchronized readout.

**Signature element (the one memorable thing):** a **synchronized multi-channel metric TIMELINE** — stacked translucent curves (attention / virality / engagement …) on a shared 1 Hz time axis, with a **draggable playhead locked to the video scrubber**; hover → crosshair + per-metric values; **click a peak → seek the media to that moment**. *"Scrub the media, watch the brain; click a spike, jump there."* Built with **Plotly** (hover/zoom/click→seek), custom-styled (not default plot chrome). Below: a summary strip (peak/mean per metric).

**States (quality floor):**
- **Empty:** inviting panel + "Try a sample clip" CTA + one-line "what this does".
- **Loading:** real per-window progress ("Window 2/4 · extracting V-JEPA2 features…"), not a bare spinner.
- **Error (plain voice, actionable):** e.g. *"This clip is 7 min; the max is 5. Trim it and try again."*; gated case: *"Model access is pending approval — check back soon."*
- **Result:** the timeline + summary.
- Responsive to mobile (zones stack), visible keyboard focus, `prefers-reduced-motion` respected. Boldness spent only on the timeline; everything else quiet.

**Code shape (mirrors qwen-image-editor):** `theme.py` (`gr.themes.Base` + `PALETTE` tokens + CSS string), `ui.py` (per-mode builders returning component dicts, `info=` tooltips), `app.py` wires events. A frontend-design review pass critiques the rendered result before sign-off.

---

## 7. ZeroGPU Space architecture (item 5 — single Space)

- **Hardware:** ZeroGPU (H200 70 GB slice). GPU attaches only inside `@spaces.GPU`; the `spaces` runtime registers CUDA at **startup**, so we **build the model at module import** (downloads 708 MB ckpt only — backbones load lazily at first `predict()`), mirroring cbensimon + qwen eager-startup.
- **`@spaces.GPU` boundary (v1):** one decorated inference call per Run, `duration` sized generously (start ~300–600 s). **On the first real Space run, empirically measure GPU-seconds per 60 s of video**, then set `duration` + `win_s` from data.
- **GPU-budget fallback (strategy C)** if a 5-min single call exceeds the duration ceiling: one bounded `@spaces.GPU` call extracts + caches features (backbones load once), then the cheap 8-layer head runs per window across short GPU calls with a progress bar.
- **Memory:** backbones load→free sequentially (LLaMA ~6.5 GB, V-JEPA2 ~3 GB, W2V-BERT ~2.3 GB, DINOv2 ~1.2 GB); peak ≈ largest single backbone + activations + head (0.7 GB) ≪ 70 GB. **Time, not memory, is the binding constraint.**
- **Gated model handling:** `HF_TOKEN` set as a **Space secret** (token from techfreakworm, Llama license accepted). `transformers`/`huggingface_hub` auto-use it. Set `HF_HUB_ENABLE_HF_TRANSFER="0"` (whisperx/huggingface_hub compat, per cbensimon).
- **Serving:** `models.on_spaces()` guard for Space-only UI (quota banner); `.queue(default_concurrency_limit=1)` (one heavy GPU task at a time); `.launch(show_error=True, ssr_mode=False)`.
- **Caching:** `cache_folder` on persistent `/data` if available, else `./cache`.

---

## 8. Repo layout + license (item 6) — scaffolded ✅

```
tribe-manager/
├── app.py                      # Gradio entrypoint (mode tabs, eager startup on Spaces)
├── src/tribescore/
│   ├── inference.py            # TribeModel load + per-window predict wrapper (guarded import)
│   ├── windowing.py            # plan_windows + stitch  (synthetic-testable)
│   ├── metrics.py              # ROI masks + brain→metric curves
│   └── plotting.py             # Plotly synchronized timeline
├── theme.py / ui.py            # frontend-design "Cortical Observatory" (to add)
├── tests/                      # pytest, synthetic infer_fn — NO model
├── requirements.txt            # cbensimon-pinned (tribev2[plotting] git+, torch==2.6.0, gradio==6.11.0, spaces, nilearn, plotly, numpy)
├── README.md                   # HF Space front-matter + methodology + attribution
├── LICENSE (Apache-2.0)        # our code
├── NOTICE                      # tribev2 = CC-BY-NC-4.0; Llama-3.2 license; backbone licenses
└── docs/PLAN.md                # this file
```
**License posture:** our code Apache-2.0; deployed demo is **non-commercial** (tribev2 CC-BY-NC) — stated in README + NOTICE + UI footer.

---

## 9. Turnkey task breakdown (item 7 — coding agents execute directly on approval + access)

Dependency-ordered; each unit has a clear interface so they parallelize. **All build locally / test on synthetic data; real validation on the Space.**

- **T-A `inference.py`** — `load_model(config_update)`; `infer_window(events_slice) → (preds (T,20484), abs_times)`. Guarded import (loads without torch locally). Caller wraps in `@spaces.GPU`.
- **T-B `windowing.py`** — `plan_windows`, `stitch` per §4. **Unit tests with synthetic `infer_fn`.** *(core)*
- **T-C `metrics.py`** — `roi_masks(fsaverage5)` (nilearn Yeo/Schaefer, cached) + `to_metrics(timeline) → {name: curve}` per §5. Synthetic tests.
- **T-D `plotting.py`** — Plotly synchronized multi-channel timeline (playhead, hover crosshair, click→seek). Pure function of `(t_axis, curves)`.
- **T-E `theme.py` + `ui.py`** — frontend-design "Cortical Observatory" tokens + CSS + 3-mode builders + states.
- **T-F `app.py`** — wire per mode: input → validate (≤5 min) → `get_events_dataframe` → `@spaces.GPU` windowed inference → `stitch` → `to_metrics` → timeline; per-window progress; quota banner; `queue(1)`; `show_error`; `ssr_mode=False`; eager startup.
- **T-G packaging** — finalize requirements/README/NOTICE; set `HF_TOKEN` secret.
- **T-H deploy + validate** — push to the ONE Space; measure GPU-time/60 s; tune `duration`+`win_s`; full end-to-end validation once Llama approved.

---

## 10. Risks & mitigations

1. **5-min single `@spaces.GPU` exceeds duration ceiling** → strategy C (feature-cache split); measure first.
2. **Llama gate pending Meta** → blocks live inference only; everything else ships + boots now.
3. **"Virality" is a proxy** → label as research proxy everywhere; cortical-only limitation stated.
4. **Independent-window boundary artifacts** → overlap + crossfade + warm-up trim.
5. **ASR (whisper) + gTTS** need CPU/network time in preprocess → show progress; cache.
6. **ZeroGPU per-user quota** → quota banner + concurrency=1; Video is heaviest, Text lightest.

---

## 11. Open questions for operator (async — batch)

1. **Metric set:** attention / virality / engagement confirmed — add valence, arousal, or memorability if the ROI mapping supports them?
2. **Window default:** 100 s window / 80 s hop (in-distribution, slower) vs. shorter windows (faster, OOD risk) — prioritize fidelity or speed?
3. **Max input length:** hard cap at 5 min?
4. **UI identity:** "Cortical Observatory" (deep-slate + amber/cyan) — approve, or prefer a lighter / different direction?
5. **Space name:** e.g. `techfreakworm/tribev2-brain-timeline` (or your preference)?

---

## 12. Sequence to first working milestone

1. Operator reviews this plan (+ answers §11).
2. Coders execute T-A…T-G locally (synthetic-tested) — *independent of Llama gate*.
3. Deploy skeleton → Space to the ONE Space; confirm UI boots + model object builds (no Llama needed yet).
4. **On Meta approval:** run end-to-end on the Space, measure GPU-time, tune window/duration, validate the timeline across all 3 modes → first working milestone.
