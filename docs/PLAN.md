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
model = TribeModel.from_pretrained(
    "facebook/tribev2",
    cache_folder=CACHE_DIR,           # str; persistent /data if available (see §7)
    device="auto",
    config_update={"data.overlap_trs_train": 20},  # v1: 20 TR (20 s) overlap; see §4
)
events = model.get_events_dataframe(video_path=... | audio_path=... | text_path=...)
preds, segments = model.predict(events, verbose=True)
```

**Verified facts (from `model.py`, `main.py`, `demo_utils.py`, `pl_module.py`, `config.yaml`):**
- `from_pretrained` downloads `config.yaml` + `best.ckpt` via `hf_hub_download`, builds `FmriEncoderModel`, loads `state_dict` (strict), `.to(device).eval()`. It **unconditionally sets `config["average_subjects"]=True`** (demo_utils.py:218) → single "average subject"; we do **not** need to pass it. It then applies `config.update(config_update)` (flat dotted keys on a `ConfDict`), so `config_update` is our hook for loader-time params (e.g. `data.overlap_trs_train`).
- **Backbones load lazily** inside `extractor.prepare()` *during `predict()`*, then are **freed** (`_free_extractor_model`) → ⇒ **building the model at startup does NOT touch gated Llama; only `predict()` does.** (So the Space can deploy + boot before Meta approval; only live inference 403s.)
- **`predict()` returns `(preds, all_segments)`**:
  - `preds`: `np.ndarray` shape **`(n_kept_TRs, 20484)`** — per-TR predicted activity. `20484 = 2 × 10242` fsaverage5 vertices, **left hemisphere `[0:10242]` then right `[10242:20484]`** (`main.py:515` `n_outputs = 2 * FSAVERAGE_SIZES[mesh]`; `utils_fmri.apply` `np.vstack([left, right])`). The shipped ROI helper (§5) already returns indices in this exact space, so we never split hemispheres by hand.
  - `all_segments`: list aligned 1:1 to `preds` rows; each is a per-TR `segment.copy(offset=t, duration=TR)` carrying `.start` (absolute time on the input clock, in seconds), `.offset`, `.duration` (=`TR`=1.0), `.ns_events`. **x-axis source = `round(seg.start)`** (the same attribute `plotting/utils.get_clip` treats as absolute clip time).
- **TR = 1.0 s** (`neuro.frequency=1.0`, `Data.TR = 1/frequency`) → **1 prediction per second**.
- **⚠ POOLER IS LOCKED AT 100 BY THE CHECKPOINT — do NOT change `data.duration_trs`.** The model is rebuilt from `ckpt["model_build_args"]` (`demo_utils.py:229,235`), and `model_build_args["n_output_timesteps"]` was saved at train time (`pl_module.py:48-51`) = **100**. The forward pools to exactly `n_output_timesteps=100` frames per window (`model.py:104` `AdaptiveAvgPool1d(100)`). `predict()` then expands each loader segment into `floor(segment.duration/TR)` per-TR rows and indexes the pooled output with a `[keep]` mask of that length. **If `config.duration_trs ≠ 100`, row-count vs pooled-frame-count misalign → `[keep]` length mismatch / `ValueError`.** Therefore window length is **fixed at 100 s**; the only loader knob we use is `overlap_trs_train`. Empirically corroborated: the shipped demo runs a ~52 s clip with `duration_trs=100` and works — `list_segments` emits a full 100 s segment even when data is shorter, and the out-of-data TRs are dropped by `remove_empty_segments` (so short/partial clips do **not** crash and the real tail is always covered).
- **Native segmentation:** `Data.get_loaders` tiles the timeline via `ns.segments.list_segments(stride=(duration_trs−overlap_trs)·TR, duration=duration_trs·TR)`. For the `"all"` split that `predict()` uses, `overlap_trs = self.overlap_trs_train` (**not** `_val`). Shipped: `duration_trs=100`, `overlap_trs_train=0`, `batch_size=8`, `max_seq_len=1024`, `hidden=1152`, `depth=8`. With our `overlap_trs_train=20` ⇒ **overlapping 100 s windows, stride 80 s**. (`max_seq_len=1024` ⇒ head could take ~1024 TRs, but we never exceed 100 because of the pooler lock.)
- **Forward:** features `(B,L,D,T)` → per-modality `projectors` → concat → transformer (`time_pos_embed`, `max_seq_len=1024`) → `SubjectLayers` predictor → `(B, 20484, T)` → `AdaptiveAvgPool1d(100)`. `predict()` then expands each segment into per-TR rows and drops empty TRs (`remove_empty_segments=True`).

---

## 4. Long-video windowing + stitching (item 2 — the core feature)

**Goal:** score a 4–5 min (240–300 s) input and emit a continuous `(T_seconds, 20484)` timeline at 1 Hz, smooth across window seams, within ZeroGPU budget.

**Chosen approach = (B) configure overlap INTO tribev2's own segmenter (single `predict()`), then absolute-time stitch.** ONE `get_events_dataframe` + ONE `predict()` per Run ⇒ **one backbone load/free cycle** (cheapest GPU path). Rejected: (A) default non-overlapping tiling → hard 100 s seams; (C) external per-window `ffmpeg`-trim + N× `predict()` → N× backbone load/free = prohibitive — **kept as the documented FALLBACK** (see "Fallback" below + §7) if on-Space testing shows the overlap-config or `.start` realignment misbehaving.

**LOCKED parameters (decided — no further tuning needed pre-Space):**
- **`win_s = 100`** — *forced* by the checkpoint pooler (`n_output_timesteps=100`, §3); not a free choice. Also = training `duration_trs` ⇒ in-distribution.
- **`hop_s = 80` → `overlap_trs_train = 20` (20 s / 20 % overlap).** *Justification (HRF + receptive field):* the canonical hemodynamic response peaks ~5 s and returns toward baseline by ~20–24 s after an impulse; the BOLD signal a window predicts at time *t* is driven by stimulus up to ~20 s earlier. A 20 s overlap therefore gives every retained TR a full HRF-length of left-context inside its window and a 20 s blend zone — enough to absorb the worst-case HRF tail while keeping window count low. Tighter overlap (<~10 s) would leave HRF-tail discontinuities at seams; wider overlap costs GPU time for no added context.
- **`warmup_trim = 5` TRs** — drop/down-weight the **leading** 5 TRs of every window except window 0. *Justification:* those TRs have <1 HRF-width of left-context within their window and no `time_pos_embed` history (the encoder + the model's learned 5 s acquisition offset, `neuro.offset=5.0`, both want left-context). Trailing TRs need **no** trim (right-context isn't required to predict a causal BOLD response, and the final window's tail is the unique end-of-video signal — trimming it would drop real data).
- **normalization = per-window, per-vertex z-score over time** (NOT detrend). *Justification:* the training target was z-scored **per sample** + detrended (`config.yaml` `data.neuro.cleaning: standardize=zscore_sample, detrend=true`), so each window's absolute level/scale is arbitrary → naive concat = stepped seams. Per-window z-score cancels exactly that per-sample offset+scale. We avoid linear **detrend** because over a 100 s window it would flatten genuine slow dynamics (e.g. a real sustained-attention ramp); we only claim *relative* dynamics, and the per-metric global z-score in §5 handles the rest.
- **crossfade = trapezoidal overlap-add weighting** (flat=1 core, linear ramps across the 20 s overlap) folded into the stitch step 3c (below) — no separate pass.
- **smoothing** is applied later, per metric curve (§5), not to the 20484-wide timeline.

**Algorithm (`src/tribescore/windowing.py`):** the real path does NOT iterate windows itself — tribev2's segmenter does. The module's job is `plan_windows` (for the progress UI + the fallback) and `stitch` (the real work, pure-numpy, fully unit-testable).
1. `plan_windows(duration_s: float, win_s: int = 100, hop_s: int = 80) -> list[tuple[float,float]]` → window spans `[(0,100),(80,180),…]`; **force the last span to `(max(0.0, duration_s-100), duration_s)`** so the tail is covered by a full-length window; dedup spans whose start repeats. Used for the "Window k/N" progress label and (only) by the fallback path. Pure function — no model.
2. Real inference = ONE call: `preds, segs = model.predict(events)` with `overlap_trs_train=20` already set (§3). Build `abs_times = np.array([round(s.start) for s in segs])` (float seconds). `infer_fn` is the injected seam so tests pass `(preds, abs_times)` directly.
3. `stitch(preds: np.ndarray[(R,20484)], abs_times: np.ndarray[(R,)], *, warmup_trim:int=5, hop_s:int=80, win_s:int=100) -> tuple[np.ndarray[(T,20484)], np.ndarray[(T,)]]`:
   a. **Segment into windows** by detecting `abs_times` resets/contiguous runs (each run = one tribev2 window, ascending then a backward jump marks the next window's start). For each window assign a window-index `w` and an intra-window position `p` (0-based).
   b. **Per-window per-vertex z-score** over that window's rows: `(x - mean_t)/(std_t + 1e-6)`.
   c. **Weight** each row with a **trapezoidal crossfade** (flat=1 in the non-overlap core, linear ramp only across the `overlap = win_s - hop_s = 20` s edges): leading edge ramps `0→1` over the first `overlap` rows, trailing edge ramps `1→0` over the last `overlap` rows, `=1` between. This keeps `Σ weight ≈ 1` everywhere (no low-SNR seam band a whole-window Hann would create). Then apply **warm-up suppression**: for `w > 0`, force `weight = 0` for `p < warmup_trim` (the leading ramp simply starts at `p = warmup_trim`). Window 0 keeps its leading rows (no earlier window covers them). The final window's trailing rows keep `weight = 1` up to the last row (its tail is unique end-of-video signal — do not ramp it down).
   d. **Overlap-add** onto an integer-second grid `t_axis = arange(0, ceil(max(abs_times))+1)`: `timeline[t] = Σ_w weight·zscored_row  /  Σ_w weight` for all rows whose `round(abs_time)==t` (the weighted mean *is* the crossfade; non-overlap seconds have one contributor).
   e. Any grid second with **zero total weight** (e.g. an all-warmup-suppressed gap — shouldn't happen with 20 s overlap > 5 TR trim) is linearly interpolated from neighbors; assert this set is empty in tests.
   f. Returns `(timeline (T,20484), t_axis (T,))`, `T == duration_s` rounded.
4. **Fully unit-testable** with a synthetic `infer_fn` emitting overlapping windows of numpy ramps/sines at known `abs_times`: assert output shape `== (T,20484)`, `t_axis` strictly increasing 0..T-1, **seam continuity** (max |Δ| between adjacent stitched seconds across a known seam < tol), warm-up rows excluded, and duplicate seconds averaged. No model, no torch.

**Fallback (only if on-Space validation fails approach B):** `infer_window(clip_path)` = `ffmpeg`-trim the source to each `plan_windows` span (each exactly 100 s; last = `[D-100, D]`) → `get_events_dataframe` → `predict` → rows tagged `abs_time = window_start + p`. One `@spaces.GPU` per window (bounded duration, natural progress). Correct by construction (deterministic x-axis, no `list_segments`/`.start` assumptions, no overlap-dedup) at the cost of N× backbone loads + N× ASR. `stitch` is reused unchanged.

---

## 5. Brain-activity → metric curves (item 3)

tribev2 outputs **brain activity, not metrics** — we derive named curves by ROI reduction over fsaverage5, grounded in the neuroforecasting literature.

**ATLAS — DECIDED: HCP-MMP1 (Glasser 2016), via tribev2's OWN shipped helper — NOT nilearn Yeo/Schaefer.** `tribev2/utils.py` ships `get_hcp_labels()`, `get_hcp_roi_indices(rois, hemi="both", mesh="fsaverage5")`, `summarize_by_roi()`. `get_hcp_roi_indices` returns vertex indices **already in the 20484 output index space** (right hemi offset +10242 applied), strips the `L_`/`R_` prefix + `_ROI` suffix (so keys are bare Glasser names like `FEF`, `LIPv`, `TPOJ1`, `p32`, `10r`), pools both hemispheres under one key, and supports wildcards (`"IP*"`, `"*PFm"`). This removes the entire nilearn-atlas-fetch + manual vertex-alignment surface and guarantees index agreement with `preds`. (The demo notebook's "Yeo" mention is illustrative; the shipped, runnable code is Glasser. We use `combine=False` — fine-grained, canonical, verifiable names — not the coarse `combine=True` 22-region set, whose label strings are unverifiable offline.)

> **Dependency + first-run cost (handle in T-C/T-G/§7):** `get_hcp_labels` calls `mne.datasets.fetch_hcp_mmp_parcellation(accept=True)` and `mne.datasets.sample.data_path()` → **downloads the HCP-MMP annot + the MNE `sample` dataset (~1.5 GB) once.** Add **`mne`** to `requirements.txt`. The metric masks are **static** (input-independent), so **precompute once at startup and cache to `CACHE_DIR/roi_masks_hcpmmp1_fsaverage5.npz`**; load from cache thereafter. If the download fails, raise a clear startup error (do not silently fall back to wrong indices).

**Pipeline (`src/tribescore/metrics.py`):**
- `build_roi_masks(cache_dir: str, mesh: str = "fsaverage5") -> dict[str, np.ndarray]` — for each metric, `np.unique(np.concatenate([get_hcp_roi_indices(p, hemi="both", mesh=mesh) for p in PARCELS[metric] if p in valid]))`. **Startup guard:** `valid = set(get_hcp_labels(mesh=mesh, combine=False, hemi="both").keys())`; drop any parcel key not in `valid` and log it (atlas naming can vary by mne version). Persist/load via `.npz`.
- `to_metrics(timeline: np.ndarray[(T,20484)], masks: dict[str,np.ndarray]) -> dict[str, np.ndarray[(T,)]]` — per metric, per TR: **mean over that metric's vertices** → `raw[t]`; then **z-score over the full timeline** (`(raw-mean)/(std+1e-6)`) as the analytic series; then **Gaussian smooth σ = 2 s** (`scipy.ndimage.gaussian_filter1d`, truncate≈3 ⇒ ~13-tap kernel; precedent: repo `TemporalSmoothing(kernel_size=9)`). σ=2 s matches BOLD sluggishness while preserving peaks for click-to-seek (§6). Provide an optional **display** scaling to 0–100 via the repo's `robust_normalize(percentile=99)` for the summary strip only; the **z-scored** curve is the canonical/plotted series.
- All metrics share the z-scale ⇒ cross-metric comparison is in **relative** terms only (see caveat).

**v1 metric → exact Glasser parcels** (bare `combine=False` keys; reduction = mean over the union of these parcels' vertices per TR). Ship as a `PARCELS` dict constant:

| Metric (default ON) | Glasser parcels | Networks / rationale |
|---|---|---|
| **Attention** | `FEF, LIPv, LIPd, VIP, MIP, AIP, IP0, IP1, IP2, TPOJ1, TPOJ2, PGi, PGs, PFm, IFJa, IFJp, p9-46v, a9-46v, 9-46d, 46, 8C, i6-8, s6-8` | Dorsal attention (FEF + IPS complex) + Ventral attention (TPOJ/PGi/PGs/PFm + IFJ) + Frontoparietal control (DLPFC) |
| **Engagement / arousal** | `V1, V2, V3, V4, V3A, V3B, V6, V6A, MT, MST, A1, LBelt, MBelt, PBelt, A4, A5, STSdp, STSvp, STSda, STSva, TE1p, TE2p` | Sensory drive (early visual + motion + early/assoc. auditory) + associative integration (STS) |
| **Virality (proxy)** | `10r, 10v, 10d, 10pp, p32, s32, a24, d32, 25, OFC, pOFC, 11l, 13l, 9m` | vmPFC / mOFC / pgACC / mPFC cortical **value** signal |

| Metric (default OFF, toggleable) | Glasser parcels | Rationale |
|---|---|---|
| **Language / semantic load** | `44, 45, IFSa, STSdp, STSvp, STGa, TE1a, A5, PSL, SFL, 55b` | Core language network (text + audio driven) |
| **Self-relevance / DMN** | `7m, POS2, v23ab, d23ab, 31pv, 31pd, RSC, PCV, 9m, 10r, PGs, PGi` | Default-mode / self & social relevance — secondary sharing cue |

(*Declined:* "valence" and "memorability" — no clean, defensible cortical-ROI mapping on this model; including them would over-claim.)

**Neuroforecasting citations (virality proxy) — pin in README + UI tooltip:** Genevsky & Knutson 2015; **Genevsky, Yoon & Knutson 2017 (J. Neurosci.)** (NAcc + MPFC forecast aggregate market behavior beyond self-report); **Scholz et al. 2017 (PNAS)** and **Baek/Falk "brain-as-predictor"** (value system incl. VMPFC predicts population-level sharing/virality); **Doré et al. 2019**.

> **Honesty caveat (must surface in UI + README):** "Virality" is a **research proxy** from cortical value-region activity — *not* a guarantee of going viral. `facebook/tribev2` is **cortical-only** (no ventral striatum/NAcc — the strongest neuroforecasting node), so the vmPFC/mPFC signal is the validated *complement*, not a substitute. Because the training target was per-sample z-scored + detrended, **only relative temporal dynamics are interpretable — absolute "scores" are meaningless.** The summary strip therefore reports **relative** peaks (z-units / percentile), and every metric is labeled a proxy with its underlying ROI viewable.

---

## 6. Gradio UI + frontend-design (item 4)

Applying the **frontend-design** skill: a distinctive, subject-grounded identity — **not** default Gradio Blocks, and **not** one of the three AI-default looks.

**Design direction — "Cortical Observatory":** a calm, clinical-but-warm scientific instrument that reads a video and shows how the average brain would respond over time.

**Token system:**
- **Color (deep slate + BOLD-colormap accents):** `bg #0E1116`, `panel #161B22`, `ink #E8EDF2`, `dim #8A93A0`, `border #232A33`; **accent (activation hot) `#FFB454` amber**, **counter-accent (cool) `#36D1C4` cyan** — the two ends of an fMRI activation colormap, used with restraint. Metric curves use a colorblind-safe scientific ramp (viridis/cividis sample).
- **Type (3 roles):** display = **Space Grotesk** (wordmark + section headers, sparingly); body = **Inter** (Gradio default chain); **data/timestamps = a tabular-figure mono (JetBrains Mono / IBM Plex Mono)** — *justified*: a timeline needs aligned tabular numerals.
- **Layout:** two zones. Left rail = the **mode-switcher (Video / Audio / Text tabs)** + per-mode input widget + metric toggles + Run + quota banner (Space-only). Right hero = the synchronized readout.

**Signature element (the one memorable thing):** a **synchronized multi-channel metric TIMELINE** — stacked translucent curves (attention / virality / engagement …) on a shared 1 Hz time axis, with a **draggable playhead locked to the video scrubber**; hover → crosshair + per-metric values; **click a peak → seek the media to that moment**. *"Scrub the media, watch the brain; click a spike, jump there."* Built with **Plotly** (hover/zoom/click→seek), custom-styled (not default plot chrome). Below: a summary strip (peak/mean per metric).

> **Known frontend risk + DECIDED fallback (so coders aren't blocked):** Gradio's `gr.Video` exposes **no reliable programmatic seek API**. Plan: render the media via a **custom `gr.HTML` `<video id="tm-video">`** plus a tiny JS handler; Plotly `plotly_click` → JS sets `document.getElementById('tm-video').currentTime = t` (and a draggable playhead line via `relayout`). If the custom-component route proves flaky, **fallback = display `t = NN s` on click + a "Jump" control that remounts the `<video>` with a `#t=NN` media-fragment URL** (universally supported, no JS state). Treat the live-locked playhead as best-effort; the click→timestamp→jump path is the guaranteed floor.

**States (quality floor):**
- **Empty:** inviting panel + "Try a sample clip" CTA + one-line "what this does".
- **Loading:** real per-window progress ("Window 2/4 · extracting V-JEPA2 features…"), not a bare spinner.
- **Error (plain voice, actionable):** e.g. *"This clip is 7 min; the max is 5. Trim it and try again."*; gated case: *"Model access is pending approval — check back soon."*
- **Result:** the timeline + summary.
- Responsive to mobile (zones stack), visible keyboard focus, `prefers-reduced-motion` respected. Boldness spent only on the timeline; everything else quiet.

**Code shape (mirrors qwen-image-editor):** `theme.py` (`gr.themes.Base` + `PALETTE` tokens + CSS string), `ui.py` (per-mode builders returning component dicts, `info=` tooltips), `app.py` wires events. A frontend-design review pass critiques the rendered result before sign-off.

---

## 7. ZeroGPU Space architecture (item 5 — single Space)

- **Hardware:** ZeroGPU (H200 70 GB slice). GPU attaches only inside `@spaces.GPU`; the `spaces` runtime registers CUDA at **startup**, so we **build the model at module import** (downloads 708 MB ckpt only — backbones load lazily at first `predict()`), mirroring cbensimon + qwen eager-startup. The module-level model persists across `@spaces.GPU` calls.
- **`@spaces.GPU` boundary (v1):** approach B = **one decorated `predict()` call per Run** (covers all internal overlapping windows in a single GPU session). Set **`@spaces.GPU(duration=480)`** for v1 (matches cbensimon's reservation for one ~52 s clip; a 5-min input does ~4 internal windows of feature-extraction in that session). **First action in T-H: measure actual GPU-seconds per 60 s of input**, then trim `duration` from data.
- **⚠ ZeroGPU duration ALLOWANCE is the #1 unknown (R1, §10):** `duration=480` worked on **cbensimon's** account; the free ZeroGPU tier caps much lower (~120 s). **If techfreakworm's account caps below what one whole-video `predict()` needs, approach B won't fit a single call → switch to the §4 Fallback (one `@spaces.GPU(duration≈200)` PER window) or the feature-cache split below.** Confirm the account's allowance before committing (T-H, before tuning).
- **GPU-budget fallback ladder:** (1) §4 per-window `ffmpeg` path = N bounded `@spaces.GPU` calls (each loads backbones once). (2) deeper fallback (strategy C) if even that is tight: one bounded call extracts + caches features, then the cheap 8-layer head runs per window across short calls. Implement (1) first; (2) only if needed.
- **Memory:** backbones load→free sequentially (LLaMA ~6.5 GB, V-JEPA2 ~3 GB, W2V-BERT ~2.3 GB, DINOv2 ~1.2 GB); peak ≈ largest single backbone + activations + head (0.7 GB) ≪ 70 GB. **Time, not memory, is the binding constraint.**
- **ASR runtime dependency (whisperx via `uvx` — easy to miss):** `ExtractWordsFromAudio` shells out to **`uvx whisperx … --model large-v3 …`** as a subprocess (`eventstransforms.py:111-135`). The Space image therefore needs **`uv` on PATH** (add to `apt`/`packages` or `pip install uv`), plus first-run network to fetch whisperx + download whisper `large-v3` (~3 GB) and the wav2vec align model. Cache under `CACHE_DIR`. The **`audio_only` interim path skips whisperx entirely** (and Llama) → much faster for pre-approval validation.
- **MNE atlas data:** `mne` required (§5); set `MNE_DATA`/subjects-dir under `CACHE_DIR` and **pre-warm `build_roi_masks` at startup** so the ~1.5 GB `sample` + HCP-MMP annot download happens once and is cached to `.npz`.
- **Gated model handling:** `HF_TOKEN` set as a **Space secret** (token from techfreakworm, Llama license accepted). `transformers`/`huggingface_hub` auto-use it. Set `HF_HUB_ENABLE_HF_TRANSFER="0"` (whisperx/huggingface_hub compat, per cbensimon).
- **Serving:** `spaces`-based Space-only guard for the quota banner; `.queue(default_concurrency_limit=1)` (one heavy GPU task at a time); `.launch(show_error=True, ssr_mode=False)`.
- **Caching:** `CACHE_DIR` = persistent `/data` if writable, else `./cache`; reused for the ckpt, whisper/whisperx, MNE data, and the ROI `.npz`.

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

Exact signatures (coders implement to these; all heavy calls behind the injected `infer_fn` so tests use pure numpy):

- **T-A `inference.py`** —
  - `load_model(cache_dir: str) -> "TribeModel"`: `TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache_dir, device="auto", config_update={"data.overlap_trs_train": 20})`. Module-level singleton; **guarded import** of `tribev2`/`torch` so the file imports locally without them (raise a clear error only if `load_model` is actually called off-Space).
  - `run_inference(model, mode: str, src_path: str, *, audio_only: bool=False) -> tuple[np.ndarray, np.ndarray]`: normal path `events = model.get_events_dataframe(**{f"{mode}_path": src_path})` (mode ∈ `{"video","audio","text"}`). **Interim `audio_only`** (video/audio modes only; skips ASR + Llama): build the one-row event DataFrame exactly as `get_events_dataframe` does — `pd.DataFrame([{"type": "Video"|"Audio", "filepath": src_path, "start": 0, "timeline": "default", "subject": "default"}])` — then `events = get_audio_and_text_events(df, audio_only=True)`. Either way: `preds, segs = model.predict(events)`; return `preds (R,20484)`, `abs_times = np.array([round(s.start) for s in segs], float)`. **Caller wraps this in `@spaces.GPU(duration=480)`.**
  - (Fallback only) `infer_window(model, clip_path: str, window_start: float) -> tuple[np.ndarray,np.ndarray]` per §4 Fallback.
- **T-B `windowing.py`** — `plan_windows(duration_s, win_s=100, hop_s=80)` and `stitch(preds, abs_times, *, warmup_trim=5, hop_s=80, win_s=100)` **exactly per §4** (signatures, z-score, Hann weights, overlap-add, zero-weight assert). **Unit tests with synthetic `infer_fn`** (overlapping ramps/sines) asserting shape, monotone `t_axis`, seam continuity, warm-up exclusion, duplicate averaging. *(core; no model)*
- **T-C `metrics.py`** — `PARCELS: dict[str, list[str]]` constant (§5 tables); `build_roi_masks(cache_dir, mesh="fsaverage5") -> dict[str,np.ndarray]` (validate keys against `get_hcp_labels(...).keys()`, `.npz` cache, startup pre-warm); `to_metrics(timeline, masks) -> dict[str, np.ndarray]` (ROI mean → full-timeline z-score → `gaussian_filter1d(σ=2)`); `summary(curves) -> dict[str, dict]` (peak/mean in z-units + percentile). Synthetic tests (random `timeline`, hand-made `masks`).
- **T-D `plotting.py`** — `timeline_figure(t_axis: np.ndarray, curves: dict[str,np.ndarray], *, selected: list[str]) -> go.Figure`: stacked translucent Plotly traces on a shared 1 Hz axis, crosshair hover, click→seek hook (§6), styled per theme. **Pure function** of `(t_axis, curves)` — synthetic tests assert trace count == len(selected) and x-range == t_axis range.
- **T-E `theme.py` + `ui.py`** — frontend-design "Cortical Observatory" tokens + CSS + 3-mode builders + states (empty/loading/error/result), `info=` tooltips, custom `<video>` HTML per §6.
- **T-F `app.py`** — per mode: input → **validate `0 < duration_s ≤ 300`** (and a friendly note if `< ~10 s`) → `@spaces.GPU` `run_inference` → `stitch` → `to_metrics` → `timeline_figure`; `gr.Progress` "Window k/N" from `plan_windows`; quota banner; `queue(default_concurrency_limit=1)`; `launch(show_error=True, ssr_mode=False)`; eager `load_model` + `build_roi_masks` at import; optional `audio_only` debug toggle (default off).
- **T-G packaging** — `requirements.txt` (`tribev2[plotting] @ git+…`, `torch==2.6.0`, `gradio==6.11.0`, `spaces`, `nilearn`, **`mne`**, `plotly`, `numpy`, `scipy`); ensure **`uv`** in the image (whisperx); env `HF_HUB_ENABLE_HF_TRANSFER=0`, `MNE_DATA`→`CACHE_DIR`; finalize README/NOTICE; set `HF_TOKEN` secret.
- **T-H deploy + validate** — push to the ONE Space; **(1) confirm ZeroGPU duration allowance (R1)**; (2) `audio_only` end-to-end smoke test (no Llama) — exercises windowing/stitch/metrics now; (3) measure GPU-time/60 s; (4) verify `.start`-based x-axis matches the media clock (else switch to §4 Fallback); (5) full end-to-end across all 3 modes once Llama approved.

---

## 10. Risks & mitigations

1. **(R1, highest) ZeroGPU per-call duration ALLOWANCE unknown on techfreakworm's account** → a whole-video `predict()` (approach B) may exceed the account ceiling (free tier ~120 s; cbensimon used 480). **Mitigation:** confirm allowance FIRST on-Space (T-H); if too low, switch to §4 Fallback (per-window `@spaces.GPU`) then the feature-cache split. Do not tune `duration` before this is known.
2. **whisperx-via-`uvx` subprocess + MNE `sample` dataset = hidden deps + heavy first run** → image needs `uv`; first run downloads whisper `large-v3` (~3 GB), wav2vec align model, and MNE `sample` (~1.5 GB). **Mitigation:** `uv` + `mne` in image, cache to `CACHE_DIR`, pre-warm ROI `.npz` at startup, use `audio_only` (skips whisperx + Llama) for interim validation.
3. **`.start`-based x-axis realignment is unverified offline** (neuralset not installable here) → if absolute times don't match the media clock, the timeline x-axis is wrong. **Mitigation:** verify on first Space run (T-H step 4); deterministic §4 Fallback (`window_start + p`) eliminates the assumption entirely.
4. **Llama gate pending Meta** → blocks the **text-feature path** of live inference only; the Space boots without it, and the `audio_only` path validates the full heavy pipeline (V-JEPA2 + DINOv2 + W2V-BERT) end-to-end now. `modality_dropout=0.3` means the model tolerates the missing text modality (degraded but valid).
5. **"Virality" is a proxy / cortical-only / relative-only** → label as research proxy everywhere; state no-NAcc limitation; report **relative** (z-unit) values only; show underlying ROI.
6. **Window seams from overlapping windows** → per-window z-score + Hann overlap-add + leading warm-up suppression + per-curve σ=2 s smoothing.
7. **ASR English-only** (`ExtractWordsFromAudio.language="english"`, hard-coded) → non-English audio yields wrong/empty transcripts and silently degrades the text path (no error). **Mitigation:** README + UI note; consider exposing language later (out of v1 scope).
8. **`duration_trs` must stay 100** (pooler locked by ckpt, §3) → never expose/auto-tune it; changing it crashes `predict()`. Documented; no code path sets it.
9. **ZeroGPU per-user quota** → quota banner + `concurrency=1`; Video heaviest, Text lightest.
10. **Domain shift** → trained on naturalistic movies + English narrative speech (Algonauts2025/CNeuroMod Friends, Lahner2024, Lebel2023, Wen2017); fast-cut UGC / music videos / animation degrade reliability. README methodology note.

---

## 11. Decided defaults (coders build against these now; operator may override at go)

No open *high-level* questions remain — each prior question is resolved to a concrete default with rationale. Items marked *(operator may override at go)* are genuine preference calls that do not block implementation.

1. **Metric set — DECIDED.** Ship **Attention, Engagement/Arousal, Virality (proxy)** ON by default; **Language/semantic-load** and **Self-relevance/DMN** as toggles (default OFF). Exact Glasser parcels in §5. *Declined:* valence, memorability (no defensible cortical-ROI mapping — would over-claim). *(operator may override which curves are shown.)*
2. **Window / hop — DECIDED & LOCKED (not a preference).** `win_s=100` is **forced** by the checkpoint pooler (§3); shorter windows are **unsupported** (crash `predict()`), not merely lower-fidelity. `hop_s=80` / `overlap_trs_train=20` per the HRF justification in §4. No speed-vs-fidelity dial exists here; speed comes from GPU-time tuning (§7), not window length.
3. **Max input length — DECIDED.** Hard cap **300 s (5 min)**; reject longer with the friendly §6 error. Short clips work (≥ ~1 window; `<~10 s` shows an advisory note). *(operator may raise the cap; GPU-time grows ~linearly and re-validates against R1.)*
4. **UI identity — DEFAULT.** "Cortical Observatory" (deep-slate + amber/cyan, §6). *(operator may override at go.)*
5. **Space name — DEFAULT.** `techfreakworm/tribev2-brain-timeline`. *(operator may override at go.)*
6. **Interim validation before Llama — DECIDED.** Wire the `audio_only` path (skips ASR+Llama) as a debug toggle so the full heavy pipeline + windowing + metrics are validated on-Space *before* Meta approval (T-H step 2).

---

## 12. Sequence to first working milestone

1. Operator reviews this plan (+ confirms or overrides the §11 decided defaults).
2. Coders execute T-A…T-G locally (synthetic-tested) — *independent of Llama gate*.
3. Deploy skeleton → Space to the ONE Space; confirm UI boots + model object builds (no Llama needed yet).
4. **On Meta approval:** run end-to-end on the Space, measure GPU-time, tune window/duration, validate the timeline across all 3 modes → first working milestone.

---

## 13. Brain review & sign-off

Adversarial review by `brain` (max rigor + sequential-thinking) against tribev2 source (`demo_utils.py`, `model.py`, `main.py`, `pl_module.py`, `eventstransforms.py`, `utils.py`, `utils_fmri.py`, `tribe_demo.ipynb`) and `fb_config.yaml`. Changes made:

- **§3 — Keystone correction (pooler lock).** Documented that `n_output_timesteps=100` is baked into the checkpoint (`pl_module.py:48-51` → `demo_utils.py:229,235` → `model.py:104`), so **`data.duration_trs` must stay 100** — changing it misaligns `predict()`'s `[keep]` mask vs the pooled 100 frames and raises `ValueError`. Window length is therefore *fixed*, not a tuning choice. Also corrected: `average_subjects=True` is set unconditionally by `from_pretrained` (we don't pass it); the live overlap knob is `overlap_trs_train` (used by the `"all"` split), set via `config_update={"data.overlap_trs_train": 20}`; x-axis source is `round(seg.start)`.
- **§4 — Deleted the "pending brain-spec refinement" caveat by deciding everything.** Locked `win_s=100` (forced), `hop_s=80`/`overlap_trs_train=20` (HRF ≤~20 s justification), `warmup_trim=5` leading-only (left-context justification), per-window per-vertex **z-score** (not detrend; per-sample-zscore artifact justification), and a **trapezoidal crossfade** overlap-add (flat=1 core, 20 s linear ramps; avoids the low-SNR seam a whole-window Hann would cause). Gave exact `plan_windows`/`stitch` signatures + the 6-step stitch algorithm + the partial/short-clip non-crash reasoning (empirically corroborated by the working 52 s demo). Kept the external `ffmpeg` per-window method as an explicit, fully-specified fallback.
- **§5 — Replaced the nilearn/Yeo placeholder with the in-repo HCP-MMP1 (Glasser) atlas.** Pinned `tribev2.utils.get_hcp_roi_indices` (indices already in the 20484 space, rh +10242, wildcard support); exact bare-parcel lists for Attention / Engagement / Virality (+ optional Language, Self-relevance); normalization = full-timeline **z-score**, smoothing = Gaussian **σ=2 s**; corrected citations (added Scholz 2017 PNAS, Genevsky 2017); flagged the `mne` dependency + ~1.5 GB `sample` download + `.npz` startup pre-cache; reaffirmed cortical-only / relative-only interpretation.
- **§6 — Added the `gr.Video`-has-no-seek risk with a decided fallback** (custom `<video>` + Plotly-click JS `currentTime`, or `#t=` media-fragment "Jump").
- **§7 — Reconciled to approach B** (one `@spaces.GPU(duration=480)` per Run), made **R1 (ZeroGPU duration allowance)** explicit with the fallback ladder, and documented the **whisperx-via-`uvx` subprocess** (needs `uv` in the image + ~3 GB model download) and **MNE data** caching.
- **§9 — Exact signatures** for every task (`load_model`, `run_inference` incl. the precise `audio_only` branch, `plan_windows`, `stitch`, `build_roi_masks`, `to_metrics`, `summary`, `timeline_figure`), with the duration-validation and startup-prewarm wired into T-F/T-H.
- **§10 — Expanded risks** to 10, risk-ranked (R1 ZeroGPU allowance highest), adding whisperx/mne first-run cost, `.start` realignment unverified-offline, English-only ASR, `duration_trs` lock, and domain-shift.
- **§11 — Converted all 5 open questions into decided defaults** (metric set; window/hop locked-not-preference; 300 s cap; UI identity; Space name) plus the `audio_only` interim-validation decision; preference-only items marked *(operator may override at go)*.
- **§12 — Reworded** step 1 to "confirms or overrides defaults."

**Residual risks (all have a concrete mitigation in-doc, none block local T-A…T-G):**
1. **R1 — ZeroGPU per-call duration allowance** on techfreakworm's account (free tier ~120 s vs the 480 s approach-B may need). Verify first on-Space (T-H step 1); fallback ladder in §7.
2. **`.start`-based x-axis** unverifiable offline (neuralset not installable). Verify T-H step 4; deterministic §4 fallback removes the assumption.
3. **whisperx/`uvx` + MNE `sample`** heavy first run / image deps. Mitigated by `uv`+`mne` in image, caching, `audio_only` interim.
4. **Validity** of the metric proxies (relative-only, cortical-only) — an inherent model limitation, surfaced honestly in UI/README, not a build blocker.

BRAIN SIGN-OFF v1: docs/PLAN.md is gap-free and implementation-ready.
