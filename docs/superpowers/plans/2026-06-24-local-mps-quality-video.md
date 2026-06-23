# Local long-video Quality scoring on MPS — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the tribev2-brain-timeline app score a 2–3 min video in Quality mode **locally on Apple-silicon (MPS)** — which HF ZeroGPU cannot do (480/720 s `@gpu` cap) — by bounding the V-JEPA2 per-clip memory under the OS-crash ceiling and surfacing/curing the long-video correctness issues.

**Architecture:** The V-JEPA2 OOM is **per-clip, not per-video** (one ~1.5 s clip encoded serially, `B=1`, `inference_mode`, output flushed to a tiny numpy row). Bounding one clip's forward → any length runs. We add a **per-layer `empty_cache` forward-hook** on the 40 V-JEPA2 encoder layers (numerically identical — frees only unreferenced O(N²) attention transients), keep the already-live bf16 autocast, tighten the MPS HIGH watermark, and add a **`vm_stat`-based system-memory governor** (Metal *wired* memory is invisible to RSS). Then diagnose-and-cure the long-video Quality issues (ASR `0it`, WIN-1 seam).

**Tech Stack:** Python 3.12, PyTorch 2.8 MPS, `transformers` (VJEPA2), neuralset/tribev2, numpy, pytest. macOS `vm_stat` / `ffprobe`.

## Global Constraints

- **MPS-only changes must be no-ops on CUDA/CPU** — gate every change on `torch.backends.mps.is_available()`; the HF Space (CUDA) behavior must not change.
- **Memory hard gate:** system memory (NOT process RSS) must stay **< 90 GB** during any run; OS crashes near ~125 GB on the 128 GB M5 Max. Guard on `vm_stat` (active + wired + compressed) — **never psutil RSS** (Metal wired memory is invisible to it; it reclaims ~25 s after `kill -9`).
- **`B=1`** for the V-JEPA2 forward — hard-coded; do not batch clips (compute-bound; one 8192-token clip over-saturates the 40-core GPU).
- **Parity:** keep bf16 autocast (matmuls bf16, softmax/LN fp32) — matches how the model runs on the HF Space. Do NOT switch to native `.to(bf16)`.
- **Test location:** unit tests live in the repo `tests/` (pytest). MPS integration/validation scripts that load the model live in **`~/Projects/tests/`** (with venv `~/Projects/tests/tribe-local-venv`), NEVER the repo root or `~/Projects` root.
- **Git authorship:** commits are sole-author (Mayank Gupta); no Claude co-author / "Generated with" footer.
- **Scope:** Phases P1–P3 only. **P4 (adaptive speed tiering) is OUT OF SCOPE** for this plan.
- Spec: `docs/superpowers/specs/2026-06-23-local-mps-quality-video-design.md`.

---

## Task 1: `memguard` — system-memory governor (vm_stat)

**Files:**
- Create: `src/tribescore/memguard.py`
- Test: `tests/test_memguard.py`

**Interfaces:**
- Produces:
  - `_parse_vm_stat(text: str) -> float` — system-used GB from raw `vm_stat` text.
  - `system_used_gb() -> float` — calls `vm_stat`, returns used GB.
  - `headroom_gb(gate_gb: float = 105.0, safety_gb: float = 10.0) -> float` — `gate − used − safety`.
  - `require_headroom(min_gb: float = 25.0) -> None` — raise `MemoryError` if `headroom_gb() < min_gb`.
  - `check_or_abort(abort_gb: float = 90.0) -> None` — raise `MemoryError` if `system_used_gb() > abort_gb`.
  - Module constants `GATE_GB=105.0`, `SAFETY_GB=10.0`, `PER_CLIP_FLOOR_GB=25.0`, `ABORT_GB=90.0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memguard.py
import pytest
from tribescore import memguard

# A representative `vm_stat` block (page size 16384). active=3,000,000 +
# wired=1,000,000 + compressor=500,000 = 4,500,000 pages × 16384 ≈ 73.7 GB.
SAMPLE = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                          100000.
Pages active:                       3000000.
Pages inactive:                      200000.
Pages speculative:                     5000.
Pages throttled:                          0.
Pages wired down:                   1000000.
Pages purgeable:                       1000.
"Translation faults":              999999999.
Pages copy-on-write:                 100000.
Pages zero filled:                 500000000.
Pages reactivated:                  1000000.
Pages purged:                        100000.
File-backed pages:                   400000.
Anonymous pages:                    2800000.
Pages stored in compressor:         1200000.
Pages occupied by compressor:        500000.
"""

def test_parse_vm_stat_sums_active_wired_compressed():
    used = memguard._parse_vm_stat(SAMPLE)
    assert used == pytest.approx(4_500_000 * 16384 / 1e9, rel=1e-6)  # ≈ 73.7 GB

def test_headroom_uses_gate_minus_used_minus_safety(monkeypatch):
    monkeypatch.setattr(memguard, "system_used_gb", lambda: 73.7)
    assert memguard.headroom_gb(gate_gb=105.0, safety_gb=10.0) == pytest.approx(21.3, abs=0.1)

def test_require_headroom_raises_when_below_floor(monkeypatch):
    monkeypatch.setattr(memguard, "system_used_gb", lambda: 90.0)  # headroom = 105-90-10 = 5
    with pytest.raises(MemoryError, match="headroom"):
        memguard.require_headroom(min_gb=25.0)

def test_require_headroom_ok_when_room(monkeypatch):
    monkeypatch.setattr(memguard, "system_used_gb", lambda: 40.0)  # headroom = 55
    memguard.require_headroom(min_gb=25.0)  # no raise

def test_check_or_abort_raises_above_ceiling(monkeypatch):
    monkeypatch.setattr(memguard, "system_used_gb", lambda: 92.0)
    with pytest.raises(MemoryError, match="90"):
        memguard.check_or_abort(abort_gb=90.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/techfreakworm/Projects/llm/tribev2-brain-timeline && python -m pytest tests/test_memguard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tribescore.memguard'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/tribescore/memguard.py
"""System-memory governor for local MPS runs — guards on `vm_stat`, NOT RSS.

MPS/Metal WIRED memory is invisible to process RSS, so psutil-based guards are
blind to it and would let the OS OOM (it crashes near ~125 GB on a 128 GB Mac).
This reads `vm_stat` (active + wired + compressed pages) for true SYSTEM memory
use and gates the local MPS video encode under a safe ceiling. macOS-only; the
callers gate on MPS so this never runs on the CUDA Space.
"""

from __future__ import annotations

import subprocess

GATE_GB: float = 105.0          # operational ceiling (OS crashes ~125 GB)
SAFETY_GB: float = 10.0         # reserved for the ~25 s post-kill wired reclaim
PER_CLIP_FLOOR_GB: float = 25.0  # a single bounded clip's expected need
ABORT_GB: float = 90.0          # mid-run hard abort threshold (system used)


def _parse_vm_stat(text: str) -> float:
    """System memory in use (GB) = (active + wired + compressed) pages × page size."""
    page = 16384
    active = wired = compressed = 0
    for line in text.splitlines():
        if "page size of" in line:
            page = int(line.split()[-2])
        elif "Pages active" in line:
            active = int(line.split()[-1].rstrip("."))
        elif "Pages wired down" in line:
            wired = int(line.split()[-1].rstrip("."))
        elif "occupied by compressor" in line:
            compressed = int(line.split()[-1].rstrip("."))
    return (active + wired + compressed) * page / 1e9


def system_used_gb() -> float:
    return _parse_vm_stat(subprocess.check_output(["vm_stat"]).decode())


def headroom_gb(gate_gb: float = GATE_GB, safety_gb: float = SAFETY_GB) -> float:
    return gate_gb - system_used_gb() - safety_gb


def require_headroom(min_gb: float = PER_CLIP_FLOOR_GB) -> None:
    h = headroom_gb()
    if h < min_gb:
        raise MemoryError(
            f"Insufficient memory headroom to start: {h:.1f} GB < {min_gb} GB "
            f"(system using {system_used_gb():.1f} GB). Close other apps and retry."
        )


def check_or_abort(abort_gb: float = ABORT_GB) -> None:
    used = system_used_gb()
    if used > abort_gb:
        raise MemoryError(
            f"System memory {used:.1f} GB exceeded the {abort_gb} GB safety ceiling "
            f"mid-run; aborting before the OS OOMs."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/techfreakworm/Projects/llm/tribev2-brain-timeline && python -m pytest tests/test_memguard.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/tribescore/memguard.py tests/test_memguard.py
git commit -m "feat(memguard): vm_stat system-memory governor for local MPS runs"
```

---

## Task 2: Per-layer `empty_cache` hook on the V-JEPA2 encoder layers

**Files:**
- Modify: `src/tribescore/fast_encode.py` (add `_register_layer_empty_cache_hooks`; call it in `build_video_model` before caching, ~line 115)
- Test: `tests/test_layer_hooks.py`

**Interfaces:**
- Consumes: `build_video_model(...)` returns an `_HFVideoModel` whose `.model` is the HF VJEPA2 model with `.encoder.layer` (an `nn.ModuleList`).
- Produces: `_register_layer_empty_cache_hooks(hf_model, every: int) -> int` — registers a forward-hook on each encoder layer (firing `mps.synchronize()+empty_cache()` every `every`-th layer), idempotent via a flag **on the model object**; returns the number of hooks registered (0 if already registered or no encoder layers found).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_layer_hooks.py
import torch
import torch.nn as nn
from tribescore import fast_encode

class _Layer(nn.Module):
    def forward(self, x):
        return x

class _Encoder(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layer = nn.ModuleList([_Layer() for _ in range(n)])

class _HFModel(nn.Module):
    def __init__(self, n=40):
        super().__init__()
        self.encoder = _Encoder(n)

def test_registers_one_hook_per_encoder_layer():
    m = _HFModel(n=40)
    n = fast_encode._register_layer_empty_cache_hooks(m, every=1)
    assert n == 40
    assert all(len(l._forward_hooks) == 1 for l in m.encoder.layer)

def test_idempotent_no_double_registration():
    m = _HFModel(n=40)
    fast_encode._register_layer_empty_cache_hooks(m, every=1)
    n2 = fast_encode._register_layer_empty_cache_hooks(m, every=1)  # second call
    assert n2 == 0
    assert all(len(l._forward_hooks) == 1 for l in m.encoder.layer)  # not stacked

def test_no_encoder_layers_returns_zero():
    assert fast_encode._register_layer_empty_cache_hooks(nn.Linear(2, 2), every=1) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/techfreakworm/Projects/llm/tribev2-brain-timeline && python -m pytest tests/test_layer_hooks.py -v`
Expected: FAIL — `AttributeError: module 'tribescore.fast_encode' has no attribute '_register_layer_empty_cache_hooks'`

- [ ] **Step 3: Write minimal implementation**

Add near the top of `src/tribescore/fast_encode.py` (after the imports / module globals, before `build_video_model`):

```python
def _register_layer_empty_cache_hooks(hf_model, every: int) -> int:
    """Register a forward-hook on each V-JEPA2 ENCODER layer that frees the
    per-layer attention transients during the forward, bounding the per-clip peak.

    Targets ONLY `hf_model.encoder.layer` (NOT the predictor / pooler `.layer`
    lists). Fires `torch.mps.synchronize()` + `torch.mps.empty_cache()` every
    `every`-th layer. Idempotent via a flag set ON THE MODEL OBJECT (the model is
    cached in `_VIDEO_MODEL_CACHE` across Score runs, so a module-level flag would
    let hooks stack each run). Returns the number of hooks registered (0 if already
    done or no encoder layers found). MPS-only; callers gate on MPS.
    """
    import torch as _torch

    if getattr(hf_model, "_tribescore_layerhook", False):
        return 0
    enc = getattr(hf_model, "encoder", None)
    layers = getattr(enc, "layer", None)
    if layers is None:
        logger.warning("layer-hook: no encoder.layer ModuleList found; relying on per-clip empty_cache")
        return 0

    counter = {"i": 0}

    def _hook(_module, _inp, _out):
        counter["i"] += 1
        if counter["i"] % every == 0 and hasattr(_torch.mps, "empty_cache"):
            _torch.mps.synchronize()
            _torch.mps.empty_cache()

    n = 0
    for layer in layers:
        layer.register_forward_hook(_hook)
        n += 1
    hf_model._tribescore_layerhook = True
    logger.info("layer-hook: registered per-layer empty_cache on %d encoder layers (every=%d)", n, every)
    return n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/techfreakworm/Projects/llm/tribev2-brain-timeline && python -m pytest tests/test_layer_hooks.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Wire it into `build_video_model` (MPS-only)**

In `src/tribescore/fast_encode.py`, in `build_video_model`, immediately before `_VIDEO_MODEL_CACHE[key] = model` (currently ~line 115), add:

```python
    # Bound the per-clip MPS peak: free the O(N²) attention transients per-layer
    # during the forward (numerically identical — frees only unreferenced memory).
    # MPS-only; on CUDA the per-clip empty_cache + VRAM headroom suffice.
    if _torch.backends.mps.is_available() and "vjepa2" in model_name:
        every = int(os.environ.get("TRIBE_MPS_EMPTY_EVERY", "4"))
        _register_layer_empty_cache_hooks(model.model, every=every)
```

- [ ] **Step 6: Run the unit tests again (regression)**

Run: `cd /Users/techfreakworm/Projects/llm/tribev2-brain-timeline && python -m pytest tests/test_layer_hooks.py -v`
Expected: PASS (still 3 passed)

- [ ] **Step 7: Commit**

```bash
git add src/tribescore/fast_encode.py tests/test_layer_hooks.py
git commit -m "feat(mps): per-layer empty_cache hook on V-JEPA2 encoder layers (TRIBE_MPS_EMPTY_EVERY)"
```

---

## Task 3: Tighten the MPS HIGH watermark + document `B=1`

**Files:**
- Modify: `app.py:37` (HIGH watermark 0.6 → 0.45)
- Modify: `src/tribescore/fast_encode.py` (`B=1` rationale comment at the per-clip loop, ~line 262-264)

No new behavior to unit-test (env default + comment); verified end-to-end in Task 8.

- [ ] **Step 1: Lower the HIGH watermark**

In `app.py`, change line 37 from:
```python
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
```
to:
```python
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
```
(Graceful ordering: HIGH ~0.45 raises a catchable RuntimeError ~43–48 GB pool, surfaced by Trap-A, well below the vm_stat abort at 90 GB system, below the 105 GB gate. Leave `LOW=0.3` at line 36.)

- [ ] **Step 2: Document `B=1`**

In `src/tribescore/fast_encode.py`, immediately above the per-clip loop (currently `for k, ts_list in enumerate(clip_ts):`, ~line 262), add:

```python
                # B=1 is intentional and load-bearing: V-JEPA2 ViT-g is
                # compute-bound and ONE 8192-token clip already over-saturates the
                # GPU (measured: B=4 ≈ 231 s vs B=1 bf16 ≈ 173 s on CUDA; the M5 Max
                # 40-core GPU saturates even harder). Batching clips adds memory with
                # NO throughput gain — do not "optimize" this into a batched forward.
```

- [ ] **Step 3: Sanity-check the file still imports / parses**

Run: `cd /Users/techfreakworm/Projects/llm/tribev2-brain-timeline && python -c "import ast; ast.parse(open('app.py').read()); ast.parse(open('src/tribescore/fast_encode.py').read()); print('AST OK')"`
Expected: `AST OK`

- [ ] **Step 4: Commit**

```bash
git add app.py src/tribescore/fast_encode.py
git commit -m "perf(mps): HIGH watermark 0.6->0.45 (graceful OOM under the gate) + document B=1"
```

---

## Task 4: Wire the memguard into the encode loop

**Files:**
- Modify: `src/tribescore/fast_encode.py` (`_dedup_get_data`: start gate before the event loop; periodic `check_or_abort` inside the per-clip loop) — MPS-only.

No unit test (it's integration glue around `vm_stat` + the live forward); the start-gate + periodic-abort behavior is exercised in Task 8's staged validation. Guarded so it is a strict no-op off MPS.

- [ ] **Step 1: Add the start gate**

In `_dedup_get_data`, right after `dev = model.model.device` (~line 194), add:

```python
            # Local MPS safety gate: refuse to start if there isn't headroom for a
            # bounded clip under the system memory ceiling (MPS wired mem is invisible
            # to RSS — guard on vm_stat). No-op off MPS.
            if torch.backends.mps.is_available():
                from tribescore import memguard
                memguard.require_headroom()
```

- [ ] **Step 2: Add the periodic abort check**

In the per-clip loop, immediately after `output[k] = embd` and before the `if _mps:` cleanup block (~line 274-275), add:

```python
                    # Every ~8 clips, abort cleanly if system memory crosses the
                    # ceiling (e.g. another app grabbed RAM) — before the OS OOMs.
                    if _mps and k % 8 == 0:
                        from tribescore import memguard
                        memguard.check_or_abort()
```

- [ ] **Step 3: Confirm Trap-A re-raises the abort (read-only check)**

The `except Exception` at the end of `_dedup_get_data` (Trap A, ~line 310-322) must let a `MemoryError` surface (not fall back to the heavier original path, which would re-OOM). Verify the branch: `isinstance(exc, MemoryError)` is already in the re-raise condition (~line 316). No change needed — just confirm it reads as expected.

- [ ] **Step 4: Sanity-check parse**

Run: `cd /Users/techfreakworm/Projects/llm/tribev2-brain-timeline && python -c "import ast; ast.parse(open('src/tribescore/fast_encode.py').read()); print('AST OK')"`
Expected: `AST OK`

- [ ] **Step 5: Commit**

```bash
git add src/tribescore/fast_encode.py
git commit -m "feat(mps): wire vm_stat memguard into the dedup encode loop (start gate + periodic abort)"
```

---

## Task 5: Diagnose the ASR `0it` drop (spike — no pre-committed code change)

**Files:**
- Create: `~/Projects/tests/diag_asr_audio.py`

This is a **diagnosis spike**, not a code fix. tribe-brain traced `0it` to `ExtractAudioFromVideo`'s `if not audio: continue` (the video had no audio track), NOT ChunkEvents/`min_duration`. Confirm before changing anything.

- [ ] **Step 1: Write the diagnostic script**

```python
# ~/Projects/tests/diag_asr_audio.py
"""Diagnose the ASR `0it` drop: does the failing test video have an audio track,
and how many Audio events survive ExtractAudioFromVideo? Run with the test video
that produced `Extracting words from audio: 0it`."""
import subprocess, sys, os
sys.path.insert(0, "/Users/techfreakworm/Projects/llm/tribev2-brain-timeline/src")

VIDEO = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Projects/tests/tribe_clip90.mp4")

# 1) Does ffprobe see an audio stream?
out = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
     "stream=index,codec_name,channels", "-of", "default=nw=1", VIDEO],
    capture_output=True, text=True)
print("=== ffprobe audio streams ===")
print(out.stdout.strip() or "(NONE — no audio stream)")

# 2) Build events the way the app does and count Audio rows.
from tribescore.mps import enable_mps; enable_mps()
from tribescore.inference import load_model
model = load_model(os.path.expanduser("~/.cache/tribe-local"))
events = model.get_events_dataframe(video_path=VIDEO)
print("=== events df type counts ===")
print(events["type"].value_counts().to_string())
print("Audio rows:", int((events["type"] == "Audio").sum()),
      "| Word rows:", int((events["type"] == "Word").sum()))
```

- [ ] **Step 2: Run the spike on the failing clip**

Run: `/Users/techfreakworm/Projects/tests/tribe-local-venv/bin/python ~/Projects/tests/diag_asr_audio.py ~/Projects/tests/<the-95s-clip>.mp4`
Record: (a) does ffprobe list an audio stream? (b) Audio/Word row counts.

- [ ] **Step 3: Branch on the finding (decision rule)**

- **If ffprobe shows NO audio stream** (expected): the bug is benign — the clip simply has no speech. **No code fix to the chain.** Re-test the whole pipeline with a real speech video (a clip that ffprobe shows HAS audio). Proceed to **Task 6** (graceful UI message). Record "confirmed: no-audio-track" in the commit message.
- **If ffprobe SHOWS audio but Audio rows == 0:** extraction genuinely fails — the fix is in `ExtractAudioFromVideo` (moviepy `.audio` read). Capture the exact failure, and add a Task 5b (out of this plan's pre-written scope) to fix it; do NOT touch `ChunkEvents`/`min_duration` (proven irrelevant).

- [ ] **Step 4: Commit the diagnostic script + finding**

```bash
cd ~/Projects/tests && git -C /Users/techfreakworm/Projects/llm/tribev2-brain-timeline add docs/superpowers/plans/2026-06-24-local-mps-quality-video.md  # if you annotate the finding
# (the diag script lives in ~/Projects/tests which is not the repo; keep it there)
```
Note the finding inline in this plan or the spec; no repo code changed by the spike itself.

---

## Task 6: Graceful "no speech track → video-only" message

**Files:**
- Modify: `src/tribescore/inference.py` (`run_inference`: detect no-Word-events in Quality/full path; surface via the existing `out_info` dict)
- Modify: `app.py` (`_score_impl`: read the flag and prepend a one-line notice to the summary)
- Test: `tests/test_no_speech_notice.py`

**Interfaces:**
- Consumes: `run_inference(..., out_info: dict | None = None)` (already exists — see `inference.py`).
- Produces: when `mode != "video" `... actually any full-multimodal run where the built events contain **zero `type=="Word"` rows**, `run_inference` sets `out_info["no_speech"] = True`. `_score_impl` turns that into a visible notice.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_no_speech_notice.py
import types
import numpy as np
from tribescore import inference

def test_run_inference_flags_no_speech(monkeypatch):
    # Stub a model whose events have NO Word rows; assert out_info["no_speech"] is set.
    import pandas as pd
    events = pd.DataFrame({"type": ["Audio", "Video"], "filepath": ["a", "v"]})

    class _Model:
        def get_events_dataframe(self, **kw):
            return events
        def predict(self, ev):
            seg = types.SimpleNamespace(start=0.0)
            return np.zeros((1, 20484)), [seg]

    monkeypatch.setattr(inference, "assert_model_runtime", lambda: None)
    info = {}
    inference.run_inference(_Model(), "video", "v.mp4", audio_only=False, out_info=info)
    assert info.get("no_speech") is True

def test_run_inference_no_flag_when_words_present(monkeypatch):
    import pandas as pd
    events = pd.DataFrame({"type": ["Audio", "Word"], "filepath": ["a", "a"],
                           "text": [None, "hi"]})
    class _Model:
        def get_events_dataframe(self, **kw):
            return events
        def predict(self, ev):
            seg = types.SimpleNamespace(start=0.0)
            return np.zeros((1, 20484)), [seg]
    monkeypatch.setattr(inference, "assert_model_runtime", lambda: None)
    info = {}
    inference.run_inference(_Model(), "video", "v.mp4", audio_only=False, out_info=info)
    assert "no_speech" not in info
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/techfreakworm/Projects/llm/tribev2-brain-timeline && python -m pytest tests/test_no_speech_notice.py -v`
Expected: FAIL — `assert info.get("no_speech") is True` fails (flag not set)

- [ ] **Step 3: Implement in `run_inference`**

In `src/tribescore/inference.py`, in `run_inference`, right after the events are built and the existing `out_info` text-media block (after `events = model.get_events_dataframe(...)`; near where `out_info["media_path"]` is set for text mode), add:

```python
    # Full-multimodal run with no transcribed speech → flag it so the UI can say so
    # (e.g. the video has no audio track). Quality silently degrades to video-only
    # otherwise. Best-effort.
    if out_info is not None and not audio_only:
        try:
            if int((events["type"] == "Word").sum()) == 0:
                out_info["no_speech"] = True
        except Exception:
            pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/techfreakworm/Projects/llm/tribev2-brain-timeline && python -m pytest tests/test_no_speech_notice.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Surface it in the UI**

In `app.py`, `_score_impl` already unpacks `_gpu_infer` and builds `summary_str`. `_gpu_infer` must thread `info["no_speech"]` out (it already threads `info.get("media_path")`). Change `_gpu_infer`'s return to also carry it and prepend a notice. Minimal approach — in `_gpu_infer`, return `(preds, abs_times, info.get("media_path"), info.get("no_speech", False))`; in `_score_impl` unpack the extra value and, when true, prepend to `summary_str`:

```python
        # in _gpu_infer's return:
        return preds, abs_times, info.get("media_path"), bool(info.get("no_speech"))
```
```python
        # in _score_impl, replace the unpack:
        preds, abs_times, text_media, no_speech = _gpu_infer(mode, src_path, bool(audio_only))
        ...
        if no_speech:
            summary_str = (
                '<div class="co-notice">⚠ No speech track detected — scored on '
                'video + audio only (Language/semantic-load not from speech).</div>'
                + summary_str
            )
```

- [ ] **Step 6: Verify app still parses**

Run: `cd /Users/techfreakworm/Projects/llm/tribev2-brain-timeline && python -c "import ast; ast.parse(open('app.py').read()); print('AST OK')"`
Expected: `AST OK`

- [ ] **Step 7: Commit**

```bash
git add src/tribescore/inference.py app.py tests/test_no_speech_notice.py
git commit -m "feat(quality): surface 'no speech track -> video-only' notice when no Word events"
```

---

## Task 7: WIN-1 seam — instrument, then cure the z-score degeneracy

**Files:**
- Create: `~/Projects/tests/diag_seam.py` (instrumentation)
- Modify: `src/tribescore/windowing.py` (`_zscore_over_time`: guard degenerate near-constant windows)
- Test: `tests/test_windowing.py` (add a degenerate-window case)

tribe-brain: `stitch` already weight-normalizes (lines 271-273); the ~22× seam step is consistent with a **degenerate per-window z-score** — `(x-mean)/(std+1e-6)` blows up when a window is near-constant (std≈0). Instrument first, then fix the real cause.

- [ ] **Step 1: Instrumentation spike**

```python
# ~/Projects/tests/diag_seam.py
"""Log per-window std + weight_sum around the failing seam to confirm the WIN-1
cause is a degenerate z-score (std≈0), not the crossfade. Feed real (preds,
abs_times) from a >150 s score (pickle them from a run, or synth a constant window)."""
import sys, numpy as np
sys.path.insert(0, "/Users/techfreakworm/Projects/llm/tribev2-brain-timeline/src")
from tribescore.windowing import _segment_windows, _zscore_over_time

preds = np.load(sys.argv[1])      # (R, 20484) from a real >150s run
abs_times = np.load(sys.argv[2])  # (R,)
for lo, hi in _segment_windows(abs_times):
    w = preds[lo:hi]
    std = w.std(axis=0)
    print(f"window [{lo}:{hi}] rows={hi-lo} min_std={std.min():.3e} "
          f"frac_near_zero={(std < 1e-3).mean():.2%} z_max={np.abs(_zscore_over_time(w)).max():.1f}")
```
Run it on a real >150 s `(preds, abs_times)`. **Expected finding:** the seam-adjacent window has parcels with `std ≈ 0` and `z_max` in the tens (≈22×), confirming z-score degeneracy.

- [ ] **Step 2: Write the failing test (degenerate window)**

Add to `tests/test_windowing.py`:

```python
def test_zscore_does_not_blow_up_on_near_constant_window():
    import numpy as np
    from tribescore.windowing import _zscore_over_time
    # A window whose signal is ~constant over time (std≈0) with tiny float jitter.
    win = np.full((20, 4), 5.0) + np.array([0, 1e-7, -1e-7, 0])
    z = _zscore_over_time(win)
    # A degenerate (near-constant) vertex must map to ~0, never explode to tens.
    assert np.abs(z).max() < 3.0, f"z-score exploded to {np.abs(z).max():.1f} on a near-constant window"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /Users/techfreakworm/Projects/llm/tribev2-brain-timeline && python -m pytest tests/test_windowing.py::test_zscore_does_not_blow_up_on_near_constant_window -v`
Expected: FAIL — z explodes (tiny jitter / (≈0 std + 1e-6) → large values)

- [ ] **Step 4: Fix `_zscore_over_time`**

In `src/tribescore/windowing.py`, replace `_zscore_over_time` (lines 305-313) body with a **relative epsilon** that treats a near-constant window as flat (→ ~0) instead of amplifying float noise:

```python
def _zscore_over_time(win_preds: np.ndarray) -> np.ndarray:
    """Per-vertex z-score over time for one window (``§4`` step b).

    ``(x - mean_t) / max(std_t, eps)`` along the time (row) axis. ``eps`` is
    RELATIVE to the window's signal scale so a near-constant vertex maps to ~0
    instead of amplifying float jitter into a huge z (the WIN-1 seam bug). An
    absolute 1e-6 floor (the old value) divides ~0-std jitter by ~1e-6 → ~22×
    spurious steps at window seams on long clips.
    """
    mean_t = win_preds.mean(axis=0, keepdims=True)
    std_t = win_preds.std(axis=0, keepdims=True)
    # Relative floor: a fraction of the window's typical per-vertex spread.
    scale = np.median(std_t) if std_t.size else 0.0
    eps = max(_ZSCORE_EPS, 1e-3 * float(scale))
    return (win_preds - mean_t) / np.maximum(std_t, eps)
```

- [ ] **Step 5: Run the new test + the existing windowing suite**

Run: `cd /Users/techfreakworm/Projects/llm/tribev2-brain-timeline && python -m pytest tests/test_windowing.py -v`
Expected: PASS — the new test passes AND all existing stitch/seam tests still pass (the relative eps only changes degenerate windows; normal windows have `std ≫ eps`).

- [ ] **Step 6: Commit**

```bash
git add src/tribescore/windowing.py tests/test_windowing.py
git commit -m "fix(windowing): relative z-score epsilon to cure WIN-1 seam blow-up on near-constant windows"
```

---

## Task 8: Staged MPS validation under the gate + bf16 parity

**Files:**
- Create: `~/Projects/tests/validate_local_video.py`

End-to-end gate to flip `KNOWN_ISSUES.md` Video·Local ⛔→✅. Runs entirely from `~/Projects/tests` with the local venv. **Run each stage guarded** (the memguard aborts before any OS risk).

- [ ] **Step 1: Write the staged validation harness**

```python
# ~/Projects/tests/validate_local_video.py
"""Staged local MPS video validation: peak system memory + s/clip per stage.
Usage: validate_local_video.py <clip.mp4> [audio_only=0]"""
import os, sys, time, threading
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.3"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.45"
sys.path.insert(0, "/Users/techfreakworm/Projects/llm/tribev2-brain-timeline/src")
from tribescore import memguard

CLIP = sys.argv[1]
AUDIO_ONLY = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
PEAK = [0.0]
def sample():
    while True:
        PEAK[0] = max(PEAK[0], memguard.system_used_gb())
        time.sleep(0.4)
threading.Thread(target=sample, daemon=True).start()

from tribescore.mps import enable_mps; enable_mps()
from tribescore import fast_encode; fast_encode.apply_frame_dedup_encode()
from tribescore.inference import load_model, run_inference
model = load_model(os.path.expanduser("~/.cache/tribe-local"))
print(f"baseline sys={memguard.system_used_gb():.1f}GB peak_so_far={PEAK[0]:.1f}")
t = time.perf_counter()
info = {}
preds, at = run_inference(model, "video", CLIP, audio_only=AUDIO_ONLY, out_info=info)[:2]
dt = time.perf_counter() - t
print(f"DONE clip={os.path.basename(CLIP)} TRs={len(at)} time={dt:.1f}s "
      f"s/clip={dt/max(len(at),1):.2f} SYS_PEAK={PEAK[0]:.1f}GB no_speech={info.get('no_speech')}")
assert PEAK[0] < 80.0, f"system peak {PEAK[0]:.1f}GB exceeded the 80GB target"
print("PASS: under the 80GB target")
```

- [ ] **Step 2: Stage A — short single-window clip (Fast)**

Run: `/Users/techfreakworm/Projects/tests/tribe-local-venv/bin/python ~/Projects/tests/validate_local_video.py ~/Projects/tests/tribe_clip15.mp4 1`
Expected: completes; `SYS_PEAK < 80 GB`; record s/clip.

- [ ] **Step 3: Stage B — 60 s clip (Fast, multi-window)**

Run: `... validate_local_video.py ~/Projects/tests/tribe_clip60.mp4 1`
Expected: completes; `SYS_PEAK < 80 GB`.

- [ ] **Step 4: Stage C — 2–3 min clip (Quality, full multimodal — a clip WITH a real speech track)**

Run: `... validate_local_video.py ~/Projects/tests/<2-3min-speech>.mp4 0`
Expected: completes; `SYS_PEAK < 80 GB`; `no_speech` is `None`/`False`; record total wall-time (expect ~3–6 min forward floor) + whether Llama is co-resident (watch the peak vs the §6 budget).

- [ ] **Step 5: bf16 parity A/B (vs the HF/CUDA result)**

Score the SAME short clip on the HF Space (CUDA, bf16) and locally; compare the output curves. Expect `max|Δ|` ≈ float-noise on the z-scored metrics. (If it diverges beyond noise → escalate: switch to Approach C / fp32 per the spec.)

- [ ] **Step 6: Full UI smoke (Playwright, local)**

Launch the local Gradio app (`~/Projects/tests/run_local.py`), upload the 2–3 min speech clip, Quality mode, all metrics; confirm the timeline renders, the seam is continuous (no ~22× step), and the audio/Language curves are populated. Watch `~/Projects/tests` memgate during the run.

- [ ] **Step 7: Flip the status + commit the docs**

When all stages pass, update `KNOWN_ISSUES.md`: Video·Local ⛔→✅ (with the "long videos run locally, ~3–6 min/3-min clip" note), and note the local-video Quality path is verified.

```bash
git add KNOWN_ISSUES.md
git commit -m "docs: local MPS video Quality verified end-to-end (long videos run locally)"
```

---

## Self-review notes

- **Spec coverage:** P1 → Tasks 1–4 (memguard, hook, watermark/B=1, wiring). P2 → Tasks 5–6 (ASR diagnosis + no-speech notice; Llama concurrency verified in Task 8 Stage C). P3 → Task 7 (seam instrument + z-score fix). Verification → Task 8. P4 explicitly out of scope.
- **Diagnostic-first tasks (5, 7-step-1)** carry decision rules + the most-likely fix pre-written (no-speech notice; relative-eps z-score), so they are not placeholders.
- **Type consistency:** `out_info` dict keys (`media_path`, `no_speech`) and `_gpu_infer`'s 4-tuple return are threaded consistently between Task 6's `inference.py` and `app.py` edits.
- **Test env:** unit tests assume `pytest` in the env running them (`python -m pytest`); install if absent (`pip install pytest`). MPS scripts use `~/Projects/tests/tribe-local-venv/bin/python`.
