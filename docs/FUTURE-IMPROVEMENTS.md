# Future improvements (backlog)

Parked ideas — not scheduled. Each entry notes origin, why it's non-trivial, and a sketch.

---

## Cross-video "engagement benchmark" comparator (exemplar-based)

**Status:** backlog (operator's explicit call — do NOT implement until prioritized).
**Origin:** operator wants to upload a video and see where it falls short vs viral / aspirational
clips — calibrated curves + numbers comparable *across* videos, not just within one.

**Why it's not free today:** the timeline curves are z-scored **per-video** (the global-z-score seam
fix), so every clip averages ~0 → absolute levels aren't comparable across videos. Today you only get
within-video diagnostics + rough shape-eyeballing.

**Core unlock — population/exemplar normalization:** keep the raw (pre-self-normalization) metric
values; normalize a new clip against a fixed **reference set's** mean/spread (not itself); and
**time-align to % of duration (0–100%)** so different lengths overlay. Then "+1.5" means "1.5 SD above a
typical reference clip," consistent across videos. (Assumes the model's raw output scale is stable across
videos — plausible since it's one model; sanity-check it.)

**Honest ceiling — frame it as a content-engagement benchmark, NOT a P(viral) oracle.** Virality is
mostly non-content (algorithm, follower count, timing, thumbnail, luck); the brain-metric signal explains
only a slice. Setting that expectation is both honest and the actually-useful framing.

**Recommended MVP (Tier 1 — needs NO labeled dataset): exemplar comparator.**
- User picks niche-matched reference clips (own best performers / admired creators). Score them → a
  reference "band."
- For a new clip: overlay its curves on the band, highlight segments below it, and report 3–5
  interpretable **gap features** as percentiles vs the reference set:
  - **hook** (peak in first ~5–10%), **sustain** (trough depth/length), **dynamism**,
    **climax alignment** (peak on the payoff/CTA?), **coherence** (Language rises where dialogue matters?).
- Output reads like: "hook = 30th pct; dead zone at 20–30%; climax 8s early."

**Heavier tiers (later):** Tier 2 = single profile-match similarity score + ranked gaps (same exemplars).
Tier 3 = trained predictor with attribution (needs a **labeled** corpus + real-outcome validation;
closest to an "oracle" but capped by data + the content ceiling).

**Caveats to record:** (1) the reference set IS the oracle — niche-match it or it gives
confidently-wrong advice; (2) proxy-on-a-proxy → only "real" once validated against actual
retention/share data (close the loop); (3) main build pieces = store raw/population stats, a
reference-clip library, time-alignment, and the comparator view.

**Why it's relative — the absolute-score ceiling (conceptual foundation).** Operator question: can
tribev2 say a clip *is* exciting/boring in absolute terms, so a wholly-boring clip doesn't show misleading
peaks? **No — absoluteness isn't recoverable**, for three stacked reasons:
1. **fMRI has no absolute scale** — BOLD is % change from an arbitrary baseline; there is no "0–100 arousal
   meter" in the ground truth.
2. **tribev2 was trained on per-run z-scored fMRI** — it learned the *relative pattern* only; even its *raw*
   output lives in a normalized space, with no notion of "this whole clip is more aroused than that one."
3. Our pipeline then adds the **global z-score** (the seam fix); removing it still leaves (1) and (2).

**Direct consequence (the operator's exact worry — and it's correct):** self-normalized curves **always**
show internal peaks, *even in a boring clip*. An internal peak = "the most active moment *relative to this
clip*," NOT real arousal — the model cannot tell you "the whole thing is flat." This is precisely why the
comparator above must use **raw** (pre-self-z-score) output vs a reference population, validated against real
retention/like data — not the z-scored curves.

**Arousal ≠ valence.** Attention/Engagement/Virality are **arousal/engagement INTENSITY** proxies, not
valence — they can't cleanly separate happy-excited from angry-excited, or calm-content from sad. So an
absolute "happy vs sad" is **doubly** out of reach (weak/absent valence signal + no absolute scale).

**Honest reframe of the goal:** "absolute arousal/happiness score" = impossible from this model;
"**this clip is above/below a typical video**" = the achievable, honest version — i.e. exactly the
exemplar/population comparator + IG-validation loop above.

**Next step when prioritized:** tribe-brain to write the full Tier-1 design spec.
