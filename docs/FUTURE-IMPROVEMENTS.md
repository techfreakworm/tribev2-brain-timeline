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

**Next step when prioritized:** tribe-brain to write the full Tier-1 design spec.
