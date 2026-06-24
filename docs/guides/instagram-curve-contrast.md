# Brain curves vs. Instagram reel curves — a creator's guide

*How the "in-silico neuroscience" timeline from this tool lines up against the analytics Instagram already gives you — and what that comparison can (and can't) tell you.*

---

## 📌 TL;DR — read this into the camera

- Instagram now shows you **two curves plotted second-by-second through your reel**: a **retention curve** (how many viewers are still watching at each second) and a **like-timing curve** (the exact moments people tapped ❤️).
- This tool produces **five brain-response curves on the *same* second-by-second axis** — so you can lay them side by side.
- The promise: **the brain curves predict *where* the Instagram curves will move** — engagement dips *before* people swipe away; engagement/virality peaks *before* people like.
- The honest limit: this predicts the **shape of attention through your video**, not whether it goes viral. Virality is mostly *not* about the content. **Use it to find your weak seconds, not to forecast your view count.**

---

## Why these even compare

Here's the unlock most people miss: **all of these curves share one x-axis — time into the video (seconds, or % of the way through).**

| Instagram gives you | What it is |
|---|---|
| **Retention chart** (added 2025) | % of viewers still watching at each second of the reel |
| **Like-timing chart** ("when people liked") | the playback moments viewers tapped ❤️ — IG notes these cluster at **1–3s (the hook), a mid value-delivery beat, and the end** |

Because this tool's curves are also on the *time-into-the-video* axis, you can **stack them on top of each other** and ask: *does what the model predicts about a brain match what real viewers actually did?*

---

## The five brain curves, in plain English

This tool runs your video through a model of how an average human cortex would respond, and plots five things over time:

| Curve | In one phrase |
|---|---|
| **Attention** | how *locked-on* the eyes/focus are |
| **Engagement / arousal** | how *into it* the viewer is (excitement, intensity) |
| **Virality (proxy)** | the *urge to react / share* — the softest, least certain signal |
| **Language / semantic load** | how hard the *words and meaning* are being processed |
| **Self-relevance / DMN** | *reflective / "this is about me"* vs. mind-wandering |

**How to read the numbers:** the curves are **z-scored per video** — meaning they're scored relative to *that clip's own average*:

- **0** = average for this video
- **+1 / +2** = a standout peak (a strong moment)
- **−1 / −2** = a dead spot

👉 **Read the *shape and timing*, not the absolute number.** A "+1.5" doesn't mean "good" in any universal sense — it means "a high point *for this video*." (Comparing *across* videos needs a different normalization — see the caveats.)

---

## The contrast guide (the core)

Three comparisons worth making — and one that's a trap.

### 1. Engagement / Attention ↔ the retention curve
Compare against the retention **drop-rate**, not its level. (Retention only ever falls; the brain curves oscillate up and down — so don't expect the lines to *match*, expect the *turning points* to.)

> **Expect engagement *troughs* to land just *before* retention cliffs — roughly a 0.5–2 s lead.**

That lead is the gold: **those troughs are your cut points.** If engagement dips at 0:07 and retention falls off a cliff at 0:08, that second of video is what's losing people. Tighten it or cut it.

### 2. Virality / Engagement peaks ↔ the like spikes
This is the cleanest validation: **peak aligns with peak.** When engagement or the virality proxy spikes, you should see a like spike land **~0.5–1.5 s later** (tap latency — people feel it, *then* their thumb moves).

Instagram's own observation that likes cluster at the **hook, a value-delivery beat, and the end** maps directly onto where these curves *should* be peaking. If your model peaks but the likes don't follow, the moment isn't landing the way the content "should."

### 3. ⚠️ The trap: totals vs. average score
**Do not** compare your total likes / views / shares against your average brain score. They won't correlate, and that's expected — **virality is mostly non-content**: the algorithm, your skip rate, audience size, posting time, thumbnail, luck. The brain signal lives in the **per-moment curves**, not the totals. (More in the caveats.)

---

## How to actually run the comparison

1. **Put both on the same axis.** Resample everything to **% of duration (0–100%)** so different reel lengths overlay cleanly.
2. **Find the lead/lag with cross-correlation.** Slide one curve against the other and find the offset where they line up best — that tells you the consistent lead time (e.g. "engagement leads retention by ~1 s").
3. **Do it across ~10–20 reels, not one.** A single reel is noise. You're looking for a *pattern that repeats* — that's what's real.
4. Self-normalized curves are fine for **within-video shape**. Comparing *levels across videos* needs population normalization (a planned feature; for now, compare shapes).

---

## The one practical friction (be honest about this)

Both Instagram curves live **inside the app** (Reel Insights) — there's no clean export. The **Graph API gives you scalars** (plays, reach, likes, average watch time) but **not the per-second curves.**

So for a real study, you'll **screenshot the in-app retention/like graphs and trace them** to get the curve data. It's manual. Say so honestly if you cover it.

---

## ⚖️ Honest caveats (keep these on screen)

- **It's a model *proxy*, not an oracle.** It estimates how an average cortex *would* respond — it has not measured your specific viewers.
- **The virality ceiling is real.** Content is a *minority* factor in whether a reel pops. This finds *content* weak spots; it can't beat the algorithm.
- **Z-scored = relative.** There is no universal "good number" — only high and low points *within a video*.
- **One reel is anecdote.** The comparison only becomes a finding across a batch.

---

## Sources

- Instagram adds retention insights for Reels — Social Media Today: https://www.socialmediatoday.com/news/instagram-adds-retention-insights-reels/758464/
- Instagram Reel analytics — Metricool: https://metricool.com/instagram-reel-analytics/
- Instagram Reels insights — Inro: https://www.inro.social/blog/instagram-reels-insights
- Instagram Reel retention chart — Website Builder Expert: https://www.websitebuilderexpert.com/news/instagram-reel-retention-chart/
- Instagram media insights (Graph API) — Meta for Developers: https://developers.facebook.com/docs/instagram-platform/reference/instagram-media/insights/

---

*This is a research-grade proxy framing for content diagnosis, not a performance guarantee. The curves describe predicted per-moment attention, not outcomes.*
