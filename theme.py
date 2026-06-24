"""Cortical Observatory theme — a calm, clinical-but-warm scientific instrument.

The visual identity for the TRIBE v2 brain-score Space (PLAN.md §6): a deep
slate surface read like an fMRI viewer, with the two ends of a BOLD activation
colormap (hot amber / cool cyan) used with restraint as accents. Metric curves
themselves use a colorblind-safe scientific ramp (see ``plotting.py``).

This module exposes the same three things ``qwen_theme.py`` does, so ``ui.py``
and ``app.py`` consume it identically:

* ``PALETTE`` -- the single source of truth for colors + a couple of metrics.
* ``build_theme()`` -- a configured :class:`gradio.themes.Base`.
* ``CSS`` -- the touches Gradio's theme tokens can't express alone (the
  wordmark, the synchronized-readout chrome, the state panels, the seek
  hint, and the licensing footer).

Type roles (three, per §6):
    * display = **Space Grotesk** -- wordmark + section headers, used sparingly.
    * body    = **Inter** -- Gradio's default chain (the bulk of the UI text).
    * data    = **JetBrains Mono** -- tabular figures for timestamps / metric
      values, where aligned numerals matter (the timeline is a clock).

Fonts are pulled from Google Fonts via :class:`gradio.themes.GoogleFont`; the
``@import`` in :data:`CSS` is a belt-and-suspenders fallback so the display +
mono faces are present even if the theme font chain is overridden.
"""

from __future__ import annotations

import gradio as gr

# Single source of truth for the palette. Variant: "Cortical Observatory" --
# deep slate panels + the hot/cool ends of an fMRI activation colormap.
PALETTE: dict[str, str] = {
    # Surfaces
    "body_bg": "#0E1116",      # page: near-black slate
    "panel_bg": "#161B22",     # cards / blocks
    "input_bg": "#0E1116",     # inputs sit flush with the page
    "elev_bg": "#1B2230",      # hover / raised state
    # Ink
    "ink": "#E8EDF2",          # primary text
    "dim": "#8A93A0",          # secondary / labels / info
    "faint": "#5A626E",        # tertiary / separators-as-text
    # Lines
    "border": "#232A33",
    "border_strong": "#313B47",
    # Accents -- the two ends of a BOLD activation colormap.
    "accent": "#FFB454",       # activation hot (amber) -- primary action
    "accent_text": "#0E1116",  # ink ON the amber button
    "cool": "#36D1C4",         # cool cyan -- counter-accent / live indicators
    "cool_dim": "#1F7A73",
    # Status
    "error": "#FF6B6B",
    "warn": "#FFB454",
    "ok": "#36D1C4",
    "radius": "10px",
}


def build_theme() -> gr.themes.Base:
    """Return a Gradio theme matching the Cortical Observatory palette.

    Three font roles are registered (display / body / mono). The neutral and
    primary hue ramps are derived from :data:`PALETTE` so Gradio's own
    components (sliders, focus rings, primary buttons) stay on-palette without
    per-component CSS overrides.
    """
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            # Amber "activation hot" ramp around PALETTE["accent"].
            c50="#FFF6E9",
            c100="#FFE9C7",
            c200="#FFD79A",
            c300="#FFC774",
            c400="#FFBE63",
            c500=PALETTE["accent"],
            c600="#E69A3C",
            c700="#BE7C2C",
            c800="#8C5A1E",
            c900="#5A3A12",
            c950="#2E1E09",
        ),
        secondary_hue=gr.themes.Color(
            # Cool cyan counter-accent ramp around PALETTE["cool"].
            c50="#E7FBF8",
            c100="#C2F4ED",
            c200="#8DE9DE",
            c300="#5BDCCE",
            c400="#42D6C7",
            c500=PALETTE["cool"],
            c600="#23A99D",
            c700="#1F7A73",
            c800="#155350",
            c900="#0E3634",
            c950="#071E1D",
        ),
        neutral_hue=gr.themes.Color(
            # Slate ramp: light ink -> near-black page.
            c50="#E8EDF2",
            c100="#CDD5DE",
            c200="#AAB4C0",
            c300="#8A93A0",
            c400="#5A626E",
            c500="#3B434E",
            c600="#313B47",
            c700="#232A33",
            c800="#161B22",
            c900="#0E1116",
            c950="#080B0F",
        ),
        font=(
            gr.themes.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
        font_mono=(
            gr.themes.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "SFMono-Regular",
            "Menlo",
            "monospace",
        ),
        radius_size=gr.themes.sizes.radius_md,
        spacing_size=gr.themes.sizes.spacing_md,
        text_size=gr.themes.sizes.text_md,
    ).set(
        # --- Surfaces -------------------------------------------------------
        body_background_fill=PALETTE["body_bg"],
        body_text_color=PALETTE["ink"],
        body_text_color_subdued=PALETTE["dim"],
        background_fill_primary=PALETTE["panel_bg"],
        background_fill_secondary=PALETTE["body_bg"],
        block_background_fill=PALETTE["panel_bg"],
        block_border_color=PALETTE["border"],
        block_border_width="1px",
        block_label_background_fill=PALETTE["panel_bg"],
        block_label_text_color=PALETTE["dim"],
        block_title_text_color=PALETTE["ink"],
        block_radius=PALETTE["radius"],
        panel_background_fill=PALETTE["panel_bg"],
        panel_border_color=PALETTE["border"],
        # --- Inputs ---------------------------------------------------------
        input_background_fill=PALETTE["input_bg"],
        input_background_fill_focus=PALETTE["elev_bg"],
        input_border_color=PALETTE["border"],
        input_border_color_focus=PALETTE["accent"],
        input_placeholder_color=PALETTE["faint"],
        # --- Primary button (activation hot) --------------------------------
        button_primary_background_fill=PALETTE["accent"],
        button_primary_background_fill_hover="#FFC774",
        button_primary_text_color=PALETTE["accent_text"],
        button_primary_border_color=PALETTE["accent"],
        # --- Secondary button (quiet, bordered) -----------------------------
        button_secondary_background_fill=PALETTE["elev_bg"],
        button_secondary_background_fill_hover=PALETTE["border_strong"],
        button_secondary_text_color=PALETTE["ink"],
        button_secondary_border_color=PALETTE["border_strong"],
        # --- Accents / focus ------------------------------------------------
        slider_color=PALETTE["accent"],
        color_accent=PALETTE["accent"],
        color_accent_soft="rgba(255,180,84,0.12)",
        checkbox_background_color=PALETTE["input_bg"],
        checkbox_background_color_selected=PALETTE["accent"],
        checkbox_border_color=PALETTE["border_strong"],
        checkbox_border_color_focus=PALETTE["accent"],
        checkbox_label_background_fill=PALETTE["panel_bg"],
        checkbox_label_background_fill_selected=PALETTE["elev_bg"],
        # --- Links ----------------------------------------------------------
        link_text_color=PALETTE["cool"],
        link_text_color_hover="#5BDCCE",
        link_text_color_active=PALETTE["cool"],
    )


CSS: str = """
/* Cortical Observatory -- chrome Gradio's theme tokens can't express alone. */

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --co-bg: #0E1116;
    --co-panel: #161B22;
    --co-elev: #1B2230;
    --co-ink: #E8EDF2;
    --co-dim: #8A93A0;
    --co-faint: #5A626E;
    --co-border: #232A33;
    --co-border-strong: #313B47;
    --co-amber: #FFB454;
    --co-cyan: #36D1C4;
    --co-error: #FF6B6B;
    --co-display: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif;
    --co-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}

/* Keep the heavy hero contained on very wide monitors. */
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; }

/* ---- Wordmark / masthead ------------------------------------------------ */
.co-masthead {
    display: flex;
    align-items: baseline;
    gap: 14px;
    flex-wrap: wrap;
    padding: 4px 2px 2px;
    margin-bottom: 2px;
}
.co-wordmark {
    font-family: var(--co-display);
    font-weight: 700;
    font-size: 26px;
    letter-spacing: -0.01em;
    color: var(--co-ink);
    line-height: 1.1;
}
/* The "·" between the two activation-colormap ends. */
.co-wordmark .co-hot { color: var(--co-amber); }
.co-wordmark .co-cool { color: var(--co-cyan); }
.co-tagline {
    font-family: var(--co-mono);
    font-size: 12px;
    color: var(--co-dim);
    letter-spacing: 0.02em;
}

/* Section headers use the display face, sparingly. */
.co-section-title {
    font-family: var(--co-display);
    font-weight: 600;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--co-dim);
    margin: 2px 2px 6px;
}

/* ---- Live / quota banner (Space-only) ----------------------------------- */
.co-quota {
    margin: 2px 0 10px;
    padding: 9px 14px;
    font-size: 12.5px;
    line-height: 1.5;
    color: var(--co-dim);
    background: linear-gradient(180deg, rgba(54,209,196,0.06), rgba(54,209,196,0.02));
    border: 1px solid var(--co-border);
    border-radius: 10px;
}
.co-quota strong { color: var(--co-ink); font-weight: 600; }
.co-quota .co-dot::before {
    content: "";
    display: inline-block;
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--co-cyan);
    box-shadow: 0 0 0 0 rgba(54,209,196,0.5);
    margin-right: 7px;
    vertical-align: middle;
    animation: co-pulse 2.4s ease-out infinite;
}
@keyframes co-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(54,209,196,0.45); }
    70%  { box-shadow: 0 0 0 6px rgba(54,209,196,0); }
    100% { box-shadow: 0 0 0 0 rgba(54,209,196,0); }
}

/* ---- Synchronized readout (the hero / signature element) ----------------- */
.co-readout {
    border: 1px solid var(--co-border);
    border-radius: 12px;
    background:
        radial-gradient(1200px 280px at 20% -10%, rgba(255,180,84,0.05), transparent 60%),
        radial-gradient(900px 260px at 90% -20%, rgba(54,209,196,0.05), transparent 55%),
        var(--co-panel);
    padding: 10px;
}

/* The custom <video id="tm-video"> mount (the seek fallback target, §6). */
.co-video-wrap { position: relative; }
#tm-video {
    width: 100%;
    display: block;
    border-radius: 10px;
    background: #000;
    border: 1px solid var(--co-border);
    aspect-ratio: 16 / 9;
    object-fit: contain;
}
.co-video-empty {
    display: flex;
    align-items: center;
    justify-content: center;
    aspect-ratio: 16 / 9;
    border: 1px dashed var(--co-border-strong);
    border-radius: 10px;
    color: var(--co-faint);
    font-family: var(--co-mono);
    font-size: 13px;
    background:
        repeating-linear-gradient(45deg, rgba(255,255,255,0.012) 0 12px, transparent 12px 24px),
        var(--co-bg);
}

/* The "click a spike -> jump there" hint sits under the timeline. */
.co-seek-hint {
    font-family: var(--co-mono);
    font-size: 11.5px;
    color: var(--co-faint);
    margin-top: 4px;
    text-align: right;
}
.co-seek-hint kbd {
    font-family: var(--co-mono);
    color: var(--co-dim);
    border: 1px solid var(--co-border-strong);
    border-bottom-width: 2px;
    border-radius: 5px;
    padding: 0 5px;
    background: var(--co-elev);
}

/* ---- Summary strip (peak / mean per metric) ----------------------------- */
.co-summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
    margin-top: 10px;
}
.co-stat {
    border: 1px solid var(--co-border);
    border-left: 3px solid var(--co-border-strong);
    border-radius: 8px;
    padding: 9px 12px;
    background: var(--co-bg);
}
.co-stat .co-stat-name {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--co-dim);
}
.co-stat .co-stat-val {
    font-family: var(--co-mono);
    font-size: 19px;
    font-weight: 500;
    color: var(--co-ink);
    line-height: 1.3;
}
.co-stat .co-stat-sub {
    font-family: var(--co-mono);
    font-size: 11px;
    color: var(--co-faint);
}

/* ---- State panels: empty / loading / error ------------------------------ */
.co-state {
    border: 1px solid var(--co-border);
    border-radius: 12px;
    padding: 30px 26px;
    background: var(--co-panel);
    text-align: center;
}
.co-state-icon {
    font-size: 26px;
    line-height: 1;
    margin-bottom: 12px;
    opacity: 0.9;
}
.co-state-title {
    font-family: var(--co-display);
    font-weight: 600;
    font-size: 17px;
    color: var(--co-ink);
    margin-bottom: 6px;
}
.co-state-body {
    font-size: 13.5px;
    color: var(--co-dim);
    line-height: 1.6;
    max-width: 460px;
    margin: 0 auto;
}
.co-state.co-state-error {
    border-color: rgba(255,107,107,0.35);
    background: linear-gradient(180deg, rgba(255,107,107,0.06), transparent), var(--co-panel);
}
.co-state.co-state-error .co-state-title { color: var(--co-error); }

/* Loading: theme Gradio's native gr.Progress widget to the amber accent so the
   real per-clip determinate bar (app.py per-clip sink) matches the card. The
   old indeterminate .co-loading-bar scan-line is gone — the native bar is the
   single source of truth. Class names span Gradio's progress markup defensively
   (the determinate fill + the "clip k/N · window p/P" label). */
.progress-bar,
.progress-level .progress-level-inner,
.progress-level-inner,
.meta-text + .progress-bar {
    background: var(--co-amber) !important;
    color: var(--co-accent-text, #0E1116) !important;
}
.progress-text,
.meta-text,
.meta-text-center {
    color: var(--co-amber) !important;
    font-family: var(--co-mono, ui-monospace, monospace) !important;
    letter-spacing: 0.01em;
}
.progress-level {
    background: var(--co-border) !important;
    border-radius: 3px;
}

/* ---- Proxy / honesty caveat (the value-region disclaimer, §5) ----------- */
.co-caveat {
    font-size: 12px;
    line-height: 1.55;
    color: var(--co-dim);
    border-left: 2px solid var(--co-amber);
    padding: 2px 0 2px 12px;
    margin: 8px 2px;
}
.co-caveat strong { color: var(--co-ink); }

/* ---- Footer (license posture) ------------------------------------------- */
.co-footer {
    margin-top: 18px;
    padding-top: 12px;
    border-top: 1px solid var(--co-border);
    font-size: 11.5px;
    color: var(--co-faint);
    line-height: 1.6;
}
.co-footer a { color: var(--co-dim); text-decoration: underline; text-underline-offset: 2px; }
.co-footer a:hover { color: var(--co-ink); }

/* Monospace numerals anywhere we explicitly tag tabular data. */
.co-mono, .co-tabnum { font-family: var(--co-mono); font-variant-numeric: tabular-nums; }

/* ---- Accessibility ------------------------------------------------------ */
/* Visible keyboard focus everywhere (don't rely on the faint default ring). */
:focus-visible {
    outline: 2px solid var(--co-amber) !important;
    outline-offset: 2px !important;
    border-radius: 4px;
}

@media (prefers-reduced-motion: reduce) {
    .co-quota .co-dot::before,
    .co-stat { animation: none !important; }
}

/* ---- Responsive: zones stack on small screens --------------------------- */
@media (max-width: 820px) {
    .co-wordmark { font-size: 22px; }
    .co-readout { padding: 7px; }
    .co-summary { grid-template-columns: repeat(2, 1fr); }
    .co-state { padding: 22px 16px; }
}
@media (max-width: 480px) {
    .co-summary { grid-template-columns: 1fr; }
    .co-tagline { display: block; }
}
""".strip()
