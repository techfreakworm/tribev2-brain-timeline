"""Gradio UI builders for the TRIBE v2 brain-score Space (Cortical Observatory).

Mirrors the qwen-image-editor shape (PLAN.md §6 "Code shape"): per-mode
``build_*_tab()`` builders each return a ``dict[str, gr.components.Component]``
so ``app.py`` can wire ``.click()`` / ``.change()`` handlers without reaching
into local scopes. **No event wiring lives here** -- app.py owns it.

The three modes (PLAN.md §2) all feed the *same* metric-timeline pipeline, so
each tab differs only in its input widget (+ an ``audio_only`` debug toggle on
the two media tabs). The shared right-hand readout is built once by
:func:`build_results`.

Help text flows through Gradio's native ``info=`` parameter (a dim subtitle
under each label). Components that don't support ``info=`` (``gr.Video``,
``gr.HTML``) carry their guidance in the label or adjacent markdown instead.
"""

from __future__ import annotations

import gradio as gr

# ---------------------------------------------------------------------------
# Metric catalogue (PLAN.md §5). The checkbox VALUES are these exact display
# names, so app.py can pass the selection straight through to
# ``plotting.timeline_figure(..., selected=...)`` and key into the curve dict.
# ---------------------------------------------------------------------------
METRICS_DEFAULT_ON: tuple[str, ...] = (
    "Attention",
    "Engagement / arousal",
    "Virality (proxy)",
)
METRICS_OPTIONAL: tuple[str, ...] = (
    "Language / semantic load",
    "Self-relevance / DMN",
)
METRIC_CHOICES: tuple[str, ...] = METRICS_DEFAULT_ON + METRICS_OPTIONAL

# One-line rationale per metric, surfaced as the checkbox group's ``info=``.
_METRIC_INFO = (
    "Brain-derived curves over time. Attention (dorsal/ventral + control), "
    "Engagement (sensory + STS), Virality (vmPFC/mOFC value proxy) are on by "
    "default; Language and Self-relevance/DMN are optional."
)

# Accepted upload suffixes per mode (PLAN.md §2 table).
_VIDEO_SUFFIXES = [".mp4", ".mov", ".mkv", ".avi", ".webm"]
_AUDIO_SUFFIXES = [".wav", ".mp3", ".flac", ".ogg"]

# Honesty caveat (PLAN.md §5) -- shown once near the metric toggles.
_PROXY_CAVEAT = (
    '<div class="co-caveat"><strong>Virality is a research proxy</strong>, not '
    "a guarantee. <code>facebook/tribev2</code> is cortical-only (no ventral "
    "striatum / NAcc), so this is the validated vmPFC/mPFC <em>complement</em>. "
    "The model's target was per-sample z-scored, so only <strong>relative "
    "temporal dynamics</strong> are interpretable &mdash; absolute scores are "
    "meaningless.</div>"
)


def _metric_toggles() -> dict[str, gr.components.Component]:
    """Metric-selection checkboxes + the proxy caveat, shared by every tab.

    Returns ``{metrics, proxy_note}``: a single ``gr.CheckboxGroup`` whose value
    is the list of selected metric display names (defaults = the three ON
    metrics), and the caveat HTML beneath it.
    """
    with gr.Group():
        gr.HTML('<div class="co-section-title">Metrics</div>')
        metrics = gr.CheckboxGroup(
            choices=list(METRIC_CHOICES),
            value=list(METRICS_DEFAULT_ON),
            label="Brain metrics to plot",
            info=_METRIC_INFO,
        )
        proxy_note = gr.HTML(_PROXY_CAVEAT)
    return dict(metrics=metrics, proxy_note=proxy_note)


def _audio_only_toggle() -> gr.components.Component:
    """The ``audio_only`` debug switch (Video/Audio tabs only, PLAN.md §11.6).

    Skips ASR (whisperx) + the gated Llama text path -> much faster, and lets
    the heavy pipeline be validated on-Space before Meta approval.
    """
    return gr.Checkbox(
        value=False,
        label="Audio-only (debug)",
        info=(
            "Skip speech-to-text + the gated Llama text path. Faster; "
            "validates the pipeline before model access is granted."
        ),
    )


def _run_button(label: str = "Score") -> gr.components.Component:
    """The primary action button (activation-hot, per the theme)."""
    return gr.Button(label, variant="primary", elem_classes=["co-run"])


# ===========================================================================
# Per-mode input tabs
# ===========================================================================
def build_video_tab() -> dict[str, gr.components.Component]:
    """Build the Video mode (primary) input column.

    Keys: ``video, sample_btn, metrics, proxy_note, audio_only, run_btn``.
    Full multimodal path: V-JEPA2 + DINOv2 + extracted-audio W2V-BERT + Llama.
    """
    with gr.Column():
        gr.HTML('<div class="co-section-title">Video &middot; full multimodal</div>')
        video = gr.Video(
            label="Video (.mp4 .mov .mkv .avi .webm, up to 5 min)",
            sources=["upload"],
            height=200,
        )
        sample_btn = gr.Button(
            "Try a sample clip", variant="secondary", size="sm",
            elem_classes=["co-sample"],
        )
        toggles = _metric_toggles()
        audio_only = _audio_only_toggle()
        run_btn = _run_button("Score video")
    return dict(
        video=video,
        sample_btn=sample_btn,
        audio_only=audio_only,
        run_btn=run_btn,
        **toggles,
    )


def build_audio_tab() -> dict[str, gr.components.Component]:
    """Build the Audio mode input column.

    Keys: ``audio, metrics, proxy_note, audio_only, run_btn``.
    Path: W2V-BERT + Llama (ASR word context).
    """
    with gr.Column():
        gr.HTML('<div class="co-section-title">Audio</div>')
        audio = gr.Audio(
            label="Audio (.wav .mp3 .flac .ogg, up to 5 min)",
            sources=["upload"],
            type="filepath",
        )
        toggles = _metric_toggles()
        audio_only = _audio_only_toggle()
        run_btn = _run_button("Score audio")
    return dict(
        audio=audio,
        audio_only=audio_only,
        run_btn=run_btn,
        **toggles,
    )


def build_text_tab() -> dict[str, gr.components.Component]:
    """Build the Text mode input column.

    Keys: ``text, metrics, proxy_note, run_btn``.
    Path: gTTS-synthesized speech -> W2V-BERT + Llama. (No ``audio_only`` here:
    text *is* the ASR-equivalent input, so the debug toggle doesn't apply.)
    """
    with gr.Column():
        gr.HTML('<div class="co-section-title">Text</div>')
        text = gr.Textbox(
            label="Text",
            info=(
                "Synthesized to speech (gTTS), then scored over the spoken "
                "length. English narrative works best."
            ),
            lines=6,
            placeholder=(
                "Paste a script or passage. It is read aloud and the average "
                "brain's response to that narration is plotted over time."
            ),
        )
        toggles = _metric_toggles()
        run_btn = _run_button("Score text")
    return dict(
        text=text,
        run_btn=run_btn,
        **toggles,
    )


# ===========================================================================
# Shared right-hand readout (the signature synchronized timeline + states)
# ===========================================================================

# Empty-state markup: inviting, one-line "what this does" (PLAN.md §6 states).
_EMPTY_HTML = """
<div class="co-state" id="co-empty">
  <div class="co-state-icon">&#129504;</div>
  <div class="co-state-title">Read a clip, watch the brain</div>
  <div class="co-state-body">
    Pick a mode on the left, drop in a clip (or text), choose your metrics, and
    press <strong>Score</strong>. You'll get a synchronized timeline of how the
    <em>average</em> brain would respond &mdash; scrub the media, watch the
    curves; click a spike to jump there.
  </div>
</div>
""".strip()

# The custom <video id="tm-video"> mount: the seek-fallback target (PLAN.md §6).
# Rendered as a placeholder until app.py swaps in a real <video> on result.
_VIDEO_PLACEHOLDER = """
<div class="co-video-wrap">
  <div class="co-video-empty">media appears here on Score</div>
</div>
""".strip()


def build_results() -> dict[str, gr.components.Component]:
    """Build the shared results readout (right hero zone).

    Keys:
    ``empty, loading, error, result_grp, media_html, timeline, summary,
    seek_hint``.

    Layout (one zone, four mutually-exclusive states app.py toggles ``visible``):
      * ``empty``     -- inviting empty state (visible at start).
      * ``loading``   -- real per-window progress + a scan-line bar.
      * ``error``     -- plain-voice, actionable error panel.
      * ``result_grp`` -- the media + timeline + summary (the win state):
          - ``media_html`` : custom ``<video id="tm-video">`` (gr.HTML) so a
            Plotly click can seek it (the §6 fallback the timeline drives).
          - ``timeline``   : the ``gr.Plot`` Plotly figure (elem_id
            ``co-timeline`` so ``plotting.seek_js`` can find it).
          - ``summary``    : the per-metric peak/mean strip (gr.HTML).
          - ``seek_hint``  : the "click a spike to jump" affordance.
    """
    with gr.Column(elem_classes=["co-readout"]):
        # --- Empty (initial) ---------------------------------------------
        empty = gr.HTML(_EMPTY_HTML, visible=True)

        # --- Loading ------------------------------------------------------
        with gr.Column(visible=False) as loading:
            gr.HTML(
                """
                <div class="co-state">
                  <div class="co-state-icon">&#128300;</div>
                  <div class="co-state-title">Scoring&hellip;</div>
                  <div class="co-state-body" id="co-progress">
                    Extracting features and predicting cortical activity. This
                    runs window-by-window over the clip.
                  </div>
                  <div class="co-loading-bar"></div>
                </div>
                """.strip()
            )

        # --- Error --------------------------------------------------------
        # app.py sets the inner text via this component's value on failure.
        error = gr.HTML(visible=False)

        # --- Result -------------------------------------------------------
        with gr.Column(visible=False) as result_grp:
            media_html = gr.HTML(_VIDEO_PLACEHOLDER)
            timeline = gr.Plot(
                label="Synchronized brain-metric timeline",
                show_label=False,
                elem_id="co-timeline",
            )
            seek_hint = gr.HTML(
                '<div class="co-seek-hint">click a spike to '
                '<kbd>seek</kbd> the media to that moment</div>'
            )
            summary = gr.HTML(elem_classes=["co-summary-wrap"])

    return dict(
        empty=empty,
        loading=loading,
        error=error,
        result_grp=result_grp,
        media_html=media_html,
        timeline=timeline,
        summary=summary,
        seek_hint=seek_hint,
    )


# ---------------------------------------------------------------------------
# Small HTML helpers app.py can call to render the dynamic states. Pure
# string builders (no Gradio state) so they stay unit-checkable.
# ---------------------------------------------------------------------------
def error_html(message: str) -> str:
    """Wrap an actionable, plain-voice message in the error-panel markup."""
    return (
        '<div class="co-state co-state-error">'
        '<div class="co-state-icon">&#9888;&#65039;</div>'
        '<div class="co-state-title">Couldn\'t score that</div>'
        f'<div class="co-state-body">{message}</div>'
        "</div>"
    )


def video_html(src: str, *, mime: str = "video/mp4") -> str:
    """Render the custom ``<video id="tm-video">`` the timeline seeks (§6).

    ``src`` is a URL or a Gradio-served file path (e.g. ``/file=...``). The
    element id is fixed (``tm-video``) so ``plotting.seek_js`` can target it.
    """
    return (
        '<div class="co-video-wrap">'
        '<video id="tm-video" controls preload="metadata" playsinline>'
        f'<source src="{src}" type="{mime}">'
        "Your browser can't play this clip."
        "</video></div>"
    )


def summary_html(stats: dict[str, dict]) -> str:
    """Render the per-metric summary strip from ``metrics.summary`` output.

    ``stats`` maps metric name -> ``{"peak": float, "peak_t": float|int,
    "mean": float, ...}`` (extra keys ignored). Values render in tabular mono
    (z-units); ``peak_t`` is the second of the peak, for the click-to-seek cue.
    """
    cards = []
    for name, s in stats.items():
        peak = s.get("peak")
        mean = s.get("mean")
        peak_t = s.get("peak_t")
        peak_str = f"{peak:+.2f} z" if isinstance(peak, (int, float)) else "&mdash;"
        sub_bits = []
        if isinstance(peak_t, (int, float)):
            sub_bits.append(f"@ {round(peak_t)} s")
        if isinstance(mean, (int, float)):
            sub_bits.append(f"mean {mean:+.2f}")
        sub = " &middot; ".join(sub_bits) if sub_bits else "&nbsp;"
        cards.append(
            '<div class="co-stat">'
            f'<div class="co-stat-name">{name}</div>'
            f'<div class="co-stat-val co-tabnum">{peak_str}</div>'
            f'<div class="co-stat-sub">{sub}</div>'
            "</div>"
        )
    return '<div class="co-summary">' + "".join(cards) + "</div>"
