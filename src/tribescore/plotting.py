"""Plotly timeline: the synchronized multi-channel brain-metric readout.

This is the **signature element** of the Cortical Observatory UI (PLAN.md §6):
stacked translucent metric curves on a shared 1 Hz time axis, with a crosshair
hover that shows every selected metric's value at the cursored second, and a
``plotly_click`` -> media-seek hook so clicking a spike jumps the video to that
moment ("scrub the media, watch the brain; click a spike, jump there").

The two public entry points:

* :func:`timeline_figure` -- the spec'd ``(t_axis, curves, *, selected)`` ->
  :class:`plotly.graph_objects.Figure` builder (PLAN.md §9 T-D). A **pure
  function** of its inputs (no globals, no model); plotly is imported lazily so
  this module imports cleanly where plotly is absent (a bare local box).
* :func:`seek_js` -- returns the tiny JS snippet that wires a Plotly
  ``plotly_click`` to ``document.getElementById('tm-video').currentTime = t``
  (the §6 seek fallback). Returned as a string so ``app.py`` can attach it to a
  ``gr.Plot`` without this module importing Gradio.

The legacy :func:`plot_metric_timeline` is kept for the older call site /
contract; new code should prefer :func:`timeline_figure`.

Theme tokens are duplicated here as plain hex (rather than importing
``theme.py``, which pulls in Gradio) so the figure styling stays self-contained
and the module remains a leaf with only a numpy hard-dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Theme tokens (mirror of theme.PALETTE; kept local to avoid a Gradio import).
# ---------------------------------------------------------------------------
_BG = "#0E1116"
_PANEL = "#161B22"
_INK = "#E8EDF2"
_DIM = "#8A93A0"
_GRID = "rgba(35,42,51,0.85)"   # #232A33 at ~panel contrast
_ZEROLINE = "rgba(49,59,71,0.9)"  # #313B47
_MONO = "JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
_SANS = "Inter, ui-sans-serif, system-ui, sans-serif"

#: Colorblind-safe scientific ramp (a viridis/cividis-style sample, PLAN.md §6).
#: Ordered so adjacent metrics stay distinguishable for deuteranopia/protanopia.
#: Stable per-metric color comes from indexing this by the metric's slot.
_RAMP: tuple[str, ...] = (
    "#FDE725",  # viridis yellow
    "#35B779",  # viridis green
    "#31688E",  # viridis blue
    "#B5367A",  # magenta (cividis-adjacent warm)
    "#FFB454",  # observatory amber (5th+)
    "#36D1C4",  # observatory cyan
    "#90D743",  # light green
    "#443983",  # deep indigo
)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """``'#RRGGBB'`` -> ``'rgba(r,g,b,alpha)'`` (for translucent fills)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def color_for(index: int) -> str:
    """Stable curve color for the metric in slot ``index`` (wraps the ramp)."""
    return _RAMP[index % len(_RAMP)]


def timeline_figure(
    t_axis: np.ndarray,
    curves: Mapping[str, np.ndarray],
    *,
    selected: Sequence[str] | None = None,
) -> "go.Figure":
    """Build the synchronized brain-metric timeline figure.

    Parameters
    ----------
    t_axis:
        Shared, monotonic time axis of shape ``(T,)`` in **seconds** (the 1 Hz
        grid from :func:`tribescore.windowing.stitch`).
    curves:
        Mapping ``{metric_name: curve}``; each ``curve`` has shape ``(T,)``
        (z-scored, smoothed -- the canonical analytic series).
    selected:
        Which metric names to draw, in the order their colors are assigned.
        ``None`` (default) draws every key of ``curves`` in iteration order.
        Names not present in ``curves`` are skipped.

    Returns
    -------
    plotly.graph_objects.Figure
        One translucent line+fill trace per selected metric on a shared x-axis,
        crosshair (``hovermode='x unified'``) hover with per-metric values, dark
        Cortical-Observatory styling, x ticks in seconds (tabular mono).

    Raises
    ------
    ValueError
        If a selected curve's length does not match ``t_axis``.

    Notes
    -----
    Pure function -- no globals are read or written and no model is touched.
    Plotly is imported lazily inside the body so importing this module never
    requires plotly. The colorblind-safe ramp is :data:`_RAMP`; color is
    assigned by the metric's position in ``selected`` (stable across renders so
    a metric keeps its color when others are toggled, as long as the caller
    passes a stable ``selected`` order).
    """
    import plotly.graph_objects as go  # lazy: keep module import-safe

    t_axis = np.asarray(t_axis, dtype=float)
    if selected is None:
        selected = list(curves.keys())

    # Keep only requested metrics that we actually have data for, preserving
    # the caller's order (which fixes color assignment).
    names = [n for n in selected if n in curves]

    fig = go.Figure()
    for slot, name in enumerate(names):
        curve = np.asarray(curves[name], dtype=float)
        if curve.shape[0] != t_axis.shape[0]:
            raise ValueError(
                f"curve '{name}' has length {curve.shape[0]} but t_axis has "
                f"length {t_axis.shape[0]}"
            )
        line_color = color_for(slot)
        fig.add_scatter(
            x=t_axis,
            y=curve,
            name=name,
            mode="lines",
            line=dict(color=line_color, width=2, shape="spline", smoothing=0.5),
            # Translucent fill to the zero baseline -> the "stacked translucent
            # curves" look without occluding lines beneath.
            fill="tozeroy",
            fillcolor=_hex_to_rgba(line_color, 0.10),
            hovertemplate="%{y:.2f} z<extra>" + name + "</extra>",
        )

    # Crosshair-style unified hover: one tooltip listing every metric at the
    # hovered second, with a vertical spike line down to the x-axis.
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_BG,
        plot_bgcolor=_PANEL,
        font=dict(family=_SANS, color=_INK, size=12),
        margin=dict(l=54, r=18, t=20, b=44),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=_PANEL,
            bordercolor="#313B47",
            font=dict(family=_MONO, color=_INK, size=12),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(family=_SANS, color=_DIM, size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        dragmode="zoom",
        showlegend=True,
    )
    fig.update_xaxes(
        title=dict(text="Time (s)", font=dict(family=_MONO, color=_DIM, size=11)),
        color=_DIM,
        gridcolor=_GRID,
        zeroline=False,
        showspikes=True,
        spikecolor="#FFB454",  # amber crosshair
        spikethickness=1,
        spikedash="dot",
        spikemode="across",
        spikesnap="cursor",
        tickfont=dict(family=_MONO, color=_DIM, size=10),
        ticksuffix=" s",
        rangemode="tozero",
    )
    fig.update_yaxes(
        title=dict(
            text="Activity (z)", font=dict(family=_MONO, color=_DIM, size=11)
        ),
        color=_DIM,
        gridcolor=_GRID,
        zeroline=True,
        zerolinecolor=_ZEROLINE,
        zerolinewidth=1,
        tickfont=dict(family=_MONO, color=_DIM, size=10),
    )
    return fig


def seek_js(video_id: str = "tm-video", plot_elem_id: str = "co-timeline") -> str:
    """Return JS that wires Plotly ``plotly_click`` -> media seek (PLAN.md §6).

    The returned snippet finds the Plotly graph div inside the Gradio
    ``gr.Plot`` (identified by ``plot_elem_id``), registers a ``plotly_click``
    handler, and on click sets ``document.getElementById(video_id).currentTime``
    to the clicked point's x (the timestamp in seconds) -- the guaranteed
    click->timestamp->seek floor from §6. It also moves a draggable playhead
    line via ``Plotly.relayout`` for the best-effort live readout, and is
    idempotent (re-binding the handler is safe across re-renders).

    Returned as a plain string so ``app.py`` can pass it to e.g.
    ``demo.load(js=...)`` or a component's ``.then(js=...)`` without this module
    importing Gradio. Pure function -- depends only on its arguments.
    """
    return f"""
() => {{
    const PLOT_ID = {plot_elem_id!r};
    const VIDEO_ID = {video_id!r};
    const bind = () => {{
        const host = document.getElementById(PLOT_ID);
        if (!host) return false;
        // Gradio wraps Plotly in a .js-plotly-plot div.
        const gd = host.querySelector('.js-plotly-plot') || host;
        if (!gd || !gd.on) return false;
        if (gd.dataset.coSeekBound === '1') return true;
        gd.dataset.coSeekBound = '1';
        gd.on('plotly_click', (ev) => {{
            if (!ev || !ev.points || !ev.points.length) return;
            const t = ev.points[0].x;
            const v = document.getElementById(VIDEO_ID);
            if (v && Number.isFinite(t)) {{
                try {{ v.currentTime = t; v.play && v.play().catch(() => {{}}); }} catch (e) {{}}
            }}
            // Best-effort: draw/move a playhead line at the clicked second.
            try {{
                if (window.Plotly && Number.isFinite(t)) {{
                    window.Plotly.relayout(gd, {{
                        'shapes': [{{
                            type: 'line', xref: 'x', yref: 'paper',
                            x0: t, x1: t, y0: 0, y1: 1,
                            line: {{ color: '#36D1C4', width: 1.5, dash: 'solid' }}
                        }}]
                    }});
                }}
            }} catch (e) {{}}
        }});
        return true;
    }};
    // Retry briefly: the Plotly div may mount a tick after the callback fires.
    if (!bind()) {{
        let n = 0;
        const id = setInterval(() => {{ if (bind() || ++n > 40) clearInterval(id); }}, 100);
    }}
}}
""".strip()


# ---------------------------------------------------------------------------
# Legacy contract (kept for the original call site). Prefer timeline_figure.
# ---------------------------------------------------------------------------
def plot_metric_timeline(
    time_axis: np.ndarray,
    metric_curves: Mapping[str, np.ndarray],
    *,
    title: str = "Derived brain metrics over time",
    x_title: str = "Time (s)",
    y_title: str = "Activity (z)",
    selected: Sequence[str] | None = None,
) -> "go.Figure":
    """Backwards-compatible alias that delegates to :func:`timeline_figure`.

    The original placeholder raised ``NotImplementedError`` (the body was to be
    written on the Space). It is now implemented in :func:`timeline_figure`;
    this wrapper preserves the older name/signature and forwards ``selected``.
    The ``title`` argument is accepted for compatibility but no longer rendered
    as a chart title (the section header in the UI carries the title instead).
    """
    fig = timeline_figure(time_axis, metric_curves, selected=selected)
    # Honor the legacy axis-title overrides if a caller passed custom ones.
    if x_title:
        fig.update_xaxes(title=dict(text=x_title))
    if y_title:
        fig.update_yaxes(title=dict(text=y_title))
    return fig
