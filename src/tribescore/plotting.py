"""Build the timeline figure: derived brain-metric curves vs. time.

Renders the metric curves produced by :func:`tribescore.metrics.reduce_to_metrics`
as a single multi-line plot over the full video timeline, suitable for
display in the Gradio UI (``gr.Plot``).

Plotly is imported lazily inside the function so this module imports cleanly
even where plotly is not installed (e.g. a bare local box). Plotly is present
on the Space (see requirements.txt).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go


def plot_metric_timeline(
    time_axis: np.ndarray,
    metric_curves: Mapping[str, np.ndarray],
    *,
    title: str = "Derived brain metrics over time",
    x_title: str = "Time (s)",
    y_title: str = "Metric (normalized)",
) -> "go.Figure":
    """Create a Plotly line figure of metric curves against time.

    Parameters
    ----------
    time_axis:
        Monotonic time axis of shape ``(T,)``, in seconds (from
        :func:`tribescore.windowing.run_windowed`).
    metric_curves:
        Mapping ``{metric_name: curve}``; each ``curve`` has shape ``(T,)``
        and is plotted as one line.
    title, x_title, y_title:
        Figure / axis labels.

    Returns
    -------
    plotly.graph_objects.Figure
        One trace per metric, shared x-axis, unified hover.

    Raises
    ------
    ValueError
        If a curve length does not match ``time_axis``.

    Notes
    -----
    Figure assembly is deferred (TODO). The signature and the input contract
    -- a shared ``time_axis`` plus a name->curve mapping -- are stable; the
    body is a handful of ``fig.add_scatter`` calls plus layout, e.g.::

        import plotly.graph_objects as go
        fig = go.Figure()
        for name, curve in metric_curves.items():
            fig.add_scatter(x=time_axis, y=curve, mode="lines", name=name)
        fig.update_layout(title=title, xaxis_title=x_title,
                          yaxis_title=y_title, hovermode="x unified")
        return fig
    """
    time_axis = np.asarray(time_axis, dtype=float)
    for name, curve in metric_curves.items():
        if np.asarray(curve).shape[0] != time_axis.shape[0]:
            raise ValueError(
                f"curve '{name}' has length {np.asarray(curve).shape[0]} but "
                f"time_axis has length {time_axis.shape[0]}"
            )

    raise NotImplementedError(
        "plot_metric_timeline is implemented on the Space (plotly). See the "
        "docstring for the trace-assembly body."
    )
