"""Assemble the HTML dashboard.

Hard requirement: ONE self-contained .html file, no CDN, no server. It has to
survive being emailed to a stakeholder, opened offline, and attached to a
ticket. That constraint drives every decision below.
"""

from __future__ import annotations

from octopus.types import SelectionResult


def build_report(result: SelectionResult, path: str, title: str = "Octopus") -> str:
    """Render `result` to a standalone HTML file and return the path.

    TODO(you):
    1. import jinja2/plotly here; raise MissingDependencyError with the
       `pip install "octopus[report]"` hint if absent
    2. build each figure with plotly and embed with
       fig.to_html(full_html=False, include_plotlyjs=False), then include
       plotly.js ONCE inline (include_plotlyjs='inline' on the first figure).
       Otherwise a 5-figure report is 5x 3MB.
    3. render templates/report.html.j2 with the figures + tables
    4. write UTF-8, return path

    Panels, in order of usefulness (build them in this order):
      a) Consensus table — feature, aggregated rank, per-arm ranks, disagreement
      b) Agreement heatmap — Spearman correlation BETWEEN arms' rankings. Often
         the most surprising panel: it shows which techniques are redundant on
         YOUR data, and it is what justifies not running all eight next time.
      c) Rank slope chart / bump chart — how each feature moves across arms
      d) Stability panel — selection frequency across bootstraps (if computed)
      e) Skipped arms panel — arm, reason. Never hide a skip.
      f) Run manifest — collapsible <details> at the bottom
    """
    raise NotImplementedError
