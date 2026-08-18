"""Individual plotly figures. One function per panel, each returning a Figure.

Keeping them separate makes each one usable directly in a notebook
(`plots.agreement_heatmap(result).show()`), which is how you will actually
debug them.
"""

from __future__ import annotations

from octopus.types import SelectionResult


def consensus_bar(result: SelectionResult, k: int = 25) -> object:
    """Horizontal bar of the top-k aggregated scores.

    TODO(you): color by disagreement, not by score — the score is already the
    bar length, so spending color on it wastes the channel.
    """
    raise NotImplementedError


def agreement_heatmap(result: SelectionResult) -> object:
    """Spearman correlation between arms' full rankings (n_arms x n_arms).

    TODO(you): a diverging colorscale centered at 0. Expect the filters to
    cluster together and the post-hoc arms to cluster together — when they
    don't, that is a finding worth reading the data over.
    """
    raise NotImplementedError


def rank_slope(result: SelectionResult, features: list[str] | None = None) -> object:
    """Bump chart: one line per feature, x = arm, y = rank.

    TODO(you): limit to ~15 features or it becomes spaghetti. Default to the
    consensus top-10 plus the 5 most disagreed-upon.
    """
    raise NotImplementedError


def stability_plot(result: SelectionResult) -> object:
    """Selection frequency across bootstrap resamples."""
    raise NotImplementedError
