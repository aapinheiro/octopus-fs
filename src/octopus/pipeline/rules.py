"""Turning consensus scores into a concrete subset. Deliberately separate.

Keeping the cut here (and not inside the arms) means you can re-cut a saved
result with a different rule without recomputing anything.
"""

from __future__ import annotations

from octopus.types import ConsensusResult


def top_k(consensus: ConsensusResult, k: int) -> list[str]:
    """The k highest-scoring features.

    TODO(you): deterministic tie-breaking. When scores tie, sort by feature name
    — otherwise dict ordering decides and two identical runs disagree.
    """
    raise NotImplementedError


def threshold(consensus: ConsensusResult, value: float) -> list[str]:
    """Features whose aggregated score exceeds `value`."""
    raise NotImplementedError


def quantile(consensus: ConsensusResult, q: float) -> list[str]:
    """Top q fraction of features. Scales across datasets better than a fixed k."""
    raise NotImplementedError


def min_arms(arms: dict[str, object], k: int, min_votes: int) -> list[str]:
    """Features in the top-k of at least `min_votes` arms.

    The most intuitive rule for stakeholders: 'six of eight techniques agree
    this feature matters'. Consider making it the default in the report.
    """
    raise NotImplementedError
