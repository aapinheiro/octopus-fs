"""Rank aggregation — the actual intellectual contribution of this library.

Why ranks and not scores: an MI of 0.34, a chi2 statistic of 812 and a mean
|SHAP| of 0.007 live on incomparable scales. Normalizing them to [0,1] and
averaging looks reasonable and is quietly wrong — it makes the aggregate
sensitive to each arm's score DISTRIBUTION rather than its opinion. Ranks are
the common currency.

This is also the most testable module in the repo: pure functions, no I/O, no
sklearn. Property-based tests with hypothesis belong here.
"""

from __future__ import annotations

from typing import Literal

from octopus.types import ArmResult, ConsensusResult


def borda(rankings: dict[str, dict[str, int]], weights: dict[str, float] | None = None) -> dict[str, float]:
    """Borda count: each arm gives (n - rank) points to each feature.

    TODO(you):
    - points_j = sum over arms of (n_features - rank_{arm,j})
    - handle arms that ranked a SUBSET of features (a skipped arm contributes
      nothing; an arm that dropped constant columns must not silently penalize
      them). Decide: missing -> worst rank, or exclude the arm from that
      feature's average? Document the choice; they give different answers.
    - `weights` lets a user trust SHAP more than chi2. Default: uniform.
    """
    raise NotImplementedError


def mean_reciprocal_rank(rankings: dict[str, dict[str, int]]) -> dict[str, float]:
    """Mean of 1/rank across arms.

    Contrast with Borda: MRR is dominated by the TOP of each list, so a feature
    ranked #1 by two arms and #400 by six can still win. That is desirable when
    you want to surface niche-but-strong signals, and undesirable when you want
    broad agreement. Offer both; explain the difference in the report.
    """
    raise NotImplementedError


def robust_rank_aggregation(rankings: dict[str, dict[str, int]]) -> dict[str, float]:
    """RRA (Kolde et al., 2012): p-value that a feature ranks this well by chance.

    TODO(you): stretch goal for v0.2. The payoff is a principled significance
    cut instead of an arbitrary top-k, and it downweights features that are
    merely mediocre everywhere. Returns lower_is_better p-values — remember to
    flip them before mixing with the other methods.
    """
    raise NotImplementedError


def aggregate(
    arms: dict[str, ArmResult],
    method: Literal["borda", "mean_rank", "rra"] = "borda",
    weights: dict[str, float] | None = None,
) -> ConsensusResult:
    """Public entry point used by the runner.

    TODO(you):
    1. drop arms whose status != "ok" (but keep them in per_arm_ranks as empty
       so the report can show 'chi2: skipped — negative values')
    2. build {arm: result.ranks}
    3. dispatch on method
    4. return ConsensusResult with per_arm_ranks attached, so nothing
       downstream ever has to recompute ranks
    """
    raise NotImplementedError
