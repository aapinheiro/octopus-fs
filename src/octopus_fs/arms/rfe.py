"""Arm 8/8 — Recursive Feature Elimination.

Family: wrapper. By far the most expensive arm: it refits the model N times,
dropping the weakest feature(s) each round.

What it measures: the elimination ORDER. Note the shape mismatch with every
other arm — RFE natively produces a ranking, not a score. Converting a ranking
into a score is trivial; the reverse is not. This arm is the reason the whole
consensus layer aggregates ranks.

Implement this one LAST: it needs the estimator plumbing from tree_importance,
the scaling lesson from lasso, and a cost budget the others do not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from octopus_fs.core.base import ArmContext, BaseArm
from octopus_fs.core.registry import register
from octopus_fs.types import ArmFamily, TaskType


@register
class RFEArm(BaseArm):
    """Feature ranking from recursive elimination."""

    name = "rfe"
    family = ArmFamily.WRAPPER
    supported_tasks = frozenset({TaskType.REGRESSION, TaskType.BINARY, TaskType.MULTICLASS})
    requires_estimator = True

    def _score(self, X: pd.DataFrame, y: pd.Series, ctx: ArmContext) -> np.ndarray:
        """Return a score derived from RFE's elimination ranking.

        TODO(you):
        - hint: sklearn.feature_selection.RFE(...).ranking_ gives 1 for kept
          features and increasing integers for the order of elimination
        - convert to higher_is_better: score = n_features - ranking + 1
          (or set direction="lower_is_better" and return ranking_ raw — pick
          one, document it, and let ArmResult.ranks do the work)
        - param `step` (default 0.1 = drop 10% per round): the single biggest
          cost lever. step=1 on 500 features means 500 model fits
        - param `n_features_to_select` (default 1): going all the way down gives
          a complete ordering, which is what consensus wants
        - RFECV is the better tool when the user wants a subset SIZE rather than
          an ordering. Consider exposing it as a separate `rfecv` arm later
        - COST GUARD: estimate n_fits before starting and raise (or warn loudly)
          above a threshold. This arm is how someone accidentally launches a
          6-hour job from a notebook cell.
        """
        raise NotImplementedError
