"""Arm 3/8 — Mutual Information.

Family: filter. Works for regression and classification.

What it measures: how much knowing x_j reduces uncertainty about y — including
NON-LINEAR and non-monotonic relationships. This is the filter that catches
what correlation misses.

Costs more than correlation (k-nearest-neighbour density estimation) and is
stochastic — so `random_state` genuinely matters here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from octopus.core.base import ArmContext, BaseArm
from octopus.core.registry import register
from octopus.types import ArmFamily, TaskType


@register
class MutualInformationArm(BaseArm):
    """Estimated mutual information between each feature and the target."""

    name = "mutual_info"
    family = ArmFamily.FILTER
    supported_tasks = frozenset({TaskType.REGRESSION, TaskType.BINARY, TaskType.MULTICLASS})

    def _score(self, X: pd.DataFrame, y: pd.Series, ctx: ArmContext) -> np.ndarray:
        """Return estimated MI per column.

        TODO(you):
        - dispatch on ctx.task: mutual_info_regression vs mutual_info_classif
        - ALWAYS pass random_state=ctx.random_state — the kNN estimator adds
          noise to break ties, so without a seed this arm is not reproducible
          and your manifest becomes a lie
        - `discrete_features`: pass a boolean mask from
          validation.split_column_types(). Getting this wrong is the most
          common source of nonsense MI scores
        - param `n_neighbors` (default 3): higher = smoother, lower variance
          estimate, but blurs sharp relationships
        - MI is unbounded and scale-free-ish: never compare its raw magnitude
          with a correlation. Another argument for rank-based consensus.
        """
        raise NotImplementedError
