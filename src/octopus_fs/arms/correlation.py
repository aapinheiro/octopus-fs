"""Arm 1/8 — Correlation coefficient (Pearson / Spearman).

Family: filter. Univariate. Cheapest arm; run it always.

What it measures: monotonic (Spearman) or linear (Pearson) association between
each feature and the target, ignoring every other feature.

Blind spots to state in the docstring (and in the report tooltip):
- misses interactions entirely (XOR gets a score of ~0)
- a feature perfectly correlated with another gets the same high score — this
  arm cannot detect redundancy
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from octopus_fs.core.base import ArmContext, BaseArm
from octopus_fs.core.registry import register
from octopus_fs.types import ArmFamily, TaskType


@register
class CorrelationArm(BaseArm):
    """Absolute Pearson or Spearman correlation with the target."""

    name = "correlation"
    family = ArmFamily.FILTER
    supported_tasks = frozenset({TaskType.REGRESSION, TaskType.BINARY})

    def _score(self, X: pd.DataFrame, y: pd.Series, ctx: ArmContext) -> np.ndarray:
        """Return |corr(x_j, y)| per column.

        TODO(you):
        - param `method`: "pearson" | "spearman" (default: "spearman" — it is
          rank-based, so it survives outliers and monotone transforms)
        - hint: X.corrwith(y, method=self.params["method"]).abs().to_numpy()
          keeps column alignment for free; a manual loop invites ordering bugs
        - a constant column yields NaN -> map to 0.0 explicitly
        - MULTICLASS is unsupported on purpose: correlation with an arbitrary
          class encoding is meaningless. Say so in the skip message.
        - stretch: return the p-value as a secondary score and expose both.
          If you do, remember direction="lower_is_better" for p-values.
        """
        raise NotImplementedError
