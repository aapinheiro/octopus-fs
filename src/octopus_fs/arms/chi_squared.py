"""Arm 2/8 — Chi-squared test of independence.

Family: filter. Classification only. Requires NON-NEGATIVE features.

What it measures: dependence between each feature and the class label, treating
the feature as a count/frequency. Classic for text (bag-of-words) and one-hot
categoricals; questionable on continuous features.

This is the first arm that must REFUSE data — it is the reference
implementation for `supports()`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from octopus_fs.core.base import ArmContext, BaseArm
from octopus_fs.core.registry import register
from octopus_fs.types import ArmFamily, TaskType


@register
class ChiSquaredArm(BaseArm):
    """Chi-squared statistic between each feature and the target."""

    name = "chi2"
    family = ArmFamily.FILTER
    supported_tasks = frozenset({TaskType.BINARY, TaskType.MULTICLASS})
    requires_non_negative = True

    def _score(self, X: pd.DataFrame, y: pd.Series, ctx: ArmContext) -> np.ndarray:
        """Return the chi2 statistic per column.

        TODO(you):
        - hint: sklearn.feature_selection.chi2 returns (stat, p_value)
        - decide which you expose. Statistic: higher_is_better. p-value:
          lower_is_better AND needs multiple-testing correction if you want to
          claim significance across 500 columns (Benjamini-Hochberg).
          Recommendation: expose the statistic, mention the caveat in the report.
        - param `bins`: optionally discretize continuous columns first
          (KBinsDiscretizer, strategy="quantile") — otherwise chi2 on a raw
          continuous column is close to meaningless. Default: don't, and warn.
        - override supports() to add a precise message naming the offending
          columns: 'chi2 requires non-negative values; negative in: age_delta'.
        """
        raise NotImplementedError
