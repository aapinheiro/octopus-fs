"""Arm 6/8 — Permutation Importance.

Family: post-hoc. Model-agnostic.

What it measures: how much a chosen metric DROPS when a single column is
shuffled. Unlike impurity importance, it is measured in units of model
performance — which is why stakeholders understand it.

The decision that changes the meaning of the output: permute on train or on a
held-out set? On train, it measures what the model USED. On validation, it
measures what actually GENERALIZES. Overfitted models make these wildly
different. Default to a held-out split and make the choice explicit.

Also: with correlated features, shuffling one leaves its twin intact, so both
look unimportant. Report that caveat next to the numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from octopus_fs.core.base import ArmContext, BaseArm
from octopus_fs.core.registry import register
from octopus_fs.types import ArmFamily, TaskType


@register
class PermutationImportanceArm(BaseArm):
    """Mean metric drop under column shuffling."""

    name = "permutation"
    family = ArmFamily.POST_HOC
    supported_tasks = frozenset({TaskType.REGRESSION, TaskType.BINARY, TaskType.MULTICLASS})
    requires_estimator = True

    def _score(self, X: pd.DataFrame, y: pd.Series, ctx: ArmContext) -> np.ndarray:
        """Return the mean importance per column.

        TODO(you):
        - hint: sklearn.inspection.permutation_importance -> .importances_mean
        - params: `n_repeats` (default 5-10; cost is linear in it),
          `scoring` (default None -> estimator's own score; be explicit instead:
          roc_auc for binary, r2 for regression), `eval_on` in
          {"holdout", "train"} (default "holdout")
        - pass random_state and n_jobs from ctx
        - .importances_std is free and valuable — stash it in
          ArmResult.params["std"] so the report can draw error bars. A feature
          whose importance is 0.01 ± 0.03 is noise, and only the std shows it.
        - negative importances are normal (shuffling helped) — clip to 0 for
          RANKING but keep the raw value for display, and document the choice
        """
        raise NotImplementedError
