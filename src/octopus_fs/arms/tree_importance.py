"""Arm 4/8 — Tree-based (impurity) feature importance.

Family: embedded. Comes free from fitting one tree ensemble.

What it measures: total impurity decrease attributable to each feature across
all splits.

Known bias — put this in the report tooltip, it is the most misused metric in
applied ML: impurity importance is inflated for high-cardinality and continuous
features, because they offer more possible split points. Two features carrying
the same information split the credit between them. This arm exists partly so
the report can SHOW that bias next to permutation and SHAP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from octopus_fs.core.base import ArmContext, BaseArm
from octopus_fs.core.registry import register
from octopus_fs.types import ArmFamily, TaskType


@register
class TreeImportanceArm(BaseArm):
    """`feature_importances_` from the shared tree ensemble."""

    name = "tree_importance"
    family = ArmFamily.EMBEDDED
    supported_tasks = frozenset({TaskType.REGRESSION, TaskType.BINARY, TaskType.MULTICLASS})
    requires_estimator = True

    def _score(self, X: pd.DataFrame, y: pd.Series, ctx: ArmContext) -> np.ndarray:
        """Return the impurity importance per column.

        TODO(you):
        - use ctx.estimator (already fitted by the runner) — do NOT fit again
        - guard with hasattr(est, "feature_importances_") and raise
          ArmNotApplicableError with a helpful message otherwise (e.g. someone
          passed a LogisticRegression as estimator)
        - lightgbm: importance_type="gain" is closer to sklearn's than "split";
          record which one you used in `params` so the manifest explains the
          numbers later
        - the vector already sums to 1 for sklearn ensembles — do not
          re-normalize, it destroys the only interpretable property it has
        """
        raise NotImplementedError
