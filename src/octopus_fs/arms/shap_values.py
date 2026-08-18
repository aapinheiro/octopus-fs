"""Arm 7/8 — SHAP values.

Family: post-hoc. Optional dependency (`pip install "octopus_fs[shap]"`).

What it measures: the average magnitude of each feature's contribution to
individual predictions, mean(|phi_j|). Additive, locally faithful, and the only
arm that also gives you per-row explanations — which is why the discovery
report should surface a beeswarm, not just a bar chart.

Cost is the reason it is optional: TreeExplainer is fast, but KernelExplainer on
a wide table can take hours. Subsample aggressively and say so in the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from octopus_fs.core.base import ArmContext, BaseArm
from octopus_fs.core.registry import register
from octopus_fs.types import ArmFamily, TaskType


@register
class ShapArm(BaseArm):
    """Global importance as mean(|SHAP value|) per feature."""

    name = "shap"
    family = ArmFamily.POST_HOC
    supported_tasks = frozenset({TaskType.REGRESSION, TaskType.BINARY, TaskType.MULTICLASS})
    requires_estimator = True
    optional_extra = "shap"

    def _score(self, X: pd.DataFrame, y: pd.Series, ctx: ArmContext) -> np.ndarray:
        """Return mean(|phi_j|) per column.

        TODO(you):
        - import shap INSIDE this method; a top-level import would make the
          whole library depend on it
        - explainer choice matters:
            tree ensembles     -> shap.TreeExplainer (exact, fast)
            linear models      -> shap.LinearExplainer
            anything else      -> shap.Explainer (auto) with a background sample
        - param `max_samples` (default ~1000): subsample ROWS, stratified on y,
          seeded. Record the actual n used in params — a SHAP ranking from 200
          rows and one from 200k rows are not the same artifact
        - shape landmine: for multiclass, shap returns a list (or 3D array) of
          (n_samples, n_features) per class. Reduce with mean(|.|) over both
          samples and classes, and unit-test that shape handling — it is where
          this arm will break when shap bumps a major version
        - stretch: persist the raw values (parquet) so the report can draw a
          beeswarm without recomputation
        """
        raise NotImplementedError
