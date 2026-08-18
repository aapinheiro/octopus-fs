"""Arm 5/8 — L1 regularization (Lasso).

Family: embedded. The only arm that produces exact zeros, i.e. a real subset.

What it measures: the coefficient each feature keeps when the model is
penalized for using features at all.

Two traps that make or break this arm:
1. SCALING. L1 penalizes coefficient magnitude, and coefficient magnitude
   depends on the feature's unit. Without standardization this arm reports the
   units of your columns, not their importance.
2. CORRELATED FEATURES. Lasso arbitrarily keeps one of a correlated group and
   zeroes the rest — a tiny data change flips which one. That instability is
   real signal for the stability panel, but must never be read as
   'the other features are useless'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from octopus.core.base import ArmContext, BaseArm
from octopus.core.registry import register
from octopus.types import ArmFamily, TaskType


@register
class LassoArm(BaseArm):
    """Absolute L1 coefficients from a standardized linear model."""

    name = "lasso"
    family = ArmFamily.EMBEDDED
    supported_tasks = frozenset({TaskType.REGRESSION, TaskType.BINARY, TaskType.MULTICLASS})

    def _score(self, X: pd.DataFrame, y: pd.Series, ctx: ArmContext) -> np.ndarray:
        """Return |coef_| per column.

        TODO(you):
        - build a Pipeline([StandardScaler(), model]) INSIDE this arm; never
          mutate the caller's X
        - regression -> LassoCV; classification -> LogisticRegression(
          penalty="l1", solver="liblinear"|"saga", C=...) or LogisticRegressionCV
        - param `alpha`/`C`: default to the CV-selected value, but expose it and
          RECORD the chosen value in params — 'which alpha did we use' is the
          first question anyone asks about a lasso selection
        - multiclass gives coef_ with shape (n_classes, n_features): reduce with
          mean(|coef|) over classes and say so in the docstring
        - exact zeros are meaningful. Do not add epsilon to make ranks pretty;
          let ties at zero be ties.
        """
        raise NotImplementedError
