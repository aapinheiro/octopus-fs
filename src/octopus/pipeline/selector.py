"""OctopusSelector — the production face.

The entire point of this class: feature selection is PART OF THE MODEL. If you
select on the full dataset and then cross-validate, your CV score is optimistic
and you will not find out until production. As a sklearn transformer, selection
is refit on each training fold automatically.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OctopusSelector(BaseEstimator, TransformerMixin):
    """Select features with Octopus inside a scikit-learn pipeline.

    Example:
        >>> Pipeline([("sel", OctopusSelector(arms="fast", rule="top_k", k=30)),
        ...           ("clf", LGBMClassifier())])
    """

    def __init__(
        self,
        arms: str | list[str] = "fast",
        rule: Literal["top_k", "threshold", "quantile", "min_arms"] = "top_k",
        k: int | None = 30,
        threshold: float | None = None,
        random_state: int | None = 42,
        n_jobs: int = 1,
    ) -> None:
        """Store params ONLY.

        sklearn contract: __init__ assigns every argument to a same-named
        attribute and does nothing else — no validation, no computation.
        get_params/set_params and clone() depend on it. Validate in fit().
        """
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OctopusSelector":
        """Run the arms on this fold and decide the subset.

        TODO(you):
        1. result = Octopus(...).fit(X, y)
        2. self.selected_ = apply_rule(result.consensus, self.rule, ...)
        3. set the sklearn-expected attributes:
           self.n_features_in_, self.feature_names_in_, self.support_ (bool mask)
        4. keep self.result_ for introspection, but make it opt-in
           (`keep_result=False` by default) — a fitted pipeline that carries
           eight full score dicts per fold gets fat fast
        5. guard: if the rule selects 0 features, raise instead of returning an
           empty frame. An empty X fails 10 steps later with a cryptic error.
        """
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Subset the columns.

        TODO(you): check_is_fitted(self), then verify the incoming columns are a
        superset of self.selected_ and raise a clear error naming the missing
        ones. Silent column drift is a top cause of production model decay.
        """
        raise NotImplementedError

    def get_feature_names_out(self, input_features: object = None) -> object:
        """Required for `set_output(transform='pandas')` and ColumnTransformer."""
        raise NotImplementedError
