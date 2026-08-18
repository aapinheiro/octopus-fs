"""Octopus — the orchestrator. This is the object users actually touch.

Responsibilities (and nothing else):
  resolve config -> validate inputs -> infer task -> fit the shared estimator
  -> run each arm -> aggregate -> pack a SelectionResult.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from octopus_fs.types import SelectionResult


class Octopus:
    """High-level entry point for a feature selection run.

    Example:
        >>> result = Octopus(arms="fast", random_state=42).fit(X, y)
        >>> result.consensus.top(10)
    """

    def __init__(
        self,
        arms: str | list[str] = "all",
        task: Literal["auto", "regression", "binary", "multiclass"] = "auto",
        random_state: int | None = 42,
        n_jobs: int = 1,
        on_error: Literal["skip", "raise"] = "skip",
        consensus_method: str = "borda",
        estimator: object | None = None,
        **arm_params: dict[str, object],
    ) -> None:
        """Build a run configuration. No data is touched here.

        TODO(you): `arm_params` lets a user do
            Octopus(arms=["correlation"], correlation={"method": "spearman"})
        Convert everything into a RunConfig now so the whole configuration is
        validated before `fit` runs.
        """
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SelectionResult:
        """Run every configured arm and aggregate the results.

        TODO(you), suggested order:
        1. X, y = check_inputs(X, y)
        2. task = infer_task(y) if self.task == "auto" else self.task
        3. estimator = self._resolve_estimator(task)   # see below
        4. fit that estimator ONCE if any post-hoc arm needs it — refitting a
           model per arm is the single biggest waste in a naive implementation
        5. results = {name: arm.run(X, y, ctx) for name in arms}
           (parallelize with joblib.Parallel when n_jobs != 1, but only over
           arms that are not already internally parallel — nested parallelism
           will oversubscribe your cores and get SLOWER)
        6. consensus = aggregate(results, method=...)
        7. manifest = build_manifest(config, X, versions)
        8. return SelectionResult(...)
        """
        raise NotImplementedError

    def _resolve_estimator(self, task: object) -> object:
        """Pick the shared model used by rfe / tree_importance / shap / permutation.

        TODO(you): default to RandomForest* (no extra deps, handles mixed
        scales, gives both impurity importance and a fast TreeExplainer).
        If lightgbm is installed prefer LGBM* — much faster on wide tables.
        The user-supplied `estimator` always wins.

        Design question worth answering in a docstring: should this estimator be
        fitted on the FULL data? For discovery, yes. Inside a pipeline, it must
        only ever see the training fold — which is why OctopusSelector calls
        this per-fold rather than reusing a global fit.
        """
        raise NotImplementedError
