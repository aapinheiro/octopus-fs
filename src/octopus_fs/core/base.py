"""The BaseArm contract — implement this once, get eight arms for free.

Read this file before writing any arm. Every arm is a small class that answers
three questions:

    1. Can I run on this task and this data?   -> supports()
    2. What are my scores?                     -> _score()
    3. What did I actually do?                 -> params reported back

The template method `run()` handles timing, error policy and result packing so
no arm ever has to. Do not override `run()`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
import pandas as pd

from octopus_fs.types import ArmFamily, ArmResult, TaskType


@dataclass
class ArmContext:
    """Everything an arm may need beyond (X, y).

    Passing a context object instead of 12 keyword arguments keeps the arm
    signature stable as the library grows.
    """

    task: TaskType
    random_state: int | None
    n_jobs: int
    estimator: object | None = None  # shared fitted model for post-hoc arms
    feature_names: list[str] | None = None


class BaseArm(abc.ABC):
    """Abstract base for the eight techniques."""

    name: str
    family: ArmFamily
    supported_tasks: frozenset[TaskType]
    requires_estimator: bool = False
    requires_non_negative: bool = False
    optional_extra: str | None = None  # e.g. "shap"

    def __init__(self, **params: object) -> None:
        """Store arm-specific params; validate them here, not at run time."""
        self.params = params
        # TODO(you): validate params eagerly and raise InvalidConfigError.
        # Failing at construction beats failing 40 minutes into a pipeline.

    # ------------------------------------------------------------ contract --
    @abc.abstractmethod
    def _score(self, X: pd.DataFrame, y: pd.Series, ctx: ArmContext) -> np.ndarray:
        """Return one score per column of X, in column order.

        Contract:
        - length == X.shape[1], no NaNs (map failures to 0.0 or -inf explicitly)
        - respect `direction`: if you return p-values, set
          `direction = "lower_is_better"` on the class
        - must be deterministic given ctx.random_state
        """

    def supports(self, X: pd.DataFrame, y: pd.Series, ctx: ArmContext) -> tuple[bool, str]:
        """Return (can_run, reason_if_not).

        TODO(you): default implementation checks
        1. ctx.task in self.supported_tasks
        2. self.requires_non_negative and (X < 0).any().any()
        3. self.requires_estimator and ctx.estimator is None
        4. self.optional_extra installed (importlib.util.find_spec)
        Arms override this only for their own extra rules.
        """
        raise NotImplementedError

    # -------------------------------------------------------------- engine --
    def run(self, X: pd.DataFrame, y: pd.Series, ctx: ArmContext) -> ArmResult:
        """Template method — DO NOT override.

        TODO(you):
        1. ok, reason = self.supports(...); if not ok -> ArmResult(status="skipped")
        2. t0 = time.perf_counter()
        3. scores = self._score(X, y, ctx)
        4. assert len(scores) == X.shape[1]  (catch the classic off-by-one when
           an arm drops constant columns internally)
        5. zip into dict(zip(X.columns, scores)) and pack an ArmResult
        6. wrap 3-5 in try/except -> status="failed", message=repr(exc)
           (never let one arm kill the other seven)
        """
        raise NotImplementedError
