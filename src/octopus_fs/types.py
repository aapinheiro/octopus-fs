"""The shared contract.

Everything in Octopus flows through these objects. Get this file right and the
rest of the library is mostly plumbing — so implement it FIRST, and write its
tests before writing any arm.

Design rule: an arm produces *scores*, never *decisions*. Turning scores into a
selected subset is the job of `pipeline.rules`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np


class TaskType(str, Enum):
    """Supervised task inferred from y (or forced by the user)."""

    REGRESSION = "regression"
    BINARY = "binary"
    MULTICLASS = "multiclass"


class ArmFamily(str, Enum):
    """How an arm gets its scores. Drives cost estimates and report grouping."""

    FILTER = "filter"        # statistics on (X, y) only
    EMBEDDED = "embedded"    # falls out of fitting one model
    WRAPPER = "wrapper"      # repeatedly refits a model over feature subsets
    POST_HOC = "post_hoc"    # explains an already fitted model


ArmStatus = Literal["ok", "skipped", "failed"]
Direction = Literal["higher_is_better", "lower_is_better"]


@dataclass(frozen=True)
class ArmResult:
    """The output of one arm. Immutable on purpose — results are artifacts."""

    arm: str
    family: ArmFamily
    status: ArmStatus
    scores: dict[str, float] = field(default_factory=dict)
    params: dict[str, object] = field(default_factory=dict)
    direction: Direction = "higher_is_better"
    elapsed_s: float = 0.0
    message: str | None = None  # why it was skipped/failed
    warnings: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- hints --
    @property
    def ranks(self) -> dict[str, int]:
        """Feature -> rank, 1 = most important.

        TODO(you):
        - respect `self.direction` (p-values are lower_is_better!)
        - ties must get the SAME rank ("min" method) — otherwise column order
          silently becomes a tiebreaker and results stop being deterministic
        - hint: scipy.stats.rankdata(..., method="min") on the negated array
        """
        raise NotImplementedError

    def normalized(self) -> dict[str, float]:
        """Scores rescaled to [0, 1], higher = better.

        TODO(you): min-max is fine, BUT decide what to do when all scores are
        equal (division by zero) and when scores are unbounded (SHAP mean |phi|
        vs. a correlation in [-1, 1] are not comparable in absolute terms).
        This is exactly why consensus should aggregate RANKS, not raw scores.
        """
        raise NotImplementedError

    def top(self, k: int) -> list[str]:
        """The k best features according to this arm."""
        raise NotImplementedError


@dataclass(frozen=True)
class ConsensusResult:
    """Aggregated view across arms."""

    scores: dict[str, float]           # aggregated score, higher = better
    per_arm_ranks: dict[str, dict[str, int]]  # arm -> {feature: rank}
    method: str
    stability: dict[str, float] | None = None  # feature -> selection frequency

    def top(self, k: int) -> list[str]:
        """The k best features after aggregation."""
        raise NotImplementedError

    def disagreement(self) -> dict[str, float]:
        """Per-feature spread of ranks across arms.

        TODO(you): std or IQR of the ranks. This is one of the most useful
        columns in the report — it points at features whose importance depends
        entirely on which technique you asked.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class SelectionResult:
    """What `Octopus.fit()` returns."""

    arms: dict[str, ArmResult]
    consensus: ConsensusResult
    feature_names: list[str]
    task: TaskType
    manifest: dict[str, object]  # see pipeline.artifacts.build_manifest

    def to_frame(self) -> "np.ndarray":  # noqa: UP037
        """Tidy table: one row per (feature, arm) with score and rank.

        TODO(you): return a pandas.DataFrame. It is the single input the report
        layer and the notebook user both want. Annotate it properly once you
        decide whether pandas is a hard dependency (it is, per pyproject).
        """
        raise NotImplementedError

    def to_html(self, path: str) -> str:
        """Render the discovery dashboard. Delegates to octopus_fs.report.

        TODO(you): import the report module INSIDE this method and raise
        MissingDependencyError with the install hint if jinja2/plotly are absent.
        """
        raise NotImplementedError

    def to_json(self, path: str) -> None:
        """Persist the result as a portable artifact (no pickles).

        TODO(you): pickles of sklearn objects rot across versions. Serialize
        scores + manifest as JSON so a selection made today is still readable
        in two years. Round floats — full float64 noise is not signal.
        """
        raise NotImplementedError
