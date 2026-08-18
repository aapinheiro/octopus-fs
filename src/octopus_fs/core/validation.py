"""Input validation and task inference — the boring code that saves you later.

Everything here runs ONCE per run, before any arm, so a bad input fails in
milliseconds instead of after the SHAP arm has been running for 20 minutes.
"""

from __future__ import annotations

import pandas as pd

from octopus_fs.types import TaskType


def check_inputs(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Validate and normalize (X, y).

    TODO(you), in order:
    1. accept np.ndarray too -> wrap into a DataFrame with f0..fn names
    2. len(X) == len(y), and align indexes (a misaligned index silently
       shuffles your target — this is the #1 real-world bug here)
    3. reject empty frames, duplicated column names, all-constant columns
       (warn + drop; a constant column breaks correlation with a ZeroDivision)
    4. decide the NaN policy ONCE and document it: most sklearn estimators
       reject NaN, mutual_info too. Options: raise / impute / arm-by-arm.
       Recommendation for v0.1: raise, with a clear message. Imputation is a
       modeling decision and should not be smuggled inside a selection library.
    5. return copies, never mutate the caller's frame
    """
    raise NotImplementedError


def infer_task(y: pd.Series) -> TaskType:
    """Infer regression / binary / multiclass from y.

    TODO(you): heuristic —
    - non-numeric dtype or few unique values -> classification
    - nunique == 2 -> BINARY, else MULTICLASS
    - float dtype with many unique values -> REGRESSION
    Beware the trap: an int-encoded target with 10 classes and a genuine
    integer count target look identical. Log the inferred task at INFO level
    and let the user override with task=...
    """
    raise NotImplementedError


def split_column_types(X: pd.DataFrame) -> dict[str, list[str]]:
    """Group columns into numeric / categorical / boolean.

    TODO(you): several arms need this — chi2 wants non-negative (often
    categorical/count) features, correlation wants numeric. Returning the
    grouping once avoids each arm re-deriving it.
    """
    raise NotImplementedError
