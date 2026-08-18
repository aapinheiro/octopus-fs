"""Run manifest — what makes a selection auditable six months later.

If you only build one thing beyond the arms themselves, build this. 'Why does
the model use these 30 columns?' is a question that gets asked long after the
notebook is gone.
"""

from __future__ import annotations

import pandas as pd


def build_manifest(config: object, X: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    """Capture everything needed to explain and reproduce a run.

    TODO(you), suggested fields:
    - octopus_version, python_version
    - versions of numpy/pandas/sklearn/shap/lightgbm (importlib.metadata)
    - config fingerprint (see RunConfig.fingerprint)
    - input schema hash: sorted (column, dtype) pairs -> sha256. Compare this
      on the next run to detect schema drift for free.
    - n_rows, n_features, task, class balance
    - random_state, per-arm status and elapsed_s
    - created_at (UTC, ISO-8601)
    Deliberately NOT included: the data itself, or anything row-level.
    """
    raise NotImplementedError


def diff_manifests(old: dict[str, object], new: dict[str, object]) -> dict[str, object]:
    """What changed between two runs.

    TODO(you): the useful output is a short human-readable list — 'schema hash
    changed', 'sklearn 1.3 -> 1.5', 'shap arm skipped this time'. This is the
    hook a monitoring job would call to explain why today's selection differs
    from last month's.
    """
    raise NotImplementedError
