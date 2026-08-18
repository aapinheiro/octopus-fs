"""Stability: does this selection survive a slightly different dataset?

An unstable selection is the failure mode nobody notices — the pipeline runs,
the model trains, and next month a different 30 features come out. For anyone
who has to audit or govern a model, stability is more actionable than the
ranking itself.
"""

from __future__ import annotations

import pandas as pd


def bootstrap_selection(
    X: pd.DataFrame,
    y: pd.Series,
    config: object,
    n_bootstrap: int = 20,
    k: int = 30,
) -> dict[str, float]:
    """Selection frequency per feature across bootstrap resamples.

    TODO(you):
    - resample rows with replacement (stratified on y for classification),
      seeded from config.random_state + i so each round is reproducible
    - run the SAME arms on each resample, take top-k of the consensus
    - return {feature: times_selected / n_bootstrap}
    - cost warning: this multiplies the whole run by n_bootstrap. Default to
      the cheap arms only, and expose n_bootstrap prominently.
    """
    raise NotImplementedError


def nogueira_stability(selections: list[set[str]], n_features: int) -> float:
    """Nogueira et al. (2018) stability index for a set of selections.

    One number in roughly [0, 1]: 1 = identical subsets every time, ~0 = no
    better than random. Corrects for subset size, unlike a naive mean Jaccard.

    TODO(you): implement from the paper's formula; it is ~10 lines of numpy.
    Good target for a unit test with a known-value fixture.
    """
    raise NotImplementedError
