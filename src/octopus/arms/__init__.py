"""The eight arms.

Importing this package is what populates the registry — every module below
registers its class on import. Keep the imports here even though linters will
call them unused (hence the noqa).

Suggested implementation order (each one teaches the next):
    1. correlation      — simplest possible arm, gets the contract right
    2. mutual_info      — same shape, non-linear, introduces discrete_features
    3. chi2             — first arm that must REFUSE data (supports())
    4. tree_importance  — first arm that needs a fitted estimator
    5. lasso            — first arm that needs scaling + a path/alpha choice
    6. permutation      — first arm with a scoring metric and repeats
    7. shap             — first optional dependency
    8. rfe              — most expensive, benefits from everything above
"""

from octopus.arms import (  # noqa: F401
    chi_squared,
    correlation,
    lasso,
    mutual_info,
    permutation,
    rfe,
    shap_values,
    tree_importance,
)
