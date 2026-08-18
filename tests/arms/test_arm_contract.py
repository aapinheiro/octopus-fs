"""One parameterized suite every arm must pass. Cheaper than eight test files.

    @pytest.mark.parametrize("name", registry.available())

For each arm:
- len(scores) == X.shape[1] and the keys equal X.columns exactly
- no NaN, no inf in scores
- determinism: two runs with the same seed produce identical scores
- column-order invariance: shuffling X's columns produces the same per-feature
  scores (catches every place you indexed positionally by accident)
- it never mutates X or y (compare a copy before/after)
- elapsed_s > 0 and params is JSON-serializable

Then, per arm, only the SPECIFIC behaviour: chi2 refuses negatives, correlation
fails on XOR, lasso zeroes redundant features, rfe respects a cost guard.
"""
