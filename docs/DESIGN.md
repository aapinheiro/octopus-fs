# Design notes

Short, opinionated, and updated as decisions get made. Open questions first —
this file is where you argue with yourself before writing code.

## Decisions taken

| # | Decision | Rationale |
|---|---|---|
| 1 | Arms return scores, not subsets | Cutting is a policy, scoring is a measurement. Separating them lets you re-cut a saved run without recomputing. |
| 2 | Consensus aggregates ranks, not normalized scores | The eight scales are not comparable; normalizing hides that instead of fixing it. |
| 3 | One shared fitted estimator for the post-hoc arms | Refitting per arm triples the cost of a run for zero benefit. |
| 4 | Selection is a sklearn transformer | Otherwise users select on the full dataset and leak. |
| 5 | Optional deps behind extras | A production pipeline should not install plotly to select features. |
| 6 | Skips are loud and recorded | A silent zero from a non-applicable arm corrupts the consensus invisibly. |

## Open questions

- **NaN policy.** Raise (v0.1) vs. arm-by-arm handling. Imputation is a
  modeling decision — does it belong in a selection library at all?
- **Categorical features.** Encode internally (and hide it) or require the user
  to pass an encoded frame? Internal encoding makes `chi2` and `correlation`
  mean different things than the user expects.
- **What is the unit of a "feature"?** After one-hot encoding, is `city` one
  feature or 40? The consensus is computed on columns, but humans think in
  source features. A `feature_groups` mapping may be the answer.
- **Multicollinearity.** No arm here detects redundancy directly. Worth adding a
  VIF / clustered-correlation pre-pass that groups near-duplicates and reports
  them alongside the ranking?
- **Scale.** pandas covers most cases. Wide tables (>10k columns) or Spark-
  resident data need a backend protocol. Defer to v0.4, but do not paint
  yourself into a corner: keep arms taking a DataFrame-like, not a
  `pandas.DataFrame` literal, in type hints.
