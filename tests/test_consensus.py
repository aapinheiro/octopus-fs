"""Tests for rank aggregation — the best place for property-based testing.

Properties worth asserting with hypothesis:
- PERMUTATION INVARIANCE: shuffling the order of arms never changes the result
- UNANIMITY: if every arm ranks feature A first, A wins under every method
- IDEMPOTENCE: aggregating one arm's ranking returns that same ranking
- MONOTONICITY: improving a feature's rank in one arm never worsens its
  aggregated position
- a skipped arm changes nothing (it contributes no votes)

Plus one hand-computed Borda fixture with the arithmetic written out in a
comment — property tests prove consistency, a worked example proves correctness.
"""
