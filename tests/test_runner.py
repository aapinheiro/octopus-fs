"""Orchestration tests. Use fakes, not real arms — these must run in <1s.

Cases:
- a registered arm that raises -> status='failed', the other arms still run
- on_error='raise' propagates instead
- an arm that doesn't support the task -> status='skipped' with a reason
- SAME SEED, SAME RESULT: run fit() twice, assert identical scores. This is the
  test that catches every unseeded random_state you forgot to pass down.
- n_jobs=2 produces byte-identical output to n_jobs=1
"""
