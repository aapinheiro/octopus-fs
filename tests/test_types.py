"""Tests for the shared contract. Write these BEFORE any arm.

Checklist:
- ranks: 1 is the best; ties share a rank; lower_is_better is respected
- ranks on an empty/one-feature result does not explode
- normalized: all-equal scores does not divide by zero
- top(k) with k > n_features returns everything rather than raising
- ArmResult is frozen: mutating it raises
- to_json -> read back -> equal (round-trip is the real test of a contract)
"""
