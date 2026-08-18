"""OctopusSelector must behave like a real sklearn transformer.

- sklearn.utils.estimator_checks.check_estimator(OctopusSelector()) — it will
  fail on things you did not expect, and every failure is a real bug
- clone(sel) preserves params; get_params/set_params round-trip
- fit on train, transform on test with columns in a DIFFERENT ORDER -> correct
  columns selected (test it by name, never by position)
- transform with a missing column raises a message naming that column
- works inside Pipeline + cross_val_score without leaking: assert the selector
  is refit per fold (count fit calls with a spy)
"""
