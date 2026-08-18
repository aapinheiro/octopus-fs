"""Shared fixtures.

Design the fixtures FIRST. A dataset where you know the right answer turns
every arm test from 'it ran without crashing' into 'it found what it should'.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def informative_regression():
    """(X, y, truth) where `truth` is the set of genuinely informative columns.

    TODO(you): sklearn.datasets.make_regression(n_features=20, n_informative=5,
    shuffle=False, random_state=0) puts the informative columns FIRST — that is
    your ground truth. Wrap in a DataFrame with names f00..f19.
    """
    raise NotImplementedError


@pytest.fixture
def informative_classification():
    """Same idea with make_classification.

    TODO(you): set n_redundant>0 too — redundant columns are linear combinations
    of informative ones, and they are exactly what distinguishes the arms
    (lasso zeroes them, tree importance splits credit, correlation loves them).
    """
    raise NotImplementedError


@pytest.fixture
def nonlinear_xor():
    """y = XOR(x0 > 0, x1 > 0), plus noise columns.

    The discriminating fixture: correlation MUST fail here (near-zero scores for
    x0/x1) and mutual_info MUST succeed. Assert both. A test suite that only
    uses linear data will happily pass while your MI arm is broken.
    """
    raise NotImplementedError


@pytest.fixture
def dirty_frame():
    """Constant column, NaN column, negative column, duplicated name, object dtype.

    Every validation and every supports() rule gets tested against this one.
    """
    raise NotImplementedError
