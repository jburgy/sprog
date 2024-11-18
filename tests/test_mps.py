"""Test :mod:`sprog.mps`."""

# ruff: noqa: S101

import numpy as np
import pytest

from sprog.mps import solve_mps


@pytest.fixture
def problem() -> str:
    """Sample problem from https://lpsolve.sourceforge.net/5.5/mps-format.htm."""
    return """NAME          TESTPROB
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  MYEQN
COLUMNS
    XONE      COST                 1   LIM1                 1
    XONE      LIM2                 1
    YTWO      COST                 4   LIM1                 1
    YTWO      MYEQN               -1
    ZTHREE    COST                 9   LIM2                 1
    ZTHREE    MYEQN                1
RHS
    RHS1      LIM1                 5   LIM2                10
    RHS1      MYEQN                7
BOUNDS
 UP BND1      XONE                 4
 LO BND1      YTWO                -1
 UP BND1      YTWO                 1
ENDATA
"""


def test_mps(problem: str) -> None:
    """Check that solve_mps works as expected."""
    solution = solve_mps(problem.splitlines())
    assert np.allclose(solution.x, [4, -1, 6])
