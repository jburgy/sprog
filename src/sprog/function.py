"""Common functions in Linear Programming."""

# ruff: noqa: S101

import pandas as pd

from sprog.extension import LinearVariable, LinearVariableArray


def pos(s: pd.Series | LinearVariableArray) -> pd.Series | LinearVariableArray:
    """Positive part of :code:`s`.

    Args:
        s: input series (:code:`dtype` must be :class:`sprog.extension.LinearVariable`)

    Returns:
        new Series (and side-effect on slacks)

    """
    assert isinstance(s, LinearVariableArray) or isinstance(s.dtype, LinearVariable)

    excess = LinearVariableArray(len(s))
    excess.slacks.append((s.array if isinstance(s, pd.Series) else s) - excess)
    return (
        pd.Series(
            excess,
            index=s.index,
            name=f"({s.name})₊" if s.name else None,
        )
        if isinstance(s, pd.Series)
        else excess
    )
