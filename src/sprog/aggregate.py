"""Pandas-aware aggregation functions."""

import pandas as pd
from pandas.api.typing import SeriesGroupBy

from sprog.sparse import scatter


def sum(groups: SeriesGroupBy) -> pd.Series:  # noqa: A001
    """Faster sum aggregates for LinearVariableArray series."""
    row = groups.ngroup().to_numpy()
    return groups._wrap_applied_output(  # noqa: SLF001
        data=groups.obj,
        values=scatter(row, grouping=True) @ groups.obj.array,
        not_indexed_same=True,
        is_transform=False,
    )
