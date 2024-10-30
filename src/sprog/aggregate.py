"""Pandas-aware aggregation functions."""

import numpy as np
import pandas as pd
from pandas.api.typing import SeriesGroupBy
from scipy import sparse


def sum(groups: SeriesGroupBy) -> pd.Series:  # noqa: A001
    """Faster sum aggregates for LinearVariableArray series."""
    row = groups.ngroup().to_numpy()
    return groups._wrap_applied_output(  # noqa: SLF001
        data=groups.obj,
        values=sparse.csr_array(
            (np.ones(row.size), (row, np.arange(row.size, dtype=row.dtype)))
        )
        @ groups.obj.array,
        not_indexed_same=True,
        is_transform=False,
    )
