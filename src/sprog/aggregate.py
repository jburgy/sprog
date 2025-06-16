"""Pandas-aware aggregation functions."""

from typing import TYPE_CHECKING, cast

import pandas as pd
from pandas.api.typing import SeriesGroupBy

from sprog.sparse import scatter

if TYPE_CHECKING:
    from collections.abc import Sequence
    from numbers import Integral


def sum(groups: SeriesGroupBy) -> pd.Series:  # noqa: A001
    """Faster sum aggregates for LinearVariableArray series."""
    row = cast("Sequence[Integral]", groups.ngroup().to_numpy())
    return groups._wrap_applied_output(  # noqa: SLF001
        data=groups.obj,
        values=scatter(row, grouping=True) @ groups.obj.array,  # type: ignore[operator]
        not_indexed_same=True,
        is_transform=False,
    )
