"""Test "Hello, World!" type stuff."""

# ruff: noqa: S101

import pandas as pd
from pandas._libs import lib
from pandas.core.construction import extract_array
from scipy import sparse

from sprog.extension import LinearVariable, LinearVariableArray


def test_construct() -> None:
    """Test LinearVariableArray.__init__."""
    obj = LinearVariableArray(sparse.eye_array(3, format="csr"))
    assert lib.is_list_like(obj)
    assert extract_array(obj, extract_numpy=True, extract_range=True) is obj

    s = pd.Series(obj, index=[*"ABC"])
    assert isinstance(s.dtype, LinearVariable)

    frame = pd.DataFrame({"data": obj}, index=[*"ABC"])
    assert isinstance(frame.dtypes["data"], LinearVariable)
    assert frame.shape == (3, 1)


def test_take() -> None:
    """Test indexing and selecting data."""
    obj = LinearVariableArray(sparse.eye_array(3, format="csr"))
    ser = pd.Series(obj, index=[*"ABC"])
    loc = ser.loc[["A", "C"]]
    assert isinstance(loc, pd.Series)


def test_melt() -> None:
    """Invoke LinearVariableArray._concat_same_type."""
    wide = pd.DataFrame(
        {
            "A": LinearVariableArray(sparse.eye_array(m=3)),
            "B": LinearVariableArray(sparse.eye_array(m=3, n=6, k=3)),
        },
        index=[*"abc"],
    )
    narrow = wide.melt(ignore_index=False)
    assert (sparse.eye_array(6) != narrow["value"].array).nnz == 0


if __name__ == "__main__":
    test_melt()
