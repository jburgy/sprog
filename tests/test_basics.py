"""Test "Hello, World!" type stuff."""

# ruff: noqa: S101

import pandas as pd
import pytest
from pandas._libs import lib
from pandas.core.construction import extract_array, sanitize_array
from scipy import sparse

from sprog.extension import LinearVariable, LinearVariableArray


def test_construct() -> None:
    """Test LinearVariableArray.__init__."""
    obj = LinearVariableArray(sparse.eye_array(3, format="csr"))
    assert lib.is_list_like(obj)
    assert extract_array(obj, extract_numpy=True, extract_range=True) is obj
    assert sanitize_array(obj, index=[*"ABC"]) is obj

    a = pd.array([3, 4], dtype=LinearVariable())
    assert all(a.indptr == [0] * 4 + [1, 2])
    assert (a != a.astype(float)).nnz == 0

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


def test_unstack() -> None:
    """Invoke LinearVariable.construct_array_type."""
    a = pd.Series(  # noqa: PD010
        LinearVariableArray(sparse.eye_array(4)),
        index=pd.MultiIndex.from_product([["one", "two"], ["a", "b"]]),
    ).unstack(level=0)
    b = pd.DataFrame(
        {
            "one": LinearVariableArray(sparse.eye_array(m=2, n=4)),
            "two": LinearVariableArray(sparse.eye_array(m=2, n=4, k=2)),
        },
        index=["a", "b"],
    )
    pd.testing.assert_frame_equal(a, b)


def test_resize() -> None:
    """Test .resize method."""
    a = LinearVariableArray(sparse.eye_array(3))

    with pytest.raises(ValueError, match=r"shape\[0\]=2 must not be less than m=3"):
        a.resize(2, 3)

    with pytest.raises(ValueError, match=r"shape\[1\]=2 must not be less than n=3"):
        a.resize(3, 2)

    assert a.resize(4, 4).shape == (4, 4)


if __name__ == "__main__":
    test_construct()
