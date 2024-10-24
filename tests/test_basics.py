"""Test "Hello, World!" type stuff."""

# ruff: noqa: S101

import pandas as pd
from pandas._libs import lib
from pandas.core.construction import extract_array
from scipy import sparse

from sprog.extension import LinearVariable, LinearVariableArray


def test_construct() -> None:
    """Test LinearVariableArray.__init__."""
    obj = LinearVariableArray(sparse.eye(3, format="csr"))
    assert lib.is_list_like(obj)
    assert extract_array(obj, extract_numpy=True, extract_range=True) is obj

    s = pd.Series(obj, index=[*"ABC"])
    assert isinstance(s.dtype, LinearVariable)

    df = pd.DataFrame({"data": obj}, index=[*"ABC"])
    assert isinstance(df.dtypes["data"], LinearVariable)
    assert df.shape == (3, 1)


if __name__ == "__main__":
    test_construct()
