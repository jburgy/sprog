"""See https://pandas.pydata.org/docs/development/extending.html#extension-types."""  # noqa: E501

import os
from collections.abc import Sequence
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas._typing import (
    Dtype,
    InterpolateOptions,
    PositionalIndexer,
    TakeIndexer,
)
from pandas.core.arrays.base import ExtensionArray
from pandas.core.dtypes.dtypes import (
    ExtensionDtype,
    register_extension_dtype,
)
from pandas.core.indexers import is_scalar_indexer
from scipy import sparse

os.environ["MKL_RT"] = os.path.join(  # noqa: PTH118
    os.environ["VIRTUAL_ENV"],
    "lib/libmkl_rt.so.2",
)
os.environ["KMP_AFFINITY"] = "disabled"

from sparse_dot_mkl import dot_product_mkl  # noqa: E402

from sprog.sparse import gather, scatter  # noqa: E402


@register_extension_dtype
class LinearVariable(ExtensionDtype):
    """See extension types on "Extending pandas".

    https://pandas.pydata.org/docs/development/extending.html#extension-types.
    """

    type = sparse.sparray
    name = "unknown"

    @classmethod
    def construct_array_type(cls) -> "type[ExtensionArray]":
        """Return the array type associated with this dtype."""
        return LinearVariableArray


class LinearVariableArray(sparse.csr_array, ExtensionArray):
    """An instance of ExtensionDtype to represent unknowns."""

    @classmethod
    def _from_sequence_of_strings(
        cls,
        strings: Sequence[str],
        dtype: Dtype | None = None,  # noqa: ARG003
        copy: bool = False,  # noqa: ARG003, FBT001, FBT002
    ) -> Self:
        return cls(scatter(pd.Series(strings).str.extract(r"^x(\d+)$")))

    @classmethod
    def _from_factorized(
        cls,
        values: npt.NDArray[np.int_],
        original: Self,
    ) -> Self:
        return original.take(values)

    def __rmatmul__(self, lhs: sparse.sparray) -> Self:
        """Matrix multiplication using binary `@` operator."""
        return type(self)(dot_product_mkl(lhs, self))

    def __getitem__(self, key: PositionalIndexer) -> Self:
        """Object indexing using the `[]` operator."""
        return self.take([key] if is_scalar_indexer(key) else key)

    def dtype(self) -> ExtensionDtype:
        """Return an instance of ExtensionDtype."""
        return LinearVariable

    def take(
        self,
        indices: "TakeIndexer",
        allow_fill: bool = False,  # noqa: ARG002, FBT001, FBT002
        fill_value: Any = None,  # noqa: ANN401
    ) -> Self:
        """Take elements from an array."""
        assert np.isnan(fill_value)  # noqa: S101
        n = len(self)
        return gather(np.arange(n)[indices], n=n) @ self

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self:
        """Concatenate multiple array of this dtype."""
        return cls(sparse.vstack(to_concat, format="csr"))

    def interpolate(
        self,
        *,
        method: InterpolateOptions,  # noqa: ARG002
        axis: int,  # noqa: ARG002
        index: pd.Index,  # noqa: ARG002
        limit,  # noqa: ANN001, ARG002
        limit_direction,  # noqa: ANN001, ARG002
        limit_area,  # noqa: ANN001, ARG002
        copy: bool,  # noqa: ARG002
        **kwargs,  # noqa: ANN003, ARG002
    ) -> Self:
        """See DataFrame.interpolate.__doc__."""
        return self
