import os
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from pandas import Index
from pandas._typing import (
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

os.environ["MKL_RT"] = "/opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so"

from sparse_dot_mkl import dot_product_mkl  # noqa E402

from sprog.sparse import gather  # noqa E402


@register_extension_dtype
class LinearVariable(ExtensionDtype):
    type = sparse.sparray
    name = "unknown"

    @classmethod
    def construct_array_type(cls) -> "type[ExtensionArray]":
        return LinearVariableArray


class LinearVariableArray(sparse.csr_array, ExtensionArray):
    @classmethod
    def _from_sequence_of_strings(cls, strings, dtype=None, copy=False):
        return NotImplemented

    @classmethod
    def _from_factorized(cls, values: npt.NDArray[np.int_], original: Self):
        return original.take(values)

    def __rmatmul__(self, lhs: sparse.sparray) -> Self:
        return type(self)(dot_product_mkl(lhs, self))

    def __getitem__(self, key: PositionalIndexer) -> Self:
        return self.take([key] if is_scalar_indexer(key) else key)

    def dtype(self):
        return LinearVariable

    def take(
        self,
        indices: "TakeIndexer",
        allow_fill: bool = False,
        fill_value: Any = None,
    ) -> Self:
        assert np.isnan(fill_value)
        n = len(self)
        return gather(np.arange(n)[indices], n=n) @ self

    @classmethod
    def _concat_same_type(cls, to_concat):
        return NotImplemented

    def interpolate(
        self,
        *,
        method: InterpolateOptions,
        axis: int,
        index: Index,
        limit,
        limit_direction,
        limit_area,
        copy: bool,
        **kwargs,
    ) -> Self:
        return self
