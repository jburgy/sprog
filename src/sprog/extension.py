"""See https://pandas.pydata.org/docs/development/extending.html#extension-types."""  # noqa: E501

import inspect
import os
from collections.abc import Sequence
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas._typing import (
    ArrayLike,
    Dtype,
    PositionalIndexer,
    TakeIndexer,
)
from pandas.core.arrays.base import ExtensionArray
from pandas.core.dtypes.dtypes import (
    ExtensionDtype,
    register_extension_dtype,
)
from pandas.core.indexers import is_scalar_indexer
from pandas.core.ops import unpack_zerodim_and_defer
from scipy import sparse

os.environ["MKL_RT"] = os.path.join(  # noqa: PTH118
    os.environ["VIRTUAL_ENV"],
    "lib/libmkl_rt.so.2",
)
os.environ["KMP_AFFINITY"] = "disabled"

from sparse_dot_mkl import dot_product_mkl  # noqa: E402

from sprog.sparse import gather, repeat, scatter  # noqa: E402


@register_extension_dtype
class LinearVariable(np.float64, ExtensionDtype):
    """See extension types on "Extending pandas".

    https://pandas.pydata.org/docs/development/extending.html#extension-types.
    """

    type = float
    name = "unknown"
    _supports_2d = False

    @classmethod
    def construct_array_type(cls) -> "type[ExtensionArray]":
        """Return the array type associated with this dtype."""
        return LinearVariableArray


class LinearVariableArray(sparse.csr_array, ExtensionArray):
    """An instance of ExtensionDtype to represent unknowns."""

    lower: ArrayLike | None = None
    upper: ArrayLike | None = None

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
        values: npt.NDArray[np.intp],
        original: Self,
    ) -> Self:
        return original.take(values)

    def __len__(self) -> int:
        """Avoid "sparse array length is ambiguous; use getnnz()".

        CSR matrix represents a linear operator.  We care about the
        length of its image.
        """
        return (
            self.lower.size
            if self.lower is not None
            else self.upper.size
            if self.upper is not None
            else self.shape[0]
        )

    @property
    def ndim(self) -> int:
        """Trick pandas into thinking we're one dimensional."""
        if "pandas" in inspect.currentframe().f_back.f_code.co_filename:
            return 1
        return super().ndim

    def __rmatmul__(self, lhs: sparse.sparray) -> Self:
        """Matrix multiplication using binary `@` operator."""
        return type(self)(dot_product_mkl(lhs, self))

    def __getitem__(self, key: PositionalIndexer) -> Self:
        """Object indexing using the `[]` operator."""
        return self.take([key] if is_scalar_indexer(key, ndim=1) else key)

    @unpack_zerodim_and_defer("__add__")
    def __add__(self, other: ArrayLike) -> Self:
        """Implement self - other."""
        # handle np.ndarray
        m, n = self.shape
        if n < (n1 := other.shape[1]):
            pad = sparse.csr_array((m, n1 - n))
            return type(self)(sparse.hstack([self, pad], format="csr") + other)
        return super().__add__(other)

    @unpack_zerodim_and_defer("__sub__")
    def __sub__(self, other: ArrayLike) -> Self:
        """Implement self + other."""
        m, n = self.shape
        if sparse.issparse(other) and 1 == len(other) < m:
            other = repeat(m) @ other
        if (n1 := other.shape[1]) < n:
            pad = sparse.csr_array((m, n - n1))
            other = sparse.hstack([other, pad], format="csr")
        res = super().__sub__(other)
        res.lower = self.lower
        res.upper = self.upper
        return res

    @unpack_zerodim_and_defer("__le__")
    def __le__(self, other: ArrayLike) -> Self:
        """Generate <= constraint."""
        constraint: Self = self._with_data(self.data)
        constraint.upper = np.broadcast_to(other, shape=(len(self),))
        return constraint

    @unpack_zerodim_and_defer("__eq__")
    def __eq__(self, other: ArrayLike) -> Self:
        """Generate == constraint."""
        constraint: Self = self._with_data(self.data)
        constraint.lower = other
        constraint.upper = other
        return constraint

    @property
    def dtype(self) -> ExtensionDtype:
        """Return an instance of ExtensionDtype."""
        if "sparse_dot_mkl" in inspect.currentframe().f_back.f_code.co_filename:
            return np.float64
        return LinearVariable()

    def astype(self, dtype: Dtype, *, copy: bool = False) -> ArrayLike:
        """Avoid unnecessary copy."""
        if isinstance(dtype, LinearVariable) and not copy:
            return self
        return super().astype(dtype=dtype, copy=copy)

    def take(
        self,
        indices: "TakeIndexer",
        *,
        allow_fill: bool = False,  # noqa: ARG002
        fill_value: Any = None,  # noqa: ANN401
    ) -> Self:
        """Take elements from an array."""
        assert pd.isna(fill_value)  # noqa: S101
        n = len(self)
        return gather(np.arange(n)[indices], n=n) @ self

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[Self]) -> Self:
        """Concatenate multiple array of this dtype."""
        return cls(sparse.vstack(to_concat, format="csr"))

    def isna(self) -> npt.NDArray[np.bool_]:
        """Implement pd.isna."""
        return np.zeros(len(self), dtype=np.bool_)

    def __abs__(self) -> Self:
        """Epigraph for |x|.

        y = abs(x)
            ⇒ y ≥ x ∧ y ≥ -x
            ⇒ y - x ≥ 0 ∧ y + x ≥
        """
        eye = sparse.eye(m=len(self), format="csr")
        res = type(self)(
            sparse.block_array(
                [[-self, eye], [self, eye]],
                format="csr",
            )
        )
        res.lower = np.zeros(len(self))
        return res
