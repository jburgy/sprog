"""See :external+pandas:ref:`extending.extension-types`."""

# flake8: noqa: E203
# ruff: noqa: SLF001

import numbers
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from packaging.version import parse
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
from scipy.sparse._sputils import check_shape

os.environ["MKL_RT"] = str(  # noqa: I001
    max(
        Path(sys.prefix, sys.platlibdir).glob("libmkl_rt.so.*"),
        key=lambda path: parse(path.name.rpartition(".")[-1]),
    )
)
os.environ["KMP_AFFINITY"] = "disabled"

from sparse_dot_mkl import dot_product_mkl  # noqa: E402

from sprog.sparse import gather, scatter  # noqa: E402

_offset: int = 0


@register_extension_dtype
class LinearVariable(np.float64, ExtensionDtype):
    """See :external+pandas:ref:`extending.extension-types`.

    :class:`pandas.api.extensions.ExtensionDtype` subclass
    """

    type = float
    name = "unknown"
    _supports_2d = False
    char = "d"

    @classmethod
    def construct_array_type(cls) -> "type[ExtensionArray]":
        """Return the array type associated with this dtype."""
        return LinearVariableArray


class LinearVariableArray(sparse.csr_array, ExtensionArray):
    """An instance of :class:`pandas.api.extensions.ExtensionArray` to represent unknowns.

    This class contains *shared* :term:`mutable` state (:code:`slacks` class variable)
    even though :external+python:ref:`tut-class-and-instance-variables` warns against it.
    """  # noqa: E501

    slacks: ClassVar = []

    def __init__(
        self,
        arg1: Any,  # noqa: ANN401
        *,
        shape: tuple | None = None,
        dtype: ExtensionDtype | None = None,
        copy: bool = False,
    ) -> None:
        """Instantiate helper variables."""
        if isinstance(arg1, numbers.Integral):
            global _offset  # noqa: PLW0603
            n = arg1 + _offset
            arg1 = gather(range(_offset, n), m=arg1, n=n)
            _offset = n
        super().__init__(arg1, shape=shape, dtype=dtype, copy=copy)

    @classmethod
    def _from_sequence(
        cls,
        data: Sequence[int],
        *,
        dtype: Dtype | None = None,
        copy: bool = False,  # noqa: ARG003
    ) -> Self:
        assert isinstance(dtype, LinearVariable)  # noqa: S101
        return cls(scatter(data))

    def __len__(self) -> int:
        """Avoid "sparse array length is ambiguous; use getnnz()".

        CSR matrix represents a linear operator.  We care about the
        length of its image.
        """
        return self.shape[0]

    @property
    def ndim(self) -> int:
        """Trick pandas into thinking we're one dimensional."""
        return 1

    def __matmul__(self, rhs: sparse.sparray) -> sparse.csr_array:
        """Matrix multiplication using binary `@` operator."""
        n = len(rhs)
        return dot_product_mkl(
            sparse.csr_array(self.resize(len(self), n) if self.shape[1] < n else self),
            rhs,
        )

    def __rmatmul__(self, lhs: sparse.sparray) -> Self:
        """Matrix multiplication using binary `@` operator."""
        return self.__class__(dot_product_mkl(lhs, sparse.csr_array(self)))

    def __getitem__(self, key: PositionalIndexer) -> Self:
        """Object indexing using the `[]` operator."""
        return self.take([key] if is_scalar_indexer(key, ndim=1) else key)

    def resize(self, *shape: int) -> Self:
        """Resize to dimensions given by shape."""
        shape = check_shape(shape)
        m, n = self.shape
        if shape[0] < m:
            msg = f"{shape[0]=} must not be less than {m=}"
            raise ValueError(msg)
        if shape[1] < m:
            msg = f"{shape[1]=} must not be less than {n=}"
            raise ValueError(msg)
        m = shape[0] - m
        return self.__class__(
            (
                self.data,
                self.indices,
                np.concatenate(
                    (
                        self.indptr,
                        np.full(m, self.indptr[-1], dtype=self.indptr.dtype),
                    )
                )
                if m > 0
                else self.indptr,
            ),
            shape=shape,
            dtype=self.dtype,
        )

    @unpack_zerodim_and_defer("__add__")
    def __add__(self, other: ArrayLike) -> Self:
        """Implement self - other."""
        # handle np.ndarray
        if isinstance(other, int) and other == 0:
            return self
        m, n = self.shape
        if n < (n1 := other.shape[1]):
            return self.resize(m, n1) + other
        return super().__add__(other)

    @unpack_zerodim_and_defer("__sub__")
    def __sub__(self, other: Self) -> Self:
        """Implement self + other."""
        m, n = self.shape
        m1, n1 = other.shape
        if sparse.issparse(other) and 1 == m1 < m:
            other = gather([0] * m) @ other
        if n < n1:
            return self.resize(m, n1) - other
        if n1 < n:
            other = other.resize(m, n)
        return super().__sub__(other)

    @property
    def dtype(self) -> ExtensionDtype:
        """Return an instance of ExtensionDtype."""
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
        from scipy.sparse._sputils import get_index_dtype

        data = np.concatenate([b.data for b in to_concat])
        constant_dim = max(b._shape_as_2d[1] for b in to_concat)
        idx_dtype = get_index_dtype(
            arrays=[b.indptr for b in to_concat], maxval=max(data.size, constant_dim)
        )
        indices = np.empty(data.size, dtype=idx_dtype)
        indptr = np.empty(
            sum(b._shape_as_2d[0] for b in to_concat) + 1, dtype=idx_dtype
        )
        last_indptr = idx_dtype(0)
        sum_dim = 0
        sum_indices = 0
        for b in to_concat:
            indices[sum_indices : sum_indices + b.indices.size] = b.indices
            sum_indices += b.indices.size
            idxs = slice(sum_dim, sum_dim + b._shape_as_2d[0])
            indptr[idxs] = b.indptr[:-1]
            indptr[idxs] += last_indptr
            sum_dim += b._shape_as_2d[0]
            last_indptr += b.indptr[-1]
        indptr[-1] = last_indptr
        return cls((data, indices, indptr), shape=(sum_dim, constant_dim))

    def isna(self) -> npt.NDArray[np.bool_]:
        """Implement pd.isna."""
        return np.zeros(len(self), dtype=np.bool_)

    def __abs__(self) -> Self:
        """Epigraph for :math:`|x|`.

        .. math::

            y = |x|

            ⇒ y ≥ x ∧ y ≥ -x

            ⇒ x - y ≤ 0 ∧ -x - y ≤ 0

        """
        slack = self.__class__(len(self))
        self.slacks.extend([self - slack, -self - slack])
        return slack
