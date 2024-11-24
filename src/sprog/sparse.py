"""Utilities to implement pandas API with Yale sparse matrices."""

# ruff: noqa: S101

from collections.abc import Sequence
from numbers import Integral

import numpy as np
from scipy import sparse


def scatter(indices: Sequence[Integral], m: int = -1, n: int = -1) -> sparse.csr_array:
    """Scatter consecutive indices of x into (larger) result vector y.

    Args:
        indices: subset of range to populate (rest will be 0)
        m: length of range (defaults to :code:`max(indices) + 1`)
        n: length of domain (defaults to :code:`len(indices)`)

    Returns:
        sparse array in CSR format

    Roughly equivalent to::

        y[indices] = x

    >>> scatter([1, 3]) @ [6, 7]
    array([0., 6., 0., 7.])

    """
    if m < 0:
        m = max(indices) + 1
    k = len(indices)
    if n < 0:
        n = k
    assert m >= max(indices) + 1
    assert n >= k
    assert m >= n
    return sparse.csr_array((np.ones(shape=k), (indices, range(k))), shape=(m, n))


def gather(indices: Sequence[Integral], m: int = -1, n: int = -1) -> sparse.csr_array:
    """Gather subset of x into (smaller) consecutive result vector y.

    Args:
        indices: subset of domain to select
        m: length of range (defaults to :code:`len(indices)`)
        n: length of domain (defaults to :code:`max(indices) + 1`)

    Returns:
        sparse array in CSR format

    Roughly equivalent to::

        y = x[indices]

    >>> gather([1, 3]) @ [4, 5, 6, 7]
    array([5., 7.])

    """
    k = len(indices)
    if m < 0:
        m = k
    if n < 0:
        n = max(indices) + 1
    assert n >= max(indices) + 1
    return sparse.csr_array((np.ones(shape=k), (range(k), indices)), shape=(m, n))
