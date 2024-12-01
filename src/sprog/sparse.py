"""Utilities to implement pandas API with Yale sparse matrices."""

# ruff: noqa: S101

from collections.abc import Iterable, Sequence
from itertools import pairwise, starmap
from numbers import Integral
from operator import __eq__, __sub__

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


def _increments(s: Sequence[Integral]) -> Iterable[Integral]:
    return starmap(__sub__, pairwise(s))


def isrelation(s: sparse.csr_array) -> bool:
    """Check that all entries are 0 or 1."""
    return s.nnz <= min(s.shape) and all(map(np.float64(1.0).__eq__, s.data))


def isonto(s: sparse.csr_array) -> bool:
    """Each element of the range has at least one pre-image."""
    return max(_increments(s.indptr)) < 0


def isgather(s: sparse.csr_array) -> bool:
    """Test for surjective relation.

    >>> isgather(gather([1, 3]))
    np.True_
    >>> isgather(scatter([1, 3]))
    False
    >>> isgather(scatter(range(3)))  # also onto
    np.True_
    """
    m, n = s.shape
    return m <= n and isrelation(s) and isonto(s)


def isscatter(s: sparse.csr_array) -> bool:
    """Test for injective relation.

    >>> isscatter(scatter([1, 3]))
    True
    >>> isscatter(gather([1, 3]))
    False
    >>> isscatter(gather(range(3)))  # also sequential
    True
    """
    m, n = s.shape
    return m >= n and isrelation(s) and all(map(__eq__, s.indices, range(n)))


def isvariable(s: sparse.csr_array) -> bool:
    """Test for sequential injective relation.

    >>> isvariable(gather(range(3)))
    True
    >>> isvariable(gather(range(1, 3)))
    True
    >>> isvariable(scatter(range(3)))
    True
    >>> isvariable(scatter(range(1, 3)))  # not onto
    False
    >>> isvariable(gather([1, 3]))  # not sequential
    False
    >>> isvariable(scatter([1, 3]))  # neither sequential nor onto
    False
    """
    return isgather(s) and set(_increments(s.indices)) == {-1}
