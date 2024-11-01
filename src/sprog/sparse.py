"""Utilities to implement pandas API with Yale sparse matrices."""

# ruff: noqa: S101

from collections.abc import Sequence
from numbers import Integral

import numpy as np
from scipy import sparse


def repeat(repeats: int) -> sparse.csr_array:
    """(1 × n) → (repeats × n).

    Args:
        repeats: number of repetitions

    Returns:
        sparse array in CSR format

    >>> repeat(3) @ [4]
    array([4., 4., 4.])

    """
    return sparse.csr_array(
        (np.ones(repeats), (range(repeats), [0] * repeats)), shape=(repeats, 1)
    )


def scatter(indices: Sequence[Integral], m: int = -1, n: int = -1) -> sparse.csr_array:
    """Scatter consecutive indices of x into (larger) result vector y.

    Args:
        indices: subset of range to populate (rest will be 0)
        m: length of range (defaults to :code:`max(indices) + 1`)
        n: length of domain (defaults to :code:`len(indices)`)

    Returns:
        sparse array in CSR format

    Roughly equivalent to::

        for i, j in enumerate(indices):
            y[j] = x[i]

    >>> scatter([1, 3]) @ [6, 7]
    array([0., 6., 0., 7.])

    """
    if m < 0:
        m = max(indices) + 1
    if n < 0:
        n = len(indices)
    assert m >= max(indices) + 1
    assert n >= len(indices)
    assert m >= n
    return sparse.csr_array(
        (np.ones(shape=len(indices)), (indices, range(len(indices)))),
        shape=(m, n),
    )


def gather(indices: Sequence[Integral], n: int = -1) -> sparse.csr_array:
    """Gather subset of x into (smaller) consecutive result vector y.

    Args:
        indices: subset of domain to select
        n: length of domain (defaults to :code:`max(indices) + 1`)

    Returns:
        sparse array in CSR format

    Roughly equivalent to::

        for i, j in enumerate(indices):
            y[i] = y[j]

    >>> gather([1, 3]) @ [4, 5, 6, 7]
    array([5., 7.])

    """
    m = len(indices)
    if n < 0:
        n = max(indices) + 1
    assert n >= max(indices) + 1
    assert m <= n
    return sparse.csr_array(
        (np.ones(shape=len(indices)), indices, range(len(indices) + 1)),
        shape=(m, n),
    )


def sumby(by: Sequence[Integral]) -> sparse.csr_array:
    """Compute partial sums defined by unique tuples.

    Roughly equivalent to::

        sums = defaultdict(float)
        for i, key in enumerate(by):
            sums[key] += x[i]

    >>> sumby([(0, 0), (0, 1), (1, 0), (1, 0), (1, 1), (1, 1)]) @ range(6)
    array([0., 1., 5., 9.])
    """
    keys, inverse = np.unique(by, axis=0, return_inverse=True)
    return sparse.csr_array(
        (np.ones(shape=len(by)), (inverse, range(len(by)))),
        shape=(len(keys), len(by)),
    )
