"""Parse and solve fixed MPS problems."""

# ruff: noqa: S101

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import partial
from itertools import count

import numpy as np
from scipy import optimize, sparse

_OFFSETS = 14, 39


@dataclass(slots=True)
class CscBuilder:
    """Build CSC sparse array incrementally.

    :class:`scipy.sparse.csc_array` is made up of three :class:`numpy.array`
    which are not optimized for appending.  Simple python :class:`list` perform
    better.
    """

    indptr: list[str] = field(default_factory=list)
    indices: list[int] = field(default_factory=list)
    data: list[float] = field(default_factory=list)

    def insert(self, row: int, col: int, value: float) -> None:
        """Insert `value` at `(row, col)`."""
        while col >= len(self.indptr):
            self.indptr.append(len(self.indices))
        self.indices.append(row)
        self.data.append(value)


def _array_from_dict(n: int, values: dict[int, float]) -> np.ndarray:
    a = np.zeros(n)
    a.put(list(values.keys()), list(values.values()))
    return a


def solve_mps(lines: Iterable[str]) -> optimize.OptimizeResult:
    """Solve linear program in Mathematical Programming System."""
    flip = set()
    setter = {}
    n_ub = count()
    n_eq = count()
    columns = {}

    c = {}
    a_ub = CscBuilder()
    a_eq = CscBuilder()
    b_ub = {}
    b_eq = {}
    rhs = {}
    bounds = {"LO": {}, "UP": {}}
    for line in lines:
        if not line[:1].isspace():
            section = line[:14].strip()
            continue
        if section == "ROWS":
            kind = line[1]
            row = line[4:].rstrip()
            if kind == "G":
                flip.add(row)
            setter[row] = (
                c.__setitem__
                if kind == "N"
                else (
                    partial(a_ub.insert, (index := next(n_ub)))
                    if kind in "GL"
                    else (
                        partial(a_eq.insert, (index := next(n_eq)))
                        if kind == "E"
                        else None
                    )
                )
            )
            rhs[row] = (
                partial(b_ub.__setitem__, index)
                if kind in "GL"
                else partial(b_eq.__setitem__, index) if kind == "E" else None
            )
        elif section in "COLUMNS":
            col = columns.setdefault(line[4:14].rstrip(), len(columns))
            for offset in _OFFSETS:
                row = line[offset : offset + 7].rstrip()  # noqa: E203
                if not row:
                    continue
                value = float(line[offset + 10 : offset + 25])  # noqa: E203
                setter[row](col, -value if row in flip else value)
        elif section == "RHS":
            for offset in _OFFSETS:
                row = line[offset : offset + 7].rstrip()  # noqa: E203
                if not row:
                    continue
                value = float(line[offset + 10 : offset + 25])  # noqa: E203
                rhs[row](-value if row in flip else value)
        elif section == "BOUNDS":
            col = columns[line[14:21].rstrip()]
            bounds[line[1:3]][col] = float(line[24:39])
    assert section == "ENDATA"

    n_ub = next(n_ub)
    n_eq = next(n_eq)
    a_ub.indptr.append(len(a_ub.indices))
    a_eq.indptr.append(len(a_eq.indices))

    tmp = np.zeros((len(columns), 2))
    tmp[:, 1] = np.nan
    for key, val in bounds.items():
        tmp[list(val), ["LO", "UP"].index(key)] = list(val.values())

    return optimize.linprog(
        c=_array_from_dict(len(columns), c),
        A_ub=sparse.csc_array(
            (a_ub.data, a_ub.indices, a_ub.indptr), shape=(n_ub, len(columns))
        ),
        b_ub=_array_from_dict(n_ub, b_ub),
        A_eq=sparse.csc_array(
            (a_eq.data, a_eq.indices, a_eq.indptr), shape=(n_eq, len(columns))
        ),
        b_eq=_array_from_dict(n_eq, b_eq),
        bounds=tmp,
    )
