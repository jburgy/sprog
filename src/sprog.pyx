# cython: language_level=3
# https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html
import cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def matmul(
    m: cython.Py_ssize_t,
    n: cython.Py_ssize_t,
    ap: cython.int[::1],
    aj: cython.int[::1],
    ax: cython.double[::1],
    bp: cython.int[::1],
    bj: cython.int[::1],
    bx: cython.double[::1],
) -> tuple[cython.int[::1], list[cython.int], list[cython.double]]:
    indptr: cython.int[::1] = np.copy(ap)
    indices: list[cython.int] = []
    data: list[cython.double] = []

    colbeg: cython.Py_ssize_t
    row: cython.Py_ssize_t
    colend: cython.Py_ssize_t
    rowbeg: cython.Py_ssize_t
    col: cython.Py_ssize_t
    rowend: cython.Py_ssize_t
    x: cython.double
    i: cython.Py_ssize_t
    k: cython.Py_ssize_t
    hi: cython.Py_ssize_t
    mid: cython.Py_ssize_t

    colbeg = indptr[0]
    for row in range(m):
        colend = indptr[row + 1]
        rowbeg = bp[0]
        for col in range(n):
            rowend = bp[col + 1]
            x = 0.0
            for i in range(colbeg, colend):
                k = aj[i]
                hi = rowend  # bisect_left
                while rowbeg < hi:
                    mid = (rowbeg + hi) // 2
                    if k < bj[mid]:
                        hi = mid
                    else:
                        rowbeg = mid + 1
                if rowbeg == rowend:
                    break
                if k < bj[rowbeg]:
                    continue
                x += ax[i] * bx[rowbeg]
                rowbeg += 1
            else:
                rowbeg = rowend
            if np.isclose(x, 0.0):
                continue
            indices.append(col)
            data.append(x)
        indptr[row + 1] = len(indices)
        colbeg = colend
    return indptr, indices, data
