"""See https://docs.mosek.com/latest/pythonfusion/examples-list.html#doc-example-file-baker-py."""  # noqa: E501

# ruff: noqa: S101

import numpy as np
import pandas as pd
from scipy import optimize, sparse

from sprog.extension import LinearVariableArray


def test_baker() -> None:
    """Demonstrates a small linear problem.

    Source: "Lineær Algebra" by Knut Sydsaeter and Bernt Oeksendal.
    """
    ingredients = pd.DataFrame(
        {
            "cake": [3.0, 1.0, 1.2],
            "bread": [5.0, 0.5, 0.5],
            "available": [150.0, 22.0, 25.0],
        },
        index=["flour", "sugar", "butter"],
    )
    products = pd.DataFrame(
        {
            "cost": [4.0, 6.0],
            "amount": LinearVariableArray(sparse.eye(m=2, format="csr")),
        },
        index=["cake", "bread"],
    )
    solution = optimize.linprog(
        c=-products["cost"],
        A_ub=ingredients[products.index],
        b_ub=ingredients["available"],
    )
    assert solution.success
    assert np.allclose(solution.x, [10, 24])
