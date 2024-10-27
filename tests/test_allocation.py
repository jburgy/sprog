"""Test margin-minimizing (leverage maximizing) portfolio allocation."""

# ruff: noqa: S101

from importlib.resources import files

import pandas as pd
import pytest
from scipy import sparse

from sprog.extension import LinearVariableArray
from tests import resources


@pytest.fixture
def portfolio() -> pd.DataFrame:
    """Use LSEQ as sample portfolio of US equities."""
    columns = {"TICKER": "Symbol", "VALUE (USD)": "MV"}
    rc = files(resources)
    holdings = pd.read_csv(
        rc.joinpath("Harbor_Long-Short_Equity_ETF_holdings_20241022.csv"),
        usecols=columns,
    ).rename(columns=columns)
    screener = pd.read_csv(
        rc.joinpath("nasdaq_screener_1729696921350.csv"),
        index_col="Symbol",
        usecols=["Symbol", "Sector"],
    )
    holdings["Side"] = "Long"
    holdings.loc[holdings["MV"] < 0, "Side"] = "Short"
    holdings["Sector"] = holdings["Symbol"].map(screener["Sector"])
    return holdings


@pytest.mark.skip(reason="work in progress")
def test_allocation(portfolio: pd.DataFrame) -> None:
    """Allocate stocks between brokers."""
    assert portfolio.shape == (160, 4), "placeholder"
    portfolio["broker_1"] = LinearVariableArray(
        sparse.eye(len(portfolio), format="csr")
    )
    portfolio["broker_2"] = LinearVariableArray(
        sparse.eye(len(portfolio), k=len(portfolio), format="csr")
    )
    constraints = [
        portfolio["broker_1"] + portfolio["broker_2"] == abs(portfolio["MV"])
    ]
    assert isinstance(constraints, list)
