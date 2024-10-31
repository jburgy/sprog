"""Test margin-minimizing (leverage maximizing) portfolio allocation."""

# ruff: noqa: RUF018, S101

from importlib.resources import files

import pandas as pd
import pytest

from sprog import aggregate as agg
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


def test_allocation(portfolio: pd.DataFrame) -> None:
    """Allocate stocks between brokers."""
    m = len(portfolio)
    portfolio["broker_1"] = LinearVariableArray(m)
    portfolio["broker_2"] = LinearVariableArray(m)
    sides = portfolio.groupby("Side")["broker_1"].pipe(agg.sum)
    gmv = sides["Long"] - sides["Short"]
    assert isinstance(gmv, LinearVariableArray)
    assert len(gmv) == 1
    sector_nmv = abs(portfolio.groupby("Sector")["broker_1"].pipe(agg.sum))
    assert sector_nmv.index.equals(pd.Index(portfolio["Sector"]).unique().sort_values())
    constraints = [
        portfolio["broker_1"] + portfolio["broker_2"] == abs(portfolio["MV"]),
        sector_nmv - 0.1 * gmv <= 0.0,
    ]
    assert all(
        isinstance(constraint, pd.Series)
        and isinstance(array := constraint.array, LinearVariableArray)
        and (array.lower is not None or array.upper is not None)
        for constraint in constraints
    )


if __name__ == "__main__":
    pytest.main(["-v", f"{__file__}::{test_allocation.__name__}"])
