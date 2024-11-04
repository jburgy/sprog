"""Test margin-minimizing (leverage maximizing) portfolio allocation."""

# ruff: noqa: RUF018, S101, SLF001

from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, sparse

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


def _margin(
    portfolio: pd.DataFrame,
    broker: str,
    base_rate: float,
    skew_threshold: float,
    skew_penalty_rate: float,
    sector_threshold: float,
    sector_penalty_rate: float,
) -> list[LinearVariableArray]:
    sides = portfolio.groupby("Side")[broker].pipe(agg.sum)
    gmv = sides["Long"] - sides["Short"]
    sector_nmv = abs(portfolio.groupby("Sector")[broker].pipe(agg.sum))

    skew_excess = LinearVariableArray(1)
    sector_excess = LinearVariableArray(len(sector_nmv))

    return [
        gmv * base_rate
        + np.full((1, 1), skew_penalty_rate) @ skew_excess
        + np.full((1, len(sector_nmv)), sector_penalty_rate) @ sector_excess,
        sides["Long"] - skew_threshold * sides["Short"] - skew_excess,
        sector_nmv - sector_threshold * gmv - sector_excess,
    ]


def test_allocation(portfolio: pd.DataFrame) -> None:
    """Allocate stocks between brokers."""
    m = len(portfolio)
    portfolio["broker_1"] = LinearVariableArray(m)
    portfolio["broker_2"] = LinearVariableArray(m)

    margin_terms = [
        _margin(
            portfolio,
            "broker_1",
            base_rate=0.06,
            skew_threshold=0.3,
            skew_penalty_rate=0.5,
            sector_threshold=0.1,
            sector_penalty_rate=0.15,
        ),
        _margin(
            portfolio,
            "broker_2",
            base_rate=0.05,
            skew_threshold=0.4,
            skew_penalty_rate=0.7,
            sector_threshold=0.2,
            sector_penalty_rate=0.25,
        ),
    ]
    solution: optimize.OptimizeResult = optimize.linprog(
        c=np.ones(2).T
        @ LinearVariableArray._concat_same_type([item[0] for item in margin_terms]),
        A_ub=(
            a_ub := LinearVariableArray._concat_same_type(
                [
                    getattr(elem, "array", elem)
                    for item in margin_terms
                    for elem in item[1:]
                ]
            )
        ),
        b_ub=np.zeros(len(a_ub)),
        A_eq=sparse.hstack(
            [portfolio["broker_1"] + portfolio["broker_2"], sparse.csr_array((m, 20))],
            format="csr",
        ),
        b_eq=abs(portfolio["MV"]),
    )
    assert not solution.success


if __name__ == "__main__":
    pytest.main(["-v", f"{__file__}::{test_allocation.__name__}"])
