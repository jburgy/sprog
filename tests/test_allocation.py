"""Test margin-minimizing (leverage maximizing) portfolio allocation."""

# flake8: noqa: E203
# ruff: noqa: RUF018, S101, SLF001

from importlib.resources import files

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, sparse

from sprog import aggregate as agg
from sprog import function as func
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


@pytest.fixture
def broker_parameters() -> pd.DataFrame:
    """Broker-specific margin rates and penalty thresholds."""
    return pd.DataFrame.from_dict(
        {
            "broker_1": {
                "base_rate": 0.05,
                "skew_threshold": 0.4,
                "skew_penalty_rate": 0.3,
                "sector_threshold": 0.1,
                "sector_penalty_rate": 0.15,
            },
            "broker_2": {
                "base_rate": 0.055,
                "skew_threshold": 0.3,
                "skew_penalty_rate": 0.2,
                "sector_threshold": 0.12,
                "sector_penalty_rate": 0.17,
            },
        }
    )


def _margin(
    portfolio: pd.DataFrame,
    broker: str,
    base_rate: float,
    skew_threshold: float,
    skew_penalty_rate: float,
    sector_threshold: float,
    sector_penalty_rate: float,
) -> LinearVariableArray:
    sides = portfolio.groupby("Side")[broker].pipe(agg.sum)
    gmv = sides["Long"] - sides["Short"]
    sector_mv = portfolio.groupby("Sector")[broker].pipe(agg.sum)

    long_excess = func.pos(sides["Long"] + (1 + skew_threshold) * sides["Short"])
    short_excess = func.pos(sides["Long"] * (skew_threshold - 1) - sides["Short"])
    sector_nmv = abs(sector_mv)
    sector_excess = func.pos(sector_nmv - sector_threshold * gmv)

    return (
        gmv * base_rate
        + np.full((1, 1), skew_penalty_rate) @ long_excess
        + np.full((1, 1), skew_penalty_rate) @ short_excess
        + np.full((1, len(sector_excess)), sector_penalty_rate) @ sector_excess.array
    )


def _check_margin(
    portfolio: pd.DataFrame,
    broker: str,
    base_rate: float,
    skew_threshold: float,
    skew_penalty_rate: float,
    sector_threshold: float,
    sector_penalty_rate: float,
) -> dict[str, float]:
    sides = portfolio.groupby("Side")[broker].sum()
    gmv = sides["Long"] - sides["Short"]
    sector_mv = portfolio.groupby("Sector")[broker].sum()

    return {
        "base": gmv * base_rate,
        "long": max(sides["Long"] + (1 + skew_threshold) * sides["Short"], 0)
        * skew_penalty_rate,
        "short": max(sides["Long"] * (skew_threshold - 1) - sides["Short"], 0)
        * skew_penalty_rate,
        "sector": (abs(sector_mv) - sector_threshold * gmv).clip(lower=0).sum()
        * sector_penalty_rate,
    }


def test_allocation(portfolio: pd.DataFrame, broker_parameters: pd.DataFrame) -> None:
    """Allocate stocks between brokers."""
    m = len(portfolio)
    sign = sparse.diags_array([np.sign(portfolio["MV"])], offsets=[0], format="csr")
    portfolio["broker_1"] = sign @ LinearVariableArray(m)
    portfolio["broker_2"] = sign @ LinearVariableArray(m)

    c = sum(
        _margin(portfolio, broker, **parameters)
        for broker, parameters in broker_parameters.items()
    )
    slacks = LinearVariableArray._concat_same_type(LinearVariableArray.slacks)
    solution = optimize.linprog(
        c=c,
        A_ub=sparse.csr_array(slacks),
        b_ub=np.zeros(len(slacks)),
        A_eq=sparse.csr_array(
            (portfolio["broker_1"] + portfolio["broker_2"]).array, shape=(m, c.shape[1])
        ),
        b_eq=portfolio["MV"],
    )
    assert solution.success

    assert isinstance(c, sparse.csr_array)
    fun: np.ndarray = c @ np.diag(solution.x)
    assert np.isclose(fun.sum(), solution.fun)

    portfolio["broker_1"] = portfolio["broker_1"].array @ solution.x
    portfolio["broker_2"] = portfolio["broker_2"].array @ solution.x
    margin_checks = pd.DataFrame.from_dict(
        {
            broker: _check_margin(portfolio, broker, **parameters)
            for broker, parameters in broker_parameters.items()
        }
    )
    n = m + m
    assert np.isclose(fun[0, :m].sum(), margin_checks.loc["base", "broker_1"])
    assert np.isclose(fun[0, m:n].sum(), margin_checks.loc["base", "broker_2"])
    assert np.isclose(fun[0, n], margin_checks.loc["long", "broker_1"])
    assert np.isclose(fun[0, n + 1], margin_checks.loc["short", "broker_1"])
    assert np.allclose(fun[0, n + 1 : n + 10], 0.0)
    assert np.isclose(
        fun[0, n + 10 : n + 19].sum(), margin_checks.loc["sector", "broker_1"]
    )
    assert np.isclose(fun[0, n + 20], margin_checks.loc["long", "broker_2"])
    assert np.isclose(fun[0, n + 21], margin_checks.loc["short", "broker_2"])
    assert np.allclose(fun[0, n + 21 : n + 30], 0.0)
    assert np.isclose(fun[0, n + 30 :].sum(), margin_checks.loc["sector", "broker_2"])


if __name__ == "__main__":
    pytest.main(["-v", f"{__file__}::{test_allocation.__name__}"])
