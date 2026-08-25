"""
R.S MASTER STOCK GUIDE V2
FUNDAMENTAL + GROWTH + QUALITY ENGINE

Responsibilities:
- Fundamental data
- Revenue growth
- Profit growth
- EPS growth
- ROE / ROCE / ROA
- Debt / Equity
- Profit margin
- Quality checks
- Valuation snapshot
- Fundamental score
- Fundamental zone
- Data quality

This module is intentionally independent from the UI.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import yfinance as yf


# =========================================================
# CONFIGURATION
# =========================================================

MIN_DATA_QUALITY = 50.0

GOOD_SCORE = 70
AVERAGE_SCORE = 45


# =========================================================
# HELPERS
# =========================================================

def _safe_float(
    value: Any,
    default: float = np.nan
) -> float:

    try:

        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def _clean_symbol(
    symbol: str
) -> str:

    symbol = str(
        symbol
    ).strip().upper()

    if symbol.endswith(".NS"):
        return symbol

    return f"{symbol}.NS"


def _latest_value(
    series: pd.Series | None
) -> float:

    if series is None:
        return np.nan

    if not isinstance(
        series,
        pd.Series
    ):
        return np.nan

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if series.empty:
        return np.nan

    return float(
        series.iloc[-1]
    )


def _previous_value(
    series: pd.Series | None
) -> float:

    if series is None:
        return np.nan

    if not isinstance(
        series,
        pd.Series
    ):
        return np.nan

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(series) < 2:
        return np.nan

    return float(
        series.iloc[-2]
    )


def _growth_percent(
    current: float,
    previous: float
) -> float:

    if pd.isna(current):
        return np.nan

    if pd.isna(previous):
        return np.nan

    if previous == 0:
        return np.nan

    return (
        (current - previous)
        / abs(previous)
    ) * 100


def _find_row(
    df: pd.DataFrame,
    names: list[str]
) -> pd.Series | None:

    if df is None or df.empty:
        return None

    normalized = {
        str(index).strip().lower(): index
        for index in df.index
    }

    for name in names:

        key = str(
            name
        ).strip().lower()

        if key in normalized:

            return df.loc[
                normalized[key]
            ]

    return None


def _series_from_rows(
    df: pd.DataFrame,
    names: list[str]
) -> pd.Series | None:

    row = _find_row(
        df,
        names
    )

    if row is None:
        return None

    if isinstance(
        row,
        pd.DataFrame
    ):

        row = row.iloc[0]

    return pd.to_numeric(
        row,
        errors="coerce"
    )


# =========================================================
# FINANCIAL DATA DOWNLOAD
# =========================================================

def _download_ticker_info(
    ticker: yf.Ticker
) -> Dict[str, Any]:

    try:

        info = ticker.info

        if not isinstance(
            info,
            dict
        ):
            return {}

        return info

    except Exception:

        return {}


def _download_financials(
    ticker: yf.Ticker
):

    annual_income = pd.DataFrame()
    annual_balance = pd.DataFrame()
    annual_cashflow = pd.DataFrame()

    try:
        annual_income = (
            ticker.financials
        )

    except Exception:
        pass

    try:
        annual_balance = (
            ticker.balance_sheet
        )

    except Exception:
        pass

    try:
        annual_cashflow = (
            ticker.cashflow
        )

    except Exception:
        pass

    return (
        annual_income,
        annual_balance,
        annual_cashflow
    )


# =========================================================
# PROFITABILITY
# =========================================================

def _calculate_roe(
    info: Dict[str, Any],
    income: pd.DataFrame,
    balance: pd.DataFrame
) -> float:

    roe = _safe_float(
        info.get("returnOnEquity")
    )

    if not pd.isna(roe):
        return roe * 100

    net_income_series = _series_from_rows(
        income,
        [
            "Net Income",
            "NetIncome"
        ]
    )

    equity_series = _series_from_rows(
        balance,
        [
            "Stockholders Equity",
            "Total Stockholder Equity",
            "Common Stock Equity"
        ]
    )

    net_income = _latest_value(
        net_income_series
    )

    equity = _latest_value(
        equity_series
    )

    if (
        not pd.isna(net_income)
        and not pd.isna(equity)
        and equity != 0
    ):

        return (
            net_income
            / equity
        ) * 100

    return np.nan


def _calculate_roa(
    info: Dict[str, Any],
    income: pd.DataFrame,
    balance: pd.DataFrame
) -> float:

    roa = _safe_float(
        info.get("returnOnAssets")
    )

    if not pd.isna(roa):
        return roa * 100

    net_income_series = _series_from_rows(
        income,
        [
            "Net Income",
            "NetIncome"
        ]
    )

    assets_series = _series_from_rows(
        balance,
        [
            "Total Assets",
            "TotalAssets"
        ]
    )

    net_income = _latest_value(
        net_income_series
    )

    assets = _latest_value(
        assets_series
    )

    if (
        not pd.isna(net_income)
        and not pd.isna(assets)
        and assets != 0
    ):

        return (
            net_income
            / assets
        ) * 100

    return np.nan


def _calculate_roce(
    info: Dict[str, Any],
    income: pd.DataFrame,
    balance: pd.DataFrame
) -> float:

    # Prefer provider value when available.
    roce = _safe_float(
        info.get("returnOnCapitalEmployed")
    )

    if not pd.isna(roce):

        if abs(roce) < 1:
            return roce * 100

        return roce

    operating_income_series = _series_from_rows(
        income,
        [
            "Operating Income",
            "OperatingIncome"
        ]
    )

    equity_series = _series_from_rows(
        balance,
        [
            "Stockholders Equity",
            "Total Stockholder Equity",
            "Common Stock Equity"
        ]
    )

    debt_series = _series_from_rows(
        balance,
        [
            "Total Debt",
            "Long Term Debt",
            "Long Term Debt And Capital Lease Obligation"
        ]
    )

    operating_income = _latest_value(
        operating_income_series
    )

    equity = _latest_value(
        equity_series
    )

    debt = _latest_value(
        debt_series
    )

    if pd.isna(debt):
        debt = 0.0

    capital_employed = (
        equity + debt
    )

    if (
        not pd.isna(operating_income)
        and capital_employed != 0
    ):

        return (
            operating_income
            / capital_employed
        ) * 100

    return np.nan


# =========================================================
# GROWTH
# =========================================================

def _calculate_growth_metrics(
    info: Dict[str, Any],
    income: pd.DataFrame
):

    revenue_growth = np.nan
    profit_growth = np.nan
    eps_growth = np.nan

    # Provider values
    provider_revenue_growth = _safe_float(
        info.get(
            "revenueGrowth"
        )
    )

    provider_earnings_growth = _safe_float(
        info.get(
            "earningsGrowth"
        )
    )

    if not pd.isna(
        provider_revenue_growth
    ):

        revenue_growth = (
            provider_revenue_growth * 100
        )

    if not pd.isna(
        provider_earnings_growth
    ):

        profit_growth = (
            provider_earnings_growth * 100
        )

    # Revenue
    revenue_series = _series_from_rows(
        income,
        [
            "Total Revenue",
            "TotalRevenue",
            "Operating Revenue",
            "OperatingRevenue"
        ]
    )

    revenue_current = _latest_value(
        revenue_series
    )

    revenue_previous = _previous_value(
        revenue_series
    )

    calculated_revenue_growth = (
        _growth_percent(
            revenue_current,
            revenue_previous
        )
    )

    if not pd.isna(
        calculated_revenue_growth
    ):

        revenue_growth = (
            calculated_revenue_growth
        )

    # Profit
    profit_series = _series_from_rows(
        income,
        [
            "Net Income",
            "NetIncome"
        ]
    )

    profit_current = _latest_value(
        profit_series
    )

    profit_previous = _previous_value(
        profit_series
    )

    calculated_profit_growth = (
        _growth_percent(
            profit_current,
            profit_previous
        )
    )

    if not pd.isna(
        calculated_profit_growth
    ):

        profit_growth = (
            calculated_profit_growth
        )

    # EPS
    eps_series = _series_from_rows(
        income,
        [
            "Diluted EPS",
            "Basic EPS",
            "DilutedEPS",
            "BasicEPS"
        ]
    )

    eps_current = _latest_value(
        eps_series
    )

    eps_previous = _previous_value(
        eps_series
    )

    calculated_eps_growth = (
        _growth_percent(
            eps_current,
            eps_previous
        )
    )

    if not pd.isna(
        calculated_eps_growth
    ):

        eps_growth = (
            calculated_eps_growth
        )

    return (
        revenue_growth,
        profit_growth,
        eps_growth
    )


# =========================================================
# MARGINS
# =========================================================

def _calculate_profit_margin(
    info: Dict[str, Any],
    income: pd.DataFrame
) -> float:

    margin = _safe_float(
        info.get("profitMargins")
    )

    if not pd.isna(margin):

        return margin * 100

    revenue_series = _series_from_rows(
        income,
        [
            "Total Revenue",
            "TotalRevenue"
        ]
    )

    net_income_series = _series_from_rows(
        income,
        [
            "Net Income",
            "NetIncome"
        ]
    )

    revenue = _latest_value(
        revenue_series
    )

    net_income = _latest_value(
        net_income_series
    )

    if (
        not pd.isna(revenue)
        and revenue != 0
        and not pd.isna(net_income)
    ):

        return (
            net_income
            / revenue
        ) * 100

    return np.nan


# =========================================================
# DEBT
# =========================================================

def _calculate_debt_equity(
    info: Dict[str, Any],
    balance: pd.DataFrame
) -> float:

    debt_equity = _safe_float(
        info.get(
            "debtToEquity"
        )
    )

    if not pd.isna(
        debt_equity
    ):

        # Yahoo generally reports this as a percentage.
        return debt_equity / 100

    debt_series = _series_from_rows(
        balance,
        [
            "Total Debt",
            "Long Term Debt",
            "Long Term Debt And Capital Lease Obligation"
        ]
    )

    equity_series = _series_from_rows(
        balance,
        [
            "Stockholders Equity",
            "Total Stockholder Equity",
            "Common Stock Equity"
        ]
    )

    debt = _latest_value(
        debt_series
    )

    equity = _latest_value(
        equity_series
    )

    if (
        not pd.isna(debt)
        and not pd.isna(equity)
        and equity != 0
    ):

        return (
            debt
            / equity
        )

    return np.nan


# =========================================================
# QUALITY SCORE
# =========================================================

def _score_positive_metric(
    score: float,
    value: float,
    strong: float,
    positive: float
) -> float:

    if pd.isna(value):
        return score

    if value >= strong:
        return score + 10

    if value >= positive:
        return score + 5

    if value < 0:
        return score - 8

    return score


def calculate_fundamental_score(
    metrics: Dict[str, Any]
) -> float:

    score = 50.0

    # Revenue growth
    score = _score_positive_metric(
        score,
        metrics.get(
            "REVENUE_GROWTH_%"
        ),
        15,
        5
    )

    # Profit growth
    score = _score_positive_metric(
        score,
        metrics.get(
            "PROFIT_GROWTH_%"
        ),
        15,
        5
    )

    # EPS growth
    score = _score_positive_metric(
        score,
        metrics.get(
            "EPS_GROWTH_%"
        ),
        15,
        5
    )

    # ROE
    score = _score_positive_metric(
        score,
        metrics.get(
            "ROE_%"
        ),
        15,
        10
    )

    # ROCE
    score = _score_positive_metric(
        score,
        metrics.get(
            "ROCE_%"
        ),
        15,
        10
    )

    # ROA
    score = _score_positive_metric(
        score,
        metrics.get(
            "ROA_%"
        ),
        8,
        5
    )

    # Profit margin
    score = _score_positive_metric(
        score,
        metrics.get(
            "PROFIT_MARGIN_%"
        ),
        15,
        8
    )

    # Debt
    debt_equity = metrics.get(
        "DEBT_TO_EQUITY"
    )

    if not pd.isna(
        _safe_float(
            debt_equity
        )
    ):

        debt_equity = float(
            debt_equity
        )

        if debt_equity <= 0.5:
            score += 10

        elif debt_equity <= 1.0:
            score += 5

        elif debt_equity >= 2.0:
            score -= 10

    return float(
        max(
            0,
            min(
                100,
                round(
                    score,
                    1
                )
            )
        )
    )


# =========================================================
# FUNDAMENTAL ZONE
# =========================================================

def get_fundamental_zone(
    score: float
) -> str:

    if score >= GOOD_SCORE:
        return "GOOD"

    if score >= AVERAGE_SCORE:
        return "AVERAGE"

    return "POOR"


# =========================================================
# DATA QUALITY
# =========================================================

def calculate_fundamental_data_quality(
    metrics: Dict[str, Any]
) -> float:

    important_fields = [
        "REVENUE_GROWTH_%",
        "PROFIT_GROWTH_%",
        "EPS_GROWTH_%",
        "ROE_%",
        "ROCE_%",
        "ROA_%",
        "DEBT_TO_EQUITY",
        "PROFIT_MARGIN_%"
    ]

    available = 0

    for field in important_fields:

        value = metrics.get(
            field
        )

        try:

            if value is not None and not pd.isna(value):
                available += 1

        except Exception:
            pass

    return round(
        (
            available
            /
            len(important_fields)
        ) * 100,
        2
    )


# =========================================================
# VALUATION SNAPSHOT
# =========================================================

def calculate_valuation(
    info: Dict[str, Any]
) -> Dict[str, Any]:

    pe = _safe_float(
        info.get("trailingPE")
    )

    forward_pe = _safe_float(
        info.get("forwardPE")
    )

    pb = _safe_float(
        info.get("priceToBook")
    )

    peg = _safe_float(
        info.get("pegRatio")
    )

    dividend_yield = _safe_float(
        info.get("dividendYield")
    )

    market_cap = _safe_float(
        info.get("marketCap")
    )

    if not pd.isna(
        dividend_yield
    ):

        dividend_yield *= 100

    # Simple informational valuation zone.
    # It is NOT a standalone buy/sell signal.

    valuation_zone = "DATA_LIMITED"

    reference_pe = (
        forward_pe
        if not pd.isna(forward_pe)
        else pe
    )

    if not pd.isna(
        reference_pe
    ):

        if reference_pe <= 15:
            valuation_zone = "ATTRACTIVE"

        elif reference_pe <= 25:
            valuation_zone = "FAIR"

        elif reference_pe <= 40:
            valuation_zone = "EXPENSIVE"

        else:
            valuation_zone = "VERY_EXPENSIVE"

    return {

        "PE": pe,

        "FORWARD_PE": forward_pe,

        "PB": pb,

        "PEG": peg,

        "DIVIDEND_YIELD_%": (
            dividend_yield
        ),

        "MARKET_CAP": market_cap,

        "VALUATION_ZONE": (
            valuation_zone
        )
    }


# =========================================================
# MAIN FUNDAMENTAL ENGINE
# =========================================================

def fetch_fundamental_data(
    symbol: str
) -> Dict[str, Any]:

    clean_symbol = str(
        symbol
    ).strip().upper()

    ticker_symbol = _clean_symbol(
        clean_symbol
    )

    try:

        ticker = yf.Ticker(
            ticker_symbol
        )

        info = _download_ticker_info(
            ticker
        )

        (
            income,
            balance,
            cashflow
        ) = _download_financials(
            ticker
        )

        # -------------------------------------------------
        # GROWTH
        # -------------------------------------------------

        (
            revenue_growth,
            profit_growth,
            eps_growth
        ) = _calculate_growth_metrics(
            info,
            income
        )

        # -------------------------------------------------
        # PROFITABILITY
        # -------------------------------------------------

        roe = _calculate_roe(
            info,
            income,
            balance
        )

        roce = _calculate_roce(
            info,
            income,
            balance
        )

        roa = _calculate_roa(
            info,
            income,
            balance
        )

        profit_margin = (
            _calculate_profit_margin(
                info,
                income
            )
        )

        # -------------------------------------------------
        # DEBT
        # -------------------------------------------------

        debt_equity = (
            _calculate_debt_equity(
                info,
                balance
            )
        )

        # -------------------------------------------------
        # BASIC METRICS
        # -------------------------------------------------

        metrics = {

            "REVENUE_GROWTH_%": (
                revenue_growth
            ),

            "PROFIT_GROWTH_%": (
                profit_growth
            ),

            "EPS_GROWTH_%": (
                eps_growth
            ),

            "ROE_%": roe,

            "ROCE_%": roce,

            "ROA_%": roa,

            "DEBT_TO_EQUITY": (
                debt_equity
            ),

            "PROFIT_MARGIN_%": (
                profit_margin
            )
        }

        # -------------------------------------------------
        # SCORE
        # -------------------------------------------------

        fundamental_score = (
            calculate_fundamental_score(
                metrics
            )
        )

        fundamental_zone = (
            get_fundamental_zone(
                fundamental_score
            )
        )

        # -------------------------------------------------
        # DATA QUALITY
        # -------------------------------------------------

        data_quality = (
            calculate_fundamental_data_quality(
                metrics
            )
        )

        # -------------------------------------------------
        # VALUATION
        # -------------------------------------------------

        valuation = calculate_valuation(
            info
        )

        # -------------------------------------------------
        # COMPANY INFO
        # -------------------------------------------------

        company_name = info.get(
            "longName"
        )

        sector = info.get(
            "sector"
        )

        industry = info.get(
            "industry"
        )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        return {

            "SYMBOL": clean_symbol,

            "TICKER": ticker_symbol,

            "COMPANY_NAME": (
                company_name
                if company_name
                else clean_symbol
            ),

            "SECTOR": (
                sector
                if sector
                else "—"
            ),

            "INDUSTRY": (
                industry
                if industry
                else "—"
            ),

            # SCORE
            "FUNDAMENTAL_SCORE": (
                fundamental_score
            ),

            "FUNDAMENTAL_ZONE": (
                fundamental_zone
            ),

            # GROWTH
            "REVENUE_GROWTH_%": (
                revenue_growth
            ),

            "PROFIT_GROWTH_%": (
                profit_growth
            ),

            "EPS_GROWTH_%": (
                eps_growth
            ),

            # QUALITY
            "ROE_%": roe,

            "ROCE_%": roce,

            "ROA_%": roa,

            "PROFIT_MARGIN_%": (
                profit_margin
            ),

            "DEBT_TO_EQUITY": (
                debt_equity
            ),

            # VALUATION
            "PE": valuation[
                "PE"
            ],

            "FORWARD_PE": valuation[
                "FORWARD_PE"
            ],

            "PB": valuation[
                "PB"
            ],

            "PEG": valuation[
                "PEG"
            ],

            "DIVIDEND_YIELD_%": (
                valuation[
                    "DIVIDEND_YIELD_%"
                ]
            ),

            "MARKET_CAP": (
                valuation[
                    "MARKET_CAP"
                ]
            ),

            "VALUATION_ZONE": (
                valuation[
                    "VALUATION_ZONE"
                ]
            ),

            # QUALITY
            "DATA_QUALITY_%": (
                data_quality
            ),

            "STATUS": (
                "FRESH"
                if data_quality >= MIN_DATA_QUALITY
                else "LIMITED"
            ),

            "SIGNAL_ALLOWED": (
                data_quality >= MIN_DATA_QUALITY
            )
        }

    except Exception as error:

        return {

            "SYMBOL": clean_symbol,

            "TICKER": ticker_symbol,

            "COMPANY_NAME": clean_symbol,

            "SECTOR": "—",

            "INDUSTRY": "—",

            "FUNDAMENTAL_SCORE": 0.0,

            "FUNDAMENTAL_ZONE": "POOR",

            "REVENUE_GROWTH_%": np.nan,

            "PROFIT_GROWTH_%": np.nan,

            "EPS_GROWTH_%": np.nan,

            "ROE_%": np.nan,

            "ROCE_%": np.nan,

            "ROA_%": np.nan,

            "PROFIT_MARGIN_%": np.nan,

            "DEBT_TO_EQUITY": np.nan,

            "PE": np.nan,

            "FORWARD_PE": np.nan,

            "PB": np.nan,

            "PEG": np.nan,

            "DIVIDEND_YIELD_%": np.nan,

            "MARKET_CAP": np.nan,

            "VALUATION_ZONE": "DATA_LIMITED",

            "DATA_QUALITY_%": 0.0,

            "STATUS": "ERROR",

            "SIGNAL_ALLOWED": False,

            "ERROR": str(error)
        }


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    result = fetch_fundamental_data(
        "CEMPRO"
    )

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )
