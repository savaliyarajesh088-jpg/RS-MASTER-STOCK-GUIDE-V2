"""
R.S MASTER STOCK GUIDE V2
FUNDAMENTAL ENGINE
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd
import yfinance as yf


MIN_DATA_QUALITY = 50.0
GOOD_SCORE = 70
AVERAGE_SCORE = 45


def _safe_float(
    value: Any,
    default: float = np.nan
) -> float:
    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def _clean_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()

    if symbol.endswith(".NS"):
        return symbol

    return f"{symbol}.NS"


def _latest_value(series):
    if series is None:
        return np.nan

    try:
        series = pd.to_numeric(
            series,
            errors="coerce"
        ).dropna()

        if series.empty:
            return np.nan

        return float(series.iloc[-1])

    except Exception:
        return np.nan


def _previous_value(series):
    if series is None:
        return np.nan

    try:
        series = pd.to_numeric(
            series,
            errors="coerce"
        ).dropna()

        if len(series) < 2:
            return np.nan

        return float(series.iloc[-2])

    except Exception:
        return np.nan


def _growth_percent(
    current,
    previous
):
    current = _safe_float(current)
    previous = _safe_float(previous)

    if not np.isfinite(current):
        return np.nan

    if not np.isfinite(previous):
        return np.nan

    if previous == 0:
        return np.nan

    return (
        (current - previous)
        / abs(previous)
    ) * 100


def _find_row(
    df: pd.DataFrame,
    names
):
    if df is None or df.empty:
        return None

    normalized = {
        str(index).strip().lower(): index
        for index in df.index
    }

    for name in names:

        key = str(name).strip().lower()

        if key in normalized:
            return df.loc[normalized[key]]

    return None


def _series_from_rows(
    df: pd.DataFrame,
    names
):
    row = _find_row(df, names)

    if row is None:
        return None

    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    return pd.to_numeric(
        row,
        errors="coerce"
    )


def _download_info(ticker):

    try:
        info = ticker.info

        if isinstance(info, dict):
            return info

    except Exception:
        pass

    return {}


def _download_financials(ticker):

    income = pd.DataFrame()
    balance = pd.DataFrame()
    cashflow = pd.DataFrame()

    try:
        income = ticker.financials
    except Exception:
        pass

    try:
        balance = ticker.balance_sheet
    except Exception:
        pass

    try:
        cashflow = ticker.cashflow
    except Exception:
        pass

    return income, balance, cashflow


def _calculate_roe(
    info,
    income,
    balance
):

    value = _safe_float(
        info.get("returnOnEquity")
    )

    if np.isfinite(value):
        return round(value * 100, 2)

    net_income = _latest_value(
        _series_from_rows(
            income,
            ["Net Income", "NetIncome"]
        )
    )

    equity = _latest_value(
        _series_from_rows(
            balance,
            [
                "Stockholders Equity",
                "Total Stockholder Equity",
                "Common Stock Equity"
            ]
        )
    )

    if (
        np.isfinite(net_income)
        and np.isfinite(equity)
        and equity != 0
    ):
        return round(
            (net_income / equity) * 100,
            2
        )

    return np.nan


def _calculate_roa(
    info,
    income,
    balance
):

    value = _safe_float(
        info.get("returnOnAssets")
    )

    if np.isfinite(value):
        return round(value * 100, 2)

    net_income = _latest_value(
        _series_from_rows(
            income,
            ["Net Income", "NetIncome"]
        )
    )

    assets = _latest_value(
        _series_from_rows(
            balance,
            ["Total Assets", "TotalAssets"]
        )
    )

    if (
        np.isfinite(net_income)
        and np.isfinite(assets)
        and assets != 0
    ):
        return round(
            (net_income / assets) * 100,
            2
        )

    return np.nan


def _calculate_roce(
    info,
    income,
    balance
):

    value = _safe_float(
        info.get(
            "returnOnCapitalEmployed"
        )
    )

    if np.isfinite(value):

        if abs(value) < 1:
            value *= 100

        return round(value, 2)

    operating_income = _latest_value(
        _series_from_rows(
            income,
            [
                "Operating Income",
                "OperatingIncome"
            ]
        )
    )

    equity = _latest_value(
        _series_from_rows(
            balance,
            [
                "Stockholders Equity",
                "Total Stockholder Equity",
                "Common Stock Equity"
            ]
        )
    )

    debt = _latest_value(
        _series_from_rows(
            balance,
            [
                "Total Debt",
                "Long Term Debt",
                "Long Term Debt And Capital Lease Obligation"
            ]
        )
    )

    if not np.isfinite(debt):
        debt = 0.0

    capital = equity + debt

    if (
        np.isfinite(operating_income)
        and np.isfinite(capital)
        and capital != 0
    ):
        return round(
            (operating_income / capital) * 100,
            2
        )

    return np.nan


def _calculate_growth(
    info,
    income
):

    revenue_growth = _safe_float(
        info.get("revenueGrowth")
    )

    profit_growth = _safe_float(
        info.get("earningsGrowth")
    )

    if np.isfinite(revenue_growth):
        revenue_growth *= 100

    revenue = _series_from_rows(
        income,
        [
            "Total Revenue",
            "TotalRevenue",
            "Operating Revenue",
            "OperatingRevenue"
        ]
    )

    current_revenue = _latest_value(revenue)
    previous_revenue = _previous_value(revenue)

    calculated = _growth_percent(
        current_revenue,
        previous_revenue
    )

    if np.isfinite(calculated):
        revenue_growth = calculated

    profit = _series_from_rows(
        income,
        ["Net Income", "NetIncome"]
    )

    current_profit = _latest_value(profit)
    previous_profit = _previous_value(profit)

    calculated = _growth_percent(
        current_profit,
        previous_profit
    )

    if np.isfinite(calculated):
        profit_growth = calculated

    eps = _series_from_rows(
        income,
        [
            "Diluted EPS",
            "Basic EPS",
            "DilutedEPS",
            "BasicEPS"
        ]
    )

    eps_growth = _growth_percent(
        _latest_value(eps),
        _previous_value(eps)
    )

    return (
        revenue_growth,
        profit_growth,
        eps_growth
    )


def _calculate_margin(
    info,
    income
):

    value = _safe_float(
        info.get("profitMargins")
    )

    if np.isfinite(value):
        return value * 100

    revenue = _latest_value(
        _series_from_rows(
            income,
            [
                "Total Revenue",
                "TotalRevenue"
            ]
        )
    )

    profit = _latest_value(
        _series_from_rows(
            income,
            [
                "Net Income",
                "NetIncome"
            ]
        )
    )

    if (
        np.isfinite(revenue)
        and revenue != 0
        and np.isfinite(profit)
    ):
        return (
            profit / revenue
        ) * 100

    return np.nan


def _calculate_debt_equity(
    info,
    balance
):

    value = _safe_float(
        info.get("debtToEquity")
    )

    if np.isfinite(value):
        return value / 100

    debt = _latest_value(
        _series_from_rows(
            balance,
            [
                "Total Debt",
                "Long Term Debt",
                "Long Term Debt And Capital Lease Obligation"
            ]
        )
    )

    equity = _latest_value(
        _series_from_rows(
            balance,
            [
                "Stockholders Equity",
                "Total Stockholder Equity",
                "Common Stock Equity"
            ]
        )
    )

    if (
        np.isfinite(debt)
        and np.isfinite(equity)
        and equity != 0
    ):
        return debt / equity

    return np.nan


def _metric_score(
    score,
    value,
    strong,
    positive
):

    value = _safe_float(value)

    if not np.isfinite(value):
        return score

    if value >= strong:
        return score + 10

    if value >= positive:
        return score + 5

    if value < 0:
        return score - 8

    return score


def calculate_fundamental_score(metrics):

    score = 50.0

    fields = [
        ("REVENUE_GROWTH_%", 15, 5),
        ("PROFIT_GROWTH_%", 15, 5),
        ("EPS_GROWTH_%", 15, 5),
        ("ROE_%", 15, 10),
        ("ROCE_%", 15, 10),
        ("ROA_%", 8, 5),
        ("PROFIT_MARGIN_%", 15, 8)
    ]

    for field, strong, positive in fields:

        score = _metric_score(
            score,
            metrics.get(field),
            strong,
            positive
        )

    debt = _safe_float(
        metrics.get("DEBT_TO_EQUITY")
    )

    if np.isfinite(debt):

        if debt <= 0.5:
            score += 10

        elif debt <= 1:
            score += 5

        elif debt >= 2:
            score -= 10

    return round(
        max(0, min(100, score)),
        1
    )


def get_fundamental_zone(score):

    score = _safe_float(score, 0)

    if score >= GOOD_SCORE:
        return "GOOD"

    if score >= AVERAGE_SCORE:
        return "AVERAGE"

    return "POOR"


def calculate_data_quality(metrics):

    fields = [
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

    for field in fields:

        value = metrics.get(field)

        try:
            if value is not None and not pd.isna(value):
                available += 1
        except Exception:
            pass

    return round(
        available / len(fields) * 100,
        2
    )


def calculate_valuation(info):

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

    dividend = _safe_float(
        info.get("dividendYield")
    )

    market_cap = _safe_float(
        info.get("marketCap")
    )

    if np.isfinite(dividend):
        dividend *= 100

    reference_pe = (
        forward_pe
        if np.isfinite(forward_pe)
        else pe
    )

    zone = "DATA_LIMITED"

    if np.isfinite(reference_pe):

        if reference_pe <= 15:
            zone = "ATTRACTIVE"

        elif reference_pe <= 25:
            zone = "FAIR"

        elif reference_pe <= 40:
            zone = "EXPENSIVE"

        else:
            zone = "VERY_EXPENSIVE"

    return {
        "PE": pe,
        "FORWARD_PE": forward_pe,
        "PB": pb,
        "PEG": peg,
        "DIVIDEND_YIELD_%": dividend,
        "MARKET_CAP": market_cap,
        "VALUATION_ZONE": zone
    }


def fetch_fundamental_data(symbol):

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

        info = _download_info(
            ticker
        )

        income, balance, cashflow = (
            _download_financials(ticker)
        )

        revenue_growth, profit_growth, eps_growth = (
            _calculate_growth(
                info,
                income
            )
        )

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

        margin = _calculate_margin(
            info,
            income
        )

        debt_equity = _calculate_debt_equity(
            info,
            balance
        )

        metrics = {
            "REVENUE_GROWTH_%": revenue_growth,
            "PROFIT_GROWTH_%": profit_growth,
            "EPS_GROWTH_%": eps_growth,
            "ROE_%": roe,
            "ROCE_%": roce,
            "ROA_%": roa,
            "PROFIT_MARGIN_%": margin,
            "DEBT_TO_EQUITY": debt_equity
        }

        score = calculate_fundamental_score(
            metrics
        )

        quality = calculate_data_quality(
            metrics
        )

        valuation = calculate_valuation(
            info
        )

        return {

            "SYMBOL": clean_symbol,
            "TICKER": ticker_symbol,

            "COMPANY_NAME": info.get(
                "longName",
                clean_symbol
            ),

            "SECTOR": info.get(
                "sector",
                "—"
            ),

            "INDUSTRY": info.get(
                "industry",
                "—"
            ),

            "FUNDAMENTAL_SCORE": score,

            "FUNDAMENTAL_ZONE":
                get_fundamental_zone(score),

            "REVENUE_GROWTH_%":
                revenue_growth,

            "PROFIT_GROWTH_%":
                profit_growth,

            "EPS_GROWTH_%":
                eps_growth,

            "ROE_%": roe,
            "ROCE_%": roce,
            "ROA_%": roa,

            "PROFIT_MARGIN_%":
                margin,

            "DEBT_TO_EQUITY":
                debt_equity,

            "PE": valuation["PE"],
            "FORWARD_PE": valuation["FORWARD_PE"],
            "PB": valuation["PB"],
            "PEG": valuation["PEG"],

            "DIVIDEND_YIELD_%":
                valuation["DIVIDEND_YIELD_%"],

            "MARKET_CAP":
                valuation["MARKET_CAP"],

            "VALUATION_ZONE":
                valuation["VALUATION_ZONE"],

            "DATA_QUALITY_%":
                quality,

            "STATUS":
                "FRESH"
                if quality >= MIN_DATA_QUALITY
                else "LIMITED",

            "SIGNAL_ALLOWED":
                quality >= MIN_DATA_QUALITY
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

            "VALUATION_ZONE":
                "DATA_LIMITED",

            "DATA_QUALITY_%": 0.0,
            "STATUS": "ERROR",
            "SIGNAL_ALLOWED": False,

            "ERROR": str(error)
        }


# =========================================================
# BACKWARD COMPATIBILITY
# =========================================================

fetch_fundamental = fetch_fundamental_data
run_fundamental_engine = fetch_fundamental_data


# =========================================================
# LOCAL TEST
# =========================================================

if __name__ == "__main__":

    print("R.S MASTER STOCK GUIDE V2")
    print("Fundamental Engine: OK")

    result = fetch_fundamental_data("CEMPRO")

    print("SYMBOL:", result.get("SYMBOL"))
    print("FUNDAMENTAL SCORE:",
          result.get("FUNDAMENTAL_SCORE"))
    print("STATUS:",
          result.get("STATUS"))
