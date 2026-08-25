# =========================================================
# 🛡️ R.S MASTER STOCK GUIDE V2
# FUNDAMENTAL + GROWTH + QUALITY ENGINE
# UPGRADED FINAL V2.1
# =========================================================

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import yfinance as yf


# =========================================================
# CONFIG
# =========================================================

MIN_DATA_QUALITY = 60.0

GOOD_SCORE = 70.0
AVERAGE_SCORE = 50.0

MIN_SIGNAL_SCORE = 60.0


# =========================================================
# SAFE HELPERS
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

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:

        return default


def _clamp(
    value,
    minimum=0.0,
    maximum=100.0
):

    value = _safe_float(
        value,
        minimum
    )

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


def _clean_symbol(
    symbol: str
) -> str:

    symbol = str(
        symbol
    ).strip().upper()

    if symbol.endswith(".NS"):

        return symbol

    return f"{symbol}.NS"


# =========================================================
# SERIES HELPERS
# =========================================================

def _normalize_financial_df(
    df: pd.DataFrame
) -> pd.DataFrame:

    if df is None or df.empty:

        return pd.DataFrame()

    data = df.copy()

    # -----------------------------------------------------
    # Remove duplicated rows
    # -----------------------------------------------------

    data = data[
        ~data.index.duplicated(
            keep="first"
        )
    ]

    # -----------------------------------------------------
    # Convert columns to datetime where possible
    # -----------------------------------------------------

    valid_columns = []

    for col in data.columns:

        try:

            converted = pd.to_datetime(
                col,
                errors="coerce"
            )

            if not pd.isna(converted):

                valid_columns.append(
                    (col, converted)
                )

        except Exception:

            pass

    # -----------------------------------------------------
    # Sort financial periods
    # Latest period first
    # -----------------------------------------------------

    if valid_columns:

        valid_columns.sort(
            key=lambda x: x[1],
            reverse=True
        )

        data = data[
            [
                item[0]
                for item in valid_columns
            ]
        ]

    return data


def _find_row(
    df: pd.DataFrame,
    names: list[str]
):

    if (
        df is None
        or df.empty
    ):

        return None

    normalized = {}

    for index in df.index:

        key = str(
            index
        ).strip().lower()

        normalized[key] = index

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
):

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


def _latest_value(
    series
):

    if series is None:

        return np.nan

    try:

        series = pd.to_numeric(
            series,
            errors="coerce"
        ).dropna()

    except Exception:

        return np.nan

    if series.empty:

        return np.nan

    return _safe_float(
        series.iloc[0]
    )


def _previous_value(
    series
):

    if series is None:

        return np.nan

    try:

        series = pd.to_numeric(
            series,
            errors="coerce"
        ).dropna()

    except Exception:

        return np.nan

    if len(series) < 2:

        return np.nan

    return _safe_float(
        series.iloc[1]
    )


def _value_n_years_ago(
    series,
    years=3
):

    if series is None:

        return np.nan

    try:

        series = pd.to_numeric(
            series,
            errors="coerce"
        ).dropna()

    except Exception:

        return np.nan

    if len(series) <= years:

        return np.nan

    return _safe_float(
        series.iloc[years]
    )


# =========================================================
# GROWTH
# =========================================================

def _growth_percent(
    current,
    previous
):

    current = _safe_float(
        current
    )

    previous = _safe_float(
        previous
    )

    if (
        not np.isfinite(current)
        or not np.isfinite(previous)
        or previous == 0
    ):

        return np.nan

    return (
        (
            current
            -
            previous
        )
        /
        abs(previous)
    ) * 100


def _cagr(
    current,
    historical,
    years
):

    current = _safe_float(
        current
    )

    historical = _safe_float(
        historical
    )

    if (
        not np.isfinite(current)
        or not np.isfinite(historical)
        or current <= 0
        or historical <= 0
        or years <= 0
    ):

        return np.nan

    return (
        (
            current
            /
            historical
        )
        **
        (1 / years)
        - 1
    ) * 100


# =========================================================
# DOWNLOAD
# =========================================================

def _download_ticker_info(
    ticker
):

    try:

        info = ticker.info

        if isinstance(
            info,
            dict
        ):

            return info

    except Exception:

        pass

    return {}


def _download_financials(
    ticker
):

    income = pd.DataFrame()

    balance = pd.DataFrame()

    cashflow = pd.DataFrame()

    try:

        income = (
            _normalize_financial_df(
                ticker.financials
            )
        )

    except Exception:

        pass

    try:

        balance = (
            _normalize_financial_df(
                ticker.balance_sheet
            )
        )

    except Exception:

        pass

    try:

        cashflow = (
            _normalize_financial_df(
                ticker.cashflow
            )
        )

    except Exception:

        pass

    return (
        income,
        balance,
        cashflow
    )


# =========================================================
# PROFITABILITY
# =========================================================

def _calculate_roe(
    info,
    income,
    balance
):

    value = _safe_float(
        info.get(
            "returnOnEquity"
        )
    )

    if np.isfinite(value):

        return value * 100

    net_income = _latest_value(
        _series_from_rows(
            income,
            [
                "Net Income",
                "NetIncome"
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
        np.isfinite(net_income)
        and np.isfinite(equity)
        and equity != 0
    ):

        return (
            net_income
            /
            equity
        ) * 100

    return np.nan


def _calculate_roa(
    info,
    income,
    balance
):

    value = _safe_float(
        info.get(
            "returnOnAssets"
        )
    )

    if np.isfinite(value):

        return value * 100

    net_income = _latest_value(
        _series_from_rows(
            income,
            [
                "Net Income",
                "NetIncome"
            ]
        )
    )

    assets = _latest_value(
        _series_from_rows(
            balance,
            [
                "Total Assets",
                "TotalAssets"
            ]
        )
    )

    if (
        np.isfinite(net_income)
        and np.isfinite(assets)
        and assets != 0
    ):

        return (
            net_income
            /
            assets
        ) * 100

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

            return value * 100

        return value

    operating_income = _latest_value(
        _series_from_rows(
            income,
            [
                "Operating Income",
                "OperatingIncome",
                "EBIT"
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

    capital_employed = (
        equity
        +
        debt
    )

    if (
        np.isfinite(operating_income)
        and np.isfinite(capital_employed)
        and capital_employed != 0
    ):

        return (
            operating_income
            /
            capital_employed
        ) * 100

    return np.nan


# =========================================================
# GROWTH ENGINE
# =========================================================

def _calculate_growth_metrics(
    info,
    income
):

    revenue_series = _series_from_rows(
        income,
        [
            "Total Revenue",
            "TotalRevenue",
