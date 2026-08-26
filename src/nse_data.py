"""
R.S MASTER STOCK GUIDE V3.1
NSE MARKET DATA ENGINE
============================================================

ROLE
------------------------------------------------------------
Independent NSE/Yahoo market-data + technical evidence engine.

CORE
------------------------------------------------------------
• NSE / Yahoo Finance market data
• CMP / Previous Close / Change %
• OHLCV
• 52W High / Low
• EMA 10 / 20 / 50 / 100 / 200
• RSI 14 — Wilder
• MACD 12 / 26 / 9
• ATR 14
• Supertrend 10 / 3
• Volume Average / Volume Ratio
• Volume Breakout
• EMA Alignment
• Technical Score
• Technical Zone
• Data Quality
• Data Freshness
• Signal Safety
• EMS-compatible independent evidence

ARCHITECTURE LOCK
------------------------------------------------------------
• NSE ONLY
• No fabricated market data
• Missing data remains missing
• No Master Score calculation
• No EXIT decision
• EMS remains independent
• Fundamental layer remains external
============================================================
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import math
import numpy as np
import pandas as pd
import yfinance as yf


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PERIOD = "2y"
DATA_INTERVAL = "1d"

EMA_PERIODS = (
    10,
    20,
    50,
    100,
    200,
)

RSI_PERIOD = 14

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

ATR_PERIOD = 14

SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0

VOLUME_AVG_PERIOD = 20

VOLUME_BREAKOUT_THRESHOLD = 2.0

FRESHNESS_LIMIT_DAYS = 5

MIN_DATA_QUALITY = 90.0


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

        if isinstance(value, str):
            value = value.replace(",", "").strip()

        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except Exception:
        return default


def _clean_symbol(
    symbol: str
) -> str:

    symbol = str(symbol).strip().upper()

    if symbol.endswith(".NS"):
        symbol = symbol[:-3]

    return symbol


def _yahoo_symbol(
    symbol: str
) -> str:

    return f"{_clean_symbol(symbol)}.NS"


# =========================================================
# YFINANCE NORMALISATION
# =========================================================

def _flatten_yfinance_columns(
    df: pd.DataFrame
) -> pd.DataFrame:

    if not isinstance(df.columns, pd.MultiIndex):
        return df

    output = df.copy()

    required = {
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
    }

    columns = []

    for column in output.columns:

        selected = None

        if isinstance(column, tuple):

            for item in column:

                if str(item) in required:
                    selected = str(item)
                    break

        if selected is None:
            selected = str(column[0])

        columns.append(selected)

    output.columns = columns

    # Protect against duplicate columns
    output = output.loc[
        :,
        ~output.columns.duplicated()
    ]

    return output


def _normalise_ohlcv(
    df: pd.DataFrame
) -> pd.DataFrame:

    if df is None or df.empty:
        raise ValueError(
            "Empty market data"
        )

    df = _flatten_yfinance_columns(df)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing market columns: "
            + ", ".join(missing)
        )

    df = df[required].copy()

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
        ]
    )

    df = df.sort_index()

    if df.empty:
        raise ValueError(
            "No valid OHLC data"
        )

    # Basic market-data sanity checks
    invalid = (
        (df["High"] < df["Low"])
        |
        (df["High"] < df["Open"])
        |
        (df["High"] < df["Close"])
        |
        (df["Low"] > df["Open"])
        |
        (df["Low"] > df["Close"])
        |
        (df["Close"] <= 0)
    )

    if invalid.any():

        df = df.loc[
            ~invalid
        ].copy()

    if df.empty:
        raise ValueError(
            "No valid OHLC rows after validation"
        )

    return df


# =========================================================
# RSI — WILDER
# =========================================================

def calculate_rsi(
    close: pd.Series,
    period: int = RSI_PERIOD
) -> pd.Series:

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    rsi = (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )

    rsi = rsi.mask(
        avg_loss == 0,
        100.0
    )

    rsi = rsi.mask(
        (
            avg_gain == 0
        )
        &
        (
            avg_loss == 0
        ),
        50.0
    )

    return rsi


# =========================================================
# MACD
# =========================================================

def calculate_macd(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> Tuple[
    pd.Series,
    pd.Series,
    pd.Series,
]:

    ema_fast = close.ewm(
        span=fast,
        adjust=False,
        min_periods=fast,
    ).mean()

    ema_slow = close.ewm(
        span=slow,
        adjust=False,
        min_periods=slow,
    ).mean()

    macd_line = (
        ema_fast
        -
        ema_slow
    )

    signal_line = macd_line.ewm(
        span=signal,
        adjust=False,
        min_periods=signal,
    ).mean()

    histogram = (
        macd_line
        -
        signal_line
    )

    return (
        macd_line,
        signal_line,
        histogram,
    )


# =========================================================
# ATR — WILDER
# =========================================================

def calculate_atr(
    df: pd.DataFrame,
    period: int = ATR_PERIOD
) -> pd.Series:

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (
                high - previous_close
            ).abs(),
            (
                low - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    return atr


# =========================================================
# SUPERTREND 10 / 3
# =========================================================

def calculate_supertrend(
    df: pd.DataFrame,
    period: int = SUPERTREND_PERIOD,
    multiplier: float = SUPERTREND_MULTIPLIER,
) -> Tuple[
    pd.Series,
    pd.Series,
]:

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    atr = calculate_atr(
        df,
        period
    )

    hl2 = (
        high + low
    ) / 2.0

    basic_upper = (
        hl2
        +
        multiplier * atr
    )

    basic_lower = (
        hl2
        -
        multiplier * atr
    )

    final_upper = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    final_lower = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    supertrend = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    direction = pd.Series(
        "NO_DATA",
        index=df.index,
        dtype=object,
    )

    for i in range(len(df)):

        if pd.isna(
            atr.iloc[i]
        ):
            continue

        # First valid ATR row
        if i == 0:

            final_upper.iloc[i] = (
                basic_upper.iloc[i]
            )

            final_lower.iloc[i] = (
                basic_lower.iloc[i]
            )

            continue

        previous_upper = (
            final_upper.iloc[i - 1]
        )

        previous_lower = (
            final_lower.iloc[i - 1]
        )

        previous_close = (
            close.iloc[i - 1]
        )

        # Final upper band
        if pd.isna(previous_upper):

            final_upper.iloc[i] = (
                basic_upper.iloc[i]
            )

        elif (
            basic_upper.iloc[i]
            < previous_upper
            or
            previous_close
            > previous_upper
        ):

            final_upper.iloc[i] = (
                basic_upper.iloc[i]
            )

        else:

            final_upper.iloc[i] = (
                previous_upper
            )

        # Final lower band
        if pd.isna(previous_lower):

            final_lower.iloc[i] = (
                basic_lower.iloc[i]
            )

        elif (
            basic_lower.iloc[i]
            > previous_lower
            or
            previous_close
            < previous_lower
        ):

            final_lower.iloc[i] = (
                basic_lower.iloc[i]
            )

        else:

            final_lower.iloc[i] = (
                previous_lower
            )

        previous_direction = (
            direction.iloc[i - 1]
        )

        # Initialize direction using price
        if previous_direction == "NO_DATA":

            if close.iloc[i] >= hl2.iloc[i]:
                current_direction = "BULLISH"
            else:
                current_direction = "BEARISH"

        elif previous_direction == "BULLISH":

            if close.iloc[i] < final_lower.iloc[i]:
                current_direction = "BEARISH"
            else:
                current_direction = "BULLISH"

        else:

            if close.iloc[i] > final_upper.iloc[i]:
                current_direction = "BULLISH"
            else:
                current_direction = "BEARISH"

        direction.iloc[i] = (
            current_direction
        )

        if current_direction == "BULLISH":

            supertrend.iloc[i] = (
                final_lower.iloc[i]
            )

        else:

            supertrend.iloc[i] = (
                final_upper.iloc[i]
            )

    return (
        supertrend,
        direction,
    )


# =========================================================
# EMA ALIGNMENT
# =========================================================

def get_ema_alignment(
    row: pd.Series
) -> str:

    values = []

    for period in EMA_PERIODS:

        value = _safe_float(
            row.get(
                f"EMA_{period}"
            )
        )

        values.append(value)

    if any(
        pd.isna(value)
        for value in values
    ):
        return "INSUFFICIENT_DATA"

    e10, e20, e50, e100, e200 = values

    if (
        e10 > e20
        and e20 > e50
        and e50 > e100
        and e100 > e200
    ):
        return "BULLISH"

    if (
        e10 < e20
        and e20 < e50
        and e50 < e100
        and e100 < e200
    ):
        return "BEARISH"

    return "MIXED"


# =========================================================
# RSI STATE
# =========================================================

def get_rsi_state(
    rsi: float
) -> str:

    if pd.isna(rsi):
        return "NO_DATA"

    if rsi >= 70:
        return "OVERBOUGHT"

    if rsi >= 60:
        return "STRONG"

    if rsi >= 50:
        return "POSITIVE"

    if rsi >= 40:
        return "NEUTRAL"

    if rsi >= 30:
        return "WEAK"

    return "OVERSOLD"


# =========================================================
# MACD STATE
# =========================================================

def get_macd_state(
    macd: float,
    signal: float,
    histogram: float,
) -> str:

    if any(
        pd.isna(value)
        for value in (
            macd,
            signal,
            histogram,
        )
    ):
        return "NO_DATA"

    if (
        macd > signal
        and histogram > 0
    ):
        return "BULLISH"

    if (
        macd < signal
        and histogram < 0
    ):
        return "BEARISH"

    return "MIXED"


# =========================================================
# VOLUME BREAKOUT
# =========================================================

def get_volume_breakout(
    volume_ratio: float
) -> str:

    if pd.isna(volume_ratio):
        return "NO_DATA"

    if volume_ratio >= 2.0:
        return "STRONG_BREAKOUT"

    if volume_ratio >= 1.5:
        return "BREAKOUT"

    if volume_ratio >= 1.2:
        return "BUILDING"

    return "NO"


# =========================================================
# TECHNICAL SCORE
# =========================================================

def calculate_technical_score(
    row: pd.Series
) -> float:

    score = 50.0

    alignment = row.get(
        "EMA_ALIGNMENT"
    )

    if alignment == "BULLISH":
        score += 15

    elif alignment == "BEARISH":
        score -= 15

    rsi = _safe_float(
        row.get("RSI_14")
    )

    if not pd.isna(rsi):

        if 55 <= rsi < 70:
            score += 10

        elif 45 <= rsi < 55:
            score += 3

        elif 30 <= rsi < 45:
            score -= 5

        elif rsi >= 70:
            score -= 3

        elif rsi < 30:
            score += 2

    macd_state = row.get(
        "MACD_STATE"
    )

    if macd_state == "BULLISH":
        score += 10

    elif macd_state == "BEARISH":
        score -= 10

    trend = row.get(
        "SUPERTREND_STATUS"
    )

    if trend == "BULLISH":
        score += 10

    elif trend == "BEARISH":
        score -= 10

    volume_ratio = _safe_float(
        row.get(
            "VOLUME_RATIO"
        )
    )

    if not pd.isna(volume_ratio):

        if volume_ratio >= 1.5:
            score += 5

        elif volume_ratio < 0.7:
            score -= 3

    return float(
        max(
            0.0,
            min(
                100.0,
                round(score, 1)
            )
        )
    )


# =========================================================
# TECHNICAL ZONE
# =========================================================

def get_technical_zone(
    score: float
) -> str:

    if pd.isna(score):
        return "NO_DATA"

    if score >= 75:
        return "STRONG"

    if score >= 60:
        return "POSITIVE"

    if score >= 45:
        return "NEUTRAL"

    if score >= 30:
        return "WEAK"

    return "VERY_WEAK"


# =========================================================
# DATA QUALITY
# =========================================================

def calculate_data_quality(
    df: pd.DataFrame
) -> float:

    if df is None or df.empty:
        return 0.0

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    checks = []

    for column in required:

        if column not in df.columns:

            checks.append(0.0)

        else:

            checks.append(
                float(
                    df[column]
                    .notna()
                    .mean()
                )
            )

    return round(
        float(
            np.mean(checks)
        ) * 100.0,
        2
    )


# =========================================================
# FRESHNESS
# =========================================================

def get_freshness_status(
    last_date: Any
) -> Tuple[
    Optional[int],
    str,
]:

    try:

        timestamp = pd.Timestamp(
            last_date
        )

        if timestamp.tzinfo is None:

            timestamp = timestamp.tz_localize(
                "UTC"
            )

        else:

            timestamp = timestamp.tz_convert(
                "UTC"
            )

        now = pd.Timestamp.now(
            tz="UTC"
        )

        age_seconds = (
            now - timestamp
        ).total_seconds()

        age_days = max(
            0,
            int(
                age_seconds / 86400
            )
        )

        if age_days <= 1:
            status = "FRESH"

        elif age_days <= FRESHNESS_LIMIT_DAYS:
            status = "RECENT"

        else:
            status = "STALE"

        return (
            age_days,
            status
        )

    except Exception:

        return (
            None,
            "UNKNOWN"
        )


# =========================================================
# DOWNLOAD MARKET DATA
# =========================================================

def download_market_data(
    symbol: str
) -> pd.DataFrame:

    ticker = _yahoo_symbol(
        symbol
    )

    data = yf.download(
        ticker,
        period=DATA_PERIOD,
        interval=DATA_INTERVAL,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data is None or data.empty:

        raise ValueError(
            f"No market data available for {ticker}"
        )

    return _normalise_ohlcv(
        data
    )


# =========================================================
# EMS INDEPENDENT EVIDENCE
# =========================================================

def build_ems_evidence(
    current: pd.Series
) -> Dict[str, Any]:

    ema_alignment = current.get(
        "EMA_ALIGNMENT"
    )

    macd_state = current.get(
        "MACD_STATE"
    )

    supertrend_status = current.get(
        "SUPERTREND_STATUS"
    )

    cmp = _safe_float(
        current.get("CMP")
    )

    ema20 = _safe_float(
        current.get("EMA_20")
    )

    ema50 = _safe_float(
        current.get("EMA_50")
    )

    volume_ratio = _safe_float(
        current.get(
            "VOLUME_RATIO"
        )
    )

    trend_breakdown = None

    if supertrend_status in {
        "BULLISH",
        "BEARISH",
    }:

        trend_breakdown = (
            supertrend_status == "BEARISH"
        )

    momentum_breakdown = None

    if macd_state in {
        "BULLISH",
        "BEARISH",
    }:

        momentum_breakdown = (
            macd_state == "BEARISH"
        )

    support_breakdown = None

    if (
        not pd.isna(cmp)
        and not pd.isna(ema20)
    ):

        support_breakdown = (
            cmp < ema20
        )

    relative_strength_breakdown = None

    if ema_alignment in {
        "BULLISH",
        "BEARISH",
        "MIXED",
    }:

        relative_strength_breakdown = (
            ema_alignment == "BEARISH"
        )

    risk_deterioration = None

    if (
        not pd.isna(cmp)
        and not pd.isna(ema50)
    ):

        risk_deterioration = (
            cmp < ema50
        )

    volume_confirmation = None

    if not pd.isna(volume_ratio):

        volume_confirmation = (
            volume_ratio
            >= VOLUME_BREAKOUT_THRESHOLD
        )

    return {

        "trend_breakdown":
            trend_breakdown,

        "momentum_breakdown":
            momentum_breakdown,

        "support_breakdown":
            support_breakdown,

        "volume_confirmation":
            volume_confirmation,

        "relative_strength_breakdown":
            relative_strength_breakdown,

        "risk_deterioration":
            risk_deterioration,

        # External EMS layer
        "above_exit_price":
            None,

        "ath_profit":
            None,

        "outperformance":
            None,
    }


# =========================================================
# ERROR RESULT
# =========================================================

def _error_result(
    clean_symbol: str,
    ticker: str,
    error: Exception
) -> Dict[str, Any]:

    return {

        "SYMBOL": clean_symbol,
        "TICKER": ticker,

        "STATUS": "ERROR",
        "NSE_STATUS": "ERROR",

        "DATA_DATE": None,
        "DATA_AGE_DAYS": None,
        "DATA_QUALITY_%": 0.0,
        "SIGNAL_ALLOWED": False,

        "CMP": np.nan,
        "PREVIOUS_CLOSE": np.nan,
        "CHANGE": np.nan,
        "CHANGE_%": np.nan,

        "OPEN": np.nan,
        "HIGH": np.nan,
        "LOW": np.nan,
        "CLOSE": np.nan,
        "VOLUME": np.nan,

        "AVG_VOLUME_20": np.nan,
        "VOLUME_RATIO": np.nan,
        "VOLUME_BREAKOUT": "NO_DATA",

        "52W_HIGH": np.nan,
        "52W_LOW": np.nan,
        "DISTANCE_FROM_52W_HIGH_%": np.nan,

        "EMA_10": np.nan,
        "EMA_20": np.nan,
        "EMA_50": np.nan,
        "EMA_100": np.nan,
        "EMA_200": np.nan,

        "EMA_ALIGNMENT": "NO_DATA",

        "RSI_14": np.nan,
        "RSI_STATE": "NO_DATA",

        "MACD": np.nan,
        "MACD_SIGNAL": np.nan,
        "MACD_HIST": np.nan,
        "MACD_STATE": "NO_DATA",

        "SUPERTREND": np.nan,
        "SUPERTREND_STATUS": "NO_DATA",

        "ATR_14": np.nan,

        "TECHNICAL_SCORE": np.nan,
        "TECHNICAL_ZONE": "NO_DATA",

        "trend_breakdown": None,
        "momentum_breakdown": None,
        "support_breakdown": None,
        "volume_confirmation": None,
        "relative_strength_breakdown": None,
        "risk_deterioration": None,

        "above_exit_price": None,
        "ath_profit": None,
        "outperformance": None,

        "FUNDAMENTAL_DATA_AVAILABLE": False,
        "REFERENCE_MATCH": "NONE",

        "ERROR": str(error),
    }


# =========================================================
# MAIN NSE ENGINE
# =========================================================

def fetch_nse_data(
    symbol: str
) -> Dict[str, Any]:

    clean_symbol = _clean_symbol(
        symbol
    )

    ticker = _yahoo_symbol(
        clean_symbol
    )

    try:

        df = download_market_data(
            clean_symbol
        )

        if df.empty:
            raise ValueError(
                "Empty market dataframe"
            )

        close = df["Close"]

        # -------------------------------------------------
        # PRICE
        # -------------------------------------------------

        cmp = _safe_float(
            close.iloc[-1]
        )

        previous_close = (
            _safe_float(
                close.iloc[-2]
            )
            if len(close) >= 2
            else cmp
        )

        change = (
            cmp - previous_close
        )

        change_pct = (
            (
                change
                /
                previous_close
            )
            * 100.0
            if (
                not pd.isna(previous_close)
                and previous_close != 0
            )
            else np.nan
        )

        # -------------------------------------------------
        # EMA
        # -------------------------------------------------

        for period in EMA_PERIODS:

            df[
                f"EMA_{period}"
            ] = close.ewm(
                span=period,
                adjust=False,
                min_periods=period,
            ).mean()

        # -------------------------------------------------
        # RSI
        # -------------------------------------------------

        df["RSI_14"] = calculate_rsi(
            close,
            RSI_PERIOD
        )

        # -------------------------------------------------
        # MACD
        # -------------------------------------------------

        (
            df["MACD"],
            df["MACD_SIGNAL"],
            df["MACD_HIST"],
        ) = calculate_macd(
            close
        )

        # -------------------------------------------------
        # ATR
        # -------------------------------------------------

        df["ATR_14"] = calculate_atr(
            df,
            ATR_PERIOD
        )

        # -------------------------------------------------
        # SUPERTREND
        # -------------------------------------------------

        (
            df["SUPERTREND"],
            df["SUPERTREND_STATUS"],
        ) = calculate_supertrend(
            df,
            SUPERTREND_PERIOD,
            SUPERTREND_MULTIPLIER,
        )

        # -------------------------------------------------
        # VOLUME
        # -------------------------------------------------

        df["AVG_VOLUME_20"] = (
            df["Volume"]
            .rolling(
                VOLUME_AVG_PERIOD,
                min_periods=VOLUME_AVG_PERIOD,
            )
            .mean()
        )

        df["VOLUME_RATIO"] = (
            df["Volume"]
            /
            df["AVG_VOLUME_20"]
        )

        df["VOLUME_BREAKOUT"] = (
            df["VOLUME_RATIO"]
            .apply(
                get_volume_breakout
            )
        )

        # -------------------------------------------------
        # CURRENT ROW
        # -------------------------------------------------

        current = df.iloc[-1].copy()

        current["CMP"] = cmp

        # -------------------------------------------------
        # EMA ALIGNMENT
        # -------------------------------------------------

        current["EMA_ALIGNMENT"] = (
            get_ema_alignment(
                current
            )
        )

        # -------------------------------------------------
        # RSI STATE
        # -------------------------------------------------

        current["RSI_STATE"] = (
            get_rsi_state(
                _safe_float(
                    current.get(
                        "RSI_14"
                    )
                )
            )
        )

        # -------------------------------------------------
        # MACD STATE
        # -------------------------------------------------

        current["MACD_STATE"] = (
            get_macd_state(
                _safe_float(
                    current.get(
                        "MACD"
                    )
                ),
                _safe_float(
                    current.get(
                        "MACD_SIGNAL"
                    )
                ),
                _safe_float(
                    current.get(
                        "MACD_HIST"
                    )
                ),
            )
        )

        # -------------------------------------------------
        # TECHNICAL SCORE
        # -------------------------------------------------

        technical_score = (
            calculate_technical_score(
                current
            )
        )

        technical_zone = (
            get_technical_zone(
                technical_score
            )
        )

        # -------------------------------------------------
        # 52 WEEK
        # -------------------------------------------------

        recent_52w = df.tail(
            252
        )

        high_52w = _safe_float(
            recent_52w["High"].max()
        )

        low_52w = _safe_float(
            recent_52w["Low"].min()
        )

        distance_from_high = (
            (
                high_52w - cmp
            )
            /
            high_52w
            * 100.0
            if (
                not pd.isna(high_52w)
                and high_52w > 0
            )
            else np.nan
        )

        # -------------------------------------------------
        # FRESHNESS
        # -------------------------------------------------

        last_date = df.index[-1]

        age_days, freshness = (
            get_freshness_status(
                last_date
            )
        )

        # -------------------------------------------------
        # DATA QUALITY
        # -------------------------------------------------

        data_quality = (
            calculate_data_quality(
                df
            )
        )

        # -------------------------------------------------
        # SIGNAL SAFETY
        # -------------------------------------------------

        signal_allowed = (
            freshness in {
                "FRESH",
                "RECENT",
            }
            and
            data_quality
            >= MIN_DATA_QUALITY
            and
            not pd.isna(cmp)
        )

        # -------------------------------------------------
        # EMS
        # -------------------------------------------------

        ems_evidence = (
            build_ems_evidence(
                current
            )
        )

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        return {

            "SYMBOL":
                clean_symbol,

            "TICKER":
                ticker,

            "STATUS":
                freshness,

            "NSE_STATUS":
                freshness,

            "DATA_DATE":
                pd.Timestamp(
                    last_date
                ).strftime(
                    "%Y-%m-%d"
                ),

            "DATA_AGE_DAYS":
                age_days,

            "DATA_QUALITY_%":
                data_quality,

            "SIGNAL_ALLOWED":
                signal_allowed,

            "CMP":
                cmp,

            "PREVIOUS_CLOSE":
                previous_close,

            "CHANGE":
                change,

            "CHANGE_%":
                change_pct,

            "OPEN":
                _safe_float(
                    current.get("Open")
                ),

            "HIGH":
                _safe_float(
                    current.get("High")
                ),

            "LOW":
                _safe_float(
                    current.get("Low")
                ),

            "CLOSE":
                cmp,

            "VOLUME":
                _safe_float(
                    current.get("Volume")
                ),

            "AVG_VOLUME_20":
                _safe_float(
                    current.get(
                        "AVG_VOLUME_20"
                    )
                ),

            "VOLUME_RATIO":
                _safe_float(
                    current.get(
                        "VOLUME_RATIO"
                    )
                ),

            "VOLUME_BREAKOUT":
                current.get(
                    "VOLUME_BREAKOUT",
                    "NO_DATA"
                ),

            "52W_HIGH":
                high_52w,

            "52W_LOW":
                low_52w,

            "DISTANCE_FROM_52W_HIGH_%":
                distance_from_high,

            "EMA_10":
                _safe_float(
                    current.get(
                        "EMA_10"
                    )
                ),

            "EMA_20":
                _safe_float(
                    current.get(
                        "EMA_20"
                    )
                ),

            "EMA_50":
                _safe_float(
                    current.get(
                        "EMA_50"
                    )
                ),

            "EMA_100":
                _safe_float(
                    current.get(
                        "EMA_100"
                    )
                ),

            "EMA_200":
                _safe_float(
                    current.get(
                        "EMA_200"
                    )
                ),

            "EMA_ALIGNMENT":
                current.get(
                    "EMA_ALIGNMENT",
                    "INSUFFICIENT_DATA"
                ),

            "RSI_14":
                _safe_float(
                    current.get(
                        "RSI_14"
                    )
                ),

            "RSI_STATE":
                current.get(
                    "RSI_STATE",
                    "NO_DATA"
                ),

            "MACD":
                _safe_float(
                    current.get(
                        "MACD"
                    )
                ),

            "MACD_SIGNAL":
                _safe_float(
                    current.get(
                        "MACD_SIGNAL"
                    )
                ),

            "MACD_HIST":
                _safe_float(
                    current.get(
                        "MACD_HIST"
                    )
                ),

            "MACD_STATE":
                current.get(
                    "MACD_STATE",
                    "NO_DATA"
                ),

            "SUPERTREND":
                _safe_float(
                    current.get(
                        "SUPERTREND"
                    )
                ),

            "SUPERTREND_STATUS":
                current.get(
                    "SUPERTREND_STATUS",
                    "NO_DATA"
                ),

            "ATR_14":
                _safe_float(
                    current.get(
                        "ATR_14"
                    )
                ),

            "TECHNICAL_SCORE":
                technical_score,

            "TECHNICAL_ZONE":
                technical_zone,

            **ems_evidence,

            "FUNDAMENTAL_DATA_AVAILABLE":
                False,

            "REFERENCE_MATCH":
                "NONE",
        }

    except Exception as error:

        return _error_result(
            clean_symbol,
            ticker,
            error
        )


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    TEST_SYMBOL = "CEMPRO"

    result = fetch_nse_data(
        TEST_SYMBOL
    )

    print()
    print("=" * 75)
    print("🐂 R.S MASTER STOCK GUIDE V3.1")
    print("NSE MARKET DATA ENGINE TEST")
    print("=" * 75)

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

    print("=" * 75)
