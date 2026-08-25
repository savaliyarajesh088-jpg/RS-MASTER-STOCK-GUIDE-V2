"""
R.S MASTER STOCK GUIDE V2
NSE Market Data + Core Technical Engine

Responsibilities:
- NSE/Yahoo price data
- CMP / change
- OHLCV
- 52-week high / low
- EMA 10/20/50/100/200
- RSI 14
- MACD 12/26/9
- Supertrend 10/3
- Volume ratio
- Data freshness / quality
- Basic ATR
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd
import yfinance as yf


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PERIOD = "2y"
DATA_INTERVAL = "1d"

EMA_PERIODS = [10, 20, 50, 100, 200]

RSI_PERIOD = 14

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

SUPERTREND_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0

ATR_PERIOD = 14

VOLUME_AVG_PERIOD = 20

FRESHNESS_LIMIT_DAYS = 5


# =========================================================
# BASIC HELPERS
# =========================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""

    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def _clean_symbol(symbol: str) -> str:
    """Convert NSE symbol to Yahoo Finance NSE ticker."""

    symbol = str(symbol).strip().upper()

    if symbol.endswith(".NS"):
        return symbol

    return f"{symbol}.NS"


def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Protect against yfinance MultiIndex columns."""

    if not isinstance(df.columns, pd.MultiIndex):
        return df

    flattened = []

    for column in df.columns:

        if isinstance(column, tuple):

            value = None

            for item in column:

                if item in {
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Adj Close",
                    "Volume",
                }:
                    value = item
                    break

            flattened.append(
                value if value else str(column[0])
            )

        else:
            flattened.append(str(column))

    df = df.copy()
    df.columns = flattened

    return df


def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only required OHLCV columns and clean numeric data."""

    df = _flatten_yfinance_columns(df)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
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
            "Close"
        ]
    )

    df = df.sort_index()

    return df


# =========================================================
# RSI
# =========================================================

def calculate_rsi(
    close: pd.Series,
    period: int = RSI_PERIOD
) -> pd.Series:
    """Calculate RSI using Wilder-style smoothing."""

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    rsi = rsi.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return rsi


# =========================================================
# MACD
# =========================================================

def calculate_macd(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL
):
    """Calculate MACD line, signal line and histogram."""

    ema_fast = close.ewm(
        span=fast,
        adjust=False
    ).mean()

    ema_slow = close.ewm(
        span=slow,
        adjust=False
    ).mean()

    macd_line = (
        ema_fast - ema_slow
    )

    signal_line = macd_line.ewm(
        span=signal,
        adjust=False
    ).mean()

    histogram = (
        macd_line - signal_line
    )

    return (
        macd_line,
        signal_line,
        histogram
    )


# =========================================================
# TRUE RANGE / ATR
# =========================================================

def calculate_atr(
    df: pd.DataFrame,
    period: int = ATR_PERIOD
) -> pd.Series:
    """Calculate Average True Range."""

    high = df["High"]

    low = df["Low"]

    close = df["Close"]

    previous_close = close.shift(1)

    tr1 = (
        high - low
    )

    tr2 = (
        high - previous_close
    ).abs()

    tr3 = (
        low - previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    return atr


# =========================================================
# SUPERTREND
# =========================================================

def calculate_supertrend(
    df: pd.DataFrame,
    period: int = SUPERTREND_PERIOD,
    multiplier: float = SUPERTREND_MULTIPLIER
):
    """
    Calculate Supertrend.

    Returns:
        supertrend series
        direction series
    """

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    atr = calculate_atr(
        df,
        period
    )

    hl2 = (
        high + low
    ) / 2

    basic_upper = (
        hl2 + multiplier * atr
    )

    basic_lower = (
        hl2 - multiplier * atr
    )

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()

    direction = pd.Series(
        index=df.index,
        dtype="object"
    )

    supertrend = pd.Series(
        index=df.index,
        dtype="float64"
    )

    if len(df) == 0:
        return supertrend, direction

    direction.iloc[0] = "BULLISH"

    supertrend.iloc[0] = np.nan

    for i in range(1, len(df)):

        if (
            pd.isna(
                basic_upper.iloc[i]
            )
            or
            pd.isna(
                basic_lower.iloc[i]
            )
        ):

            final_upper.iloc[i] = (
                basic_upper.iloc[i]
            )

            final_lower.iloc[i] = (
                basic_lower.iloc[i]
            )

            direction.iloc[i] = (
                direction.iloc[i - 1]
            )

            continue

        previous_close = close.iloc[
            i - 1
        ]

        previous_upper = final_upper.iloc[
            i - 1
        ]

        previous_lower = final_lower.iloc[
            i - 1
        ]

        if (
            pd.isna(previous_upper)
            or
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

        if (
            pd.isna(previous_lower)
            or
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

        if previous_direction == "BEARISH":

            if (
                close.iloc[i]
                > final_upper.iloc[i]
            ):

                direction.iloc[i] = (
                    "BULLISH"
                )

            else:

                direction.iloc[i] = (
                    "BEARISH"
                )

        else:

            if (
                close.iloc[i]
                < final_lower.iloc[i]
            ):

                direction.iloc[i] = (
                    "BEARISH"
                )

            else:

                direction.iloc[i] = (
                    "BULLISH"
                )

        if direction.iloc[i] == "BULLISH":

            supertrend.iloc[i] = (
                final_lower.iloc[i]
            )

        else:

            supertrend.iloc[i] = (
                final_upper.iloc[i]
            )

    return (
        supertrend,
        direction
    )


# =========================================================
# EMA ALIGNMENT
# =========================================================

def get_ema_alignment(row: pd.Series) -> str:
    """Classify EMA structure."""

    ema10 = _safe_float(
        row.get("EMA_10"),
        np.nan
    )

    ema20 = _safe_float(
        row.get("EMA_20"),
        np.nan
    )

    ema50 = _safe_float(
        row.get("EMA_50"),
        np.nan
    )

    ema100 = _safe_float(
        row.get("EMA_100"),
        np.nan
    )

    ema200 = _safe_float(
        row.get("EMA_200"),
        np.nan
    )

    values = [
        ema10,
        ema20,
        ema50,
        ema100,
        ema200
    ]

    if any(
        pd.isna(value)
        for value in values
    ):
        return "INSUFFICIENT_DATA"

    if (
        ema10
        > ema20
        > ema50
        > ema100
        > ema200
    ):
        return "BULLISH"

    if (
        ema10
        < ema20
        < ema50
        < ema100
        < ema200
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
    histogram: float
) -> str:

    if any(
        pd.isna(value)
        for value in [
            macd,
            signal,
            histogram
        ]
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
# VOLUME STATE
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

    # EMA
    alignment = row.get(
        "EMA_ALIGNMENT"
    )

    if alignment == "BULLISH":
        score += 15

    elif alignment == "BEARISH":
        score -= 15

    # RSI
    rsi = _safe_float(
        row.get("RSI_14"),
        np.nan
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

    # MACD
    macd_state = row.get(
        "MACD_STATE"
    )

    if macd_state == "BULLISH":
        score += 10

    elif macd_state == "BEARISH":
        score -= 10

    # Supertrend
    supertrend_status = row.get(
        "SUPERTREND_STATUS"
    )

    if supertrend_status == "BULLISH":
        score += 10

    elif supertrend_status == "BEARISH":
        score -= 10

    # Volume
    volume_ratio = _safe_float(
        row.get("VOLUME_RATIO"),
        np.nan
    )

    if not pd.isna(volume_ratio):

        if volume_ratio >= 1.5:
            score += 5

        elif volume_ratio < 0.7:
            score -= 3

    return float(
        max(
            0,
            min(
                100,
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

    if df.empty:
        return 0.0

    checks = []

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in required:

        checks.append(
            float(
                df[column].notna().mean()
            )
        )

    return round(
        float(
            np.mean(checks)
        ) * 100,
        2
    )


# =========================================================
# DATA FRESHNESS
# =========================================================

def get_freshness_status(
    last_date
):

    try:

        if isinstance(
            last_date,
            pd.Timestamp
        ):

            last_date = (
                last_date.to_pydatetime()
            )

        if last_date.tzinfo is None:

            last_date = last_date.replace(
                tzinfo=timezone.utc
            )

        now = datetime.now(
            timezone.utc
        )

        age_days = (
            now - last_date
        ).days

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

    ticker = _clean_symbol(
        symbol
    )

    data = yf.download(
        ticker,
        period=DATA_PERIOD,
        interval=DATA_INTERVAL,
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if data is None or data.empty:

        raise ValueError(
            f"No market data available for {ticker}"
        )

    return _normalise_ohlcv(
        data
    )


# =========================================================
# MAIN NSE ENGINE
# =========================================================

def fetch_nse_data(
    symbol: str
) -> Dict[str, Any]:
    """
    Main NSE data engine.

    Returns a dictionary consumed by app.py.
    """

    clean_symbol = (
        str(symbol)
        .strip()
        .upper()
    )

    ticker = _clean_symbol(
        clean_symbol
    )

    try:

        df = download_market_data(
            clean_symbol
        )

        if df.empty:

            return {
                "SYMBOL": clean_symbol,
                "STATUS": "NO_DATA",
                "ERROR": "Empty market data"
            }

        close = df["Close"]

        # -------------------------------------------------
        # CHANGE
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
            (change / previous_close) * 100
            if previous_close != 0
            else 0
        )

        # -------------------------------------------------
        # EMA
        # -------------------------------------------------

        for period in EMA_PERIODS:

            df[
                f"EMA_{period}"
            ] = close.ewm(
                span=period,
                adjust=False
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
            df["MACD_HIST"]
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
            df["SUPERTREND_STATUS"]
        ) = calculate_supertrend(
            df,
            SUPERTREND_PERIOD,
            SUPERTREND_MULTIPLIER
        )

        # -------------------------------------------------
        # VOLUME
        # -------------------------------------------------

        df["AVG_VOLUME_20"] = (
            df["Volume"]
            .rolling(
                VOLUME_AVG_PERIOD
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
                current.get(
                    "RSI_14"
                )
            )
        )

        # -------------------------------------------------
        # MACD STATE
        # -------------------------------------------------

        current["MACD_STATE"] = (
            get_macd_state(
                current.get("MACD"),
                current.get("MACD_SIGNAL"),
                current.get("MACD_HIST")
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

        # -------------------------------------------------
        # ATR
        # -------------------------------------------------

        atr = _safe_float(
            current.get(
                "ATR_14"
            )
        )

        # -------------------------------------------------
        # DATA FRESHNESS
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
            freshness
            in {
                "FRESH",
                "RECENT"
            }
            and data_quality >= 90
        )

        # -------------------------------------------------
        # RETURN
        # -------------------------------------------------

        return {

            "SYMBOL": clean_symbol,

            "TICKER": ticker,

            "STATUS": freshness,

            "DATA_DATE": (
                last_date.strftime(
                    "%Y-%m-%d"
                )
            ),

            "DATA_AGE_DAYS": age_days,

            "DATA_QUALITY_%": data_quality,

            "SIGNAL_ALLOWED": (
                signal_allowed
            ),

            # PRICE
            "CMP": cmp,

            "PREVIOUS_CLOSE": previous_close,

            "CHANGE": change,

            "CHANGE_%": change_pct,

            "OPEN": _safe_float(
                current.get("Open")
            ),

            "HIGH": _safe_float(
                current.get("High")
            ),

            "LOW": _safe_float(
                current.get("Low")
            ),

            "CLOSE": cmp,

            "VOLUME": _safe_float(
                current.get("Volume")
            ),

            "AVG_VOLUME_20": _safe_float(
                current.get(
                    "AVG_VOLUME_20"
                )
            ),

            "VOLUME_RATIO": _safe_float(
                current.get(
                    "VOLUME_RATIO"
                )
            ),

            "VOLUME_BREAKOUT": current.get(
                "VOLUME_BREAKOUT",
                "NO"
            ),

            # 52 WEEK
            "52W_HIGH": high_52w,

            "52W_LOW": low_52w,

            "DISTANCE_FROM_52W_HIGH_%": (
                (
                    (high_52w - cmp)
                    / high_52w
                ) * 100
                if high_52w
                else 0
            ),

            # EMA
            "EMA_10": _safe_float(
                current.get("EMA_10")
            ),

            "EMA_20": _safe_float(
                current.get("EMA_20")
            ),

            "EMA_50": _safe_float(
                current.get("EMA_50")
            ),

            "EMA_100": _safe_float(
                current.get("EMA_100")
            ),

            "EMA_200": _safe_float(
                current.get("EMA_200")
            ),

            "EMA_ALIGNMENT": current.get(
                "EMA_ALIGNMENT",
                "MIXED"
            ),

            # RSI
            "RSI_14": _safe_float(
                current.get("RSI_14"),
                np.nan
            ),

            "RSI_STATE": current.get(
                "RSI_STATE",
                "NO_DATA"
            ),

            # MACD
            "MACD": _safe_float(
                current.get("MACD"),
                np.nan
            ),

            "MACD_SIGNAL": _safe_float(
                current.get(
                    "MACD_SIGNAL"
                ),
                np.nan
            ),

            "MACD_HIST": _safe_float(
                current.get(
                    "MACD_HIST"
                ),
                np.nan
            ),

            "MACD_STATE": current.get(
                "MACD_STATE",
                "NO_DATA"
            ),

            # SUPERTREND
            "SUPERTREND": _safe_float(
                current.get(
                    "SUPERTREND"
                ),
                np.nan
            ),

            "SUPERTREND_STATUS": current.get(
                "SUPERTREND_STATUS",
                "NO_DATA"
            ),

            # ATR
            "ATR_14": atr,

            # SCORE
            "TECHNICAL_SCORE": (
                technical_score
            ),

            "TECHNICAL_ZONE": (
                technical_zone
            ),

        }

    except Exception as error:

        return {

            "SYMBOL": clean_symbol,

            "TICKER": ticker,

            "STATUS": "ERROR",

            "DATA_DATE": None,

            "DATA_AGE_DAYS": None,

            "DATA_QUALITY_%": 0.0,

            "SIGNAL_ALLOWED": False,

            "CMP": 0.0,

            "CHANGE": 0.0,

            "CHANGE_%": 0.0,

            "VOLUME": 0.0,

            "AVG_VOLUME_20": 0.0,

            "VOLUME_RATIO": 0.0,

            "VOLUME_BREAKOUT": "NO",

            "52W_HIGH": 0.0,

            "52W_LOW": 0.0,

            "EMA_10": 0.0,

            "EMA_20": 0.0,

            "EMA_50": 0.0,

            "EMA_100": 0.0,

            "EMA_200": 0.0,

            "EMA_ALIGNMENT": "NO_DATA",

            "RSI_14": np.nan,

            "RSI_STATE": "NO_DATA",

            "MACD": np.nan,

            "MACD_SIGNAL": np.nan,

            "MACD_HIST": np.nan,

            "MACD_STATE": "NO_DATA",

            "SUPERTREND": np.nan,

            "SUPERTREND_STATUS": "NO_DATA",

            "ATR_14": 0.0,

            "TECHNICAL_SCORE": 0.0,

            "TECHNICAL_ZONE": "NO_DATA",

            "ERROR": str(error)

        }


# =========================================================
# OPTIONAL DIRECT TEST
# =========================================================

if __name__ == "__main__":

    test_symbol = "CEMPRO"

    result = fetch_nse_data(
        test_symbol
    )

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )
