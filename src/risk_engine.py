# =========================================================
# 🛡️ R.S MASTER STOCK GUIDE V2
# RISK ENGINE V2 — SINGLE SOURCE OF TRUTH
# =========================================================

import numpy as np
import pandas as pd


# =========================================================
# SAFE HELPERS
# =========================================================

def safe_float(value, default=np.nan):

    try:
        if value is None or pd.isna(value):
            return default

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def clamp(value, minimum=0.0, maximum=100.0):

    value = safe_float(value, minimum)

    return max(
        minimum,
        min(maximum, value)
    )


# =========================================================
# OHLCV PREPARATION
# =========================================================

def prepare_ohlcv(df):

    if df is None or df.empty:
        return None

    data = df.copy()

    if isinstance(data.columns, pd.MultiIndex):

        data.columns = [
            col[0]
            for col in data.columns
        ]

    required = ["High", "Low", "Close"]

    if not all(
        col in data.columns
        for col in required
    ):
        return None

    for col in required:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data.dropna(
        subset=required,
        inplace=True
    )

    if data.empty:
        return None

    return data


# =========================================================
# ATR
# =========================================================

def calculate_atr(df, period=14):

    data = prepare_ohlcv(df)

    if data is None:
        return np.nan

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = (
        tr
        .rolling(
            period,
            min_periods=period
        )
        .mean()
    )

    if atr.dropna().empty:
        return np.nan

    return safe_float(
        atr.dropna().iloc[-1]
    )


# =========================================================
# ATR %
# =========================================================

def calculate_atr_percent(cmp, atr):

    cmp = safe_float(cmp)
    atr = safe_float(atr)

    if (
        not np.isfinite(cmp)
        or cmp <= 0
        or not np.isfinite(atr)
        or atr < 0
    ):
        return np.nan

    return round(
        atr / cmp * 100,
        2
    )


# =========================================================
# VOLATILITY
# =========================================================

def calculate_volatility(df, period=20):

    data = prepare_ohlcv(df)

    if data is None:
        return np.nan

    returns = (
        data["Close"]
        .pct_change()
        .dropna()
    )

    if len(returns) < period:
        return np.nan

    value = (
        returns
        .tail(period)
        .std()
        * np.sqrt(252)
        * 100
    )

    return safe_float(value)


# =========================================================
# MAX DRAWDOWN
# =========================================================

def calculate_max_drawdown(df):

    data = prepare_ohlcv(df)

    if data is None:
        return np.nan

    close = data["Close"]

    running_high = close.cummax()

    drawdown = (
        close / running_high - 1
    ) * 100

    return safe_float(
        drawdown.min()
    )


# =========================================================
# 52 WEEK
# =========================================================

def calculate_52_week_levels(df):

    result = {
        "52W_HIGH": np.nan,
        "52W_LOW": np.nan,
        "DISTANCE_FROM_52W_HIGH_%": np.nan,
        "DISTANCE_FROM_52W_LOW_%": np.nan
    }

    data = prepare_ohlcv(df)

    if data is None:
        return result

    data = data.tail(252)

    high = safe_float(
        data["High"].max()
    )

    low = safe_float(
        data["Low"].min()
    )

    cmp = safe_float(
        data["Close"].iloc[-1]
    )

    result["52W_HIGH"] = high
    result["52W_LOW"] = low

    if np.isfinite(cmp) and cmp > 0:

        if np.isfinite(high) and high > 0:

            result[
                "DISTANCE_FROM_52W_HIGH_%"
            ] = round(
                (cmp - high) / high * 100,
                2
            )

        if np.isfinite(low) and low > 0:

            result[
                "DISTANCE_FROM_52W_LOW_%"
            ] = round(
                (cmp - low) / low * 100,
                2
            )

    return result


# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

def calculate_support_resistance(
    df,
    lookback=20
):

    result = {
        "SUPPORT": np.nan,
        "RESISTANCE": np.nan
    }

    data = prepare_ohlcv(df)

    if data is None:
        return result

    data = data.tail(
        max(int(lookback), 5)
    )

    result["SUPPORT"] = safe_float(
        data["Low"].min()
    )

    result["RESISTANCE"] = safe_float(
        data["High"].max()
    )

    return result


# =========================================================
# RISK SCORE
# =========================================================

def calculate_risk_score(
    atr_percent,
    volatility,
    max_drawdown,
    distance_from_high
):

    score = 100.0

    atr_percent = safe_float(
        atr_percent,
        0
    )

    volatility = safe_float(
        volatility,
        0
    )

    max_drawdown = safe_float(
        max_drawdown,
        0
    )

    distance_from_high = safe_float(
        distance_from_high,
        0
    )

    # ATR
    if atr_percent > 8:
        score -= 30
    elif atr_percent > 6:
        score -= 22
    elif atr_percent > 4:
        score -= 14
    elif atr_percent > 3:
        score -= 8
    elif atr_percent > 2:
        score -= 3

    # Volatility
    if volatility > 60:
        score -= 25
    elif volatility > 45:
        score -= 18
    elif volatility > 35:
        score -= 12
    elif volatility > 25:
        score -= 6

    # Drawdown
    drawdown = abs(max_drawdown)

    if drawdown > 40:
        score -= 25
    elif drawdown > 30:
        score -= 18
    elif drawdown > 20:
        score -= 12
    elif drawdown > 10:
        score -= 6

    # 52W distance
    if distance_from_high < -30:
        score -= 15
    elif distance_from_high < -20:
        score -= 10
    elif distance_from_high < -10:
        score -= 5

    return round(
        clamp(score),
        1
    )


# =========================================================
# RISK LEVEL
# =========================================================

def get_risk_level(score):

    score = safe_float(score, 0)

    if score >= 80:
        return "🟢 LOW"

    if score >= 65:
        return "🟡 MODERATE"

    if score >= 45:
        return "🟠 HIGH"

    return "🔴 VERY HIGH"


# =========================================================
# RISK %
# =========================================================

def calculate_risk_percent(score):

    return round(
        100 - clamp(score),
        1
    )


# =========================================================
# STOP LOSS
# =========================================================

def calculate_stop_loss(
    cmp,
    atr,
    support=np.nan
):

    cmp = safe_float(cmp)
    atr = safe_float(atr)
    support = safe_float(support)

    if (
        not np.isfinite(cmp)
        or cmp <= 0
    ):
        return np.nan

    if np.isfinite(atr) and atr > 0:

        atr_stop = cmp - (
            2.0 * atr
        )

    else:

        atr_stop = cmp * 0.95

    if (
        np.isfinite(support)
        and support > 0
        and support < cmp
    ):

        support_stop = support * 0.98

        stop = min(
            atr_stop,
            support_stop
        )

    else:

        stop = atr_stop

    # Maximum allowed loss = 20%
    minimum_stop = cmp * 0.80

    stop = max(
        stop,
        minimum_stop
    )

    return round(
        stop,
        2
    )


# =========================================================
# SWING TARGET
# =========================================================

def calculate_swing_target(
    cmp,
    atr,
    resistance=np.nan
):

    cmp = safe_float(cmp)
    atr = safe_float(atr)
    resistance = safe_float(resistance)

    if (
        not np.isfinite(cmp)
        or cmp <= 0
    ):
        return np.nan

    if np.isfinite(atr) and atr > 0:

        target = cmp + (
            2.5 * atr
        )

    else:

        target = cmp * 1.08

    if (
        np.isfinite(resistance)
        and resistance > cmp
    ):

        target = max(
            target,
            resistance
        )

    return round(
        target,
        2
    )


# =========================================================
# LONG TERM TARGET
# =========================================================

def calculate_long_term_target(
    cmp,
    atr,
    high_52=np.nan
):

    cmp = safe_float(cmp)
    atr = safe_float(atr)
    high_52 = safe_float(high_52)

    if (
        not np.isfinite(cmp)
        or cmp <= 0
    ):
        return np.nan

    if np.isfinite(atr) and atr > 0:

        target = cmp + (
            5.0 * atr
        )

    else:

        target = cmp * 1.20

    if (
        np.isfinite(high_52)
        and high_52 > cmp
    ):

        target = max(
            target,
            high_52
        )

    return round(
        target,
        2
    )


# =========================================================
# RISK / REWARD
# =========================================================

def calculate_risk_reward(
    cmp,
    stop_loss,
    target
):

    cmp = safe_float(cmp)
    stop_loss = safe_float(stop_loss)
    target = safe_float(target)

    if not all(
        np.isfinite(x)
        for x in [
            cmp,
            stop_loss,
            target
        ]
    ):
        return np.nan

    risk = cmp - stop_loss
    reward = target - cmp

    if risk <= 0:
        return np.nan

    return round(
        reward / risk,
        2
    )


# =========================================================
# SAFETY GATE
# =========================================================

def calculate_safety_gate(
    risk_score,
    risk_reward,
    cmp,
    stop_loss
):

    risk_score = safe_float(
        risk_score,
        0
    )

    risk_reward = safe_float(
        risk_reward
    )

    cmp = safe_float(cmp)
    stop_loss = safe_float(stop_loss)

    if risk_score < 30:
        return "🔴 BLOCKED"

    if (
        np.isfinite(risk_reward)
        and risk_reward < 1
    ):
        return "🔴 BLOCKED"

    if (
        np.isfinite(cmp)
        and np.isfinite(stop_loss)
        and stop_loss >= cmp
    ):
        return "🔴 BLOCKED"

    if risk_score < 45:
        return "🟠 CAUTION"

    if risk_score < 65:
        return "🟡 WATCH"

    return "🟢 SAFE"


# =========================================================
# MASTER RISK ENGINE
# =========================================================

def calculate_risk(
    df,
    cmp=None,
    support=None,
    resistance=None
):

    data = prepare_ohlcv(df)

    if data is None:

        return {
            "RISK_SCORE": 0.0,
            "RISK_LEVEL": "🔴 VERY HIGH",
            "RISK_%": 100.0,
            "SAFETY_GATE": "🔴 BLOCKED"
        }

    if cmp is None:

        cmp = safe_float(
            data["Close"].iloc[-1]
        )

    else:

        cmp = safe_float(cmp)

    atr = calculate_atr(
        data,
        14
    )

    atr_percent = calculate_atr_percent(
        cmp,
        atr
    )

    volatility = calculate_volatility(
        data,
        20
    )

    max_drawdown = calculate_max_drawdown(
        data
    )

    week = calculate_52_week_levels(
        data
    )

    sr = calculate_support_resistance(
        data,
        20
    )

    if support is None:
        support = sr["SUPPORT"]

    if resistance is None:
        resistance = sr["RESISTANCE"]

    risk_score = calculate_risk_score(
        atr_percent,
        volatility,
        max_drawdown,
        week[
            "DISTANCE_FROM_52W_HIGH_%"
        ]
    )

    risk_level = get_risk_level(
        risk_score
    )

    risk_percent = calculate_risk_percent(
        risk_score
    )

    stop_loss = calculate_stop_loss(
        cmp,
        atr,
        support
    )

    swing_target = calculate_swing_target(
        cmp,
        atr,
        resistance
    )

    long_target = calculate_long_term_target(
        cmp,
        atr,
        week["52W_HIGH"]
    )

    risk_reward = calculate_risk_reward(
        cmp,
        stop_loss,
        swing_target
    )

    safety_gate = calculate_safety_gate(
        risk_score,
        risk_reward,
        cmp,
        stop_loss
    )

    return {

        "CMP": round(cmp, 2),

        "ATR": round(atr, 2)
        if np.isfinite(atr)
        else np.nan,

        "ATR_%": atr_percent,

        "VOLATILITY_%": round(
            volatility,
            2
        ) if np.isfinite(volatility)
        else np.nan,

        "MAX_DRAWDOWN_%": round(
            max_drawdown,
            2
        ) if np.isfinite(max_drawdown)
        else np.nan,

        "52W_HIGH": week["52W_HIGH"],

        "52W_LOW": week["52W_LOW"],

        "DISTANCE_FROM_52W_HIGH_%":
            week[
                "DISTANCE_FROM_52W_HIGH_%"
            ],

        "DISTANCE_FROM_52W_LOW_%":
            week[
                "DISTANCE_FROM_52W_LOW_%"
            ],

        "SUPPORT": round(
            support,
            2
        ) if np.isfinite(
            safe_float(support)
        ) else np.nan,

        "RESISTANCE": round(
            resistance,
            2
        ) if np.isfinite(
            safe_float(resistance)
        ) else np.nan,

        "RISK_SCORE": risk_score,

        "RISK_LEVEL": risk_level,

        "RISK_%": risk_percent,

        "STOP_LOSS": stop_loss,

        "SWING_TARGET": swing_target,

        "LONG_TERM_TARGET": long_target,

        "RISK_REWARD": risk_reward,

        "SAFETY_GATE": safety_gate
    }


# =========================================================
# ALIAS
# =========================================================

run_risk_engine = calculate_risk
