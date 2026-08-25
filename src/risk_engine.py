# =========================================================
# 🛡️ R.S MASTER STOCK GUIDE V2.1
# RISK ENGINE
# =========================================================

import pandas as pd
import numpy as np


# =========================================================
# SAFE HELPERS
# =========================================================

def safe_float(value, default=np.nan):

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


def clamp(value, minimum=0.0, maximum=100.0):

    value = safe_float(value, minimum)

    return max(
        minimum,
        min(maximum, value)
    )


# =========================================================
# ATR
# =========================================================

def calculate_atr(
    df,
    period=14
):

    if df is None or df.empty:

        return np.nan

    required = [
        "High",
        "Low",
        "Close"
    ]

    if not all(
        col in df.columns
        for col in required
    ):

        return np.nan

    high = pd.to_numeric(
        df["High"],
        errors="coerce"
    )

    low = pd.to_numeric(
        df["Low"],
        errors="coerce"
    )

    close = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    previous_close = close.shift(1)

    tr1 = high - low

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

    atr = (
        true_range
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

def calculate_atr_percent(
    cmp,
    atr
):

    cmp = safe_float(cmp)
    atr = safe_float(atr)

    if (
        not np.isfinite(cmp)
        or cmp <= 0
        or not np.isfinite(atr)
        or atr < 0
    ):

        return np.nan

    return (
        atr / cmp
    ) * 100


# =========================================================
# VOLATILITY
# =========================================================

def calculate_volatility(
    df,
    period=20
):

    if df is None or df.empty:

        return np.nan

    if "Close" not in df.columns:

        return np.nan

    close = pd.to_numeric(
        df["Close"],
        errors="coerce"
    ).dropna()

    if len(close) < period + 1:

        return np.nan

    returns = (
        close
        .pct_change()
        .dropna()
    )

    if returns.empty:

        return np.nan

    volatility = (
        returns
        .tail(period)
        .std()
        * np.sqrt(252)
        * 100
    )

    return safe_float(
        volatility
    )


# =========================================================
# MAX DRAWDOWN
# =========================================================

def calculate_max_drawdown(
    df
):

    if df is None or df.empty:

        return np.nan

    if "Close" not in df.columns:

        return np.nan

    close = pd.to_numeric(
        df["Close"],
        errors="coerce"
    ).dropna()

    if close.empty:

        return np.nan

    running_high = (
        close
        .cummax()
    )

    drawdown = (
        (
            close
            /
            running_high
        )
        - 1
    ) * 100

    return safe_float(
        drawdown.min()
    )


# =========================================================
# 52 WEEK DATA
# =========================================================

def calculate_52_week_levels(
    df
):

    result = {
        "52W_HIGH": np.nan,
        "52W_LOW": np.nan,
        "DISTANCE_FROM_52W_HIGH_%": np.nan,
        "DISTANCE_FROM_52W_LOW_%": np.nan
    }

    if df is None or df.empty:

        return result

    required = [
        "High",
        "Low",
        "Close"
    ]

    if not all(
        col in df.columns
        for col in required
    ):

        return result

    data = df.tail(252)

    high = pd.to_numeric(
        data["High"],
        errors="coerce"
    ).max()

    low = pd.to_numeric(
        data["Low"],
        errors="coerce"
    ).min()

    cmp = safe_float(
        data["Close"].iloc[-1]
    )

    result[
        "52W_HIGH"
    ] = safe_float(high)

    result[
        "52W_LOW"
    ] = safe_float(low)

    if (
        np.isfinite(cmp)
        and cmp > 0
    ):

        if np.isfinite(high) and high > 0:

            result[
                "DISTANCE_FROM_52W_HIGH_%"
            ] = (
                (cmp - high)
                / high
            ) * 100

        if np.isfinite(low) and low > 0:

            result[
                "DISTANCE_FROM_52W_LOW_%"
            ] = (
                (cmp - low)
                / low
            ) * 100

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

    if df is None or df.empty:

        return result

    if not all(
        col in df.columns
        for col in [
            "High",
            "Low"
        ]
    ):

        return result

    data = df.tail(
        max(lookback, 5)
    )

    result["SUPPORT"] = safe_float(
        pd.to_numeric(
            data["Low"],
            errors="coerce"
        ).min()
    )

    result["RESISTANCE"] = safe_float(
        pd.to_numeric(
            data["High"],
            errors="coerce"
        ).max()
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

    # -----------------------------------------------------
    # ATR RISK
    # -----------------------------------------------------

    atr_percent = safe_float(
        atr_percent,
        0
    )

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

    # -----------------------------------------------------
    # VOLATILITY RISK
    # -----------------------------------------------------

    volatility = safe_float(
        volatility,
        0
    )

    if volatility > 60:

        score -= 25

    elif volatility > 45:

        score -= 18

    elif volatility > 35:

        score -= 12

    elif volatility > 25:

        score -= 6

    # -----------------------------------------------------
    # DRAWDOWN RISK
    # -----------------------------------------------------

    max_drawdown = abs(
        safe_float(
            max_drawdown,
            0
        )
    )

    if max_drawdown > 40:

        score -= 25

    elif max_drawdown > 30:

        score -= 18

    elif max_drawdown > 20:

        score -= 12

    elif max_drawdown > 10:

        score -= 6

    # -----------------------------------------------------
    # 52W HIGH DISTANCE
    # -----------------------------------------------------

    distance_from_high = (
        safe_float(
            distance_from_high,
            0
        )
    )

    if distance_from_high < -30:

        score -= 15

    elif distance_from_high < -20:

        score -= 10

    elif distance_from_high < -10:

        score -= 5

    # -----------------------------------------------------
    # FINAL
    # -----------------------------------------------------

    return round(
        clamp(score),
        1
    )


# =========================================================
# RISK LEVEL
# =========================================================

def get_risk_level(
    risk_score
):

    risk_score = safe_float(
        risk_score,
        0
    )

    if risk_score >= 80:

        return "🟢 LOW"

    elif risk_score >= 65:

        return "🟡 MODERATE"

    elif risk_score >= 45:

        return "🟠 HIGH"

    else:

        return "🔴 VERY HIGH"


# =========================================================
# RISK %
# =========================================================

def calculate_risk_percent(
    risk_score
):

    risk_score = clamp(
        risk_score
    )

    return round(
        100 - risk_score,
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

    if (
        not np.isfinite(cmp)
        or cmp <= 0
    ):

        return np.nan

    # -----------------------------------------------------
    # ATR STOP
    # -----------------------------------------------------

    if (
        np.isfinite(atr)
        and atr > 0
    ):

        atr_stop = (
            cmp
            -
            (2.0 * atr)
        )

    else:

        atr_stop = (
            cmp * 0.95
        )

    # -----------------------------------------------------
    # SUPPORT AWARE STOP
    # -----------------------------------------------------

    support = safe_float(
        support
    )

    if (
        np.isfinite(support)
        and support > 0
        and support < cmp
    ):

        support_stop = (
            support * 0.98
        )

        stop = min(
            atr_stop,
            support_stop
        )

    else:

        stop = atr_stop

    # -----------------------------------------------------
    # SAFETY FLOOR
    # -----------------------------------------------------

    minimum_stop = (
        cmp * 0.80
    )

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

    if (
        not np.isfinite(cmp)
        or cmp <= 0
    ):

        return np.nan

    if (
        np.isfinite(atr)
        and atr > 0
    ):

        target = (
            cmp
            +
            (2.5 * atr)
        )

    else:

        target = (
            cmp * 1.08
        )

    resistance = safe_float(
        resistance
    )

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

    if (
        not np.isfinite(cmp)
        or cmp <= 0
    ):

        return np.nan

    if (
        np.isfinite(atr)
        and atr > 0
    ):

        target = (
            cmp
            +
            (5.0 * atr)
        )

    else:

        target = (
            cmp * 1.20
        )

    high_52 = safe_float(
        high_52
    )

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

    if (
        not np.isfinite(cmp)
        or not np.isfinite(stop_loss)
        or not np.isfinite(target)
    ):

        return np.nan

    risk = (
        cmp
        -
        stop_loss
    )

    reward = (
        target
        -
        cmp
    )

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
    stop_loss = safe_float(
        stop_loss
    )

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

    result = {}

    # -----------------------------------------------------
    # CMP
    # -----------------------------------------------------

    if cmp is None:

        if (
            df is not None
            and not df.empty
            and "Close" in df.columns
        ):

            cmp = safe_float(
                df["Close"].iloc[-1]
            )

        else:

            cmp = np.nan

    cmp = safe_float(cmp)

    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------

    atr = calculate_atr(
        df,
        period=14
    )

    atr_percent = (
        calculate_atr_percent(
            cmp,
            atr
        )
    )

    # -----------------------------------------------------
    # VOLATILITY
    # -----------------------------------------------------

    volatility = (
        calculate_volatility(
            df
        )
    )

    # -----------------------------------------------------
    # DRAWDOWN
    # -----------------------------------------------------

    max_drawdown = (
        calculate_max_drawdown(
            df
        )
    )

    # -----------------------------------------------------
    # 52W
    # -----------------------------------------------------

    week_data = (
        calculate_52_week_levels(
            df
        )
    )

    # -----------------------------------------------------
    # SUPPORT / RESISTANCE
    # -----------------------------------------------------

    sr_data = (
        calculate_support_resistance(
            df
        )
    )

    if support is None:

        support = sr_data[
            "SUPPORT"
        ]

    if resistance is None:

        resistance = sr_data[
            "RESISTANCE"
        ]

    # -----------------------------------------------------
    # RISK SCORE
    # -----------------------------------------------------

    risk_score = (
        calculate_risk_score(
            atr_percent,
            volatility,
            max_drawdown,
            week_data[
                "DISTANCE_FROM_52W_HIGH_%"
            ]
        )
    )

    # -----------------------------------------------------
    # RISK LEVEL
    # -----------------------------------------------------

    risk_level = (
        get_risk_level(
            risk_score
        )
    )

    # -----------------------------------------------------
    # RISK %
    # -----------------------------------------------------

    risk_percent = (
        calculate_risk_percent(
            risk_score
        )
    )

    # -----------------------------------------------------
    # STOP LOSS
    # -----------------------------------------------------

    stop_loss = (
        calculate_stop_loss(
            cmp,
            atr,
            support
        )
    )

    # -----------------------------------------------------
    # SWING TARGET
    # -----------------------------------------------------

    swing_target = (
        calculate_swing_target(
            cmp,
            atr,
            resistance
        )
    )

    # -----------------------------------------------------
    # LONG TARGET
    # -----------------------------------------------------

    long_target = (
        calculate_long_term_target(
            cmp,
            atr,
            week_data[
                "52W_HIGH"
            ]
        )
    )

    # -----------------------------------------------------
    # RISK / REWARD
    # -----------------------------------------------------

    risk_reward = (
        calculate_risk_reward(
            cmp,
            stop_loss,
            swing_target
        )
    )

    # -----------------------------------------------------
    # SAFETY GATE
    # -----------------------------------------------------

    safety_gate = (
        calculate_safety_gate(
            risk_score,
            risk_reward,
            cmp,
            stop_loss
        )
    )

    # =====================================================
    # OUTPUT
    # =====================================================

    result.update(
        {
            "CMP": round(cmp, 2)
            if np.isfinite(cmp)
            else np.nan,

            "ATR": round(atr, 2)
            if np.isfinite(atr)
            else np.nan,

            "ATR_%": round(
                atr_percent,
                2
            )
            if np.isfinite(atr_percent)
            else np.nan,

            "VOLATILITY_%": round(
                volatility,
                2
            )
            if np.isfinite(volatility)
            else np.nan,

            "MAX_DRAWDOWN_%": round(
                max_drawdown,
                2
            )
            if np.isfinite(max_drawdown)
            else np.nan,

            "52W_HIGH": week_data[
                "52W_HIGH"
            ],

            "52W_LOW": week_data[
                "52W_LOW"
            ],

            "DISTANCE_FROM_52W_HIGH_%":
                week_data[
                    "DISTANCE_FROM_52W_HIGH_%"
                ],

            "DISTANCE_FROM_52W_LOW_%":
                week_data[
                    "DISTANCE_FROM_52W_LOW_%"
                ],

            "SUPPORT": support,

            "RESISTANCE": resistance,

            "RISK_SCORE": risk_score,

            "RISK_LEVEL": risk_level,

            "RISK_%": risk_percent,

            "STOP_LOSS": stop_loss,

            "SWING_TARGET": swing_target,

            "LONG_TERM_TARGET": long_target,

            "RISK_REWARD": risk_reward,

            "SAFETY_GATE": safety_gate
        }
    )

    return result


# =========================================================
# ALIAS
# =========================================================

run_risk_engine = calculate_risk
