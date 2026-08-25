# =========================================================
# 🛡️ R.S MASTER STOCK GUIDE V2
# RISK ENGINE — FINAL V2.1
# =========================================================

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

ATR_PERIOD = 14
VOLATILITY_PERIOD = 20
SR_LOOKBACK = 20
WEEK_52_PERIOD = 252

MIN_RISK_SCORE = 0.0
MAX_RISK_SCORE = 100.0


# =========================================================
# SAFE HELPERS
# =========================================================

def safe_float(
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


def clamp(
    value,
    minimum=0.0,
    maximum=100.0
):

    value = safe_float(
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


# =========================================================
# OHLCV PREPARATION
# =========================================================

def prepare_ohlcv(
    df
):

    if (
        df is None
        or not isinstance(
            df,
            pd.DataFrame
        )
        or df.empty
    ):

        return None

    data = df.copy()

    # -----------------------------------------------------
    # MultiIndex
    # -----------------------------------------------------

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        data.columns = [
            col[0]
            for col in data.columns
        ]

    # -----------------------------------------------------
    # Normalize column names
    # -----------------------------------------------------

    rename_map = {}

    for col in data.columns:

        name = str(
            col
        ).strip().lower()

        if name == "high":
            rename_map[col] = "High"

        elif name == "low":
            rename_map[col] = "Low"

        elif name == "close":
            rename_map[col] = "Close"

        elif name == "volume":
            rename_map[col] = "Volume"

    data.rename(
        columns=rename_map,
        inplace=True
    )

    required = [
        "High",
        "Low",
        "Close"
    ]

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

    data.sort_index(
        inplace=True
    )

    return data


# =========================================================
# ATR
# =========================================================

def calculate_atr(
    df,
    period=ATR_PERIOD
):

    data = prepare_ohlcv(
        df
    )

    if data is None:

        return np.nan

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    previous_close = (
        close.shift(1)
    )

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
    ).max(
        axis=1
    )

    # -----------------------------------------------------
    # Wilder-style EWM ATR
    # -----------------------------------------------------

    atr = (
        true_range
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )

    valid = (
        atr.dropna()
    )

    if valid.empty:

        return np.nan

    return safe_float(
        valid.iloc[-1]
    )


# =========================================================
# ATR %
# =========================================================

def calculate_atr_percent(
    cmp,
    atr
):

    cmp = safe_float(
        cmp
    )

    atr = safe_float(
        atr
    )

    if (
        not np.isfinite(cmp)
        or cmp <= 0
        or not np.isfinite(atr)
        or atr < 0
    ):

        return np.nan

    return round(
        (
            atr
            /
            cmp
        ) * 100,
        2
    )


# =========================================================
# VOLATILITY
# =========================================================

def calculate_volatility(
    df,
    period=VOLATILITY_PERIOD
):

    data = prepare_ohlcv(
        df
    )

    if data is None:

        return np.nan

    close = data[
        "Close"
    ]

    returns = (
        close
        .pct_change()
        .dropna()
    )

    if len(
        returns
    ) < period:

        return np.nan

    volatility = (
        returns
        .tail(period)
        .std()
        *
        np.sqrt(252)
        *
        100
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

    data = prepare_ohlcv(
        df
    )

    if data is None:

        return np.nan

    close = data[
        "Close"
    ]

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
# 52 WEEK LEVELS
# =========================================================

def calculate_52_week_levels(
    df
):

    result = {

        "52W_HIGH": np.nan,

        "52W_LOW": np.nan,

        "DISTANCE_FROM_52W_HIGH_%":
            np.nan,

        "DISTANCE_FROM_52W_LOW_%":
            np.nan

    }

    data = prepare_ohlcv(
        df
    )

    if data is None:

        return result

    data = data.tail(
        WEEK_52_PERIOD
    )

    if data.empty:

        return result

    high = safe_float(
        data[
            "High"
        ].max()
    )

    low = safe_float(
        data[
            "Low"
        ].min()
    )

    cmp = safe_float(
        data[
            "Close"
        ].iloc[-1]
    )

    result[
        "52W_HIGH"
    ] = high

    result[
        "52W_LOW"
    ] = low

    if (
        np.isfinite(cmp)
        and cmp > 0
    ):

        if (
            np.isfinite(high)
            and high > 0
        ):

            result[
                "DISTANCE_FROM_52W_HIGH_%"
            ] = round(
                (
                    (
                        cmp
                        -
                        high
                    )
                    /
                    high
                ) * 100,
                2
            )

        if (
            np.isfinite(low)
            and low > 0
        ):

            result[
                "DISTANCE_FROM_52W_LOW_%"
            ] = round(
                (
                    (
                        cmp
                        -
                        low
                    )
                    /
                    low
                ) * 100,
                2
            )

    return result


# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

def calculate_support_resistance(
    df,
    lookback=SR_LOOKBACK
):

    result = {

        "SUPPORT": np.nan,

        "RESISTANCE": np.nan

    }

    data = prepare_ohlcv(
        df
    )

    if data is None:

        return result

    lookback = max(
        int(lookback),
        5
    )

    data = data.tail(
        lookback
    )

    if data.empty:

        return result

    result[
        "SUPPORT"
    ] = safe_float(
        data[
            "Low"
        ].min()
    )

    result[
        "RESISTANCE"
    ] = safe_float(
        data[
            "High"
        ].max()
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

    valid_inputs = 0

    # =====================================================
    # ATR RISK
    # =====================================================

    atr_percent = safe_float(
        atr_percent
    )

    if np.isfinite(
        atr_percent
    ):

        valid_inputs += 1

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

    # =====================================================
    # VOLATILITY RISK
    # =====================================================

    volatility = safe_float(
        volatility
    )

    if np.isfinite(
        volatility
    ):

        valid_inputs += 1

        if volatility > 60:

            score -= 25

        elif volatility > 45:

            score -= 18

        elif volatility > 35:

            score -= 12

        elif volatility > 25:

            score -= 6

    # =====================================================
    # DRAWDOWN RISK
    # =====================================================

    max_drawdown = safe_float(
        max_drawdown
    )

    if np.isfinite(
        max_drawdown
    ):

        valid_inputs += 1

        drawdown = abs(
            max_drawdown
        )

        if drawdown > 40:

            score -= 25

        elif drawdown > 30:

            score -= 18

        elif drawdown > 20:

            score -= 12

        elif drawdown > 10:

            score -= 6

    # =====================================================
    # 52W HIGH DISTANCE
    # =====================================================

    distance_from_high = safe_float(
        distance_from_high
    )

    if np.isfinite(
        distance_from_high
    ):

        valid_inputs += 1

        if distance_from_high < -30:

            score -= 15

        elif distance_from_high < -20:

            score -= 10

        elif distance_from_high < -10:

            score -= 5

    # =====================================================
    # DATA SAFETY
    # =====================================================

    if valid_inputs == 0:

        return 50.0

    return round(
        clamp(
            score,
            MIN_RISK_SCORE,
            MAX_RISK_SCORE
        ),
        1
    )


# =========================================================
# RISK LEVEL
# =========================================================

def get_risk_level(
    risk_score
):

    score = safe_float(
        risk_score,
        50
    )

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

def calculate_risk_percent(
    risk_score
):

    score = clamp(
        risk_score
    )

    return round(
        100 - score,
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

    cmp = safe_float(
        cmp
    )

    atr = safe_float(
        atr
    )

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
            (
                2.0
                *
                atr
            )
        )

    else:

        atr_stop = (
            cmp
            *
            0.95
        )

    # -----------------------------------------------------
    # SUPPORT STOP
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
            support
            *
            0.98
        )

        stop = min(
            atr_stop,
            support_stop
        )

    else:

        stop = atr_stop

    # -----------------------------------------------------
    # HARD FLOOR
    # -----------------------------------------------------

    minimum_stop = (
        cmp
        *
        0.80
    )

    stop = max(
        stop,
        minimum_stop
    )

    # Safety
    if stop >= cmp:

        stop = (
            cmp
            *
            0.95
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

    cmp = safe_float(
        cmp
    )

    atr = safe_float(
        atr
    )

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
            (
                2.5
                *
                atr
            )
        )

    else:

        target = (
            cmp
            *
            1.08
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

    cmp = safe_float(
        cmp
    )

    atr = safe_float(
        atr
    )

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
            (
                5.0
                *
                atr
            )
        )

    else:

        target = (
            cmp
            *
            1.20
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

    cmp = safe_float(
        cmp
    )

    stop_loss = safe_float(
        stop_loss
    )

    target = safe_float(
        target
    )

    if not all(
        np.isfinite(x)
        for x in [
            cmp,
            stop_loss,
            target
        ]
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

    if (
        risk <= 0
        or reward <= 0
    ):

        return np.nan

    return round(
        reward
        /
        risk,
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
        50
    )

    risk_reward = safe_float(
        risk_reward
    )

    cmp = safe_float(
        cmp
    )

    stop_loss = safe_float(
        stop_loss
    )

    # =====================================================
    # HARD BLOCK
    # =====================================================

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

    # =====================================================
    # WARNING
    # =====================================================

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

    # =====================================================
    # DATA CHECK
    # =====================================================

    data = prepare_ohlcv(
        df
    )

    if data is None:

        return {

            "CMP": np.nan,

            "ATR": np.nan,

            "ATR_%": np.nan,

            "VOLATILITY_%": np.nan,

            "MAX_DRAWDOWN_%": np.nan,

            "52W_HIGH": np.nan,

            "52W_LOW": np.nan,

            "DISTANCE_FROM_52W_HIGH_%":
                np.nan,

            "DISTANCE_FROM_52W_LOW_%":
                np.nan,

            "SUPPORT": np.nan,

            "RESISTANCE": np.nan,

            "RISK_SCORE": 0.0,

            "RISK_LEVEL":
                "🔴 VERY HIGH",

            "RISK_%": 100.0,

            "STOP_LOSS": np.nan,

            "SWING_TARGET": np.nan,

            "LONG_TERM_TARGET":
                np.nan,

            "RISK_REWARD": np.nan,

            "SAFETY_GATE":
                "🔴 BLOCKED",

            "STATUS":
                "NO_DATA"

        }

    # =====================================================
    # CMP
    # =====================================================

    if cmp is None:

        cmp = safe_float(
            data[
                "Close"
            ].iloc[-1]
        )

    else:

        cmp = safe_float(
            cmp
        )

    # =====================================================
    # ATR
    # =====================================================

    atr = calculate_atr(
        data,
        ATR_PERIOD
    )

    atr_percent = (
        calculate_atr_percent(
            cmp,
            atr
        )
    )

    # =====================================================
    # VOLATILITY
    # =====================================================

    volatility = (
        calculate_volatility(
            data,
            VOLATILITY_PERIOD
        )
    )

    # =====================================================
    # DRAWDOWN
    # =====================================================

    max_drawdown = (
        calculate_max_drawdown(
            data
        )
    )

    # =====================================================
    # 52W
    # =====================================================

    week_data = (
        calculate_52_week_levels(
            data
        )
    )

    # =====================================================
    # SUPPORT / RESISTANCE
    # =====================================================

    sr_data = (
        calculate_support_resistance(
            data,
            SR_LOOKBACK
        )
    )

    if support is None:

        support = sr_data[
            "SUPPORT"
        ]

    else:

        support = safe_float(
            support
        )

    if resistance is None:

        resistance = sr_data[
            "RESISTANCE"
        ]

    else:

        resistance = safe_float(
            resistance
        )

    # =====================================================
    # RISK SCORE
    # =====================================================

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

    # =====================================================
    # RISK LEVEL
    # =====================================================

    risk_level = (
        get_risk_level(
            risk_score
        )
    )

    # =====================================================
    # RISK %
    # =====================================================

    risk_percent = (
        calculate_risk_percent(
            risk_score
        )
    )

    # =====================================================
    # STOP LOSS
    # =====================================================

    stop_loss = (
        calculate_stop_loss(
            cmp,
            atr,
            support
        )
    )

    # =====================================================
    # TARGETS
    # =====================================================

    swing_target = (
        calculate_swing_target(
            cmp,
            atr,
            resistance
        )
    )

    long_target = (
        calculate_long_term_target(
            cmp,
            atr,
            week_data[
                "52W_HIGH"
            ]
        )
    )

    # =====================================================
    # RISK / REWARD
    # =====================================================

    risk_reward = (
        calculate_risk_reward(
            cmp,
            stop_loss,
            swing_target
        )
    )

    # =====================================================
    # SAFETY GATE
    # =====================================================

    safety_gate = (
        calculate_safety_gate(
            risk_score,
            risk_reward,
            cmp,
            stop_loss
        )
    )

    # =====================================================
    # STATUS
    # =====================================================

    status = "FRESH"

    if not np.isfinite(
        atr
    ):

        status = "LIMITED"

    # =====================================================
    # FINAL RESULT
    # =====================================================

    return {

        "CMP":
            round(cmp, 2)
            if np.isfinite(cmp)
            else np.nan,

        "ATR":
            round(atr, 2)
            if np.isfinite(atr)
            else np.nan,

        "ATR_%":
            round(atr_percent, 2)
            if np.isfinite(atr_percent)
            else np.nan,

        "VOLATILITY_%":
            round(volatility, 2)
            if np.isfinite(volatility)
            else np.nan,

        "MAX_DRAWDOWN_%":
            round(max_drawdown, 2)
            if np.isfinite(max_drawdown)
            else np.nan,

        "52W_HIGH":
            round(
                week_data[
                    "52W_HIGH"
                ],
                2
            )
            if np.isfinite(
                week_data[
                    "52W_HIGH"
                ]
            )
            else np.nan,

        "52W_LOW":
            round(
                week_data[
                    "52W_LOW"
                ],
                2
            )
            if np.isfinite(
                week_data[
                    "52W_LOW"
                ]
            )
            else np.nan,

        "DISTANCE_FROM_52W_HIGH_%":
            week_data[
                "DISTANCE_FROM_52W_HIGH_%"
            ],

        "DISTANCE_FROM_52W_LOW_%":
            week_data[
                "DISTANCE_FROM_52W_LOW_%"
            ],

        "SUPPORT":
            round(
                support,
                2
            )
            if np.isfinite(support)
            else np.nan,

        "RESISTANCE":
            round(
                resistance,
                2
            )
            if np.isfinite(resistance)
            else np.nan,

        "RISK_SCORE":
            risk_score,

        "RISK_LEVEL":
            risk_level,

        "RISK_%":
            risk_percent,

        "STOP_LOSS":
            stop_loss,

        "SWING_TARGET":
            swing_target,

        "LONG_TERM_TARGET":
            long_target,

        "RISK_REWARD":
            risk_reward,

        "SAFETY_GATE":
            safety_gate,

        "STATUS":
            status

    }


# =========================================================
# ALIAS
# =========================================================

run_risk_engine = (
    calculate_risk
)


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    print(
        "🛡️ R.S MASTER STOCK GUIDE V2"
    )

    print(
        "RISK ENGINE V2.1 READY"
    )
