# =====================================================
# V3 E.M.S. — INDEPENDENT EXIT MANAGEMENT SYSTEM
# =====================================================

ems_input = {

    # -------------------------------------------------
    # MASTER SCORE — CONTEXT ONLY
    # -------------------------------------------------
    "master_score": master_score,

    # -------------------------------------------------
    # STRUCTURAL EVIDENCE
    # -------------------------------------------------

    "trend_breakdown": (
        technical_score < 35
    ),

    "momentum_breakdown": (
        str(
            result.get(
                "MOMENTUM_LEVEL",
                ""
            )
        ).upper()
        in {
            "BEARISH",
            "WEAK",
            "NEGATIVE"
        }
    ),

    # -------------------------------------------------
    # SUPPORT
    # -------------------------------------------------
    # Actual V2 support-breakdown mapping
    # available નથી → DO NOT GUESS.

    "support_breakdown": None,

    # -------------------------------------------------
    # VOLUME
    # -------------------------------------------------

    "volume_confirmation": (
        str(
            result.get(
                "VOLUME_BREAKOUT",
                ""
            )
        ).upper()
        in {
            "YES",
            "TRUE",
            "BREAKOUT"
        }
    ),

    # -------------------------------------------------
    # RELATIVE STRENGTH
    # -------------------------------------------------
    # Actual benchmark comparison available નથી.

    "relative_strength_breakdown": None,

    # -------------------------------------------------
    # RISK DETERIORATION
    # -------------------------------------------------

    "risk_deterioration": (
        str(
            result.get(
                "RISK_LEVEL",
                ""
            )
        ).upper()
        in {
            "HIGH",
            "VERY HIGH",
            "CRITICAL",
            "SEVERE",
            "EXTREME"
        }
    ),

    # -------------------------------------------------
    # ABOVE EXIT PRICE
    # -------------------------------------------------
    # Actual V2 Exit Price mapping નથી.

    "above_exit_price": None,

    # -------------------------------------------------
    # ATH PROFIT
    # -------------------------------------------------
    # 52W HIGH ≠ ATH.

    "ath_profit": None,

    # -------------------------------------------------
    # OUTPERFORMANCE
    # -------------------------------------------------
    # NIFTY benchmark comparison mapping pending.

    "outperformance": None,

    # -------------------------------------------------
    # REFERENCE CALIBRATION
    # -------------------------------------------------

    "reference_match": None,
}


# =====================================================
# RUN EMS
# =====================================================

ems_result = evaluate_ems(
    ems_input
)


ems_ui = ems_display(
    ems_result
)


# =====================================================
# V3 FINAL EXIT STATUS
# =====================================================

ems_status = str(
    ems_result.get(
        "status",
        "DATA LIMITED"
    )
).upper()


ems_reason = ems_result.get(
    "reason",
    "EMS data unavailable"
)


if ems_status == "EXIT":

    exit_signal = "EXIT"

    exit_reason = ems_reason


elif ems_status == "REDUCE":

    exit_signal = "REDUCE"

    exit_reason = ems_reason


elif ems_status == "WATCH":

    exit_signal = "HOLD"

    exit_reason = (
        "EMS WATCH — confirmation required"
    )


elif ems_status == "SAFE":

    exit_signal = "HOLD"

    exit_reason = (
        "EMS SAFE — no confirmed exit"
    )


else:

    exit_signal = "HOLD"

    exit_reason = (
        "EMS DATA LIMITED — "
        "no exit generated"
    )
