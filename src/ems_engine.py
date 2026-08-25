"""
R.S MASTER STOCK GUIDE V3
E.M.S. — Exit Management System

V3 upgrade layer.

IMPORTANT:
- V2 Master Score is preserved.
- Master Score alone can NEVER generate EXIT.
- EMS uses independent evidence.
- CEMPRO and SHRIRAMFIN are reference/calibration cases.
- Missing evidence => DATA LIMITED.
- No fabricated score or EXIT.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import math


# =========================================================
# EMS CONFIG
# =========================================================

EMS_VERSION = "V3.0"
EMS_NAME = "E.M.S."
EMS_FULL_NAME = "Exit Management System"

REFERENCE_CASES = (
    "CEMPRO",
    "SHRIRAMFIN",
)


# =========================================================
# EMS RESULT
# =========================================================

@dataclass
class EMSResult:

    status: str
    confidence: Optional[float]
    severity: str
    reason: str

    confirmed_factors: int
    warning_factors: int

    data_limited: bool

    reference_match: str

    version: str = EMS_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================
# SAFE CONVERSION
# =========================================================

def _num(value: Any) -> Optional[float]:

    try:

        if value is None:
            return None

        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):

        return None


def _bool(value: Any) -> Optional[bool]:

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, str):

        value = value.strip().lower()

        if value in {
            "true",
            "yes",
            "y",
            "1",
            "confirmed",
            "broken",
            "negative",
        }:
            return True

        if value in {
            "false",
            "no",
            "n",
            "0",
            "safe",
            "positive",
        }:
            return False

    return None


# =========================================================
# EMS ENGINE
# =========================================================

def evaluate_ems(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate E.M.S.

    Expected optional inputs:

        trend_breakdown
        momentum_breakdown
        support_breakdown
        volume_confirmation
        relative_strength_breakdown
        risk_deterioration

        above_exit_price
        ath_profit
        outperformance

        master_score

        data_fresh
        reference_match

    IMPORTANT:
    Master Score is contextual only.
    It is NOT an independent EXIT trigger.
    """

    evidence_keys = (

        "trend_breakdown",

        "momentum_breakdown",

        "support_breakdown",

        "volume_confirmation",

        "relative_strength_breakdown",

        "risk_deterioration",

        "above_exit_price",

        "ath_profit",

        "outperformance",

    )

    # -----------------------------------------------------
    # DATA AVAILABILITY
    # -----------------------------------------------------

    available = sum(
        _bool(data.get(key)) is not None
        for key in evidence_keys
    )

    # No usable EMS evidence
    if available == 0:

        return EMSResult(

            status="DATA LIMITED",

            confidence=None,

            severity="UNKNOWN",

            reason=(
                "EMS evidence is unavailable; "
                "no EXIT is generated."
            ),

            confirmed_factors=0,

            warning_factors=0,

            data_limited=True,

            reference_match=str(
                data.get("reference_match") or "NONE"
            ),

        ).to_dict()

    # -----------------------------------------------------
    # FACTOR COUNTERS
    # -----------------------------------------------------

    confirmed = 0

    warnings = 0

    # -----------------------------------------------------
    # STRUCTURAL DETERIORATION
    # -----------------------------------------------------

    structural_factors = (

        "trend_breakdown",

        "momentum_breakdown",

        "support_breakdown",

        "relative_strength_breakdown",

        "risk_deterioration",

    )

    for key in structural_factors:

        flag = _bool(data.get(key))

        if flag is True:

            confirmed += 1

        elif flag is False:

            warnings += 1

    # -----------------------------------------------------
    # CONFIRMATION / CONTEXT FACTORS
    # -----------------------------------------------------

    confirmation_factors = (

        "volume_confirmation",

        "above_exit_price",

        "ath_profit",

        "outperformance",

    )

    for key in confirmation_factors:

        flag = _bool(data.get(key))

        if flag is True:

            confirmed += 1

    # -----------------------------------------------------
    # REFERENCE CASE
    # -----------------------------------------------------

    reference = str(
        data.get("reference_match") or "NONE"
    ).upper()

    if reference not in REFERENCE_CASES:

        reference = "NONE"

    # -----------------------------------------------------
    # MASTER SCORE
    # -----------------------------------------------------
    #
    # IMPORTANT:
    # Master Score is ONLY contextual.
    #
    # It cannot independently trigger EXIT.
    # -----------------------------------------------------

    master_score = _num(
        data.get("master_score")
    )

    score_context = ""

    if master_score is not None:

        score_context = (

            f" Master Score context="
            f"{master_score:.1f}; "
            "not used as a standalone EXIT trigger."
        )

    # -----------------------------------------------------
    # EMS CONFIDENCE
    # -----------------------------------------------------

    confidence = round(

        min(
            100.0,

            (
                confirmed
                / max(available, 1)
            )
            * 100.0
        ),

        1

    )

    # -----------------------------------------------------
    # FINAL EMS STATUS
    # -----------------------------------------------------

    if confirmed >= 5:

        status = "EXIT"

        severity = "HIGH"

        reason = (
            "Multiple independent deterioration "
            "factors confirmed."
        )

    elif confirmed >= 3:

        status = "REDUCE"

        severity = "ELEVATED"

        reason = (
            "Meaningful deterioration detected; "
            "position risk should be reviewed."
        )

    elif confirmed >= 2:

        status = "WATCH"

        severity = "MODERATE"

        reason = (
            "Early deterioration detected; "
            "await confirmation."
        )

    else:

        status = "SAFE"

        severity = "LOW"

        reason = (
            "No sufficient independent "
            "EXIT confirmation."
        )

    # -----------------------------------------------------
    # FINAL RESULT
    # -----------------------------------------------------

    return EMSResult(

        status=status,

        confidence=confidence,

        severity=severity,

        reason=reason + score_context,

        confirmed_factors=confirmed,

        warning_factors=warnings,

        data_limited=(
            available < 4
        ),

        reference_match=reference,

    ).to_dict()


# =========================================================
# EMS DISPLAY HELPER
# =========================================================

def ems_display(
    result: Dict[str, Any]
) -> Dict[str, str]:

    status = str(
        result.get(
            "status",
            "DATA LIMITED"
        )
    )

    styles = {

        "EXIT": (
            "🔴",
            "#ff304f"
        ),

        "REDUCE": (
            "🟠",
            "#ff9800"
        ),

        "WATCH": (
            "🟡",
            "#ffd740"
        ),

        "SAFE": (
            "🟢",
            "#00e676"
        ),

        "DATA LIMITED": (
            "🩵",
            "#00cfff"
        ),

    }

    symbol, color = styles.get(

        status,

        (
            "🩵",
            "#00cfff"
        )

    )

    return {

        "label": (
            f"{symbol} E.M.S. {status}"
        ),

        "color": color,

    }


# =========================================================
# EMS REFERENCE CASE HELPER
# =========================================================

def get_reference_cases():

    return list(
        REFERENCE_CASES
    )


# =========================================================
# EMS SAFE STATUS
# =========================================================

def is_exit_confirmed(
    result: Dict[str, Any]
) -> bool:

    return (
        str(
            result.get("status", "")
        ).upper()
        == "EXIT"
    )


def is_data_limited(
    result: Dict[str, Any]
) -> bool:

    return bool(
        result.get(
            "data_limited",
            True
        )
    )


# =========================================================
# END OF EMS ENGINE
# =========================================================
