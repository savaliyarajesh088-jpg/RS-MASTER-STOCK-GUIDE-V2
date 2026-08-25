import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

from plotly.subplots import make_subplots

from src.nse_data import fetch_nse_data
from src.fundamental_engine import fetch_fundamental_data
from src.ems_engine import (
    evaluate_ems,
    ems_display,
    is_exit_confirmed,
    is_data_limited,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="R.S MASTER STOCK GUIDE V2",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# MOBILE COLOR UI
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #050505;
    color: #ffffff;
}

.block-container {
    max-width: 1250px;
    padding: 0.35rem 0.35rem 1.2rem 0.35rem;
}

.main-title {
    text-align: center;
    font-size: 1.45rem;
    font-weight: 950;
    margin: 0;
}

.sub-title {
    text-align: center;
    font-size: 0.70rem;
    opacity: 0.65;
    margin-bottom: 0.45rem;
}

.stock-card {
    background: linear-gradient(
        145deg,
        #111111,
        #080808
    );
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 18px;
    padding: 12px;
    margin-bottom: 14px;
}

.section-title {
    font-size: 0.95rem;
    font-weight: 900;
    margin-top: 13px;
    margin-bottom: 7px;
}

.score-card {
    background: #111111;
    border-radius: 16px;
    padding: 13px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.12);
}

.score-title {
    font-size: 0.68rem;
    font-weight: 800;
    opacity: 0.70;
}

.score-number {
    font-size: 1.65rem;
    font-weight: 950;
}

.score-bar {
    width: 100%;
    height: 8px;
    background: #252525;
    border-radius: 20px;
    overflow: hidden;
    margin-top: 8px;
}

.score-fill {
    height: 100%;
    border-radius: 20px;
}

.signal-card {
    border-radius: 15px;
    padding: 12px;
    text-align: center;
    font-weight: 950;
    font-size: 1.02rem;
    margin: 8px 0;
}

.signal-buy {
    background: linear-gradient(
        135deg,
        rgba(0,220,100,.22),
        rgba(0,150,70,.08)
    );
    border: 1px solid #00dc64;
    color: #00ff77;
}

.signal-hold {
    background: linear-gradient(
        135deg,
        rgba(0,170,255,.20),
        rgba(0,100,180,.08)
    );
    border: 1px solid #00aaff;
    color: #45c5ff;
}

.signal-wait {
    background: linear-gradient(
        135deg,
        rgba(255,200,0,.20),
        rgba(180,130,0,.08)
    );
    border: 1px solid #ffc400;
    color: #ffd633;
}

.signal-reduce {
    background: linear-gradient(
        135deg,
        rgba(255,152,0,.22),
        rgba(180,80,0,.08)
    );
    border: 1px solid #ff9800;
    color: #ffb13b;
}

.signal-exit {
    background: linear-gradient(
        135deg,
        rgba(255,48,79,.24),
        rgba(160,0,30,.08)
    );
    border: 1px solid #ff304f;
    color: #ff5570;
}

.data-card {
    background: #101010;
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 12px;
    padding: 8px;
    margin-bottom: 7px;
}

.data-label {
    font-size: 0.62rem;
    opacity: 0.62;
    font-weight: 800;
}

.data-value {
    font-size: 0.92rem;
    font-weight: 900;
}

.bull {
    color: #00e676;
}

.bear {
    color: #ff3d5a;
}

.neutral {
    color: #ffd740;
}

.white {
    color: #ffffff;
}

.target-green {
    border: 1px solid rgba(0,220,100,.35);
}

.target-blue {
    border: 1px solid rgba(0,170,255,.35);
}

.target-red {
    border: 1px solid rgba(255,48,79,.35);
}

.summary-card {
    background: #101010;
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 15px;
    padding: 12px;
    text-align: center;
}

.summary-number {
    font-size: 1.35rem;
    font-weight: 950;
}

.mobile-table {
    font-size: 0.75rem;
}

@media (max-width: 640px) {

    .block-container {
        padding-left: 0.18rem;
        padding-right: 0.18rem;
    }

    .main-title {
        font-size: 1.12rem;
    }

    .sub-title {
        font-size: 0.57rem;
    }

    .stock-card {
        padding: 9px;
        border-radius: 15px;
    }

    .score-number {
        font-size: 1.35rem;
    }

    .section-title {
        font-size: 0.84rem;
    }

    .signal-card {
        font-size: 0.90rem;
        padding: 10px;
    }

    .data-value {
        font-size: 0.82rem;
    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================

def safe_float(value, default=0.0):

    try:

        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def money(value):

    try:

        if value is None or pd.isna(value):
            return "—"

        return f"₹{float(value):,.2f}"

    except Exception:

        return "—"


def pct(value):

    try:

        if value is None or pd.isna(value):
            return "—"

        return f"{float(value):.2f}%"

    except Exception:

        return "—"


def number(value, decimals=2):

    try:

        if value is None or pd.isna(value):
            return "—"

        return f"{float(value):,.{decimals}f}"

    except Exception:

        return "—"


def integer_number(value):

    try:

        if value is None or pd.isna(value):
            return "—"

        return f"{float(value):,.0f}"

    except Exception:

        return "—"


def display_value(value):

    try:

        if value is None or pd.isna(value):
            return "—"

    except Exception:
        pass

    return str(value)


def clean_symbol(symbol):

    symbol = str(symbol).strip().upper()

    if symbol.endswith(".NS"):
        symbol = symbol[:-3]

    return symbol


# =========================================================
# SIGNAL HELPERS
# =========================================================

def decision_style(decision):

    if decision == "BUY":
        return (
            "🟢🚀 BUY / વધારો",
            "signal-buy",
            "#00e676"
        )

    if decision == "HOLD":
        return (
            "🔵🛡️ HOLD / જાળવો",
            "signal-hold",
            "#00aaff"
        )

    if decision == "WAIT":
        return (
            "🟡⏳ WAIT / રાહ જુઓ",
            "signal-wait",
            "#ffd740"
        )

    if decision == "SELL":
        return (
            "🟠⚠️ REDUCE / ઘટાડો",
            "signal-reduce",
            "#ff9800"
        )

    return (
        "🔴🚪 EXIT / બહાર નીકળો",
        "signal-exit",
        "#ff304f"
    )


def zone_style(score):

    if score >= 75:
        return "🐂🟢 BULL", "bull"

    if score >= 55:
        return "🟡↔️ NEUTRAL", "neutral"

    return "🐻🔴 BEAR", "bear"


def technical_zone_style(zone):

    z = str(zone).upper()

    if "STRONG" in z or "POSITIVE" in z or "BULL" in z:
        return "🟢📈 " + z

    if "AVERAGE" in z or "MIXED" in z or "NEUTRAL" in z:
        return "🟡↔️ " + z

    if (
        "WEAK" in z
        or "BEAR" in z
        or "POOR" in z
    ):
        return "🔴📉 " + z

    return "⚪ " + z


def alignment_style(value):

    text = str(value).upper()

    if "BULLISH" in text:
        return "🟢📈 BULLISH"

    if "BEARISH" in text:
        return "🔴📉 BEARISH"

    if "MIXED" in text:
        return "🟡↔️ MIXED"

    return "⚪ —"


def trend_style(value):

    text = str(value).upper()

    if "BULL" in text:
        return "🟢🐂 BULLISH"

    if "BEAR" in text:
        return "🔴🐻 BEARISH"

    return "🟡↔️ MIXED"


def breakout_style(value):

    text = str(value).upper()

    if (
        "YES" in text
        or "BREAKOUT" in text
        or text == "TRUE"
    ):
        return "🟢🚀 YES"

    return "🔴 NO"


def valuation_style(value):

    text = str(value).upper()

    if "CHEAP" in text or "UNDER" in text:
        return "🟢💰 " + text

    if "FAIR" in text:
        return "🟡⚖️ " + text

    if "EXPENSIVE" in text or "HIGH" in text:
        return "🔴💸 " + text

    if "LIMITED" in text:
        return "⚪ DATA_LIMITED"

    return "⚪ " + text


# =========================================================
# ATR
# =========================================================

def calculate_atr(df, period=14):

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

    return true_range.rolling(period).mean()


# =========================================================
# ADVANCED PRICE CHART
# =========================================================

def build_price_chart(
    symbol,
    stop_loss,
    swing_target,
    long_target
):

    st.markdown(
        '<div class="section-title">'
        '📊 ADVANCED PRICE CHART'
        '</div>',
        unsafe_allow_html=True
    )

    ticker_symbol = clean_symbol(symbol) + ".NS"

    try:

        data = yf.download(
            ticker_symbol,
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if data.empty:

            st.warning(
                "📊 Historical chart data ઉપલબ્ધ નથી."
            )

            return

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):

            data.columns = [
                col[0]
                for col in data.columns
            ]

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        missing = [
            c
            for c in required
            if c not in data.columns
        ]

        if missing:

            st.warning(
                "Chart data missing: "
                + ", ".join(missing)
            )

            return

        data = data[required].copy()

        for col in required:

            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

        data.dropna(inplace=True)

        if len(data) < 50:

            st.warning(
                "Chart માટે historical data ઓછો છે."
            )

            return

        close = data["Close"]

        # =================================================
        # EMA
        # =================================================

        for period in [10, 20, 50, 100, 200]:

            data[f"EMA{period}"] = (
                close
                .ewm(
                    span=period,
                    adjust=False
                )
                .mean()
            )

        # =================================================
        # RSI
        # =================================================

        delta = close.diff()

        gain = (
            delta.clip(lower=0)
            .rolling(14)
            .mean()
        )

        loss = (
            -delta.clip(upper=0)
            .rolling(14)
            .mean()
        )

        rs = gain / loss.replace(
            0,
            np.nan
        )

        data["RSI"] = (
            100 -
            (
                100 /
                (1 + rs)
            )
        )

        # =================================================
        # MACD
        # =================================================

        ema12 = (
            close
            .ewm(
                span=12,
                adjust=False
            )
            .mean()
        )

        ema26 = (
            close
            .ewm(
                span=26,
                adjust=False
            )
            .mean()
        )

        data["MACD"] = ema12 - ema26

        data["MACD_SIGNAL"] = (
            data["MACD"]
            .ewm(
                span=9,
                adjust=False
            )
            .mean()
        )

        data["MACD_HIST"] = (
            data["MACD"]
            -
            data["MACD_SIGNAL"]
        )

        # =================================================
        # VOLUME
        # =================================================

        data["VOL20"] = (
            data["Volume"]
            .rolling(20)
            .mean()
        )

        data["VOL_RATIO"] = (
            data["Volume"]
            /
            data["VOL20"]
        )

        # =================================================
        # BREAKOUT
        # =================================================

        data["PREVIOUS_20_HIGH"] = (
            data["High"]
            .rolling(20)
            .max()
            .shift(1)
        )

        data["BREAKOUT"] = (
            data["Close"]
            >
            data["PREVIOUS_20_HIGH"]
        )

        # =================================================
        # CURRENT VALUES
        # =================================================

        cmp = float(close.iloc[-1])

        ema10 = safe_float(
            data["EMA10"].iloc[-1]
        )

        ema20 = safe_float(
            data["EMA20"].iloc[-1]
        )

        rsi = safe_float(
            data["RSI"].iloc[-1],
            50
        )

        macd = safe_float(
            data["MACD"].iloc[-1]
        )

        macd_signal = safe_float(
            data["MACD_SIGNAL"].iloc[-1]
        )

        vol_ratio = safe_float(
            data["VOL_RATIO"].iloc[-1]
        )

        breakout = bool(
            data["BREAKOUT"].iloc[-1]
        )

        momentum_ready = (
            ema10 > ema20
            and rsi >= 55
            and macd > macd_signal
            and vol_ratio >= 1.2
        )

        # =================================================
        # 52 WEEK
        # =================================================

        one_year = data.tail(252)

        high_52 = float(
            one_year["High"].max()
        )

        low_52 = float(
            one_year["Low"].min()
        )

        # =================================================
        # ATR
        # =================================================

        atr_series = (
            calculate_atr(data)
            .dropna()
        )

        if atr_series.empty:
            atr = max(cmp * 0.02, 1)
        else:
            atr = float(
                atr_series.iloc[-1]
            )

        stop_loss = safe_float(
            stop_loss,
            cmp - 2 * atr
        )

        swing_target = safe_float(
            swing_target,
            cmp + 2 * atr
        )

        long_target = safe_float(
            long_target,
            cmp + 5 * atr
        )

        # =================================================
        # SUBTITLE
        # =================================================

        st.caption(
            "🤏 Pinch zoom • Drag • Double tap reset • "
            "🔍 1M / 3M / 6M / 1Y / ALL"
        )

        # =================================================
        # FIGURE
        # =================================================

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.025,
            row_heights=[0.78, 0.22]
        )

        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="PRICE"
            ),
            row=1,
            col=1
        )

        # =================================================
        # EMA
        # =================================================

        for column, name, width in [
            ("EMA10", "EMA 10", 1.1),
            ("EMA20", "EMA 20", 1.1),
            ("EMA50", "EMA 50", 1.4),
            ("EMA100", "EMA 100", 1.4),
            ("EMA200", "EMA 200", 1.8)
        ]:

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[column],
                    name=name,
                    mode="lines",
                    line=dict(width=width)
                ),
                row=1,
                col=1
            )

        # =================================================
        # VOLUME
        # =================================================

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["Volume"],
                name="Volume",
                opacity=0.55
            ),
            row=2,
            col=1
        )

        # =================================================
        # BREAKOUT
        # =================================================

        breakout_data = data[
            data["BREAKOUT"]
        ]

        if not breakout_data.empty:

            fig.add_trace(
                go.Scatter(
                    x=breakout_data.index,
                    y=breakout_data["High"],
                    mode="markers",
                    name="🚀 BREAKOUT",
                    marker=dict(
                        size=9,
                        symbol="triangle-up"
                    )
                ),
                row=1,
                col=1
            )

        # =================================================
        # MOMENTUM
        # =================================================

        if momentum_ready:

            fig.add_trace(
                go.Scatter(
                    x=[data.index[-1]],
                    y=[cmp],
                    mode="markers+text",
                    text=["⚡ MOMENTUM"],
                    textposition="top center",
                    name="⚡ MOMENTUM"
                ),
                row=1,
                col=1
            )

        # =================================================
        # PRICE LEVELS
        # =================================================

        levels = [
            (
                cmp,
                f"CMP ₹{cmp:,.2f}",
                "dot"
            ),
            (
                stop_loss,
                f"🛑 SL ₹{stop_loss:,.2f}",
                "dash"
            ),
            (
                swing_target,
                f"🎯 SWING ₹{swing_target:,.2f}",
                "dot"
            ),
            (
                long_target,
                f"🚀 LONG ₹{long_target:,.2f}",
                "dot"
            ),
            (
                high_52,
                f"52W HIGH ₹{high_52:,.2f}",
                "dashdot"
            ),
            (
                low_52,
                f"52W LOW ₹{low_52:,.2f}",
                "dashdot"
            )
        ]

        for level, label, dash in levels:

            fig.add_hline(
                y=level,
                row=1,
                col=1,
                line_dash=dash,
                line_width=1,
                annotation_text=label,
                annotation_position="top right"
            )

        # =================================================
        # RANGE BUTTONS
        # =================================================

        buttons = [

            dict(
                count=1,
                label="1M",
                step="month",
                stepmode="backward"
            ),

            dict(
                count=3,
                label="3M",
                step="month",
                stepmode="backward"
            ),

            dict(
                count=6,
                label="6M",
                step="month",
                stepmode="backward"
            ),

            dict(
                count=1,
                label="1Y",
                step="year",
                stepmode="backward"
            ),

            dict(
                step="all",
                label="ALL"
            )
        ]

        # =================================================
        # LAYOUT
        # =================================================

        fig.update_layout(
            height=650,
            template="plotly_dark",
            paper_bgcolor="#050505",
            plot_bgcolor="#050505",
            hovermode="x unified",
            dragmode="pan",
            margin=dict(
                l=5,
                r=5,
                t=40,
                b=5
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.01,
                xanchor="center",
                x=0.5
            ),
            xaxis=dict(
                type="date",
                fixedrange=False,
                rangeslider=dict(
                    visible=True,
                    thickness=0.06
                ),
                rangeselector=dict(
                    buttons=buttons
                )
            ),
            xaxis2=dict(
                type="date",
                fixedrange=False
            ),
            yaxis=dict(
                fixedrange=False,
                autorange=True
            ),
            yaxis2=dict(
                fixedrange=False,
                autorange=True
            )
        )

        config = {
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "doubleClick": "reset",
            "displayModeBar": True,
            "modeBarButtonsToRemove": [
                "lasso2d",
                "select2d"
            ]
        }

        st.plotly_chart(
            fig,
            width="stretch",
            config=config,
            key=f"chart_{clean_symbol(symbol)}"
        )

        # =================================================
        # CHART SIGNAL
        # =================================================

        if breakout:

            st.success(
                "🚀 BREAKOUT CONFIRMED"
            )

        elif momentum_ready:

            st.info(
                "⚡ MOMENTUM BUILDING"
            )

        else:

            st.caption(
                "⏳ Breakout / momentum confirmation pending"
            )

        st.caption(
            f"📍 CMP {money(cmp)} | "
            f"🛑 SL {money(stop_loss)} | "
            f"🎯 Swing {money(swing_target)} | "
            f"🚀 Long {money(long_target)}"
        )

    except Exception as error:

        st.error(
            f"📊 Chart Error: "
            f"{type(error).__name__}: {error}"
        )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '📈 R.S MASTER STOCK GUIDE V2'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'NSE • Technical • Fundamental • Momentum • '
    'Breakout • Exit Matra • Risk • Target'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# PORTFOLIO
# =========================================================

st.markdown(
    '<div class="section-title">📁 પોર્ટફોલિયો</div>',
    unsafe_allow_html=True
)

try:

    portfolio = pd.read_csv(
        "portfolio.csv"
    )

except Exception as error:

    st.error(
        f"Portfolio loading error: {error}"
    )

    st.stop()


if portfolio.empty:

    st.warning(
        "portfolio.csv empty છે."
    )

    st.stop()


if "SYMBOL" not in portfolio.columns:

    st.error(
        "portfolio.csv માં SYMBOL column જરૂરી છે."
    )

    st.stop()


portfolio["SYMBOL"] = (
    portfolio["SYMBOL"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# =========================================================
# PROCESSING
# =========================================================

all_scores = []


for raw_symbol in portfolio["SYMBOL"]:

    symbol = clean_symbol(raw_symbol)

    # =====================================================
    # NSE DATA
    # =====================================================

    try:

        result = fetch_nse_data(symbol)

    except Exception as error:

        st.error(
            f"{symbol}: NSE error — {error}"
        )

        continue


    if not isinstance(result, dict):

        st.error(
            f"{symbol}: NSE response invalid."
        )

        continue


    # =====================================================
    # FUNDAMENTAL
    # =====================================================

    try:

        fundamental = fetch_fundamental_data(
            symbol
        )

    except Exception as error:

        fundamental = {
            "FUNDAMENTAL_SCORE": 0,
            "FUNDAMENTAL_ZONE": "DATA ERROR",
            "DATA_QUALITY_%": 0,
            "ERROR": str(error)
        }


    if not isinstance(
        fundamental,
        dict
    ):

        fundamental = {}


    # =====================================================
    # SCORES
    # =====================================================

    technical_score = safe_float(
        result.get(
            "TECHNICAL_SCORE"
        )
    )

    fundamental_score = safe_float(
        fundamental.get(
            "FUNDAMENTAL_SCORE"
        )
    )

    risk_score = safe_float(
        result.get(
            "RISK_SCORE"
        )
    )


    # =====================================================
    # MASTER SCORE
    # =====================================================

    master_score = round(
        (
            technical_score * 0.40
            +
            fundamental_score * 0.40
            +
            risk_score * 0.20
        ),
        1
    )


    # =====================================================
    # DECISION
    # =====================================================

    if master_score >= 75:

        decision = "BUY"

    elif master_score >= 60:

        decision = "HOLD"

    elif master_score >= 45:

        decision = "WAIT"

    elif master_score >= 30:

        decision = "SELL"

    else:

        decision = "EXIT"


    decision_text, decision_class, score_color = (
        decision_style(decision)
    )


    # =====================================================
    # EXIT MATRA
    # =====================================================

    cmp = safe_float(
        result.get(
            "CMP"
        )
    )

    stop_loss = safe_float(
        result.get(
            "STOP_LOSS"
        ),
        cmp * 0.95
    )

    swing_target = safe_float(
        result.get(
            "SWING_TARGET"
        ),
        cmp * 1.08
    )

    long_target = safe_float(
        result.get(
            "LONG_TERM_TARGET"
        ),
        cmp * 1.20
    )


    exit_signal = "HOLD"
    exit_reason = "Setup active"


    if cmp > 0:

        if cmp <= stop_loss:

            exit_signal = "EXIT"
            exit_reason = "Stop-loss breached"

        elif master_score < 30:

            exit_signal = "EXIT"
            exit_reason = "Master score weak"

        elif (
            technical_score < 35
            and
            fundamental_score < 35
        ):

            exit_signal = "REDUCE"
            exit_reason = (
                "Technical + fundamental weakness"
            )

        elif cmp >= long_target:

            exit_signal = "BOOK"
            exit_reason = (
                "Long-term target reached"
            )


    # =====================================================
    # MARKET ZONE
    # =====================================================

    market_zone, market_class = zone_style(
        master_score
    )


    # =====================================================
    # STOCK CARD
    # =====================================================

    st.markdown(
        '<div class="stock-card">',
        unsafe_allow_html=True
    )


    st.markdown(
        f'<div style="font-size:1.05rem;'
        f'font-weight:950;margin-bottom:8px;">'
        f'📌 {symbol}'
        f'</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # PRICE
    # =====================================================

    p1, p2, p3 = st.columns(3)

    with p1:

        st.metric(
            "CMP",
            money(cmp)
        )

    with p2:

        st.metric(
            "CHANGE",
            pct(
                result.get(
                    "CHANGE_%"
                )
            )
        )

    with p3:

        st.metric(
            "MOMENTUM",
            display_value(
                result.get(
                    "MOMENTUM_LEVEL"
                )
            )
        )


    # =====================================================
    # MASTER SCORE
    # =====================================================

    st.markdown(
        f"""
        <div class="score-card">

            <div class="score-title">
                🏦 MASTER SCORE
            </div>

            <div
                class="score-number"
                style="color:{score_color};"
            >
                {master_score:.1f}
                <span
                    style="
                    font-size:0.78rem;
                    opacity:0.55;
                    "
                >
                    /100
                </span>
            </div>

            <div class="score-bar">

                <div
                    class="score-fill"
                    style="
                    width:{max(0,min(master_score,100))}%;
                    background:{score_color};
                    "
                >
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
# E.M.S. — EXIT MANAGEMENT SYSTEM
# =====================================================

ems_input = {
    "master_score": master_score,

    # V2 analysisમાંથી available values હોય ત્યારે જ આપવાના.
    # Missing values માટે EMS DATA LIMITED આપશે.
    "trend_breakdown": None,
    "momentum_breakdown": None,
    "support_breakdown": None,
    "volume_confirmation": None,
    "relative_strength_breakdown": None,
    "risk_deterioration": None,

    "above_exit_price": None,
    "ath_profit": None,
    "outperformance": None,

    # Reference calibration
    "reference_match": None,
}

ems_result = evaluate_ems(ems_input)

ems_ui = ems_display(ems_result)

st.markdown(
    f"""
    <div class="score-card"
         style="
         border:1px solid {ems_ui['color']};
         margin-top:10px;
         ">

        <div class="score-title">
            🧠 E.M.S.
        </div>

        <div
            style="
            color:{ems_ui['color']};
            font-size:1.25rem;
            font-weight:950;
            margin-top:6px;
            "
        >
            {ems_ui['label']}
        </div>

        <div
            style="
            font-size:0.70rem;
            opacity:0.65;
            margin-top:6px;
            "
        >
            Independent Exit Management Layer
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


    # =====================================================
    # DECISION
    # =====================================================

    st.markdown(
        f"""
        <div class="signal-card {decision_class}">
            {decision_text}
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # MARKET ZONE
    # =====================================================

    st.markdown(
        f"""
        <div class="signal-card">
            {market_zone}
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # EXIT MATRA
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🚪 EXIT MATRA'
        '</div>',
        unsafe_allow_html=True
    )

    ex1, ex2 = st.columns(2)

    with ex1:

        if exit_signal == "EXIT":

            exit_display = "🔴 EXIT"

        elif exit_signal == "REDUCE":

            exit_display = "🟠 REDUCE"

        elif exit_signal == "BOOK":

            exit_display = "🟢 BOOK"

        else:

            exit_display = "🟢 HOLD"


        st.metric(
            "EXIT SIGNAL",
            exit_display
        )

    with ex2:

        st.metric(
            "REASON",
            exit_reason
        )


    # =====================================================
    # TARGET
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🎯 TARGET & RISK'
        '</div>',
        unsafe_allow_html=True
    )

    t1, t2, t3 = st.columns(3)

    with t1:

        st.markdown(
            '<div class="data-card target-green">'
            '<div class="data-label">🎯 SWING</div>'
            f'<div class="data-value">{money(swing_target)}</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with t2:

        st.markdown(
            '<div class="data-card target-blue">'
            '<div class="data-label">🚀 LONG</div>'
            f'<div class="data-value">{money(long_target)}</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with t3:

        st.markdown(
            '<div class="data-card target-red">'
            '<div class="data-label">🛑 STOP LOSS</div>'
            f'<div class="data-value">{money(stop_loss)}</div>'
            '</div>',
            unsafe_allow_html=True
        )


    # =====================================================
    # CHART
    # =====================================================

    build_price_chart(
        symbol,
        stop_loss,
        swing_target,
        long_target
    )


    # =====================================================
    # TECHNICAL
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📈 TECHNICAL'
        '</div>',
        unsafe_allow_html=True
    )

    tc1, tc2, tc3 = st.columns(3)

    with tc1:

        st.metric(
            "TECHNICAL",
            f"{technical_score:.0f}/100"
        )

    with tc2:

        st.metric(
            "ZONE",
            technical_zone_style(
                result.get(
                    "TECHNICAL_ZONE"
                )
            )
        )

    with tc3:

        st.metric(
            "RSI",
            number(
                result.get(
                    "RSI_14"
                ),
                2
            )
        )


    # =====================================================
    # EMA
    # =====================================================

    st.caption(
        "📊 EMA 10 / 20 / 50 / 100 / 200"
    )

    e1, e2, e3, e4, e5 = st.columns(5)

    with e1:
        st.metric(
            "10",
            money(result.get("EMA_10"))
        )

    with e2:
        st.metric(
            "20",
            money(result.get("EMA_20"))
        )

    with e3:
        st.metric(
            "50",
            money(result.get("EMA_50"))
        )

    with e4:
        st.metric(
            "100",
            money(result.get("EMA_100"))
        )

    with e5:
        st.metric(
            "200",
            money(result.get("EMA_200"))
        )


    # =====================================================
    # ALIGNMENT
    # =====================================================

    st.markdown(
        f"""
        <div class="data-card">

            <div class="data-label">
                EMA ALIGNMENT
            </div>

            <div class="data-value">
                {alignment_style(
                    result.get("EMA_ALIGNMENT")
                )}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # RSI MACD
    # =====================================================

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "RSI 14",
            number(
                result.get(
                    "RSI_14"
                ),
                2
            )
        )

    with m2:

        st.metric(
            "MACD",
            number(
                result.get(
                    "MACD"
                ),
                2
            )
        )

    with m3:

        st.metric(
            "HISTOGRAM",
            number(
                result.get(
                    "MACD_HIST"
                ),
                2
            )
        )


    # =====================================================
    # TREND / CPR / PIVOT
    # =====================================================

    st.caption(
        "🧭 TREND / CPR / PIVOT"
    )

    s1, s2, s3 = st.columns(3)

    with s1:

        st.metric(
            "SUPERTREND",
            money(
                result.get(
                    "SUPERTREND"
                )
            )
        )

    with s2:

        st.metric(
            "TREND",
            trend_style(
                result.get(
                    "SUPERTREND_STATUS"
                )
            )
        )

    with s3:

        st.metric(
            "PIVOT",
            money(
                result.get(
                    "PIVOT"
                )
            )
        )


    # =====================================================
    # VOLUME
    # =====================================================

    v1, v2, v3 = st.columns(3)

    with v1:

        st.metric(
            "VOLUME",
            integer_number(
                result.get(
                    "VOLUME"
                )
            )
        )

    with v2:

        st.metric(
            "VOLUME RATIO",
            number(
                result.get(
                    "VOLUME_RATIO"
                ),
                2
            ) + "x"
        )

    with v3:

        st.metric(
            "BREAKOUT",
            breakout_style(
                result.get(
                    "VOLUME_BREAKOUT"
                )
            )
        )


    # =====================================================
    # FUNDAMENTAL
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🏢 FUNDAMENTAL • GROWTH • QUALITY'
        '</div>',
        unsafe_allow_html=True
    )

    f1, f2, f3 = st.columns(3)

    with f1:

        st.metric(
            "FUNDAMENTAL",
            f"{fundamental_score:.0f}/100"
        )

    with f2:

        st.metric(
            "ZONE",
            technical_zone_style(
                fundamental.get(
                    "FUNDAMENTAL_ZONE"
                )
            )
        )

    with f3:

        st.metric(
            "QUALITY",
            pct(
                fundamental.get(
                    "DATA_QUALITY_%"
                )
            )
        )


    f4, f5, f6 = st.columns(3)

    with f4:

        st.metric(
            "REVENUE GROWTH",
            pct(
                fundamental.get(
                    "REVENUE_GROWTH_%"
                )
            )
        )

    with f5:

        st.metric(
            "PROFIT GROWTH",
            pct(
                fundamental.get(
                    "PROFIT_GROWTH_%"
                )
            )
        )

    with f6:

        st.metric(
            "EPS GROWTH",
            pct(
                fundamental.get(
                    "EPS_GROWTH_%"
                )
            )
        )


    f7, f8, f9 = st.columns(3)

    with f7:

        st.metric(
            "ROE",
            pct(
                fundamental.get(
                    "ROE_%"
                )
            )
        )

    with f8:

        st.metric(
            "ROCE",
            pct(
                fundamental.get(
                    "ROCE_%"
                )
            )
        )

    with f9:

        st.metric(
            "DEBT / EQUITY",
            number(
                fundamental.get(
                    "DEBT_TO_EQUITY"
                ),
                2
            )
        )


    # =====================================================
    # VALUATION
    # =====================================================

    st.caption(
        "💰 VALUATION"
    )

    q1, q2, q3, q4 = st.columns(4)

    with q1:

        st.metric(
            "PE",
            number(
                fundamental.get(
                    "PE"
                ),
                2
            )
        )

    with q2:

        st.metric(
            "FORWARD PE",
            number(
                fundamental.get(
                    "FORWARD_PE"
                ),
                2
            )
        )

    with q3:

        st.metric(
            "PB",
            number(
                fundamental.get(
                    "PB"
                ),
                2
            )
        )

    with q4:

        st.metric(
            "VALUATION",
            valuation_style(
                fundamental.get(
                    "VALUATION_ZONE"
                )
            )
        )


    # =====================================================
    # RISK
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🛡️ RISK'
        '</div>',
        unsafe_allow_html=True
    )

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "RISK SCORE",
            f"{risk_score:.0f}/100"
        )

    with r2:

        st.metric(
            "RISK LEVEL",
            display_value(
                result.get(
                    "RISK_LEVEL"
                )
            )
        )

    with r3:

        st.metric(
            "RISK %",
            pct(
                result.get(
                    "RISK_%"
                )
            )
        )


    # =====================================================
    # DATA STATUS
    # =====================================================

    st.caption(
        "📅 Data Date: "
        +
        display_value(
            result.get(
                "DATA_DATE"
            )
        )
        +
        " | NSE Status: "
        +
        display_value(
            result.get(
                "STATUS"
            )
        )
        +
        " | Fundamental: "
        +
        display_value(
            fundamental.get(
                "STATUS"
            )
        )
    )


    # =====================================================
    # SHARE / COPY
    # =====================================================

    share_text = f"""
📈 R.S MASTER STOCK GUIDE V2

📌 STOCK: {symbol}
CMP: {money(cmp)}
CHANGE: {pct(result.get("CHANGE_%"))}

🏦 MASTER SCORE: {master_score:.1f}/100
🎯 DECISION: {decision_text}
{market_zone}

📈 TECHNICAL: {technical_score:.0f}/100
🏢 FUNDAMENTAL: {fundamental_score:.0f}/100
🛡️ RISK: {risk_score:.0f}/100

🎯 SWING: {money(swing_target)}
🚀 LONG: {money(long_target)}
🛑 STOP LOSS: {money(stop_loss)}

🚪 EXIT MATRA: {exit_signal}
REASON: {exit_reason}

📊 RSI 14: {number(result.get("RSI_14"), 2)}
📊 MACD: {number(result.get("MACD"), 2)}
📊 HISTOGRAM: {number(result.get("MACD_HIST"), 2)}

🚀 BREAKOUT:
{breakout_style(result.get("VOLUME_BREAKOUT"))}

🏢 REVENUE GROWTH:
{pct(fundamental.get("REVENUE_GROWTH_%"))}

💰 PROFIT GROWTH:
{pct(fundamental.get("PROFIT_GROWTH_%"))}

EPS GROWTH:
{pct(fundamental.get("EPS_GROWTH_%"))}

ROE:
{pct(fundamental.get("ROE_%"))}

ROCE:
{pct(fundamental.get("ROCE_%"))}

DATA DATE:
{display_value(result.get("DATA_DATE"))}
"""

    with st.expander(
        "📤 SHARE / COPY RESULT"
    ):

        st.code(
            share_text.strip(),
            language="text"
        )


    # =====================================================
    # SCORE STORAGE
    # =====================================================

    all_scores.append(
        {
            "SYMBOL": symbol,
            "MASTER_SCORE": master_score,
            "DECISION": decision,
            "EXIT_MATRA": exit_signal,
            "ZONE": market_zone
        }
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    st.divider()


# =========================================================
# PORTFOLIO SUMMARY
# =========================================================

st.markdown(
    '<div class="section-title">'
    '📊 PORTFOLIO SUMMARY'
    '</div>',
    unsafe_allow_html=True
)


if all_scores:

    scores_df = pd.DataFrame(
        all_scores
    )

    health = round(
        scores_df[
            "MASTER_SCORE"
        ].mean(),
        1
    )

    buy = int(
        (
            scores_df["DECISION"]
            == "BUY"
        ).sum()
    )

    hold = int(
        (
            scores_df["DECISION"]
            == "HOLD"
        ).sum()
    )

    wait = int(
        (
            scores_df["DECISION"]
            == "WAIT"
        ).sum()
    )

    sell = int(
        (
            scores_df["DECISION"]
            == "SELL"
        ).sum()
    )

    exit_count = int(
        (
            scores_df["DECISION"]
            == "EXIT"
        ).sum()
    )


    b1, b2, b3 = st.columns(3)

    with b1:

        st.markdown(
            f"""
            <div class="summary-card">

                <div class="data-label">
                    📊 STOCKS
                </div>

                <div class="summary-number">
                    {len(scores_df)}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with b2:

        health_color = (
            "#00e676"
            if health >= 60
            else "#ffd740"
            if health >= 40
            else "#ff304f"
        )

        st.markdown(
            f"""
            <div class="summary-card">

                <div class="data-label">
                    ❤️ PORTFOLIO HEALTH
                </div>

                <div
                    class="summary-number"
                    style="color:{health_color};"
                >
                    {health}/100
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with b3:

        st.markdown(
            f"""
            <div class="summary-card">

                <div class="data-label">
                    🟢 BUY
                </div>

                <div
                    class="summary-number"
                    style="color:#00e676;"
                >
                    {buy}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    st.markdown(
        f"""
        <div
            style="
            text-align:center;
            margin:10px 0;
            font-size:0.78rem;
            font-weight:800;
            "
        >
            🟢 BUY: {buy}
            &nbsp; | &nbsp;
            🔵 HOLD: {hold}
            &nbsp; | &nbsp;
            🟡 WAIT: {wait}
            &nbsp; | &nbsp;
            🟠 REDUCE: {sell}
            &nbsp; | &nbsp;
            🔴 EXIT: {exit_count}
        </div>
        """,
        unsafe_allow_html=True
    )


    st.dataframe(
        scores_df,
        width="stretch",
        hide_index=True
    )


else:

    st.info(
        "Portfolio summary માટે stock data ઉપલબ્ધ નથી."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "📈 R.S MASTER STOCK GUIDE V2 | "
    "NSE Stock Decision System"
)
