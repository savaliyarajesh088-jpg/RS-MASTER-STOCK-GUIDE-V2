import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.nse_data import fetch_nse_data
from src.fundamental_engine import fetch_fundamental_data


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
# MOBILE FULL COLOR UI
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at top left, #172554 0%, #050505 32%),
        #050505;
    color: #ffffff;
}

.block-container {
    max-width: 1250px;
    padding: 0.45rem 0.35rem 1.5rem 0.35rem;
}

/* ---------- HEADER ---------- */

.rs-header {
    text-align:center;
    padding:12px 8px 8px 8px;
}

.rs-title {
    font-size:1.65rem;
    font-weight:950;
    letter-spacing:0.4px;
}

.rs-subtitle {
    font-size:0.72rem;
    opacity:0.70;
}

/* ---------- STOCK CARD ---------- */

.stock-card {
    background:
        linear-gradient(145deg,
        rgba(20,20,30,0.98),
        rgba(7,7,12,0.98));
    border:1px solid rgba(255,255,255,0.14);
    border-radius:20px;
    padding:12px;
    margin:10px 0 18px 0;
    box-shadow:
        0 8px 30px rgba(0,0,0,0.35);
}

/* ---------- STOCK NAME ---------- */

.stock-name {
    font-size:1.25rem;
    font-weight:950;
    margin-bottom:8px;
}

/* ---------- COLOR CARDS ---------- */

.color-card {
    border-radius:15px;
    padding:10px;
    text-align:center;
    margin:4px 0;
    border:1px solid rgba(255,255,255,0.15);
}

.card-label {
    font-size:0.64rem;
    opacity:0.72;
    font-weight:800;
}

.card-value {
    font-size:1.05rem;
    font-weight:950;
    margin-top:2px;
}

.blue-card {
    background:linear-gradient(145deg,
        rgba(0,120,255,0.22),
        rgba(0,60,180,0.08));
}

.green-card {
    background:linear-gradient(145deg,
        rgba(0,220,110,0.22),
        rgba(0,100,50,0.08));
}

.red-card {
    background:linear-gradient(145deg,
        rgba(255,40,60,0.22),
        rgba(120,0,20,0.08));
}

.orange-card {
    background:linear-gradient(145deg,
        rgba(255,150,0,0.23),
        rgba(120,60,0,0.08));
}

.yellow-card {
    background:linear-gradient(145deg,
        rgba(255,210,0,0.20),
        rgba(120,90,0,0.08));
}

.purple-card {
    background:linear-gradient(145deg,
        rgba(160,80,255,0.22),
        rgba(80,20,150,0.08));
}

/* ---------- SIGNAL ---------- */

.signal-box {
    border-radius:18px;
    padding:13px 8px;
    text-align:center;
    margin:9px 0;
    border:1px solid rgba(255,255,255,0.20);
    font-size:1.18rem;
    font-weight:950;
}

.signal-small {
    font-size:0.68rem;
    opacity:0.75;
    font-weight:800;
}

/* ---------- SCORE ---------- */

.score-wrap {
    background:#111318;
    border:1px solid rgba(255,255,255,0.12);
    border-radius:17px;
    padding:12px;
    margin:8px 0;
}

.score-heading {
    font-size:0.68rem;
    font-weight:900;
    opacity:0.7;
}

.score-number {
    font-size:1.7rem;
    font-weight:950;
}

.score-bar {
    width:100%;
    height:9px;
    border-radius:20px;
    background:#252525;
    overflow:hidden;
    margin-top:6px;
}

.score-fill {
    height:100%;
    border-radius:20px;
}

/* ---------- SECTION ---------- */

.section-title {
    font-size:0.96rem;
    font-weight:950;
    margin-top:12px;
    margin-bottom:6px;
    padding-left:3px;
}

/* ---------- MINI SIGNAL ---------- */

.mini-signal {
    display:inline-block;
    padding:4px 8px;
    border-radius:10px;
    font-size:0.70rem;
    font-weight:900;
    border:1px solid rgba(255,255,255,0.15);
}

/* ---------- MOBILE ---------- */

@media(max-width:640px){

    .block-container {
        padding-left:0.18rem;
        padding-right:0.18rem;
    }

    .rs-title {
        font-size:1.20rem;
    }

    .rs-subtitle {
        font-size:0.59rem;
    }

    .stock-card {
        border-radius:16px;
        padding:9px;
    }

    .stock-name {
        font-size:1.10rem;
    }

    .card-value {
        font-size:0.92rem;
    }

    .score-number {
        font-size:1.42rem;
    }

    .signal-box {
        font-size:1.02rem;
        padding:11px 6px;
    }

    .section-title {
        font-size:0.86rem;
    }

}

/* Remove dataframe excess width */
div[data-testid="stDataFrame"] {
    border-radius:14px;
    overflow:hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================

def safe_float(v, default=0.0):
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def money(v):
    try:
        if v is None or pd.isna(v):
            return "—"
        return f"₹{float(v):,.2f}"
    except Exception:
        return "—"


def pct(v):
    try:
        if v is None or pd.isna(v):
            return "—"
        return f"{float(v):.2f}%"
    except Exception:
        return "—"


def number2(v):
    try:
        if v is None or pd.isna(v):
            return "—"
        return f"{float(v):.2f}"
    except Exception:
        return "—"


def clean_symbol(symbol):
    symbol = str(symbol).strip().upper()

    if symbol.endswith(".NS"):
        symbol = symbol[:-3]

    return symbol


def display_value(v):
    try:
        if v is None or pd.isna(v):
            return "—"
    except Exception:
        pass

    return v


# =========================================================
# COLOR LOGIC
# =========================================================

def score_color(score):

    if score >= 75:
        return "#00e676"

    if score >= 60:
        return "#2196f3"

    if score >= 45:
        return "#ffd000"

    if score >= 30:
        return "#ff9800"

    return "#ff304f"


def decision_info(score):

    if score >= 75:
        return (
            "🟢🔥 BUY",
            "buy",
            "Strong setup"
        )

    if score >= 60:
        return (
            "🔵🛡️ HOLD",
            "hold",
            "Good setup"
        )

    if score >= 45:
        return (
            "🟡⏳ WAIT",
            "wait",
            "Confirmation required"
        )

    if score >= 30:
        return (
            "🟠⚠️ REDUCE",
            "reduce",
            "Weak setup"
        )

    return (
        "🔴🚪 EXIT",
        "exit",
        "Master score weak"
    )


def market_info(score):

    if score >= 75:
        return "🐂🟢 BULL", "green-card"

    if score >= 55:
        return "🟡 NEUTRAL", "yellow-card"

    return "🐻🔴 BEAR", "red-card"


def signal_class(kind):

    mapping = {
        "buy": "green-card",
        "hold": "blue-card",
        "wait": "yellow-card",
        "reduce": "orange-card",
        "exit": "red-card"
    }

    return mapping.get(kind, "blue-card")


# =========================================================
# COLOR SCORE
# =========================================================

def score_html(score):

    color = score_color(score)

    return f"""
    <div class="score-wrap">
        <div class="score-heading">
            🏦 MASTER SCORE
        </div>

        <div class="score-number">
            {score:.1f}<span style="font-size:0.8rem;opacity:0.6">/100</span>
        </div>

        <div class="score-bar">
            <div class="score-fill"
                 style="width:{max(0,min(score,100))}%;
                        background:{color};">
            </div>
        </div>
    </div>
    """


# =========================================================
# CARD
# =========================================================

def color_card(label, val, cls="blue-card"):

    st.markdown(
        f"""
        <div class="color-card {cls}">
            <div class="card-label">{label}</div>
            <div class="card-value">{val}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# ATR
# =========================================================

def calculate_atr(df, period=14):

    high = pd.to_numeric(
        df["High"], errors="coerce"
    )

    low = pd.to_numeric(
        df["Low"], errors="coerce"
    )

    close = pd.to_numeric(
        df["Close"], errors="coerce"
    )

    previous_close = close.shift(1)

    tr1 = high - low

    tr2 = (
        high - previous_close
    ).abs()

    tr3 = (
        low - previous_close
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return tr.rolling(period).mean()


# =========================================================
# ADVANCED CHART
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

    ticker = clean_symbol(symbol) + ".NS"

    try:

        data = yf.download(
            ticker,
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if data.empty:
            st.caption(
                "📊 Historical chart data ઉપલબ્ધ નથી."
            )
            return

        if isinstance(
            data.columns,
            pd.MultiIndex
        ):
            data.columns = [
                c[0]
                for c in data.columns
            ]

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if any(
            c not in data.columns
            for c in required
        ):
            st.caption(
                "📊 Chart data incomplete."
            )
            return

        data = data[required].copy()

        for c in required:
            data[c] = pd.to_numeric(
                data[c],
                errors="coerce"
            )

        data.dropna(inplace=True)

        if len(data) < 50:
            return

        close = data["Close"]

        # EMA
        for p in [10,20,50,100,200]:

            data[f"EMA{p}"] = (
                close.ewm(
                    span=p,
                    adjust=False
                ).mean()
            )

        # RSI
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
            100 / (1 + rs)
        )

        # MACD
        ema12 = close.ewm(
            span=12,
            adjust=False
        ).mean()

        ema26 = close.ewm(
            span=26,
            adjust=False
        ).mean()

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

        # Volume
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

        # Breakout
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

        cmp = float(close.iloc[-1])

        one_year = data.tail(252)

        high_52 = float(
            one_year["High"].max()
        )

        low_52 = float(
            one_year["Low"].min()
        )

        atr_series = (
            calculate_atr(data)
            .dropna()
        )

        atr = (
            float(atr_series.iloc[-1])
            if not atr_series.empty
            else max(cmp * 0.02, 1)
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

        ema10 = float(
            data["EMA10"].iloc[-1]
        )

        ema20 = float(
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

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.025,
            row_heights=[0.78,0.22]
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

        for column,name,width in [
            ("EMA10","EMA 10",1.1),
            ("EMA20","EMA 20",1.1),
            ("EMA50","EMA 50",1.4),
            ("EMA100","EMA 100",1.4),
            ("EMA200","EMA 200",1.8)
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

        if momentum_ready:

            fig.add_trace(
                go.Scatter(
                    x=[data.index[-1]],
                    y=[cmp],
                    mode="markers+text",
                    text=["⚡ MOMENTUM"],
                    textposition="top center",
                    name="⚡ Momentum"
                ),
                row=1,
                col=1
            )

        levels = [
            (cmp, f"CMP ₹{cmp:,.2f}", "dot"),
            (stop_loss, f"🛑 SL ₹{stop_loss:,.2f}", "dash"),
            (swing_target, f"🎯 SWING ₹{swing_target:,.2f}", "dot"),
            (long_target, f"🚀 LONG ₹{long_target:,.2f}", "dot"),
            (high_52, f"52W HIGH ₹{high_52:,.2f}", "dashdot"),
            (low_52, f"52W LOW ₹{low_52:,.2f}", "dashdot")
        ]

        for level,label,dash in levels:

            fig.add_hline(
                y=level,
                row=1,
                col=1,
                line_dash=dash,
                line_width=1,
                annotation_text=label,
                annotation_position="top right"
            )

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

        fig.update_layout(
            height=620,
            template="plotly_dark",
            paper_bgcolor="#050505",
            plot_bgcolor="#050505",
            hovermode="x unified",
            dragmode="pan",
            margin=dict(
                l=5,r=5,t=45,b=5
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

        st.caption(
            "🤏 Pinch zoom • Drag • Double tap reset • "
            "🔍 1M / 3M / 6M / 1Y / ALL"
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config=config,
            key=f"chart_{clean_symbol(symbol)}"
        )

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
            f"📍 CMP {money(cmp)}  |  "
            f"🛑 SL {money(stop_loss)}  |  "
            f"🎯 Swing {money(swing_target)}  |  "
            f"🚀 Long {money(long_target)}"
        )

    except Exception as error:

        st.error(
            f"📊 Chart Error: {type(error).__name__}: {error}"
        )


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="rs-header">
    <div class="rs-title">
        📈 R.S MASTER STOCK GUIDE V2
    </div>
    <div class="rs-subtitle">
        NSE • Technical • Fundamental • Momentum • Breakout •
        Exit Matra • Risk • Target
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()


# =========================================================
# PORTFOLIO
# =========================================================

st.header("📁 પોર્ટફોલિયો")

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
# PROCESS
# =========================================================

all_scores = []


for raw_symbol in portfolio["SYMBOL"]:

    symbol = clean_symbol(raw_symbol)

    # -----------------------------------------------------
    # NSE
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # FUNDAMENTAL
    # -----------------------------------------------------

    try:

        fundamental = (
            fetch_fundamental_data(symbol)
        )

    except Exception as error:

        fundamental = {
            "FUNDAMENTAL_SCORE":0,
            "FUNDAMENTAL_ZONE":"DATA ERROR",
            "DATA_QUALITY_%":0,
            "ERROR":str(error)
        }


    if not isinstance(
        fundamental,
        dict
    ):

        fundamental = {}


    # -----------------------------------------------------
    # SCORE
    # -----------------------------------------------------

    technical_score = safe_float(
        result.get("TECHNICAL_SCORE")
    )

    fundamental_score = safe_float(
        fundamental.get("FUNDAMENTAL_SCORE")
    )

    risk_score = safe_float(
        result.get("RISK_SCORE")
    )

    master_score = round(
        technical_score * 0.40
        +
        fundamental_score * 0.40
        +
        risk_score * 0.20,
        1
    )


    decision_text, decision_kind, decision_reason = (
        decision_info(master_score)
    )

    market_zone, market_class = (
        market_info(master_score)
    )


    # -----------------------------------------------------
    # PRICE
    # -----------------------------------------------------

    cmp = safe_float(
        result.get("CMP")
    )

    stop_loss = safe_float(
        result.get("STOP_LOSS"),
        cmp * 0.95
    )

    swing_target = safe_float(
        result.get("SWING_TARGET"),
        cmp * 1.08
    )

    long_target = safe_float(
        result.get("LONG_TERM_TARGET"),
        cmp * 1.20
    )


    # -----------------------------------------------------
    # EXIT MATRA
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # STOCK CARD
    # -----------------------------------------------------

    st.markdown(
        '<div class="stock-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="stock-name">📌 {symbol}</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # PRICE CARDS
    # -----------------------------------------------------

    p1,p2,p3 = st.columns(3)

    with p1:
        color_card(
            "CMP",
            money(cmp),
            "blue-card"
        )

    with p2:
        color_card(
            "CHANGE",
            pct(result.get("CHANGE_%")),
            "green-card"
        )

    with p3:
        color_card(
            "MOMENTUM",
            display_value(
                result.get("MOMENTUM_LEVEL")
            ),
            "purple-card"
        )


    # -----------------------------------------------------
    # MASTER SCORE
    # -----------------------------------------------------

    st.markdown(
        score_html(master_score),
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # DECISION
    # -----------------------------------------------------

    st.markdown(
        f"""
        <div class="signal-box {signal_class(decision_kind)}">
            {decision_text}
            <div class="signal-small">
                {decision_reason}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # MARKET
    # -----------------------------------------------------

    st.markdown(
        f"""
        <div class="signal-box {market_class}">
            {market_zone}
        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # EXIT MATRA
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🚪 EXIT MATRA</div>',
        unsafe_allow_html=True
    )

    ex1,ex2 = st.columns(2)

    with ex1:

        color_card(
            "EXIT SIGNAL",
            (
                "🔴 " + exit_signal
                if exit_signal == "EXIT"
                else
                "🟢 " + exit_signal
            ),
            (
                "red-card"
                if exit_signal == "EXIT"
                else "green-card"
            )
        )

    with ex2:

        color_card(
            "REASON",
            exit_reason,
            "orange-card"
        )


    # -----------------------------------------------------
    # TARGET
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🎯 TARGET & RISK</div>',
        unsafe_allow_html=True
    )

    t1,t2,t3 = st.columns(3)

    with t1:
        color_card(
            "🎯 SWING",
            money(swing_target),
            "green-card"
        )

    with t2:
        color_card(
            "🚀 LONG",
            money(long_target),
            "purple-card"
        )

    with t3:
        color_card(
            "🛑 STOP LOSS",
            money(stop_loss),
            "red-card"
        )


    # -----------------------------------------------------
    # CHART
    # -----------------------------------------------------

    build_price_chart(
        symbol,
        stop_loss,
        swing_target,
        long_target
    )


    # -----------------------------------------------------
    # TECHNICAL
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📈 TECHNICAL</div>',
        unsafe_allow_html=True
    )

    tc1,tc2,tc3 = st.columns(3)

    with tc1:
        color_card(
            "TECHNICAL",
            f"{technical_score:.0f}/100",
            (
                "green-card"
                if technical_score >= 60
                else
                "orange-card"
                if technical_score >= 40
                else
                "red-card"
            )
        )

    with tc2:
        color_card(
            "ZONE",
            display_value(
                result.get("TECHNICAL_ZONE")
            ),
            "blue-card"
        )

    with tc3:
        color_card(
            "RSI",
            number2(
                result.get("RSI_14")
            ),
            (
                "green-card"
                if safe_float(result.get("RSI_14")) >= 50
                else
                "red-card"
            )
        )


    # -----------------------------------------------------
    # EMA
    # -----------------------------------------------------

    st.caption(
        "📊 EMA 10 / 20 / 50 / 100 / 200"
    )

    e1,e2,e3,e4,e5 = st.columns(5)

    ema_values = [
        ("10","EMA_10"),
        ("20","EMA_20"),
        ("50","EMA_50"),
        ("100","EMA_100"),
        ("200","EMA_200")
    ]

    for col,(label,key) in zip(
        [e1,e2,e3,e4,e5],
        ema_values
    ):

        with col:

            color_card(
                label,
                money(result.get(key)),
                "blue-card"
            )


    alignment = display_value(
        result.get("EMA_ALIGNMENT")
    )

    alignment_text = str(alignment)

    if "BULL" in alignment_text.upper():

        align_class = "green-card"
        align_symbol = "🟢📈"

    elif "BEAR" in alignment_text.upper():

        align_class = "red-card"
        align_symbol = "🔴📉"

    else:

        align_class = "yellow-card"
        align_symbol = "🟡↔️"


    color_card(
        "EMA ALIGNMENT",
        f"{align_symbol} {alignment}",
        align_class
    )


    # -----------------------------------------------------
    # RSI MACD
    # -----------------------------------------------------

    m1,m2,m3 = st.columns(3)

    with m1:

        color_card(
            "RSI 14",
            number2(
                result.get("RSI_14")
            ),
            "blue-card"
        )

    with m2:

        macd_value = safe_float(
            result.get("MACD"),
            np.nan
        )

        color_card(
            "MACD",
            number2(macd_value),
            (
                "green-card"
                if macd_value > 0
                else "red-card"
            )
        )

    with m3:

        hist = safe_float(
            result.get("MACD_HIST"),
            np.nan
        )

        color_card(
            "HISTOGRAM",
            number2(hist),
            (
                "green-card"
                if hist > 0
                else "red-card"
            )
        )


    # -----------------------------------------------------
    # TREND
    # -----------------------------------------------------

    st.caption(
        "🧭 Trend / CPR / Pivot"
    )

    s1,s2,s3 = st.columns(3)

    with s1:

        color_card(
            "SUPERTREND",
            money(
                result.get("SUPERTREND")
            ),
            "purple-card"
        )

    with s2:

        trend = display_value(
            result.get("SUPERTREND_STATUS")
        )

        trend_text = str(trend).upper()

        if "BULL" in trend_text:

            cls = "green-card"
            sym = "🟢🐂"

        elif "BEAR" in trend_text:

            cls = "red-card"
            sym = "🔴🐻"

        else:

            cls = "yellow-card"
            sym = "🟡↔️"

        color_card(
            "TREND",
            f"{sym} {trend}",
            cls
        )

    with s3:

        color_card(
            "PIVOT",
            money(
                result.get("PIVOT")
            ),
            "blue-card"
        )


    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

    v1,v2,v3 = st.columns(3)

    with v1:

        color_card(
            "VOLUME",
            display_value(
                result.get("VOLUME")
            ),
            "blue-card"
        )

    with v2:

        ratio = safe_float(
            result.get("VOLUME_RATIO")
        )

        color_card(
            "VOLUME RATIO",
            f"{ratio:.2f}x",
            (
                "green-card"
                if ratio >= 1.2
                else "yellow-card"
            )
        )

    with v3:

        breakout = display_value(
            result.get("VOLUME_BREAKOUT")
        )

        breakout_text = str(
            breakout
        ).upper()

        if breakout_text in [
            "YES",
            "TRUE",
            "1"
        ]:

            breakout_display = "🚀 YES"
            breakout_class = "green-card"

        else:

            breakout_display = "🔴 NO"
            breakout_class = "red-card"

        color_card(
            "BREAKOUT",
            breakout_display,
            breakout_class
        )


    # -----------------------------------------------------
    # FUNDAMENTAL
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '🏢 FUNDAMENTAL • GROWTH • QUALITY'
        '</div>',
        unsafe_allow_html=True
    )

    f1,f2,f3 = st.columns(3)

    with f1:

        color_card(
            "FUNDAMENTAL",
            f"{fundamental_score:.0f}/100",
            (
                "green-card"
                if fundamental_score >= 60
                else
                "orange-card"
                if fundamental_score >= 40
                else
                "red-card"
            )
        )

    with f2:

        color_card(
            "ZONE",
            display_value(
                fundamental.get(
                    "FUNDAMENTAL_ZONE"
                )
            ),
            "blue-card"
        )

    with f3:

        quality = safe_float(
            fundamental.get(
                "DATA_QUALITY_%"
            )
        )

        color_card(
            "QUALITY",
            f"{quality:.2f}%",
            "purple-card"
        )


    f4,f5,f6 = st.columns(3)

    with f4:

        color_card(
            "REVENUE GROWTH",
            pct(
                fundamental.get(
                    "REVENUE_GROWTH_%"
                )
            ),
            (
                "green-card"
                if safe_float(
                    fundamental.get(
                        "REVENUE_GROWTH_%"
                    )
                ) >= 0
                else "red-card"
            )
        )

    with f5:

        color_card(
            "PROFIT GROWTH",
            pct(
                fundamental.get(
                    "PROFIT_GROWTH_%"
                )
            ),
            (
                "green-card"
                if safe_float(
                    fundamental.get(
                        "PROFIT_GROWTH_%"
                    )
                ) >= 0
                else "red-card"
            )
        )

    with f6:

        color_card(
            "EPS GROWTH",
            pct(
                fundamental.get(
                    "EPS_GROWTH_%"
                )
            ),
            (
                "green-card"
                if safe_float(
                    fundamental.get(
                        "EPS_GROWTH_%"
                    )
                ) >= 0
                else "red-card"
            )
        )


    f7,f8,f9 = st.columns(3)

    with f7:

        color_card(
            "ROE",
            pct(
                fundamental.get(
                    "ROE_%"
                )
            ),
            "green-card"
        )

    with f8:

        color_card(
            "ROCE",
            pct(
                fundamental.get(
                    "ROCE_%"
                )
            ),
            "green-card"
        )

    with f9:

        de = safe_float(
            fundamental.get(
                "DEBT_TO_EQUITY"
            ),
            np.nan
        )

        color_card(
            "DEBT / EQUITY",
            number2(de),
            (
                "green-card"
                if de <= 1
                else "orange-card"
            )
        )


    # -----------------------------------------------------
    # VALUATION
    # -----------------------------------------------------

    st.caption("💰 Valuation")

    q1,q2,q3,q4 = st.columns(4)

    with q1:
        color_card(
            "PE",
            number2(
                fundamental.get("PE")
            ),
            "blue-card"
        )

    with q2:
        color_card(
            "FORWARD PE",
            number2(
                fundamental.get("FORWARD_PE")
            ),
            "blue-card"
        )

    with q3:
        color_card(
            "PB",
            number2(
                fundamental.get("PB")
            ),
            "blue-card"
        )

    with q4:
        color_card(
            "VALUATION",
            display_value(
                fundamental.get(
                    "VALUATION_ZONE"
                )
            ),
            "yellow-card"
        )


    # -----------------------------------------------------
    # RISK
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🛡️ RISK</div>',
        unsafe_allow_html=True
    )

    r1,r2,r3 = st.columns(3)

    with r1:

        color_card(
            "RISK SCORE",
            f"{risk_score:.0f}/100",
            (
                "green-card"
                if risk_score >= 60
                else
                "orange-card"
                if risk_score >= 30
                else
                "red-card"
            )
        )

    with r2:

        color_card(
            "RISK LEVEL",
            display_value(
                result.get("RISK_LEVEL")
            ),
            "orange-card"
        )

    with r3:

        color_card(
            "RISK %",
            pct(
                result.get("RISK_%")
            ),
            "red-card"
        )


    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    st.caption(
        "📅 Data Date: "
        + str(
            display_value(
                result.get("DATA_DATE")
            )
        )
        + " | NSE Status: "
        + str(
            display_value(
                result.get("STATUS")
            )
        )
        + " | Fundamental: "
        + str(
            display_value(
                fundamental.get("STATUS")
            )
        )
    )


    # -----------------------------------------------------
    # SHARE
    # -----------------------------------------------------

    share_text = f"""
📈 R.S MASTER STOCK GUIDE V2

📌 STOCK: {symbol}
💰 CMP: ₹{cmp:.2f}

🏦 MASTER SCORE: {master_score:.1f}/100
🎯 DECISION: {decision_text}
{market_zone}

📈 TECHNICAL: {technical_score:.0f}/100
🏢 FUNDAMENTAL: {fundamental_score:.0f}/100
🛡️ RISK: {risk_score:.0f}/100

🎯 SWING: ₹{swing_target:.2f}
🚀 LONG: ₹{long_target:.2f}
🛑 STOP LOSS: ₹{stop_loss:.2f}

🚪 EXIT MATRA: {exit_signal}
Reason: {exit_reason}

📊 RSI: {number2(result.get("RSI_14"))}
📈 MACD: {number2(result.get("MACD"))}
📉 HISTOGRAM: {number2(result.get("MACD_HIST"))}

📊 EMA ALIGNMENT:
{display_value(result.get("EMA_ALIGNMENT"))}

🧭 TREND:
{display_value(result.get("SUPERTREND_STATUS"))}

🚀 BREAKOUT:
{display_value(result.get("VOLUME_BREAKOUT"))}

🏢 REVENUE GROWTH:
{pct(fundamental.get("REVENUE_GROWTH_%"))}

💰 PROFIT GROWTH:
{pct(fundamental.get("PROFIT_GROWTH_%"))}

ROE:
{pct(fundamental.get("ROE_%"))}

ROCE:
{pct(fundamental.get("ROCE_%"))}

📅 DATA DATE:
{display_value(result.get("DATA_DATE"))}
"""

    with st.expander(
        "📤 SHARE / COPY RESULT"
    ):

        st.code(
            share_text.strip(),
            language="text"
        )


    # -----------------------------------------------------
    # SCORE STORAGE
    # -----------------------------------------------------

    all_scores.append(
        {
            "SYMBOL":symbol,
            "MASTER_SCORE":master_score,
            "DECISION":decision_text,
            "EXIT_MATRA":exit_signal,
            "ZONE":market_zone
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

st.header("📊 PORTFOLIO SUMMARY")


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
        scores_df["DECISION"]
        .str.contains("BUY")
        .sum()
    )

    hold = int(
        scores_df["DECISION"]
        .str.contains("HOLD")
        .sum()
    )

    wait = int(
        scores_df["DECISION"]
        .str.contains("WAIT")
        .sum()
    )

    reduce_count = int(
        scores_df["DECISION"]
        .str.contains("REDUCE")
        .sum()
    )

    exit_count = int(
        scores_df["DECISION"]
        .str.contains("EXIT")
        .sum()
    )


    b1,b2,b3 = st.columns(3)

    with b1:

        color_card(
            "📊 STOCKS",
            len(scores_df),
            "blue-card"
        )

    with b2:

        color_card(
            "❤️ PORTFOLIO HEALTH",
            f"{health}/100",
            (
                "green-card"
                if health >= 60
                else
                "orange-card"
                if health >= 40
                else
                "red-card"
            )
        )

    with b3:

        color_card(
            "🟢 BUY",
            buy,
            "green-card"
        )


    st.caption(
        f"🔵 HOLD: {hold}  |  "
        f"🟡 WAIT: {wait}  |  "
        f"🟠 REDUCE: {reduce_count}  |  "
        f"🔴 EXIT: {exit_count}"
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

st.markdown(
    """
    <div style="
        text-align:center;
        opacity:0.65;
        font-size:0.70rem;
        padding:8px;">
        📈 R.S MASTER STOCK GUIDE V2<br>
        NSE Stock Decision System
    </div>
    """,
    unsafe_allow_html=True
)
