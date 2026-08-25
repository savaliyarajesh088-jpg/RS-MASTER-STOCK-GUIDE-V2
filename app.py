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
# BLACK MOBILE UI
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #050505;
    }

    .block-container {
        padding-top: 0.45rem;
        padding-left: 0.35rem;
        padding-right: 0.35rem;
        padding-bottom: 1rem;
        max-width: 1250px;
    }

    .main-title {
        text-align: center;
        font-size: 1.55rem;
        font-weight: 900;
        margin-bottom: 0;
    }

    .sub-title {
        text-align: center;
        font-size: 0.72rem;
        opacity: 0.70;
        margin-bottom: 0.5rem;
    }

    .stock-card {
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 16px;
        padding: 11px;
        margin-bottom: 15px;
        background: #0b0b0b;
    }

    .score-card {
        border-radius: 14px;
        padding: 12px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.13);
        background: #101010;
    }

    .score-title {
        font-size: 0.67rem;
        opacity: 0.7;
        font-weight: 800;
    }

    .score-value {
        font-size: 1.55rem;
        font-weight: 900;
    }

    .signal {
        border-radius: 13px;
        padding: 10px;
        text-align: center;
        font-size: 1rem;
        font-weight: 900;
        margin: 8px 0;
    }

    .buy {
        background: rgba(0,220,100,0.13);
        border: 1px solid rgba(0,220,100,0.38);
    }

    .hold {
        background: rgba(0,170,255,0.13);
        border: 1px solid rgba(0,170,255,0.38);
    }

    .wait {
        background: rgba(255,200,0,0.13);
        border: 1px solid rgba(255,200,0,0.38);
    }

    .sell {
        background: rgba(255,60,60,0.13);
        border: 1px solid rgba(255,60,60,0.38);
    }

    .section-title {
        font-size: 0.98rem;
        font-weight: 900;
        margin-top: 10px;
        margin-bottom: 6px;
    }

    @media (max-width: 640px) {

        .block-container {
            padding-left: 0.22rem;
            padding-right: 0.22rem;
        }

        .main-title {
            font-size: 1.20rem;
        }

        .sub-title {
            font-size: 0.62rem;
        }

        .score-value {
            font-size: 1.22rem;
        }

        .section-title {
            font-size: 0.88rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


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


def value(value):

    try:

        if value is None or pd.isna(value):
            return "—"

    except Exception:
        pass

    return value


def clean_symbol(symbol):

    symbol = str(
        symbol
    ).strip().upper()

    if symbol.endswith(".NS"):
        return symbol[:-3]

    return symbol


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

    return (
        true_range
        .rolling(period)
        .mean()
    )


# =========================================================
# ADVANCED CHART
# =========================================================

def build_price_chart(
    symbol,
    stop_loss,
    swing_target,
    long_target
):

    ticker_symbol = (
        clean_symbol(symbol)
        + ".NS"
    )

    st.markdown(
        "### 📊 ADVANCED PRICE CHART"
    )

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

        # -------------------------------------------------
        # MULTIINDEX FIX
        # -------------------------------------------------

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
            col
            for col in required
            if col not in data.columns
        ]

        if missing:

            st.warning(
                "Chart data missing: "
                + ", ".join(missing)
            )

            return

        data = data[
            required
        ].copy()

        for col in required:

            data[col] = pd.to_numeric(
                data[col],
                errors="coerce"
            )

        data.dropna(
            inplace=True
        )

        if len(data) < 50:

            st.warning(
                "Chart માટે historical data ઓછો છે."
            )

            return

        close = data["Close"]

        # -------------------------------------------------
        # EMA
        # -------------------------------------------------

        for period in [
            10,
            20,
            50,
            100,
            200
        ]:

            data[
                f"EMA{period}"
            ] = (
                close
                .ewm(
                    span=period,
                    adjust=False
                )
                .mean()
            )

        # -------------------------------------------------
        # RSI
        # -------------------------------------------------

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
            100
            -
            (
                100
                /
                (1 + rs)
            )
        )

        # -------------------------------------------------
        # MACD
        # -------------------------------------------------

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

        data["MACD"] = (
            ema12 - ema26
        )

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

        # -------------------------------------------------
        # VOLUME AVERAGE
        # -------------------------------------------------

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

        # -------------------------------------------------
        # BREAKOUT
        # -------------------------------------------------

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

        # -------------------------------------------------
        # CMP
        # -------------------------------------------------

        cmp = float(
            close.iloc[-1]
        )

        # -------------------------------------------------
        # 52 WEEK
        # -------------------------------------------------

        one_year = data.tail(252)

        high_52 = float(
            one_year["High"].max()
        )

        low_52 = float(
            one_year["Low"].min()
        )

        # -------------------------------------------------
        # ATR
        # -------------------------------------------------

        atr_series = (
            calculate_atr(
                data
            )
            .dropna()
        )

        if atr_series.empty:

            atr = max(
                cmp * 0.02,
                1
            )

        else:

            atr = float(
                atr_series.iloc[-1]
            )

        # -------------------------------------------------
        # FALLBACK TARGETS
        # -------------------------------------------------

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

        # -------------------------------------------------
        # MOMENTUM
        # -------------------------------------------------

        ema10 = float(
            data["EMA10"].iloc[-1]
        )

        ema20 = float(
            data["EMA20"].iloc[-1]
        )

        ema50 = float(
            data["EMA50"].iloc[-1]
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
            and
            rsi >= 55
            and
            macd > macd_signal
            and
            vol_ratio >= 1.2
        )

        # -------------------------------------------------
        # CHART
        # -------------------------------------------------

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.025,
            row_heights=[
                0.78,
                0.22
            ]
        )

        # -------------------------------------------------
        # CANDLE
        # -------------------------------------------------

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

        # -------------------------------------------------
        # EMA LINES
        # -------------------------------------------------

        ema_settings = [
            ("EMA10", "EMA 10", 1.1),
            ("EMA20", "EMA 20", 1.1),
            ("EMA50", "EMA 50", 1.4),
            ("EMA100", "EMA 100", 1.4),
            ("EMA200", "EMA 200", 1.8)
        ]

        for column, name, width in ema_settings:

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[column],
                    name=name,
                    mode="lines",
                    line=dict(
                        width=width
                    )
                ),
                row=1,
                col=1
            )

        # -------------------------------------------------
        # VOLUME
        # -------------------------------------------------

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

        # -------------------------------------------------
        # BREAKOUT MARKERS
        # -------------------------------------------------

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
                        size=10,
                        symbol="triangle-up"
                    )
                ),
                row=1,
                col=1
            )

        # -------------------------------------------------
        # CURRENT MOMENTUM MARKER
        # -------------------------------------------------

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

        # -------------------------------------------------
        # PRICE LEVELS
        # -------------------------------------------------

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

        # -------------------------------------------------
        # RANGE BUTTONS
        # -------------------------------------------------

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

        # -------------------------------------------------
        # LAYOUT
        # -------------------------------------------------

        fig.update_layout(

            height=680,

            template="plotly_dark",

            paper_bgcolor="#050505",

            plot_bgcolor="#050505",

            hovermode="x unified",

            dragmode="pan",

            margin=dict(
                l=5,
                r=5,
                t=45,
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

        # -------------------------------------------------
        # MOBILE INTERACTION
        # -------------------------------------------------

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

        # -------------------------------------------------
        # CHART SIGNAL
        # -------------------------------------------------

        if breakout:

            st.success(
                "🚀 BREAKOUT CONFIRMED"
            )

        elif momentum_ready:

            st.info(
                "⚡ MOMENTUM BUILDING — "
                "breakout પહેલાં strength દેખાઈ રહી છે"
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
# PORTFOLIO LOAD
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


st.dataframe(
    portfolio,
    width="stretch",
    hide_index=True
)

st.divider()


# =========================================================
# PROCESSING
# =========================================================

all_scores = []


for raw_symbol in portfolio["SYMBOL"]:

    symbol = clean_symbol(
        raw_symbol
    )

    # =====================================================
    # NSE
    # =====================================================

    try:

        result = fetch_nse_data(
            symbol
        )

    except Exception as error:

        st.error(
            f"{symbol}: NSE error — {error}"
        )

        continue


    if not isinstance(
        result,
        dict
    ):

        st.error(
            f"{symbol}: NSE response invalid."
        )

        continue


    # =====================================================
    # FUNDAMENTAL
    # =====================================================

    try:

        fundamental = (
            fetch_fundamental_data(
                symbol
            )
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

        decision_text = (
            "🟢 BUY / વધારો"
        )

        decision_class = "buy"

    elif master_score >= 60:

        decision = "HOLD"

        decision_text = (
            "🔵 HOLD / જાળવો"
        )

        decision_class = "hold"

    elif master_score >= 45:

        decision = "WAIT"

        decision_text = (
            "🟡 WAIT / રાહ જુઓ"
        )

        decision_class = "wait"

    elif master_score >= 30:

        decision = "SELL"

        decision_text = (
            "🟠 REDUCE / ઘટાડો"
        )

        decision_class = "wait"

    else:

        decision = "EXIT"

        decision_text = (
            "🔴 EXIT / બહાર નીકળો"
        )

        decision_class = "sell"


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

            exit_reason = (
                "Stop-loss breached"
            )

        elif master_score < 30:

            exit_signal = "EXIT"

            exit_reason = (
                "Master score weak"
            )

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

    if master_score >= 75:

        market_zone = "🐂 BULL"

    elif master_score >= 55:

        market_zone = "🐷 NEUTRAL"

    else:

        market_zone = "🐻 BEAR"


    # =====================================================
    # STOCK CARD
    # =====================================================

    st.markdown(
        '<div class="stock-card">',
        unsafe_allow_html=True
    )

    st.subheader(
        f"📌 {symbol}"
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
            "Change",
            pct(
                result.get(
                    "CHANGE_%"
                )
            )
        )

    with p3:

        st.metric(
            "Momentum",
            money(
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
            <div class="score-value">
                {master_score}/100
            </div>
        </div>

        <div class="signal {decision_class}">
            🎯 {decision_text}
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class="signal hold">
            MARKET ZONE<br>
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

        st.metric(
            "Exit Signal",
            exit_signal
        )

    with ex2:

        st.metric(
            "Reason",
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

        st.metric(
            "Swing",
            money(
                swing_target
            )
        )

    with t2:

        st.metric(
            "Long",
            money(
                long_target
            )
        )

    with t3:

        st.metric(
            "Stop Loss",
            money(
                stop_loss
            )
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

    tc1.metric(
        "Technical",
        f"{technical_score:.0f}/100"
    )

    tc2.metric(
        "Zone",
        value(
            result.get(
                "TECHNICAL_ZONE"
            )
        )
    )

    tc3.metric(
        "RSI",
        value(
            result.get(
                "RSI_14"
            )
        )
    )


    # =====================================================
    # EMA
    # =====================================================

    st.caption(
        "📊 EMA 10 / 20 / 50 / 100 / 200"
    )

    e1, e2, e3, e4, e5 = st.columns(5)

    e1.metric(
        "10",
        money(
            result.get(
                "EMA_10"
            )
        )
    )

    e2.metric(
        "20",
        money(
            result.get(
                "EMA_20"
            )
        )
    )

    e3.metric(
        "50",
        money(
            result.get(
                "EMA_50"
            )
        )
    )

    e4.metric(
        "100",
        money(
            result.get(
                "EMA_100"
            )
        )
    )

    e5.metric(
        "200",
        money(
            result.get(
                "EMA_200"
            )
        )
    )


    st.caption(
        "EMA Alignment: "
        +
        str(
            value(
                result.get(
                    "EMA_ALIGNMENT"
                )
            )
        )
    )


    # =====================================================
    # MOMENTUM INDICATORS
    # =====================================================

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "RSI",
        value(
            result.get(
                "RSI_14"
            )
        )
    )

    m2.metric(
        "MACD",
        value(
            result.get(
                "MACD"
            )
        )
    )

    m3.metric(
        "Histogram",
        value(
            result.get(
                "MACD_HIST"
            )
        )
    )


    # =====================================================
    # SUPERTREND / CPR / PIVOT
    # =====================================================

    st.caption(
        "🧭 Trend / CPR / Pivot"
    )

    s1, s2, s3 = st.columns(3)

    s1.metric(
        "Supertrend",
        money(
            result.get(
                "SUPERTREND"
            )
        )
    )

    s2.metric(
        "Trend",
        value(
            result.get(
                "SUPERTREND_STATUS"
            )
        )
    )

    s3.metric(
        "Pivot",
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

    v1.metric(
        "Volume",
        value(
            result.get(
                "VOLUME"
            )
        )
    )

    v2.metric(
        "Volume Ratio",
        str(
            value(
                result.get(
                    "VOLUME_RATIO"
                )
            )
        )
        + "x"
    )

    v3.metric(
        "Breakout",
        value(
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

    f1.metric(
        "Fundamental",
        f"{fundamental_score:.0f}/100"
    )

    f2.metric(
        "Zone",
        value(
            fundamental.get(
                "FUNDAMENTAL_ZONE"
            )
        )
    )

    f3.metric(
        "Quality",
        pct(
            fundamental.get(
                "DATA_QUALITY_%"
            )
        )
    )


    f4, f5, f6 = st.columns(3)

    f4.metric(
        "Revenue Growth",
        pct(
            fundamental.get(
                "REVENUE_GROWTH_%"
            )
        )
    )

    f5.metric(
        "Profit Growth",
        pct(
            fundamental.get(
                "PROFIT_GROWTH_%"
            )
        )
    )

    f6.metric(
        "EPS Growth",
        pct(
            fundamental.get(
                "EPS_GROWTH_%"
            )
        )
    )


    f7, f8, f9 = st.columns(3)

    f7.metric(
        "ROE",
        pct(
            fundamental.get(
                "ROE_%"
            )
        )
    )

    f8.metric(
        "ROCE",
        pct(
            fundamental.get(
                "ROCE_%"
            )
        )
    )

    f9.metric(
        "Debt/Equity",
        value(
            fundamental.get(
                "DEBT_TO_EQUITY"
            )
        )
    )


    # =====================================================
    # VALUATION
    # =====================================================

    st.caption(
        "💰 Valuation"
    )

    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        "PE",
        value(
            fundamental.get(
                "PE"
            )
        )
    )

    q2.metric(
        "Forward PE",
        value(
            fundamental.get(
                "FORWARD_PE"
            )
        )
    )

    q3.metric(
        "PB",
        value(
            fundamental.get(
                "PB"
            )
        )
    )

    q4.metric(
        "Valuation",
        value(
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

    r1.metric(
        "Risk Score",
        f"{risk_score:.0f}/100"
    )

    r2.metric(
        "Risk Level",
        value(
            result.get(
                "RISK_LEVEL"
            )
        )
    )

    r3.metric(
        "Risk %",
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
        str(
            value(
                result.get(
                    "DATA_DATE"
                )
            )
        )
        +
        " | NSE Status: "
        +
        str(
            value(
                result.get(
                    "STATUS"
                )
            )
        )
        +
        " | Fundamental: "
        +
        str(
            value(
                fundamental.get(
                    "STATUS"
                )
            )
        )
    )


    # =====================================================
    # SHARE / COPY
    # =====================================================

    share_text = f"""
📈 R.S MASTER STOCK GUIDE V2

Stock: {symbol}
CMP: ₹{cmp:.2f}

MASTER SCORE: {master_score}/100
DECISION: {decision_text}
MARKET ZONE: {market_zone}

TECHNICAL: {technical_score:.0f}/100
FUNDAMENTAL: {fundamental_score:.0f}/100
RISK: {risk_score:.0f}/100

🎯 Swing Target: ₹{swing_target:.2f}
🚀 Long Target: ₹{long_target:.2f}
🛑 Stop Loss: ₹{stop_loss:.2f}

🚪 EXIT MATRA: {exit_signal}
Reason: {exit_reason}

📈 RSI: {value(result.get("RSI_14"))}
📊 MACD: {value(result.get("MACD"))}
🚀 Breakout: {value(result.get("VOLUME_BREAKOUT"))}

🏢 Revenue Growth:
{pct(fundamental.get("REVENUE_GROWTH_%"))}

💰 Profit Growth:
{pct(fundamental.get("PROFIT_GROWTH_%"))}

ROE:
{pct(fundamental.get("ROE_%"))}

ROCE:
{pct(fundamental.get("ROCE_%"))}

Data Date:
{value(result.get("DATA_DATE"))}
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

st.header(
    "📊 PORTFOLIO SUMMARY"
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

    b1.metric(
        "Stocks",
        len(scores_df)
    )

    b2.metric(
        "Portfolio Health",
        f"{health}/100"
    )

    b3.metric(
        "BUY",
        buy
    )

    st.caption(
        f"HOLD: {hold} | "
        f"WAIT: {wait} | "
        f"REDUCE/SELL: {sell} | "
        f"EXIT: {exit_count}"
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
