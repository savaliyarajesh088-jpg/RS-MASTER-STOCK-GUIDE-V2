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
# MOBILE + FULL COLOR UI
# =========================================================

st.markdown(
    """
    <style>

    /* =========================
       GLOBAL
    ========================= */

    .stApp {
        background:
            radial-gradient(
                circle at top,
                #171717 0%,
                #080808 42%,
                #030303 100%
            );
        color: #ffffff;
    }

    .block-container {
        max-width: 1250px;
        padding:
            0.35rem
            0.30rem
            1rem
            0.30rem;
    }

    header {
        visibility: hidden;
        height: 0;
    }

    footer {
        visibility: hidden;
    }


    /* =========================
       TITLE
    ========================= */

    .main-title {
        text-align: center;
        font-size: 1.65rem;
        font-weight: 950;
        letter-spacing: 0.5px;
        margin-top: 0;
        margin-bottom: 0;
        background:
            linear-gradient(
                90deg,
                #00e676,
                #00b0ff,
                #ffea00,
                #ff4081
            );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sub-title {
        text-align: center;
        font-size: 0.72rem;
        color: #a8a8a8;
        margin-bottom: 0.35rem;
    }


    /* =========================
       STOCK CARD
    ========================= */

    .stock-card {
        border-radius: 20px;
        padding: 12px;
        margin-top: 8px;
        margin-bottom: 15px;

        background:
            linear-gradient(
                145deg,
                rgba(30,30,30,0.98),
                rgba(8,8,8,0.98)
            );

        border:
            1px solid
            rgba(255,255,255,0.12);

        box-shadow:
            0 8px 30px
            rgba(0,0,0,0.35);
    }


    /* =========================
       SECTION
    ========================= */

    .section-title {
        font-size: 0.96rem;
        font-weight: 950;
        margin-top: 12px;
        margin-bottom: 6px;
        padding-left: 3px;
    }


    /* =========================
       SCORE CARD
    ========================= */

    .score-card {
        border-radius: 18px;
        padding: 12px;
        text-align: center;

        background:
            linear-gradient(
                135deg,
                #151515,
                #090909
            );

        border:
            1px solid
            rgba(255,255,255,0.13);
    }

    .score-title {
        font-size: 0.68rem;
        color: #9e9e9e;
        font-weight: 900;
        letter-spacing: 0.5px;
    }

    .score-number {
        font-size: 2.0rem;
        font-weight: 950;
        line-height: 1.1;
    }


    /* =========================
       SIGNAL
    ========================= */

    .signal-box {
        border-radius: 16px;
        padding: 11px;
        margin-top: 8px;
        text-align: center;
        font-size: 1.05rem;
        font-weight: 950;
        border: 1px solid rgba(255,255,255,0.13);
    }

    .signal-buy {
        background:
            linear-gradient(
                135deg,
                rgba(0,230,118,0.30),
                rgba(0,150,70,0.08)
            );
        border-color: rgba(0,230,118,0.55);
        color: #00ff88;
    }

    .signal-hold {
        background:
            linear-gradient(
                135deg,
                rgba(0,176,255,0.30),
                rgba(0,100,200,0.08)
            );
        border-color: rgba(0,176,255,0.55);
        color: #40c4ff;
    }

    .signal-wait {
        background:
            linear-gradient(
                135deg,
                rgba(255,235,59,0.28),
                rgba(200,150,0,0.08)
            );
        border-color: rgba(255,235,59,0.55);
        color: #ffe600;
    }

    .signal-reduce {
        background:
            linear-gradient(
                135deg,
                rgba(255,152,0,0.30),
                rgba(180,80,0,0.08)
            );
        border-color: rgba(255,152,0,0.55);
        color: #ffab19;
    }

    .signal-exit {
        background:
            linear-gradient(
                135deg,
                rgba(255,48,79,0.32),
                rgba(150,0,30,0.08)
            );
        border-color: rgba(255,48,79,0.60);
        color: #ff405c;
    }


    /* =========================
       DATA BOX
    ========================= */

    .data-box {
        border-radius: 14px;
        padding: 9px 8px;
        margin: 4px 0;

        background:
            rgba(255,255,255,0.035);

        border:
            1px solid
            rgba(255,255,255,0.08);

        text-align: center;
    }

    .data-label {
        font-size: 0.60rem;
        color: #8e8e8e;
        font-weight: 850;
        text-transform: uppercase;
    }

    .data-value {
        font-size: 0.90rem;
        font-weight: 900;
        margin-top: 2px;
    }


    /* =========================
       TARGET COLORS
    ========================= */

    .target-swing {
        border-left: 4px solid #00e676;
    }

    .target-long {
        border-left: 4px solid #00b0ff;
    }

    .target-sl {
        border-left: 4px solid #ff304f;
    }


    /* =========================
       MOBILE
    ========================= */

    @media (max-width: 640px) {

        .block-container {
            padding-left: 0.18rem;
            padding-right: 0.18rem;
        }

        .main-title {
            font-size: 1.25rem;
        }

        .sub-title {
            font-size: 0.58rem;
        }

        .score-number {
            font-size: 1.65rem;
        }

        .signal-box {
            font-size: 0.92rem;
            padding: 9px;
        }

        .section-title {
            font-size: 0.86rem;
        }

        .data-value {
            font-size: 0.78rem;
        }

        .data-label {
            font-size: 0.54rem;
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

        number = float(value)

        if number == 0:
            return "—"

        return f"₹{number:,.2f}"

    except Exception:

        return "—"


def num2(value):

    try:

        if value is None or pd.isna(value):
            return "—"

        return f"{float(value):.2f}"

    except Exception:

        return "—"


def pct(value):

    try:

        if value is None or pd.isna(value):
            return "—"

        return f"{float(value):.2f}%"

    except Exception:

        return "—"


def integer(value):

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

    symbol = str(
        symbol
    ).strip().upper()

    if symbol.endswith(".NS"):
        return symbol[:-3]

    return symbol


# =========================================================
# SYMBOL
# =========================================================

def signal_symbol(decision):

    return {
        "BUY": "🟢📈",
        "HOLD": "🔵🛡️",
        "WAIT": "🟡⏳",
        "SELL": "🟠⚠️",
        "EXIT": "🔴🚪"
    }.get(
        decision,
        "⚪"
    )


def decision_class(decision):

    return {
        "BUY": "signal-buy",
        "HOLD": "signal-hold",
        "WAIT": "signal-wait",
        "SELL": "signal-reduce",
        "EXIT": "signal-exit"
    }.get(
        decision,
        "signal-wait"
    )


def zone_symbol(zone):

    text = str(zone).upper()

    if "BULL" in text or "POSITIVE" in text or "STRONG" in text:
        return "🟢📈"

    if "BEAR" in text or "WEAK" in text or "POOR" in text:
        return "🔴📉"

    if "AVERAGE" in text or "NEUTRAL" in text:
        return "🟡↔️"

    return "⚪"


def alignment_symbol(alignment):

    text = str(alignment).upper()

    if "BULL" in text:
        return "🟢📈 BULLISH"

    if "BEAR" in text:
        return "🔴📉 BEARISH"

    if "MIX" in text:
        return "🟡↔️ MIXED"

    return "⚪ NO DATA"


def trend_symbol(trend):

    text = str(trend).upper()

    if "BULL" in text:
        return "🟢🐂 BULLISH"

    if "BEAR" in text:
        return "🔴🐻 BEARISH"

    return "🟡↔️ NEUTRAL"


def breakout_symbol(value):

    text = str(value).upper()

    if text in ["YES", "TRUE", "1", "BREAKOUT"]:
        return "🟢🚀 YES"

    if text in ["NO", "FALSE", "0"]:
        return "🔴 NO"

    return "⚪ —"


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
# YFINANCE CHART CACHE
# =========================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def download_chart_data(symbol):

    ticker_symbol = (
        clean_symbol(symbol)
        + ".NS"
    )

    try:

        data = yf.download(
            ticker_symbol,
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        return data

    except Exception:

        return pd.DataFrame()


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

    data = download_chart_data(
        symbol
    )

    if data.empty:

        st.warning(
            "⚠️ Yahoo Finance data હાલમાં unavailable / rate-limit છે. "
            "Stock analysis ચાલુ રહેશે; chart પછી ફરી load થશે."
        )

        return

    try:

        # -------------------------------------------------
        # MULTIINDEX
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

        if not all(
            col in data.columns
            for col in required
        ):

            st.warning(
                "⚠️ Chart data incomplete."
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
                "⚠️ Historical chart data ઓછો છે."
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
            ] = close.ewm(
                span=period,
                adjust=False
            ).mean()

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

        rs = (
            gain /
            loss.replace(
                0,
                np.nan
            )
        )

        data["RSI"] = (
            100 -
            (
                100 /
                (1 + rs)
            )
        )

        # -------------------------------------------------
        # MACD
        # -------------------------------------------------

        ema12 = close.ewm(
            span=12,
            adjust=False
        ).mean()

        ema26 = close.ewm(
            span=26,
            adjust=False
        ).mean()

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
        # VOLUME
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
        # TARGET FALLBACK
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
            and
            rsi >= 55
            and
            macd > macd_signal
            and
            vol_ratio >= 1.2
        )

        # -------------------------------------------------
        # FIGURE
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
        # EMA
        # -------------------------------------------------

        ema_settings = [
            ("EMA10", "EMA 10", 1.1),
            ("EMA20", "EMA 20", 1.1),
            ("EMA50", "EMA 50", 1.3),
            ("EMA100", "EMA 100", 1.4),
            ("EMA200", "EMA 200", 1.7)
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
        # BREAKOUT
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
                        size=9,
                        symbol="triangle-up"
                    )
                ),
                row=1,
                col=1
            )

        # -------------------------------------------------
        # MOMENTUM
        # -------------------------------------------------

        if momentum_ready:

            fig.add_trace(
                go.Scatter(
                    x=[
                        data.index[-1]
                    ],
                    y=[
                        cmp
                    ],
                    mode="markers+text",
                    text=[
                        "⚡ MOMENTUM"
                    ],
                    textposition="top center",
                    name="⚡ Momentum"
                ),
                row=1,
                col=1
            )

        # -------------------------------------------------
        # LEVELS
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
        # RANGE
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

            height=650,

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
            f"📍 CMP {money(cmp)} | "
            f"🛑 SL {money(stop_loss)} | "
            f"🎯 Swing {money(swing_target)} | "
            f"🚀 Long {money(long_target)}"
        )

    except Exception as error:

        st.warning(
            f"📊 Chart unavailable: "
            f"{type(error).__name__}"
        )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        📈 R.S MASTER STOCK GUIDE V2
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
        NSE • Technical • Fundamental • Momentum •
        Breakout • Exit Matra • Risk • Target
    </div>
    """,
    unsafe_allow_html=True
)

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
            "BUY / વધારો"
        )

    elif master_score >= 60:

        decision = "HOLD"

        decision_text = (
            "HOLD / જાળવો"
        )

    elif master_score >= 45:

        decision = "WAIT"

        decision_text = (
            "WAIT / રાહ જુઓ"
        )

    elif master_score >= 30:

        decision = "SELL"

        decision_text = (
            "REDUCE / ઘટાડો"
        )

    else:

        decision = "EXIT"

        decision_text = (
            "EXIT / બહાર નીકળો"
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

        market_zone = "🟡 NEUTRAL"

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

    score_color = "#00e676"

    if master_score < 30:
        score_color = "#ff304f"

    elif master_score < 45:
        score_color = "#ff9800"

    elif master_score < 60:
        score_color = "#ffea00"

    elif master_score < 75:
        score_color = "#00b0ff"


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
                {master_score}
                <span
                    style="
                    font-size:0.8rem;
                    opacity:0.55;
                    "
                >
                    /100
                </span>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # REAL STREAMLIT PROGRESS BAR
    st.progress(
        min(
            max(
                int(master_score),
                0
            ),
            100
        )
    )


    # =====================================================
    # DECISION SIGNAL
    # =====================================================

    dclass = decision_class(
        decision
    )

    dsymbol = signal_symbol(
        decision
    )

    st.markdown(
        f"""
        <div class="signal-box {dclass}">
            {dsymbol}
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
        <div class="signal-box signal-reduce">
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

        exit_icon = {
            "EXIT": "🔴",
            "REDUCE": "🟠",
            "HOLD": "🟢",
            "BOOK": "💰"
        }.get(
            exit_signal,
            "⚪"
        )

        st.metric(
            "EXIT SIGNAL",
            f"{exit_icon} {exit_signal}"
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
            '<div class="data-box target-swing">'
            '<div class="data-label">🎯 SWING</div>'
            f'<div class="data-value">{money(swing_target)}</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with t2:

        st.markdown(
            '<div class="data-box target-long">'
            '<div class="data-label">🚀 LONG</div>'
            f'<div class="data-value">{money(long_target)}</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with t3:

        st.markdown(
            '<div class="data-box target-sl">'
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

        tech_zone = display_value(
            result.get(
                "TECHNICAL_ZONE"
            )
        )

        st.metric(
            "ZONE",
            f"{zone_symbol(tech_zone)} {tech_zone}"
        )

    with tc3:

        st.metric(
            "RSI",
            num2(
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

    with e1:
        st.metric(
            "10",
            money(
                result.get(
                    "EMA_10"
                )
            )
        )

    with e2:
        st.metric(
            "20",
            money(
                result.get(
                    "EMA_20"
                )
            )
        )

    with e3:
        st.metric(
            "50",
            money(
                result.get(
                    "EMA_50"
                )
            )
        )

    with e4:
        st.metric(
            "100",
            money(
                result.get(
                    "EMA_100"
                )
            )
        )

    with e5:
        st.metric(
            "200",
            money(
                result.get(
                    "EMA_200"
                )
            )
        )


    alignment = display_value(
        result.get(
            "EMA_ALIGNMENT"
        )
    )

    st.markdown(
        f"""
        <div class="data-box">
            <div class="data-label">
                EMA ALIGNMENT
            </div>

            <div class="data-value">
                {alignment_symbol(alignment)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # RSI / MACD
    # =====================================================

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "RSI 14",
            num2(
                result.get(
                    "RSI_14"
                )
            )
        )

    with m2:

        st.metric(
            "MACD",
            num2(
                result.get(
                    "MACD"
                )
            )
        )

    with m3:

        st.metric(
            "HISTOGRAM",
            num2(
                result.get(
                    "MACD_HIST"
                )
            )
        )


    # =====================================================
    # TREND
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🧭 TREND / CPR / PIVOT'
        '</div>',
        unsafe_allow_html=True
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

        trend = display_value(
            result.get(
                "SUPERTREND_STATUS"
            )
        )

        st.metric(
            "TREND",
            trend_symbol(
                trend
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
            integer(
                result.get(
                    "VOLUME"
                )
            )
        )

    with v2:

        ratio = safe_float(
            result.get(
                "VOLUME_RATIO"
            )
        )

        st.metric(
            "VOLUME RATIO",
            f"{ratio:.2f}x"
        )

    with v3:

        st.metric(
            "BREAKOUT",
            breakout_symbol(
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

        fzone = display_value(
            fundamental.get(
                "FUNDAMENTAL_ZONE"
            )
        )

        st.metric(
            "ZONE",
            f"{zone_symbol(fzone)} {fzone}"
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
            num2(
                fundamental.get(
                    "DEBT_TO_EQUITY"
                )
            )
        )


    # =====================================================
    # VALUATION
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '💰 VALUATION'
        '</div>',
        unsafe_allow_html=True
    )

    q1, q2, q3, q4 = st.columns(4)

    with q1:

        st.metric(
            "PE",
            num2(
                fundamental.get(
                    "PE"
                )
            )
        )

    with q2:

        st.metric(
            "FORWARD PE",
            num2(
                fundamental.get(
                    "FORWARD_PE"
                )
            )
        )

    with q3:

        st.metric(
            "PB",
            num2(
                fundamental.get(
                    "PB"
                )
            )
        )

    with q4:

        valuation = display_value(
            fundamental.get(
                "VALUATION_ZONE"
            )
        )

        st.metric(
            "VALUATION",
            f"{zone_symbol(valuation)} {valuation}"
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

        risk_level = display_value(
            result.get(
                "RISK_LEVEL"
            )
        )

        st.metric(
            "RISK LEVEL",
            risk_level
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
    # STATUS
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
CMP: ₹{cmp:.2f}

🏦 MASTER SCORE: {master_score}/100

🎯 DECISION:
{signal_symbol(decision)} {decision_text}

MARKET ZONE:
{market_zone}

📈 TECHNICAL:
{technical_score:.0f}/100

🏢 FUNDAMENTAL:
{fundamental_score:.0f}/100

🛡️ RISK:
{risk_score:.0f}/100

🎯 SWING TARGET:
₹{swing_target:.2f}

🚀 LONG TARGET:
₹{long_target:.2f}

🛑 STOP LOSS:
₹{stop_loss:.2f}

🚪 EXIT MATRA:
{exit_signal}

Reason:
{exit_reason}

📊 RSI:
{num2(result.get("RSI_14"))}

📊 MACD:
{num2(result.get("MACD"))}

📊 MACD HISTOGRAM:
{num2(result.get("MACD_HIST"))}

📈 EMA ALIGNMENT:
{alignment_symbol(
    result.get("EMA_ALIGNMENT")
)}

🧭 TREND:
{trend_symbol(
    result.get("SUPERTREND_STATUS")
)}

🚀 BREAKOUT:
{breakout_symbol(
    result.get("VOLUME_BREAKOUT")
)}

🏢 REVENUE GROWTH:
{pct(
    fundamental.get("REVENUE_GROWTH_%")
)}

💰 PROFIT GROWTH:
{pct(
    fundamental.get("PROFIT_GROWTH_%")
)}

📈 EPS GROWTH:
{pct(
    fundamental.get("EPS_GROWTH_%")
)}

ROE:
{pct(
    fundamental.get("ROE_%")
)}

ROCE:
{pct(
    fundamental.get("ROCE_%")
)}

📅 DATA DATE:
{display_value(
    result.get("DATA_DATE")
)}
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


    # =====================================================
    # SUMMARY CARDS
    # =====================================================

    b1, b2, b3 = st.columns(3)

    with b1:

        st.metric(
            "📊 STOCKS",
            len(scores_df)
        )

    with b2:

        st.metric(
            "❤️ PORTFOLIO HEALTH",
            f"{health}/100"
        )

    with b3:

        st.metric(
            "🟢 BUY",
            buy
        )


    st.markdown(
        f"""
        <div class="signal-box signal-reduce">

        🟢 BUY: {buy}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        🔵 HOLD: {hold}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        🟡 WAIT: {wait}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        🟠 REDUCE: {sell}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        🔴 EXIT: {exit_count}

        </div>
        """,
        unsafe_allow_html=True
    )


    # =====================================================
    # SUMMARY TABLE
    # =====================================================

    summary_display = scores_df.copy()

    summary_display["SIGNAL"] = (
        summary_display["DECISION"]
        .map(signal_symbol)
    )

    summary_display = summary_display[
        [
            "SYMBOL",
            "MASTER_SCORE",
            "SIGNAL",
            "DECISION",
            "EXIT_MATRA",
            "ZONE"
        ]
    ]

    st.dataframe(
        summary_display,
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
