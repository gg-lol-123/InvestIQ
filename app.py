"""
app.py
Streamlit dashboard for InvestIQ
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Local modules
from data_fetch import (
    fetch_stock_data,
    get_stock_info,
    normalize_ticker
)

from indicators import add_indicators

from model import (
    train_model,
    predict,
    get_model_metrics
)

from explain import (
    explain_latest,
    plot_shap_bar,
    plot_shap_latest
)

from recommendation import (
    generate_recommendation
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="InvestIQ",
    page_icon="📈",
    layout="wide"
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

with st.sidebar:

    st.title("📈 InvestIQ")

    st.caption(
        "AI Investment Decision Support"
    )

    st.markdown("---")

    ticker_input = st.text_input(
        "NSE Stock Ticker",
        value="RELIANCE"
    )

    period = st.selectbox(
        "Historical Period",
        ["6mo", "1y", "2y"],
        index=1
    )

    analyze_btn = st.button(
        "⚡ Analyze"
    )

    st.markdown("---")

    st.caption(
        "Educational prototype only. "
        "Not financial advice."
    )

# ---------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------

st.title(
    "📊 AI Investment Decision Support"
)

st.caption(
    "Powered by XGBoost + SHAP Explainability"
)

# ---------------------------------------------------
# WAIT FOR BUTTON
# ---------------------------------------------------

if not analyze_btn:

    st.info(
        "Enter a stock ticker and click Analyze."
    )

    st.stop()

# ---------------------------------------------------
# FETCH DATA
# ---------------------------------------------------

ticker_norm = normalize_ticker(
    ticker_input
)

with st.spinner(
    f"Fetching data for {ticker_norm}..."
):

    try:

        raw_df = fetch_stock_data(
            ticker_input,
            period=period
        )

        info = get_stock_info(
            ticker_input
        )

    except Exception as e:

        st.error(str(e))

        st.stop()

# ---------------------------------------------------
# INDICATORS
# ---------------------------------------------------

with st.spinner(
    "Computing technical indicators..."
):

    enriched_df = add_indicators(
        raw_df
    )

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

with st.spinner(
    "Training prediction model..."
):

    (
        model,
        scaler,
        acc,
        feature_cols
    ) = train_model(enriched_df)

    pred_result = predict(
        model,
        scaler,
        enriched_df
    )

    metrics = get_model_metrics(
        model,
        scaler,
        enriched_df
    )

# ---------------------------------------------------
# SHAP
# ---------------------------------------------------

with st.spinner(
    "Running explainability engine..."
):

    shap_result = explain_latest(
        model,
        scaler,
        enriched_df
    )

# ---------------------------------------------------
# RECOMMENDATION
# ---------------------------------------------------

with st.spinner(
    "Generating recommendation..."
):

    rec = generate_recommendation(
        pred_result,
        enriched_df
    )

# ---------------------------------------------------
# COMPANY INFO
# ---------------------------------------------------

st.markdown(
    f"## {info.get('name', ticker_norm)}"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Sector",
    info.get("sector", "N/A")
)

col2.metric(
    "52W High",
    f"₹{info.get('52w_high', 'N/A')}"
)

col3.metric(
    "52W Low",
    f"₹{info.get('52w_low', 'N/A')}"
)

pe = info.get("pe_ratio")

if pe:
    pe_display = f"{pe:.1f}"
else:
    pe_display = "N/A"

col4.metric(
    "P/E Ratio",
    pe_display
)

st.markdown("---")

# ---------------------------------------------------
# PRICE CHART
# ---------------------------------------------------

col_chart, col_pred = st.columns([2, 1])

with col_chart:

    st.subheader(
        "📈 Price History"
    )

    fig_price, ax = plt.subplots(
        figsize=(9, 4)
    )

    close = enriched_df["Close"]

    sma20 = enriched_df["SMA_20"]

    sma50 = enriched_df["SMA_50"]

    ax.plot(
        close.index,
        close.values,
        label="Close"
    )

    ax.plot(
        sma20.index,
        sma20.values,
        linestyle="--",
        label="SMA 20"
    )

    ax.plot(
        sma50.index,
        sma50.values,
        linestyle="--",
        label="SMA 50"
    )

    ax.legend()

    plt.tight_layout()

    st.pyplot(fig_price)

    plt.close(fig_price)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

with col_pred:

    st.subheader(
        "🤖 Prediction"
    )

    direction = pred_result["direction"]

    if direction == "UP":
        emoji = "⬆️"
    else:
        emoji = "⬇️"

    st.metric(
        "Direction",
        f"{emoji} {direction}"
    )

    st.metric(
        "Confidence",
        f"{pred_result['confidence']:.1%}"
    )

    st.metric(
        "Probability Up",
        f"{pred_result['prob_up']:.1%}"
    )

    st.metric(
        "Probability Down",
        f"{pred_result['prob_down']:.1%}"
    )

    st.metric(
        "Model Accuracy",
        f"{metrics['accuracy']}%"
    )

st.markdown("---")

# ---------------------------------------------------
# RECOMMENDATION + RISK
# ---------------------------------------------------

col_rec, col_risk = st.columns(2)

with col_rec:

    st.subheader(
        "💡 Recommendation"
    )

    st.success(
        f"{rec['emoji']} "
        f"{rec['action']}"
    )

    st.write(
        rec["rationale"]
    )

with col_risk:

    st.subheader(
        "⚠️ Risk Metrics"
    )

    rm = rec["risk_metrics"]

    st.metric(
        "1Y Return",
        f"{rm['total_return_1y']}%"
    )

    st.metric(
        "Annualized Return",
        f"{rm['annualized_return']}%"
    )

    st.metric(
        "Sharpe Ratio",
        rm["sharpe_ratio"]
    )

    st.metric(
        "Max Drawdown",
        f"{rm['max_drawdown']}%"
    )

    st.metric(
        "VaR (95%)",
        f"{rm['var_95']}%"
    )

    st.metric(
        "Volatility",
        f"{rm['current_volatility']}%"
    )

st.markdown("---")

# ---------------------------------------------------
# SHAP EXPLAINABILITY
# ---------------------------------------------------

st.subheader(
    "🔍 Explainable AI"
)

col_shap1, col_shap2 = st.columns(2)

with col_shap1:

    st.caption(
        "Global Feature Importance"
    )

    fig_bar = plot_shap_bar(
        shap_result["shap_values"]
    )

    st.pyplot(fig_bar)

    plt.close(fig_bar)

with col_shap2:

    st.caption(
        "Latest Prediction Drivers"
    )

    fig_local = plot_shap_latest(
        shap_result
    )

    st.pyplot(fig_local)

    plt.close(fig_local)

# ---------------------------------------------------
# EXPLANATION
# ---------------------------------------------------

st.subheader(
    "📝 Plain-English Explanation"
)

st.info(
    shap_result["plain_english"]
)

st.markdown("---")

# ---------------------------------------------------
# DATA TABLE
# ---------------------------------------------------

with st.expander(
    "📐 Technical Indicators"
):

    recent = enriched_df[
        [
            "Close",
            "RSI",
            "MACD",
            "MACD_Signal",
            "SMA_20",
            "Volatility"
        ]
    ].tail(20)

    st.dataframe(
        recent
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "InvestIQ — AI Investment "
    "Decision Support Prototype "
    "| Final Year Project"
)