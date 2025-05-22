import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

st.set_page_config(page_title="Stock Trend Predictor", layout="wide")
st.set_option('client.showErrorDetails', True)
st.title("📈 Stock Trend Predictor (AutoML Edition)")

# === User Inputs ===
ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA):", value="AAPL")
start_date = st.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End date", pd.to_datetime("2024-12-31"))

# === Company Summary ===
with st.expander("🏢 Company Snapshot"):
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        st.markdown(f"**Name:** {info.get('longName', '-')}")
        st.markdown(f"**Sector:** {info.get('sector', '-')}")
        st.markdown(f"**Market Cap:** {info.get('marketCap', '-'):,}")
        st.markdown(f"**Trailing P/E:** {info.get('trailingPE', '-')}")
        st.markdown(f"**Forward P/E:** {info.get('forwardPE', '-')}")
        st.markdown(f"**PEG Ratio:** {info.get('pegRatio', '-')}")
        st.markdown(f"**EPS (TTM):** {info.get('trailingEps', '-')}")
        st.markdown(f"**Profit Margin:** {info.get('profitMargins', '-')}")
        st.markdown(f"**Revenue Growth:** {info.get('revenueGrowth', '-')}")
        st.markdown(f"**Dividend Yield:** {info.get('dividendYield', '-')}")
        st.markdown(f"**Return on Equity (ROE):** {info.get('returnOnEquity', '-')}")
        st.markdown(f"**Free Cash Flow:** {info.get('freeCashflow', '-'):,}")
        st.markdown(f"**Operating Margins:** {info.get('operatingMargins', '-')}")
        st.markdown(f"**Insider Ownership:** {info.get('heldPercentInsiders', '-')}")
    except:
        st.warning("Unable to load company fundamentals.")

# === Historical Valuation Charts ===
with st.expander("📊 Historical Valuation Metrics"):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5y")

        if t.earnings.empty or t.balance_sheet.empty:
            raise ValueError("Missing earnings or balance sheet data")

        earnings = t.earnings
        revenue = earnings['Revenue'].reindex(hist.index.year).ffill().values
        eps = earnings['Earnings'].reindex(hist.index.year).ffill().values

        book = t.balance_sheet.loc['TotalStockholderEquity']
        shares = info.get('sharesOutstanding', 1)
        book_value = book / shares

        pe_ratio = hist['Close'] / eps
        ps_ratio = hist['Close'] / (revenue / 1e9)
        pb_ratio = hist['Close'] / book_value.ffill().reindex(hist.index, method='ffill')

        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        axs[0].plot(hist.index, pe_ratio, label='P/E Ratio')
        axs[0].axhline(pe_ratio.mean(), color='gray', linestyle='--', label='Avg P/E')
        axs[0].legend()
        axs[0].set_title("Price-to-Earnings Ratio")

        axs[1].plot(hist.index, ps_ratio, label='P/S Ratio', color='purple')
        axs[1].axhline(ps_ratio.mean(), color='gray', linestyle='--', label='Avg P/S')
        axs[1].legend()
        axs[1].set_title("Price-to-Sales Ratio")

        axs[2].plot(hist.index, pb_ratio, label='P/B Ratio', color='green')
        axs[2].axhline(pb_ratio.mean(), color='gray', linestyle='--', label='Avg P/B')
        axs[2].legend()
        axs[2].set_title("Price-to-Book Ratio")

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"Could not generate valuation charts: {e}")
