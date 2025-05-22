import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import ta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Trend Predictor", layout="wide")
st.set_option('client.showErrorDetails', True)
st.title("📈 Stock Trend Predictor (XGBoost Edition)")

# === User Inputs ===
ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA):", value="AAPL")
start_date = st.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End date", pd.to_datetime("2024-12-31"))

# === FUNDAMENTALS PREVIEW ===
st.subheader("📊 Fundamental Analysis: Key Financials (from Yahoo Finance)")
ticker_obj = yf.Ticker(ticker)

try:
    info = ticker_obj.info

    col1, col2, col3 = st.columns(3)

    col1.metric("Market Cap", f"${info.get('marketCap', 0) / 1e9:.2f}B")
    col2.metric("P/E Ratio (TTM)", info.get('trailingPE', 'N/A'))
    col3.metric("Forward P/E", info.get('forwardPE', 'N/A'))

    col1.metric("Return on Equity (ROE)", f"{info.get('returnOnEquity', 0)*100:.2f}%")
    col2.metric("Profit Margin", f"{info.get('profitMargins', 0)*100:.2f}%")
    col3.metric("Debt to Equity", info.get('debtToEquity', 'N/A'))

    col1.metric("EPS (TTM)", info.get('trailingEps', 'N/A'))
    col2.metric("Revenue (TTM)", f"${info.get('totalRevenue', 0) / 1e9:.2f}B")
    col3.metric("Free Cash Flow", f"${info.get('freeCashflow', 0) / 1e6:.0f}M")

    # === Long-Term Investment Recommendation ===
    st.subheader("📌 Long-Term Investment Recommendation")
    pe = info.get('trailingPE', None)
    roe = info.get('returnOnEquity', 0)
    pm = info.get('profitMargins', 0)
    fcf = info.get('freeCashflow', 0)
    debt_eq = info.get('debtToEquity', None)

    score = 0
    if pe and pe < 25:
        score += 1
    if roe and roe > 0.15:
        score += 1
    if pm and pm > 0.1:
        score += 1
    if fcf and fcf > 0:
        score += 1
    if debt_eq and float(debt_eq) < 1:
        score += 1

    if score >= 4:
        st.success("🟢 Strong fundamentals — potential long-term buy")
    elif score == 3:
        st.info("🟡 Moderate strength — watchlist candidate")
    else:
        st.warning("🔴 Weak fundamentals — proceed with caution")

except Exception as e:
    st.warning("⚠️ Could not retrieve fundamentals.")
    st.exception(e)

# === Download and Prepare Data ===
data = yf.download(ticker, start=start_date, end=end_date)
data.dropna(inplace=True)

# Feature Engineering
data['Daily_Change_%'] = data['Close'].pct_change() * 100
data['MA_5'] = data['Close'].rolling(window=5).mean()
data['MA_10'] = data['Close'].rolling(window=10).mean()

close_series = pd.Series(data['Close'].values.flatten(), index=data.index)
rsi = ta.momentum.RSIIndicator(close=close_series, window=14)
macd = ta.trend.MACD(close=close_series)

data['RSI'] = rsi.rsi()
data['MACD'] = macd.macd()
data['MACD_Signal'] = macd.macd_signal()

for i in range(1, 6):
    data[f'Close_lag_{i}'] = data['Close'].shift(i)

data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
data.dropna(inplace=True)

features = [
    'Volume', 'Daily_Change_%', 'MA_5', 'MA_10', 'RSI',
    'MACD', 'MACD_Signal'
] + [f'Close_lag_{i}' for i in range(1, 6)]

X = data[features]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === Model Training ===
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

data_test = data.iloc[-len(y_test):].copy()
data_test['Prediction'] = y_pred

with st.expander("📋 Model Performance"):
    st.write(f"Accuracy: **{accuracy_score(y_test, y_pred):.2%}**")
    st.text(classification_report(y_test, y_pred))

with st.expander("📈 Price with Prediction Markers"):
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(data_test.index, data_test['Close'], label="Close Price", color='blue')
    ax1.scatter(data_test.index[data_test['Prediction'] == 1], data_test['Close'][data_test['Prediction'] == 1], color='green', label='Predicted Up')
    ax1.scatter(data_test.index[data_test['Prediction'] == 0], data_test['Close'][data_test['Prediction'] == 0], color='red', label='Predicted Down')
    ax1.legend()
    ax1.grid()
    st.pyplot(fig1)

with st.expander("📌 Trade Signals on Chart"):
    data_test['Signal'] = data_test['Prediction'].shift(1)
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(data_test.index, data_test['Close'], label="Close Price", color='gray')
    buy_signals = (data_test['Signal'] == 1) & (data_test['Signal'].shift(1) != 1)
    ax2.scatter(data_test.index[buy_signals], data_test['Close'][buy_signals], color='green', label='Buy Signal', marker='^')
    sell_signals = (data_test['Signal'] == 0) & (data_test['Signal'].shift(1) == 1)
    ax2.scatter(data_test.index[sell_signals], data_test['Close'][sell_signals], color='red', label='Sell Signal', marker='v')
    ax2.set_title(f"{ticker} Close Price with Trade Signals")
    ax2.legend()
    ax2.grid()
    st.pyplot(fig2)
