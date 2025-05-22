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
st.title("📈 Stock Trend Predictor (XGBoost + Strategy Analysis)")

# === User Inputs ===
ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA):", value="AAPL")
start_date = st.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End date", pd.to_datetime("2024-12-31"))
initial_cash = st.number_input("💵 Initial investment ($)", min_value=1000, max_value=1000000, value=10000, step=1000)

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

# === Expanders for Clean UI ===
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

with st.expander("🔍 Feature Importance"):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    st.bar_chart(feature_importance_df.set_index('Feature'))

with st.expander("💰 Strategy Backtest (Model vs Buy & Hold)"):
    cash = initial_cash
    position = 0
    strategy_values = []
    buy_and_hold_values = []
    data_test['Signal'] = data_test['Prediction'].shift(1)
    buy_and_hold_shares = initial_cash / data_test['Close'].iloc[0]

    for i in range(1, len(data_test)):
        price_today = data_test['Close'].iloc[i]
        signal = data_test['Signal'].iloc[i]
        if signal == 1 and position == 0:
            position = cash / price_today
            cash = 0
        elif signal == 0 and position > 0:
            cash = position * price_today
            position = 0
        total = cash + position * price_today
        strategy_values.append(total)
        bh_total = buy_and_hold_shares * price_today
        buy_and_hold_values.append(bh_total)

    final_strategy = strategy_values[-1]
    final_bh = buy_and_hold_values[-1]
    col1, col2 = st.columns(2)
    col1.metric("📈 Strategy Final Value", f"${final_strategy:,.2f}")
    col2.metric("📊 Buy & Hold Final Value", f"${final_bh:,.2f}")

    returns_df = pd.DataFrame({
        "Strategy": strategy_values,
        "Buy & Hold": buy_and_hold_values
    }, index=data_test.index[1:])
    st.line_chart(returns_df)

with st.expander("📌 Trade Signals on Chart"):
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
