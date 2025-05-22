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
    except:
        st.warning("Unable to load company fundamentals.")

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

# === AutoML Logic ===
models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='lbfgs')
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_copy = data.iloc[-len(y_test):].copy()
    test_copy['Prediction'] = y_pred
    test_copy['Signal'] = test_copy['Prediction'].shift(1)
    test_copy['Daily Return'] = test_copy['Close'].pct_change()
    test_copy['Strategy Return'] = test_copy['Daily Return'] * test_copy['Signal']

    cumulative_strategy = (1 + test_copy['Strategy Return'].fillna(0)).cumprod()
    sharpe = np.sqrt(252) * test_copy['Strategy Return'].mean() / test_copy['Strategy Return'].std()

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Sharpe": sharpe,
        "Final Value": cumulative_strategy.iloc[-1],
        "Predictions": y_pred
    })

# Select best by Sharpe Ratio
best_model = max(results, key=lambda x: x['Sharpe'])

data_test = data.iloc[-len(y_test):].copy()
data_test['Prediction'] = best_model['Predictions']

st.success(f"✅ Best model: {best_model['Model']} (Sharpe: {best_model['Sharpe']:.2f})")

with st.expander("📋 Model Performance"):
    st.write(f"Model Used: **{best_model['Model']}**")
    st.write(f"Accuracy: **{best_model['Accuracy']:.2%}**")

with st.expander("📈 Price with Prediction Markers"):
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(data_test.index, data_test['Close'], label="Close Price", color='blue')
    ax1.scatter(data_test.index[data_test['Prediction'] == 1], data_test['Close'][data_test['Prediction'] == 1], color='green', label='Predicted Up')
    ax1.scatter(data_test.index[data_test['Prediction'] == 0], data_test['Close'][data_test['Prediction'] == 0], color='red', label='Predicted Down')
    ax1.legend()
    ax1.grid()
    st.pyplot(fig1)

with st.expander("📊 Quant Backtest Metrics"):
    data_test['Signal'] = data_test['Prediction'].shift(1)
    data_test['Daily Return'] = data_test['Close'].pct_change()
    data_test['Strategy Return'] = data_test['Daily Return'] * data_test['Signal']

    cumulative_strategy = (1 + data_test['Strategy Return'].fillna(0)).cumprod()
    cumulative_market = (1 + data_test['Daily Return'].fillna(0)).cumprod()

    sharpe = np.sqrt(252) * data_test['Strategy Return'].mean() / data_test['Strategy Return'].std()

    st.metric("📈 Strategy Sharpe Ratio", f"{sharpe:.2f}")
    st.line_chart(pd.DataFrame({
        "Strategy": cumulative_strategy,
        "Buy & Hold": cumulative_market
    }))

    st.dataframe(pd.DataFrame({
        "Final Strategy Value ($1)": [cumulative_strategy.iloc[-1]],
        "Final Market Value ($1)": [cumulative_market.iloc[-1]],
        "Annualized Volatility": [data_test['Strategy Return'].std() * np.sqrt(252)]
    }).T.rename(columns={0: "Value"}))

with st.expander("📌 Trade Signals on Chart"):
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(data_test.index, data_test['Close'], label="Close Price", color='gray')
    buy_signals = (data_test['Signal'] == 1) & (data_test['Signal'].shift(1) != 1)
    sell_signals = (data_test['Signal'] == 0) & (data_test['Signal'].shift(1) == 1)
    ax2.scatter(data_test.index[buy_signals], data_test['Close'][buy_signals], color='green', label='Buy Signal', marker='^')
    ax2.scatter(data_test.index[sell_signals], data_test['Close'][sell_signals], color='red', label='Sell Signal', marker='v')
    ax2.set_title(f"{ticker} Close Price with Trade Signals")
    ax2.legend()
    ax2.grid()
    st.pyplot(fig2)
