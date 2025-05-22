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

# Streamlit setup
st.set_page_config(page_title="Stock Trend Predictor (AutoML Edition)", layout="wide")
st.set_option('client.showErrorDetails', True)
st.title("📈 Stock Trend Predictor (AutoML Edition)")

# === User Inputs ===
ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA):", value="AAPL")
start_date = st.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End date", pd.to_datetime("2024-12-31"))

# === Company Snapshot ===
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
    except Exception:
        st.warning("Unable to load company fundamentals.")

# === Historical Valuation Metrics ===
with st.expander("📊 Historical Valuation Metrics"):
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5y")

        # Safe check for earnings and balance sheet
        if t.earnings is None or t.balance_sheet is None or t.earnings.empty or t.balance_sheet.empty:
            raise ValueError("Missing earnings or balance sheet data")

        # Extract financials
        earnings = t.earnings
        revenue = earnings['Revenue'].reindex(hist.index.year).ffill().values
        eps = earnings['Earnings'].reindex(hist.index.year).ffill().values

        balance_sheet = t.balance_sheet
        book = balance_sheet.loc['TotalStockholderEquity']
        shares = info.get('sharesOutstanding', 1)
        book_value = (book / shares).reindex(hist.index.year).ffill().values

        # Calculate ratios
        pe_ratio = hist['Close'] / eps
        ps_ratio = hist['Close'] / (revenue / 1e9)
        pb_ratio = hist['Close'] / book_value

        # Plot ratios with average lines
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        axs[0].plot(hist.index, pe_ratio, label='P/E Ratio')
        axs[0].axhline(np.nanmean(pe_ratio), color='gray', linestyle='--', label='Avg P/E')
        axs[0].legend()
        axs[0].set_title("Price-to-Earnings Ratio")

        axs[1].plot(hist.index, ps_ratio, label='P/S Ratio', color='purple')
        axs[1].axhline(np.nanmean(ps_ratio), color='gray', linestyle='--', label='Avg P/S')
        axs[1].legend()
        axs[1].set_title("Price-to-Sales Ratio")

        axs[2].plot(hist.index, pb_ratio, label='P/B Ratio', color='green')
        axs[2].axhline(np.nanmean(pb_ratio), color='gray', linestyle='--', label='Avg P/B')
        axs[2].legend()
        axs[2].set_title("Price-to-Book Ratio")

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"Could not generate valuation charts: {e}")

# === Download and Prepare Stock Data ===
raw = yf.download(ticker, start=start_date, end=end_date)
data = raw[['Close']].copy()
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
    'Volume' if 'Volume' in raw.columns else 'Close',
    'Daily_Change_%', 'MA_5', 'MA_10', 'RSI',
    'MACD', 'MACD_Signal'
] + [f'Close_lag_{i}' for i in range(1, 6)]

X = data[features]
if 'Volume' not in raw.columns:
    X = X.drop(columns=['Volume'])
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
    test_df = data.iloc[-len(y_test):].copy()
    test_df['Prediction'] = y_pred
    test_df['Signal'] = test_df['Prediction'].shift(1)
    test_df['Daily Return'] = test_df['Close'].pct_change()
    test_df['Strategy Return'] = test_df['Daily Return'] * test_df['Signal']

    cum_strat = (1 + test_df['Strategy Return'].fillna(0)).cumprod()
    sharpe = np.sqrt(252) * test_df['Strategy Return'].mean() / test_df['Strategy Return'].std()

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Sharpe": sharpe,
        "Final Value": cum_strat.iloc[-1],
        "Predictions": y_pred
    })

best = max(results, key=lambda x: x['Sharpe'])

# Prepare final test data
final_df = data.iloc[-len(y_test):].copy()
final_df['Prediction'] = best['Predictions']

st.success(f"✅ Best model: {best['Model']} (Sharpe: {best['Sharpe']:.2f})")

# Model Performance
with st.expander("📋 Model Performance"):
    st.write(f"Model Used: **{best['Model']}**")
    st.write(f"Accuracy: **{best['Accuracy']:.2%}**")

# Price Chart
with st.expander("📈 Price with Prediction Markers"):
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(final_df.index, final_df['Close'], label="Close Price", color='blue')
    ax1.scatter(final_df.index[final_df['Prediction'] == 1], final_df['Close'][final_df['Prediction'] == 1], color='green', label='Predicted Up')
    ax1.scatter(final_df.index[final_df['Prediction'] == 0], final_df['Close'][final_df['Prediction'] == 0], color='red', label='Predicted Down')
    ax1.legend(); ax1.grid(); st.pyplot(fig1)

# Quant Backtest Metrics
with st.expander("📊 Quant Backtest Metrics"):
    final_df['Signal'] = final_df['Prediction'].shift(1)
    final_df['Daily Return'] = final_df['Close'].pct_change()
    final_df['Strategy Return'] = final_df['Daily Return'] * final_df['Signal']

    cum_strat = (1 + final_df['Strategy Return'].fillna(0)).cumprod()
    cum_hold = (1 + final_df['Daily Return'].fillna(0)).cumprod()
    sharpe_final = np.sqrt(252) * final_df['Strategy Return'].mean() / final_df['Strategy Return'].std()

    st.metric("📈 Strategy Sharpe Ratio", f"{sharpe_final:.2f}")
    st.line_chart(pd.DataFrame({"Strategy": cum_strat, "Buy & Hold": cum_hold}))
    st.dataframe(pd.DataFrame({
        "Final Strategy Value ($1)": [cum_strat.iloc[-1]],
        "Final Market Value ($1)": [cum_hold.iloc[-1]],
        "Annualized Volatility": [final_df['Strategy Return'].std() * np.sqrt(252)]
    }).T.rename(columns={0: "Value"}))

# Trade Signals Chart
with st.expander("📌 Trade Signals on Chart"):
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(final_df.index, final_df['Close'], label="Close Price", color='gray')
    buys = (final_df['Signal'] == 1) & (final_df['Signal'].shift(1) != 1)
    sells = (final_df['Signal'] == 0) & (final_df['Signal'].shift(1) == 1)
    ax2.scatter(final_df.index[buys], final_df['Close'][buys], color='green', marker='^', label='Buy')
    ax2.scatter(final_df.index[sells], final_df['Close'][sells], color='red', marker='v', label='Sell')
    ax2.legend(); ax2.grid(); st.pyplot(fig2)
