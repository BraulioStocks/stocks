import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import ta
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# --- Streamlit setup ---------------------------------
st.set_page_config(page_title="Stock Trend Predictor (FMP Edition)", layout="wide")
st.set_option('client.showErrorDetails', True)
st.title("📈 Stock Trend Predictor (AutoML + FMP Valuations)")

# --- User inputs -------------------------------------
ticker     = st.text_input("Enter ticker (e.g. AAPL):", value="AAPL")
start_date = st.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date   = st.date_input("End   date", pd.to_datetime("2024-12-31"))

# --- Your FMP API key (hard-coded) -------------------
apikey = "t2QaKaEK7z4WnLBnj7EiyWy1ViDoaEY9"

# --- Company snapshot -------------------------------
with st.expander("🏢 Company Snapshot"):
    try:
        t    = yf.Ticker(ticker)
        info = t.info
        st.markdown(f"**Name:** {info.get('longName','-')}")
        st.markdown(f"**Sector:** {info.get('sector','-')}")
        st.markdown(f"**Market Cap:** {info.get('marketCap',0):,}")
        st.markdown(f"**Trailing P/E:** {info.get('trailingPE','-')}")
        st.markdown(f"**Forward P/E:** {info.get('forwardPE','-')}")
        st.markdown(f"**Dividend Yield:** {info.get('dividendYield','-')}")
        st.markdown(f"**ROE:** {info.get('returnOnEquity','-')}")
    except Exception:
        st.warning("⚠️ Unable to load company fundamentals.")

# --- Annual Valuations via FMP ----------------------
with st.expander("📊 Annual Valuation Metrics (via FMP API)"):
    try:
        # 1) Income statements (last 5 years)
        url_inc = (
            f"https://financialmodelingprep.com/api/v3/income-statement/"
            f"{ticker}?limit=5&apikey={apikey}"
        )
        df_inc = pd.DataFrame(requests.get(url_inc).json())
        if df_inc.empty or 'eps' not in df_inc or 'revenue' not in df_inc:
            raise ValueError("Income statement data incomplete")

        # 2) Balance sheet (last 5 years)
        url_bs = (
            f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/"
            f"{ticker}?limit=5&apikey={apikey}"
        )
        df_bs = pd.DataFrame(requests.get(url_bs).json())
        if df_bs.empty or 'totalStockholdersEquity' not in df_bs:
            raise ValueError("Balance sheet data incomplete")

        # 3) Index by fiscal year-end
        df_inc['date'] = pd.to_datetime(df_inc['date'])
        df_inc.set_index('date', inplace=True)
        df_bs['date']  = pd.to_datetime(df_bs['date'])
        df_bs.set_index('date', inplace=True)

        # 4) Year-end prices aligned to those dates
        hist = yf.Ticker(ticker).history(start=start_date, end=end_date)['Close']
        prices = hist.reindex(df_inc.index, method='ffill').values

        # 5) Extract financials
        eps    = df_inc['eps'].astype(float).values
        rev    = df_inc['revenue'].astype(float).values
        equity = df_bs['totalStockholdersEquity'].astype(float).values
        shares = yf.Ticker(ticker).info.get('sharesOutstanding', np.nan)
        bv_ps  = equity / shares

        # 6) Compute ratios
        pe = prices / eps
        ps = prices / (rev / 1e9)   # rev in billions
        pb = prices / bv_ps

        # 7) Plot bar chart
        df_ratios = pd.DataFrame({
            'P/E': pe,
            'P/S': ps,
            'P/B': pb
        }, index=df_inc.index.date)
        st.bar_chart(df_ratios)

    except Exception as e:
        st.warning(f"Could not generate annual valuations: {e}")

# --- Download & prepare price data ------------------
raw = yf.download(ticker, start=start_date, end=end_date)
if raw.empty:
    st.error("No historical price data found.")
    st.stop()

data = raw[['Close']].copy()
cs   = pd.Series(data['Close'].values.flatten(), index=data.index)

# --- Technical indicators ----------------------------
data['Daily_Change_%'] = cs.pct_change() * 100
data['MA_5']           = cs.rolling(5).mean()
data['MA_10']          = cs.rolling(10).mean()
data['RSI']            = ta.momentum.RSIIndicator(close=cs, window=14).rsi()
macd_obj               = ta.trend.MACD(close=cs)
data['MACD']           = macd_obj.macd()
data['MACD_Signal']    = macd_obj.macd_signal()
for i in range(1, 6):
    data[f'Close_lag_{i}'] = cs.shift(i)

data['Target'] = (cs.shift(-1) > cs).astype(int)
data.dropna(inplace=True)

# --- Features & train/test --------------------------
features = ['Daily_Change_%','MA_5','MA_10','RSI','MACD','MACD_Signal'] \
         + [f'Close_lag_{i}' for i in range(1,6)]
if 'Volume' in raw.columns:
    data['Volume'] = raw['Volume']
    features = ['Volume'] + features

X = data[features]; y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- AutoML & backtest ------------------------------
models = {
    "XGBoost":       XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Reg.": LogisticRegression(max_iter=1000, solver='lbfgs')
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    df_t  = data.iloc[-len(y_test):].copy()
    df_t['Pred']     = preds
    df_t['Signal']   = df_t['Pred'].shift(1)
    df_t['DailyRet'] = df_t['Close'].pct_change()
    df_t['StratRet'] = df_t['DailyRet'] * df_t['Signal']
    cum              = (1 + df_t['StratRet'].fillna(0)).cumprod()
    sharpe           = np.sqrt(252) * df_t['StratRet'].mean() / df_t['StratRet'].std()
    results.append({
        "Model":    name,
        "Accuracy": accuracy_score(y_test, preds),
        "Sharpe":   sharpe,
        "FinalVal": cum.iloc[-1],
        "Preds":    preds
    })

best  = max(results, key=lambda x: x['Sharpe'])
final = data.iloc[-len(y_test):].copy()
final['Pred'] = best['Preds']

st.success(f"✅ Best model: {best['Model']} (Sharpe {best['Sharpe']:.2f})")

with st.expander("📋 Model Performance"):
    st.write(f"Model: **{best['Model']}**   Accuracy: **{best['Accuracy']:.2%}**")

with st.expander("📈 Price & Predictions"):
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(final.index, final['Close'], color='blue', label='Close')
    ax.scatter(final.index[final['Pred'] == 1],
               final['Close'][final['Pred'] == 1], color='green', marker='o', label='Up')
    ax.scatter(final.index[final['Pred'] == 0],
               final['Close'][final['Pred'] == 0], color='red', marker='o', label='Down')
    ax.legend(); ax.grid(); st.pyplot(fig)

with st.expander("📊 Quant Backtest Metrics"):
    final['Signal']   = final['Pred'].shift(1)
    final['DailyRet'] = final['Close'].pct_change()
    final['StratRet'] = final['DailyRet'] * final['Signal']
    cum_str = (1 + final['StratRet'].fillna(0)).cumprod()
    cum_hld = (1 + final['DailyRet'].fillna(0)).cumprod()
    sr_f    = np.sqrt(252) * final['StratRet'].mean() / final['StratRet'].std()
    st.metric("Sharpe Ratio", f"{sr_f:.2f}")
    st.line_chart(pd.DataFrame({"Strategy": cum_str, "Buy & Hold": cum_hld}))
    st.dataframe(pd.DataFrame({
        "Final Strat $1": [cum_str.iloc[-1]],
        "Final HLD  $1":  [cum_hld.iloc[-1]],
        "Ann Vol":        [final['StratRet'].std() * np.sqrt(252)]
    }).T.rename(columns={0: "Value"}))

with st.expander("📌 Trade Signals on Chart"):
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(final.index, final['Close'], color='gray', label='Close')
    buys  = (final['Signal'] == 1) & (final['Signal'].shift(1) != 1)
    sells = (final['Signal'] == 0) & (final['Signal'].shift(1) == 1)
    ax.scatter(final.index[buys],  final['Close'][buys],  color='green', marker='^', label='Buy')
    ax.scatter(final.index[sells], final['Close'][sells], color='red',   marker='v', label='Sell')
    ax.legend(); ax.grid(); st.pyplot(fig)
