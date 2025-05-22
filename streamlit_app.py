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

# --- Streamlit setup ---------------------------------
st.set_page_config(page_title="Stock Trend Predictor (AutoML Edition)", layout="wide")
st.set_option('client.showErrorDetails', True)
st.title("📈 Stock Trend Predictor (AutoML Edition)")

# --- User inputs -------------------------------------
ticker     = st.text_input("Enter ticker (e.g. AAPL):", value="AAPL")
start_date = st.date_input("Start date", pd.to_datetime("2020-01-01"))
end_date   = st.date_input("End   date", pd.to_datetime("2024-12-31"))

# --- Company snapshot -------------------------------
# --- Annual Valuation Ratios ------------------------
with st.expander("📊 Annual Valuation Metrics (Bar Chart)"):
    try:
        t    = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="5y")

        # 1️⃣ Year-end price
        ye_price = hist['Close'].resample('Y').last()

        # 2️⃣ Annual earnings & revenue (indexed by year integer)
        earn = t.earnings['Earnings']      # Series: idx = [2023,2022,...]
        rev   = t.earnings['Revenue']      # Series same index

        # Map year-end prices to those same year ints
        years = ye_price.index.year
        earn_yr = earn.reindex(years).values
        rev_yr  = rev.reindex(years).values

        # 3️⃣ Balance sheet → Total equity, then book-value per share
        bs      = t.balance_sheet.loc['TotalStockholderEquity']  # idx = dates
        eq_yr   = bs.resample('Y').last().reindex(ye_price.index).values
        shares  = info.get('sharesOutstanding', np.nan)
        bv_yr   = eq_yr / shares

        # 4️⃣ Compute ratios
        pe = ye_price.values / earn_yr
        ps = ye_price.values / (rev_yr / 1e9)
        pb = ye_price.values / bv_yr

        # 5️⃣ Build DataFrame & plot bars
        df_ratios = pd.DataFrame({
            'P/E': pe,
            'P/S': ps,
            'P/B': pb
        }, index=ye_price.index)

        st.bar_chart(df_ratios)

    except Exception as e:
        st.warning(f"Could not generate annual valuation charts: {e}")

# --- Historical Valuation Charts --------------------
with st.expander("📊 Historical Valuation Metrics"):
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="5y")

        # safe‐check
        if t.earnings is None or t.balance_sheet is None or t.earnings.empty or t.balance_sheet.empty:
            raise ValueError("Missing earnings or balance sheet data")

        # year‐aligned arrays
        earnings = t.earnings
        rev_yr   = earnings['Revenue'].reindex(hist.index.year).ffill().values
        eps_yr   = earnings['Earnings'].reindex(hist.index.year).ffill().values
        bs       = t.balance_sheet
        bv_yr    = (bs.loc['TotalStockholderEquity'] / info.get('sharesOutstanding',1))\
                     .reindex(hist.index.year).ffill().values

        pe = hist['Close'].values / eps_yr
        ps = hist['Close'].values / (rev_yr/1e9)
        pb = hist['Close'].values / bv_yr

        fig, axs = plt.subplots(3,1,figsize=(10,10),sharex=True)
        axs[0].plot(hist.index, pe, label='P/E'); axs[0].axhline(np.nanmean(pe), color='gray', ls='--', label='Avg'); axs[0].set_title("P/E Ratio");    axs[0].legend()
        axs[1].plot(hist.index, ps, label='P/S', color='purple'); axs[1].axhline(np.nanmean(ps), color='gray', ls='--', label='Avg'); axs[1].set_title("P/S Ratio"); axs[1].legend()
        axs[2].plot(hist.index, pb, label='P/B', color='green');   axs[2].axhline(np.nanmean(pb), color='gray', ls='--', label='Avg'); axs[2].set_title("P/B Ratio");   axs[2].legend()
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"Could not generate valuation charts: {e}")

# --- Download & prepare price data ------------------
raw = yf.download(ticker, start=start_date, end=end_date)
if raw.empty:
    st.error("No price data found for this ticker.")
    st.stop()

data = raw[['Close']].copy()
# flatten with .flatten() to ensure 1-D
cs   = pd.Series(data['Close'].values.flatten(), index=data.index)

# technicals
data['Daily_Change_%'] = cs.pct_change()*100
data['MA_5']            = cs.rolling(5).mean()
data['MA_10']           = cs.rolling(10).mean()
data['RSI']             = ta.momentum.RSIIndicator(close=cs,window=14).rsi()
macd_obj               = ta.trend.MACD(close=cs)
data['MACD']            = macd_obj.macd()
data['MACD_Signal']     = macd_obj.macd_signal()

# lags
for i in range(1,6):
    data[f'Close_lag_{i}'] = cs.shift(i)

# binary target
data['Target'] = (cs.shift(-1) > cs).astype(int)
data.dropna(inplace=True)

# build feature list
features = ['Daily_Change_%','MA_5','MA_10','RSI','MACD','MACD_Signal'] + \
            [f'Close_lag_{i}' for i in range(1,6)]
# add Volume if present
if 'Volume' in raw.columns:
    data['Volume'] = raw['Volume']
    features = ['Volume'] + features

X = data[features]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

# --- AutoML & backtest ------------------------------
models = {
    "XGBoost":        XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Random Forest":  RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Reg.":  LogisticRegression(max_iter=1000, solver='lbfgs')
}

results = []
for name,model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    df_t  = data.iloc[-len(y_test):].copy()
    df_t['Pred']      = preds
    df_t['Signal']    = df_t['Pred'].shift(1)
    df_t['DailyRet']  = df_t['Close'].pct_change()
    df_t['StratRet']  = df_t['DailyRet'] * df_t['Signal']
    cum               = (1 + df_t['StratRet'].fillna(0)).cumprod()
    sharpe            = np.sqrt(252)*df_t['StratRet'].mean()/df_t['StratRet'].std()
    results.append({
        "Model":name,
        "Accuracy":accuracy_score(y_test,preds),
        "Sharpe":sharpe,
        "FinalVal":cum.iloc[-1],
        "Preds":preds
    })

best = max(results, key=lambda x: x['Sharpe'])
final = data.iloc[-len(y_test):].copy()
final['Pred'] = best['Preds']

st.success(f"✅ Best model: {best['Model']} (Sharpe {best['Sharpe']:.2f})")

with st.expander("📋 Model Performance"):
    st.write(f"Model: **{best['Model']}**   Accuracy: **{best['Accuracy']:.2%}**")

with st.expander("📈 Price & Predictions"):
    fig,ax = plt.subplots(figsize=(12,5))
    ax.plot(final.index, final['Close'], color='blue', label='Close')
    ax.scatter(final.index[final['Pred']==1], final['Close'][final['Pred']==1], color='green', marker='o', label='Up')
    ax.scatter(final.index[final['Pred']==0], final['Close'][final['Pred']==0], color='red', marker='o', label='Down')
    ax.legend(); ax.grid(); st.pyplot(fig)

with st.expander("📊 Quant Backtest Metrics"):
    final['Signal']   = final['Pred'].shift(1)
    final['DailyRet'] = final['Close'].pct_change()
    final['StratRet'] = final['DailyRet'] * final['Signal']
    cum_str = (1+final['StratRet'].fillna(0)).cumprod()
    cum_hld = (1+final['DailyRet'].fillna(0)).cumprod()
    sr_f    = np.sqrt(252)*final['StratRet'].mean()/final['StratRet'].std()
    st.metric("Sharpe Ratio", f"{sr_f:.2f}")
    st.line_chart(pd.DataFrame({"Strategy":cum_str, "Buy & Hold":cum_hld}))
    st.dataframe(pd.DataFrame({
        "Final Strat $1":[cum_str.iloc[-1]],
        "Final HLD  $1":[cum_hld.iloc[-1]],
        "Ann Vol":[final['StratRet'].std()*np.sqrt(252)]
    }).T.rename(columns={0:"Value"}))

with st.expander("📌 Trade Signals on Chart"):
    fig,ax = plt.subplots(figsize=(12,5))
    ax.plot(final.index, final['Close'], color='gray', label='Close')
    buys  = (final['Signal']==1) & (final['Signal'].shift(1)!=1)
    sells = (final['Signal']==0) & (final['Signal'].shift(1)==1)
    ax.scatter(final.index[buys],  final['Close'][buys],  color='green', marker='^', label='Buy')
    ax.scatter(final.index[sells], final['Close'][sells], color='red',   marker='v', label='Sell')
    ax.legend(); ax.grid(); st.pyplot(fig)
