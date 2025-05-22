# --- Annual Valuation Ratios (Bar Chart) -------------
with st.expander("📊 Annual Valuation Metrics (Bar Chart)"):
    try:
        # 1) Get Ticker and info
        t    = yf.Ticker(ticker)
        info = t.info

        # 2) Get 5-year price history and year-end prices
        hist = t.history(period="5y")
        if hist.empty:
            raise ValueError("No price history")
        ye_price = hist['Close'].resample('Y').last()

        # 3) Earnings & Revenue (annual)
        if t.earnings is None or t.earnings.empty:
            raise ValueError("Missing earnings data")
        earn = t.earnings['Earnings']
        rev   = t.earnings['Revenue']

        # align years
        years    = ye_price.index.year
        earn_yr  = earn.reindex(years).ffill().values
        rev_yr   = rev.reindex(years).ffill().values

        # 4) Balance sheet → Total shareholder equity
        if (
            t.balance_sheet is None or 
            t.balance_sheet.empty or 
            'TotalStockholderEquity' not in t.balance_sheet.index
        ):
            raise ValueError("Missing balance sheet data")
        equity = t.balance_sheet.loc['TotalStockholderEquity']
        # year-end equity
        eq_yr = equity.resample('Y').last().reindex(ye_price.index).values

        # 5) Compute per-share book value
        shares   = info.get('sharesOutstanding', np.nan)
        bv_yr    = eq_yr / shares

        # 6) Compute ratios
        pe = ye_price.values / earn_yr
        ps = ye_price.values / (rev_yr / 1e9)      # Rev in billions
        pb = ye_price.values / bv_yr

        # 7) Build DataFrame & plot
        df_ratios = pd.DataFrame({
            'P/E': pe,
            'P/S': ps,
            'P/B': pb
        }, index=ye_price.index)

        st.bar_chart(df_ratios)

    except Exception as e:
        st.warning(f"Could not generate annual valuation charts: {e}")
