import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import google.generativeai as genai

# 🔥 1. 제목 및 페이지 설정
st.set_page_config(layout="wide", page_title="Quant Dashboard")
st.title("🚀 Quant Dashboard")

# ---------------------------------------------------------
# 🔑 API 키 자동 로드
# ---------------------------------------------------------
try:
    api_key = st.secrets["general"]["GEMINI_API_KEY"]
except:
    api_key = st.sidebar.text_input("🔑 API Key가 없습니다. 수동으로 입력하세요:", type="password")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("⚠️ API Key가 설정되지 않았습니다. .streamlit/secrets.toml 파일을 확인해주세요.")
    st.stop()

# ---------------------------------------------------------
# 1. 사이드바: 매매일지 입력
# ---------------------------------------------------------
st.sidebar.header("📝 Portfolio Inputs")

if st.sidebar.button("🔄 Refresh Data (Click)"):
    st.cache_data.clear()
    st.rerun()

target_currency = st.sidebar.radio("💱 Display Currency", ["KRW (₩)", "USD ($)"])
target_sym = "₩" if target_currency == "KRW (₩)" else "$"

st.sidebar.info("💡 입력은 '현지 통화' 그대로 하세요! (삼성=원, 애플=달러)")

default_data = pd.DataFrame([
    {"Market": "🇺🇸 US", "Ticker": "SCHD", "Date": datetime(2023, 1, 15), "Price": 75.5, "Qty": 100},
    {"Market": "🇰🇷 KOSPI", "Ticker": "005930", "Date": datetime(2023, 6, 20), "Price": 72000.0, "Qty": 10},
    {"Market": "🇺🇸 Coin", "Ticker": "BTC-USD", "Date": datetime(2024, 1, 10), "Price": 45000.0, "Qty": 0.1},
])

edited_df = st.sidebar.data_editor(
    default_data,
    num_rows="dynamic",
    column_config={
        "Market": st.column_config.SelectboxColumn(
            "Market",
            options=["🇺🇸 US", "🇰🇷 KOSPI", "🇰🇷 KOSDAQ", "🇺🇸 Coin"],
            required=True
        ),
        "Ticker": st.column_config.TextColumn("Ticker", validate="^[A-Za-z0-9.-]+$"),
        "Date": st.column_config.DateColumn("Buy Date", format="YYYY-MM-DD"),
        "Price": st.column_config.NumberColumn("Buy Price (Local)", min_value=0.01, format="%.2f"),
        "Qty": st.column_config.NumberColumn("Quantity", min_value=0.0001, format="%.4f"),
    },
    hide_index=True
)

if edited_df.empty:
    st.warning("👈 Please enter at least one ticker in the sidebar!")
    st.stop()

# ---------------------------------------------------------
# 2. 데이터 처리 및 환율 계산
# ---------------------------------------------------------
with st.spinner('Fetching market data & Exchange rates... ⏳'):
    
    @st.cache_data(ttl=600)
    def get_exchange_rate():
        ex_data = yf.download("KRW=X", period="10y", progress=False)['Close']
        if isinstance(ex_data, pd.Series):
            ex_data = ex_data.to_frame(name="KRW=X")
        ex_data.index = ex_data.index.tz_localize(None)
        return ex_data.ffill().fillna(1000)

    exchange_rate_history = get_exchange_rate()
    current_exchange_rate = exchange_rate_history.iloc[-1].item()

    final_tickers = []
    edited_df["RealTicker"] = edited_df["Ticker"] 
    edited_df["Currency"] = "USD"

    for index, row in edited_df.iterrows():
        raw_ticker = str(row["Ticker"]).strip().upper()
        market = row["Market"]
        
        if market == "🇰🇷 KOSPI":
            if not raw_ticker.endswith(".KS"): raw_ticker += ".KS"
            edited_df.at[index, "Currency"] = "KRW"
        elif market == "🇰🇷 KOSDAQ":
            if not raw_ticker.endswith(".KQ"): raw_ticker += ".KQ"
            edited_df.at[index, "Currency"] = "KRW"
        else:
            edited_df.at[index, "Currency"] = "USD"
        
        final_tickers.append(raw_ticker)
        edited_df.at[index, "RealTicker"] = raw_ticker

    unique_tickers = list(set(final_tickers))
    
    @st.cache_data(ttl=600) 
    def get_market_data(ticker_list):
        try:
            data = yf.download(ticker_list, period="10y", progress=False)['Close']
            if isinstance(data, pd.Series):
                data = data.to_frame(name=ticker_list[0])
            data.index = data.index.tz_localize(None)
            return data.ffill().fillna(0)
        except Exception as e:
            return pd.DataFrame()

    raw_data = get_market_data(unique_tickers)
    
    if raw_data.empty:
        st.error("Failed to load data. Please check tickers.")
        st.stop()

    common_index = raw_data.index.intersection(exchange_rate_history.index)
    raw_data = raw_data.loc[common_index]
    exchange_rate_history = exchange_rate_history.loc[common_index]

    current_prices = raw_data.iloc[-1]
    last_updated = raw_data.index[-1].strftime('%Y-%m-%d %H:%M')

    earliest_input_date = pd.to_datetime(edited_df["Date"].min())
    sim_data = raw_data[raw_data.index >= earliest_input_date].copy()
    sim_ex_rate = exchange_rate_history[exchange_rate_history.index >= earliest_input_date]["KRW=X"]
    
    portfolio_history = pd.Series(0.0, index=sim_data.index)
    invested_capital_history = pd.Series(0.0, index=sim_data.index)

    total_invested_converted = 0.0
    current_portfolio_value_converted = 0.0
    details = []

    for index, row in edited_df.iterrows():
        real_ticker = row["RealTicker"]
        display_ticker = row["Ticker"]
        asset_currency = row["Currency"]
        
        buy_date = pd.to_datetime(row["Date"])
        price_at_buy_native = float(row["Price"])
        qty = float(row["Qty"])
        
        if real_ticker not in sim_data.columns:
            st.toast(f"⚠️ Data missing for '{display_ticker}'")
            continue

        invest_amt_native = price_at_buy_native * qty
        current_price_native = current_prices[real_ticker]
        current_val_native = current_price_native * qty
        
        if target_currency == "KRW (₩)":
            if asset_currency == "USD":
                asset_val_series = (sim_data[real_ticker] * qty) * sim_ex_rate
                invest_amt_final = invest_amt_native * current_exchange_rate
                current_val_final = current_val_native * current_exchange_rate
            else: 
                asset_val_series = sim_data[real_ticker] * qty
                invest_amt_final = invest_amt_native
                current_val_final = current_val_native
        else:
            if asset_currency == "KRW":
                asset_val_series = (sim_data[real_ticker] * qty) / sim_ex_rate
                invest_amt_final = invest_amt_native / current_exchange_rate
                current_val_final = current_val_native / current_exchange_rate
            else:
                asset_val_series = sim_data[real_ticker] * qty
                invest_amt_final = invest_amt_native
                current_val_final = current_val_native

        total_invested_converted += invest_amt_final
        current_portfolio_value_converted += current_val_final
        
        asset_val_series.loc[asset_val_series.index < buy_date] = 0.0
        portfolio_history = portfolio_history.add(asset_val_series, fill_value=0)
        
        cap_series = pd.Series(0.0, index=sim_data.index)
        cap_series.loc[cap_series.index >= buy_date] = invest_amt_final
        invested_capital_history = invested_capital_history.add(cap_series, fill_value=0)

        roi_native = ((current_price_native - price_at_buy_native) / price_at_buy_native) * 100

        details.append({
            "Ticker": display_ticker,
            "Market": row["Market"],
            "Currency": asset_currency,
            "Qty": qty,
            "Avg Buy (Local)": price_at_buy_native,
            "Current (Local)": current_price_native,
            "Current Val (Converted)": current_val_final,
            "Return (%)": roi_native
        })

    if total_invested_converted > 0:
        total_return_money = current_portfolio_value_converted - total_invested_converted
        total_return_pct = (total_return_money / total_invested_converted) * 100
    else:
        total_return_money = 0
        total_return_pct = 0
        
    df_details = pd.DataFrame(details)
    if not df_details.empty:
        df_details["Weight (%)"] = (df_details["Current Val (Converted)"] / current_portfolio_value_converted * 100).fillna(0)

# ---------------------------------------------------------
# 📊 3. 대시보드 출력
# ---------------------------------------------------------
st.markdown(f"### 💰 Portfolio Status (Total in {target_currency})")
st.caption(f"ℹ️ Applied Exchange Rate (USD/KRW): {current_exchange_rate:,.2f}")

# 🔥 여기가 수정된 부분 (2단 그리드)
c1, c2 = st.columns(2)
c1.metric("Total Invested", f"{target_sym}{total_invested_converted:,.0f}")
c2.metric("Current Value", f"{target_sym}{current_portfolio_value_converted:,.0f}")

st.write("") # 여백

c3, c4 = st.columns(2)
c3.metric("Net Profit", f"{target_sym}{total_return_money:,.0f}", delta=f"{total_return_pct:.2f}%")
c4.metric("Tickers", f"{len(df_details)}")

st.subheader("📈 Asset Growth (Converted)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio_history.index, y=portfolio_history, mode='lines', name='Total Value', line=dict(color='#FF4B4B', width=3)))
fig.add_trace(go.Scatter(x=invested_capital_history.index, y=invested_capital_history, mode='lines', name='Invested Capital', line=dict(color='gray', dash='dash')))
fig.update_layout(hovermode="x unified", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🧾 Holdings Detail")
st.dataframe(
    df_details.style.format({
        "Qty": "{:,.4f}",
        "Avg Buy (Local)": "{:,.2f}", 
        "Current (Local)": "{:,.2f}",
        "Current Val (Converted)": f"{target_sym}{{:,.0f}}",
        "Return (%)": "{:,.2f}%",
        "Weight (%)": "{:,.1f}%"
    }).background_gradient(cmap='RdYlGn', subset=['Return (%)']),
    use_container_width=True
)

# ---------------------------------------------------------
# 🔮 4. Gemini AI 진단
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔮 Gemini AI Analyst Report")

ai_portfolio_summary = df_details[["Ticker", "Currency", "Weight (%)", "Return (%)"]].to_string(index=False)
chart_trend = "Upward (Profit)" if total_return_pct > 0 else "Downward (Loss)"

prompt = f"""
You are a professional Quant Analyst. Analyze this user's GLOBAL portfolio.
The user holds assets in both USD and KRW, but the summary is converted to {target_currency}.

[Summary in {target_currency}]
- Total Invested: {target_sym}{total_invested_converted:,.0f}
- Current Value: {target_sym}{current_portfolio_value_converted:,.0f}
- Total Return: {total_return_pct:.2f}% ({chart_trend})
- Exchange Rate Used: {current_exchange_rate:,.2f} KRW/USD

[Holdings]
{ai_portfolio_summary}

[Request]
1. Analyze the portfolio performance considering Currency Risks (USD vs KRW exposure).
2. Identify the main profit drivers.
3. Suggest a rebalancing strategy or risk management tip for this mix.

Please write in **Korean** (한국어). Use Markdown.
"""

if st.button("🤖 Analyze Portfolio (Click)"):
    with st.spinner("AI Analyst is evaluating currency risks and assets..."):
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            model_name = 'models/gemini-1.5-flash'
            for m in available_models:
                if 'flash' in m: model_name = m; break
                elif 'pro' in m: model_name = m
            
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            st.success(f"✅ Analysis Complete (Model: {model_name})")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"Error: {e}")
