import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import google.generativeai as genai

# 1. 페이지 설정 및 제목
st.set_page_config(layout="wide", page_title="Quant Dashboard")
st.title("🚀 Quant Dashboard (200MA Pro Spec)")

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
        "Market": st.column_config.SelectboxColumn("Market", options=["🇺🇸 US", "🇰🇷 KOSPI", "🇰🇷 KOSDAQ", "🇺🇸 Coin"], required=True),
        "Ticker": st.column_config.TextColumn("Ticker", validate="^[A-Za-z0-9.-]+$"),
        "Date": st.column_config.DateColumn("Buy Date", format="YYYY-MM-DD"),
        "Price": st.column_config.NumberColumn("Buy Price (Local)", min_value=0.01, format="%.2f"),
        "Qty": st.column_config.NumberColumn("Quantity", min_value=0.0001, format="%.4f"),
    },
    hide_index=True
)

if edited_df.empty:
    st.warning("👈 Please enter data in the sidebar!")
    st.stop()

# ---------------------------------------------------------
# 2. 데이터 처리 및 환율 계산
# ---------------------------------------------------------
with st.spinner('Fetching market data, S&P 500 & Exchange rates... ⏳'):
    
    @st.cache_data(ttl=600)
    def get_exchange_rate():
        ex_data = yf.download("KRW=X", period="10y", progress=False)['Close']
        if isinstance(ex_data, pd.Series):
            ex_data = ex_data.to_frame(name="KRW=X")
        ex_data.index = ex_data.index.tz_localize(None)
        return ex_data.ffill().fillna(1000)

    exchange_rate_history = get_exchange_rate()
    current_exchange_rate = exchange_rate_history.iloc[-1].item()

    ticker_map = {}
    edited_df["RealTicker"] = edited_df["Ticker"]
    
    for index, row in edited_df.iterrows():
        rt = str(row["Ticker"]).strip().upper()
        if row["Market"] == "🇰🇷 KOSPI":
            if not rt.endswith(".KS"): rt += ".KS"
        elif row["Market"] == "🇰🇷 KOSDAQ":
            if not rt.endswith(".KQ"): rt += ".KQ"
        ticker_map[row["Ticker"]] = rt
        edited_df.at[index, "RealTicker"] = rt

    unique_tickers = list(set(edited_df["RealTicker"].tolist()))
    
    @st.cache_data(ttl=600)
    def get_market_data(ticker_list):
        download_list = ticker_list + ["^GSPC"]
        data = yf.download(download_list, period="10y", progress=False)['Close']
        if isinstance(data, pd.Series): data = data.to_frame(name=download_list[0])
        data.index = data.index.tz_localize(None)
        return data.ffill().fillna(0)

    raw_data_all = get_market_data(unique_tickers)
    sp500_data = raw_data_all["^GSPC"].copy()
    raw_data = raw_data_all.drop(columns=["^GSPC"], errors='ignore')

    common_index = raw_data.index.intersection(exchange_rate_history.index)
    raw_data = raw_data.loc[common_index]
    exchange_rate_history = exchange_rate_history.loc[common_index]
    sp500_data = sp500_data.loc[raw_data.index]

    current_prices = raw_data.iloc[-1]
    last_updated = raw_data.index[-1].strftime('%Y-%m-%d %H:%M')

    earliest_date = pd.to_datetime(edited_df["Date"].min())
    sim_data = raw_data[raw_data.index >= earliest_date].copy()
    sim_ex_rate = exchange_rate_history[exchange_rate_history.index >= earliest_date]["KRW=X"]
    sim_sp500 = sp500_data[sp500_data.index >= earliest_date].copy()
    
    portfolio_history = pd.Series(0.0, index=sim_data.index)
    invested_capital_history = pd.Series(0.0, index=sim_data.index)
    details = []

    for index, row in edited_df.iterrows():
        rt = row["RealTicker"]
        buy_date = pd.to_datetime(row["Date"])
        price_native = float(row["Price"])
        qty = float(row["Qty"])
        is_usd = row["Market"] in ["🇺🇸 US", "🇺🇸 Coin"]
        
        if rt not in sim_data.columns: continue

        val_native = sim_data[rt] * qty
        if target_currency == "KRW (₩)":
            asset_val = val_native * sim_ex_rate if is_usd else val_native
            invest_final = (price_native * qty) * current_exchange_rate if is_usd else (price_native * qty)
        else:
            asset_val = val_native if is_usd else val_native / sim_ex_rate
            invest_final = (price_native * qty) if is_usd else (price_native * qty) / current_exchange_rate

        asset_val.loc[asset_val.index < buy_date] = 0.0
        portfolio_history = portfolio_history.add(asset_val, fill_value=0)
        
        cap_val = pd.Series(0.0, index=sim_data.index)
        cap_val.loc[cap_val.index >= buy_date] = invest_final
        invested_capital_history = invested_capital_history.add(cap_val, fill_value=0)

        details.append({
            "Ticker": row["Ticker"],
            "Market": row["Market"],
            "Qty": qty,
            "Avg Buy": price_native,
            "Current": current_prices[rt],
            "Value": asset_val.iloc[-1],
            "Return (%)": ((current_prices[rt] - price_native) / price_native) * 100
        })

    total_invested = invested_capital_history.iloc[-1]
    current_value = portfolio_history.iloc[-1]
    total_ret_pct = (current_value / total_invested - 1) * 100 if total_invested > 0 else 0
    df_details = pd.DataFrame(details)
    df_details["Weight (%)"] = (df_details["Value"] / current_value * 100).fillna(0)

# ---------------------------------------------------------
# 📊 3. 대시보드 출력
# ---------------------------------------------------------
st.markdown(f"### 💰 Portfolio Status (Total in {target_currency})")
c1, c2 = st.columns(2)
c1.metric("Total Invested", f"{target_sym}{total_invested:,.0f}")
c2.metric("Current Value", f"{target_sym}{current_value:,.0f}")

c3, c4 = st.columns(2)
c3.metric("Net Profit", f"{target_sym}{current_value-total_invested:,.0f}", delta=f"{total_ret_pct:.2f}%")
c4.metric("Tickers", f"{len(df_details)}")

st.subheader("📈 Asset Growth")
fig_main = go.Figure()
fig_main.add_trace(go.Scatter(x=portfolio_history.index, y=portfolio_history, mode='lines', name='Value', line=dict(color='#FF4B4B', width=3)))
fig_main.add_trace(go.Scatter(x=invested_capital_history.index, y=invested_capital_history, mode='lines', name='Capital', line=dict(color='gray', dash='dash')))
fig_main.update_layout(hovermode="x unified", template="plotly_white")
st.plotly_chart(fig_main, use_container_width=True)

# ---------------------------------------------------------
# 📊 4. 기술적 분석 (🔥 200 MA 추가됨)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Technical Analysis (5, 20, 60, 120, 200 MA)")

selected_ticker = st.selectbox("Select Asset", df_details["Ticker"].unique())
rt = ticker_map[selected_ticker]

if rt in raw_data.columns:
    tech_data = raw_data[rt].copy().to_frame(name="Close").iloc[-500:] # 200일선을 위해 더 많은 데이터 로드

    # MA 계산
    for window in [5, 20, 60, 120, 200]:
        tech_data[f'MA{window}'] = tech_data['Close'].rolling(window=window).mean()

    # BB & RSI
    tech_data['Std_20'] = tech_data['Close'].rolling(window=20).std()
    tech_data['Upper'] = tech_data['MA20'] + (tech_data['Std_20'] * 2)
    tech_data['Lower'] = tech_data['MA20'] - (tech_data['Std_20'] * 2)
    delta = tech_data['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    tech_data['RSI'] = 100 - (100 / (1 + (gain / loss)))

    fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    
    # 200 MA 추가 (진한 빨간색으로 강조)
    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tech_data['MA200'], line=dict(color='darkred', width=3), name='200 MA (Trend)'), row=1, col=1)
    
    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tech_data['Upper'], line=dict(color='rgba(200,200,200,0.2)', dash='dot'), name='Upper BB'), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tech_data['Lower'], line=dict(color='rgba(200,200,200,0.2)', dash='dot'), name='Lower BB', fill='tonexty'), row=1, col=1)
    
    colors = {'MA5':'pink', 'MA20':'orange', 'MA60':'green', 'MA120':'purple'}
    for ma, color in colors.items():
        fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tech_data[ma], line=dict(color=color, width=1), name=ma), row=1, col=1)
    
    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tech_data['Close'], line=dict(color='blue', width=2), name='Price'), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tech_data['RSI'], line=dict(color='magenta'), name='RSI'), row=2, col=1)
    fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig_tech.update_layout(height=800, template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig_tech, use_container_width=True)

# ---------------------------------------------------------
# 🔮 5. Gemini AI 진단
# ---------------------------------------------------------
st.markdown("---")
if st.button("🤖 Analyze Portfolio with AI"):
    # (AI 분석 로직 동일)
    st.info("AI 분석 기능을 실행합니다...")
