import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import google.generativeai as genai

# 🔥 1. 페이지 설정
st.set_page_config(layout="wide", page_title="Quant Dashboard")
st.title("🚀 Quant Dashboard (V36. Master)")

# ---------------------------------------------------------
# 🔑 API 키 로딩 및 AI 설정
# ---------------------------------------------------------
try:
    if "general" in st.secrets and "GEMINI_API_KEY" in st.secrets["general"]:
        api_key = st.secrets["general"]["GEMINI_API_KEY"]
    else:
        api_key = st.secrets.get("GEMINI_API_KEY")
except:
    api_key = None

if api_key:
    genai.configure(api_key=api_key)
else:
    api_key_input = st.sidebar.text_input("🔑 API Key를 입력하세요:", type="password")
    if api_key_input:
        genai.configure(api_key=api_key_input)
        api_key = api_key_input

# ---------------------------------------------------------
# 2. 사이드바 및 시장 드롭다운
# ---------------------------------------------------------
st.sidebar.header("📝 Portfolio Inputs")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

target_currency = st.sidebar.radio("💱 Display Currency", ["KRW (₩)", "USD ($)"])
target_sym = "₩" if target_currency == "KRW (₩)" else "$"

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
        "Ticker": st.column_config.TextColumn("Ticker"),
        "Date": st.column_config.DateColumn("Buy Date", format="YYYY-MM-DD"),
        "Price": st.column_config.NumberColumn("Price (Local)", min_value=0.01),
        "Qty": st.column_config.NumberColumn("Qty", min_value=0.0001),
    },
    hide_index=True
)

if edited_df.empty:
    st.warning("👈 데이터를 입력해주세요.")
    st.stop()

# ---------------------------------------------------------
# 3. 데이터 로딩 및 처리
# ---------------------------------------------------------
with st.spinner('시장 데이터를 불러오는 중... ⏳'):
    @st.cache_data(ttl=600)
    def fetch_data(ticker_list):
        download_list = ticker_list + ["^GSPC", "KRW=X"]
        data = yf.download(download_list, period="10y", progress=False)['Close']
        if isinstance(data, pd.Series): data = data.to_frame(name=download_list[0])
        data.index = data.index.tz_localize(None)
        return data.ffill()

    ticker_map = {}; final_tickers = []
    for idx, row in edited_df.iterrows():
        rt = str(row["Ticker"]).strip().upper()
        if row["Market"] == "🇰🇷 KOSPI" and not rt.endswith(".KS"): rt += ".KS"
        elif row["Market"] == "🇰🇷 KOSDAQ" and not rt.endswith(".KQ"): rt += ".KQ"
        ticker_map[row["Ticker"]] = rt
        final_tickers.append(rt)

    raw_data_all = fetch_data(list(set(final_tickers)))
    exchange_rate_history = raw_data_all["KRW=X"]
    sp500_history = raw_data_all["^GSPC"]
    raw_data = raw_data_all.drop(columns=["KRW=X", "^GSPC"], errors='ignore')
    
    current_ex_rate = exchange_rate_history.iloc[-1]
    earliest_date = pd.to_datetime(edited_df["Date"].min())
    
    portfolio_history = pd.Series(0.0, index=raw_data.index)
    invested_history = pd.Series(0.0, index=raw_data.index)
    details = []

    for idx, row in edited_df.iterrows():
        rt = ticker_map[row["Ticker"]]; buy_date = pd.to_datetime(row["Date"])
        is_usd = row["Market"] in ["🇺🇸 US", "🇺🇸 Coin"]
        val_native = raw_data[rt] * float(row["Qty"])
        if target_currency == "KRW (₩)":
            val_converted = val_native * exchange_rate_history if is_usd else val_native
            invest_converted = (float(row["Price"]) * float(row["Qty"])) * current_ex_rate if is_usd else (float(row["Price"]) * float(row["Qty"]))
        else:
            val_converted = val_native if is_usd else val_native / exchange_rate_history
            invest_converted = (float(row["Price"]) * float(row["Qty"])) if is_usd else (float(row["Price"]) * float(row["Qty"])) / current_ex_rate
        val_converted.loc[val_converted.index < buy_date] = 0.0
        portfolio_history = portfolio_history.add(val_converted, fill_value=0)
        cap_val = pd.Series(0.0, index=raw_data.index); cap_val.loc[cap_val.index >= buy_date] = invest_converted
        invested_history = invested_history.add(cap_val, fill_value=0)
        details.append({"Ticker": row["Ticker"], "Qty": row["Qty"], "Avg Buy": row["Price"], "Current": raw_data[rt].iloc[-1], "Value": val_converted.iloc[-1], "Return (%)": ((raw_data[rt].iloc[-1] - row["Price"]) / row["Price"]) * 100})

    total_invested = invested_history.iloc[-1]; current_value = portfolio_history.iloc[-1]; df_details = pd.DataFrame(details)
    df_details["Weight (%)"] = (df_details["Value"] / current_value * 100).fillna(0)

# ---------------------------------------------------------
# 4. 상단 메트릭 & 포트폴리오 차트
# ---------------------------------------------------------
st.markdown(f"### 💰 Portfolio Status ({target_currency})")
c1, c2, c3 = st.columns(3)
c1.metric("Total Invested", f"{target_sym}{total_invested:,.0f}")
c2.metric("Current Value", f"{target_sym}{current_value:,.0f}")
c3.metric("Profit/Loss", f"{target_sym}{current_value-total_invested:,.0f}", delta=f"{(current_value/total_invested-1)*100:.2f}%")

st.subheader("📈 Portfolio Growth")
mask = portfolio_history > 0
f_history = portfolio_history[mask]; f_invested = invested_history[mask]
fig_growth = go.Figure()
fig_growth.add_trace(go.Scatter(x=f_history.index, y=f_history, name="자산 가치", line=dict(color='#FF4B4B', width=3)))
fig_growth.add_trace(go.Scatter(x=f_invested.index, y=f_invested, name="투자 원금", line=dict(color='gray', dash='dash')))
fig_growth.update_layout(hovermode="x unified", template="plotly_white")
st.plotly_chart(fig_growth, use_container_width=True)

# ---------------------------------------------------------
# 5. 벤치마크 & 히트맵
# ---------------------------------------------------------
st.markdown("---")
col_bench, col_heat = st.columns(2)
with col_bench:
    st.subheader("🆚 vs S&P 500 (Benchmark)")
    my_ret = (portfolio_history / invested_history - 1) * 100
    sp_sliced = sp500_history.loc[earliest_date:]
    sp_ret = (sp_sliced / sp_sliced.iloc[0] - 1) * 100
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(x=my_ret[mask].index, y=my_ret[mask], name="내 포트폴리오", line=dict(color='#FF4B4B')))
    fig_b.add_trace(go.Scatter(x=sp_ret.index, y=sp_ret, name="S&P 500", line=dict(color='blue', dash='dot')))
    st.plotly_chart(fig_b, use_container_width=True)

with col_heat:
    st.subheader("🔥 Correlation Heatmap")
    st.plotly_chart(px.imshow(raw_data.pct_change().corr(), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)

st.subheader("🧾 Holdings Detail")
st.dataframe(df_details.style.format({"Qty":"{:,.4f}", "Avg Buy":"{:,.2f}", "Current":"{:,.2f}", "Value":f"{target_sym}{{:,.0f}}", "Return (%)":"{:,.2f}%", "Weight (%)":"{:,.1f}%"}).background_gradient(cmap='RdYlGn', subset=['Return (%)']), use_container_width=True)

# ---------------------------------------------------------
# 📊 6. 기술적 분석 (MA, BB, RSI) + 개별 종목 AI 분석
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Detailed Technical Analysis")
sel_ticker = st.selectbox("분석 종목 선택", df_details["Ticker"].unique())
rt_sel = ticker_map[sel_ticker]; tech_df = raw_data[rt_sel].to_frame(name="Close").iloc[-500:]

for ma in [5, 20, 60, 120, 200]: tech_df[f'MA{ma}'] = tech_df['Close'].rolling(window=ma).mean()
tech_df['Std_20'] = tech_df['Close'].rolling(window=20).std()
tech_df['Upper'] = tech_df['MA20'] + (tech_df['Std_20'] * 2); tech_df['Lower'] = tech_df['MA20'] - (tech_df['Std_20'] * 2)
delta = tech_df['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
tech_df['RSI'] = 100 - (100 / (1 + (gain / loss)))

fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Upper'], line=dict(color='rgba(200,200,200,0.2)', dash='dot'), name='Upper BB'), row=1, col=1)
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Lower'], line=dict(color='rgba(200,200
