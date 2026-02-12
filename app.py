import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import google.generativeai as genai

# 1. 페이지 설정
st.set_page_config(layout="wide", page_title="Quant Dashboard")
st.title("🚀 Quant Dashboard (V27. Stable Release)")

# ---------------------------------------------------------
# 🔑 API 키 로딩 및 AI 설정
# ---------------------------------------------------------
# 팁: Streamlit Cloud Secrets에 [general] GEMINI_API_KEY = "키" 형태로 저장하세요.
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
# 2. 사이드바: 매매일지 입력
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
        "Ticker": st.column_config.TextColumn("Ticker", validate="^[A-Za-z0-9.-]+$"),
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
        data.index = data.index.tz_localize(None)
        return data.ffill()

    ticker_map = {}
    final_tickers = []
    for idx, row in edited_df.iterrows():
        rt = str(row["Ticker"]).strip().upper()
        if row["Market"] == "🇰🇷 KOSPI" and not rt.endswith(".KS"): rt += ".KS"
        elif row["Market"] == "🇰🇷 KOSDAQ" and not rt.endswith(".KQ"): rt += ".KQ"
        ticker_map[row["Ticker"]] = rt
        final_tickers.append(rt)

    raw_data_all = fetch_data(list(set(final_tickers)))
    
    # 데이터 분리
    exchange_rate_history = raw_data_all["KRW=X"]
    sp500_history = raw_data_all["^GSPC"]
    raw_data = raw_data_all.drop(columns=["KRW=X", "^GSPC"], errors='ignore')
    
    current_ex_rate = exchange_rate_history.iloc[-1]
    earliest_date = pd.to_datetime(edited_df["Date"].min())
    
    portfolio_history = pd.Series(0.0, index=raw_data.index)
    invested_history = pd.Series(0.0, index=raw_data.index)
    details = []

    for idx, row in edited_df.iterrows():
        rt = ticker_map[row["Ticker"]]
        buy_date = pd.to_datetime(row["Date"])
        is_usd = row["Market"] in ["🇺🇸 US", "🇺🇸 Coin"]
        
        # 포트폴리오 가치 계산
        val_native = raw_data[rt] * float(row["Qty"])
        if target_currency == "KRW (₩)":
            val_converted = val_native * exchange_rate_history if is_usd else val_native
            invest_converted = (float(row["Price"]) * float(row["Qty"])) * current_ex_rate if is_usd else (float(row["Price"]) * float(row["Qty"]))
        else:
            val_converted = val_native if is_usd else val_native / exchange_rate_history
            invest_converted = (float(row["Price"]) * float(row["Qty"])) if is_usd else (float(row["Price"]) * float(row["Qty"])) / current_ex_rate

        val_converted.loc[val_converted.index < buy_date] = 0.0
        portfolio_history = portfolio_history.add(val_converted, fill_value=0)
        
        cap_val = pd.Series(0.0, index=raw_data.index)
        cap_val.loc[cap_val.index >= buy_date] = invest_converted
        invested_history = invested_history.add(cap_val, fill_value=0)

        details.append({
            "Ticker": row["Ticker"],
            "Value": val_converted.iloc[-1],
            "Return (%)": ((raw_data[rt].iloc[-1] - row["Price"]) / row["Price"]) * 100
        })

    total_invested = invested_history.iloc[-1]
    current_value = portfolio_history.iloc[-1]
    df_details = pd.DataFrame(details)

# ---------------------------------------------------------
# 4. UI 출력 (메트릭 및 차트)
# ---------------------------------------------------------
st.markdown(f"### 💰 Portfolio ({target_currency})")
c1, c2 = st.columns(2)
c1.metric("Total Invested", f"{target_sym}{total_invested:,.0f}")
c2.metric("Current Value", f"{target_sym}{current_value:,.0f}")

st.plotly_chart(px.line(portfolio_history, title="Portfolio Growth"), use_container_width=True)

# 상관관계 히트맵
st.subheader("🔥 Correlation Heatmap")
st.plotly_chart(px.imshow(raw_data.pct_change().corr(), text_auto=True, color_continuous_scale="RdBu_r"), use_container_width=True)

# 기술적 분석 (MA 200 포함)
st.subheader("📊 Technical Analysis")
sel_ticker = st.selectbox("종목 선택", df_details["Ticker"].unique())
rt_sel = ticker_map[sel_ticker]
tech_df = raw_data[rt_sel].to_frame(name="Close").iloc[-500:]

for ma in [5, 20, 60, 120, 200]:
    tech_df[f'MA{ma}'] = tech_df['Close'].rolling(window=ma).mean()

fig_tech = go.Figure()
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Close'], name="Price", line=dict(color='blue', width=2)))
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['MA200'], name="200 MA", line=dict(color='red', width=3)))
st.plotly_chart(fig_tech, use_container_width=True)

# ---------------------------------------------------------
# 🔮 5. Gemini AI 분석 (무한 로딩 방지 강화)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔮 Gemini AI Analyst")

if st.button("🤖 Analyze Portfolio with AI"):
    if not api_key:
        st.error("❌ API Key가 설정되지 않았습니다. Secrets를 확인해주세요.")
    else:
        status = st.empty()
        status.info("AI 분석 중... 잠시만 기다려주세요. ⏳")
        try:
            # 모델 탐색 및 설정
            model_name = 'gemini-1.5-flash'
            model = genai.GenerativeModel(model_name)
            
            summary = df_details.to_string(index=False)
            prompt = f"다음 포트폴리오의 수익률과 종목 구성을 분석하고 투자 조언을 한국어로 해줘:\n{summary}"
            
            response = model.generate_content(prompt)
            status.empty()
            st.success(f"✅ 분석 완료 (Model: {model_name})")
            st.markdown(response.text)
            
        except Exception as e:
            status.empty()
            st.error(f"❌ 에러 발생: {str(e)}")
            st.info("💡 API 키가 차단되었거나, 모델명이 다를 수 있습니다.")
