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
st.title("🚀 Quant Dashboard (V31. Anti-Error)")

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
# 2. 사이드바 및 데이터 로딩 (기존 로직 유지)
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

edited_df = st.sidebar.data_editor(default_data, num_rows="dynamic", hide_index=True)

if edited_df.empty:
    st.warning("👈 데이터를 입력해주세요.")
    st.stop()

# 데이터 처리 (상세 로직 생략, 기존 V30과 동일)
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
        rt = ticker_map[row["Ticker"]]
        buy_date = pd.to_datetime(row["Date"])
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
# 3. UI 출력 (메트릭, 차트, 결산 표)
# ---------------------------------------------------------
st.markdown(f"### 💰 Portfolio Status ({target_currency})")
c1, c2, c3 = st.columns(3)
c1.metric("Total Invested", f"{target_sym}{total_invested:,.0f}")
c2.metric("Current Value", f"{target_sym}{current_value:,.0f}")
c3.metric("Profit/Loss", f"{target_sym}{current_value-total_invested:,.0f}", delta=f"{(current_value/total_invested-1)*100:.2f}%")

st.plotly_chart(px.line(portfolio_history, title="Portfolio Growth"), use_container_width=True)
st.subheader("🧾 Holdings Detail")
st.dataframe(df_details.style.format({"Qty":"{:,.4f}", "Avg Buy":"{:,.2f}", "Current":"{:,.2f}", "Value":f"{target_sym}{{:,.0f}}", "Return (%)":"{:,.2f}%", "Weight (%)":"{:,.1f}%"}).background_gradient(cmap='RdYlGn', subset=['Return (%)']), use_container_width=True)

# ---------------------------------------------------------
# 4. 기술적 분석 (RSI, MA, BB 완벽 포함)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Technical Analysis")
sel_ticker = st.selectbox("종목 선택", df_details["Ticker"].unique())
rt_sel = ticker_map[sel_ticker]
tech_df = raw_data[rt_sel].to_frame(name="Close").iloc[-500:]

for ma in [5, 20, 60, 120, 200]: tech_df[f'MA{ma}'] = tech_df['Close'].rolling(window=ma).mean()
tech_df['Std_20'] = tech_df['Close'].rolling(window=20).std()
tech_df['Upper'] = tech_df['MA20'] + (tech_df['Std_20'] * 2); tech_df['Lower'] = tech_df['MA20'] - (tech_df['Std_20'] * 2)
delta = tech_df['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
tech_df['RSI'] = 100 - (100 / (1 + (gain / loss)))

fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Upper'], line=dict(color='rgba(200,200,200,0.2)', dash='dot'), name='Upper BB'), row=1, col=1)
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Lower'], line=dict(color='rgba(200,200,200,0.2)', dash='dot'), name='Lower BB', fill='tonexty'), row=1, col=1)
colors = {'MA5':'pink', 'MA20':'orange', 'MA60':'green', 'MA120':'purple', 'MA200':'darkred'}
for ma, color in colors.items(): fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df[ma], line=dict(color=color, width=2 if ma=='MA200' else 1), name=ma), row=1, col=1)
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Close'], line=dict(color='blue', width=2), name='Price'), row=1, col=1)
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['RSI'], line=dict(color='magenta'), name='RSI'), row=2, col=1)
fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1); fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
fig_tech.update_layout(height=800, template="plotly_white", hovermode="x unified")
st.plotly_chart(fig_tech, use_container_width=True)

# ---------------------------------------------------------
# 5. Gemini AI 분석 (🔥 모델 탐색 로직 대폭 강화)
# ---------------------------------------------------------
st.markdown("---")
if st.button("🤖 Analyze Portfolio with AI"):
    if not api_key: st.error("❌ API Key를 설정해주세요.")
    else:
        status = st.empty(); status.info("사용 가능한 AI 모델을 탐색 중입니다... 🔎")
        try:
            # 1. 사용 가능한 모델 목록에서 최적의 모델 찾기
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            
            # 우선순위: gemini-1.5-flash -> gemini-1.5-pro -> 기타
            target_model = None
            for m in models:
                if 'gemini-1.5-flash' in m: target_model = m; break
            if not target_model:
                for m in models:
                    if 'gemini-1.5-pro' in m: target_model = m; break
            if not target_model and models: target_model = models[0]
            
            if not target_model:
                st.error("❌ 사용 가능한 Gemini 모델을 찾을 수 없습니다.")
            else:
                status.info(f"AI 분석 실행 중... (연결된 모델: {target_model}) ⏳")
                model = genai.GenerativeModel(target_model)
                summary = df_details[["Ticker", "Return (%)", "Weight (%)"]].to_string(index=False)
                response = model.generate_content(f"다음 포트폴리오를 퀀트 관점에서 분석하고 한국어로 조언해줘:\n{summary}")
                status.empty(); st.success(f"✅ 분석 완료! ({target_model})"); st.markdown(response.text)
                
        except Exception as e:
            status.empty(); st.error(f"❌ AI 연결 에러: {str(e)}")
            st.info("💡 팁: API 키가 노출되어 차단되었을 수 있습니다. [Google AI Studio]에서 새 키를 발급받아 교체해보세요.")
