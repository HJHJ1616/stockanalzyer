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
st.title("🚀 Quant Dashboard (V49. Master)")

# ---------------------------------------------------------
# 🔑 API 및 모델 설정
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

# 🛠️ [지능형 모델 자동 매칭 함수]
def safe_generate_content(prompt):
    try:
        # 1. 현재 이 API Key로 사용 가능한 모델 리스트를 실시간으로 가져옵니다.
        available_models = [m.name for m in genai.list_models() 
                           if 'generateContent' in m.supported_generation_methods]
        
        # 2. 효진 님에게 가장 좋은 모델 순서대로 우선순위 정하기
        # 최신 2.0 -> 안정적인 1.5 순으로 리스트를 만듭니다.
        priority_list = [
            "models/gemini-2.0-flash", 
            "models/gemini-1.5-flash", 
            "models/gemini-1.5-flash-latest",
            "models/gemini-pro"
        ]
        
        # 3. 리스트에 있는 모델 중 실제 사용 가능한 첫 번째 모델을 선택
        target_model = None
        for p_model in priority_list:
            if p_model in available_models:
                target_model = p_model
                break
        
        # 만약 우선순위에 없으면 리스트 중 아무거나 첫 번째 거라도 씁니다.
        if not target_model and available_models:
            target_model = available_models[0]
            
        if not target_model:
            raise Exception("사용 가능한 Gemini 모델이 계정에 없습니다.")

        # 4. 선택된 모델로 분석 진행
        model = genai.GenerativeModel(target_model)
        response = model.generate_content(prompt)
        return response.text, target_model # 모델 이름도 같이 반환해서 확인용으로 씀

    except Exception as e:
        raise Exception(f"AI 엔진 오류: {str(e)}")

# ---------------------------------------------------------
# 🔍 AI 분석 버튼 (자동 모델 매칭 적용)
# ---------------------------------------------------------
if st.button(f"🔍 AI {sel_ticker} 분석"):
    if not api_key: st.error("❌ API Key를 설정해주세요.")
    else:
        status = st.empty()
        status.info("최적의 AI 모델을 찾는 중...")
        try:
            l_p, l_r = tech_df['Close'].iloc[-1], tech_df['RSI'].iloc[-1]
            prompt = f"{sel_ticker} 현재가 {l_p:.2f}, RSI {l_r:.2f}. 투자 전략 요약해줘."
            
            # 여기서 자동 매칭 발생!
            result_text, used_model = safe_generate_content(prompt)
            
            status.empty()
            st.success(f"✅ 분석 완료 (사용 모델: {used_model})")
            st.info(result_text)
        except Exception as e:
            status.empty()
            if "429" in str(e):
                st.error("🚨 사용량 초과! 30초만 쉬었다가 다시 눌러주세요.")
            else:
                st.error(f"⚠️ {str(e)}")

# ---------------------------------------------------------
# 2. 사이드바 입력 (소수점 지원)
# ---------------------------------------------------------
st.sidebar.header("📝 Portfolio Inputs")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

target_currency = st.sidebar.radio("💱 Display Currency", ["KRW (₩)", "USD ($)"])
target_sym = "₩" if target_currency == "KRW (₩)" else "$"

default_data = pd.DataFrame([
    {"Market": "🇺🇸 US", "Ticker": "SCHD", "Date": datetime(2023, 1, 15), "Price": 75.5, "Qty": 100.0},
    {"Market": "🇰🇷 KOSPI", "Ticker": "005930", "Date": datetime(2023, 6, 20), "Price": 72000.0, "Qty": 10.0},
    {"Market": "🇺🇸 Coin", "Ticker": "BTC-USD", "Date": datetime(2024, 1, 10), "Price": 45000.0, "Qty": 0.015},
])

edited_df = st.sidebar.data_editor(
    default_data,
    num_rows="dynamic",
    column_config={
        "Market": st.column_config.SelectboxColumn("Market", options=["🇺🇸 US", "🇰🇷 KOSPI", "🇰🇷 KOSDAQ", "🇺🇸 Coin"], required=True),
        "Ticker": st.column_config.TextColumn("Ticker"),
        "Date": st.column_config.DateColumn("Buy Date"),
        "Price": st.column_config.NumberColumn("Price (Local)", format="%.2f"),
        "Qty": st.column_config.NumberColumn("Qty", step=0.000001, format="%.6f"),
    },
    hide_index=True
)

if edited_df.empty:
    st.warning("👈 데이터를 입력해주세요.")
    st.stop()

# ---------------------------------------------------------
# 3. 데이터 로딩 및 계산
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
        qty = float(row["Qty"])
        val_native = raw_data[rt] * qty
        if target_currency == "KRW (₩)":
            val_converted = val_native * exchange_rate_history if is_usd else val_native
            invest_converted = (float(row["Price"]) * qty) * (current_ex_rate if is_usd else 1.0)
        else:
            val_converted = val_native if is_usd else val_native / exchange_rate_history
            invest_converted = (float(row["Price"]) * qty) if is_usd else (float(row["Price"]) * qty) / current_ex_rate
        val_converted.loc[val_converted.index < buy_date] = 0.0
        portfolio_history = portfolio_history.add(val_converted, fill_value=0)
        cap_val = pd.Series(0.0, index=raw_data.index); cap_val.loc[cap_val.index >= buy_date] = invest_converted
        invested_history = invested_history.add(cap_val, fill_value=0)
        details.append({"Ticker": row["Ticker"], "Qty": qty, "Avg Buy": row["Price"], "Current": raw_data[rt].iloc[-1], "Value": val_converted.iloc[-1], "Return (%)": ((raw_data[rt].iloc[-1] - row["Price"]) / row["Price"]) * 100})

    total_invested = invested_history.iloc[-1]
    current_value = portfolio_history.iloc[-1]
    df_details = pd.DataFrame(details)
    df_details["Weight (%)"] = (df_details["Value"] / current_value * 100).fillna(0)

# ---------------------------------------------------------
# 4. 차트 출력 영역
# ---------------------------------------------------------
st.markdown(f"### 💰 Portfolio Status ({target_currency})")
c1, c2, c3 = st.columns(3)
c1.metric("Total Invested", f"{target_sym}{total_invested:,.0f}")
c2.metric("Current Value", f"{target_sym}{current_value:,.0f}")
c3.metric("Profit/Loss", f"{target_sym}{current_value-total_invested:,.0f}", delta=f"{(current_value/total_invested-1)*100:.2f}%")

st.subheader("📈 Portfolio Growth")
mask = portfolio_history > 0
fig_growth = go.Figure()
fig_growth.add_trace(go.Scatter(x=portfolio_history[mask].index, y=portfolio_history[mask], name="자산 가치", line=dict(color='#FF4B4B', width=3)))
fig_growth.add_trace(go.Scatter(x=invested_history[mask].index, y=invested_history[mask], name="투자 원금", line=dict(color='gray', dash='dash')))
st.plotly_chart(fig_growth, use_container_width=True)

col_bench, col_heat = st.columns(2)
with col_bench:
    st.subheader("🆚 vs S&P 500")
    my_ret = (portfolio_history / invested_history - 1) * 100
    sp_ret = (sp500_history.loc[earliest_date:] / sp500_history.loc[earliest_date:].iloc[0] - 1) * 100
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(x=my_ret[mask].index, y=my_ret[mask], name="내 포트폴리오", line=dict(color='#FF4B4B')))
    fig_b.add_trace(go.Scatter(x=sp_ret.index, y=sp_ret, name="S&P 500", line=dict(color='blue', dash='dot')))
    st.plotly_chart(fig_b, use_container_width=True)

with col_heat:
    st.subheader("🔥 Correlation Heatmap")
    st.plotly_chart(px.imshow(raw_data.pct_change().corr(), text_auto=".2f", color_continuous_scale="RdBu_r"), use_container_width=True)

st.subheader("🧾 Holdings Detail")
st.dataframe(df_details.style.format({"Qty":"{:,.6f}", "Avg Buy":"{:,.2f}", "Current":"{:,.2f}", "Value":f"{target_sym}{{:,.0f}}", "Return (%)":"{:,.2f}%", "Weight (%)":"{:,.1f}%"}).background_gradient(cmap='RdYlGn', subset=['Return (%)']), use_container_width=True)

# ---------------------------------------------------------
# 5. 기술적 분석 (에러 수정됨!)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Detailed Technical Analysis")
sel_ticker = st.selectbox("분석 종목 선택", df_details["Ticker"].unique())
rt_sel = ticker_map[sel_ticker]; tech_df = raw_data[rt_sel].to_frame(name="Close").iloc[-500:]

for ma in [5, 20, 60, 120, 200]: tech_df[f'MA{ma}'] = tech_df['Close'].rolling(window=ma).mean()
tech_df['Std_20'] = tech_df['Close'].rolling(window=20).std()
tech_df['Upper'] = tech_df['MA20'] + (tech_df['Std_20'] * 2)
tech_df['Lower'] = tech_df['MA20'] - (tech_df['Std_20'] * 2)
delta = tech_df['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
tech_df['RSI'] = 100 - (100 / (1 + (gain / loss)))

fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

# ✅ 에러 수정 포인트: showlegend 위치를 line 밖으로 뺐습니다.
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Upper'], line=dict(color='rgba(200,200,200,0)'), name='Upper BB', showlegend=False), row=1, col=1)
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Lower'], line=dict(color='rgba(200,200,200,0)'), fill='tonexty', fillcolor='rgba(200,200,200,0.2)', name='BB Range', showlegend=False), row=1, col=1)

colors = {'MA5':'pink', 'MA20':'orange', 'MA60':'green', 'MA120':'purple', 'MA200':'darkred'}
for ma, color in colors.items():
    fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df[ma], line=dict(color=color, width=1), name=ma), row=1, col=1)

fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Close'], line=dict(color='blue', width=2), name='Price'), row=1, col=1)
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['RSI'], line=dict(color='magenta'), name='RSI'), row=2, col=1)
fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
fig_tech.update_layout(height=800, template="plotly_white", hovermode="x unified")
st.plotly_chart(fig_tech, use_container_width=True)

# ---------------------------------------------------------
# 6. AI 분석
# ---------------------------------------------------------
col_ai1, col_ai2 = st.columns(2)
with col_ai1:
    if st.button(f"🔍 AI {sel_ticker} 기술적 분석"):
        if not api_key: st.error("❌ API Key를 설정해주세요.")
        else:
            with st.spinner("분석 중..."):
                try:
                    prompt = f"{sel_ticker} 현재가 {tech_df['Close'].iloc[-1]:.2f}, RSI {tech_df['RSI'].iloc[-1]:.2f}. 대응 전략 요약해줘."
                    st.info(safe_generate_content(prompt))
                except Exception as e: st.error(f"AI 에러: {str(e)}")

with col_ai2:
    if st.button("🤖 전체 포트폴리오 진단"):
        if not api_key: st.error("❌ API Key를 설정해주세요.")
        else:
            with st.spinner("진단 중..."):
                try:
                    summary = df_details[["Ticker", "Return (%)", "Weight (%)"]].to_string(index=False)
                    st.success("진단 완료!"); st.markdown(safe_generate_content(f"포트폴리오 분석해줘:\n{summary}"))
                except Exception as e: st.error(f"AI 에러: {str(e)}")
