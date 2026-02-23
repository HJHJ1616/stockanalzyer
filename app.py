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
st.title("🚀 Quant Dashboard (V57. Master)")

# ---------------------------------------------------------
# 🔑 API 및 지능형 AI 엔진 (경량화 로직)
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

def safe_generate_content(prompt):
    # 무료 티어에서 가장 안정적인 1.5-flash 사용
    target_model = "models/gemini-1.5-flash"
    try:
        model = genai.GenerativeModel(target_model)
        # 답변 길이를 제한하여 토큰을 아끼고 속도 향상
        response = model.generate_content(prompt, generation_config={"max_output_tokens": 400})
        return response.text, target_model
    except Exception as e:
        if "429" in str(e):
            raise Exception("🚨 현재 사용량(Quota)이 소진되었습니다. 잠시 후 시도하거나 내일 다시 이용해주세요.")
        raise e

# ---------------------------------------------------------
# 2. 사이드바 및 포트폴리오 데이터 입력 (소수점 완벽 지원)
# ---------------------------------------------------------
st.sidebar.header("📝 Portfolio Inputs")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

target_currency = st.sidebar.radio("💱 Display Currency", ["KRW (₩)", "USD ($)"])
target_sym = "₩" if target_currency == "KRW (₩)" else "$"

# 기본 예시 데이터 (비트코인 소수점 포함)
default_data = pd.DataFrame([
    {"Market": "🇺🇸 US", "Ticker": "SCHD", "Date": datetime(2023, 1, 15), "Price": 75.5, "Qty": 100.0},
    {"Market": "🇰🇷 KOSPI", "Ticker": "005930", "Date": datetime(2023, 6, 20), "Price": 72000.0, "Qty": 10.0},
    {"Market": "🇺🇸 Coin", "Ticker": "BTC-USD", "Date": datetime(2024, 1, 10), "Price": 45000.0, "Qty": 0.012345},
])

edited_df = st.sidebar.data_editor(
    default_data, num_rows="dynamic",
    column_config={
        "Market": st.column_config.SelectboxColumn("Market", options=["🇺🇸 US", "🇰🇷 KOSPI", "🇰🇷 KOSDAQ", "🇺🇸 Coin"], required=True),
        "Ticker": st.column_config.TextColumn("Ticker"),
        "Date": st.column_config.DateColumn("Buy Date"),
        "Price": st.column_config.NumberColumn("Price (Local)", format="%.2f"),
        "Qty": st.column_config.NumberColumn("Qty", step=0.000001, format="%.6f"), # 🛠️ 소수점 6자리 허용
    }, hide_index=True
)

if edited_df.empty:
    st.warning("👈 좌측 사이드바에 데이터를 입력해주세요.")
    st.stop()

# ---------------------------------------------------------
# 3. 데이터 처리 및 계산
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
# 4. 상단 성과 지표 및 성장 차트
# ---------------------------------------------------------
st.markdown(f"### 💰 Portfolio Status ({target_currency})")
c1, c2, c3 = st.columns(3)
c1.metric("Total Invested", f"{target_sym}{total_invested:,.0f}")
c2.metric("Current Value", f"{target_sym}{current_value:,.0f}")
c3.metric("Profit/Loss", f"{target_sym}{current_value-total_invested:,.0f}", delta=f"{(current_value/total_invested-1)*100:.2f}%")

st.subheader("📈 Portfolio Growth")
mask = portfolio_history > 0
fig_growth = go.Figure()
fig_growth.add_trace(go.Scatter(x=portfolio_history[mask].index, y=portfolio_history[mask], name="Value", line=dict(color='#FF4B4B', width=3)))
fig_growth.add_trace(go.Scatter(x=invested_history[mask].index, y=invested_history[mask], name="Capital", line=dict(color='gray', dash='dash')))
st.plotly_chart(fig_growth, use_container_width=True)

col_bench, col_heat = st.columns(2)
with col_bench:
    st.subheader("🆚 vs S&P 500")
    my_ret = (portfolio_history / invested_history - 1) * 100
    sp_ret = (sp500_history.loc[earliest_date:] / sp500_history.loc[earliest_date:].iloc[0] - 1) * 100
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(x=my_ret[mask].index, y=my_ret[mask], name="My Port", line=dict(color='#FF4B4B')))
    fig_b.add_trace(go.Scatter(x=sp_ret.index, y=sp_ret, name="S&P 500", line=dict(color='blue', dash='dot')))
    st.plotly_chart(fig_b, use_container_width=True)

with col_heat:
    st.subheader("🔥 Correlation Heatmap")
    st.plotly_chart(px.imshow(raw_data.pct_change().corr(), text_auto=".2f", color_continuous_scale="RdBu_r"), use_container_width=True)

st.subheader("🧾 Holdings Detail")
st.dataframe(df_details.style.format({"Qty":"{:,.6f}", "Avg Buy":"{:,.2f}", "Current":"{:,.2f}", "Value":f"{target_sym}{{:,.0f}}", "Return (%)":"{:,.2f}%", "Weight (%)":"{:,.1f}%"}).background_gradient(cmap='RdYlGn', subset=['Return (%)']), use_container_width=True)

# ---------------------------------------------------------
# 5. 기술적 분석 (볼린저 밴드 + RSI)
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
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Upper'], line=dict(color='rgba(200,200,200,0)'), showlegend=False), row=1, col=1)
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Lower'], line=dict(color='rgba(200,200,200,0)'), fill='tonexty', fillcolor='rgba(200,200,200,0.2)', showlegend=False), row=1, col=1)
colors = {'MA5':'pink', 'MA20':'orange', 'MA60':'green', 'MA120':'purple', 'MA200':'darkred'}
for ma, color in colors.items(): fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df[ma], line=dict(color=color, width=1), name=ma), row=1, col=1)
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Close'], line=dict(color='blue', width=2), name='Price'), row=1, col=1)
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['RSI'], line=dict(color='magenta'), name='RSI'), row=2, col=1)
fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1); fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
fig_tech.update_layout(height=800, template="plotly_white", hovermode="x unified")
st.plotly_chart(fig_tech, use_container_width=True)

# ---------------------------------------------------------
# 6. 🍏 AI 60일 추세 다이어트 분석
# ---------------------------------------------------------
c_ai1, c_ai2 = st.columns(2)
with c_ai1:
    if st.button(f"🔍 AI {sel_ticker} 60일 추세 분석"):
        if not api_key: st.error("❌ API Key를 설정해주세요.")
        else:
            with st.spinner("최근 두 달간의 흐름을 요약 중..."):
                try:
                    recent_60 = tech_df.tail(60)
                    cur_p = recent_60['Close'].iloc[-1]; start_p = recent_60['Close'].iloc[0]
                    chg = ((cur_p - start_p) / start_p) * 100
                    
                    # 3일 간격 추출로 토큰 다이어트
                    summary = f"""
                    종목: {sel_ticker} | 현재가: {cur_p:.2f} (60일전 대비 {chg:+.2f}%)
                    RSI: {recent_60['RSI'].iloc[-1]:.1f}
                    이평선: MA20({recent_60['MA20'].iloc[-1]:.2f}), MA60({recent_60['MA60'].iloc[-1]:.2f})
                    주가 추이(3일간격): {recent_60['Close'].iloc[::3].tolist()}
                    """
                    prompt = f"{summary}\n위 60일 데이터를 기반으로 중기 추세와 대응 전략을 한국어로 3문장 요약해줘."
                    
                    txt, model_name = safe_generate_content(prompt)
                    st.success(f"✅ 분석 완료 (Model: {model_name})")
                    st.info(txt)
                except Exception as e: st.error(str(e))

with c_ai2:
    if st.button("🤖 포트폴리오 요약 진단"):
        if not api_key: st.error("❌ API Key를 설정해주세요.")
        else:
            with st.spinner("비중 분석 중..."):
                try:
                    port_summary = df_details[["Ticker", "Return (%)", "Weight (%)"]].to_string(index=False)
                    prompt = f"내 포트폴리오 요약:\n{port_summary}\n\n리밸런싱 조언을 딱 한 문단으로 한국어로 해줘."
                    txt, model_name = safe_generate_content(prompt)
                    st.success(f"✅ 진단 완료 (Model: {model_name})")
                    st.markdown(txt)
                except Exception as e: st.error(str(e))
