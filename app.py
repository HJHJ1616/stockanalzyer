import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import google.generativeai as genai

# 🔥 1. 페이지 설정 (숫자 안 잘리게 넓게!)
st.set_page_config(layout="wide", page_title="Quant Dashboard")
st.title("🚀 Quant Dashboard (Ultimate Full Version)")

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
    api_key_input = st.sidebar.text_input("🔑 API Key를 입력하세요 (Secrets 권장):", type="password")
    if api_key_input:
        genai.configure(api_key=api_key_input)
        api_key = api_key_input

# ---------------------------------------------------------
# 1. 사이드바: 매매일지 입력
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
# 2. 데이터 처리 및 환율 계산
# ---------------------------------------------------------
with st.spinner('시장 데이터를 불러오는 중... ⏳'):
    @st.cache_data(ttl=600)
    def fetch_data(ticker_list):
        download_list = ticker_list + ["^GSPC", "KRW=X"]
        data = yf.download(download_list, period="10y", progress=False)['Close']
        if isinstance(data, pd.Series): data = data.to_frame(name=download_list[0])
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
        
        cap_val = pd.Series(0.0, index=raw_data.index)
        cap_val.loc[cap_val.index >= buy_date] = invest_converted
        invested_history = invested_history.add(cap_val, fill_value=0)

        details.append({
            "Ticker": row["Ticker"],
            "Market": row["Market"],
            "Qty": row["Qty"],
            "Avg Buy": row["Price"],
            "Current": raw_data[rt].iloc[-1],
            "Value": val_converted.iloc[-1],
            "Return (%)": ((raw_data[rt].iloc[-1] - row["Price"]) / row["Price"]) * 100
        })

    total_invested = invested_history.iloc[-1]
    current_value = portfolio_history.iloc[-1]
    df_details = pd.DataFrame(details)
    df_details["Weight (%)"] = (df_details["Value"] / current_value * 100).fillna(0)

# ---------------------------------------------------------
# 3. UI 출력 (상단 대시보드)
# ---------------------------------------------------------
st.markdown(f"### 💰 Portfolio Status ({target_currency})")
c1, c2 = st.columns(2)
c1.metric("Total Invested", f"{target_sym}{total_invested:,.0f}")
c2.metric("Current Value", f"{target_sym}{current_value:,.0f}")

st.write("")
c3, c4 = st.columns(2)
c3.metric("Net Profit", f"{target_sym}{current_value-total_invested:,.0f}", delta=f"{(current_value/total_invested-1)*100:.2f}%")
c4.metric("Tickers", f"{len(df_details)}")

st.subheader("📈 Portfolio Growth")
fig_growth = go.Figure()
fig_growth.add_trace(go.Scatter(x=portfolio_history.index, y=portfolio_history, name="Total Value", line=dict(color='#FF4B4B', width=3)))
fig_growth.add_trace(go.Scatter(x=invested_history.index, y=invested_history, name="Invested Capital", line=dict(color='gray', dash='dash')))
st.plotly_chart(fig_growth, use_container_width=True)

# ---------------------------------------------------------
# 4. 리스크 분석 (벤치마크 & 히트맵)
# ---------------------------------------------------------
st.markdown("---")
col_b, col_h = st.columns(2)
with col_b:
    st.subheader("🆚 vs S&P 500")
    my_ret = (portfolio_history / invested_history - 1) * 100
    sp_ret = (sp500_history / sp500_history.loc[earliest_date:].iloc[0] - 1) * 100
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(x=my_ret.index, y=my_ret, name="My Port", line=dict(color='#FF4B4B')))
    fig_b.add_trace(go.Scatter(x=sp_ret.index, y=sp_ret, name="S&P 500", line=dict(color='blue', dash='dot')))
    st.plotly_chart(fig_b, use_container_width=True)

with col_h:
    st.subheader("🔥 Correlation Heatmap")
    st.plotly_chart(px.imshow(raw_data.pct_change().corr(), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)

# ---------------------------------------------------------
# 5. 수익 결산 표 (Holdings Detail 부활!)
# ---------------------------------------------------------
st.subheader("🧾 Holdings Detail")
st.dataframe(
    df_details.style.format({
        "Qty": "{:,.4f}", "Avg Buy": "{:,.2f}", "Current": "{:,.2f}",
        "Value": f"{target_sym}{{:,.0f}}", "Return (%)": "{:,.2f}%", "Weight (%)": "{:,.1f}%"
    }).background_gradient(cmap='RdYlGn', subset=['Return (%)']),
    use_container_width=True
)

# ---------------------------------------------------------
# 6. 기술적 분석 (MA 5/20/60/120/200 + BB + RSI 부활!)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Technical Analysis")
sel_ticker = st.selectbox("분석할 종목 선택", df_details["Ticker"].unique())
rt_sel = ticker_map[sel_ticker]
tech_df = raw_data[rt_sel].to_frame(name="Close").iloc[-500:]

for ma in [5, 20, 60, 120, 200]:
    tech_df[f'MA{ma}'] = tech_df['Close'].rolling(window=ma).mean()
tech_df['Std_20'] = tech_df['Close'].rolling(window=20).std()
tech_df['Upper'] = tech_df['MA20'] + (tech_df['Std_20'] * 2)
tech_df['Lower'] = tech_df['MA20'] - (tech_df['Std_20'] * 2)

delta = tech_df['Close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
tech_df['RSI'] = 100 - (100 / (1 + (gain / loss)))

fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3], subplot_titles=("Price & Indicators", "RSI (14)"))
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Upper'], line=dict(color='rgba(200,200,200,0.2)', dash='dot'), name='Upper BB'), row=1, col=1)
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Lower'], line=dict(color='rgba(200,200,200,0.2)', dash='dot'), name='Lower BB', fill='tonexty'), row=1, col=1)

colors = {'MA5':'pink', 'MA20':'orange', 'MA60':'green', 'MA120':'purple', 'MA200':'darkred'}
for ma, color in colors.items():
    fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df[ma], line=dict(color=color, width=2 if ma=='MA200' else 1), name=ma), row=1, col=1)

fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['Close'], line=dict(color='blue', width=2), name='Price'), row=1, col=1)
fig_tech.add_trace(go.Scatter(x=tech_df.index, y=tech_df['RSI'], line=dict(color='magenta'), name='RSI'), row=2, col=1)
fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1); fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
fig_tech.update_layout(height=800, hovermode="x unified", template="plotly_white")
st.plotly_chart(fig_tech, use_container_width=True)

# ---------------------------------------------------------
# 7. Gemini AI 분석 (안전장치 포함)
# ---------------------------------------------------------
st.markdown("---")
if st.button("🤖 Analyze Portfolio with AI"):
    if not api_key: st.error("❌ API Key를 설정해주세요.")
    else:
        status = st.empty(); status.info("AI 분석 중... ⏳")
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            summary = df_details[["Ticker", "Return (%)", "Weight (%)"]].to_string(index=False)
            response = model.generate_content(f"다음 포트폴리오를 분석하고 투자 조언을 한국어로 해줘:\n{summary}")
            status.empty(); st.success("✅ 분석 완료!"); st.markdown(response.text)
        except Exception as e: status.empty(); st.error(f"❌ 에러: {str(e)}")
