import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

    final_tickers = []
    ticker_map = {} 
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
        ticker_map[row["Ticker"]] = raw_ticker
        edited_df.at[index, "RealTicker"] = raw_ticker

    unique_tickers = list(set(final_tickers))
    
    # 🔥 S&P 500 (^GSPC) 데이터도 같이 가져오기
    @st.cache_data(ttl=600) 
    def get_market_data(ticker_list):
        download_list = ticker_list + ["^GSPC"] # 벤치마크 추가
        try:
            data = yf.download(download_list, period="10y", progress=False)['Close']
            if isinstance(data, pd.Series):
                data = data.to_frame(name=download_list[0])
            data.index = data.index.tz_localize(None)
            return data.ffill().fillna(0)
        except Exception as e:
            return pd.DataFrame()

    raw_data_all = get_market_data(unique_tickers)
    
    if raw_data_all.empty:
        st.error("Failed to load data. Please check tickers.")
        st.stop()

    # S&P 500 분리
    sp500_data = raw_data_all["^GSPC"].copy()
    raw_data = raw_data_all.drop(columns=["^GSPC"], errors='ignore') # 내 종목만 남김

    # 인덱스 정렬
    common_index = raw_data.index.intersection(exchange_rate_history.index)
    raw_data = raw_data.loc[common_index]
    exchange_rate_history = exchange_rate_history.loc[common_index]
    sp500_data = sp500_data.loc[raw_data.index]

    current_prices = raw_data.iloc[-1]
    last_updated = raw_data.index[-1].strftime('%Y-%m-%d %H:%M')

    earliest_input_date = pd.to_datetime(edited_df["Date"].min())
    
    # 시뮬레이션용 데이터 자르기
    sim_data = raw_data[raw_data.index >= earliest_input_date].copy()
    sim_ex_rate = exchange_rate_history[exchange_rate_history.index >= earliest_input_date]["KRW=X"]
    sim_sp500 = sp500_data[sp500_data.index >= earliest_input_date].copy()
    
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

c1, c2 = st.columns(2)
c1.metric("Total Invested", f"{target_sym}{total_invested_converted:,.0f}")
c2.metric("Current Value", f"{target_sym}{current_portfolio_value_converted:,.0f}")

st.write("") 

c3, c4 = st.columns(2)
c3.metric("Net Profit", f"{target_sym}{total_return_money:,.0f}", delta=f"{total_return_pct:.2f}%")
c4.metric("Tickers", f"{len(df_details)}")

st.subheader("📈 Asset Growth (Converted)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio_history.index, y=portfolio_history, mode='lines', name='Total Value', line=dict(color='#FF4B4B', width=3)))
fig.add_trace(go.Scatter(x=invested_capital_history.index, y=invested_capital_history, mode='lines', name='Invested Capital', line=dict(color='gray', dash='dash')))
fig.update_layout(hovermode="x unified", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# 🔥 [NEW] 1. 벤치마크 비교 (VS S&P 500)
st.markdown("---")
col_bench, col_corr = st.columns(2)

with col_bench:
    st.subheader("🆚 Benchmark (vs S&P 500)")
    # 수익률 정규화 (시작일 기준 0%로 맞춤)
    my_cum_ret = (portfolio_history / invested_capital_history - 1) * 100
    sp500_cum_ret = (sim_sp500 / sim_sp500.iloc[0] - 1) * 100
    
    fig_bench = go.Figure()
    fig_bench.add_trace(go.Scatter(x=my_cum_ret.index, y=my_cum_ret, mode='lines', name='My Portfolio', line=dict(color='#FF4B4B', width=2)))
    fig_bench.add_trace(go.Scatter(x=sp500_cum_ret.index, y=sp500_cum_ret, mode='lines', name='S&P 500', line=dict(color='blue', dash='dot')))
    fig_bench.update_layout(hovermode="x unified", template="plotly_white", yaxis_title="Cumulative Return (%)", height=400)
    st.plotly_chart(fig_bench, use_container_width=True)

# 🔥 [NEW] 2. 상관관계 히트맵 (Correlation Heatmap)
with col_corr:
    st.subheader("🔥 Correlation Heatmap")
    # 상관관계 계산
    corr_matrix = sim_data.pct_change().corr()
    
    fig_corr = px.imshow(corr_matrix, 
                         text_auto=True, 
                         aspect="auto", 
                         color_continuous_scale="RdBu_r", # 빨강=높음, 파랑=낮음
                         zmin=-1, zmax=1)
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)


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

# 🔥 [NEW] 3. 기술적 분석 (RSI & Bollinger Bands)
st.markdown("---")
st.subheader("📊 Technical Analysis (RSI & Bollinger Bands)")
st.info("Select an asset to see detailed charts.")

selected_ticker_display = st.selectbox("Select Asset", df_details["Ticker"].unique())
selected_real_ticker = ticker_map[selected_ticker_display] 

if selected_real_ticker in raw_data.columns:
    tech_data = raw_data[selected_real_ticker].copy().to_frame(name="Close")
    tech_data = tech_data.iloc[-252:] 

    # BB
    tech_data['SMA_20'] = tech_data['Close'].rolling(window=20).mean()
    tech_data['Std_20'] = tech_data['Close'].rolling(window=20).std()
    tech_data['Upper_BB'] = tech_data['SMA_20'] + (tech_data['Std_20'] * 2)
    tech_data['Lower_BB'] = tech_data['SMA_20'] - (tech_data['Std_20'] * 2)

    # RSI
    delta = tech_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    tech_data['RSI'] = 100 - (100 / (1 + rs))

    fig_tech = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                             vertical_spacing=0.1, row_heights=[0.7, 0.3],
                             subplot_titles=(f"{selected_ticker_display} Price & Bollinger Bands", "RSI (14)"))

    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tech_data['Upper_BB'], line=dict(color='gray', width=1, dash='dot'), name='Upper Band'), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tech_data['Lower_BB'], line=dict(color='gray', width=1, dash='dot'), name='Lower Band', fill='tonexty', fillcolor='rgba(200,200,200,0.1)'), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tech_data['SMA_20'], line=dict(color='orange', width=1), name='20 MA'), row=1, col=1)
    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tech_data['Close'], line=dict(color='blue', width=2), name='Price'), row=1, col=1)

    fig_tech.add_trace(go.Scatter(x=tech_data.index, y=tech_data['RSI'], line=dict(color='purple', width=2), name='RSI'), row=2, col=1)
    fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig_tech.update_layout(height=600, showlegend=True, hovermode="x unified")
    st.plotly_chart(fig_tech, use_container_width=True)
else:
    st.warning("Not enough data to calculate indicators.")

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
2. Compare the portfolio against general market trends (S&P 500).
3. Evaluate diversification based on asset correlations (mention if assets are too correlated).
4. Provide a strategy using technical indicators (RSI, Bollinger Bands) if applicable.

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
