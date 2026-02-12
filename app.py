import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import google.generativeai as genai

st.set_page_config(layout="wide", page_title="Quant Dashboard")
st.title("🚀 Quant Dashboard (15m update)")

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
# 1. 사이드바: 매매일지 입력 (시장 선택 기능 추가)
# ---------------------------------------------------------
st.sidebar.header("📝 내 주식 장부 작성")

if st.sidebar.button("🔄 현재가 새로고침 (Click)"):
    st.cache_data.clear()
    st.rerun()

currency_choice = st.sidebar.radio("🌍 기준 통화 (표시용)", ["달러 ($)", "원화 (₩)"])
sym = "$" if currency_choice == "달러 ($)" else "₩"

st.sidebar.info("시장(미국/한국)을 선택하면 티커 뒤에 .KS/.KQ가 자동 입력됩니다.")

# 🔥 시장 구분 컬럼 추가
default_data = pd.DataFrame([
    {"시장": "🇺🇸 미국", "티커": "SCHD", "매수일": datetime(2023, 1, 15), "매수 단가": 75.5, "수량": 100},
    {"시장": "🇰🇷 코스피", "티커": "005930", "매수일": datetime(2023, 6, 20), "매수 단가": 70000.0, "수량": 10},
    {"시장": "🇺🇸 코인", "티커": "BTC-USD", "매수일": datetime(2024, 1, 10), "매수 단가": 45000.0, "수량": 0.1},
])

edited_df = st.sidebar.data_editor(
    default_data,
    num_rows="dynamic",
    column_config={
        "시장": st.column_config.SelectboxColumn(
            "시장 선택",
            options=["🇺🇸 미국", "🇰🇷 코스피", "🇰🇷 코스닥", "🇺🇸 코인"],
            required=True
        ),
        "티커": st.column_config.TextColumn("종목 티커 (예: 005930)", validate="^[A-Za-z0-9.-]+$"),
        "매수일": st.column_config.DateColumn("매수 날짜", format="YYYY-MM-DD"),
        "매수 단가": st.column_config.NumberColumn(f"매수 단가 ({sym})", min_value=0.01, format="%.2f"),
        "수량": st.column_config.NumberColumn("보유 수량", min_value=0.0001, format="%.4f"),
    },
    hide_index=True
)

if edited_df.empty:
    st.warning("👈 사이드바에 최소 1개 이상의 종목을 입력해주세요!")
    st.stop()

# ---------------------------------------------------------
# 2. 데이터 처리 및 계산 (자동 접미사 처리)
# ---------------------------------------------------------
with st.spinner('최신 시장 데이터를 가져오는 중... ⏳'):
    
    # 🔥 [핵심] 사용자가 입력한 티커를 야후 파이낸스용으로 변환하는 로직
    final_tickers = []
    
    # 원본 데이터프레임에 '실제티커' 컬럼 추가를 위해 미리 계산
    edited_df["실제티커"] = edited_df["티커"] # 초기값
    
    for index, row in edited_df.iterrows():
        raw_ticker = str(row["티커"]).strip().upper()
        market = row["시장"]
        
        # 이미 .KS나 .KQ를 붙여서 썼다면 그대로 두고, 안 붙였으면 붙여줌
        if market == "🇰🇷 코스피":
            if not raw_ticker.endswith(".KS"):
                raw_ticker += ".KS"
        elif market == "🇰🇷 코스닥":
            if not raw_ticker.endswith(".KQ"):
                raw_ticker += ".KQ"
        
        final_tickers.append(raw_ticker)
        # 변환된 티커를 데이터프레임에 업데이트 (나중에 매칭 위해)
        edited_df.at[index, "실제티커"] = raw_ticker

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
        st.error("데이터 로드 실패. 티커를 확인해주세요.")
        st.stop()

    current_prices = raw_data.iloc[-1]
    last_updated = raw_data.index[-1].strftime('%Y-%m-%d')

    earliest_input_date = pd.to_datetime(edited_df["매수일"].min())
    sim_data = raw_data[raw_data.index >= earliest_input_date].copy()
    
    portfolio_history = pd.Series(0.0, index=sim_data.index)
    invested_capital_history = pd.Series(0.0, index=sim_data.index)

    total_invested = 0.0
    current_portfolio_value = 0.0
    details = []

    for index, row in edited_df.iterrows():
        real_ticker = row["실제티커"] # 변환된 티커 사용
        display_ticker = row["티커"] # 보여줄 때는 입력한 그대로
        
        buy_date = pd.to_datetime(row["매수일"])
        price_at_buy = float(row["매수 단가"])
        qty = float(row["수량"])
        
        if real_ticker not in sim_data.columns:
            st.toast(f"⚠️ '{display_ticker}' 데이터 없음")
            continue

        invest_amt = price_at_buy * qty
        total_invested += invest_amt
        
        curr_price = current_prices[real_ticker]
        curr_val = curr_price * qty
        current_portfolio_value += curr_val
        
        asset_val_series = sim_data[real_ticker] * qty
        asset_val_series.loc[asset_val_series.index < buy_date] = 0.0
        portfolio_history = portfolio_history.add(asset_val_series, fill_value=0)
        
        cap_series = pd.Series(0.0, index=sim_data.index)
        cap_series.loc[cap_series.index >= buy_date] = invest_amt
        invested_capital_history = invested_capital_history.add(cap_series, fill_value=0)

        roi = ((curr_price - price_at_buy) / price_at_buy) * 100 if price_at_buy > 0 else 0
        details.append({
            "종목": display_ticker, # 화면엔 '005930'으로 표시
            "시장": row["시장"],
            "수량": qty,
            "매수 평균가": price_at_buy,
            "현재가": curr_price,
            "투자 원금": invest_amt,
            "현재 평가금": curr_val,
            "수익률(%)": roi
        })

    if total_invested > 0:
        total_return_money = current_portfolio_value - total_invested
        total_return_pct = (total_return_money / total_invested) * 100
    else:
        total_return_money = 0
        total_return_pct = 0
        
    df_details = pd.DataFrame(details)
    if not df_details.empty:
        df_details["비중(%)"] = (df_details["현재 평가금"] / current_portfolio_value * 100).fillna(0)

# ---------------------------------------------------------
# 📊 3. 대시보드 출력
# ---------------------------------------------------------
st.markdown(f"### 💰 내 계좌 현황판 (기준일: {last_updated})")
c1, c2, c3, c4 = st.columns(4)
c1.metric("총 투자 원금", f"{sym}{total_invested:,.0f}")
c2.metric("현재 총 자산", f"{sym}{current_portfolio_value:,.0f}")
c3.metric("순수익금", f"{sym}{total_return_money:,.0f}", delta=f"{total_return_pct:.2f}%")
c4.metric("분석 종목 수", f"{len(df_details)}개")

st.subheader("📈 자산 성장 그래프")
fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio_history.index, y=portfolio_history, mode='lines', name='평가 금액', line=dict(color='#FF4B4B', width=3)))
fig.add_trace(go.Scatter(x=invested_capital_history.index, y=invested_capital_history, mode='lines', name='투자 원금', line=dict(color='gray', dash='dash')))
fig.update_layout(hovermode="x unified", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🧾 보유 종목 상세")
st.dataframe(
    df_details.style.format({
        "수량": "{:,.4f}",
        "매수 평균가": f"{sym}{{:,.2f}}", 
        "현재가": f"{sym}{{:,.2f}}",
        "투자 원금": f"{sym}{{:,.0f}}",
        "현재 평가금": f"{sym}{{:,.0f}}",
        "수익률(%)": "{:,.2f}%",
        "비중(%)": "{:,.1f}%"
    }).background_gradient(cmap='RdYlGn', subset=['수익률(%)']),
    use_container_width=True
)

# ---------------------------------------------------------
# 🔮 4. Gemini AI 진단
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔮 Gemini AI 투자 애널리스트 진단")

ai_portfolio_summary = df_details[["종목", "비중(%)", "수익률(%)"]].to_string(index=False)
chart_trend = "수익 중 (Good)" if total_return_pct > 0 else "손실 중 (Bad)"

prompt = f"""
당신은 냉철한 퀀트 투자 애널리스트입니다. 사용자 계좌를 진단해주세요.

[계좌 요약]
- 총 투자금: {sym}{total_invested:,.0f}
- 현재 평가금: {sym}{current_portfolio_value:,.0f}
- 수익률: {total_return_pct:.2f}% ({chart_trend})

[보유 종목]
{ai_portfolio_summary}

[요청사항]
1. 수익/손실의 주원인을 분석하세요.
2. 현재 비중에서 리스크가 큰 부분을 지적하고, 리밸런싱 아이디어를 주세요.
3. 향후 시장 상황에 따른 대응 전략을 간략히 조언하세요.

마크다운으로 작성해주세요.
"""

if st.button("🤖 AI 진단 요청 (Click)"):
    with st.spinner("AI가 분석 중입니다..."):
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            model_name = 'models/gemini-1.5-flash'
            for m in available_models:
                if 'flash' in m: model_name = m; break
                elif 'pro' in m: model_name = m
            
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            st.success(f"✅ 진단 완료! (Model: {model_name})")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"오류 발생: {e}")
