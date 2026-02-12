import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import google.generativeai as genai

st.set_page_config(layout="wide", page_title="Quant Dashboard")
st.title("🚀 Quant Dashboard (ver. 16)")

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
# 1. 사이드바: 매매일지 입력 (엑셀 스타일)
# ---------------------------------------------------------
st.sidebar.header("📝 내 주식 장부 작성")

currency_choice = st.sidebar.radio("🌍 기준 통화 (표시용)", ["달러 ($)", "원화 (₩)"])
sym = "$" if currency_choice == "달러 ($)" else "₩"

st.sidebar.info("아래 표에 보유 종목을 입력하세요. (행 추가 가능)")

# 기본 데이터 (예시)
default_data = pd.DataFrame([
    {"티커": "SCHD", "매수일": datetime(2023, 1, 15), "매수 단가": 75.5, "수량": 100},
    {"티커": "SSO", "매수일": datetime(2023, 6, 20), "매수 단가": 50.0, "수량": 50},
    {"티커": "BTC-USD", "매수일": datetime(2024, 1, 10), "매수 단가": 45000.0, "수량": 0.1},
])

# 엑셀처럼 편집 가능한 데이터 에디터
edited_df = st.sidebar.data_editor(
    default_data,
    num_rows="dynamic", # 행 추가/삭제 가능
    column_config={
        "티커": st.column_config.TextColumn("종목 티커 (예: AAPL)", validate="^[A-Za-z0-9.-]+$"),
        "매수일": st.column_config.DateColumn("매수 날짜", format="YYYY-MM-DD"),
        "매수 단가": st.column_config.NumberColumn(f"매수 단가 ({sym})", min_value=0.01),
        "수량": st.column_config.NumberColumn("보유 수량", min_value=0.0001),
    },
    hide_index=True
)

if edited_df.empty:
    st.warning("👈 사이드바에 최소 1개 이상의 종목을 입력해주세요!")
    st.stop()

# ---------------------------------------------------------
# 2. 데이터 처리 및 계산
# ---------------------------------------------------------
with st.spinner('장부를 분석하고 시장 데이터를 가져오는 중... ⏳'):
    tickers = edited_df["티커"].str.upper().unique().tolist()
    earliest_date = pd.to_datetime(edited_df["매수일"].min())
    start_date_yf = earliest_date - pd.Timedelta(days=365*2) # 차트 여유분 및 지표 계산용
    
    # 데이터 다운로드 (2010년부터 가져와서 AI용으로도 씀)
    raw_data = yf.download(tickers, start="2010-01-01", progress=False)['Close']
    
    # 단일 종목일 경우 Series -> DataFrame 변환
    if isinstance(raw_data, pd.Series):
        raw_data = raw_data.to_frame(name=tickers[0])
    
    # 현재가(가장 최근 데이터)
    current_prices = raw_data.iloc[-1]

    # 포트폴리오 가치 역산 (Time Series)
    # 전체 기간에 대한 빈 프레임 생성
    sim_data = raw_data[raw_data.index >= earliest_date].copy()
    portfolio_history = pd.Series(0.0, index=sim_data.index)
    invested_capital_history = pd.Series(0.0, index=sim_data.index)

    total_invested = 0
    current_portfolio_value = 0
    
    # 종목별 상세 분석용 리스트
    details = []

    for index, row in edited_df.iterrows():
        ticker = row["티커"].upper()
        buy_date = pd.to_datetime(row["매수일"])
        price_at_buy = row["매수 단가"]
        qty = row["수량"]
        
        if ticker not in sim_data.columns:
            continue # 데이터 없는 종목 스킵

        # 1. 총 투자금 계산 (입력한 단가 기준)
        invest_amt = price_at_buy * qty
        total_invested += invest_amt
        
        # 2. 현재 평가금 계산 (시장가 기준)
        curr_price = current_prices[ticker]
        curr_val = curr_price * qty
        current_portfolio_value += curr_val
        
        # 3. 시계열 자산 가치 누적 (매수일 이후부터 가치 반영)
        # 해당 종목의 가격 흐름 * 수량
        asset_value_series = sim_data[ticker] * qty
        # 매수일 이전은 0 처리
        asset_value_series[asset_value_series.index < buy_date] = 0
        portfolio_history += asset_value_series
        
        # 4. 투자 원금 시계열 (매수일에 원금 투입됨)
        capital_series = pd.Series(0.0, index=sim_data.index)
        capital_series[capital_series.index >= buy_date] = invest_amt
        invested_capital_history += capital_series

        details.append({
            "종목": ticker,
            "수량": qty,
            "매수 평균가": price_at_buy,
            "현재가": curr_price,
            "투자 원금": invest_amt,
            "현재 평가금": curr_val,
            "수익률(%)": (curr_price - price_at_buy) / price_at_buy * 100
        })

    # 비중 재계산 (현재 평가금 기준)
    df_details = pd.DataFrame(details)
    df_details["비중(%)"] = (df_details["현재 평가금"] / current_portfolio_value * 100)

    # 수익률 계산
    total_return_money = current_portfolio_value - total_invested
    total_return_pct = (total_return_money / total_invested) * 100 if total_invested > 0 else 0

# ---------------------------------------------------------
# 📊 3. 대시보드 출력
# ---------------------------------------------------------
# 상단 요약
st.markdown("### 💰 내 계좌 현황판")
c1, c2, c3, c4 = st.columns(4)
c1.metric("총 투자 원금", f"{sym}{total_invested:,.0f}")
c2.metric("현재 총 자산", f"{sym}{current_portfolio_value:,.0f}")
c3.metric("순수익금", f"{sym}{total_return_money:,.0f}", delta=f"{total_return_pct:.2f}%")
c4.metric("종목 수", f"{len(df_details)}개")

# 차트: 내 돈 vs 불어난 돈
st.subheader("📈 자산 성장 그래프 (원금 vs 평가금)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio_history.index, y=portfolio_history, mode='lines', name='총 자산 평가액', line=dict(color='#FF4B4B', width=3)))
fig.add_trace(go.Scatter(x=invested_capital_history.index, y=invested_capital_history, mode='lines', name='투입 원금', line=dict(color='gray', dash='dash')))
fig.update_layout(hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# 상세 보유 현황 표
st.subheader("🧾 보유 종목 상세 명세서")
st.dataframe(
    df_details.style.format({
        "수량": "{:,.2f}",
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
# 🔮 4. Gemini AI 포트폴리오 진단
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔮 Gemini AI 투자 애널리스트의 포트폴리오 진단")

# AI에게 보낼 깔끔한 데이터 정리
ai_portfolio_summary = df_details[["종목", "비중(%)", "수익률(%)"]].to_string(index=False)
chart_trend = "우상향" if total_return_pct > 0 else "우하향"

prompt = f"""
당신은 냉철한 퀀트 투자 애널리스트입니다. 아래 사용자의 실제 주식 보유 현황(매매일지)을 보고 진단해주세요.

[계좌 요약]
- 총 투자금: {sym}{total_invested:,.0f}
- 현재 평가금: {sym}{current_portfolio_value:,.0f}
- 총 수익률: {total_return_pct:.2f}%
- 자산 추세: {chart_trend}

[보유 종목 상세]
{ai_portfolio_summary}

[요청사항]
1. **현재 상태 팩트 체크:** 수익이 난 이유(또는 손실 이유)를 종목 비중과 연결해 분석하세요. (예: "SCHD가 든든하게 받쳐주고 있지만, 코인 비중이 너무 커서 변동성이 큽니다.")
2. **비중 리밸런싱 조언:** 현재 비중(%)을 기준으로, 너무 쏠려있는 종목이 있다면 줄이거나 늘리라고 조언하세요.
3. **미래 대응 전략:** 이 포트폴리오가 앞으로의 시장(금리 인하/인상, 경기 침체 등)에서 유리할지 불리할지 예측하세요.

말투는 전문적이지만 이해하기 쉽게, 마크다운으로 작성하세요.
"""

if st.button("🤖 내 장부 AI에게 검사받기 (Click)"):
    with st.spinner("AI가 장부를 꼼꼼히 살피는 중입니다..."):
        try:
            # 모델 자동 선택 로직
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            model_name = 'models/gemini-1.5-flash'
            for m in available_models:
                if 'flash' in m: model_name = m; break
                elif 'pro' in m: model_name = m
            
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            st.success(f"✅ 진단 완료! (Based on {model_name})")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"오류 발생: {e}")
