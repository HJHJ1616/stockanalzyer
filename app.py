import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import google.generativeai as genai

st.set_page_config(layout="wide", page_title="Quant Dashboard")
st.title("🚀 Quant Dashboard (ver. 17)")

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

st.sidebar.info("아래 표에 보유 종목을 입력하세요. (티커는 정확하게!)")

# 기본 데이터 (예시)
default_data = pd.DataFrame([
    {"티커": "SCHD", "매수일": datetime(2023, 1, 15), "매수 단가": 75.5, "수량": 100},
    {"티커": "SSO", "매수일": datetime(2023, 6, 20), "매수 단가": 50.0, "수량": 50},
    {"티커": "BTC-USD", "매수일": datetime(2024, 1, 10), "매수 단가": 45000.0, "수량": 0.1},
])

edited_df = st.sidebar.data_editor(
    default_data,
    num_rows="dynamic",
    column_config={
        "티커": st.column_config.TextColumn("종목 티커 (예: AAPL)", validate="^[A-Za-z0-9.-]+$"),
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
# 2. 데이터 처리 및 계산 (🔥 NaN 해결 핵심 로직)
# ---------------------------------------------------------
with st.spinner('장부를 분석하고 시장 데이터를 가져오는 중... ⏳'):
    tickers = edited_df["티커"].str.strip().str.upper().unique().tolist()
    
    # 1. 데이터 다운로드
    try:
        raw_data = yf.download(tickers, start="2015-01-01", progress=False)['Close']
    except Exception as e:
        st.error(f"데이터 다운로드 중 오류 발생: {e}")
        st.stop()
    
    # 단일 종목일 경우 Series -> DataFrame 변환
    if isinstance(raw_data, pd.Series):
        raw_data = raw_data.to_frame(name=tickers[0])
    
    # 🔥 [중요] NaN(빈 값) 처리 및 시간대(Timezone) 제거
    # 1) 시간대 제거: tz_localize(None)을 해야 사용자 입력 날짜와 비교 가능
    raw_data.index = raw_data.index.tz_localize(None)
    # 2) 빈 값 채우기: 주말/휴일 데이터를 전날 데이터로 채움 (ffill) 후 0으로 채움
    raw_data = raw_data.ffill().fillna(0)

    # 현재가 가져오기 (가장 최근 데이터)
    current_prices = raw_data.iloc[-1]

    # 포트폴리오 계산용 빈 그릇 만들기
    earliest_input_date = pd.to_datetime(edited_df["매수일"].min())
    sim_data = raw_data[raw_data.index >= earliest_input_date].copy()
    
    portfolio_history = pd.Series(0.0, index=sim_data.index)
    invested_capital_history = pd.Series(0.0, index=sim_data.index)

    total_invested = 0.0
    current_portfolio_value = 0.0
    details = []

    for index, row in edited_df.iterrows():
        ticker = row["티커"].strip().upper()
        # 날짜 형식 변환 (시간대 없는 Timestamp로 통일)
        buy_date = pd.to_datetime(row["매수일"])
        price_at_buy = float(row["매수 단가"])
        qty = float(row["수량"])
        
        # 데이터에 없는 티커는 건너뛰기 (에러 방지)
        if ticker not in sim_data.columns:
            st.toast(f"⚠️ 경고: '{ticker}'에 대한 시장 데이터를 찾을 수 없습니다. 티커를 확인하세요.")
            continue

        # 1. 투자 원금 누적
        invest_amt = price_at_buy * qty
        total_invested += invest_amt
        
        # 2. 현재 평가금 누적
        curr_price = current_prices[ticker]
        curr_val = curr_price * qty
        current_portfolio_value += curr_val
        
        # 3. 차트용 시계열 데이터 만들기
        # 해당 종목의 가격 흐름 * 수량
        asset_val_series = sim_data[ticker] * qty
        
        # 🔥 [핵심] 매수일 이전의 가치는 0으로 만듦
        asset_val_series.loc[asset_val_series.index < buy_date] = 0.0
        portfolio_history = portfolio_history.add(asset_val_series, fill_value=0)
        
        # 4. 투자 원금 시계열 (매수일부터 원금 그래프 상승)
        cap_series = pd.Series(0.0, index=sim_data.index)
        cap_series.loc[cap_series.index >= buy_date] = invest_amt
        invested_capital_history = invested_capital_history.add(cap_series, fill_value=0)

        # 상세 정보 저장
        roi = ((curr_price - price_at_buy) / price_at_buy) * 100 if price_at_buy > 0 else 0
        details.append({
            "종목": ticker,
            "수량": qty,
            "매수 평균가": price_at_buy,
            "현재가": curr_price,
            "투자 원금": invest_amt,
            "현재 평가금": curr_val,
            "수익률(%)": roi
        })

    # 최종 계산 (0으로 나누기 방지)
    if total_invested > 0:
        total_return_money = current_portfolio_value - total_invested
        total_return_pct = (total_return_money / total_invested) * 100
    else:
        total_return_money = 0
        total_return_pct = 0
        
    # 데이터프레임 변환
    df_details = pd.DataFrame(details)
    if not df_details.empty:
        df_details["비중(%)"] = (df_details["현재 평가금"] / current_portfolio_value * 100).fillna(0)
    else:
        st.error("유효한 종목이 하나도 없습니다. 티커를 다시 확인해주세요.")
        st.stop()

# ---------------------------------------------------------
# 📊 3. 대시보드 출력
# ---------------------------------------------------------
# 상단 요약
st.markdown("### 💰 내 계좌 현황판")
c1, c2, c3, c4 = st.columns(4)
c1.metric("총 투자 원금", f"{sym}{total_invested:,.0f}")
c2.metric("현재 총 자산", f"{sym}{current_portfolio_value:,.0f}")
c3.metric("순수익금", f"{sym}{total_return_money:,.0f}", delta=f"{total_return_pct:.2f}%")
c4.metric("분석 종목 수", f"{len(df_details)}개")

# 차트
st.subheader("📈 자산 성장 그래프 (원금 vs 평가금)")
fig = go.Figure()
# NaN이 제거된 깔끔한 데이터로 차트 그리기
fig.add_trace(go.Scatter(x=portfolio_history.index, y=portfolio_history, mode='lines', name='총 자산 평가액', line=dict(color='#FF4B4B', width=3)))
fig.add_trace(go.Scatter(x=invested_capital_history.index, y=invested_capital_history, mode='lines', name='투입 원금', line=dict(color='gray', dash='dash')))
fig.update_layout(hovermode="x unified", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# 상세 표
st.subheader("🧾 보유 종목 상세 명세서")
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
# 🔮 4. Gemini AI 포트폴리오 진단
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔮 Gemini AI 투자 애널리스트의 포트폴리오 진단")

ai_portfolio_summary = df_details[["종목", "비중(%)", "수익률(%)"]].to_string(index=False)
chart_trend = "우상향 (수익 구간)" if total_return_pct > 0 else "우하향 (손실 구간)"

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
1. **현재 상태 팩트 체크:** 수익이 난 이유(또는 손실 이유)를 종목 비중과 연결해 분석하세요.
2. **비중 리밸런싱 조언:** 현재 비중(%)을 기준으로, 너무 쏠려있는 종목이 있다면 줄이거나 늘리라고 조언하세요.
3. **미래 대응 전략:** 이 포트폴리오가 앞으로의 시장(금리 인하/인상, 경기 침체 등)에서 유리할지 불리할지 예측하세요.

말투는 전문적이지만 이해하기 쉽게, 마크다운으로 작성하세요.
"""

if st.button("🤖 내 장부 AI에게 검사받기 (Click)"):
    with st.spinner("AI가 장부를 꼼꼼히 살피는 중입니다..."):
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            model_name = 'models/gemini-1.5-flash'
            for m in available_models:
                if 'flash' in m: model_name = m; break
                elif 'pro' in m: model_name = m
            
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            st.success(f"✅ 진단 완료! (Using {model_name})")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"오류 발생: {e}")
