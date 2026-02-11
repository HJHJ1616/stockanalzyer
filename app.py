import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide", page_title="Quant Dashboard")
st.title("🚀 Quant Dashboard (Report)")

# 1. 사이드바: 설정 영역
st.sidebar.header("⚙️ 백테스트 설정")

# 티커 입력
tickers_input = st.sidebar.text_input("🔍 티커 (쉼표 구분)", "SSO, SCHD, IAU, BTC-USD")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if not tickers:
    st.warning("최소 1개 이상의 티커를 입력해주세요!")
    st.stop()

# 기간 설정 (단타/장기 테스트를 위해 추가!)
st.sidebar.subheader("🗓️ 기간 설정")
start_date = st.sidebar.date_input("시작일", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("종료일", datetime.today())

# 비중 설정
st.sidebar.subheader("⚖️ 비중 설정 (%)")
weights_dict = {}
for ticker in tickers:
    default_w = 100 // len(tickers)
    weights_dict[ticker] = st.sidebar.slider(f"{ticker}", 0, 100, default_w)

total_weight = sum(weights_dict.values())
if total_weight == 0:
    st.sidebar.error("비중의 합이 0이 될 수 없습니다.")
    st.stop()

# 2. 데이터 가져오기
@st.cache_data
def load_data(ticker_list, start, end):
    df = yf.download(ticker_list, start=start, end=end, progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame(name=ticker_list[0])
    return df.dropna()

with st.spinner('해당 기간의 시장 데이터를 분석 중입니다... ⏳'):
    data = load_data(tickers, start_date, end_date)

if data.empty:
    st.error("해당 기간의 데이터가 없습니다. 주말이나 휴일인지, 티커가 맞는지 확인해 주세요.")
    st.stop()

# 3. 수익률 및 통계 계산 로직
daily_returns = data.pct_change().dropna()
trading_days = len(daily_returns)
years_passed = trading_days / 252 # 1년의 평균 주식 거래일은 252일

# 포트폴리오 일일 수익률 계산
portfolio_daily_return = pd.Series(0.0, index=daily_returns.index)
for ticker in tickers:
    normalized_weight = weights_dict[ticker] / total_weight
    portfolio_daily_return += daily_returns[ticker] * normalized_weight

# 통계 지표 계산 함수 (CAGR, MDD 등)
def calculate_stats(returns_series):
    cum_ret = (1 + returns_series).cumprod()
    total_return = (cum_ret.iloc[-1] - 1) * 100
    
    # 연평균 수익률 (CAGR)
    cagr = ((cum_ret.iloc[-1] ** (1 / max(years_passed, 0.01))) - 1) * 100
    
    # 최대 낙폭 (MDD: 고점 대비 얼마나 떨어졌었나)
    roll_max = cum_ret.cummax()
    drawdown = (cum_ret - roll_max) / roll_max
    mdd = drawdown.min() * 100
    
    return total_return, cagr, mdd

# 전체 누적 수익률 계산 (차트용)
cum_returns = (1 + portfolio_daily_return).cumprod() * 100
cum_returns.name = 'My Portfolio'
all_cum_returns = (1 + daily_returns).cumprod() * 100
all_cum_returns['My Portfolio'] = cum_returns

# 4. 화면 출력 (대시보드)
st.markdown("---")
st.subheader("📝 백테스트 요약 리포트")
st.write(f"**분석 기간:** {start_date} ~ {end_date} (총 **{trading_days}건**의 일별 거래 데이터, 약 **{years_passed:.1f}년**)")

# 핵심 지표 카드 (눈에 확 들어오게)
port_tot, port_cagr, port_mdd = calculate_stats(portfolio_daily_return)

col1, col2, col3 = st.columns(3)
col1.metric("🔥 총 누적 수익률", f"{port_tot:.2f}%")
col2.metric("📈 연평균 수익률 (CAGR)", f"{port_cagr:.2f}%", help="복리로 매년 평균 몇 %씩 굴러갔는지 보여줍니다.")
col3.metric("📉 최대 낙폭 (MDD)", f"{port_mdd:.2f}%", help="투자 기간 중 고점 대비 가장 심하게 깨졌을 때의 마이너스 비율입니다. (멘탈 스트레스 지수)")

# 차트 그리기
st.subheader("📊 누적 수익률 추이 (시작 = 100)")
fig = px.line(all_cum_returns, x=all_cum_returns.index, y=all_cum_returns.columns, labels={'value':'자산 가치', 'Date':'날짜', 'variable':'종목'})
fig.update_traces(line=dict(width=1), opacity=0.4)
fig.for_each_trace(lambda trace: trace.update(line=dict(width=4, color='#FF4B4B'), opacity=1.0) if trace.name == 'My Portfolio' else ())
st.plotly_chart(fig, use_container_width=True)

# 5. 상세 데이터 표
st.subheader("📋 개별 종목 vs 포트폴리오 상세 비교표")
stats_data = []

# 개별 종목 스탯 계산
for col in daily_returns.columns:
    tot, cagr, mdd = calculate_stats(daily_returns[col])
    stats_data.append({"종목/포트폴리오": col, "총 누적 수익률(%)": round(tot, 2), "연평균(CAGR %)": round(cagr, 2), "최대 낙폭(MDD %)": round(mdd, 2)})

# 포트폴리오 스탯 추가
stats_data.append({"종목/포트폴리오": "⭐️ My Portfolio", "총 누적 수익률(%)": round(port_tot, 2), "연평균(CAGR %)": round(port_cagr, 2), "최대 낙폭(MDD %)": round(port_mdd, 2)})

df_stats = pd.DataFrame(stats_data).set_index("종목/포트폴리오")

# 표 예쁘게 출력
st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['총 누적 수익률(%)', '연평균(CAGR %)']).background_gradient(cmap='RdYlGn_r', subset=['최대 낙폭(MDD %)']), use_container_width=True)

st.info("💡 **어떻게 해석하나요?** \n* **연평균(CAGR)**이 높을수록 돈이 빨리 불어납니다. \n* **최대 낙폭(MDD)**이 0에 가까울수록(마이너스가 작을수록) 하락장에서 방어가 잘 된, 마음 편한 투자입니다.")
