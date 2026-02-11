import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide", page_title="Quant Dashboard")
st.title("🚀 Quant Dashboard (Ver. 5)")

# 🚨 현금 방치형 Disclaimer (안내문) 추가
st.warning("⚠️ **[백테스트 로직 안내] 현금 방치형 (Cash Drag) 적용:** \n"
           "각 종목별로 설정한 '매도일' 이후(또는 '매수일' 이전)의 해당 자산은 추가적인 수익이나 손실 없이 **수익률 0%의 '현금' 상태로 계좌에 방치(보관)**되는 것으로 계산됩니다. "
           )

# 1. 사이드바: 전체 설정 및 종목별 개별 설정
st.sidebar.header("⚙️ 백테스트 설정")
tickers_input = st.sidebar.text_input("🔍 분석할 티커 (쉼표 구분)", "SSO, SCHD, IAU, BTC-USD")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if not tickers:
    st.warning("최소 1개 이상의 티커를 입력해주세요!")
    st.stop()

st.sidebar.subheader("🗓️ 전체 백테스트 기간 (조회 기간)")
global_start = st.sidebar.date_input("전체 시작일", pd.to_datetime("2020-01-01"))
global_end = st.sidebar.date_input("전체 종료일", datetime.today())

st.sidebar.markdown("---")
st.sidebar.subheader("📦 종목별 상세 설정 (비중 및 매매일)")

ticker_settings = {}
total_weight = 0

# 종목별로 설정 창을 아코디언(Expander) 형태로 생성
for ticker in tickers:
    with st.sidebar.expander(f"🔧 {ticker} 설정", expanded=True):
        w = st.slider(f"비중 (%)", 0, 100, 100 // len(tickers), key=f"weight_{ticker}")
        t_start = st.date_input(f"매수일", global_start, key=f"start_{ticker}")
        t_end = st.date_input(f"매도일", global_end, key=f"end_{ticker}")
        
        # 매수/매도일 오류 방지
        if t_start > t_end:
            st.sidebar.error(f"{ticker}의 매수일이 매도일보다 늦을 수 없습니다!")
            st.stop()
            
        ticker_settings[ticker] = {'weight': w, 'start': pd.to_datetime(t_start), 'end': pd.to_datetime(t_end)}
        total_weight += w

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
    data = load_data(tickers, global_start, global_end)

if data.empty:
    st.error("해당 기간의 데이터가 없습니다.")
    st.stop()

# 3. 실전 Buy & Hold (현금 방치) 수익률 계산
daily_returns = data.pct_change().dropna()

# 전체 포트폴리오의 가치 흐름 (초기 자본을 1.0으로 가정)
portfolio_value = pd.Series(0.0, index=daily_returns.index)
adjusted_cum_returns = pd.DataFrame(index=daily_returns.index)

for ticker in tickers:
    norm_w = ticker_settings[ticker]['weight'] / total_weight
    t_start = ticker_settings[ticker]['start']
    t_end = ticker_settings[ticker]['end']
    
    # 해당 종목의 일일 수익률 복사
    t_ret = daily_returns[ticker].copy()
    
    # 핵심 로직: 매수일 이전, 매도일 이후는 수익률 0% (현금 처리)
    t_ret.loc[t_ret.index < t_start] = 0.0
    t_ret.loc[t_ret.index > t_end] = 0.0
    
    # 이 종목이 할당받은 자본금의 성장 과정
    t_cum = (1 + t_ret).cumprod()
    adjusted_cum_returns[ticker] = t_cum * 100 # 차트 표시용
    
    # 포트폴리오 전체 가치에 합산
    portfolio_value += t_cum * norm_w

# 전체 포트폴리오 가치를 100 기준으로 변환 및 일일 수익률 역산
cum_returns = portfolio_value * 100
cum_returns.name = 'My Portfolio'
portfolio_daily_return = portfolio_value.pct_change().fillna(0)

adjusted_cum_returns['My Portfolio'] = cum_returns

# 통계 계산 함수
def calculate_stats(returns_series, is_price_series=False):
    if is_price_series:
        cum_ret = returns_series / 100
    else:
        cum_ret = (1 + returns_series).cumprod()
        
    total_return = (cum_ret.iloc[-1] - 1) * 100
    trading_days = len(cum_ret)
    years_passed = max(trading_days / 252, 0.01)
    
    # 연평균(CAGR) 계산 시 현금 방치 기간도 시간에 포함 (보유 기간 대비 기회비용 반영)
    cagr = ((cum_ret.iloc[-1] ** (1 / years_passed)) - 1) * 100
    
    roll_max = cum_ret.cummax()
    drawdown = (cum_ret - roll_max) / roll_max
    mdd = drawdown.min() * 100
    return total_return, cagr, mdd

port_tot, port_cagr, port_mdd = calculate_stats(cum_returns, is_price_series=True)

# 4. 상단 요약 대시보드
st.markdown("---")
st.subheader("📝 1. 실전 백테스트 요약 리포트 (현금 방치 반영)")
col1, col2, col3 = st.columns(3)
col1.metric("🔥 과거 누적 수익률", f"{port_tot:.2f}%")
col2.metric("📈 연평균 수익률 (CAGR)", f"{port_cagr:.2f}%")
col3.metric("📉 최대 낙폭 (MDD)", f"{port_mdd:.2f}%")

st.subheader("📊 종목별 매매 타이밍이 반영된 수익률 추이 (시작 = 100)")
fig1 = px.line(adjusted_cum_returns, x=adjusted_cum_returns.index, y=adjusted_cum_returns.columns)
fig1.update_traces(line=dict(width=1), opacity=0.4)
fig1.for_each_trace(lambda trace: trace.update(line=dict(width=4, color='#FF4B4B'), opacity=1.0) if trace.name == 'My Portfolio' else ())
st.plotly_chart(fig1, use_container_width=True)
st.info("💡 **차트 해석:** 개별 종목의 선이 중간에 'ㅡ' 자로 평평해진다면, 해당 기간 동안은 팔고 현금으로 들고 있었다는 뜻입니다.")

# 📋 개별 종목 vs 포트폴리오 상세 비교표
st.subheader("📋 개별 종목 vs 포트폴리오 상세 비교표")
stats_data = []

for col in tickers:
    tot, cagr, mdd = calculate_stats(adjusted_cum_returns[col], is_price_series=True)
    stats_data.append({"종목/포트폴리오": col, "총 누적 수익률(%)": round(tot, 2), "연평균(CAGR %)": round(cagr, 2), "최대 낙폭(MDD %)": round(mdd, 2)})

stats_data.append({"종목/포트폴리오": "⭐️ My Portfolio", "총 누적 수익률(%)": round(port_tot, 2), "연평균(CAGR %)": round(port_cagr, 2), "최대 낙폭(MDD %)": round(port_mdd, 2)})

df_stats = pd.DataFrame(stats_data).set_index("종목/포트폴리오")
st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['총 누적 수익률(%)', '연평균(CAGR %)']).background_gradient(cmap='RdYlGn_r', subset=['최대 낙폭(MDD %)']), use_container_width=True)

# 🎯 승률 분석 (Rolling Returns)
st.markdown("---")
st.subheader("🎯 2. 내가 이 시스템대로 굴린다면, 돈을 벌 확률은?")
periods = {'1개월(단타)': 21, '6개월(스윙)': 126, '1년(장투)': 252, '3년(기절)': 252*3}
win_rates = {}

for label, days in periods.items():
    if len(portfolio_daily_return) > days:
        rolling_ret = portfolio_daily_return.rolling(window=days).apply(lambda x: (1+x).prod() - 1)
        win_rate = (rolling_ret > 0).mean() * 100
        win_rates[label] = f"{win_rate:.1f}%"
    else:
        win_rates[label] = "데이터 부족"

df_win = pd.DataFrame([win_rates], index=['수익 발생 확률(승률)'])
st.table(df_win)

# 🔮 몬테카를로 미래 시뮬레이션
st.markdown("---")
st.subheader("🔮 3. 향후 3년 미래 예측 시뮬레이션 (몬테카를로)")
st.write("포트폴리오에 발생한 '현금 방치 기간'의 0% 수익률(안정성)까지 모두 포함하여 미래를 돌려봅니다.")

sim_days = 252 * 3 
num_simulations = 1000

mu = portfolio_daily_return.mean()
sigma = portfolio_daily_return.std()

if sigma == 0:
    st.warning("변동성이 0입니다. (모든 기간을 현금으로 설정하셨습니다)")
else:
    np.random.seed(42)
    simulated_daily_returns = np.random.normal(mu, sigma, (sim_days, num_simulations))
    simulated_cum_returns = (1 + simulated_daily_returns).cumprod(axis=0) * 100 

    percentile_10 = np.percentile(simulated_cum_returns, 10, axis=1)
    percentile_50 = np.percentile(simulated_cum_returns, 50, axis=1)
    percentile_90 = np.percentile(simulated_cum_returns, 90, axis=1)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(sim_days)) + list(range(sim_days))[::-1],
                              y=list(percentile_90) + list(percentile_10)[::-1],
                              fill='toself', fillcolor='rgba(0,176,246,0.2)', line=dict(color='rgba(255,255,255,0)'),
                              name='예측 범위 (상/하위 10%)'))
    fig2.add_trace(go.Scatter(x=list(range(sim_days)), y=percentile_90, mode='lines', line=dict(color='green', dash='dash'), name='운이 아주 좋을 때 (상위 10%)'))
    fig2.add_trace(go.Scatter(x=list(range(sim_days)), y=percentile_50, mode='lines', line=dict(color='blue', width=3), name='가장 현실적인 평균 (50%)'))
    fig2.add_trace(go.Scatter(x=list(range(sim_days)), y=percentile_10, mode='lines', line=dict(color='red', dash='dash'), name='운이 아주 나쁠 때 (하위 10%)'))

    fig2.update_layout(xaxis_title="미래 경과 일수 (총 3년)", yaxis_title="자산 가치 (현재=100)")
    st.plotly_chart(fig2, use_container_width=True)

    final_avg = percentile_50[-1] - 100
    st.success(f"🤖 **AI 통계 결론:** 이 시스템의 평균 수익률과 현금 비중(변동성 하락 효과)을 고려했을 때, 3년 뒤 평균적으로 **약 {final_avg:.1f}%의 수익**을 기대할 수 있습니다.")
