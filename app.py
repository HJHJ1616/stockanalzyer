import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide", page_title="Quant Dashboard")
st.title("🚀 Quant Dashbaord (Ver. 4)")

# 1. 사이드바: 설정 영역
st.sidebar.header("⚙️ 백테스트 설정")
tickers_input = st.sidebar.text_input("🔍 티커 (쉼표 구분)", "SSO, SCHD, IAU, BTC-USD")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if not tickers:
    st.warning("최소 1개 이상의 티커를 입력해주세요!")
    st.stop()

st.sidebar.subheader("🗓️ 과거 데이터 추출 기간")
start_date = st.sidebar.date_input("시작일", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("종료일", datetime.today())

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
    st.error("해당 기간의 데이터가 없습니다.")
    st.stop()

# 3. 수익률 및 통계 계산
daily_returns = data.pct_change().dropna()
portfolio_daily_return = pd.Series(0.0, index=daily_returns.index)
for ticker in tickers:
    normalized_weight = weights_dict[ticker] / total_weight
    portfolio_daily_return += daily_returns[ticker] * normalized_weight

def calculate_stats(returns_series):
    cum_ret = (1 + returns_series).cumprod()
    total_return = (cum_ret.iloc[-1] - 1) * 100
    trading_days = len(returns_series)
    years_passed = trading_days / 252
    cagr = ((cum_ret.iloc[-1] ** (1 / max(years_passed, 0.01))) - 1) * 100
    roll_max = cum_ret.cummax()
    drawdown = (cum_ret - roll_max) / roll_max
    mdd = drawdown.min() * 100
    return total_return, cagr, mdd

port_tot, port_cagr, port_mdd = calculate_stats(portfolio_daily_return)

# 차트용 누적 수익률
cum_returns = (1 + portfolio_daily_return).cumprod() * 100
cum_returns.name = 'My Portfolio'
all_cum_returns = (1 + daily_returns).cumprod() * 100
all_cum_returns['My Portfolio'] = cum_returns

# 4. 상단 요약 대시보드
st.markdown("---")
st.subheader("📝 1. 과거 백테스트 요약 리포트")
col1, col2, col3 = st.columns(3)
col1.metric("🔥 과거 누적 수익률", f"{port_tot:.2f}%")
col2.metric("📈 연평균 수익률 (CAGR)", f"{port_cagr:.2f}%")
col3.metric("📉 최대 낙폭 (MDD)", f"{port_mdd:.2f}%")

st.subheader("📊 과거 누적 수익률 추이 (시작 = 100)")
fig1 = px.line(all_cum_returns, x=all_cum_returns.index, y=all_cum_returns.columns)
fig1.update_traces(line=dict(width=1), opacity=0.4)
fig1.for_each_trace(lambda trace: trace.update(line=dict(width=4, color='#FF4B4B'), opacity=1.0) if trace.name == 'My Portfolio' else ())
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------------
# 🔥 RESTORED: 개별 종목 vs 포트폴리오 상세 비교표 (V2 기능 부활!)
# ---------------------------------------------------------
st.subheader("📋 개별 종목 vs 포트폴리오 상세 비교표")
stats_data = []

for col in daily_returns.columns:
    tot, cagr, mdd = calculate_stats(daily_returns[col])
    stats_data.append({"종목/포트폴리오": col, "총 누적 수익률(%)": round(tot, 2), "연평균(CAGR %)": round(cagr, 2), "최대 낙폭(MDD %)": round(mdd, 2)})

stats_data.append({"종목/포트폴리오": "⭐️ My Portfolio", "총 누적 수익률(%)": round(port_tot, 2), "연평균(CAGR %)": round(port_cagr, 2), "최대 낙폭(MDD %)": round(port_mdd, 2)})

df_stats = pd.DataFrame(stats_data).set_index("종목/포트폴리오")
st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['총 누적 수익률(%)', '연평균(CAGR %)']).background_gradient(cmap='RdYlGn_r', subset=['최대 낙폭(MDD %)']), use_container_width=True)

# ---------------------------------------------------------
# 🔥 NEW 1: 보유 기간별 승률 분석 (Rolling Returns)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🎯 2. 내가 이 포트폴리오를 샀다면, 돈을 벌 확률은?")
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

# ---------------------------------------------------------
# 🔥 NEW 2: 몬테카를로 미래 시뮬레이션 (Monte Carlo)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🔮 3. 향후 3년 미래 예측 시뮬레이션 (몬테카를로)")
st.write("과거의 평균 수익률과 변동성(위험도)을 바탕으로, 컴퓨터가 1,000가지의 가상 미래를 돌려본 결과입니다.")

sim_days = 252 * 3 
num_simulations = 1000

mu = portfolio_daily_return.mean()
sigma = portfolio_daily_return.std()

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
st.success(f"🤖 **AI 통계 결론:** 현재 세팅하신 비율대로 3년을 더 투자한다면, 평균적으로 **약 {final_avg:.1f}%의 수익**을 기대할 수 있습니다.")
