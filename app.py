import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide", page_title="Quant Dashboard")
st.title("🚀 Quant Dashboard (Ver. 8)")

st.warning("⚠️ **[백테스트 로직 안내] 현금 방치형 (Cash Drag) 적용:** \n"
           "설정한 '매도일' 이후(또는 '매수일' 이전)의 자산은 추가 손익 없이 **수익률 0%의 '현금' 상태로 방치**되는 것으로 계산됩니다.")

# 1. 사이드바: 전체 설정 및 종목별 개별 설정
st.sidebar.header("⚙️ 백테스트 설정")

currency_choice = st.sidebar.radio("🌍 기준 통화 선택", ["원화 (₩)", "달러 ($)"])
if currency_choice == "원화 (₩)":
    sym = "₩"
    init_val = 10000000
    step_val = 1000000
else:
    sym = "$"
    init_val = 10000
    step_val = 1000

initial_investment = st.sidebar.number_input(f"💰 총 초기 투자금 ({sym})", min_value=100, value=init_val, step=step_val)

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

for ticker in tickers:
    with st.sidebar.expander(f"🔧 {ticker} 설정", expanded=True):
        w = st.slider(f"비중 (%)", 0, 100, 100 // len(tickers), key=f"weight_{ticker}")
        t_start = st.date_input(f"매수일", global_start, key=f"start_{ticker}")
        t_end = st.date_input(f"매도일", global_end, key=f"end_{ticker}")
        
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

with st.spinner('시장 데이터를 분석 중입니다... ⏳'):
    data = load_data(tickers, global_start, global_end)

if data.empty:
    st.error("해당 기간의 데이터가 없습니다.")
    st.stop()

# 3. 실전 Buy & Hold 수익률 계산
daily_returns = data.pct_change().dropna()
portfolio_value = pd.Series(0.0, index=daily_returns.index)
adjusted_cum_returns = pd.DataFrame(index=daily_returns.index)

receipt_data = []

for ticker in tickers:
    norm_w = ticker_settings[ticker]['weight'] / total_weight
    t_start = ticker_settings[ticker]['start']
    t_end = ticker_settings[ticker]['end']
    
    t_ret = daily_returns[ticker].copy()
    t_ret.loc[t_ret.index < t_start] = 0.0
    t_ret.loc[t_ret.index > t_end] = 0.0
    
    t_cum = (1 + t_ret).cumprod()
    adjusted_cum_returns[ticker] = t_cum * 100 
    portfolio_value += t_cum * norm_w

    # 🧾 영수증용 데이터 추출
    valid_dates = data[ticker].dropna().index
    
    try:
        buy_date = valid_dates[valid_dates >= t_start].min()
        sell_date = valid_dates[valid_dates <= t_end].max()
        
        buy_price = data.loc[buy_date, ticker]
        sell_price = data.loc[sell_date, ticker]
        
        allocated_cash = initial_investment * norm_w
        final_cash = allocated_cash * (sell_price / buy_price)
        profit_cash = final_cash - allocated_cash
        
        receipt_data.append({
            "종목": ticker,
            "매수일": buy_date.strftime('%Y-%m-%d'),
            "매수 단가 (현지)": round(buy_price, 2),
            "매도일": sell_date.strftime('%Y-%m-%d'),
            "매도 단가 (현지)": round(sell_price, 2),
            f"투자 원금({sym})": round(allocated_cash, 2),
            f"최종 평가액({sym})": round(final_cash, 2),
            f"손익금({sym})": round(profit_cash, 2),
            "수익률(%)": round((sell_price/buy_price - 1)*100, 2)
        })
    except:
        pass 

cum_returns = portfolio_value * 100
cum_returns.name = 'My Portfolio'
portfolio_daily_return = portfolio_value.pct_change().fillna(0)
adjusted_cum_returns['My Portfolio'] = cum_returns

def calculate_stats(returns_series, is_price_series=False):
    if is_price_series:
        cum_ret = returns_series / 100
    else:
        cum_ret = (1 + returns_series).cumprod()
        
    total_return = (cum_ret.iloc[-1] - 1) * 100
    trading_days = len(cum_ret)
    years_passed = max(trading_days / 252, 0.01)
    cagr = ((cum_ret.iloc[-1] ** (1 / years_passed)) - 1) * 100
    roll_max = cum_ret.cummax()
    drawdown = (cum_ret - roll_max) / roll_max
    mdd = drawdown.min() * 100
    return total_return, cagr, mdd

port_tot, port_cagr, port_mdd = calculate_stats(cum_returns, is_price_series=True)

# 4. 상단 요약 대시보드
st.markdown("---")
st.subheader("📝 1. 실전 백테스트 요약 리포트")
col1, col2, col3 = st.columns(3)
col1.metric("🔥 과거 누적 수익률", f"{port_tot:.2f}%")
col2.metric("📈 연평균 수익률 (CAGR)", f"{port_cagr:.2f}%")
col3.metric("📉 최대 낙폭 (MDD)", f"{port_mdd:.2f}%")

st.subheader("📊 종목별 매매 타이밍이 반영된 수익률 추이 (시작 = 100)")

# 🔥 NEW: 가장 빠른 매수일을 찾아서 차트의 시작점으로 자르기
earliest_buy_date = min([settings['start'] for settings in ticker_settings.values()])
chart_data = adjusted_cum_returns[adjusted_cum_returns.index >= earliest_buy_date]

fig1 = px.line(chart_data, x=chart_data.index, y=chart_data.columns)
fig1.update_traces(line=dict(width=1), opacity=0.4)
fig1.for_each_trace(lambda trace: trace.update(line=dict(width=4, color='#FF4B4B'), opacity=1.0) if trace.name == 'My Portfolio' else ())
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------------
# 🔥 영수증 
# ---------------------------------------------------------
st.markdown("---")
st.subheader(f"🧾 2. 과거 가상 매매 결산 영수증 (기준 통화: {sym})")
st.write("설정한 비중(%)에 따라 초기 투자금을 쪼개어 투자했을 때의 **실제 결산 금액**입니다. (매수/매도 단가는 해당 국가의 현지 통화 기준입니다)")

if receipt_data:
    df_receipt = pd.DataFrame(receipt_data).set_index("종목")
    
    total_principal = df_receipt[f"투자 원금({sym})"].sum()
    total_final = df_receipt[f"최종 평가액({sym})"].sum()
    total_profit = df_receipt[f"손익금({sym})"].sum()
    
    st.dataframe(df_receipt.style.format("{:,.2f}", subset=["매수 단가 (현지)", "매도 단가 (현지)", f"투자 원금({sym})", f"최종 평가액({sym})", f"손익금({sym})", "수익률(%)"]).background_gradient(cmap='RdYlGn', subset=[f'손익금({sym})']), use_container_width=True)
    
    st.success(f"결산 완료: **{sym}{total_principal:,.0f}** 를 투자하여 총 **{sym}{total_profit:,.0f}** 의 수익을 얻었으며, 최종 자산은 **{sym}{total_final:,.0f}** 가 되었습니다.")

# ---------------------------------------------------------
# 🎯 승률 분석
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🎯 3. 내가 이 시스템대로 굴린다면, 돈을 벌 확률은?")
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
# 🔮 몬테카를로 
# ---------------------------------------------------------
st.markdown("---")
st.subheader(f"🔮 4. 향후 3년 미래 예상 자산 결산액 (기준: {sym}{initial_investment:,.0f})")

sim_days = 252 * 3 
num_simulations = 1000

mu = portfolio_daily_return.mean()
sigma = portfolio_daily_return.std()

if sigma == 0:
    st.warning("변동성이 0입니다. (모든 기간을 현금으로 설정하셨습니다)")
else:
    np.random.seed(42)
    simulated_daily_returns = np.random.normal(mu, sigma, (sim_days, num_simulations))
    simulated_cash_flow = (1 + simulated_daily_returns).cumprod(axis=0) * initial_investment 

    percentile_10 = np.percentile(simulated_cash_flow, 10, axis=1)
    percentile_50 = np.percentile(simulated_cash_flow, 50, axis=1)
    percentile_90 = np.percentile(simulated_cash_flow, 90, axis=1)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(sim_days)) + list(range(sim_days))[::-1],
                              y=list(percentile_90) + list(percentile_10)[::-1],
                              fill='toself', fillcolor='rgba(0,176,246,0.2)', line=dict(color='rgba(255,255,255,0)'),
                              name='예상 자산 범위 (상/하위 10%)'))
    fig2.add_trace(go.Scatter(x=list(range(sim_days)), y=percentile_90, mode='lines', line=dict(color='green', dash='dash'), name='운이 아주 좋을 때 (상위 10%)'))
    fig2.add_trace(go.Scatter(x=list(range(sim_days)), y=percentile_50, mode='lines', line=dict(color='blue', width=3), name='가장 현실적인 평균 자산 (50%)'))
    fig2.add_trace(go.Scatter(x=list(range(sim_days)), y=percentile_10, mode='lines', line=dict(color='red', dash='dash'), name='운이 아주 나쁠 때 (하위 10%)'))

    fig2.update_layout(xaxis_title="미래 경과 일수 (총 3년)", yaxis_title=f"예상 자산 가치 ({sym})")
    st.plotly_chart(fig2, use_container_width=True)

    final_10 = percentile_10[-1]
    final_50 = percentile_50[-1]
    final_90 = percentile_90[-1]
    
    st.info(f"📊 **결산 시나리오:** 현재 세팅하신 포트폴리오에 **{sym}{initial_investment:,.0f}** 를 투자하고 3년 뒤 계좌를 열어본다면, \n"
            f"* 🔴 최악의 경우(하위 10%): **{sym}{final_10:,.0f}** \n"
            f"* 🔵 평균적인 경우: **{sym}{final_50:,.0f}** \n"
            f"* 🟢 최상의 경우(상위 10%): **{sym}{final_90:,.0f}** \n"
            f"정도의 금액이 결산되어 있을 확률이 높습니다.")
