import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide", page_title="Quant Dashboard")
st.title("🚀 Quant Dashboard (Ver. 12)")

st.warning("⚠️ **[백테스트 로직 안내] 현금 방치형 (Cash Drag) 적용:** \n"
           "설정한 '매도일' 이후(또는 '매수일' 이전)의 자산은 추가 손익 없이 **수익률 0%의 '현금' 상태로 방치**되는 것으로 계산됩니다.")

# 1. 사이드바 설정
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
global_start = st.sidebar.date_input("전체 시작일", pd.to_datetime("2020-01-01"), min_value=pd.to_datetime("1980-01-01"), max_value=datetime.today())
global_end = st.sidebar.date_input("전체 종료일", datetime.today(), min_value=pd.to_datetime("1980-01-01"), max_value=datetime.today())

st.sidebar.markdown("---")
st.sidebar.subheader("📦 종목별 상세 설정 (비중 및 매매일)")

ticker_settings = {}
total_weight = 0

for ticker in tickers:
    with st.sidebar.expander(f"🔧 {ticker} 설정", expanded=True):
        w = st.slider(f"비중 (%)", 0, 100, 100 // len(tickers), key=f"weight_{ticker}")
        t_start = st.date_input(f"매수일", global_start, min_value=pd.to_datetime("1980-01-01"), max_value=datetime.today(), key=f"start_{ticker}")
        t_end = st.date_input(f"매도일", global_end, min_value=pd.to_datetime("1980-01-01"), max_value=datetime.today(), key=f"end_{ticker}")
        
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
    return df

with st.spinner('시장 데이터를 정밀 분석 중입니다... ⏳'):
    raw_data = load_data(tickers, "2010-01-01", datetime.today().strftime('%Y-%m-%d'))
    
    if raw_data.empty:
        st.error("데이터를 불러오지 못했습니다. 티커를 확인해주세요.")
        st.stop()

    mask = (raw_data.index >= pd.to_datetime(global_start)) & (raw_data.index <= pd.to_datetime(global_end))
    data = raw_data.loc[mask].dropna()
    long_term_data = raw_data.dropna()

if data.empty:
    st.error("설정한 기간에 데이터가 없습니다. 주말/휴일이거나 아직 상장 전인 종목일 수 있습니다.")
    st.stop()

if long_term_data.empty or len(long_term_data) < 21:
    long_term_data = data 

# 3. 실전 Buy & Hold 계산
daily_returns = data.pct_change().dropna()
portfolio_value = pd.Series(0.0, index=daily_returns.index)
adjusted_cum_returns = pd.DataFrame(index=daily_returns.index)
receipt_data = []

for ticker in tickers:
    norm_w = ticker_settings[ticker]['weight'] / total_weight
    t_start = ticker_settings[ticker]['start']
    t_end = ticker_settings[ticker]['end']
    
    if ticker in daily_returns.columns:
        t_ret = daily_returns[ticker].copy()
        t_ret.loc[t_ret.index < t_start] = 0.0
        t_ret.loc[t_ret.index > t_end] = 0.0
        
        t_cum = (1 + t_ret).cumprod()
        adjusted_cum_returns[ticker] = t_cum * 100 
        portfolio_value += t_cum * norm_w

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

# 통계 계산 함수 (샤프지수 추가)
def calculate_stats(returns_series, is_price_series=False):
    if is_price_series:
        daily_ret = returns_series.pct_change().dropna()
        cum_ret = returns_series / 100
    else:
        daily_ret = returns_series
        cum_ret = (1 + returns_series).cumprod()
        
    total_return = (cum_ret.iloc[-1] - 1) * 100
    trading_days = len(cum_ret)
    years_passed = max(trading_days / 252, 0.01)
    cagr = ((cum_ret.iloc[-1] ** (1 / years_passed)) - 1) * 100
    
    roll_max = cum_ret.cummax()
    drawdown = (cum_ret - roll_max) / roll_max
    mdd = drawdown.min() * 100
    
    # 샤프지수 (무위험 수익률 2% 가정)
    risk_free_rate = 0.02
    volatility = daily_ret.std() * np.sqrt(252) # 연 변동성
    if volatility == 0:
        sharpe = 0
    else:
        sharpe = (cagr/100 - risk_free_rate) / volatility
        
    return total_return, cagr, mdd, sharpe

port_tot, port_cagr, port_mdd, port_sharpe = calculate_stats(cum_returns, is_price_series=True)

# ---------------------------------------------------------
# 📝 1. 과거 백테스트 요약 리포트
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📝 1. 과거 백테스트 성과 분석")
st.info("단순 수익률뿐만 아니라, **'샤프 지수(Sharpe)'**를 꼭 확인하세요. 1.0 이상이어야 위험 대비 돈을 잘 번 것이며, 0.5 이하라면 위험한 도박을 하고 있다는 뜻입니다.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("🔥 누적 수익률", f"{port_tot:.2f}%")
col2.metric("📈 연평균(CAGR)", f"{port_cagr:.2f}%")
col3.metric("📉 최대 낙폭(MDD)", f"{port_mdd:.2f}%")
col4.metric("🛡️ 샤프 지수", f"{port_sharpe:.2f}", help="수익률 ÷ 위험도. 높을수록 고수!")

earliest_buy_date = min([settings['start'] for settings in ticker_settings.values()])
chart_data = adjusted_cum_returns[adjusted_cum_returns.index >= earliest_buy_date]

fig1 = px.line(chart_data, x=chart_data.index, y=chart_data.columns)
fig1.update_traces(line=dict(width=1), opacity=0.4)
fig1.for_each_trace(lambda trace: trace.update(line=dict(width=4, color='#FF4B4B'), opacity=1.0) if trace.name == 'My Portfolio' else ())
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------------
# 🧾 2. 영수증
# ---------------------------------------------------------
st.markdown("---")
st.subheader(f"🧾 2. 가상 매매 결산 영수증 (기준 통화: {sym})")

if receipt_data:
    df_receipt = pd.DataFrame(receipt_data).set_index("종목")
    total_principal = df_receipt[f"투자 원금({sym})"].sum()
    total_final = df_receipt[f"최종 평가액({sym})"].sum()
    total_profit = df_receipt[f"손익금({sym})"].sum()
    st.dataframe(df_receipt.style.format("{:,.2f}", subset=["매수 단가 (현지)", "매도 단가 (현지)", f"투자 원금({sym})", f"최종 평가액({sym})", f"손익금({sym})", "수익률(%)"]).background_gradient(cmap='RdYlGn', subset=[f'손익금({sym})']), use_container_width=True)

# ---------------------------------------------------------
# 🎯 3. 승률 분석
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🎯 3. 이 조합으로 돈을 벌 확률 (보유 기간별 승률)")
periods = {'1개월': 21, '6개월': 126, '1년': 252, '3년': 252*3}
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
# 🔮 4. AI 팩트 폭격 코멘트 (기준 대폭 상향)
# ---------------------------------------------------------
st.markdown("---")
st.subheader(f"🔮 4. AI 투자 분석관의 '팩트 폭격' 리포트")

# AI 장기 시뮬레이션
lt_daily_returns = long_term_data.pct_change().dropna()
lt_portfolio_return = pd.Series(0.0, index=lt_daily_returns.index)

for ticker in tickers:
    norm_w = ticker_settings[ticker]['weight'] / total_weight
    if ticker in lt_daily_returns.columns:
        lt_portfolio_return += lt_daily_returns[ticker] * norm_w

sim_days = 252 * 3 
num_simulations = 1000
lt_mu = lt_portfolio_return.mean()
lt_sigma = lt_portfolio_return.std()

if lt_sigma == 0:
    st.warning("데이터 부족으로 시뮬레이션 불가")
else:
    np.random.seed(42)
    simulated_daily_returns = np.random.normal(lt_mu, lt_sigma, (sim_days, num_simulations))
    simulated_cash_flow = (1 + simulated_daily_returns).cumprod(axis=0) * initial_investment 
    percentile_10 = np.percentile(simulated_cash_flow, 10, axis=1)
    percentile_50 = np.percentile(simulated_cash_flow, 50, axis=1)
    percentile_90 = np.percentile(simulated_cash_flow, 90, axis=1)
    final_50 = percentile_50[-1]

    # 🔥 독설가 AI 알고리즘
    st.markdown("### 🤖 시스템 종합 평가")
    
    # 1. 효율성 평가 (샤프지수)
    if port_sharpe > 1.0:
        eff_comment = "💎 **효율성 최상:** 위험 대비 수익이 아주 훌륭합니다. 고수의 포트폴리오네요."
    elif port_sharpe > 0.7:
        eff_comment = "✅ **효율성 양호:** 적당한 위험으로 적당한 수익을 내고 있습니다."
    elif port_sharpe > 0.4:
        eff_comment = "⚠️ **효율성 부족:** 수익을 내고는 있지만, 그에 비해 감수하는 위험이 너무 큽니다. 가성비가 떨어지는 투자입니다."
    else:
        eff_comment = "🗑️ **효율성 최악:** 솔직히 말씀드리면, 그냥 예금에 넣거나 S&P 500 ETF(SPY) 하나만 사는 게 정신건강과 계좌에 더 이롭습니다."

    # 2. 리스크 평가 (MDD)
    if port_mdd < -40:
        risk_comment = "🚨 **위험도 초과:** MDD가 -40%를 넘습니다. 이건 투자가 아니라 야수의 심장을 가진 도박입니다. 하락장에서 계좌가 반토막 나도 버틸 수 있으신가요?"
    elif port_mdd < -20:
        risk_comment = "🔥 **위험도 높음:** 다소 공격적입니다. 시장이 흔들리면 꽤 아플 수 있습니다."
    else:
        risk_comment = "🛡️ **위험 관리 합격:** 비교적 안정적으로 자산을 방어하고 있습니다."

    # 3. 수익성 평가 (CAGR)
    if port_cagr > 25:
        ret_comment = "🚀 **수익성 폭발:** 연 25% 이상의 초고수익입니다. (단, 이게 운인지 실력인지 샤프 지수를 꼭 다시 확인하세요.)"
    elif port_cagr > 10:
        ret_comment = "💰 **수익성 우수:** 시장 평균을 상회하는 좋은 성과입니다."
    else:
        ret_comment = "🐢 **수익성 저조:** 시장 평균(약 10%)보다 못 벌고 있습니다. 고생해서 종목을 고른 보람이 없네요."

    st.info(f"{eff_comment}\n\n{risk_comment}\n\n{ret_comment}")
    
    st.write(f"**📉 3년 뒤 미래 예측:** 현재의 변동성을 고려할 때, 3년 뒤 자산은 평균적으로 **{sym}{final_50:,.0f}** 가 될 것으로 보입니다.")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(sim_days)), y=percentile_90, mode='lines', line=dict(color='green', dash='dash'), name='상위 10% (대박)'))
    fig2.add_trace(go.Scatter(x=list(range(sim_days)), y=percentile_50, mode='lines', line=dict(color='blue', width=3), name='평균 (현실)'))
    fig2.add_trace(go.Scatter(x=list(range(sim_days)), y=percentile_10, mode='lines', line=dict(color='red', dash='dash'), name='하위 10% (쪽박)'))
    fig2.update_layout(xaxis_title="미래 3년", yaxis_title="자산 가치")
    st.plotly_chart(fig2, use_container_width=True)
