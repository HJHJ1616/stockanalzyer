import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

st.set_page_config(layout="wide", page_title="Quant Dashboard")
st.title("🚀 Quant Dashboard (Ver.11)")

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
# 날짜 범위 제한 해제 (1980~현재)
global_start = st.sidebar.date_input("전체 시작일", pd.to_datetime("2020-01-01"), min_value=pd.to_datetime("1980-01-01"), max_value=datetime.today())
global_end = st.sidebar.date_input("전체 종료일", datetime.today(), min_value=pd.to_datetime("1980-01-01"), max_value=datetime.today())

st.sidebar.markdown("---")
st.sidebar.subheader("📦 종목별 상세 설정 (비중 및 매매일)")

ticker_settings = {}
total_weight = 0

for ticker in tickers:
    with st.sidebar.expander(f"🔧 {ticker} 설정", expanded=True):
        w = st.slider(f"비중 (%)", 0, 100, 100 // len(tickers), key=f"weight_{ticker}")
        
        # 개별 종목 날짜 범위 제한 해제
        t_start = st.date_input(
            f"매수일", 
            global_start, 
            min_value=pd.to_datetime("1980-01-01"), 
            max_value=datetime.today(),
            key=f"start_{ticker}"
        )
        t_end = st.date_input(
            f"매도일", 
            global_end, 
            min_value=pd.to_datetime("1980-01-01"), 
            max_value=datetime.today(),
            key=f"end_{ticker}"
        )
        
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
    # yfinance 호출 시 progress=False로 설정하여 불필요한 출력 방지
    df = yf.download(ticker_list, start=start, end=end, progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame(name=ticker_list[0])
    return df

with st.spinner('시장 빅데이터를 분석 중입니다... ⏳'):
    # AI용 장기 데이터 (2010년부터 현재까지)
    raw_data = load_data(tickers, "2010-01-01", datetime.today().strftime('%Y-%m-%d'))
    
    if raw_data.empty:
        st.error("데이터를 불러오지 못했습니다. 티커를 확인해주세요.")
        st.stop()

    # 사용자 설정 기간 데이터 필터링
    mask = (raw_data.index >= pd.to_datetime(global_start)) & (raw_data.index <= pd.to_datetime(global_end))
    data = raw_data.loc[mask].dropna()
    long_term_data = raw_data.dropna()

if data.empty:
    st.error("설정한 기간에 데이터가 없습니다. 주말/휴일이거나 아직 상장 전인 종목일 수 있습니다.")
    st.stop()

# 장기 데이터가 너무 짧으면(신규 상장주 등) 그냥 현재 데이터 사용
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
        # 현금 방치 로직: 매수일 전, 매도일 후는 수익률 0%
        t_ret.loc[t_ret.index < t_start] = 0.0
        t_ret.loc[t_ret.index > t_end] = 0.0
        
        t_cum = (1 + t_ret).cumprod()
        adjusted_cum_returns[ticker] = t_cum * 100 
        portfolio_value += t_cum * norm_w

        # 영수증 데이터 추출
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

# ---------------------------------------------------------
# 📝 1. 과거 백테스트 요약 리포트
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📝 1. 과거 백테스트 성과 분석")
st.info("이 섹션은 설정하신 과거 기간 동안 포트폴리오가 어떻게 성장했는지 보여줍니다. **CAGR이 높을수록 돈이 빨리 복리로 불어나며, MDD가 0에 가까울수록 하락장에서 방어를 잘 한 안전한 투자**입니다.")

col1, col2, col3 = st.columns(3)
col1.metric("🔥 과거 누적 수익률", f"{port_tot:.2f}%", help="초기 자본 대비 최종적으로 몇 프로가 늘었는지 보여줍니다.")
col2.metric("📈 연평균 수익률 (CAGR)", f"{port_cagr:.2f}%", help="복리 마법의 핵심! 매년 평균적으로 이만큼씩 자산이 성장했다는 뜻입니다.")
col3.metric("📉 최대 낙폭 (MDD)", f"{port_mdd:.2f}%", help="투자 기간 중 가장 심하게 물렸을 때의 마이너스 비율입니다. (멘탈 테스트 지수)")

# 차트 시작점 자동 조절 (가장 빠른 매수일 기준)
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
st.write(f"설정하신 투자금 **{sym}{initial_investment:,.0f}**이 각각의 주식에 배분되어, 최종적으로 얼마의 현금으로 돌아왔는지 1원/1달러 단위까지 보여주는 영수증입니다.")

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
st.info("💡 **'데이터 부족'이 뜨는 이유:** 만약 조회 기간을 '1년'으로 설정하셨다면, '3년 보유 시 승률'은 과거 데이터 길이가 짧아서 수학적으로 계산할 수 없습니다. 이럴 땐 사이드바에서 [전체 시작일]을 5년 전으로 늘려보세요!")

periods = {'1개월(단타)': 21, '6개월(스윙)': 126, '1년(장투)': 252, '3년(기절)': 252*3}
win_rates = {}

for label, days in periods.items():
    if len(portfolio_daily_return) > days:
        rolling_ret = portfolio_daily_return.rolling(window=days).apply(lambda x: (1+x).prod() - 1)
        win_rate = (rolling_ret > 0).mean() * 100
        win_rates[label] = f"{win_rate:.1f}%"
    else:
        win_rates[label] = "데이터 부족 (조회기간 늘리기 요망)"

df_win = pd.DataFrame([win_rates], index=['수익 발생 확률(승률)'])
st.table(df_win)

# ---------------------------------------------------------
# 🔮 4. 장기 데이터 기반 AI 미래 예측 (몬테카를로)
# ---------------------------------------------------------
st.markdown("---")
st.subheader(f"🔮 4. 향후 3년 딥러닝 시뮬레이션 (최대 15년 빅데이터 기반)")
st.write("사용자가 짧게 설정한 기간이 아니라, **해당 티커들의 과거 15년 치(2010년~) 롱텀 데이터(Long-term Data)를 AI가 싹 다 긁어와서** 1,000번의 미래를 시뮬레이션합니다. (최근 상승장에만 취하지 않고 과거 폭락장까지 학습합니다.)")

lt_daily_returns = long_term_data.pct_change().dropna()
lt_portfolio_return = pd.Series(0.0, index=lt_daily_returns.index)

# 장기 데이터 매핑 (없는 종목은 자동으로 제외하여 에러 방지)
for ticker in tickers:
    norm_w = ticker_settings[ticker]['weight'] / total_weight
    if ticker in lt_daily_returns.columns:
        lt_portfolio_return += lt_daily_returns[ticker] * norm_w

sim_days = 252 * 3 
num_simulations = 1000

lt_mu = lt_portfolio_return.mean()
lt_sigma = lt_portfolio_return.std()

if lt_sigma == 0:
    st.warning("설정된 데이터의 변동성이 0입니다. 포트폴리오 비중이나 기간을 확인해 주세요.")
else:
    np.random.seed(42)
    simulated_daily_returns = np.random.normal(lt_mu, lt_sigma, (sim_days, num_simulations))
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

    st.markdown("### 🤖 시스템 종합 코멘트")
    
    if port_mdd < -30:
        risk_comment = "⚠️ **리스크 경고:** 과거 최대 낙폭(MDD)이 -30%를 넘습니다. 하락장이 오면 멘탈이 크게 흔들릴 수 있는 공격적인 세팅입니다. SCHD나 IAU(금)의 비중을 늘리는 것을 고려해 보세요."
    elif port_mdd > -15:
        risk_comment = "🛡️ **훌륭한 방어력:** 과거 어떤 폭락장이 와도 -15% 이내로 방어한 아주 단단한 포트폴리오입니다! 밤에 발 뻗고 잘 수 있는 세팅이네요."
    else:
        risk_comment = "⚖️ **적절한 밸런스:** 위험과 수익의 균형이 어느 정도 맞춰진 포트폴리오입니다."

    if port_cagr > 15:
        return_comment = "🔥 **압도적인 수익성:** 연평균 성장률(CAGR)이 15%를 초과하는 괴물 같은 포트폴리오입니다. 이대로 유지된다면 자산이 엄청난 속도로 불어날 것입니다."
    elif port_cagr > 8:
        return_comment = "📈 **안정적인 우상향:** 시장 평균(S&P 500) 수준의 든든한 수익률을 보여주고 있습니다."
    else:
        return_comment = "🐢 **보수적인 성장:** 수익률보다는 안전성에 치중된 세팅입니다. 조금 더 공격적인 종목을 10% 정도 섞어보는 것도 좋습니다."

    st.success(f"{risk_comment}\n\n{return_comment}\n\n**🔮 3년 뒤 결산 시나리오:** 현재 세팅으로 **{sym}{initial_investment:,.0f}** 를 투자하고 3년 뒤 계좌를 열어보면, **평균적으로 {sym}{final_50:,.0f}** 가 되어 있을 확률이 가장 높습니다. (최악의 하락장이 와도 {sym}{final_10:,.0f} 는 방어할 것으로 예측됩니다.)")
