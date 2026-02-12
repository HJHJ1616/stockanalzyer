import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import google.generativeai as genai

st.set_page_config(layout="wide", page_title="Quant Dashboard")
st.title("🚀 Quant Dashboard (ver.)")

st.warning("⚠️ **[백테스트 로직 안내] 현금 방치형 (Cash Drag) 적용:** \n"
           "설정한 '매도일' 이후(또는 '매수일' 이전)의 자산은 추가 손익 없이 **수익률 0%의 '현금' 상태로 방치**되는 것으로 계산됩니다.")

# ---------------------------------------------------------
# 🔑 API 키 자동 로드 (비밀 금고에서 꺼내오기)
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
# 1. 사이드바 설정
# ---------------------------------------------------------
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
    st.error("설정한 기간에 데이터가 없습니다.")
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
    
    risk_free_rate = 0.02
    volatility = daily_ret.std() * np.sqrt(252)
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
# 🔮 4. Gemini AI 애널리스트 분석 리포트
# ---------------------------------------------------------
st.markdown("---")
st.subheader(f"🔮 4. Gemini AI 투자 애널리스트의 심층 분석")

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
    final_10 = percentile_10[-1]

    # Gemini에게 보낼 프롬프트 작성
    prompt = f"""
    당신은 월가에서 20년 경력을 가진 냉철한 퀀트 투자 애널리스트입니다.
    사용자의 포트폴리오 데이터를 분석하고, 한국어로 솔직하고 전문적인 피드백을 주세요.
    
    [포트폴리오 정보]
    - 구성 종목: {tickers}
    - 종목별 설정(비중 등): {ticker_settings}
    
    [핵심 성과 지표]
    - 연평균 수익률(CAGR): {port_cagr:.2f}% (높을수록 좋음)
    - 최대 낙폭(MDD): {port_mdd:.2f}% (0에 가까울수록 안전)
    - 샤프 지수(Sharpe Ratio): {port_sharpe:.2f} (1.0 이상이면 우수, 0.5 미만이면 위험)
    
    [미래 3년 예측 (몬테카를로 시뮬레이션)]
    - 예상 평균 자산: {sym}{final_50:,.0f}
    - 최악의 경우 자산: {sym}{final_10:,.0f}
    
    [요청사항]
    1. **종합 평가:** 이 포트폴리오의 상태를 한마디로 정의하세요. (예: "고위험 고수익의 전형", "안전하지만 지루함" 등)
    2. **효율성 분석:** 샤프 지수를 기반으로, 사용자가 감수하는 위험 대비 수익이 적절한지 비판하세요.
    3. **종목 구성 피드백:** 각 종목(주식, 코인, 금 등)이 이 포트폴리오에서 어떤 역할을 하고 있는지, 혹은 무엇이 문제인지 지적하세요.
    4. **구체적인 조언:** MDD를 줄이거나 수익을 높이기 위해 어떤 종목의 비중을 조절하면 좋을지 제안하세요.
    
    말투는 정중하지만 팩트에 기반하여 냉철하게 분석해주세요. 마크다운 형식을 사용하여 가독성 있게 작성하세요.
    """

    # AI 분석 요청 버튼 (🔥 여기가 핵심: 모델 자동 찾기 기능 추가)
    if st.button("🤖 Gemini에게 심층 분석 요청하기 (Click)"):
        with st.spinner("AI가 사용 가능한 모델을 찾고 분석 중입니다... (약 5~10초 소요)"):
            try:
                # 1. 사용 가능한 모델 목록 조회
                available_models = []
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        available_models.append(m.name)
                
                # 2. 우선순위에 따라 모델 선택 (Flash -> Pro -> 기본)
                selected_model_name = 'models/gemini-1.5-flash' # 기본값
                
                for m in available_models:
                    if 'flash' in m: # 1순위: 빠르고 저렴한 Flash
                        selected_model_name = m
                        break
                    elif 'pro' in m: # 2순위: 성능 좋은 Pro
                        selected_model_name = m
                
                # 3. 모델 연결 및 분석 시작
                model = genai.GenerativeModel(selected_model_name)
                response = model.generate_content(prompt)
                
                st.success(f"✅ 분석 완료! (사용 모델: {selected_model_name})")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"AI 분석 중 오류가 발생했습니다: {e}")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(sim_days)), y=percentile_90, mode='lines', line=dict(color='green', dash='dash'), name='상위 10% (대박)'))
    fig2.add_trace(go.Scatter(x=list(range(sim_days)), y=percentile_50, mode='lines', line=dict(color='blue', width=3), name='평균 (현실)'))
    fig2.add_trace(go.Scatter(x=list(range(sim_days)), y=percentile_10, mode='lines', line=dict(color='red', dash='dash'), name='하위 10% (쪽박)'))
    fig2.update_layout(xaxis_title="미래 3년", yaxis_title="자산 가치")
    st.plotly_chart(fig2, use_container_width=True)
