import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🚀 나만의 커스텀 포트폴리오 백테스터")
st.markdown("원하는 미국 주식/ETF/코인 티커를 입력하고, 황금 비중을 찾아보세요!")

# 1. 티커 입력받기 (동적 생성의 핵심)
tickers_input = st.text_input("🔍 테스트할 티커를 쉼표(,)로 구분해서 입력하세요.", "SSO, SCHD, IAU, BTC-USD")
# 입력받은 텍스트를 리스트로 변환 (공백 제거 및 대문자 변환)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if not tickers:
    st.warning("최소 1개 이상의 티커를 입력해주세요!")
    st.stop()

# 2. 사이드바: 입력된 티커 개수만큼 동적 슬라이더 생성
st.sidebar.header("⚖️ 비중 설정 (%)")
weights_dict = {}

for ticker in tickers:
    # 기본값은 100을 티커 개수로 나눈 값(균등 배분)으로 세팅
    default_w = 100 // len(tickers)
    weights_dict[ticker] = st.sidebar.slider(f"{ticker} 비중", 0, 100, default_w)

# 비중 합계 검증 및 정규화 (100%가 넘거나 모자라도 알아서 비율대로 맞춰줌)
total_weight = sum(weights_dict.values())
if total_weight == 0:
    st.sidebar.error("비중의 합이 0이 될 수 없습니다.")
    st.stop()
elif total_weight != 100:
    st.sidebar.warning(f"현재 비중 합계: {total_weight}%. (자동으로 100% 기준 비율로 환산하여 계산합니다.)")

# 3. 데이터 가져오기 (yfinance)
@st.cache_data
def load_data(ticker_list):
    # 다운로드 후 종가(Close) 데이터만 추출, 결측치가 있는 날짜는 제외
    df = yf.download(ticker_list, start="2020-01-01", progress=False)['Close']
    
    # 티커가 1개일 때와 여러 개일 때 반환되는 형태가 달라서 맞춰주는 작업
    if isinstance(df, pd.Series):
        df = df.to_frame(name=ticker_list[0])
        
    df = df.dropna() 
    return df

with st.spinner('과거 주가 데이터를 불러오는 중입니다... ⏳'):
    data = load_data(tickers)

if data.empty:
    st.error("데이터를 불러오지 못했습니다. 티커명이 정확한지 확인해 주세요. (예: 비트코인은 BTC-USD)")
    st.stop()

# 4. 수익률 계산 로직
daily_returns = data.pct_change().dropna()
portfolio_daily_return = pd.Series(0.0, index=daily_returns.index)

# 각 티커별로 (일일 수익률 * 환산된 비중)을 전체 포트폴리오에 더하기
for ticker in tickers:
    normalized_weight = weights_dict[ticker] / total_weight
    portfolio_daily_return += daily_returns[ticker] * normalized_weight

# 누적 수익률 계산 (시작점 = 100)
cumulative_returns = (1 + portfolio_daily_return).cumprod() * 100
cumulative_returns.name = 'My Portfolio'

# 개별 종목들의 누적 수익률도 비교를 위해 같이 계산
all_cum_returns = (1 + daily_returns).cumprod() * 100
all_cum_returns['My Portfolio'] = cumulative_returns

# 5. 차트 그리기
st.subheader("📈 포트폴리오 vs 개별 종목 수익률 비교 (초기 자본 = 100)")

# My Portfolio 선은 굵게, 나머지는 얇게 설정
fig = px.line(all_cum_returns, x=all_cum_returns.index, y=all_cum_returns.columns)
fig.update_traces(line=dict(width=1), opacity=0.5) # 전체 얇게
fig.for_each_trace(lambda trace: trace.update(line=dict(width=4, color='red'), opacity=1.0) if trace.name == 'My Portfolio' else ())

st.plotly_chart(fig, use_container_width=True)

# 6. 최종 수익률 요약
final_return = cumulative_returns.iloc[-1] - 100
st.success(f"🔥 **설정한 비중의 포트폴리오 최종 누적 수익률: {final_return:.2f}%**")