import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from scipy.stats import norm # 정규분포용

st.set_page_config(page_title="QRNG vs PRNG 비교", page_icon="⚖️", layout="wide")

st.title("⚖️ 순수 난수 품질 비교: QRNG vs PRNG")
st.markdown("정규분포(Normal Dist) 기반 GBM에서, 난수 생성 방식에 따른 시뮬레이션 차이를 비교합니다.")

# 사이드바
ticker_input = st.sidebar.text_input("종목 코드", value="AAPL")
TICKER = ticker_input.strip().upper()
qrng_file = st.sidebar.file_uploader("QRNG 데이터 (.bin)", type=['bin'])

if qrng_file is None:
    st.info("👈 왼쪽에서 QRNG 파일을 업로드해주세요.")
else:
    # 1. 주가 데이터 로드
    data = yf.Ticker(TICKER).history(period="1y")
    close_prices = data['Close'].dropna()
    returns = np.log(close_prices / close_prices.shift(1)).dropna()
    
    sigma = float(np.std(returns)) * np.sqrt(252)
    mu = (float(np.mean(returns)) * 252) + (0.5 * sigma**2)
    mu_d, sigma_d = mu / 252, sigma / np.sqrt(252)
    S0 = float(close_prices.iloc[-1])
    
    # 2. 난수 생성
    STEPS, NUM_PATHS = 252, 200
    
    # QRNG: 파일 읽어서 정규분포(norm)로 변환
    raw = np.frombuffer(qrng_file.read(), dtype=np.uint8).astype(np.float32) / 255.0
    u_data = np.clip(raw[:STEPS * NUM_PATHS], 1e-7, 1 - 1e-7)
    z_qrng = norm.ppf(u_data) # [핵심] 정규분포로 변환 (팻테일 제거)
    
    # PRNG: 컴퓨터 기본 난수 (표준 정규분포)
    z_prng = np.random.standard_normal(STEPS * NUM_PATHS)
    
    # 3. 시뮬레이션 함수
    def run_gbm(z):
        res = np.zeros((STEPS, NUM_PATHS))
        res[0] = S0
        for t in range(1, STEPS):
            z_t = z[(t-1)*NUM_PATHS : t*NUM_PATHS]
            res[t] = res[t-1] * np.exp((mu_d - 0.5 * sigma_d**2) + sigma_d * z_t)
        return res

    res_q = run_gbm(z_qrng)
    res_p = run_gbm(z_prng)
    
    # 4. 시각화
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Quantum Random (QRNG)")
        fig1, ax1 = plt.subplots()
        ax1.plot(res_q[:, :5], alpha=0.7)
        st.pyplot(fig1)
    with col2:
        st.subheader("Pseudo-Random (PRNG)")
        fig2, ax2 = plt.subplots()
        ax2.plot(res_p[:, :5], alpha=0.7, color='orange')
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("최종 가격 분포 비교")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.hist(res_q[-1], bins=50, alpha=0.5, label='QRNG', color='blue')
    ax3.hist(res_p[-1], bins=50, alpha=0.5, label='PRNG', color='orange')
    ax3.legend()
    st.pyplot(fig3)