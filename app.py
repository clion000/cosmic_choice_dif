import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from scipy.stats import norm

st.set_page_config(page_title="QRNG vs PRNG 비교 분석", page_icon="⚖️", layout="wide")

st.title("⚖️ QRNG vs PRNG: 경로 및 확률 원뿔 비교")
st.markdown("동일한 주가 조건(GBM)에서 난수 생성 방식에 따른 시뮬레이션 경로와 확률 범위를 겹쳐서 비교합니다.")

# 1. 사이드바 설정
ticker = st.sidebar.text_input("종목 코드", value="AAPL").strip().upper()
qrng_option = st.sidebar.selectbox("사용할 QRNG 소스", 
    ["양자 난수 세트 1 (기본)", "양자 난수 세트 2 (추가)", "양자 난수 세트 3 (추가)"])

# 2. 주가 데이터 로드
data = yf.Ticker(ticker).history(period="1y")
if data.empty:
    st.error("데이터를 찾을 수 없습니다.")
    st.stop()
close = data['Close'].dropna()
ret = np.log(close / close.shift(1)).dropna()
mu, sigma = float(np.mean(ret)) * 252, float(np.std(ret)) * np.sqrt(252)
S0 = float(close.iloc[-1])
steps, paths = 252, 200

# 3. 난수 로드 (QRNG 및 PRNG)
with open(f"{qrng_option.split(' ')[3].lower()}_data_{qrng_option.split(' ')[5][0]}.bin", "rb") as f:
    raw = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32) / 255.0
z_q = norm.ppf(np.clip(raw[:steps*paths], 1e-7, 1-1e-7))
z_p = np.random.standard_normal(steps * paths)

# 4. 시뮬레이션
def get_sim(z):
    res = np.zeros((steps, paths))
    res[0] = S0
    for t in range(1, steps):
        z_t = z[(t-1)*paths : t*paths]
        res[t] = res[t-1] * np.exp((mu/252 - 0.5*(sigma/np.sqrt(252))**2) + (sigma/np.sqrt(252)) * z_t)
    return res

res_q = get_sim(z_q)
res_p = get_sim(z_p)

# 5. 시각화 (오버레이)
dates = pd.bdate_range(start=close.index[-1] + pd.Timedelta(days=1), periods=steps)

# 원뿔 비교
st.subheader("📊 확률 원뿔 겹쳐보기")
fig1, ax1 = plt.subplots(figsize=(12, 6))
q_q = np.percentile(res_q, [5, 95], axis=1)
q_p = np.percentile(res_p, [5, 95], axis=1)

ax1.fill_between(dates, q_q[0], q_q[1], color='blue', alpha=0.1, label='QRNG (5%-95%)')
ax1.fill_between(dates, q_p[0], q_p[1], color='orange', alpha=0.1, label='PRNG (5%-95%)')
ax1.plot(dates, np.median(res_q, axis=1), color='blue', linestyle='--', label='QRNG Median')
ax1.plot(dates, np.median(res_p, axis=1), color='orange', linestyle='--', label='PRNG Median')
ax1.legend()
st.pyplot(fig1)

# 경로 비교
st.subheader("📈 경로 샘플 겹쳐보기 (상위 5개)")
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(dates, res_q[:, :5], color='blue', alpha=0.3, linewidth=1)
ax2.plot(dates, res_p[:, :5], color='orange', alpha=0.3, linewidth=1)
ax2.set_title("Blue: QRNG, Orange: PRNG")
st.pyplot(fig2)