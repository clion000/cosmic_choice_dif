import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from scipy.stats import norm

# 1. 설정
st.set_page_config(page_title="QRNG vs PRNG 비교 분석", page_icon="⚖️", layout="wide")
st.title("⚖️ QRNG vs PRNG: 경로 및 확률 원뿔 비교")
st.markdown("정규분포 기반 GBM 모델에서 난수 출처(양자 vs 컴퓨터)에 따른 미래 경로와 예측 범위를 겹쳐서 비교합니다.")

# 2. 사이드바 입력
ticker = st.sidebar.text_input("종목 코드", value="AAPL").strip().upper()
qrng_option = st.sidebar.selectbox("사용할 양자 난수(QRNG) 소스", 
    ["양자 난수 세트 1 (기본)", "양자 난수 세트 2 (추가)", "양자 난수 세트 3 (추가)"])

# 3. 데이터 로드
@st.cache_data
def get_stock_data(ticker):
    data = yf.Ticker(ticker).history(period="1y")
    return data['Close'].dropna()

close = get_stock_data(ticker)
if close.empty:
    st.error("데이터를 찾을 수 없습니다. 종목 코드를 확인하세요.")
    st.stop()

ret = np.log(close / close.shift(1)).dropna()
mu = float(np.mean(ret)) * 252
sigma = float(np.std(ret)) * np.sqrt(252)
S0 = float(close.iloc[-1])

# 4. 파일 매핑 (IndexError 방지)
file_mapping = {
    "양자 난수 세트 1 (기본)": "qrng_data_1.bin",
    "양자 난수 세트 2 (추가)": "qrng_data_2.bin",
    "양자 난수 세트 3 (추가)": "qrng_data_3.bin"
}
target_file = file_mapping[qrng_option]

try:
    with open(target_file, "rb") as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
except FileNotFoundError:
    st.error(f"❌ 파일을 찾을 수 없습니다: {target_file}. 깃허브 저장소에 .bin 파일들이 있는지 확인하세요.")
    st.stop()

# 5. 시뮬레이션 로직
STEPS, PATHS = 252, 200
raw_floats = raw_data.astype(np.float32) / 255.0
# 난수 부족 방지
required_size = STEPS * PATHS
if len(raw_floats) < required_size:
    raw_floats = np.resize(raw_floats, required_size)

# QRNG 정규분포 변환
z_q = norm.ppf(np.clip(raw_floats[:required_size], 1e-7, 1-1e-7))
# PRNG 표준정규분포
z_p = np.random.standard_normal(required_size)

def run_gbm(z):
    res = np.zeros((STEPS, PATHS))
    res[0] = S0
    dt = 1/252
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    for t in range(1, STEPS):
        z_t = z[(t-1)*PATHS : t*PATHS]
        res[t] = res[t-1] * np.exp(drift + diffusion * z_t)
    return res

res_q = run_gbm(z_q)
res_p = run_gbm(z_p)

# 6. 시각화 (오버레이)
dates = pd.bdate_range(start=close.index[-1] + pd.Timedelta(days=1), periods=STEPS)

# 원뿔 비교
st.subheader("📊 확률 원뿔(5% - 95%) 겹쳐보기")
fig1, ax1 = plt.subplots(figsize=(10, 5))
q_q = np.percentile(res_q, [5, 95], axis=1)
q_p = np.percentile(res_p, [5, 95], axis=1)

ax1.fill_between(dates, q_q[0], q_q[1], color='blue', alpha=0.15, label='QRNG Range')
ax1.fill_between(dates, q_p[0], q_p[1], color='orange', alpha=0.15, label='PRNG Range')
ax1.plot(dates, np.median(res_q, axis=1), color='blue', linestyle='--', label='QRNG Median')
ax1.plot(dates, np.median(res_p, axis=1), color='orange', linestyle='--', label='PRNG Median')
ax1.legend()
ax1.set_title("Probability Cone Overlay")
st.pyplot(fig1)

# 경로 비교
st.subheader("📈 경로 샘플(상위 5개) 겹쳐보기")
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(dates, res_q[:, :5], color='blue', alpha=0.3, linewidth=1)
ax2.plot(dates, res_p[:, :5], color='orange', alpha=0.3, linewidth=1)
ax2.set_title("Path Overlay (Blue: QRNG, Orange: PRNG)")
st.pyplot(fig2)