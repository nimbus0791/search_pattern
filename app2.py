import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mplfinance as mpf

from dtaidistance import dtw as _dtw
from upstox_client.rest import ApiException
import upstox_client

# =========================
# Constants & Defaults
# =========================
INSTRUMENT_KEY = 'NSE_INDEX|Nifty Bank'
DB_PATH = "banknifty-15min.csv"
DEFAULT_PREV_N_CANDLES = 25
DEFAULT_CURR_N_CANDLES = 3

# =========================
# Data Fetching & Resampling
# =========================
def fetch_historical_data(from_date: str, to_date: str):
    api = upstox_client.HistoryApi()
    try:
        resp = api.get_historical_candle_data1(
            INSTRUMENT_KEY, '1minute', to_date, from_date, '2.0'
        )
        return resp.data.candles
    except ApiException as e:
        st.error(f"API error: {e}")
        return []

def fetch_live_data():
    api_instance = upstox_client.HistoryApi()
    try:
        api_response = api_instance.get_intra_day_candle_data(INSTRUMENT_KEY, '1minute', '2.0')
        return api_response.data.candles
    except ApiException as e:
        st.error(f"Live API error: {e}")
        return []

def change_time_frame(df: pd.DataFrame, time_frame: str) -> pd.DataFrame:
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    df2 = df.resample(time_frame).agg(agg).dropna().reset_index()
    df2['time'] = df2['time'].dt.tz_localize(None)
    return df2

def get_prev_and_curr_day_data(test_date_str: str):
    test_date = datetime.strptime(test_date_str, "%Y-%m-%d").date()
    current_date = datetime.now().date()

    # Current-day candles
    if current_date == test_date:
        curr_candles = fetch_live_data()
    elif current_date > test_date:
        curr_candles = fetch_historical_data(test_date_str, test_date_str)
    else:
        st.error("Future date selected. No data.")
        return None, None

    if not curr_candles:
        st.error("No current-day data available.")
        return None, None

    # Previous trading day candles (skip holidays)
    offset = 1
    while True:
        prev_str = (test_date - timedelta(days=offset)).strftime("%Y-%m-%d")
        prev_candles = fetch_historical_data(prev_str, prev_str)
        if prev_candles:
            tmp = pd.DataFrame(prev_candles, columns=["time", "open", "high", "low", "close", "volume", "oi"])
            tmp['time'] = pd.to_datetime(tmp['time']).dt.tz_localize(None)
            if tmp['time'].dt.date.nunique() >= 1:
                break
        offset += 1

    prev_candles = [c[:5] for c in prev_candles]
    curr_candles = [c[:5] for c in curr_candles]

    prev_df = pd.DataFrame(prev_candles, columns=["time", "open", "high", "low", "close"])
    curr_df = pd.DataFrame(curr_candles, columns=["time", "open", "high", "low", "close"])

    prev_df['time'] = pd.to_datetime(prev_df['time']).dt.tz_localize(None)
    curr_df['time'] = pd.to_datetime(curr_df['time']).dt.tz_localize(None)

    prev_df = change_time_frame(prev_df, "15min")
    curr_df = change_time_frame(curr_df, "15min")
    return prev_df, curr_df

# =========================
# Feature Engineering
# =========================
def extract_candle_features(df: pd.DataFrame, min_quantum: float = 0.001) -> str:
    """Encode swings as 'u'/'d' run-lengths."""
    swings = []
    n = len(df)
    if n == 0:
        return ""
    direction = 'up' if df.loc[0, 'close'] > df.loc[0, 'open'] else 'down'
    start_idx = end_idx = 0

    for i in range(1, n):
        curr_close, prev_close = df.loc[i, 'close'], df.loc[i-1, 'close']
        if direction == 'up':
            if curr_close >= prev_close:
                end_idx = i
            else:
                highest_high = df.loc[start_idx:end_idx, 'high'].max()
                lowest_low  = df.loc[start_idx:end_idx, 'low'].min()
                pct = (highest_high - lowest_low) / max(lowest_low, 1e-9)
                swings.append(pct)
                direction, start_idx, end_idx = 'down', end_idx, i
        else:
            if curr_close <= prev_close:
                end_idx = i
            else:
                highest_high = df.loc[start_idx:end_idx, 'high'].max()
                lowest_low  = df.loc[start_idx:end_idx, 'low'].min()
                pct = (highest_high - lowest_low) / max(highest_high, 1e-9)
                swings.append(-pct)
                direction, start_idx, end_idx = 'up', end_idx, i

    if start_idx != end_idx:
        highest_high = df.loc[start_idx:end_idx, 'high'].max()
        lowest_low  = df.loc[start_idx:end_idx, 'low'].min()
        if direction == 'up':
            pct = (highest_high - lowest_low) / max(lowest_low, 1e-9)
            swings.append(pct)
        else:
            pct = (highest_high - lowest_low) / max(highest_high, 1e-9)
            swings.append(-pct)

    pattern = []
    for swing in swings:
        symbol = 'u' if swing > 0 else 'd'
        count = int(abs(swing) / min_quantum)
        if count > 0:
            pattern.append(symbol * count)
    return ''.join(pattern)

def _build_candle_features(df: pd.DataFrame) -> np.ndarray:
    """Multivariate features for the new DTW."""
    X = df[['open','high','low','close']].astype(float).copy()
    body = (X['close'] - X['open']).to_numpy()
    rng  = (X['high'] - X['low']).to_numpy()
    rng[rng == 0] = 1e-9

    direction = np.sign(body)
    open_arr = X['open'].to_numpy()
    open_arr[open_arr == 0] = 1e-9

    body_pct   = body / open_arr
    upper_wick = (X['high'].to_numpy() - np.maximum(X['open'].to_numpy(), X['close'].to_numpy())) / rng
    lower_wick = (np.minimum(X['open'].to_numpy(), X['close'].to_numpy()) - X['low'].to_numpy()) / rng

    close_arr = X['close'].to_numpy()
    ret = np.zeros_like(close_arr)
    ret[1:] = (close_arr[1:] - close_arr[:-1]) / np.where(close_arr[:-1] == 0, 1e-9, close_arr[:-1])

    feats = np.column_stack([ret, body_pct, upper_wick, lower_wick, direction])
    mu = np.nanmean(feats, axis=0)
    sd = np.nanstd(feats, axis=0) + 1e-8
    return (feats - mu) / sd

def _time_weights(n: int, kind: str = "exp", strength: float = 3.0) -> np.ndarray:
    if n <= 1:
        return np.ones(n)
    x = np.linspace(0, 1, n)
    return (0.5 + 0.5 * np.exp(strength*(x-1))) if kind != "lin" else (0.5 + 0.5 * x)

def _apply_recency_weights(feats: np.ndarray) -> np.ndarray:
    w = _time_weights(feats.shape[0], kind="exp", strength=3.0)
    return feats * w[:, None]

def _last_k_direction_agreement(test_df: pd.DataFrame, hist_df: pd.DataFrame, k: int = 3) -> float:
    k = max(1, min(k, len(test_df), len(hist_df)))
    t_dir = np.sign(test_df['close'].to_numpy()[-k:] - test_df['open'].to_numpy()[-k:])
    h_dir = np.sign(hist_df['close'].to_numpy()[-k:] - hist_df['open'].to_numpy()[-k:])
    return float((t_dir == h_dir).sum() / float(k))

# =========================
# Matchers (3 methods)
# =========================
def lcs_similarity(seq1: str, seq2: str):
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0.0, ""
    dp = np.zeros((m+1, n+1), dtype=int)
    for i in range(m):
        s1 = seq1[i]
        row = dp[i]
        nxt = dp[i+1]
        for j in range(n):
            nxt[j+1] = row[j] + 1 if s1 == seq2[j] else max(nxt[j], row[j+1])
    similarity = dp[m][n] / max(m, n)

    # backtrack to get last match index in seq2
    i, j = m, n
    lcs_indices_seq2 = []
    while i > 0 and j > 0:
        if seq1[i-1] == seq2[j-1]:
            lcs_indices_seq2.append(j-1)
            i -= 1; j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    last_match_index = max(lcs_indices_seq2) if lcs_indices_seq2 else -1
    return float(similarity), (seq1[last_match_index:] if last_match_index >= 0 else "")

def dtw_matching_old(test_df: pd.DataFrame, hist_df: pd.DataFrame):
    # z-score close-only
    t = test_df['close'].astype(float).to_numpy()
    h = hist_df['close'].astype(float).to_numpy()
    t = (t - t.mean()) / (t.std() + 1e-9)
    h = (h - h.mean()) / (h.std() + 1e-9)
    dist = _dtw.distance(t, h)
    sim = 1.0 / (1.0 + dist)
    return float(sim), ""

def dtw_matching_new(test_df: pd.DataFrame, hist_df: pd.DataFrame, recent_k: int = 3, lambda_recent: float = 0.35):
    T = _build_candle_features(test_df)
    H = _build_candle_features(hist_df)
    T = _apply_recency_weights(T)
    H = _apply_recency_weights(H)
    T_flat, H_flat = T.reshape(-1), H.reshape(-1)
    window = max(2, int(0.2 * len(test_df) * T.shape[1]))
    dist = _dtw.distance(T_flat, H_flat, window=window)
    sim_dtw = 1.0 / (1.0 + np.exp(dist))     # logistic on -dist
    agree = _last_k_direction_agreement(test_df, hist_df, k=recent_k)
    sim = (1.0 - lambda_recent) * sim_dtw + lambda_recent * agree
    return float(sim), ""

# =========================
# Pattern Match Runner
# =========================
def match_pattern(history_df: pd.DataFrame,
                  prev_test: pd.DataFrame,
                  curr_test: pd.DataFrame,
                  test_date_str: str,
                  top_k: int,
                  prev_n_candles: int,
                  curr_n_candles: int,
                  method: str = "DTW (new)",
                  recent_k: int = 3,
                  min_agree: float = 0.67,
                  lambda_recent: float = 0.35):

    df2 = history_df.copy()
    df2['time'] = pd.to_datetime(df2['time'], utc=True).dt.tz_convert(None)
    df2 = df2.sort_values('time')
    df2['date'] = df2['time'].dt.date

    daily = {d: g.sort_values('time') for d, g in df2.groupby('date')}
    dates = sorted(daily)
    test_date = datetime.strptime(test_date_str, "%Y-%m-%d").date()

    test_df = pd.concat([prev_test.iloc[-prev_n_candles:], curr_test.iloc[:curr_n_candles]], ignore_index=True)

    matches = []
    for i in range(1, len(dates)):
        d = dates[i]
        if d >= test_date:
            break
        prev, curr = daily[dates[i - 1]], daily[d]
        if len(prev) < prev_n_candles or len(curr) < curr_n_candles:
            continue

        hist_df_c = pd.concat([prev, curr], ignore_index=True)
        hist_df = pd.concat([prev.iloc[-prev_n_candles:], curr.iloc[:curr_n_candles]], ignore_index=True)

        if method == "LCS":
            test_feats = extract_candle_features(test_df)
            hist_feats = extract_candle_features(hist_df)
            sim, fut_sequence = lcs_similarity(hist_feats, test_feats)

        elif method == "DTW (Vanilla)":
            sim, fut_sequence = dtw_matching_old(test_df, hist_df)

        else:  # "DTW (Advance)"
            # quick reject for last-k disagreement to speed up
            if _last_k_direction_agreement(test_df, hist_df, k=min(recent_k, len(test_df))) < min_agree:
                continue
            sim, fut_sequence = dtw_matching_new(test_df, hist_df, recent_k=recent_k, lambda_recent=lambda_recent)

        matches.append((d, sim, fut_sequence, hist_df_c))

    return sorted(matches, key=lambda x: -x[1])[:top_k]

# =========================
# Utilities (noise filter, plotting, sentiment)
# =========================
def filter_noise(seq: str, min_len: int = 2) -> str:
    if not seq:
        return seq
    rle, prev, cnt = [], seq[0], 1
    for ch in seq[1:]:
        if ch == prev:
            cnt += 1
        else:
            rle.append((prev, cnt)); prev, cnt = ch, 1
    rle.append((prev, cnt))

    if len(rle) > 1 and rle[0][1] < min_len:
        rle[1] = (rle[1][0], rle[0][1] + rle[1][1]); rle.pop(0)
    if len(rle) > 1 and rle[-1][1] < min_len:
        rle[-2] = (rle[-2][0], rle[-2][1] + rle[-1][1]); rle.pop()

    i = 1
    while i < len(rle) - 1:
        char, cnt = rle[i]
        if cnt < min_len:
            pch, pc = rle[i-1]
            nch, nc = rle[i+1]
            if pch == nch:
                rle[i-1] = (pch, pc + cnt + nc)
                rle.pop(i); rle.pop(i); i -= 1
            else:
                if pc >= nc:
                    rle[i-1] = (pch, pc + cnt); rle.pop(i)
                else:
                    rle[i+1] = (nch, nc + cnt); rle.pop(i)
        else:
            i += 1
    return ''.join([c * k for c, k in rle])

def plot_candle_chart(df: pd.DataFrame, title: str):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert(None)
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close']]
    fig, ax = plt.subplots(figsize=(6, 4))
    mpf.plot(df, type='candle', ax=ax, datetime_format='%H:%M', show_nontrading=False, xrotation=45, style='yahoo')
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    return fig

def analyze_market_sentiment(top_matches, curr_n_candles: int):
    up_count = down_count = 0
    n_seq = 5
    for (match_date, sim, fut_sequence, hist_df) in top_matches:
        remaining_df = hist_df.iloc[24 + curr_n_candles:].reset_index(drop=True)
        remaining_feat = extract_candle_features(remaining_df)
        cleaned = filter_noise(fut_sequence + remaining_feat, min_len=2)
        u_cnt = sum(1 for ch in cleaned[:n_seq] if ch == 'u')
        d_cnt = sum(1 for ch in cleaned[:n_seq] if ch == 'd')
        if u_cnt > d_cnt: up_count += 1
        else: down_count += 1

    total = max(1, up_count + down_count)
    p_up = up_count / total
    return ("Up", 100 * p_up) if p_up >= 0.5 else ("Down", 100 * (1 - p_up))

# =========================
# Caching
# =========================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    return df

@st.cache_data
def get_cached_top_matches(history_df, prev_df, curr_df, test_date_str,
                           top_k, prev_n_candles, curr_n_candles,
                           method, recent_k, min_agree, lambda_recent):
    return match_pattern(history_df, prev_df, curr_df, test_date_str, top_k,
                         prev_n_candles, curr_n_candles, method, recent_k, min_agree, lambda_recent)

@st.cache_data
def get_cached_sentiment(top_matches, curr_n_candles):
    return analyze_market_sentiment(top_matches, curr_n_candles)

# =========================
# UI
# =========================
st.set_page_config(layout="wide")
st.title("Top-K Pattern Matching Grid")

history_df = load_data(DB_PATH)

test_date = st.date_input("Select test date", datetime.now().date())
top_k = st.slider("Number of top matches to show", 4, 32, 20, step=1)

prev_n_candles = st.number_input("Previous-day candles to match", 1, 25, DEFAULT_PREV_N_CANDLES, 1)
curr_candle_input = st.number_input("Current-day candles to match", 1, 25, DEFAULT_CURR_N_CANDLES, 1)

# Matching method + recency controls
matching_method = st.selectbox(
    "Matching Method:",
    ["DTW (Advance)", "DTW (Vanilla)", "LCS"],  # default to new
    index=0
)

# Always show recent_k (used by DTW new; ignored by others)
recent_k = st.number_input("How many last candles must align (recent_k)", 1, 8, 3, 1)

with st.expander("Advanced (DTW new only)"):
    lambda_recent = st.slider("Blend weight for recent agreement (λ)", 0.0, 1.0, 0.35, 0.05)
    min_agree = st.slider("Quick pre-filter: minimum agreement on last-k", 0.0, 1.0, 0.67, 0.01)

# ---------------------------
# Buttons in same row
# ---------------------------
col1, col2 = st.columns([3, 1])  # left wide for Fetch, right narrow for Clear Cache

with col1:
    fetch_clicked = st.button("Fetch & Analyze", use_container_width=True)

with col2:
    if st.button("Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared!")

# ---------------------------
# Fetch logic
# ---------------------------
if fetch_clicked:
    with st.spinner("Fetching test data..."):
        prev_day_df, curr_day_df = get_prev_and_curr_day_data(test_date.strftime("%Y-%m-%d"))

    if prev_day_df is None or curr_day_df is None:
        st.error("Failed to fetch test day data / No data available.")
    else:
        curr_n_candles = min(curr_candle_input, len(curr_day_df))
        test_date_str = test_date.strftime("%Y-%m-%d")

        with st.spinner("Matching patterns..."):
            top_matches = get_cached_top_matches(
                history_df, prev_day_df, curr_day_df, test_date_str,
                top_k, prev_n_candles, curr_n_candles,
                matching_method, recent_k, min_agree, lambda_recent
            )

        sentiment, prob = get_cached_sentiment(top_matches, curr_n_candles)

        curr_day_df_c = curr_day_df.copy()
        curr_day_df = curr_day_df.iloc[:curr_n_candles]

        st.subheader(f"Test Day Candles: {test_date_str}")
        combined_test_df = pd.concat([prev_day_df, curr_day_df]).sort_values(by='time')
        combined_test_df_c = pd.concat([prev_day_df, curr_day_df_c]).sort_values(by='time')

        st.pyplot(plot_candle_chart(combined_test_df, f"Test Day: {test_date_str} | {sentiment}: {prob:.1f}%"))

        st.subheader(f"Top {top_k} Matches (Predicted | Test | Match)")
        for i, (match_date, sim, fut_sequence, hist_df) in enumerate(top_matches, 1):
            st.markdown(f"---\n### {i}. Match Date: `{match_date}` — Similarity: `{sim:.2f}`")
            remaining_df = hist_df.iloc[24 + curr_n_candles:].reset_index(drop=True)
            remaining_feat = extract_candle_features(remaining_df)
            cleaned = filter_noise((fut_sequence or "") + remaining_feat, min_len=3)

            cols = st.columns(3)
            with cols[0]:
                st.pyplot(plot_candle_chart(combined_test_df, f"Test Day | {test_date_str}"))
            with cols[1]:
                st.pyplot(plot_candle_chart(hist_df.iloc[:25 + curr_n_candles], f"Matched Day | {match_date}"))
            with cols[2]:
                st.pyplot(plot_candle_chart(hist_df, f"Predicted Day | {match_date} | sim: {sim:.2f}"))
