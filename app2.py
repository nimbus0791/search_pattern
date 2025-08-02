import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mplfinance as mpf
from numpy.linalg import norm
import upstox_client
from upstox_client.rest import ApiException
import math

# Constants
INSTRUMENT_KEY = 'NSE_INDEX|Nifty Bank'
DB_PATH = "banknifty-15min.csv"
DEFAULT_PREV_N_CANDLES = 25
DEFAULT_CURR_N_CANDLES = 4

# ---------------------------
# Data Fetching and Preprocessing
# ---------------------------
def fetch_historical_data(from_date, to_date):
    api = upstox_client.HistoryApi()
    try:
        resp = api.get_historical_candle_data1(
            INSTRUMENT_KEY, '1minute',
            to_date, from_date,
            '2.0'
        )
        return resp.data.candles
    except ApiException as e:
        st.error(f"API error: {e}")
        return []

def fetch_live_data():
    api_version = '2.0'

    api_instance = upstox_client.HistoryApi()
    interval = '1minute'
    candles = []

    try:
        api_response = api_instance.get_intra_day_candle_data(INSTRUMENT_KEY,interval,api_version)
        candles = api_response.data.candles
    except ApiException as e:
        print("Exception when calling HistoryApi->get_historical_candle_data: %s\n" % e)

    return candles

def change_time_frame(df, time_frame):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }
    df2 = df.resample(time_frame).agg(agg).dropna().reset_index()
    df2['time'] = df2['time'].dt.tz_localize(None)
    return df2

def get_prev_and_curr_day_data(test_date_str):
    test_date = datetime.strptime(test_date_str, "%Y-%m-%d").date()
    current_date = datetime.now().date()
    
    curr_candles = []
    prev_candles = []

    # Compare
    if current_date == test_date:
        curr_candles = fetch_live_data()    
    elif current_date > test_date:
        curr_candles = fetch_historical_data(test_date_str, test_date_str)
    else:
        print("No data available")

    if len(curr_candles) == 0:
        print("No data to search")
        return None, None

    offset = 1
    raw = []
    while True:
        prev_str = (test_date - timedelta(days=offset)).strftime("%Y-%m-%d")
        prev_candles = fetch_historical_data(prev_str, prev_str)
        tmp = pd.DataFrame(prev_candles, columns=["time", "open", "high", "low", "close", "volume", "oi"])
        tmp['time'] = pd.to_datetime(tmp['time']).dt.tz_localize(None)
        if tmp['time'].dt.date.nunique() >= 1:
            break
        offset += 1

    prev_candles = [candle[:5] for candle in prev_candles]
    prev_df = pd.DataFrame(prev_candles, columns=["time", "open", "high", "low", "close"])
    prev_df['time'] = pd.to_datetime(prev_df['time']).dt.tz_localize(None)
    prev_df = change_time_frame(prev_df, "15min")
    
    curr_candles = [candle[:5] for candle in curr_candles]
    curr_df = pd.DataFrame(curr_candles, columns=["time", "open", "high", "low", "close"])
    curr_df['time'] = pd.to_datetime(curr_df['time']).dt.tz_localize(None)
    curr_df = change_time_frame(curr_df, "15min")

    return prev_df, curr_df


# ---------------------------
# Feature Extraction
# ---------------------------
def extract_candle_features(df, min_quantum=0.001):
    swings = []
    n = len(df)

    # 1. Initialize direction from first candle
    if df.loc[0, 'close'] > df.loc[0, 'open']:
        direction = 'up'
    else:
        direction = 'down'

    start_idx = 0
    end_idx = 0

    for i in range(1, n):
        curr_close = df.loc[i, 'close']
        prev_close = df.loc[i - 1, 'close']

        # Detect direction
        if direction == 'up':
            if curr_close >= prev_close:
                # Continue uptrend
                end_idx = i
            else:
                # Direction changed to down
                pct = (df.loc[end_idx, 'high'] - df.loc[start_idx, 'low']) / df.loc[start_idx, 'low']
                swings.append(pct)

                direction = 'down'
                start_idx = end_idx
                end_idx = i
        else:
            if curr_close <= prev_close:
                # Continue downtrend
                end_idx = i
            else:
                # Direction changed to up
                pct = (df.loc[start_idx, 'high'] - df.loc[end_idx, 'low']) / df.loc[start_idx, 'high']
                swings.append(-pct)

                direction = 'up'
                start_idx = end_idx
                end_idx = i

    # Final swing (optional)
    if start_idx != end_idx:
        if direction == 'up':
            pct = (df.loc[end_idx, 'high'] - df.loc[start_idx, 'low']) / df.loc[start_idx, 'low']
            swings.append(pct)
        else:
            pct = (df.loc[start_idx, 'high'] - df.loc[end_idx, 'low']) / df.loc[start_idx, 'high']
            swings.append(-pct)

    # Encode swing pattern
    pattern = ''
    for swing in swings:
        symbol = 'u' if swing > 0 else 'd'
        count = int(abs(swing) / min_quantum)
        pattern += symbol * count

    return pattern


def lcs_similarity(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m+1, n+1), dtype=int)
    for i in range(m):
        for j in range(n):
            if seq1[i] == seq2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
    
    precision = dp[m][n] / m
    recall = dp[m][n] / n
    similarity = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Backtrack to find match end index in seq2
    i, j = m, n
    lcs_indices_seq2 = []
    while i > 0 and j > 0:
        if seq1[i-1] == seq2[j-1]:
            lcs_indices_seq2.append(j-1)
            i -= 1
            j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    last_match_index = max(lcs_indices_seq2)
    return similarity, seq1[last_match_index:]


def match_pattern(df, prev_test, curr_test, test_date_str, top_k, prev_n_candles, curr_n_candles):
    df2 = df.copy()
    df2['time'] = pd.to_datetime(df2['time'], utc=True).dt.tz_convert(None)
    df2 = df2.sort_values('time')
    df2['date'] = df2['time'].dt.date

    daily = {d: group.sort_values('time') for d, group in df2.groupby('date')}
    dates = sorted(daily)
    test_date = datetime.strptime(test_date_str, "%Y-%m-%d").date()

    test_df = pd.concat([prev_test.iloc[-prev_n_candles:], curr_test.iloc[:curr_n_candles]], ignore_index=True)
    test_feats = extract_candle_features(test_df)
    
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
        hist_feats = extract_candle_features(hist_df)
        
        sim, fut_sequence = lcs_similarity(hist_feats, test_feats)
        matches.append((d, sim, fut_sequence, hist_df_c))
    return sorted(matches, key=lambda x: -x[1])[:top_k]


def filter_noise(seq, min_len=2):
    # Step 1: Run-length encode
    rle = []
    prev = seq[0]
    count = 1
    for ch in seq[1:]:
        if ch == prev:
            count += 1
        else:
            rle.append((prev, count))
            prev = ch
            count = 1
    rle.append((prev, count))

    # Step 2: Fix leading noise
    if len(rle) > 1 and rle[0][1] < min_len:
        rle[1] = (rle[1][0], rle[0][1] + rle[1][1])
        rle.pop(0)

    # Step 3: Fix trailing noise
    if len(rle) > 1 and rle[-1][1] < min_len:
        rle[-2] = (rle[-2][0], rle[-2][1] + rle[-1][1])
        rle.pop()

    # Step 4: Merge noisy segments in the middle
    i = 1
    while i < len(rle) - 1:
        char, cnt = rle[i]
        if cnt < min_len:
            prev_char, prev_cnt = rle[i - 1]
            next_char, next_cnt = rle[i + 1]
            if prev_char == next_char:
                rle[i - 1] = (prev_char, prev_cnt + cnt + next_cnt)
                rle.pop(i)
                rle.pop(i)
                i -= 1
            else:
                if prev_cnt >= next_cnt:
                    rle[i - 1] = (prev_char, prev_cnt + cnt)
                    rle.pop(i)
                else:
                    rle[i + 1] = (next_char, next_cnt + cnt)
                    rle.pop(i)
        else:
            i += 1

    # Step 5: Reconstruct
    return ''.join([char * count for char, count in rle])


def group_pattern(pattern):
    grouped = []
    if not pattern:
        return grouped

    curr_char = pattern[0]
    count = 1

    for ch in pattern[1:]:
        if ch == curr_char:
            count += 1
            if count == 4:  # flush group at count = 3
                grouped.append((curr_char, 3))
                count = 1  # current ch becomes the start of next group
        else:
            if count > 0:
                grouped.append((curr_char, count))
            curr_char = ch
            count = 1

    if count > 0:
        grouped.append((curr_char, count))

    return grouped


def generate_predicted_candles_df_grouped(base_df, pattern, step_percent=0.0005, timeframe_minutes=15):
    """
    base_df: DataFrame with 'time' and 'close' columns
    pattern: string of 'u' and 'd'
    step_percent: percent movement per unit (e.g., 0.001 = 0.1%)
    """

    if base_df.empty:
        raise ValueError("Base DataFrame is empty.")

    last_time = pd.to_datetime(base_df['time'].iloc[-1])
    last_close = base_df['close'].iloc[-1]

    grouped_pattern = group_pattern(pattern)
    curr_price = last_close
    candles = []

    for i, (direction, count) in enumerate(grouped_pattern):

        time = last_time + timedelta(minutes=(i + 1) * timeframe_minutes)
        total_change = curr_price * step_percent * count
        if direction == 'u':
            open_price = curr_price
            close_price = curr_price + total_change
            low = open_price
            high = close_price
        elif direction == 'd':
            open_price = curr_price
            close_price = curr_price - total_change
            high = open_price
            low = close_price
        else:
            raise ValueError("Pattern must contain only 'u' or 'd'.")

        candle = {
            "time": time,
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close_price, 2),
        }
        candles.append(candle)
        curr_price = close_price

    predicted_df = pd.DataFrame(candles)
    
    return pd.concat([base_df, predicted_df], ignore_index=True)


def plot_candle_chart(df, title):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert(None)
    df.set_index('time', inplace=True)
    df = df[['open', 'high', 'low', 'close']]
    fig, ax = plt.subplots(figsize=(6, 4))
    mpf.plot(df, type='candle', ax=ax, datetime_format='%H:%M', show_nontrading=False, xrotation=45, style='yahoo')
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    return fig

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    return df

# ---------------- UI ---------------- #
st.set_page_config(layout="wide")
st.title("Top-K Pattern Matching Grid")

history_df = load_data(DB_PATH)

test_date = st.date_input("Select test date", datetime.now().date())
top_k = st.slider("Number of top matches to show", 4, 32, 20, step=4)

prev_n_candles = st.number_input("Number of previous day candles to match", min_value=1, max_value=25, value=DEFAULT_PREV_N_CANDLES, step=1)
curr_candle_input = st.number_input("Number of current day candles to match", min_value=1, max_value=25, value=DEFAULT_CURR_N_CANDLES, step=1)


if st.button("Fetch & Analyze"):
    with st.spinner("Fetching test data from Web..."):
        prev_day_df, curr_day_df = get_prev_and_curr_day_data(test_date.strftime("%Y-%m-%d"))

    if prev_day_df is None or curr_day_df is None:
        st.error("Failed to fetch test day data / No data available.")
    else:
        curr_n_candles = min(curr_candle_input, len(curr_day_df))
        with st.spinner("Calculating featues and matching..."):
            top_matches = match_pattern(history_df, prev_day_df, curr_day_df, test_date.strftime("%Y-%m-%d"), top_k, prev_n_candles, curr_n_candles)
        match_dates = [d for d, _, _, _ in top_matches]

        curr_day_df_c = curr_day_df.copy()
        curr_day_df = curr_day_df.iloc[:curr_n_candles]

        st.subheader(f"Test Day Candles: {test_date.strftime('%Y-%m-%d')}")
        combined_test_df = pd.concat([prev_day_df, curr_day_df]).sort_values(by='time')
        combined_test_df_c = pd.concat([prev_day_df, curr_day_df_c]).sort_values(by='time')
        st.pyplot(plot_candle_chart(combined_test_df, f"Test Day: {test_date.strftime('%Y-%m-%d')}"))

        st.subheader(f"Top {top_k} Matches (Showing 4 charts per row)")
        up_count = 0
        down_count = 0

        for i in range(len(top_matches)):
            match_date, sim, fut_sequence, hist_df = top_matches[i]
            st.markdown(f"---\n### {i+1}. Match Date: `{match_date}` — Similarity: `{sim:.2f}`")

            remaining_df = hist_df.iloc[24 + curr_n_candles:].reset_index(drop=True)
            
            remaining_feat = extract_candle_features(remaining_df)
            fut_sequence += remaining_feat
            cleaned = filter_noise(fut_sequence, min_len=3)
            print(fut_sequence, cleaned)
            
            title = f"{match_date} | sim: {sim:.2f}"


            cols = st.columns(3)
            with cols[0]:
                combined_test_df_with_pred = generate_predicted_candles_df_grouped(combined_test_df, cleaned)
                test_fig = plot_candle_chart(combined_test_df_with_pred, f"Test Day: {test_date.strftime('%Y-%m-%d')}")
                st.pyplot(test_fig)
            with cols[1]:
                test_fig = plot_candle_chart(combined_test_df_c, f"Test Day: {test_date.strftime('%Y-%m-%d')}")
                st.pyplot(test_fig)
            with cols[2]:
                match_fig = plot_candle_chart(hist_df, title)
                st.pyplot(match_fig)
                        