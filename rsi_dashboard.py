import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from streamlit_autorefresh import st_autorefresh
import json
import os

# ========== CONFIG ==========
st.set_page_config(layout="wide")

st.markdown("""
    <style>
        .main > div:first-child {
            padding-top: 0rem;
        }
        .block-container {
            padding-top: 0rem;
        }
        header {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

st_autorefresh(interval=1000, key="auto_refresh")

symbol = "BTCUSDT"
interval = "1m"
limit = 60
starting_capital = 500.0

log_file = "dip_log.json"
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        json.dump([], f)

state_file = "app_state.json"

def load_state():
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning("⚠️ State file corrupted. Resetting state.")
            os.remove(state_file)
    return {
        "balance": starting_capital,
        "holding": 0.0,
        "buy_price": 0.0,
        "buy_log": [],
        "sell_log": [],
        "live_prices": [],
        "trading_mode": "Automatic",
        "strategy_mode": "Manual",
        "buy_range_lower": 0.0,
        "sell_point": 0.0,
        "dip_threshold": 300,
        "profit_threshold": 250,
        "last_trade_index": -1,
        "last_anchor_index": -1,
        "last_anchor_price": 0.0
    }

def save_state(state):
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

state = load_state()

def fetch_candles(symbol, interval="1m", limit=60):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close']].astype(float)
    return df

df = fetch_candles(symbol, interval, limit)

live_price = df['close'].iloc[-1] if not df.empty else None
if live_price:
    state["live_prices"].append({
        "timestamp": pd.Timestamp.now().isoformat(),
        "price": live_price
    })
    state["live_prices"] = state["live_prices"][-60:]

strategy_mode = st.radio(
    "🧠 Select Strategy Mode:",
    ('Manual', 'Automatic', 'Green Candle Dip Strategy'),
    key='strategy_mode'
)
state['strategy_mode'] = strategy_mode
st.markdown(f"🧩 **Current Strategy Mode**: `{strategy_mode}`")

st.markdown("#### Strategy Threshold Settings")
state['dip_threshold'] = st.number_input("Dip Distance (units):", value=state.get('dip_threshold', 300), step=10)
state['profit_threshold'] = st.number_input("Profit Target (units):", value=state.get('profit_threshold', 250), step=10)

message = "🛌 No active trade."

if strategy_mode == "Manual":
    pass

elif strategy_mode == "Automatic":
    if state["holding"] == 0 and live_price and state.get('buy_range_lower', 0.0) > 0 and live_price <= state.get('buy_range_lower', float('inf')):
        state["buy_price"] = live_price
        state["holding"] = state["balance"] / live_price
        state["balance"] = 0
        state["buy_log"].append((len(df) - 1, live_price))
        message = f"✅ Auto-Bought at ₱{live_price:.2f}"

    if state["holding"] > 0 and live_price and state.get('sell_point', 0.0) > 0 and live_price >= state.get('sell_point', float('inf')):
        state["balance"] = state["holding"] * live_price
        state["holding"] = 0
        state["sell_log"].append((len(df) - 1, live_price))
        message = f"💰 Auto-Sold at ₱{live_price:.2f}"

elif strategy_mode == "Green Candle Dip Strategy":
    green_candles = df[df['close'] > df['open']]
    new_anchor = None
    for idx in range(len(df)):
        if df['close'].iloc[idx] > df['open'].iloc[idx]:
            if df['high'].iloc[idx] > state['last_anchor_price'] and idx > state['last_trade_index']:
                new_anchor = idx
                state['last_anchor_index'] = new_anchor
                state['last_anchor_price'] = df['high'].iloc[idx]

    anchor_index = state['last_anchor_index']
    green_anchor_price = state['last_anchor_price']

    if anchor_index is not None:
        for i in range(anchor_index + 1, len(df)):
            current = df.iloc[i]
            if green_anchor_price and state["holding"] == 0:
                if green_anchor_price - current['low'] >= state['dip_threshold']:
                    state["buy_price"] = current['low']
                    state["holding"] = state["balance"] / current['low']
                    state["balance"] = 0
                    state["buy_log"].append((i, current['low']))
                    state['last_trade_index'] = i
                    message = f"✅ Strategy Buy at ₱{current['low']:.2f} from Green ₱{green_anchor_price:.2f}"
                    break

            if state["holding"] > 0:
                profit = current['close'] - state["buy_price"]
                if profit >= state['profit_threshold']:
                    state["balance"] = state["holding"] * current['close']
                    state["holding"] = 0
                    state["sell_log"].append((i, current['close']))
                    state['last_trade_index'] = i
                    message = f"💰 Strategy Sell at ₱{current['close']:.2f} (Profit: ₱{profit:.2f})"
                    break
    else:
        st.warning("⚠️ No green candle anchor found yet. Waiting for new data...")

fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    increasing_line_color='green',
    decreasing_line_color='red'
)])

if strategy_mode == "Green Candle Dip Strategy" and green_anchor_price and anchor_index is not None:
    fig.add_trace(go.Scatter(
        x=[df.index[anchor_index]],
        y=[green_anchor_price],
        mode="markers+text",
        marker=dict(color="lightgreen", size=10),
        text=["Green Anchor"],
        textposition="top center",
        name="Green Anchor"
    ))

if state["buy_log"]:
    last_buy = state["buy_log"].pop()
    fig.add_trace(go.Scatter(
        x=[df.index[last_buy[0]]],
        y=[last_buy[1]],
        mode="markers+text",
        marker=dict(color="blue", size=12),
        text=["Buy"],
        textposition="top center",
        name="Buy"
    ))
    state["buy_log"].append(last_buy)

if state["sell_log"]:
    last_sell = state["sell_log"].pop()
    fig.add_trace(go.Scatter(
        x=[df.index[last_sell[0]]],
        y=[last_sell[1]],
        mode="markers+text",
        marker=dict(color="lime", size=12),
        text=["Sell"],
        textposition="top center",
        name="Sell"
    ))
    state["sell_log"].append(last_sell)

if live_price:
    fig.add_trace(go.Scatter(
        x=[df.index[-1]],
        y=[live_price],
        mode="markers+text",
        marker=dict(color="yellow", size=10),
        text=["Live"],
        textposition="top center",
        name="Live Price"
    ))
    fig.add_hline(
        y=live_price,
        line_dash="dot",
        line_color="yellow",
        annotation_text=f"Live Price: {live_price:.2f}",
        annotation_position="top right"
    )

fig.update_layout(
    xaxis_rangeslider_visible=False,
    yaxis=dict(range=[df['low'].min() * 0.999, df['high'].max() * 1.001]),
    yaxis_title="Price (USDT)",
    title="📈 BTC/USDT Candle Chart (1-Min)",
    margin=dict(t=30, b=20),
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

@st.cache_data(ttl=300)
def fetch_full_day():
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit=1440"
    data = requests.get(url).json()
    df_full = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore']
    )
    df_full['timestamp'] = pd.to_datetime(df_full['timestamp'], unit='ms')
    df_full.set_index('timestamp', inplace=True)
    return df_full[['open', 'high', 'low', 'close']].astype(float)

st.markdown("---")
st.markdown("### 🔢 1-Day Full Candlestick Data")
df_day = fetch_full_day()
fig_day = go.Figure(data=[go.Candlestick(
    x=df_day.index,
    open=df_day['open'],
    high=df_day['high'],
    low=df_day['low'],
    close=df_day['close'],
    increasing_line_color='green',
    decreasing_line_color='red'
)])
fig_day.update_layout(
    xaxis_rangeslider_visible=True,
    yaxis_title="Price (USDT)",
    template="plotly_dark",
    title="📈 BTC/USDT - Full Day Candlestick Chart"
)
st.plotly_chart(fig_day, use_container_width=True)

if strategy_mode == "Manual":
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Manual Buy") and state['balance'] > 0 and live_price:
            state["buy_price"] = live_price
            state["holding"] = state["balance"] / live_price
            state["balance"] = 0
            state["buy_log"].append((len(df) - 1, live_price))
            message = f"✅ Manually Bought at ₱{live_price:.2f}"
            save_state(state)
            st.experimental_rerun()
    with col2:
        if st.button("Manual Sell") and state["holding"] > 0 and live_price:
            state["balance"] = state["holding"] * live_price
            state["holding"] = 0
            state["sell_log"].append((len(df) - 1, live_price))
            message = f"💰 Manually Sold at ₱{live_price:.2f}"
            save_state(state)
            st.experimental_rerun()

st.markdown(f"💼 Current Balance: ₱{state['balance']:.2f}")
if state["holding"] > 0:
    st.markdown(f"📅 Holding {state['holding']:.6f} BTC bought at ₱{state['buy_price']:.2f}")
st.info(message)

save_state(state)
