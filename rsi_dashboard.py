import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from streamlit_autorefresh import st_autorefresh
import json
import os

# ========== CONFIG ==========
st.set_page_config(layout="wide")

# Remove top padding / header
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

st_autorefresh(interval=1000, key="auto_refresh")  # 1 sec refresh

symbol = "BTCUSDT"
interval = "1m"
limit = 60
starting_capital = 500.0  # ₱500

# JSON log path
log_file = "dip_log.json"
if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        json.dump([], f)

# ===== APP STATE MANAGEMENT =====
state_file = "app_state.json"

def load_state():
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            return json.load(f)
    else:
        return {
            "balance": starting_capital,
            "holding": 0.0,
            "buy_price": 0.0,
            "buy_log": [],
            "sell_log": [],
            "live_prices": [],
            "trading_mode": "Automatic", # Add default trading mode
            "buy_range_lower": 0.0,      # Add default buy price point
            "sell_point": 0.0            # Add default sell price point
        }

def save_state(state):
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

state = load_state()

# ===== GET MARKET DATA =====
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

# ===== FETCH LIVE PRICE =====
def fetch_price(symbol="BTCUSDT"):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        response = requests.get(url, timeout=3).json()
        return float(response["price"])
    except Exception as e:
        st.error(f"Failed to fetch live price: {e}")
        return None

live_price = fetch_price(symbol)

# Store live price history
if live_price:
    state["live_prices"].append({
        "timestamp": pd.Timestamp.now().isoformat(), # Store timestamp as ISO format string
        "price": live_price
    })
    state["live_prices"] = state["live_prices"][-60:]

# ===== DETECT DIPS =====
def find_dips(df):
    dips = []
    for i in range(1, len(df)-1):
        if df['low'][i] < df['low'][i-1] and df['low'][i] < df['low'][i+1]:
            dips.append((i, df['low'][i]))
    return dips

dips = find_dips(df)

# ===== BUY/SELL LOGIC =====
message = "💤 No active trade."

# NEW STRATEGY: Buy if last 2 dips are consecutive
# NEW STRATEGY: Buy if last 2 dips are consecutive (Automatic Mode)
if state.get("trading_mode", "Automatic") == "Automatic": # Default to Automatic if not set
    # BUY: if live price is at or below the specified lower bound (Automatic Mode)
    if state["holding"] == 0 and live_price and state.get('buy_range_lower', 0.0) > 0 and live_price <= state.get('buy_range_lower', float('inf')):
        state["buy_price"] = live_price
        state["holding"] = state["balance"] / live_price
        state["balance"] = 0
        state["buy_log"].append((len(df) - 1, live_price))
        message = f"✅ Bought at ₱{live_price:.2f}"

    # SELL: if live price is at or above the specified sell point (Automatic Mode)
    if state["holding"] > 0 and live_price and state.get('sell_point', 0.0) > 0 and live_price >= state.get('sell_point', float('inf')):
        state["balance"] = state["holding"] * live_price
        state["holding"] = 0
        state["sell_log"].append((len(df) - 1, live_price))
        message = f"💰 Sold at ₱{live_price:.2f}"

# ===== LOG DIPS TO JSON =====
with open(log_file, "r") as f:
    logged = json.load(f)

for i, price in dips:
    if {"candle": i, "price": price} not in logged:
        logged.append({"candle": i, "price": price})

if message.startswith("✅ Bought"):
    logged = []

with open(log_file, "w") as f:
    json.dump(logged, f, indent=2)

# ===== PLOT CANDLESTICK CHART =====
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    increasing_line_color='green',
    decreasing_line_color='red'
)])

# Mark dips
for i, price in dips:
    fig.add_trace(go.Scatter(
        x=[df.index[i]],
        y=[price],
        mode="markers",
        marker=dict(color="orange", size=10),
        name="Dip"
    ))

# Mark buy
if state["buy_log"]:
    last_buy = state["buy_log"][-1]
    fig.add_trace(go.Scatter(
        x=[df.index[last_buy[0]]],
        y=[last_buy[1]],
        mode="markers+text",
        marker=dict(color="blue", size=12),
        text=["Buy"],
        textposition="top center",
        name="Buy"
    ))

# Mark sell
if state["sell_log"]:
    last_sell = state["sell_log"][-1]
    fig.add_trace(go.Scatter(
        x=[df.index[last_sell[0]]],
        y=[last_sell[1]],
        mode="markers+text",
        marker=dict(color="lime", size=12),
        text=["Sell"],
        textposition="top center",
        name="Sell"
    ))

# Add live price marker
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

# Update layout
fig.update_layout(
    xaxis_rangeslider_visible=False,
    yaxis=dict(range=[df['low'].min() * 0.999, df['high'].max() * 1.001]),
    yaxis_title="Price (USDT)",
    title="📈 BTC/USDT Candle Chart (1-Min)",
    margin=dict(t=30, b=20),
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# ===== OPTIONAL MINI LIVE PRICE CHART =====
if state["live_prices"]:
    df_live = pd.DataFrame(state["live_prices"])
    df_live['timestamp'] = pd.to_datetime(df_live['timestamp']) # Convert timestamp back to datetime
    fig_live = go.Figure()
    fig_live.add_trace(go.Scatter(
        x=df_live["timestamp"],
        y=df_live["price"],
        mode="lines+markers",
        name="Live Price"
    ))
    fig_live.update_layout(
        title="📈 Live BTC Price (1s updates)",
        yaxis_title="Price (USDT)",
        template="plotly_dark",
        margin=dict(t=30, b=20)
    )
    st.plotly_chart(fig_live, use_container_width=True)

# ===== TRADING CONTROLS AND STATUS =====
st.markdown("### 📊 Trading Controls & Status")

# Add mode selection
trading_mode = st.radio(
    "Select Trading Mode:",
    ('Automatic', 'Manual'),
    key='trading_mode'
)

# Store trading mode in state
state['trading_mode'] = trading_mode

# Add input fields for buy range and sell point
# Add input fields for buy and sell points
st.markdown("#### Automatic Trading Settings")
state['buy_range_lower'] = st.number_input(
    "Buy Price Point (Buy at or below):",
    value=state.get('buy_range_lower', 0.0),
    format="%.2f",
    key='buy_point_input' # Changed key to be unique
)

state['sell_point'] = st.number_input(
    "Sell Price Point (Sell at or above):",
    value=state.get('sell_point', 0.0),
    format="%.2f",
    key='sell_point_input'
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Manual Buy"):
        # Manual Buy Logic Placeholder
        if state['balance'] > 0 and live_price:
            state["buy_price"] = live_price
            state["holding"] = state["balance"] / live_price
            state["balance"] = 0
            state["buy_log"].append((len(df) - 1, live_price))
            message = f"✅ Manually Bought at ₱{live_price:.2f}"
            save_state(state) # Save state immediately after manual action
            st.experimental_rerun() # Rerun to update display

with col2:
    if st.button("Manual Sell"):
        # Manual Sell Logic Placeholder
        if state["holding"] > 0 and live_price:
            state["balance"] = state["holding"] * live_price
            state["holding"] = 0
            state["sell_log"].append((len(df) - 1, live_price))
            message = f"💰 Manually Sold at ₱{live_price:.2f}"
            save_state(state) # Save state immediately after manual action
            st.experimental_rerun() # Rerun to update display


st.markdown(f"💼 Current Balance: ₱{state['balance']:.2f}")
if state["holding"] > 0:
    st.markdown(f"📥 Holding {state['holding']:.6f} BTC bought at ₱{state['buy_price']:.2f}")
st.info(message)

# ===== SAVE STATE =====
save_state(state)

# ===== FULL RESET BUTTON =====
st.markdown("---")
if st.button("🔁 FULL RESET — Start New Simulation"):
    # 1. Clear all Streamlit session state variables
    st.session_state.balance = starting_capital
    st.session_state.holding = 0.0
    st.session_state.buy_price = 0.0
    st.session_state.buy_log = []
    st.session_state.sell_log = []
    st.session_state.live_prices = []

    # Ensure session state is cleared before rerunning
    st.experimental_rerun()



    # 2. Clear dip log file
    if os.path.exists(log_file):
        os.remove(log_file)


    # 3. Clear other log files
    for extra_log in ["buy_log.json", "sell_log.json", "trade_log.json"]:
        if os.path.exists(extra_log):
            os.remove(extra_log)

    st.success("✅ Simulation fully reset — all logs cleared.")
    st.experimental_rerun()


