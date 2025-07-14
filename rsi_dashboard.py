import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from streamlit_autorefresh import st_autorefresh
import json
import os

# ========== CONFIG ========== #
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

# ========== SETTINGS ========== #
symbol = "BTCUSDT"
interval = "1m"
limit = 60
starting_capital = 500.0
state_file = "app_state.json"

# ========== STATE ========== #
def load_state():
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            os.remove(state_file)
    return {
        "balance": starting_capital,
        "holding": 0.0,
        "buy_price": 0.0,
        "buy_log": [],
        "sell_log": [],
        "live_prices": [],
        "dip_threshold": 300,
        "profit_threshold": 250,
        "last_anchor_price": 0.0,
        "anchor_timestamp": ""
    }

def save_state(state):
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

state = load_state()

# ========== FETCH CANDLE DATA ========== #
def fetch_candles(symbol, interval="1m", limit=60):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        data = requests.get(url).json()
        if not data or isinstance(data, dict):
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'num_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df[['open', 'high', 'low', 'close']].astype(float)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Fetch 1-minute data for main chart
df = fetch_candles(symbol, interval, limit)

# Fetch 1-day data for daily chart (1440 minutes = 24 hours)
df_daily = fetch_candles(symbol, "1m", 1440)

# ========== PRICE TRACKING ========== #
if not df.empty:
    live_price = df['close'].iloc[-1]
    state["live_prices"].append({
        "timestamp": pd.Timestamp.now().isoformat(),
        "price": live_price
    })
    state["live_prices"] = state["live_prices"][-60:]
else:
    st.error("⚠️ No candlestick data returned. Check your network or Binance API status.")
    save_state(state)
    st.stop()

# ========== PERSISTENT ANCHOR LOGIC ========== #
# Get current highest high in the dataset
current_highest = df['high'].max()

# Initialize anchor if not set, or update if we have a new higher high
if state.get('last_anchor_price', 0) == 0:
    # First time running - set initial anchor
    anchor_price = current_highest
    state['last_anchor_price'] = anchor_price
    state['anchor_timestamp'] = df['high'].idxmax().isoformat()
    anchor_status = f"🎯 Initial anchor set at ₱{anchor_price:.2f}"
    
elif current_highest > state['last_anchor_price']:
    # New higher high detected - update anchor
    anchor_price = current_highest
    state['last_anchor_price'] = anchor_price
    state['anchor_timestamp'] = df['high'].idxmax().isoformat()
    anchor_status = f"🎯 New anchor! Updated to ₱{anchor_price:.2f}"
    
else:
    # Keep existing anchor
    anchor_price = state['last_anchor_price']
    anchor_status = f"🎯 Keeping anchor at ₱{anchor_price:.2f}"

# Get ONLY the latest candle
latest_candle = df.iloc[-1]
live_price = latest_candle['close']

message = "🛌 No active trade."

# ========== DISPLAY OPTION SETTING ========== #
display_option = st.selectbox(
    "Buy/Sell Display:",
    ["Markers", "Lines", "Both"],
    index=0,
    key="display_option_top"
)

# ========== FIXED STRATEGY LOGIC ========== #
# --- BUY LOGIC: Only check if we're not holding ---
if state['holding'] == 0 and state['balance'] > 0:
    # Check if current candle's low dips enough below anchor
    dip_amount = anchor_price - latest_candle['low']
    
    if dip_amount >= state['dip_threshold']:
        # Execute buy at current price
        state["buy_price"] = live_price
        state["holding"] = state["balance"] / live_price
        state["balance"] = 0
        state["buy_log"].append((len(df)-1, live_price))
        message = f"✅ Bought at ₱{live_price:.2f} (Dip: ₱{dip_amount:.2f} from anchor ₱{anchor_price:.2f})"
    else:
        # Show how much more dip is needed
        needed_dip = state['dip_threshold'] - dip_amount
        message = f"⏳ Waiting for dip... Need ₱{needed_dip:.2f} more (Current dip: ₱{dip_amount:.2f})"

# --- SELL LOGIC: Only check if we're holding ---
elif state["holding"] > 0 and state["buy_price"] > 0:
    profit = live_price - state['buy_price']
    
    if profit >= state['profit_threshold']:
        # Execute sell at current price
        state["balance"] = state["holding"] * live_price
        total_profit = state["balance"] - starting_capital
        state["holding"] = 0
        state["buy_price"] = 0
        state["sell_log"].append((len(df)-1, live_price))
        message = f"💰 Sold at ₱{live_price:.2f} (Profit: ₱{profit:.2f}) | Total P&L: ₱{total_profit:.2f}"
    else:
        # Show current unrealized profit/loss
        needed_profit = state['profit_threshold'] - profit
        message = f"📈 Holding at ₱{state['buy_price']:.2f} | Current: ₱{live_price:.2f} | Need ₱{needed_profit:.2f} more profit"

# ========== MAIN PLOT ========== #
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    increasing_line_color='green',
    decreasing_line_color='red'
)])

# --- MARK ANCHOR PRICE ---
fig.add_hline(y=anchor_price, line_dash="dash", line_color="yellow", 
              annotation_text=f"Anchor: ₱{anchor_price:.2f}", 
              annotation_position="bottom right")

# --- MARK DIP THRESHOLD ---
dip_line = anchor_price - state['dip_threshold']
fig.add_hline(y=dip_line, line_dash="dot", line_color="orange", 
              annotation_text=f"Buy Zone: ₱{dip_line:.2f}", 
              annotation_position="top right")

# --- MARK LAST BUY ---
if state["buy_log"]:
    i, price = state["buy_log"][-1]
    if i < len(df):
        # Add markers
        if display_option in ["Markers", "Both"]:
            fig.add_trace(go.Scatter(
                x=[df.index[i]],
                y=[price],
                mode="markers+text",
                marker=dict(color="blue", size=12),
                text=["Buy"],
                textposition="top center",
                name="Buy Point"
            ))
        
        # Add vertical line
        if display_option in ["Lines", "Both"]:
            fig.add_vline(
                x=df.index[i],
                line_dash="solid",
                line_color="blue",
                line_width=2,
                annotation_text=f"Buy: ₱{price:.2f}",
                annotation_position="top"
            )

# --- MARK LAST SELL ---
if state["sell_log"]:
    i, price = state["sell_log"][-1]
    if i < len(df):
        # Add markers
        if display_option in ["Markers", "Both"]:
            fig.add_trace(go.Scatter(
                x=[df.index[i]],
                y=[price],
                mode="markers+text",
                marker=dict(color="lime", size=12),
                text=["Sell"],
                textposition="top center",
                name="Sell Point"
            ))
        
        # Add vertical line
        if display_option in ["Lines", "Both"]:
            fig.add_vline(
                x=df.index[i],
                line_dash="solid",
                line_color="lime",
                line_width=2,
                annotation_text=f"Sell: ₱{price:.2f}",
                annotation_position="top"
            )

# --- MARK PROFIT TARGET IF HOLDING ---
if state["holding"] > 0 and state["buy_price"] > 0:
    profit_target = state["buy_price"] + state["profit_threshold"]
    fig.add_hline(y=profit_target, line_dash="dot", line_color="lime", 
                  annotation_text=f"Profit Target: ₱{profit_target:.2f}", 
                  annotation_position="bottom left")

fig.update_layout(
    xaxis_rangeslider_visible=False,
    yaxis_title="Price (USDT)",
    title="📈 BTC/USDT Trading Bot - 1 Hour View",
    template="plotly_dark",
    showlegend=True,
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# ========== COMPACT SETTINGS UI ========== #
st.markdown("---")
settings_col1, settings_col2 = st.columns(2)

with settings_col1:
    state['dip_threshold'] = st.number_input("Dip Distance (units):", value=state.get('dip_threshold', 300), step=10, key="dip_input")

with settings_col2:
    state['profit_threshold'] = st.number_input("Profit Target (units):", value=state.get('profit_threshold', 250), step=10, key="profit_input")

# ========== DAILY CHART ========== #
st.markdown("---")
st.markdown("### 📊 Daily View (24 Hours)")

if not df_daily.empty:
    fig_daily = go.Figure(data=[go.Candlestick(
        x=df_daily.index,
        open=df_daily['open'],
        high=df_daily['high'],
        low=df_daily['low'],
        close=df_daily['close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    # Add anchor and thresholds to daily chart as well
    fig_daily.add_hline(y=anchor_price, line_dash="dash", line_color="yellow", 
                  annotation_text=f"Anchor: ₱{anchor_price:.2f}", 
                  annotation_position="bottom right")

    dip_line = anchor_price - state['dip_threshold']
    fig_daily.add_hline(y=dip_line, line_dash="dot", line_color="orange", 
                  annotation_text=f"Buy Zone: ₱{dip_line:.2f}", 
                  annotation_position="top right")

    if state["holding"] > 0 and state["buy_price"] > 0:
        profit_target = state["buy_price"] + state["profit_threshold"]
        fig_daily.add_hline(y=profit_target, line_dash="dot", line_color="lime", 
                      annotation_text=f"Profit Target: ₱{profit_target:.2f}", 
                      annotation_position="bottom left")

    # Add buy/sell markers to daily chart as well
    if state["buy_log"]:
        for idx, price in state["buy_log"][-10:]:  # Show last 10 trades
            if idx < len(df_daily):
                # Add markers
                if display_option in ["Markers", "Both"]:
                    fig_daily.add_trace(go.Scatter(
                        x=[df_daily.index[idx]],
                        y=[price],
                        mode="markers",
                        marker=dict(color="blue", size=8),
                        name="Buy",
                        showlegend=False
                    ))
                
                # Add vertical lines
                if display_option in ["Lines", "Both"]:
                    fig_daily.add_vline(
                        x=df_daily.index[idx],
                        line_dash="solid",
                        line_color="blue",
                        line_width=1,
                        opacity=0.7
                    )

    if state["sell_log"]:
        for idx, price in state["sell_log"][-10:]:  # Show last 10 trades
            if idx < len(df_daily):
                # Add markers
                if display_option in ["Markers", "Both"]:
                    fig_daily.add_trace(go.Scatter(
                        x=[df_daily.index[idx]],
                        y=[price],
                        mode="markers",
                        marker=dict(color="lime", size=8),
                        name="Sell",
                        showlegend=False
                    ))
                
                # Add vertical lines
                if display_option in ["Lines", "Both"]:
                    fig_daily.add_vline(
                        x=df_daily.index[idx],
                        line_dash="solid",
                        line_color="lime",
                        line_width=1,
                        opacity=0.7
                    )

    fig_daily.update_layout(
        xaxis_rangeslider_visible=False,
        yaxis_title="Price (USDT)",
        title="📈 BTC/USDT - 24 Hour View",
        template="plotly_dark",
        showlegend=False,
        height=400
    )

    st.plotly_chart(fig_daily, use_container_width=True)

# ========== DISPLAY STATUS ========== #
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"💼 **Balance**: ₱{state['balance']:.2f}")
    if state['holding'] > 0:
        current_value = state['holding'] * live_price
        st.markdown(f"📥 **Holding**: {state['holding']:.6f} BTC")
        st.markdown(f"💎 **Current Value**: ₱{current_value:.2f}")

with col2:
    st.markdown(f"📊 **Live Price**: ₱{live_price:.2f}")
    total_value = state['balance'] + (state['holding'] * live_price if state['holding'] > 0 else 0)
    total_pnl = total_value - starting_capital
    st.markdown(f"📈 **Total Value**: ₱{total_value:.2f}")
    st.markdown(f"💰 **Total P&L**: ₱{total_pnl:.2f}")

with col3:
    st.markdown(f"🎯 **Anchor**: ₱{anchor_price:.2f}")
    st.markdown(f"📉 **Dip Threshold**: ₱{state['dip_threshold']}")
    st.markdown(f"📈 **Profit Target**: ₱{state['profit_threshold']}")

st.info(message)
st.success(anchor_status)

# ========== TRADING HISTORY ========== #
if state["buy_log"] or state["sell_log"]:
    st.markdown("---")
    st.markdown("### 📊 Trading History")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if state["buy_log"]:
            st.markdown("**🔵 Buy Orders**")
            for i, (idx, price) in enumerate(state["buy_log"][-5:]):  # Show last 5
                st.write(f"{i+1}. ₱{price:.2f}")
    
    with col2:
        if state["sell_log"]:
            st.markdown("**🟢 Sell Orders**")
            for i, (idx, price) in enumerate(state["sell_log"][-5:]):  # Show last 5
                st.write(f"{i+1}. ₱{price:.2f}")

# ========== DEBUG PANEL ========== #
st.markdown("---")
with st.expander("🔧 Debug Info"):
    st.write(f"**Holding BTC**: {state['holding']:.6f}")
    st.write(f"**Buy Price**: ₱{state['buy_price']:.2f}")
    st.write(f"**Balance**: ₱{state['balance']:.2f}")
    st.write(f"**Anchor Price**: ₱{anchor_price:.2f}")
    st.write(f"**Current Highest**: ₱{current_highest:.2f}")
    st.write(f"**Latest Candle Low**: ₱{latest_candle['low']:.2f}")
    st.write(f"**Current Dip**: ₱{anchor_price - latest_candle['low']:.2f}")
    if state['holding'] > 0:
        st.write(f"**Current Profit**: ₱{live_price - state['buy_price']:.2f}")
    
    # Reset button
    if st.button("🔄 Reset Trading Bot"):
        if os.path.exists(state_file):
            os.remove(state_file)
        st.success("Bot reset! Refresh the page.")

# ========== SAVE STATE ========== #
save_state(state)
