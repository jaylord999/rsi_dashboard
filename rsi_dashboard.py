import json
import time
import os
import requests
from datetime import datetime

# CONFIG
symbol = "BTCUSDT"
interval = "1m"
state_file = "app_state.json"
starting_capital = 500.0

def fetch_latest_candle():
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit=1"
    try:
        data = requests.get(url).json()
        if not data or isinstance(data, dict):
            return None
        candle = data[0]
        return {
            "open": float(candle[1]),
            "high": float(candle[2]),
            "low": float(candle[3]),
            "close": float(candle[4])
        }
    except Exception as e:
        print(f"[FETCH ERROR]: {e}")
        return None

def load_state():
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except:
            pass
    return {
        "balance": starting_capital,
        "holding": 0.0,
        "buy_price": 0.0,
        "buy_log": [],
        "sell_log": [],
        "dip_threshold": 300,
        "profit_threshold": 250,
        "last_anchor_price": 0.0,
        "anchor_timestamp": ""
    }

def save_state(state):
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

def run_bot():
    print("[BOT] Started trading loop.")
    while True:
        try:
            state = load_state()
            candle = fetch_latest_candle()
            if not candle:
                time.sleep(10)
                continue

            live_price = candle["close"]
            high = candle["high"]
            low = candle["low"]

            # Update anchor if new high
            if state["last_anchor_price"] == 0 or high > state["last_anchor_price"]:
                state["last_anchor_price"] = high
                state["anchor_timestamp"] = datetime.utcnow().isoformat()

            dip = state["last_anchor_price"] - low

            # BUY
            if state["holding"] == 0 and state["balance"] > 0 and dip >= state["dip_threshold"]:
                state["buy_price"] = live_price
                state["holding"] = state["balance"] / live_price
                state["balance"] = 0
                state["buy_log"].append((datetime.utcnow().isoformat(), live_price))
                print(f"[BUY] {live_price:.2f}")

            # SELL
            elif state["holding"] > 0 and state["buy_price"] > 0:
                profit = live_price - state["buy_price"]
                if profit >= state["profit_threshold"]:
                    state["balance"] = state["holding"] * live_price
                    state["sell_log"].append((datetime.utcnow().isoformat(), live_price))
                    print(f"[SELL] {live_price:.2f} | P&L: {state['balance'] - starting_capital:.2f}")
                    state["buy_price"] = 0
                    state["holding"] = 0

            save_state(state)

        except Exception as e:
            print(f"[BOT ERROR]: {e}")

        time.sleep(10)

if __name__ == "__main__":
    run_bot()
