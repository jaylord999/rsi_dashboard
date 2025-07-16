import os # New import
from flask import Flask, request, jsonify

app = Flask(__name__)

API_KEY = "your_secret_api_key"  # Use this in local Streamlit headers

# Simulated config and state (could be replaced with file/DB later)
config = {
    "buy_price": 120000,
    "auto_mode": True
}

state = {
    "balance": 500.0,
    "holding": 0.0,
    "last_trade": "None"
}

@app.route("/config", methods=["GET", "POST"])
def handle_config():
    if request.headers.get("Authorization") != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 403
    
    if request.method == "GET":
        return jsonify(config)
    
    data = request.get_json()
    config.update(data)
    return jsonify({"status": "Config updated", "config": config})

@app.route("/status", methods=["GET"])
def handle_status():
    if request.headers.get("Authorization") != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify(state)

@app.route("/trade", methods=["POST"])
def handle_trade():
    if request.headers.get("Authorization") != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 403
    
    data = request.get_json()
    price = data.get("price")
    action = data.get("action", "buy")

    if action == "buy":
        state["holding"] += 1
        state["balance"] -= price
        state["last_trade"] = f"BUY at {price}"
    elif action == "sell":
        state["holding"] -= 1
        state["balance"] += price
        state["last_trade"] = f"SELL at {price}"

    return jsonify({"status": "Trade executed", "state": state})

@app.route("/ping", methods=["GET"])
def handle_ping():
    return jsonify({"status": "pong"}), 200

@app.route("/reset", methods=["POST"])
def handle_reset():
    if request.headers.get("Authorization") != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 403
    
    global config, state
    config = {
        "buy_price": 120000,
        "auto_mode": True
    }
    state = {
        "balance": 500.0,
        "holding": 0.0,
        "last_trade": "None"
    }
    return jsonify({"status": "Bot reset", "config": config, "state": state})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000)) # Get port from environment variable, default to 10000
    app.run(host="0.0.0.0", port=port)
