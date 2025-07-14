# config.py

# Binance API Keys (not needed unless placing real orders)
API_KEY = "PrdKAtYrE7boqISswVI4nqQCwKPRXDIoLa42YMybDfQCDucGLqJYujZoSfq64wKt"
API_SECRET = "QrwLkLVBS31d8VY4GgzlRCi4snWbUhrWEh8ylsQAfmPXieMWcvzFYakNW5Tk858I"

# Trading symbol (e.g., BTCUSDT)
SYMBOL = "BTCUSDT"

# Candle resolution: '1s', '1m', etc.
INTERVAL = "1s"

# How many candles to fetch from Binance (60 = last 60s)
CANDLE_LIMIT = 60

# Strategy Parameters
DIP_THRESHOLD = 0.001       # 0.1% dip
PROFIT_TARGET = 0.0015      # 0.15% gain
STOP_LOSS = 0.002           # 0.2% drop
