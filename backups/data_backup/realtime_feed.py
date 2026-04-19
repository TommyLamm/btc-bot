
import asyncio
import json
import websockets
import requests
import polars as pl
from collections import deque
from datetime import datetime, timezone

HL_API = "https://" + "api.hyperliquid.xyz/info"

class BTCRealtimeFeed:

    def __init__(self, buffer_size=25000):
        self.buffer   = deque(maxlen=buffer_size)
        self.obi_buffer = deque(maxlen=buffer_size)  # 新增：OBI记录
        self.current_candle = None
        self.callbacks = []

    def on_candle_close(self, callback):
        self.callbacks.append(callback)

    def fetch_obi(self):
        try:
            payload = {"type": "l2Book", "coin": "BTC"}
            resp    = requests.post(HL_API, json=payload, timeout=3)
            data    = resp.json()
            levels  = data.get("levels", [[], []])
            bids    = levels[0][:10] if len(levels) > 0 else []
            asks    = levels[1][:10] if len(levels) > 1 else []
            bid_vol = sum(float(b["sz"]) for b in bids)
            ask_vol = sum(float(a["sz"]) for a in asks)
            spread  = float(asks[0]["px"]) - float(bids[0]["px"]) if bids and asks else 0
            obi     = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)
            return {"obi": obi, "spread": spread, "bid_vol": bid_vol, "ask_vol": ask_vol}
        except Exception as e:
            return {"obi": 0.0, "spread": 0.0, "bid_vol": 0.0, "ask_vol": 0.0}

    async def connect(self):
        url = "wss://fstream.binance.com/stream?streams=btcusdt@kline_5m"
        print("WebSocket 连接中...")
        async with websockets.connect(url) as ws:
            print("已连接，开始接收数据")
            async for message in ws:
                await self._handle_message(json.loads(message))

    async def _handle_message(self, msg):
        kline = msg["data"]["k"]
        candle = {
            "timestamp": kline["t"],
            "open":      float(kline["o"]),
            "high":      float(kline["h"]),
            "low":       float(kline["l"]),
            "close":     float(kline["c"]),
            "volume":    float(kline["v"]),
            "is_closed": kline["x"],
        }
        self.current_candle = candle

        if candle["is_closed"]:
            # 采集 OBI
            obi_data = self.fetch_obi()
            candle["obi"]     = obi_data["obi"]
            candle["spread"]  = obi_data["spread"]
            candle["bid_vol"] = obi_data["bid_vol"]
            candle["ask_vol"] = obi_data["ask_vol"]

            self.buffer.append(candle)
            self.obi_buffer.append({
                "timestamp": candle["timestamp"],
                "obi":       obi_data["obi"],
                "spread":    obi_data["spread"],
                "bid_vol":   obi_data["bid_vol"],
                "ask_vol":   obi_data["ask_vol"],
            })

            dt = datetime.fromtimestamp(candle["timestamp"]/1000, tz=timezone.utc)
            print(f"K线收盘 {dt.strftime('%H:%M')} | close={candle['close']:.1f} | OBI={obi_data['obi']:+.3f}")

            # 每100根保存一次OBI数据
            if len(self.obi_buffer) % 100 == 0:
                self.save_obi()

            for cb in self.callbacks:
                await cb(candle, list(self.buffer))

    def save_obi(self):
        if not self.obi_buffer:
            return
        df = pl.DataFrame(list(self.obi_buffer))
        try:
            existing = pl.read_parquet("data/btc_obi_realtime.parquet")
            df = pl.concat([existing, df]).unique("timestamp").sort("timestamp")
        except Exception:
            pass
        df.write_parquet("data/btc_obi_realtime.parquet")
        print(f"  OBI 数据已保存，共 {len(df)} 条")

    def get_dataframe(self):
        if not self.buffer:
            return pl.DataFrame()
        return pl.DataFrame(list(self.buffer))
