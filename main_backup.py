import asyncio
import sys
import os
import threading
import polars as pl
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, "/root/btc-bot")

from data.realtime_feed import BTCRealtimeFeed
from factors.factor_engine import compute_factors
from engine.genetic_engine import run_genetic_search, FACTOR_COLS
from monitor.performance_monitor import PerformanceMonitor

WINRATE_THRESHOLD = 0.50
WINDOW_SIZE       = 200
MIN_TRADES        = 50
SEARCH_LOOKBACK   = 20000
SEARCH_NGEN       = 60
SEARCH_POP        = 300

position = {
    "side":        None,
    "entry_price": None,
    "entry_time":  None,
    "signal":      None,
}


class BTCFactorBot:

    def __init__(self):
        self.feed    = BTCRealtimeFeed(buffer_size=25000)
        self.monitor = PerformanceMonitor(
            winrate_threshold=WINRATE_THRESHOLD,
            window_size=WINDOW_SIZE,
            min_trades=MIN_TRADES,
        )
        self.oi_df = pl.read_parquet("data/btc_oi.parquet").sort("timestamp")
        self.oi_df = pl.read_parquet("data/btc_oi.parquet").sort("timestamp")
        self.df_history = pl.read_parquet("data/btc_5m_factors.parquet")
        self._init_factor()

    def _init_factor(self):
        print("🔍 初始化因子搜索...")
        result = run_genetic_search(
            self.df_history,
            lookback=SEARCH_LOOKBACK,
            n_gen=SEARCH_NGEN,
            pop_size=SEARCH_POP,
            verbose=True,
        )
        if result:
            self.monitor.update_factor(result)
        else:
            print("⚠️  初始搜索失败")

    def _background_search(self):
        if self.monitor.is_searching:
            return
        self.monitor.is_searching = True
        print(f"\n🔄 后台因子搜索启动... [{datetime.now(timezone.utc).strftime('%H:%M:%S')}]")
        try:
            result = run_genetic_search(
                self.df_history,
                lookback=SEARCH_LOOKBACK,
                n_gen=SEARCH_NGEN,
                pop_size=SEARCH_POP,
                verbose=False,
            )
            if result and result["val_winrate"] >= 0.52:
                self.monitor.update_factor(result)
                print(f"✅ 新因子已切换，验证集胜率: {result['val_winrate']:.3f}")
            else:
                wr = result["val_winrate"] if result else 0
                print(f"⚠️  搜索结果不理想 (胜率={wr:.3f})，保持当前因子")
        except Exception as e:
            print(f"❌ 后台搜索出错: {e}")
        finally:
            self.monitor.is_searching = False

    async def on_candle_close(self, candle, history):
        global position

        # 关键修复：datetime 明确用 ms 精度，与历史数据一致
        new_row = pl.DataFrame({
            "timestamp": pl.Series([candle["timestamp"]], dtype=pl.Int64),
            "open":      pl.Series([float(candle["open"])],   dtype=pl.Float64),
            "high":      pl.Series([float(candle["high"])],   dtype=pl.Float64),
            "low":       pl.Series([float(candle["low"])],    dtype=pl.Float64),
            "close":     pl.Series([float(candle["close"])],  dtype=pl.Float64),
            "volume":    pl.Series([float(candle["volume"])], dtype=pl.Float64),
            "datetime":  pl.Series([candle["timestamp"]], dtype=pl.Datetime("ms", "UTC")),
        })

        self.df_history = pl.concat([self.df_history, new_row], how="diagonal")
        self.df_history = compute_factors(self.df_history.tail(500), oi_df=self.oi_df)

        latest = self.df_history.tail(1).to_dicts()[0]
        factor_values = {col: latest.get(col, 0.0) for col in FACTOR_COLS}

        signal = self.monitor.generate_signal(factor_values)
        action = self.monitor.get_action(signal)

        # 平仓逻辑
        if position["side"] is not None:
            entry = position["entry_price"]
            close = candle["close"]
            pnl   = (close - entry) / entry if position["side"] == "long" \
                    else (entry - close) / entry

            bars_held = (candle["timestamp"] - position["entry_time"]) / (5 * 60 * 1000)
            should_exit = (bars_held >= 4) or \
                          (position["side"] == "long"  and action == "short") or \
                          (position["side"] == "short" and action == "long")

            if should_exit:
                status = self.monitor.record_trade(pnl, position["signal"])
                print(f"  平仓 {position['side']:5s} | PnL: {pnl*100:+.4f}% | "
                      f"胜率: {status.get('winrate', 'N/A')}")
                position = {"side": None, "entry_price": None,
                            "entry_time": None, "signal": None}

                if status.get("should_search") and not self.monitor.is_searching:
                    print(f"  ⚠️  {status['reason']} → 启动后台搜索")
                    t = threading.Thread(target=self._background_search, daemon=True)
                    t.start()

        # 开仓逻辑
        if position["side"] is None and action != "hold":
            position = {
                "side":        action,
                "entry_price": candle["close"],
                "entry_time":  candle["timestamp"],
                "signal":      signal,
            }
            print(f"  开仓 {action:5s} | price={candle['close']:.1f} signal={signal:.3f}")

        if len(self.monitor.trades) > 0 and len(self.monitor.trades) % 50 == 0:
            print(f"\n{'─'*50}")
            print(self.monitor.get_stats())
            print(f"{'─'*50}\n")

    async def run(self):
        self.feed.on_candle_close(self.on_candle_close)
        print("\n🚀 BTC因子交易机器人启动")
        print(f"   胜率阈值: {WINRATE_THRESHOLD} | 窗口: {WINDOW_SIZE}笔\n")
        await self.feed.connect()


if __name__ == "__main__":
    for d in ["data", "factors", "engine", "monitor"]:
        open(f"{d}/__init__.py", "a").close()

    bot = BTCFactorBot()
    asyncio.run(bot.run())
