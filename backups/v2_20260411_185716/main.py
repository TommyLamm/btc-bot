"""
BTC-Bot 主程序（優化版）
整合數據源、因子引擎、遺傳搜索和性能監控。
"""

import asyncio
import polars as pl
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.realtime_feed import BTCRealtimeFeed
from factors.factor_engine import compute_factors
from engine.genetic_engine import GeneticEngine, FACTOR_COLS
from monitor.performance_monitor import PerformanceMonitor


class BTCBot:
    def __init__(self):
        # 數據源
        self.feed = BTCRealtimeFeed(buffer_size=25000)
        self.feed.on_candle_close(self.on_candle_close)

        # 載入歷史數據
        print("載入歷史數據...")
        self.df_history = pl.read_parquet("data/btc_5m.parquet").sort("timestamp")
        print(f"  歷史K線: {len(self.df_history)} 根")

        # 載入持倉量數據（★ 修復：只讀取一次）
        try:
            self.oi_df = pl.read_parquet("data/btc_oi.parquet").sort("timestamp")
            print(f"  持倉量數據: {len(self.oi_df)} 條")
        except Exception:
            self.oi_df = None
            print("  無持倉量數據")

        # 遺傳引擎
        self.engine = GeneticEngine(pop_size=300, n_generations=60)

        # 性能監控
        self.monitor = PerformanceMonitor(window_size=200, retrain_threshold=0.50)

        # 狀態（★ 修復：先初始化再搜索）
        self.position = None
        self.entry_price = 0.0
        self.search_count = 0

        # 初始搜索
        print("\n初始因子搜索...")
        self._run_search()

    def _run_search(self):
        self.search_count += 1
        print(f"\n{'='*60}")
        print(f"第 {self.search_count} 次因子搜索")
        print(f"{'='*60}")

        df = compute_factors(self.df_history.tail(5000), oi_df=self.oi_df)
        df = df.drop_nulls().fill_nan(0)

        if len(df) < 100:
            print("數據不足，跳過搜索")
            return

        result = self.engine.search(df)
        if result:
            self.monitor.set_factor(result)
            print(f"搜索完成")
        else:
            print("搜索失敗")

    async def on_candle_close(self, candle: dict, buffer: list):
        # 更新歷史數據
        new_row = pl.DataFrame([{
            "timestamp": candle["timestamp"],
            "open":      candle["open"],
            "high":      candle["high"],
            "low":       candle["low"],
            "close":     candle["close"],
            "volume":    candle["volume"],
        }])

        # 確保 schema 一致
        for col in self.df_history.columns:
            if col not in new_row.columns:
                new_row = new_row.with_columns(pl.lit(None).alias(col))
        new_row = new_row.select(self.df_history.columns)

        self.df_history = pl.concat([self.df_history, new_row])

        # 計算因子
        self.df_history = compute_factors(
            self.df_history.tail(5000), oi_df=self.oi_df
        )

        # 獲取當前因子值
        if len(self.df_history) == 0:
            return

        last_row = self.df_history.tail(1)
        factor_values = {}
        for col in FACTOR_COLS:
            if col in last_row.columns:
                val = last_row[col].to_list()[0]
                factor_values[col] = float(val) if val is not None else 0.0
            else:
                factor_values[col] = 0.0

        if "obi" in candle:
            factor_values["obi"] = candle["obi"]

        # 生成信號
        signal = self.monitor.generate_signal(factor_values)
        threshold = 0.5
        if self.monitor.current_factor:
            threshold = self.monitor.current_factor.get("threshold", 0.5)

        action = None
        if signal > threshold:
            action = "long"
        elif signal < -threshold:
            action = "short"

        # 處理持倉
        if self.position is not None:
            if (self.position == "long" and action != "long") or \
               (self.position == "short" and action != "short"):
                pnl = self._calculate_pnl(candle["close"])
                self.monitor.record_trade(pnl)
                print(f"  平倉 {self.position} @ {candle['close']:.1f} PnL={pnl:+.4f}")
                self.position = None
                self.entry_price = 0.0

        if action and self.position is None:
            self.position = action
            self.entry_price = candle["close"]
            print(f"  開倉 {action} @ {candle['close']:.1f} 信號={signal:+.3f}")

        if len(self.feed.buffer) % 50 == 0:
            self.monitor.print_status()

        if self.monitor.needs_retrain:
            print("\n觸發自動重搜...")
            self._run_search()

    def _calculate_pnl(self, exit_price: float) -> float:
        if self.position == "long":
            return (exit_price - self.entry_price) / self.entry_price
        elif self.position == "short":
            return (self.entry_price - exit_price) / self.entry_price
        return 0.0

    async def run(self):
        print("\n" + "=" * 60)
        print("BTC-Bot 啟動")
        print("=" * 60)
        await self.feed.connect()


if __name__ == "__main__":
    bot = BTCBot()
    asyncio.run(bot.run())
