"""
BTC-Bot 主程序（自適應進化版 v2）
整合：數據源、因子引擎、GP 因子挖掘、市場狀態分類、遺傳搜索、性能監控。
"""

import asyncio
import polars as pl
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.realtime_feed import BTCRealtimeFeed
from factors.factor_engine import compute_factors
from factors.regime_detector import RegimeDetector
from engine.genetic_engine import GeneticEngine, FACTOR_COLS
from engine.gp_factor_miner import GPFactorMiner, BASE_FACTORS
from monitor.performance_monitor import PerformanceMonitor


class BTCBot:
    def __init__(self):
        self.feed = BTCRealtimeFeed(buffer_size=25000)
        self.feed.on_candle_close(self.on_candle_close)

        print("載入歷史數據...")
        self.df_history = pl.read_parquet("data/btc_5m.parquet").sort("timestamp")
        print(f"  歷史K線: {len(self.df_history)} 根")

        try:
            self.oi_df = pl.read_parquet("data/btc_oi.parquet").sort("timestamp")
            print(f"  持倉量數據: {len(self.oi_df)} 條")
        except Exception:
            self.oi_df = None
            print("  無持倉量數據")

        # 核心模組
        self.engine = GeneticEngine(pop_size=300, n_generations=60)
        self.gp_miner = GPFactorMiner(
            pop_size=200, n_generations=40, max_depth=4,
            ic_threshold=0.02, max_new_factors=5,
            save_path="data/gp_factors.json",
        )
        self.regime_detector = RegimeDetector()
        self.monitor = PerformanceMonitor(window_size=200, retrain_threshold=0.50)

        # 狀態
        self.position = None
        self.entry_price = 0.0
        self.search_count = 0
        self.gp_mine_count = 0
        self.current_regime = "unknown"
        self.active_factor_cols = list(FACTOR_COLS)

        # 初始搜索
        print("\n初始因子搜索...")
        self._run_adaptive_search()

    def _prepare_factor_data(self, df):
        df = compute_factors(df, oi_df=self.oi_df)
        df = df.drop_nulls().fill_nan(0)
        if len(df) < 100:
            return df, {}
        factor_data = {}
        for col in BASE_FACTORS:
            if col in df.columns:
                factor_data[col] = df[col].to_numpy().astype(np.float64)
            else:
                factor_data[col] = np.zeros(len(df))
        gp_data = self.gp_miner.compute_gp_factors(factor_data)
        for name, values in gp_data.items():
            if len(values) == len(df):
                df = df.with_columns(pl.Series(name=name, values=values))
        return df, factor_data

    def _run_gp_mining(self, factor_data, returns):
        self.gp_mine_count += 1
        print(f"\n  GP 因子挖掘（第 {self.gp_mine_count} 次）...")
        new_factors = self.gp_miner.mine(factor_data, returns)
        if new_factors:
            gp_names = self.gp_miner.get_top_factors(n=5)
            self.active_factor_cols = list(FACTOR_COLS) + gp_names
            print(f"  活躍因子數: {len(self.active_factor_cols)}")

    def _detect_regime(self, df):
        if len(df) < 100:
            return {"regime": "unknown", "factor_weights": {}}
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        close = df["close"].to_numpy().astype(np.float64)
        result = self.regime_detector.detect(high, low, close)
        self.current_regime = result["regime"]
        return result

    def _run_adaptive_search(self):
        self.search_count += 1
        print(f"\n{'='*60}")
        print(f"第 {self.search_count} 次自適應搜索")
        print(f"{'='*60}")

        df, factor_data = self._prepare_factor_data(self.df_history.tail(5000))
        if len(df) < 100:
            print("數據不足，跳過搜索")
            return

        regime_info = self._detect_regime(df)
        regime = regime_info.get("regime", "unknown")
        confidence = regime_info.get("confidence", 0)
        adx = regime_info.get("adx", 0)
        print(f"  市場狀態: {regime} (置信度={confidence:.2f}, ADX={adx:.1f})")

        if factor_data and (self.search_count % 3 == 1 or self.search_count == 1):
            close = df["close"].to_numpy().astype(np.float64)
            returns = np.zeros(len(close))
            returns[:-1] = (close[1:] - close[:-1]) / close[:-1]
            self._run_gp_mining(factor_data, returns)
            df, factor_data = self._prepare_factor_data(self.df_history.tail(5000))

        self._update_engine_factors()

        print(f"\n  遺傳搜索（{len(self.active_factor_cols)} 個因子）...")
        result = self.engine.search(df)

        if result:
            regime_weights = regime_info.get("factor_weights", {})
            if regime_weights and result.get("weights") is not None:
                adjusted_weights = result["weights"].copy()
                for i, col in enumerate(self.engine.factor_cols_used):
                    if col in regime_weights:
                        adjusted_weights[i] *= regime_weights[col]
                norm = np.linalg.norm(adjusted_weights)
                if norm > 1e-10:
                    adjusted_weights = adjusted_weights / norm
                result["weights"] = adjusted_weights
                result["regime"] = regime
            self.monitor.set_factor(result)
            print(f"搜索完成 (市場狀態={regime})")
        else:
            print("搜索失敗")

    def _update_engine_factors(self):
        import engine.genetic_engine as ge
        ge.FACTOR_COLS = list(self.active_factor_cols)
        self.engine.n_factors = len(self.active_factor_cols)

    async def on_candle_close(self, candle, buffer):
        new_row = pl.DataFrame([{
            "timestamp": candle["timestamp"],
            "open": candle["open"], "high": candle["high"],
            "low": candle["low"], "close": candle["close"],
            "volume": candle["volume"],
        }])
        for col in self.df_history.columns:
            if col not in new_row.columns:
                new_row = new_row.with_columns(pl.lit(None).alias(col))
        new_row = new_row.select(self.df_history.columns)
        self.df_history = pl.concat([self.df_history, new_row])

        df_recent = compute_factors(self.df_history.tail(5000), oi_df=self.oi_df)
        if len(df_recent) == 0:
            return

        last_row = df_recent.tail(1)
        factor_values = {}
        for col in FACTOR_COLS:
            if col in last_row.columns:
                val = last_row[col].to_list()[0]
                factor_values[col] = float(val) if val is not None else 0.0
            else:
                factor_values[col] = 0.0

        factor_data = {}
        for col in BASE_FACTORS:
            if col in df_recent.columns:
                factor_data[col] = df_recent[col].to_numpy().astype(np.float64)
            else:
                factor_data[col] = np.zeros(len(df_recent))
        gp_values = self.gp_miner.compute_gp_factors(factor_data)
        for name, values in gp_values.items():
            if len(values) > 0:
                factor_values[name] = float(values[-1])

        if "obi" in candle:
            factor_values["obi"] = candle["obi"]

        signal = self.monitor.generate_signal(factor_values)
        threshold = 0.5
        if self.monitor.current_factor:
            threshold = self.monitor.current_factor.get("threshold", 0.5)

        action = None
        if signal > threshold:
            action = "long"
        elif signal < -threshold:
            action = "short"

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
            direction = "做多" if action == "long" else "做空"
            print(f"  開倉 {direction} @ {candle['close']:.1f} 信號={signal:+.3f} [{self.current_regime}]")
            self.entry_price = candle["close"]

        if len(self.feed.buffer) % 50 == 0:
            self.monitor.print_status()
            if len(df_recent) >= 100:
                high = df_recent["high"].to_numpy().astype(np.float64)
                low = df_recent["low"].to_numpy().astype(np.float64)
                close = df_recent["close"].to_numpy().astype(np.float64)
                self.regime_detector.detect(high, low, close)
                stats = self.regime_detector.get_regime_stats()
                print(
                    f"  市場: {stats['current']} "
                    f"(趨勢{stats.get('trending_pct', 0):.0f}% "
                    f"震盪{stats.get('ranging_pct', 0):.0f}% "
                    f"高波動{stats.get('volatile_pct', 0):.0f}%)"
                )

        if self.monitor.needs_retrain:
            print("\n觸發自適應重搜...")
            self._run_adaptive_search()

    def _calculate_pnl(self, exit_price):
        if self.position == "long":
            return (exit_price - self.entry_price) / self.entry_price
        elif self.position == "short":
            return (self.entry_price - exit_price) / self.entry_price
        return 0.0

    async def run(self):
        print("\n" + "=" * 60)
        print("BTC-Bot 啟動（自適應進化版 v2）")
        print(f"  基礎因子: {len(FACTOR_COLS)} 個")
        print(f"  GP 因子: {len(self.gp_miner.discovered_factors)} 個")
        print(f"  活躍因子: {len(self.active_factor_cols)} 個")
        print(f"  市場狀態: {self.current_regime}")
        print("=" * 60)
        await self.feed.connect()


if __name__ == "__main__":
    bot = BTCBot()
    asyncio.run(bot.run())
