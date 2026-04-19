"""
BTC-Bot 主程序 v4.1 — 高頻動態倉位版
核心改進：
  1. 高頻交易：目標每天 30~50 次（TARGET_TRADE_PCT=0.12）
  2. 動態倉位：交易量少時加大倉位，交易量多時縮小倉位
  3. 目標每筆盈利 $2：根據 ATR 反算所需倉位
  4. 短預熱：10 根 K 線（50 分鐘）
  5. ATR 止損/止盈 + 手續費感知
  6. 帳戶 $400，3x 槓桿
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

# ═══ Hyperliquid 交易配置 ═══
HL_ENABLED = os.environ.get("HL_ENABLED", "true").lower() == "true"
HL_SECRET_KEY = os.environ.get(
    "HL_SECRET_KEY",
    "0xf3ddcc9a128d6251033058f671450a929b32d47389886c664f4497fe1c100e99",
)
HL_ACCOUNT_ADDRESS = os.environ.get(
    "HL_ACCOUNT_ADDRESS",
    "0x73531FF0bAf770EaFDc25003fAD6E33F7dB682EB",
)
HL_LEVERAGE = int(os.environ.get("HL_LEVERAGE", "3"))
HL_ACCOUNT_SIZE = float(os.environ.get("HL_ACCOUNT_SIZE", "400.0"))

# ═══ 交易頻率配置 ═══
TARGET_TRADE_PCT = float(os.environ.get("TARGET_TRADE_PCT", "0.12"))
DYNAMIC_THRESHOLD_WARMUP = int(os.environ.get("DYNAMIC_THRESHOLD_WARMUP", "10"))
SIGNAL_DEBUG = os.environ.get("SIGNAL_DEBUG", "true").lower() == "true"

# ═══ 動態倉位配置 ═══
TARGET_PROFIT_PER_TRADE = float(os.environ.get("TARGET_PROFIT", "2.0"))
BASE_POSITION_USD = float(os.environ.get("BASE_POSITION_USD", "200.0"))
MIN_POSITION_USD = float(os.environ.get("MIN_POSITION_USD", "50.0"))
MAX_POSITION_USD = float(os.environ.get("MAX_POSITION_USD", "350.0"))

# ═══ 風控參數 ═══
ATR_PERIOD = 14
STOP_LOSS_ATR_MULT = 1.5
TAKE_PROFIT_ATR_MULT = 2.0
MAX_HOLD_BARS = 20
COOLDOWN_BARS = 1
ROUND_TRIP_FEE = 0.0007

# ═══ 動態倉位：交易頻率調節 ═══
TRADE_COUNT_WINDOW = 288
HIGH_FREQ_THRESHOLD = 40
LOW_FREQ_THRESHOLD = 15


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

        self.engine = GeneticEngine(pop_size=300, n_generations=60)
        self.gp_miner = GPFactorMiner(
            pop_size=200, n_generations=40, max_depth=4,
            ic_threshold=0.02, max_new_factors=5,
            save_path="data/gp_factors.json",
        )
        self.regime_detector = RegimeDetector()
        self.monitor = PerformanceMonitor(window_size=200, retrain_threshold=0.42)

        self.position = None
        self.entry_price = 0.0
        self.hold_bars = 0
        self.cooldown_remaining = 0
        self.current_atr = 0.0
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0
        self.current_position_usd = BASE_POSITION_USD

        self.search_count = 0
        self.gp_mine_count = 0
        self.current_regime = "unknown"
        self.active_factor_cols = list(FACTOR_COLS)
        self.candle_count = 0
        self.no_trade_count = 0

        self.recent_trades = []
        self.recent_pnls = []

        self.hl_executor = None
        if HL_ENABLED:
            try:
                from executor.hyperliquid_executor import HyperliquidExecutor
                self.hl_executor = HyperliquidExecutor(
                    secret_key=HL_SECRET_KEY,
                    account_address=HL_ACCOUNT_ADDRESS,
                    leverage=HL_LEVERAGE,
                    default_position_size_usd=BASE_POSITION_USD,
                    coin="BTC",
                    log_path="logs/trades.jsonl",
                )
                print(f"\n  [HL] 交易模組已啟用")
                print(f"  {self.hl_executor.status_summary()}")
            except Exception as e:
                print(f"\n  [HL] 交易模組啟動失敗: {e}")
                self.hl_executor = None

        print("\n初始因子搜索...")
        self._run_adaptive_search()

    def _calc_dynamic_position_usd(self) -> float:
        atr = max(self.current_atr, 20.0)
        tp_distance = TAKE_PROFIT_ATR_MULT * atr
        btc_needed = TARGET_PROFIT_PER_TRADE / tp_distance
        notional_needed = btc_needed * 71000
        atr_based_usd = notional_needed / HL_LEVERAGE

        freq_mult = 1.0
        cutoff = self.candle_count - TRADE_COUNT_WINDOW
        recent_count = sum(1 for t in self.recent_trades if t > cutoff)

        if recent_count < LOW_FREQ_THRESHOLD:
            freq_mult = min(1.5, 1.0 + (LOW_FREQ_THRESHOLD - recent_count) / LOW_FREQ_THRESHOLD * 0.5)
        elif recent_count > HIGH_FREQ_THRESHOLD:
            freq_mult = max(0.5, 1.0 - (recent_count - HIGH_FREQ_THRESHOLD) / HIGH_FREQ_THRESHOLD * 0.5)

        win_mult = 1.0
        if len(self.recent_pnls) >= 10:
            recent_wins = sum(1 for p in self.recent_pnls[-20:] if p > 0)
            recent_total = len(self.recent_pnls[-20:])
            win_rate = recent_wins / recent_total
            if win_rate > 0.6:
                win_mult = 1.2
            elif win_rate < 0.35:
                win_mult = 0.6

        target_usd = atr_based_usd * freq_mult * win_mult
        target_usd = max(MIN_POSITION_USD, min(MAX_POSITION_USD, target_usd))

        return round(target_usd, 2)

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

        search_cols = list(FACTOR_COLS)
        print(f"\n  遺傳搜索（{len(search_cols)} 個因子，含手續費扣除）...")
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

            self.monitor.set_factor(result, factor_cols=search_cols)

            threshold = result.get("threshold", 0.5)
            print(f"  搜索閾值: {threshold:.4f}（僅供參考）")
            print(f"  預熱: 前 {DYNAMIC_THRESHOLD_WARMUP} 根K線收集信號（約 {DYNAMIC_THRESHOLD_WARMUP*5} 分鐘）")
        else:
            print("搜索失敗")

        self.no_trade_count = 0

    def _update_engine_factors(self):
        import engine.genetic_engine as ge
        ge.FACTOR_COLS = list(self.active_factor_cols)
        self.engine.n_factors = len(self.active_factor_cols)

    def _compute_atr(self, df_recent):
        if len(df_recent) < ATR_PERIOD + 1:
            if len(df_recent) > 0:
                hl = df_recent["high"].to_numpy() - df_recent["low"].to_numpy()
                return float(np.mean(hl[-min(len(hl), 14):]))
            return 100.0

        high = df_recent["high"].to_numpy().astype(np.float64)
        low = df_recent["low"].to_numpy().astype(np.float64)
        close = df_recent["close"].to_numpy().astype(np.float64)

        n = len(close)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        atr = np.mean(tr[:ATR_PERIOD])
        for i in range(ATR_PERIOD, n):
            atr = (atr * (ATR_PERIOD - 1) + tr[i]) / ATR_PERIOD

        return float(atr)

    def _hl_open(self, direction: str, price: float, signal: float, position_usd: float):
        if not self.hl_executor:
            return None
        reason = (f"sig={signal:+.3f} regime={self.current_regime} "
                  f"atr={self.current_atr:.1f} usd={position_usd:.0f}")
        try:
            if direction == "long":
                result = self.hl_executor.open_long(reason, position_size_usd=position_usd)
            else:
                result = self.hl_executor.open_short(reason, position_size_usd=position_usd)
            if result["status"] == "ok":
                print(f"  [HL] 實盤開倉: {result['direction']} {result['size']} BTC "
                      f"@ ${result['price']:,.1f} (保證金=${position_usd:.0f})")
            elif result["status"] != "skip":
                print(f"  [HL] 開倉失敗: {result.get('msg', '')}")
            return result
        except Exception as e:
            print(f"  [HL] 開倉異常: {e}")
            return {"status": "error", "msg": str(e)}

    def _hl_close(self, reason: str = ""):
        if not self.hl_executor:
            return
        try:
            result = self.hl_executor.close_position(reason)
            if result["status"] == "ok":
                pnl = result.get("pnl", 0)
                print(f"  [HL] 實盤平倉: {result['direction']} @ ${result['price']:,.1f} "
                      f"PnL=${pnl:+.2f}")
            elif result["status"] != "skip":
                print(f"  [HL] 平倉失敗: {result.get('msg', '')}")
        except Exception as e:
            print(f"  [HL] 平倉異常: {e}")

    async def on_candle_close(self, candle, buffer):
        self.candle_count += 1

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

        self.current_atr = self._compute_atr(df_recent)

        last_row = df_recent.tail(1)
        factor_values = {}
        search_cols = self.monitor.search_factor_cols or FACTOR_COLS
        for col in search_cols:
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

        n_realtime = len(self.monitor.realtime_signal_buffer)
        dyn_thr = self.monitor.get_dynamic_threshold(TARGET_TRADE_PCT, DYNAMIC_THRESHOLD_WARMUP)

        in_warmup = (dyn_thr is None)

        if dyn_thr is not None:
            threshold = dyn_thr
            thr_source = "dynamic"
        else:
            threshold = 999.0
            thr_source = "warmup"

        action = None
        if not in_warmup:
            if signal > threshold:
                action = "long"
            elif signal < -threshold:
                action = "short"

        self.current_position_usd = self._calc_dynamic_position_usd()

        if SIGNAL_DEBUG and self.candle_count % 3 == 0:
            action_str = action if action else ("warmup" if in_warmup else "hold")
            if thr_source == "dynamic":
                thr_info = f"dyn={threshold:.4f}"
            else:
                thr_info = f"warmup ({n_realtime}/{DYNAMIC_THRESHOLD_WARMUP})"
            pos_info = f" pos={self.position}" if self.position else ""
            atr_info = f" ATR={self.current_atr:.1f}"
            usd_info = f" ${self.current_position_usd:.0f}"
            print(f"  [信號] sig={signal:+.4f} {thr_info}{pos_info}{atr_info}{usd_info} → {action_str}")

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if action:
                action = None

        traded = False
        close_reason = None

        if self.position is not None:
            self.hold_bars += 1
            current_price = candle["close"]

            if self.position == "long" and current_price <= self.stop_loss_price:
                close_reason = f"止損 SL={self.stop_loss_price:.1f}"
            elif self.position == "short" and current_price >= self.stop_loss_price:
                close_reason = f"止損 SL={self.stop_loss_price:.1f}"
            elif self.position == "long" and current_price >= self.take_profit_price:
                close_reason = f"止盈 TP={self.take_profit_price:.1f}"
            elif self.position == "short" and current_price <= self.take_profit_price:
                close_reason = f"止盈 TP={self.take_profit_price:.1f}"
            elif self.hold_bars >= MAX_HOLD_BARS:
                close_reason = f"超時 {self.hold_bars}根K線"
            elif (self.position == "long" and action == "short") or \
                 (self.position == "short" and action == "long"):
                close_reason = f"信號反轉 sig={signal:+.3f}"

            if close_reason:
                pnl_raw = self._calculate_pnl(current_price)
                pnl_net = pnl_raw - ROUND_TRIP_FEE
                usd_pnl = pnl_raw * self.current_position_usd * HL_LEVERAGE
                self.monitor.record_trade(pnl_net)
                self.recent_pnls.append(usd_pnl)
                if len(self.recent_pnls) > 100:
                    self.recent_pnls = self.recent_pnls[-100:]

                print(f"  ✖ 平倉 {self.position} @ {current_price:.1f} "
                      f"PnL≈${usd_pnl:+.2f} 持倉{self.hold_bars}根 原因: {close_reason}")
                self._hl_close(close_reason)
                self.position = None
                self.entry_price = 0.0
                self.hold_bars = 0
                self.stop_loss_price = 0.0
                self.take_profit_price = 0.0
                self.cooldown_remaining = COOLDOWN_BARS
                traded = True

        if action and self.position is None and self.cooldown_remaining <= 0:
            position_usd = self.current_position_usd
            open_result = self._hl_open(action, candle["close"], signal, position_usd)
            opened = False

            if open_result and open_result.get("status") == "ok":
                opened = True
            elif not self.hl_executor:
                opened = True

            if opened:
                self.position = action
                self.entry_price = candle["close"]
                self.hold_bars = 0
                traded = True
                self.no_trade_count = 0
                self.recent_trades.append(self.candle_count)
                if len(self.recent_trades) > 500:
                    self.recent_trades = self.recent_trades[-500:]

                atr = max(self.current_atr, 20.0)
                if action == "long":
                    self.stop_loss_price = self.entry_price - STOP_LOSS_ATR_MULT * atr
                    self.take_profit_price = self.entry_price + TAKE_PROFIT_ATR_MULT * atr
                else:
                    self.stop_loss_price = self.entry_price + STOP_LOSS_ATR_MULT * atr
                    self.take_profit_price = self.entry_price - TAKE_PROFIT_ATR_MULT * atr

                direction = "做多" if action == "long" else "做空"
                print(f"  ★ 開倉 {direction} @ {candle['close']:.1f} "
                      f"信號={signal:+.3f} 閾值={threshold:.3f} "
                      f"SL={self.stop_loss_price:.1f} TP={self.take_profit_price:.1f} "
                      f"保證金=${position_usd:.0f} ATR={atr:.1f} [{self.current_regime}]")
            elif self.hl_executor:
                print(f"  [HL] 開倉未成功，跳過本次信號")

        if not traded:
            self.no_trade_count += 1

        if self.candle_count % 20 == 0:
            self.monitor.print_status()
            if self.hl_executor:
                print(f"  {self.hl_executor.status_summary()}")

            cutoff = self.candle_count - TRADE_COUNT_WINDOW
            recent_count = sum(1 for t in self.recent_trades if t > cutoff)
            avg_pnl = np.mean(self.recent_pnls[-20:]) if self.recent_pnls else 0
            print(f"  [倉位] 當前=${self.current_position_usd:.0f} "
                  f"近期交易={recent_count}筆/天 "
                  f"近期平均PnL=${avg_pnl:+.2f}")

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
            print("\n觸發自適應重搜（勝率過低）...")
            self._run_adaptive_search()

    def _calculate_pnl(self, exit_price):
        if self.position == "long":
            return (exit_price - self.entry_price) / self.entry_price
        elif self.position == "short":
            return (self.entry_price - exit_price) / self.entry_price
        return 0.0

    async def run(self):
        print("\n" + "=" * 60)
        print("BTC-Bot 啟動（v4.1 — 高頻動態倉位版）")
        print(f"  基礎因子: 12 個")
        print(f"  GP 因子: {len(self.gp_miner.discovered_factors)} 個")
        print(f"  活躍因子: {len(self.active_factor_cols)} 個")
        print(f"  市場狀態: {self.current_regime}")
        if self.hl_executor:
            print(f"  交易模式: Hyperliquid 實盤 ({HL_LEVERAGE}x 槓桿)")
            print(f"  帳戶規模: ${HL_ACCOUNT_SIZE}")
            print(f"  {self.hl_executor.status_summary()}")
        else:
            print(f"  交易模式: 純信號（未啟用實盤）")
        print(f"  ═══ 動態倉位 ═══")
        print(f"  基礎保證金: ${BASE_POSITION_USD} (範圍 ${MIN_POSITION_USD}~${MAX_POSITION_USD})")
        print(f"  目標每筆盈利: ${TARGET_PROFIT_PER_TRADE}")
        print(f"  交易頻率調節: <{LOW_FREQ_THRESHOLD}筆/天加大 | >{HIGH_FREQ_THRESHOLD}筆/天縮小")
        print(f"  ═══ 風控參數 ═══")
        print(f"  止損: {STOP_LOSS_ATR_MULT}×ATR | 止盈: {TAKE_PROFIT_ATR_MULT}×ATR")
        print(f"  最大持倉: {MAX_HOLD_BARS} 根K線 ({MAX_HOLD_BARS*5}分鐘)")
        print(f"  冷卻期: {COOLDOWN_BARS} 根K線")
        print(f"  手續費: {ROUND_TRIP_FEE*100:.3f}% (雙邊)")
        print(f"  ═══ 閾值策略 ═══")
        print(f"  預熱: 前 {DYNAMIC_THRESHOLD_WARMUP} 根K線（{DYNAMIC_THRESHOLD_WARMUP*5}分鐘）")
        print(f"  目標觸發率: {TARGET_TRADE_PCT*100:.0f}% → 每天 {int(288*TARGET_TRADE_PCT)}~{int(288*TARGET_TRADE_PCT*1.5)} 次")
        print(f"  信號 Debug: {'開啟' if SIGNAL_DEBUG else '關閉'} (每3根K線)")
        print("=" * 60)
        await self.feed.connect()


if __name__ == "__main__":
    bot = BTCBot()
    asyncio.run(bot.run())
