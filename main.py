"""
BTC-Bot v6.0 — 勝率極致優化版
改進：
  1. 40+ 因子（動量、波動率、成交量、微觀結構、多時間框架、時間）
  2. 分離多空閾值 + 預熱期用搜索閾值 80%
  3. 動態倉位（交易量少加大、多縮小）+ 目標每筆 $2
  4. 止損 SL×ATR / 止盈 TP×ATR + 自適應波動率調整
  5. Walk-forward 6 段驗證 + 非線性交互項 + 因子 IC 三重篩選
  6. 信號確認機制（連續同向 + 置信度過濾）
  7. GP 因子挖掘（勝率導向 + 條件勝率 + 信號持續性）
"""

import asyncio
import os
import sys
import numpy as np
import polars as pl
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.realtime_feed import BTCRealtimeFeed
from factors.factor_engine import compute_factors
from factors.regime_detector import RegimeDetector
from engine.gp_factor_miner import GPFactorMiner
from engine.genetic_engine import GeneticEngine, FACTOR_COLS, ROUND_TRIP_FEE
from monitor.performance_monitor import PerformanceMonitor

# ═══ 交易參數 ═══
HL_ENABLED = os.environ.get("HL_ENABLED", "true").lower() == "true"
HL_SECRET = os.environ.get("HL_SECRET", "")
HL_ACCOUNT = os.environ.get("HL_ACCOUNT", "")
HL_LEVERAGE = int(os.environ.get("HL_LEVERAGE", "3"))
HL_ACCOUNT_SIZE = float(os.environ.get("HL_ACCOUNT_SIZE", "400"))

# ═══ 動態倉位 ═══
TARGET_PROFIT_PER_TRADE = float(os.environ.get("TARGET_PROFIT", "2.0"))
BASE_POSITION_USD = float(os.environ.get("BASE_POSITION_USD", "200"))
MIN_POSITION_USD = float(os.environ.get("MIN_POSITION_USD", "50"))
MAX_POSITION_USD = float(os.environ.get("MAX_POSITION_USD", "350"))
LOW_FREQ_THRESHOLD = int(os.environ.get("LOW_FREQ_THRESHOLD", "15"))
HIGH_FREQ_THRESHOLD = int(os.environ.get("HIGH_FREQ_THRESHOLD", "40"))

# ═══ 風控參數 ═══
STOP_LOSS_ATR_MULT = float(os.environ.get("SL_ATR", "1.0"))
TAKE_PROFIT_ATR_MULT = float(os.environ.get("TP_ATR", "3.0"))
MAX_HOLD_BARS = int(os.environ.get("MAX_HOLD_BARS", "20"))
COOLDOWN_BARS = int(os.environ.get("COOLDOWN_BARS", "1"))

# ═══ 閾值參數 ═══
TARGET_TRADE_PCT = float(os.environ.get("TARGET_TRADE_PCT", "0.12"))
DYNAMIC_THRESHOLD_WARMUP = int(os.environ.get("DYNAMIC_THRESHOLD_WARMUP", "10"))
SIGNAL_CONFIRM_BARS = int(os.environ.get("SIGNAL_CONFIRM_BARS", "2"))
SIGNAL_CONFIDENCE_MULT = float(os.environ.get("SIGNAL_CONFIDENCE_MULT", "1.15"))
SIGNAL_DEBUG = os.environ.get("SIGNAL_DEBUG", "true").lower() == "true"
SIGNAL_DEBUG_INTERVAL = int(os.environ.get("SIGNAL_DEBUG_INTERVAL", "3"))

# ═══ 數據參數 ═══
DATA_PATH = "data/btc_5m.parquet"
OI_PATH = "data/btc_oi.parquet"
HISTORY_BARS = 4000


class BTCBot:
    def __init__(self):
        self.feed = BTCRealtimeFeed()
        self.feed.on_candle_close(self.on_candle_close)
        self.regime_detector = RegimeDetector()
        self.gp_miner = GPFactorMiner(
            pop_size=300, n_generations=50, max_depth=5,
            ic_threshold=0.02, max_new_factors=5,
            max_correlation=0.7,
        )
        self.genetic_engine = GeneticEngine(
            pop_size=300, n_generations=60,
            threshold_range=(0.05, 1.2),
            fee_rate=ROUND_TRIP_FEE,
            n_interactions=10,
            sl_atr=STOP_LOSS_ATR_MULT,
            tp_atr=TAKE_PROFIT_ATR_MULT,
            max_hold=MAX_HOLD_BARS,
            cooldown_bars=COOLDOWN_BARS,
            signal_confirm_bars=max(1, SIGNAL_CONFIRM_BARS),
            confidence_multiplier=max(1.0, SIGNAL_CONFIDENCE_MULT),
        )
        self.monitor = PerformanceMonitor(window_size=200, retrain_threshold=0.40)
        # v6.0：信號確認參數（也可被搜索結果覆蓋）
        self.monitor.signal_confirm_bars = max(1, SIGNAL_CONFIRM_BARS)
        self.monitor.confidence_multiplier = max(1.0, SIGNAL_CONFIDENCE_MULT)

        self.current_regime = "unknown"
        self.active_factor_cols = list(FACTOR_COLS)
        self.candle_count = 0
        self.position = "flat"
        self.entry_price = 0.0
        self.entry_bar = 0
        self.entry_atr = 0.0
        self.cooldown_remaining = 0
        self.last_close = 0.0
        self.prev_close = 0.0   # Bug 3 Fix：ATR 需要前一根收盤價
        self.current_atr = 0.0
        self.today_trades = 0
        self.today_start_bar = 0
        self.current_position_usd = BASE_POSITION_USD
        self._retrain_lock = False  # Bug 9 Fix：防止重搜重入

        # Hyperliquid
        self.hl_executor = None
        if HL_ENABLED and HL_SECRET and HL_ACCOUNT:
            try:
                from executor.hyperliquid_executor import HyperliquidExecutor
                self.hl_executor = HyperliquidExecutor(
                    secret_key=HL_SECRET,
                    account_address=HL_ACCOUNT,
                    leverage=HL_LEVERAGE,
                    default_position_size_usd=BASE_POSITION_USD,
                    coin="BTC",
                    log_path="logs/trades_v5.jsonl",
                )
                print(f"  Hyperliquid 已連接: {HL_LEVERAGE}x 槓桿")
            except Exception as e:
                print(f"  Hyperliquid 初始化失敗: {e}")

        # Bug 1 Fix：初始搜索移至 run() 中用 asyncio.to_thread 執行，不在 __init__ 阻塞

    def _run_adaptive_search(self):
        """Bug 14 Fix：純計算，不修改任何共享狀態。返回 (result, regime, all_cols) 元組。"""
        if not os.path.exists(DATA_PATH):
            print("  歷史數據不存在，跳過搜索")
            return None

        df = pl.read_parquet(DATA_PATH).tail(HISTORY_BARS)
        oi_df = None
        if os.path.exists(OI_PATH):
            oi_df = pl.read_parquet(OI_PATH)

        df = compute_factors(df, oi_df)

        # 市場狀態
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64) if "high" in df.columns else close
        low = df["low"].to_numpy().astype(np.float64) if "low" in df.columns else close
        regime_result = self.regime_detector.detect(high, low, close)
        regime = regime_result["regime"]

        # GP 因子挖掘
        ret = np.zeros(len(close))
        ret[:-1] = (close[1:] - close[:-1]) / close[:-1]
        factor_data = {}
        for col in df.columns:
            try:
                factor_data[col] = df[col].to_numpy().astype(np.float64)
            except Exception:
                pass
        gp_factors = self.gp_miner.mine(factor_data, ret, close_arr=close)

        # 將 GP 因子加入 DataFrame
        gp_data = self.gp_miner.compute_gp_factors(factor_data)
        for name, values in gp_data.items():
            if len(values) == len(df):
                df = df.with_columns(pl.Series(name=name, values=values))

        gp_names = self.gp_miner.get_factor_names()

        from engine.genetic_engine import FACTOR_COLS as BASE_COLS
        all_cols = list(BASE_COLS) + [g for g in gp_names if g in df.columns]
        self.genetic_engine.factor_cols_used = all_cols
        # S5 Fix：不再修改全域 FACTOR_COLS，search() 已使用 self.factor_cols_used
        print(f"\n  遺傳搜索（{len(all_cols)} 個因子）...")
        result = self.genetic_engine.search(df)

        if result is not None:
            result["regime"] = regime
            result["signal_confirm_bars"] = self.monitor.signal_confirm_bars
            result["confidence_multiplier"] = self.monitor.confidence_multiplier
            if not result.get("screened_factor_cols"):
                result["screened_factor_cols"] = list(all_cols)
            # 信號分佈參考（僅打印，不修改狀態）
            signal_cols = result.get("screened_factor_cols", all_cols)
            signal_cols = [c for c in signal_cols if c in df.columns]
            if not signal_cols:
                signal_cols = [c for c in all_cols if c in df.columns]
            X_raw = df.select(signal_cols).to_numpy().astype(np.float64)
            X_raw = np.nan_to_num(X_raw, nan=0.0)
            X_mean = result["factor_mean"]
            X_std = result["factor_std"]
            n_base = min(X_raw.shape[1], len(X_mean))
            X = (X_raw[:, :n_base] - X_mean[:n_base]) / (X_std[:n_base] + 1e-10)
            signals = X @ result["weights"][:n_base]
            abs_sig = np.abs(signals)
            pos_sig = signals[signals > 0]
            neg_sig = np.abs(signals[signals < 0])
            print(f"  信號分佈: P50={np.percentile(abs_sig,50):.3f} "
                  f"P90={np.percentile(abs_sig,90):.3f} Max={abs_sig.max():.3f}")
            if len(pos_sig) > 0:
                print(f"  多頭信號: P50={np.percentile(pos_sig,50):.3f} P90={np.percentile(pos_sig,90):.3f}")
            if len(neg_sig) > 0:
                print(f"  空頭信號: P50={np.percentile(neg_sig,50):.3f} P90={np.percentile(neg_sig,90):.3f}")

        print(f"  搜索完成（市場狀態={regime}）")
        # 返回結果元組，由主線程 _apply_search_result 更新共享狀態
        return (result, regime, all_cols) if result is not None else None

    def _apply_search_result(self, search_output):
        """Bug 14 Fix：在主線程中原子性地更新共享狀態。"""
        result, regime, all_cols = search_output
        self.current_regime = regime
        effective_cols = result.get("screened_factor_cols") or all_cols
        self.active_factor_cols = list(effective_cols)
        self.monitor.set_factor(
            result,
            factor_cols=effective_cols,
            interaction_pairs=result.get("interaction_pairs", []),
        )

    def _compute_atr(self, high, low, close, period=14):
        """計算 ATR（Bug 3 Fix：TR 使用前一根收盤價）"""
        prev_close = self.prev_close if self.prev_close > 0 else close
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        if self.current_atr == 0:
            self.current_atr = tr
        else:
            self.current_atr = (self.current_atr * (period - 1) + tr) / period
        return self.current_atr

    def _calculate_position_size(self):
        """動態倉位計算"""
        # 基礎倉位：根據 ATR 和目標盈利反算
        if self.current_atr > 0 and self.last_close > 0:
            tp_price_move = self.current_atr * TAKE_PROFIT_ATR_MULT
            btc_needed = TARGET_PROFIT_PER_TRADE / tp_price_move
            notional_needed = btc_needed * self.last_close
            base_margin = notional_needed / HL_LEVERAGE
        else:
            base_margin = BASE_POSITION_USD

        # 交易頻率調節
        bars_today = self.candle_count - self.today_start_bar
        if bars_today > 0:
            trades_per_day_est = self.today_trades * (288 / bars_today)
        else:
            trades_per_day_est = 0

        freq_multiplier = 1.0
        if trades_per_day_est < LOW_FREQ_THRESHOLD and bars_today > 20:
            freq_multiplier = 1.3
        elif trades_per_day_est > HIGH_FREQ_THRESHOLD:
            freq_multiplier = 0.6

        # 勝率調節
        stats = self.monitor.get_stats()
        wr = stats.get("win_rate", 0.5)
        wr_multiplier = 1.0
        if stats["n_trades"] >= 20:
            if wr > 0.58:
                wr_multiplier = 1.2
            elif wr < 0.38:
                wr_multiplier = 0.5

        position_usd = base_margin * freq_multiplier * wr_multiplier
        position_usd = max(MIN_POSITION_USD, min(MAX_POSITION_USD, position_usd))
        self.current_position_usd = round(position_usd, 2)
        return self.current_position_usd

    def _hl_open(self, direction, reason=""):
        """Hyperliquid 開倉"""
        if not self.hl_executor:
            return None
        try:
            position_usd = self._calculate_position_size()
            self.hl_executor.default_position_size_usd = position_usd
            if direction == "long":
                result = self.hl_executor.open_long(reason)
            else:
                result = self.hl_executor.open_short(reason)
            if result and result.get("status") == "ok":
                return result
            else:
                err = result.get("error", "unknown") if result else "no result"
                print(f"  [HL] 開倉失敗: {err}")
                return None
        except Exception as e:
            print(f"  [HL] 開倉異常: {e}")
            return None

    def _hl_close(self, reason=""):
        """Hyperliquid 平倉"""
        if not self.hl_executor:
            return None
        try:
            result = self.hl_executor.close_position(reason)
            if result and result.get("status") == "ok":
                return result
            elif result and "no position" in str(result.get("error", "")).lower():
                # Bug 15 Fix：HL 確認無倉位，返回 skip 而非透傳 error status
                print(f"  [HL] HL 上已無倉位，同步本地狀態")
                return {"status": "skip", "msg": "HL 上已無倉位"}
            else:
                err = result.get("error", "unknown") if result else "no result"
                print(f"  [HL] 平倉失敗: {err}")
                return None
        except Exception as e:
            print(f"  [HL] 平倉異常: {e}")
            return None

    async def on_candle_close(self, candle, buffer_list=None):
        self.candle_count += 1
        close = candle["close"]
        high = candle.get("high", close)
        low = candle.get("low", close)
        obi = candle.get("obi", 0.0)

        # 計算 ATR（必須在更新 last_close/prev_close 之前）
        atr = self._compute_atr(high, low, close)

        # Bug 3 Fix：更新 prev_close 供下一根 K 線的 ATR 使用
        self.prev_close = self.last_close if self.last_close > 0 else close
        self.last_close = close

        # 每日重置
        if self.candle_count % 288 == 1:
            self.today_trades = 0
            self.today_start_bar = self.candle_count

        # 冷卻期
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            # S1 Fix：冷卻期內重置確認計數器，防止非連續K線被當作連續確認
            self.monitor._confirm_long_count = 0
            self.monitor._confirm_short_count = 0

        # ═══ 計算因子 ═══
        factor_values = self._compute_realtime_factors(candle)

        # S3 Fix：因子計算異常返回空字典時，重置確認計數器，跳過交易邏輯
        if not factor_values:
            self.monitor._confirm_long_count = 0
            self.monitor._confirm_short_count = 0
            return

        # ═══ 生成信號 ═══
        signal = self.monitor.generate_signal(factor_values)

        # ═══ 取得閾值 ═══
        long_thr, short_thr, is_warmup = self.monitor.get_thresholds(
            target_trade_pct=TARGET_TRADE_PCT,
            warmup=DYNAMIC_THRESHOLD_WARMUP,
        )

        # ═══ 信號 Debug ═══
        if SIGNAL_DEBUG and self.candle_count % SIGNAL_DEBUG_INTERVAL == 0:
            warmup_str = f"預熱({len(self.monitor.realtime_signal_buffer)}/{DYNAMIC_THRESHOLD_WARMUP})" if is_warmup else "動態"
            if signal > long_thr:
                action_str = "→ long"
            elif signal < -short_thr:
                action_str = "→ short"
            else:
                action_str = "→ hold"
            print(f"  [信號] sig={signal:+.4f} L閾={long_thr:.4f} S閾={short_thr:.4f} "
                  f"[{warmup_str}] {action_str}")

        # ═══ 持倉管理 ═══
        if self.position != "flat":
            bars_held = self.candle_count - self.entry_bar
            pnl_pct = self._calculate_pnl(close)

            # 止損（自適應：高波動時放寬）
            vol_adj = max(0.8, min(1.5, atr / (self.entry_atr + 1e-10)))
            sl_pct = (self.entry_atr * STOP_LOSS_ATR_MULT * vol_adj) / self.entry_price
            tp_pct = (self.entry_atr * TAKE_PROFIT_ATR_MULT) / self.entry_price

            close_reason = None
            if pnl_pct <= -sl_pct:
                close_reason = f"止損 {pnl_pct:+.4f} (SL={sl_pct:.4f})"
            elif pnl_pct >= tp_pct:
                close_reason = f"止盈 {pnl_pct:+.4f} (TP={tp_pct:.4f})"
            elif bars_held >= MAX_HOLD_BARS:
                close_reason = f"超時 {bars_held}根 PnL={pnl_pct:+.4f}"
            elif self.position in ("long", "short"):
                pos_dir = 1 if self.position == "long" else -1
                if self.monitor.should_close_position(signal, pos_dir, long_thr, short_thr):
                    close_reason = f"信號反轉 sig={signal:+.3f}"

            if close_reason:
                net_pnl = pnl_pct - ROUND_TRIP_FEE
                print(f"  平倉 {self.position} @ {close:.1f} {close_reason}")
                hl_result = self._hl_close(close_reason)
                # Bug 5 Fix：僅在 HL 真正成功平倉（或無 HL）時才更新本地狀態
                hl_ok = (self.hl_executor is None) or (
                    hl_result is not None and hl_result.get("status") in ("ok", "skip")
                )
                if hl_ok:
                    self.monitor.record_trade(net_pnl)
                    self.position = "flat"
                    self.cooldown_remaining = COOLDOWN_BARS
                    self.today_trades += 1
                    # S1 Fix：平倉時重置確認計數器
                    self.monitor._confirm_long_count = 0
                    self.monitor._confirm_short_count = 0
                else:
                    print(f"  [警告] HL 平倉失敗，保持本地持倉狀態，下根K線重試")

        # ═══ 開倉邏輯 ═══
        if self.position == "flat" and self.cooldown_remaining <= 0:
            direction, confirmed = self.monitor.should_open_position(signal, long_thr, short_thr)
            if confirmed and direction in (1, -1):
                open_dir = "long" if direction == 1 else "short"
                open_thr = long_thr if direction == 1 else short_thr
                conf_thr = open_thr * self.monitor.confidence_multiplier
                pos_usd = self._calculate_position_size()
                dir_cn = "多" if open_dir == "long" else "空"
                print(f"  ★ 開倉 做{dir_cn} @ {close:.1f} 信號={signal:+.3f} "
                      f"閾值={open_thr:.3f} 確認閾值={conf_thr:.3f} "
                      f"({self.monitor.signal_confirm_bars}根確認) 倉位=${pos_usd:.0f} [{self.current_regime}]")
                hl_result = self._hl_open(open_dir, f"sig={signal:+.3f} thr={open_thr:.3f}")
                # Bug 16 Fix：只在 HL 成功或無 HL 時才設定本地持倉
                hl_ok = (self.hl_executor is None) or (hl_result is not None)
                if hl_ok:
                    self.position = open_dir
                    self.entry_price = close
                    self.entry_bar = self.candle_count
                    self.entry_atr = atr
                else:
                    print(f"  [警告] HL 開倉失敗，不設定本地持倉")

        # ═══ 定期狀態報告 ═══
        if self.candle_count % 20 == 0:
            self.monitor.print_status()
            if self.hl_executor:
                print(f"  {self.hl_executor.status_summary()}")
            stats = self.regime_detector.get_regime_stats()
            if stats:
                print(
                    f"  市場: {stats['current']} "
                    f"(趨勢{stats.get('trending_pct', 0):.0f}% "
                    f"震盪{stats.get('ranging_pct', 0):.0f}% "
                    f"高波動{stats.get('volatile_pct', 0):.0f}%)"
                )

        # ═══ 自適應重搜 ═══
        # Bug 2 & 9 Fix：用 asyncio.to_thread 避免阻塞 event loop；加重入鎖防無限重搜
        if self.monitor.needs_retrain and not self._retrain_lock:
            self._retrain_lock = True
            print("\n觸發自適應重搜（背景執行）...")
            asyncio.create_task(self._async_retrain())

    async def _async_retrain(self):
        """Bug 2+14 Fix：背景搜索 + 主線程原子更新。"""
        try:
            search_output = await asyncio.to_thread(self._run_adaptive_search)
            if search_output is None:
                print("  重搜未找到有效因子，5 分鐘後重試")
                await asyncio.sleep(300)
            else:
                # Bug 14 Fix：回到主線程後再更新共享狀態（asyncio 是單線程，此處安全）
                self._apply_search_result(search_output)
            self.monitor.needs_retrain = False
        except Exception as e:
            print(f"  重搜異常: {e}")
            self.monitor.needs_retrain = False
        finally:
            self._retrain_lock = False

    def _compute_realtime_factors(self, candle):
        """從 K 線緩衝重算最新一根因子，與 factor_engine 保持一致。"""
        if not hasattr(self, '_candle_buffer'):
            self._candle_buffer = []
        self._candle_buffer.append(candle)
        if len(self._candle_buffer) > 300:
            self._candle_buffer = self._candle_buffer[-300:]

        def _to_float(value, default=0.0):
            try:
                if value is None:
                    return float(default)
                return float(value)
            except Exception:
                return float(default)

        try:
            rows = []
            for item in self._candle_buffer:
                close_v = _to_float(item.get("close", 0.0))
                rows.append({
                    "timestamp": _to_float(item.get("timestamp", 0.0)),
                    "open": _to_float(item.get("open", close_v), close_v),
                    "high": _to_float(item.get("high", close_v), close_v),
                    "low": _to_float(item.get("low", close_v), close_v),
                    "close": close_v,
                    "volume": _to_float(item.get("volume", 0.0), 0.0),
                    "obi": _to_float(item.get("obi", 0.0), 0.0),
                    "spread": _to_float(item.get("spread", 0.0), 0.0),
                    "bid_vol": _to_float(item.get("bid_vol", 0.0), 0.0),
                    "ask_vol": _to_float(item.get("ask_vol", 0.0), 0.0),
                })

            buf_df = pl.DataFrame(rows)
            if buf_df.height == 0:
                return {}

            factor_df = compute_factors(buf_df, oi_df=None)
            last = factor_df.tail(1).to_dicts()[0]

            factor_values = {}
            for k, v in last.items():
                if isinstance(v, (int, float, np.integer, np.floating)):
                    fv = float(v)
                    factor_values[k] = fv if np.isfinite(fv) else 0.0

            # 沒有 OI 檔時，保底欄位為 0
            factor_values.setdefault("oi_roc6", 0.0)
            factor_values.setdefault("oi_roc24", 0.0)
            factor_values.setdefault("price_oi_confirm", 0.0)

            # GP 因子：基於完整因子時序計算最新值
            if hasattr(self, 'gp_miner') and factor_df.height >= 30:
                factor_data = {}
                for col in factor_df.columns:
                    try:
                        arr = factor_df[col].to_numpy().astype(np.float64)
                        factor_data[col] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    except Exception:
                        pass
                gp_data = self.gp_miner.compute_gp_factors(factor_data)
                for name, values in gp_data.items():
                    if len(values) > 0:
                        fv = float(values[-1])
                        factor_values[name] = fv if np.isfinite(fv) else 0.0

            return factor_values
        except Exception as e:
            print(f"  即時因子計算異常: {e}")
            return {}

    def _ewm(self, arr, span):
        """計算 EWM（指數加權移動平均），返回最終純量值。"""
        alpha = 2.0 / (span + 1)
        result = float(arr[0])
        for i in range(1, len(arr)):
            result = alpha * float(arr[i]) + (1 - alpha) * result
        return result

    def _ewm_full(self, arr, span):
        """Bug 18 Fix：計算完整 EWM 陣列（O(n)），用於 MACD 等需要歷史序列的場景。"""
        alpha = 2.0 / (span + 1)
        result = np.empty(len(arr))
        result[0] = float(arr[0])
        for i in range(1, len(arr)):
            result[i] = alpha * float(arr[i]) + (1 - alpha) * result[i - 1]
        return result

    def _ewm_arr(self, arr, span):
        """保留此別名以向後相容，內部委派給 _ewm。"""
        return self._ewm(arr, span)

    def _calculate_pnl(self, exit_price):
        if self.position == "long":
            return (exit_price - self.entry_price) / self.entry_price
        elif self.position == "short":
            return (self.entry_price - exit_price) / self.entry_price
        return 0.0

    async def run(self):
        print("\n" + "=" * 60)
        print("BTC-Bot 啟動（v6.0 — 勝率極致優化版）")
        print(f"  基礎因子: {len(FACTOR_COLS)} 個")
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
        print(f"  ═══ 風控參數 ═══")
        print(f"  止損: {STOP_LOSS_ATR_MULT}×ATR | 止盈: {TAKE_PROFIT_ATR_MULT}×ATR (風險回報比 1:{TAKE_PROFIT_ATR_MULT/STOP_LOSS_ATR_MULT:.0f})")
        print(f"  最大持倉: {MAX_HOLD_BARS} 根K線 ({MAX_HOLD_BARS*5}分鐘)")
        print(f"  ═══ 閾值策略 ═══")
        print(f"  預熱: 前 {DYNAMIC_THRESHOLD_WARMUP} 根K線用搜索閾值80%")
        print(f"  目標觸發率: {TARGET_TRADE_PCT*100:.0f}%")
        print(f"  信號確認: {self.monitor.signal_confirm_bars} 根同向")
        print(f"  置信度過濾: {self.monitor.confidence_multiplier:.2f}x 閾值")
        print(f"  信號 Debug: {'開啟' if SIGNAL_DEBUG else '關閉'} (每{SIGNAL_DEBUG_INTERVAL}根K線)")
        print("=" * 60)

        # Bug 1 Fix：初始搜索改為背景非阻塞執行
        print("\n開始初始因子搜索（背景執行，不阻塞K線接收）...")
        asyncio.create_task(self._async_retrain())

        await self.feed.connect()


if __name__ == "__main__":
    bot = BTCBot()
    asyncio.run(bot.run())

