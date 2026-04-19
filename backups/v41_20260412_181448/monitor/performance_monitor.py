"""
BTC-Bot 性能監控器 v4.1 — 高頻版
改進：
  1. 動態閾值：基於即時信號分佈（短預熱 10 根 K 線）
  2. 手續費感知：PnL 統計包含手續費
  3. 信號質量追蹤：記錄多空信號比例
  4. 閾值上限 0.8，確保高頻觸發
"""

import numpy as np
from collections import deque


class PerformanceMonitor:
    def __init__(self, window_size=200, retrain_threshold=0.45):
        self.window_size = window_size
        self.retrain_threshold = retrain_threshold
        self.trade_results = deque(maxlen=window_size)
        self.all_trades = []
        self.current_factor = None
        self.needs_retrain = False

        self.search_factor_cols = None

        self.realtime_signal_buffer = deque(maxlen=500)
        self.dynamic_threshold = None

        self.long_signal_count = 0
        self.short_signal_count = 0
        self.total_signal_count = 0

        self.consecutive_losses = 0
        self.max_consecutive_losses = 0

    def set_factor(self, factor_info, factor_cols=None):
        self.current_factor = factor_info
        self.needs_retrain = False
        if factor_cols is not None:
            self.search_factor_cols = list(factor_cols)
        self.realtime_signal_buffer.clear()
        self.dynamic_threshold = None
        self.long_signal_count = 0
        self.short_signal_count = 0
        self.total_signal_count = 0

        threshold = factor_info.get("threshold")
        regime = factor_info.get("regime", "")
        if threshold is not None:
            print(f"  因子已更新，閾值={threshold:.3f}" + (f" [{regime}]" if regime else ""))
        else:
            print(f"  因子已更新")

    def record_trade(self, pnl_net):
        self.trade_results.append(1 if pnl_net > 0 else 0)
        self.all_trades.append(pnl_net)

        if pnl_net <= 0:
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        else:
            self.consecutive_losses = 0

        if len(self.trade_results) >= 30:
            recent_wr = sum(list(self.trade_results)[-30:]) / 30
            if recent_wr < self.retrain_threshold:
                print(f"  近30筆勝率 {recent_wr:.3f} < {self.retrain_threshold}，觸發重搜")
                self.needs_retrain = True

        if self.consecutive_losses >= 8:
            print(f"  連續虧損 {self.consecutive_losses} 筆，觸發重搜")
            self.needs_retrain = True

    def generate_signal(self, factor_values):
        if self.current_factor is None:
            return 0.0
        weights = self.current_factor.get("weights")
        X_mean = self.current_factor.get("factor_mean")
        X_std = self.current_factor.get("factor_std")
        if weights is None or X_mean is None or X_std is None:
            return 0.0
        try:
            if self.search_factor_cols is not None:
                cols = self.search_factor_cols
            else:
                from engine.genetic_engine import FACTOR_COLS
                cols = FACTOR_COLS

            x = np.array([factor_values.get(col, 0.0) for col in cols], dtype=np.float64)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            n = min(len(x), len(weights), len(X_mean))
            x = x[:n]
            w = weights[:n]
            m = X_mean[:n]
            s = X_std[:n]

            x = (x - m) / (s + 1e-10)
            signal = float(x @ w)

            self.realtime_signal_buffer.append(signal)

            self.total_signal_count += 1
            if signal > 0:
                self.long_signal_count += 1
            elif signal < 0:
                self.short_signal_count += 1

            return signal
        except Exception as e:
            print(f"  generate_signal 錯誤: {e}")
            return 0.0

    def get_dynamic_threshold(self, target_trade_pct=0.04, warmup=30):
        n_signals = len(self.realtime_signal_buffer)

        if n_signals < warmup:
            return None

        signals = np.array(list(self.realtime_signal_buffer))
        abs_signals = np.abs(signals)

        target_percentile = (1.0 - target_trade_pct) * 100
        threshold = float(np.percentile(abs_signals, target_percentile))

        threshold = max(threshold, 0.05)
        threshold = min(threshold, 0.8)

        self.dynamic_threshold = threshold
        return threshold

    def get_stats(self):
        if not self.trade_results:
            return {
                "win_rate": 0.0, "n_trades": 0, "total_pnl": 0.0,
                "needs_retrain": self.needs_retrain,
                "consecutive_losses": self.consecutive_losses,
                "long_signal_pct": 0.5,
            }
        win_rate = sum(self.trade_results) / len(self.trade_results)
        long_pct = self.long_signal_count / max(1, self.total_signal_count)
        return {
            "win_rate": win_rate,
            "n_trades": len(self.all_trades),
            "total_pnl": sum(self.all_trades),
            "needs_retrain": self.needs_retrain,
            "consecutive_losses": self.consecutive_losses,
            "max_consecutive_losses": self.max_consecutive_losses,
            "long_signal_pct": long_pct,
        }

    def print_status(self):
        stats = self.get_stats()
        n = len(self.trade_results)
        retrain_str = "需重搜" if stats["needs_retrain"] else "正常"

        thr_str = f" 動態閾值={self.dynamic_threshold:.4f}" if self.dynamic_threshold else " 預熱中"
        buf_str = f" 信號={len(self.realtime_signal_buffer)}根"

        long_pct = stats.get("long_signal_pct", 0.5)
        bias_str = f" 多空={long_pct:.0%}/{1-long_pct:.0%}"

        cl = stats.get("consecutive_losses", 0)
        cl_str = f" 連虧={cl}" if cl > 0 else ""

        print(f"  監控: 勝率={stats['win_rate']:.3f} ({n}/{self.window_size}) "
              f"累計淨PnL={stats['total_pnl']:+.4f} {retrain_str}"
              f"{thr_str}{buf_str}{bias_str}{cl_str}")
