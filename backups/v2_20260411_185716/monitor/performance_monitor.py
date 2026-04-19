"""
BTC-Bot 性能監控器（優化版）
200 筆滑動窗口監控勝率，勝率低於閾值時自動觸發重搜。
"""

import numpy as np
from collections import deque


class PerformanceMonitor:
    def __init__(self, window_size: int = 200, retrain_threshold: float = 0.50):
        self.window_size = window_size
        self.retrain_threshold = retrain_threshold
        self.trade_results = deque(maxlen=window_size)
        self.all_trades = []
        self.current_factor = None
        self.needs_retrain = False

    def set_factor(self, factor_info: dict):
        self.current_factor = factor_info
        self.needs_retrain = False
        threshold = factor_info.get("threshold")
        if threshold is not None:
            print(f"  因子已更新，閾值={threshold:.3f}")
        else:
            print(f"  因子已更新")

    def record_trade(self, pnl: float):
        self.trade_results.append(1 if pnl > 0 else 0)
        self.all_trades.append(pnl)
        if len(self.trade_results) >= self.window_size:
            win_rate = sum(self.trade_results) / len(self.trade_results)
            if win_rate < self.retrain_threshold:
                print(f"  勝率 {win_rate:.3f} < {self.retrain_threshold}，觸發重搜")
                self.needs_retrain = True

    def generate_signal(self, factor_values: dict) -> float:
        if self.current_factor is None:
            return 0.0
        weights = self.current_factor.get("weights")
        X_mean = self.current_factor.get("factor_mean")
        X_std = self.current_factor.get("factor_std")
        # ★ 防禦性檢查
        if weights is None or X_mean is None or X_std is None:
            return 0.0
        try:
            from engine.genetic_engine import FACTOR_COLS
            x = np.array([factor_values.get(col, 0.0) for col in FACTOR_COLS], dtype=np.float64)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = (x - X_mean) / (X_std + 1e-10)
            return float(x @ weights)
        except Exception as e:
            print(f"  generate_signal 錯誤: {e}")
            return 0.0

    def get_stats(self):
        if not self.trade_results:
            return {"win_rate": 0.0, "n_trades": 0, "total_pnl": 0.0, "needs_retrain": self.needs_retrain}
        win_rate = sum(self.trade_results) / len(self.trade_results)
        return {"win_rate": win_rate, "n_trades": len(self.all_trades),
                "total_pnl": sum(self.all_trades), "needs_retrain": self.needs_retrain}

    def print_status(self):
        stats = self.get_stats()
        n = len(self.trade_results)
        retrain_str = "需重搜" if stats["needs_retrain"] else "正常"
        print(
            f"  監控: 勝率={stats['win_rate']:.3f} "
            f"({n}/{self.window_size}) "
            f"累計PnL={stats['total_pnl']:+.4f} "
            f"{retrain_str}"
        )
