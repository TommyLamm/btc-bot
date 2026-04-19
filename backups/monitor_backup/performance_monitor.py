import polars as pl
import numpy as np
import json
import os
from datetime import datetime, timezone
from collections import deque


class PerformanceMonitor:

    def __init__(self,
                 winrate_threshold: float = 0.50,
                 window_size: int = 200,
                 min_trades: int = 50,
                 save_path: str = "monitor/state.json"):

        self.winrate_threshold = winrate_threshold
        self.window_size = window_size
        self.min_trades = min_trades
        self.save_path = save_path

        self.trades = deque(maxlen=window_size)
        self.current_factor = None
        self.is_searching = False
        self.search_count = 0
        self.last_search_time = None

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self._load_state()

    def record_trade(self, pnl_pct: float, signal: float, timestamp=None):
        trade = {
            "pnl":       pnl_pct,
            "win":       pnl_pct > 0,
            "signal":    signal,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
        self.trades.append(trade)
        self._save_state()
        return self.check_degradation()

    def check_degradation(self) -> dict:
        n = len(self.trades)
        status = {
            "n_trades":      n,
            "should_search": False,
            "reason":        None,
            "winrate":       None,
            "sharpe":        None,
        }

        if n < self.min_trades:
            status["reason"] = f"积累中 ({n}/{self.min_trades})"
            return status

        pnls = np.array([t["pnl"] for t in self.trades])
        wins = np.array([t["win"] for t in self.trades])

        winrate = wins.mean()
        sharpe  = pnls.mean() / (pnls.std() + 1e-10) * np.sqrt(105120)

        status["winrate"] = round(float(winrate), 4)
        status["sharpe"]  = round(float(sharpe), 2)

        if winrate < self.winrate_threshold:
            status["should_search"] = True
            status["reason"] = f"胜率 {winrate:.3f} < 阈值 {self.winrate_threshold}"
        elif sharpe < 0:
            status["should_search"] = True
            status["reason"] = f"Sharpe 为负 ({sharpe:.2f})"

        return status

    def generate_signal(self, factor_values: dict) -> float:
        """生成交易信号，任何字段为 None 都安全返回 0"""
        if self.current_factor is None:
            return 0.0

        weights   = self.current_factor.get("weights")
        threshold = self.current_factor.get("threshold")
        X_mean    = self.current_factor.get("factor_mean")
        X_std     = self.current_factor.get("factor_std")

        # 防御性检查，任何一个为 None 就跳过
        if weights is None or X_mean is None or X_std is None:
            return 0.0

        try:
            from engine.genetic_engine import FACTOR_COLS
            x = np.array([factor_values.get(col, 0.0) for col in FACTOR_COLS],
                         dtype=np.float64)
            x = np.nan_to_num((x - X_mean) / (X_std + 1e-10))
            return float(x @ weights)
        except Exception as e:
            print(f"  ⚠️  generate_signal 错误: {e}")
            return 0.0

    def get_action(self, signal: float) -> str:
        if self.current_factor is None:
            return "hold"
        threshold = self.current_factor.get("threshold", 0.5)
        if threshold is None:
            return "hold"
        if signal >  threshold: return "long"
        if signal < -threshold: return "short"
        return "hold"

    def get_stats(self) -> str:
        n = len(self.trades)
        if n == 0:
            return "暂无交易记录"

        pnls = np.array([t["pnl"] for t in self.trades])
        wins = np.array([t["win"] for t in self.trades])

        return "\n".join([
            f"── 最近 {n} 笔交易 ──",
            f"  胜率:     {wins.mean():.3f}  (阈值: {self.winrate_threshold})",
            f"  平均收益: {pnls.mean()*100:.4f}%",
            f"  累计收益: {pnls.sum()*100:.3f}%",
            f"  最大亏损: {pnls.min()*100:.4f}%",
            f"  重搜次数: {self.search_count}",
        ])

    def update_factor(self, new_factor: dict):
        """切换新因子组合，确保 numpy array 正确存储"""
        # 确保 numpy array 类型正确
        if new_factor.get("weights") is not None:
            new_factor["weights"]     = np.array(new_factor["weights"], dtype=np.float64)
        if new_factor.get("factor_mean") is not None:
            new_factor["factor_mean"] = np.array(new_factor["factor_mean"], dtype=np.float64)
        if new_factor.get("factor_std") is not None:
            new_factor["factor_std"]  = np.array(new_factor["factor_std"], dtype=np.float64)

        self.current_factor   = new_factor
        self.search_count    += 1
        self.last_search_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        self.trades.clear()
        self._save_state()
        print(f"✅ 因子组合已切换（第 {self.search_count} 次）"
              f" | 验证集胜率: {new_factor.get('val_winrate', 'N/A')}")

    def _save_state(self):
        state = {
            "search_count":     self.search_count,
            "last_search_time": self.last_search_time,
            "n_trades":         len(self.trades),
            "recent_winrate":   float(np.mean([t["win"] for t in self.trades])) if self.trades else None,
        }
        with open(self.save_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path) as f:
                    state = json.load(f)
                self.search_count     = state.get("search_count", 0)
                self.last_search_time = state.get("last_search_time")
            except Exception:
                pass
