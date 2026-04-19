"""
BTC-Bot 性能監控器 v5.0
改進：
  1. 分離多空閾值：多頭和空頭各自獨立動態閾值
  2. 預熱期用搜索閾值的 80% 作為初始值（不浪費機會）
  3. PF 觸發重搜（而非僅勝率）
  4. 信號質量追蹤
"""

import numpy as np
from collections import deque


class PerformanceMonitor:
    def __init__(self, window_size=200, retrain_threshold=0.40):
        self.window_size = window_size
        self.retrain_threshold = retrain_threshold
        self.trade_results = deque(maxlen=window_size)
        # Bug 19 Fix：限制 all_trades 大小，用累計變數追蹤全域統計
        self.all_trades = deque(maxlen=2000)
        self.total_trade_count = 0
        self.cumulative_pnl = 0.0
        self.cumulative_wins = 0.0
        self.cumulative_losses_abs = 0.0
        self.current_factor = None
        self.needs_retrain = False

        # 搜索時的因子列表和交互項
        self.search_factor_cols = None
        self.interaction_pairs = None

        # 即時信號緩衝（分正負）
        self.realtime_signal_buffer = deque(maxlen=500)
        self.dynamic_long_threshold = None
        self.dynamic_short_threshold = None

        # 搜索閾值（作為預熱期初始值）
        self.search_long_threshold = None
        self.search_short_threshold = None

        # 信號方向統計
        self.long_signal_count = 0
        self.short_signal_count = 0
        self.total_signal_count = 0

        # 連續虧損追蹤
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0

        # PnL 追蹤
        self.recent_pnl = deque(maxlen=30)

    def set_factor(self, factor_info, factor_cols=None, interaction_pairs=None):
        """設置因子信息"""
        self.current_factor = factor_info
        self.needs_retrain = False
        if factor_cols is not None:
            self.search_factor_cols = list(factor_cols)
        if interaction_pairs is not None:
            self.interaction_pairs = list(interaction_pairs)

        # 保存搜索閾值作為預熱期初始值
        self.search_long_threshold = factor_info.get("long_threshold")
        self.search_short_threshold = factor_info.get("short_threshold")

        # 重置即時信號緩衝
        self.realtime_signal_buffer.clear()
        self.dynamic_long_threshold = None
        self.dynamic_short_threshold = None

        # 重置信號方向統計
        self.long_signal_count = 0
        self.short_signal_count = 0
        self.total_signal_count = 0

        regime = factor_info.get("regime", "")
        lt = self.search_long_threshold
        st = self.search_short_threshold
        if lt is not None and st is not None:
            print(f"  因子已更新，多頭閾值={lt:.3f} 空頭閾值={st:.3f}" + (f" [{regime}]" if regime else ""))
        else:
            print(f"  因子已更新")

    def record_trade(self, pnl_net):
        """記錄交易結果（已扣除手續費的淨 PnL）"""
        self.trade_results.append(1 if pnl_net > 0 else 0)
        self.all_trades.append(pnl_net)
        self.recent_pnl.append(pnl_net)
        # Bug 19 Fix：累計追蹤，避免全量掃描
        self.total_trade_count += 1
        self.cumulative_pnl += pnl_net
        if pnl_net > 0:
            self.cumulative_wins += pnl_net
        else:
            self.cumulative_losses_abs += abs(pnl_net)

        # 連續虧損追蹤
        if pnl_net <= 0:
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        else:
            self.consecutive_losses = 0

        # 重搜條件 1：近 30 筆 PF < 0.8
        if len(self.recent_pnl) >= 20:
            gains = sum(p for p in self.recent_pnl if p > 0)
            losses = abs(sum(p for p in self.recent_pnl if p < 0))
            pf = gains / (losses + 1e-10)
            if pf < 0.8:
                print(f"  近{len(self.recent_pnl)}筆 PF={pf:.2f} < 0.8，觸發重搜")
                self.needs_retrain = True

        # 重搜條件 2：連續虧損超過 6 筆
        if self.consecutive_losses >= 6:
            print(f"  連續虧損 {self.consecutive_losses} 筆，觸發重搜")
            self.needs_retrain = True

    def generate_signal(self, factor_values):
        """生成交易信號"""
        if self.current_factor is None:
            return 0.0
        weights = self.current_factor.get("weights")
        X_mean = self.current_factor.get("factor_mean")
        X_std = self.current_factor.get("factor_std")
        if weights is None or X_mean is None or X_std is None:
            return 0.0
        try:
            # 使用搜索時的因子列表
            if self.search_factor_cols is not None:
                cols = self.search_factor_cols
            else:
                from engine.genetic_engine import FACTOR_COLS
                cols = FACTOR_COLS

            x = np.array([factor_values.get(col, 0.0) for col in cols], dtype=np.float64)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # 標準化（只用基礎因子的 mean/std）
            n_base = min(len(x), len(X_mean))
            x_base = x[:n_base]
            x_base = (x_base - X_mean[:n_base]) / (X_std[:n_base] + 1e-10)

            # 生成交互項
            interaction_pairs = self.interaction_pairs or self.current_factor.get("interaction_pairs", [])
            x_interactions = []
            if interaction_pairs:
                for pair_name in interaction_pairs:
                    parts = pair_name.split("×")
                    if len(parts) == 2:
                        idx_i = cols.index(parts[0]) if parts[0] in cols else -1
                        idx_j = cols.index(parts[1]) if parts[1] in cols else -1
                        if idx_i >= 0 and idx_j >= 0 and idx_i < len(x_base) and idx_j < len(x_base):
                            x_interactions.append(x_base[idx_i] * x_base[idx_j])
                        else:
                            x_interactions.append(0.0)

            # 合併基礎因子和交互項
            if x_interactions:
                x_full = np.concatenate([x_base, np.array(x_interactions)])
            else:
                x_full = x_base

            # 計算信號
            n = min(len(x_full), len(weights))
            signal = float(x_full[:n] @ weights[:n])

            # 記錄即時信號
            self.realtime_signal_buffer.append(signal)

            # 統計信號方向
            self.total_signal_count += 1
            if signal > 0:
                self.long_signal_count += 1
            elif signal < 0:
                self.short_signal_count += 1

            return signal
        except Exception as e:
            print(f"  generate_signal 錯誤: {e}")
            return 0.0

    def get_thresholds(self, target_trade_pct=0.12, warmup=10):
        """
        返回 (long_threshold, short_threshold, is_warmup)
        預熱期：用搜索閾值的 80%
        預熱後：用即時信號的動態閾值
        """
        n_signals = len(self.realtime_signal_buffer)

        if n_signals < warmup:
            # 預熱期：用搜索閾值的 80%（不浪費機會）
            lt = (self.search_long_threshold or 0.3) * 0.8
            st = (self.search_short_threshold or 0.3) * 0.8
            return lt, st, True

        signals = np.array(list(self.realtime_signal_buffer))
        pos_signals = signals[signals > 0]
        neg_signals = np.abs(signals[signals < 0])

        # 多頭閾值：正信號的百分位數
        if len(pos_signals) > 5:
            pct = (1.0 - target_trade_pct) * 100
            lt = float(np.percentile(pos_signals, pct))
        else:
            lt = self.search_long_threshold or 0.3

        # 空頭閾值：負信號的百分位數
        if len(neg_signals) > 5:
            pct = (1.0 - target_trade_pct) * 100
            st = float(np.percentile(neg_signals, pct))
        else:
            st = self.search_short_threshold or 0.3

        # 閾值保護
        lt = max(0.03, min(lt, 1.5))
        st = max(0.03, min(st, 1.5))

        self.dynamic_long_threshold = lt
        self.dynamic_short_threshold = st
        return lt, st, False

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

        # Bug 19 Fix：使用累計變數，O(1) 計算全域 PF
        pf = self.cumulative_wins / (self.cumulative_losses_abs + 1e-10)

        return {
            "win_rate": win_rate,
            "n_trades": self.total_trade_count,
            "total_pnl": self.cumulative_pnl,
            "profit_factor": pf,
            "needs_retrain": self.needs_retrain,
            "consecutive_losses": self.consecutive_losses,
            "max_consecutive_losses": self.max_consecutive_losses,
            "long_signal_pct": long_pct,
        }

    def print_status(self):
        stats = self.get_stats()
        n = len(self.trade_results)
        retrain_str = "需重搜" if stats["needs_retrain"] else "正常"

        lt = self.dynamic_long_threshold
        st = self.dynamic_short_threshold
        if lt and st:
            thr_str = f" L閾={lt:.3f} S閾={st:.3f}"
        else:
            thr_str = " 預熱中"
        buf_str = f" 信號={len(self.realtime_signal_buffer)}根"

        long_pct = stats.get("long_signal_pct", 0.5)
        bias_str = f" 多空={long_pct:.0%}/{1-long_pct:.0%}"

        pf_str = f" PF={stats.get('profit_factor', 0):.2f}"

        cl = stats.get("consecutive_losses", 0)
        cl_str = f" 連虧={cl}" if cl > 0 else ""

        print(f"  監控: WR={stats['win_rate']:.3f} ({n}/{self.window_size}) "
              f"淨PnL={stats['total_pnl']:+.4f}{pf_str} {retrain_str}"
              f"{thr_str}{buf_str}{bias_str}{cl_str}")

