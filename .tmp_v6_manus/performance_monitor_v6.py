"""
BTC-Bot 性能監控器 v6.0 — 勝率極致優化版
核心改進：
  1. 信號確認機制：連續 N 根 K 線信號同向才觸發開倉
  2. 信號置信度過濾：|signal| 必須超過閾值的 confidence_multiplier 倍
  3. 動態止損調整：根據信號強度調整止損距離
  4. 更嚴格的重搜條件：PF < 0.9 或連續虧損 5 筆
  5. 信號品質追蹤：追蹤信號準確率和盈虧比
  6. 保留 v5.5-fix 的全面類型安全保護
"""

import numpy as np
from collections import deque


class PerformanceMonitor:
    def __init__(self, window_size=200, retrain_threshold=0.40):
        self.window_size = window_size
        self.retrain_threshold = retrain_threshold
        self.trade_results = deque(maxlen=window_size)
        self.all_trades = []
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

        # v6.0：信號確認參數
        self.signal_confirm_bars = 2
        self.confidence_multiplier = 1.15
        self._confirm_long_count = 0
        self._confirm_short_count = 0
        self._last_signal_dir = 0

        # 信號方向統計
        self.long_signal_count = 0
        self.short_signal_count = 0
        self.total_signal_count = 0

        # 連續虧損追蹤
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0

        # PnL 追蹤
        self.recent_pnl = deque(maxlen=30)

        # v6.0：信號品質追蹤
        self.signal_accuracy_buffer = deque(maxlen=100)
        self.avg_win_size = deque(maxlen=50)
        self.avg_loss_size = deque(maxlen=50)

    def set_factor(self, factor_info, factor_cols=None, interaction_pairs=None):
        """設置因子信息"""
        self.current_factor = factor_info
        self.needs_retrain = False
        if factor_cols is not None:
            if isinstance(factor_cols, (list, tuple)):
                self.search_factor_cols = list(factor_cols)
            elif isinstance(factor_cols, np.ndarray):
                self.search_factor_cols = factor_cols.tolist()
            else:
                print(f"  [警告] factor_cols 類型異常: {type(factor_cols)}，忽略")
                self.search_factor_cols = None
        if interaction_pairs is not None:
            if isinstance(interaction_pairs, (list, tuple)):
                self.interaction_pairs = list(interaction_pairs)
            elif isinstance(interaction_pairs, np.ndarray):
                self.interaction_pairs = interaction_pairs.tolist()
            else:
                print(f"  [警告] interaction_pairs 類型異常: {type(interaction_pairs)}，忽略")
                self.interaction_pairs = []

        # 保存搜索閾值
        self.search_long_threshold = factor_info.get("long_threshold")
        self.search_short_threshold = factor_info.get("short_threshold")

        # v6.0：讀取信號確認參數
        self.signal_confirm_bars = factor_info.get("signal_confirm_bars", 2)
        self.confidence_multiplier = factor_info.get("confidence_multiplier", 1.15)

        # 重置
        self.realtime_signal_buffer.clear()
        self.dynamic_long_threshold = None
        self.dynamic_short_threshold = None
        self.long_signal_count = 0
        self.short_signal_count = 0
        self.total_signal_count = 0
        self._confirm_long_count = 0
        self._confirm_short_count = 0
        self._last_signal_dir = 0

        regime = factor_info.get("regime", "")
        lt = self.search_long_threshold
        st = self.search_short_threshold
        if lt is not None and st is not None:
            print(f"  因子已更新，多頭閾值={lt:.3f} 空頭閾值={st:.3f} "
                  f"確認={self.signal_confirm_bars}根 置信度={self.confidence_multiplier:.2f}x"
                  + (f" [{regime}]" if regime else ""))
        else:
            print(f"  因子已更新")

    def record_trade(self, pnl_net):
        """記錄交易結果"""
        self.trade_results.append(1 if pnl_net > 0 else 0)
        self.all_trades.append(pnl_net)
        self.recent_pnl.append(pnl_net)

        # v6.0：追蹤盈虧大小
        if pnl_net > 0:
            self.avg_win_size.append(pnl_net)
        elif pnl_net < 0:
            self.avg_loss_size.append(abs(pnl_net))

        # 連續虧損追蹤
        if pnl_net <= 0:
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        else:
            self.consecutive_losses = 0

        # v6.0 重搜條件（更嚴格）
        # 條件 1：近 20 筆 PF < 0.9
        if len(self.recent_pnl) >= 20:
            gains = sum(p for p in self.recent_pnl if p > 0)
            losses = abs(sum(p for p in self.recent_pnl if p < 0))
            pf = gains / (losses + 1e-10)
            if pf < 0.9:
                print(f"  近{len(self.recent_pnl)}筆 PF={pf:.2f} < 0.9，觸發重搜")
                self.needs_retrain = True

        # 條件 2：連續虧損超過 5 筆（v6.0 更嚴格）
        if self.consecutive_losses >= 5:
            print(f"  連續虧損 {self.consecutive_losses} 筆，觸發重搜")
            self.needs_retrain = True

        # v6.0 條件 3：近期勝率驟降
        if len(self.trade_results) >= 15:
            recent_wr = sum(list(self.trade_results)[-15:]) / 15
            if recent_wr < 0.35:
                print(f"  近15筆勝率={recent_wr:.1%} < 35%，觸發重搜")
                self.needs_retrain = True

    def _safe_to_list(self, val):
        """安全地將任何值轉為 list"""
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            return list(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, str):
            return [val]
        return None

    def generate_signal(self, factor_values):
        """v6.0 生成交易信號（含信號確認和置信度過濾）"""
        if self.current_factor is None:
            return 0.0
        weights = self.current_factor.get("weights")
        factor_mean = self.current_factor.get("factor_mean")
        factor_std = self.current_factor.get("factor_std")
        if weights is None or factor_mean is None or factor_std is None:
            return 0.0

        # 類型安全
        if isinstance(weights, (int, float)):
            weights = np.array([weights], dtype=np.float64)
        else:
            weights = np.atleast_1d(np.asarray(weights, dtype=np.float64))

        if isinstance(factor_mean, (int, float)):
            factor_mean = np.array([factor_mean], dtype=np.float64)
        else:
            factor_mean = np.atleast_1d(np.asarray(factor_mean, dtype=np.float64))

        if isinstance(factor_std, (int, float)):
            factor_std = np.array([factor_std], dtype=np.float64)
        else:
            factor_std = np.atleast_1d(np.asarray(factor_std, dtype=np.float64))

        try:
            # 因子列表獲取
            screened_cols = self._safe_to_list(
                self.current_factor.get("screened_factor_cols")
            )
            search_cols = self._safe_to_list(self.search_factor_cols)

            if screened_cols is not None and len(screened_cols) > 0:
                cols = screened_cols
            elif search_cols is not None and len(search_cols) > 0:
                cols = search_cols
            else:
                from engine.genetic_engine import FACTOR_COLS
                cols = list(FACTOR_COLS)

            x = np.array([factor_values.get(col, 0.0) for col in cols], dtype=np.float64)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # 標準化
            n_base = min(len(x), len(factor_mean))
            x_base = x[:n_base]
            safe_std = factor_std[:n_base].copy()
            safe_std[safe_std < 1e-10] = 1.0
            x_base = (x_base - factor_mean[:n_base]) / safe_std

            # 交互項
            raw_pairs = self.interaction_pairs or self.current_factor.get("interaction_pairs", [])
            interaction_pairs = self._safe_to_list(raw_pairs)
            if interaction_pairs is None:
                interaction_pairs = []

            x_interactions = []
            if interaction_pairs:
                for pair_name in interaction_pairs:
                    if not isinstance(pair_name, str):
                        x_interactions.append(0.0)
                        continue
                    parts = pair_name.split("\u00d7")
                    if len(parts) == 2:
                        idx_i = -1
                        idx_j = -1
                        if parts[0] in cols:
                            idx_i = cols.index(parts[0])
                        if parts[1] in cols:
                            idx_j = cols.index(parts[1])
                        if idx_i >= 0 and idx_j >= 0 and idx_i < len(x_base) and idx_j < len(x_base):
                            x_interactions.append(x_base[idx_i] * x_base[idx_j])
                        else:
                            x_interactions.append(0.0)

            if x_interactions:
                x_full = np.concatenate([x_base, np.array(x_interactions)])
            else:
                x_full = x_base

            # 計算原始信號
            n = min(len(x_full), len(weights))
            if n == 0:
                return 0.0
            raw_signal = float(x_full[:n] @ weights[:n])

            # 記錄即時信號
            self.realtime_signal_buffer.append(raw_signal)

            # 統計信號方向
            self.total_signal_count += 1
            if raw_signal > 0:
                self.long_signal_count += 1
            elif raw_signal < 0:
                self.short_signal_count += 1

            return raw_signal
        except Exception as e:
            print(f"  信號計算異常: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def should_open_position(self, signal, long_threshold, short_threshold):
        """v6.0 新增：信號確認 + 置信度過濾
        返回: (direction, confirmed)
        direction: 1=多, -1=空, 0=無信號
        confirmed: True=已確認可開倉
        """
        conf_long_thr = long_threshold * self.confidence_multiplier
        conf_short_thr = short_threshold * self.confidence_multiplier

        if signal > conf_long_thr:
            self._confirm_long_count += 1
            self._confirm_short_count = 0
        elif signal < -conf_short_thr:
            self._confirm_short_count += 1
            self._confirm_long_count = 0
        else:
            self._confirm_long_count = 0
            self._confirm_short_count = 0

        if self._confirm_long_count >= self.signal_confirm_bars:
            self._confirm_long_count = 0
            return 1, True
        elif self._confirm_short_count >= self.signal_confirm_bars:
            self._confirm_short_count = 0
            return -1, True

        return 0, False

    def should_close_position(self, signal, position_dir, long_threshold, short_threshold):
        """v6.0 新增：平倉信號判斷（信號反轉時平倉，不需要確認）"""
        if position_dir == 1 and signal < -short_threshold:
            return True
        if position_dir == -1 and signal > long_threshold:
            return True
        return False

    def get_thresholds(self, target_trade_pct=0.10, warmup=15):
        """
        返回 (long_threshold, short_threshold, is_warmup)
        v6.0：target_trade_pct 降到 10%（更精選交易）
        """
        n_signals = len(self.realtime_signal_buffer)

        if n_signals < warmup:
            lt = (self.search_long_threshold or 0.3) * 0.85
            st = (self.search_short_threshold or 0.3) * 0.85
            return lt, st, True

        signals = np.array(list(self.realtime_signal_buffer))
        pos_signals = signals[signals > 0]
        neg_signals = np.abs(signals[signals < 0])

        if len(pos_signals) > 5:
            pct = (1.0 - target_trade_pct) * 100
            lt = float(np.percentile(pos_signals, pct))
        else:
            lt = self.search_long_threshold or 0.3

        if len(neg_signals) > 5:
            pct = (1.0 - target_trade_pct) * 100
            st = float(np.percentile(neg_signals, pct))
        else:
            st = self.search_short_threshold or 0.3

        lt = max(0.03, min(lt, 1.5))
        st = max(0.03, min(st, 1.5))

        self.dynamic_long_threshold = lt
        self.dynamic_short_threshold = st
        return lt, st, False

    def get_signal_quality(self):
        """v6.0 新增：返回信號品質指標"""
        if len(self.avg_win_size) < 3 or len(self.avg_loss_size) < 3:
            return {"win_loss_ratio": 0.0, "signal_quality": "insufficient_data"}

        avg_win = np.mean(list(self.avg_win_size))
        avg_loss = np.mean(list(self.avg_loss_size))
        wl_ratio = avg_win / (avg_loss + 1e-10)

        if wl_ratio > 1.5:
            quality = "excellent"
        elif wl_ratio > 1.0:
            quality = "good"
        elif wl_ratio > 0.7:
            quality = "fair"
        else:
            quality = "poor"

        return {
            "win_loss_ratio": wl_ratio,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "signal_quality": quality,
        }

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

        gains = sum(p for p in self.all_trades if p > 0)
        losses = abs(sum(p for p in self.all_trades if p < 0))
        pf = gains / (losses + 1e-10)

        quality = self.get_signal_quality()

        return {
            "win_rate": win_rate,
            "n_trades": len(self.all_trades),
            "total_pnl": sum(self.all_trades),
            "profit_factor": pf,
            "needs_retrain": self.needs_retrain,
            "consecutive_losses": self.consecutive_losses,
            "max_consecutive_losses": self.max_consecutive_losses,
            "long_signal_pct": long_pct,
            "win_loss_ratio": quality.get("win_loss_ratio", 0),
            "signal_quality": quality.get("signal_quality", "unknown"),
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
        wl_str = f" 盈虧比={stats.get('win_loss_ratio', 0):.2f}"
        qual_str = f" [{stats.get('signal_quality', '?')}]"

        cl = stats.get("consecutive_losses", 0)
        cl_str = f" 連虧={cl}" if cl > 0 else ""

        print(f"  監控: WR={stats['win_rate']:.3f} ({n}/{self.window_size}) "
              f"淨PnL={stats['total_pnl']:+.4f}{pf_str}{wl_str}{qual_str} {retrain_str}"
              f"{thr_str}{buf_str}{bias_str}{cl_str}")
