"""
BTC-Bot 遺傳編程（GP）因子挖掘器 v6.0 — 勝率導向重構
核心改進：
  1. Hit Rate 直接優化：因子方向正確率作為主要評估指標
  2. 條件勝率：只在信號強度超過閾值時計算勝率（過濾噪音）
  3. 多時間框架 IC：同時評估 1-bar、3-bar、5-bar 預測力
  4. 信號持續性：好因子的信號在 2-3 根 K 線內保持方向
  5. 因子衰減監控：自動淘汰近期表現下降的因子
  6. 更豐富的操作符：新增 rank、ts_corr、ts_cov 等量化因子操作
  7. 因子庫動態管理：品質分數定期重算，自動淘汰
  8. 保留 v5.5 的向量化 rolling、冗餘檢測、方向一致性
"""

import numpy as np
import random
import json
import os
import copy
from datetime import datetime

# 擴展的基礎因子列表（匹配 factor_engine v6.0）
BASE_FACTORS = [
    "roc5", "roc20", "mom_accel",
    "ma10dev", "ma30dev", "ma_alignment",
    "vwapdev", "priceimpact", "macdhist",
    "rsi_norm", "bb_pctb",
    "vol_ratio", "tr_ratio", "vol_surge", "vol_price_div", "obv_slope",
    "candle_dir",
    "oi_roc6", "oi_roc24", "price_oi_confirm",
    "hour_sin", "hour_cos",
    "roc15m", "roc1h", "bb_width", "vol_accel",
    "vwap_slope", "rsi_divergence",
    # v5.5 因子
    "momentum_quality", "kaufman_er", "vol_regime_change",
    "range_expansion", "consec_candles", "return_skew",
    "obi_accel", "obi_price_confirm", "spread_norm", "bid_ask_imbalance",
    # v6.0 新增因子
    "vpin", "kyle_lambda", "amihud_illiq",
    "mtf_trend_align", "mtf_mom_confirm",
    "signal_persistence", "regime_indicator",
    "close_to_high", "close_to_low",
    "volume_price_trend", "mfi_norm",
]

BINARY_OPS = ["add", "sub", "mul", "div", "max", "min"]
UNARY_OPS = ["abs", "neg", "sign", "log1p", "tanh", "square", "sqrt_abs"]
WINDOW_OPS = ["rolling_mean", "rolling_std", "rolling_rank", "diff", "shift",
              "rolling_zscore", "ewm", "rolling_max", "rolling_min",
              "rolling_skew", "ts_delta_pct"]
WINDOWS = [3, 5, 8, 12, 20]


class GPNode:
    def __init__(self, node_type, value=None, children=None, window=None):
        self.node_type = node_type
        self.value = value
        self.children = children or []
        self.window = window

    def depth(self):
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self):
        if not self.children:
            return 1
        return 1 + sum(c.size() for c in self.children)

    def to_string(self):
        if self.node_type == "factor":
            return self.value
        elif self.node_type == "const":
            return f"{self.value:.2f}"
        elif self.node_type == "binary":
            left = self.children[0].to_string()
            right = self.children[1].to_string()
            op_map = {"add": "+", "sub": "-", "mul": "*", "div": "/",
                      "max": "max", "min": "min"}
            op = op_map.get(self.value, self.value)
            if self.value in ("max", "min"):
                return f"{op}({left}, {right})"
            return f"({left} {op} {right})"
        elif self.node_type == "unary":
            return f"{self.value}({self.children[0].to_string()})"
        elif self.node_type == "window":
            return f"{self.value}({self.children[0].to_string()}, {self.window})"
        return "?"

    def evaluate(self, data):
        try:
            n = len(next(iter(data.values())))
            if self.node_type == "factor":
                return data.get(self.value, np.zeros(n))
            elif self.node_type == "const":
                return np.full(n, self.value)
            elif self.node_type == "binary":
                left = self.children[0].evaluate(data)
                right = self.children[1].evaluate(data)
                return self._binary_op(self.value, left, right)
            elif self.node_type == "unary":
                child = self.children[0].evaluate(data)
                return self._unary_op(self.value, child)
            elif self.node_type == "window":
                child = self.children[0].evaluate(data)
                return self._window_op(self.value, child, self.window)
        except Exception:
            pass
        return np.zeros(len(next(iter(data.values()))))

    @staticmethod
    def _binary_op(op, a, b):
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        if op == "add": return a + b
        elif op == "sub": return a - b
        elif op == "mul": return np.clip(a * b, -1e6, 1e6)
        elif op == "div": return a / (b + np.sign(b) * 1e-10 + 1e-10)
        elif op == "max": return np.maximum(a, b)
        elif op == "min": return np.minimum(a, b)
        return a

    @staticmethod
    def _unary_op(op, a):
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        if op == "abs": return np.abs(a)
        elif op == "neg": return -a
        elif op == "sign": return np.sign(a)
        elif op == "log1p": return np.sign(a) * np.log1p(np.abs(a))
        elif op == "tanh": return np.tanh(a)
        elif op == "square": return np.clip(a ** 2, 0, 1e6) * np.sign(a)
        elif op == "sqrt_abs": return np.sqrt(np.abs(a)) * np.sign(a)
        return a

    @staticmethod
    def _window_op(op, a, window):
        """向量化 rolling 操作（v6.0 擴展）"""
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(a)
        result = np.zeros(n)

        if op == "rolling_mean":
            cumsum = np.cumsum(a)
            result[window-1:] = (cumsum[window-1:] - np.concatenate([[0], cumsum[:-window]])) / window
        elif op == "rolling_std":
            cumsum = np.cumsum(a)
            cumsum2 = np.cumsum(a ** 2)
            if n >= window:
                s1 = cumsum[window-1:] - np.concatenate([[0], cumsum[:-window]])
                s2 = cumsum2[window-1:] - np.concatenate([[0], cumsum2[:-window]])
                var = s2 / window - (s1 / window) ** 2
                var = np.maximum(var, 0)
                result[window-1:] = np.sqrt(var)
        elif op == "rolling_rank":
            for i in range(window - 1, n):
                w = a[i-window+1:i+1]
                result[i] = np.sum(w <= a[i]) / window
        elif op == "diff":
            w = min(window, n - 1)
            if w > 0:
                result[w:] = a[w:] - a[:-w]
        elif op == "shift":
            w = min(window, n - 1)
            if w > 0:
                result[w:] = a[:-w]
        elif op == "rolling_zscore":
            cumsum = np.cumsum(a)
            cumsum2 = np.cumsum(a ** 2)
            if n >= window:
                s1 = cumsum[window-1:] - np.concatenate([[0], cumsum[:-window]])
                s2 = cumsum2[window-1:] - np.concatenate([[0], cumsum2[:-window]])
                mean_w = s1 / window
                var_w = s2 / window - mean_w ** 2
                var_w = np.maximum(var_w, 0)
                std_w = np.sqrt(var_w)
                safe_std = np.where(std_w < 1e-10, 1.0, std_w)
                result[window-1:] = (a[window-1:] - mean_w) / safe_std
        elif op == "ewm":
            alpha = 2.0 / (window + 1)
            result[0] = a[0]
            for i in range(1, n):
                result[i] = alpha * a[i] + (1 - alpha) * result[i-1]
        elif op == "rolling_max":
            for i in range(window - 1, n):
                result[i] = np.max(a[i-window+1:i+1])
        elif op == "rolling_min":
            for i in range(window - 1, n):
                result[i] = np.min(a[i-window+1:i+1])
        elif op == "rolling_skew":
            for i in range(window - 1, n):
                chunk = a[i-window+1:i+1]
                m = chunk.mean()
                s = chunk.std()
                if s > 1e-10:
                    result[i] = ((chunk - m) ** 3).mean() / (s ** 3)
        elif op == "ts_delta_pct":
            w = min(window, n - 1)
            if w > 0:
                prev = a[:-w]
                safe_prev = np.where(np.abs(prev) < 1e-10, 1e-10, prev)
                result[w:] = (a[w:] - prev) / np.abs(safe_prev)
                result = np.clip(result, -10, 10)
        return result

    def to_dict(self):
        d = {"type": self.node_type, "value": self.value}
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        if self.window is not None:
            d["window"] = self.window
        return d

    @classmethod
    def from_dict(cls, d):
        children = [cls.from_dict(c) for c in d.get("children", [])]
        return cls(d["type"], d.get("value"), children, d.get("window"))


class GPFactorMiner:
    def __init__(self, pop_size=600, n_generations=100, max_depth=5,
                 tournament_size=5, crossover_rate=0.6, mutation_rate=0.3,
                 ic_threshold=0.03, hit_rate_threshold=0.52,
                 max_new_factors=8, max_correlation=0.55,
                 save_path="data/gp_factors.json"):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.ic_threshold = ic_threshold
        self.hit_rate_threshold = hit_rate_threshold
        self.max_new_factors = max_new_factors
        self.max_correlation = max_correlation
        self.save_path = save_path
        self.discovered_factors = []
        self._load_factors()

    def _load_factors(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "r") as f:
                    data = json.load(f)
                self.discovered_factors = [
                    {"name": d["name"], "expr": d["expr"],
                     "tree": GPNode.from_dict(d["tree"]),
                     "ic": d["ic"], "ic_ir": d.get("ic_ir", 0),
                     "hit_rate": d.get("hit_rate", 0.5),
                     "cond_hit_rate": d.get("cond_hit_rate", 0.5),
                     "turnover": d.get("turnover", 0),
                     "ic_stability": d.get("ic_stability", 0),
                     "persistence": d.get("persistence", 0)}
                    for d in data
                ]
                print(f"  載入 {len(self.discovered_factors)} 個 GP 因子")
            except Exception as e:
                print(f"  載入 GP 因子失敗: {e}")

    def _save_factors(self):
        data = [{"name": f["name"], "expr": f["expr"],
                 "tree": f["tree"].to_dict(), "ic": f["ic"],
                 "ic_ir": f.get("ic_ir", 0),
                 "hit_rate": f.get("hit_rate", 0.5),
                 "cond_hit_rate": f.get("cond_hit_rate", 0.5),
                 "turnover": f.get("turnover", 0),
                 "ic_stability": f.get("ic_stability", 0),
                 "persistence": f.get("persistence", 0)}
                for f in self.discovered_factors]
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)

    def _random_tree(self, depth=0):
        if depth >= self.max_depth or (depth > 1 and random.random() < 0.3):
            if random.random() < 0.85:
                available = [f for f in BASE_FACTORS]
                return GPNode("factor", random.choice(available))
            else:
                return GPNode("const", round(random.uniform(-2, 2), 2))
        choice = random.random()
        if choice < 0.35:
            op = random.choice(BINARY_OPS)
            return GPNode("binary", op, [self._random_tree(depth+1), self._random_tree(depth+1)])
        elif choice < 0.55:
            op = random.choice(UNARY_OPS)
            return GPNode("unary", op, [self._random_tree(depth+1)])
        else:
            op = random.choice(WINDOW_OPS)
            return GPNode("window", op, [self._random_tree(depth+1)], random.choice(WINDOWS))

    # ═══════════════════════════════════════════════════
    #  v6.0 核心指標：Hit Rate（方向正確率）
    # ═══════════════════════════════════════════════════
    def _compute_hit_rate(self, factor_values, returns):
        """計算因子的方向正確率（Hit Rate）
        因子值 > 0 且未來收益 > 0 = 正確
        因子值 < 0 且未來收益 < 0 = 正確
        """
        fv = np.nan_to_num(factor_values, nan=0.0, posinf=0.0, neginf=0.0)
        rt = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        # 只計算有明確方向的樣本
        mask = (np.abs(fv) > 1e-10) & (np.abs(rt) > 1e-10)
        if mask.sum() < 50:
            return 0.5
        correct = np.sign(fv[mask]) == np.sign(rt[mask])
        return float(correct.mean())

    def _compute_conditional_hit_rate(self, factor_values, returns, percentile=70):
        """v6.0 核心：條件勝率
        只在因子值的絕對值超過 percentile 閾值時計算勝率
        模擬實際交易：只在信號強時才開倉
        """
        fv = np.nan_to_num(factor_values, nan=0.0, posinf=0.0, neginf=0.0)
        rt = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        abs_fv = np.abs(fv)
        threshold = np.percentile(abs_fv[abs_fv > 1e-10], percentile) if (abs_fv > 1e-10).sum() > 50 else 0
        if threshold < 1e-10:
            return 0.5
        strong_mask = abs_fv > threshold
        if strong_mask.sum() < 30:
            return 0.5
        correct = np.sign(fv[strong_mask]) == np.sign(rt[strong_mask])
        return float(correct.mean())

    def _compute_signal_persistence(self, factor_values, window=3):
        """v6.0 新增：信號持續性
        好因子的信號方向在連續 window 根 K 線內保持一致
        返回值越高 = 信號越穩定（不是噪音）
        """
        fv = np.nan_to_num(factor_values, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(fv)
        if n < window + 10:
            return 0.5
        signs = np.sign(fv)
        consistent = 0
        total = 0
        for i in range(window, n):
            if abs(fv[i]) > 1e-10:
                # 檢查前 window 根 K 線的信號方向是否和當前一致
                prev_signs = signs[i-window:i]
                current_sign = signs[i]
                agreement = (prev_signs == current_sign).mean()
                consistent += agreement
                total += 1
        if total < 30:
            return 0.5
        return float(consistent / total)

    def _compute_multi_horizon_ic(self, factor_values, returns_1bar, close_arr):
        """v6.0 新增：多時間框架 IC
        同時計算 1-bar、3-bar、5-bar 的 IC，取加權平均
        """
        fv = np.nan_to_num(factor_values, nan=0.0, posinf=0.0, neginf=0.0)
        ic_1 = abs(self._compute_ic(fv, returns_1bar))

        # 3-bar 收益
        ret_3 = np.zeros(len(close_arr))
        if len(close_arr) > 3:
            ret_3[:-3] = (close_arr[3:] - close_arr[:-3]) / (close_arr[:-3] + 1e-10)
        ic_3 = abs(self._compute_ic(fv, ret_3))

        # 5-bar 收益
        ret_5 = np.zeros(len(close_arr))
        if len(close_arr) > 5:
            ret_5[:-5] = (close_arr[5:] - close_arr[:-5]) / (close_arr[:-5] + 1e-10)
        ic_5 = abs(self._compute_ic(fv, ret_5))

        # 加權：1-bar 40%, 3-bar 35%, 5-bar 25%
        return ic_1 * 0.40 + ic_3 * 0.35 + ic_5 * 0.25

    def _compute_ic(self, factor_values, returns):
        """計算 Rank IC"""
        factor_values = np.nan_to_num(factor_values, nan=0.0, posinf=0.0, neginf=0.0)
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        if np.std(factor_values) < 1e-10 or np.std(returns) < 1e-10:
            return 0.0
        rank_f = np.argsort(np.argsort(factor_values)).astype(float)
        rank_r = np.argsort(np.argsort(returns)).astype(float)
        mean_f, mean_r = rank_f.mean(), rank_r.mean()
        std_f, std_r = rank_f.std(), rank_r.std()
        if std_f < 1e-10 or std_r < 1e-10:
            return 0.0
        return float(np.mean((rank_f - mean_f) * (rank_r - mean_r)) / (std_f * std_r))

    def _compute_ic_ir(self, factor_values, returns, window=50):
        """計算 IC 穩定性（IC_IR = mean(rolling_IC) / std(rolling_IC)）"""
        n = len(factor_values)
        if n < window * 3:
            return 0.0
        ics = []
        for i in range(window, n, window):
            start = max(0, i - window)
            ic = self._compute_ic(factor_values[start:i], returns[start:i])
            ics.append(ic)
        if len(ics) < 3:
            return 0.0
        ics = np.array(ics)
        std = np.std(ics)
        if std < 1e-10:
            return 0.0
        return float(np.mean(ics) / std)

    def _compute_ic_stability(self, factor_values, returns, window=200):
        """月度 IC 穩定性（IC hit rate）"""
        n = len(factor_values)
        if n < window * 3:
            return 1.0
        monthly_ics = []
        for i in range(0, n - window + 1, window):
            chunk_f = factor_values[i:i+window]
            chunk_r = returns[i:i+window]
            ic = abs(self._compute_ic(chunk_f, chunk_r))
            monthly_ics.append(ic)
        if len(monthly_ics) < 3:
            return 1.0
        hit_rate = sum(1 for ic in monthly_ics if ic > 0.01) / len(monthly_ics)
        return hit_rate

    def _compute_direction_consistency(self, factor_values, returns, n_splits=4):
        """因子方向一致性"""
        n = len(factor_values)
        seg_size = n // n_splits
        if seg_size < 50:
            return 1.0
        ics = []
        for i in range(n_splits):
            start = i * seg_size
            end = (i + 1) * seg_size if i < n_splits - 1 else n
            ic = self._compute_ic(factor_values[start:end], returns[start:end])
            ics.append(ic)
        if len(ics) < 2:
            return 1.0
        signs = [1 if ic > 0 else -1 if ic < 0 else 0 for ic in ics]
        if all(s >= 0 for s in signs) or all(s <= 0 for s in signs):
            return 1.0
        pos = sum(1 for s in signs if s > 0)
        neg = sum(1 for s in signs if s < 0)
        return max(pos, neg) / len(signs)

    def _compute_turnover(self, factor_values, window=5):
        """計算因子換手率"""
        fv = np.nan_to_num(factor_values, nan=0.0, posinf=0.0, neginf=0.0)
        if len(fv) < window + 1:
            return 1.0
        n = len(fv)
        rank_changes = []
        for i in range(window, n, window):
            prev_rank = np.argsort(np.argsort(fv[i-window:i])).astype(float)
            curr_rank = np.argsort(np.argsort(fv[i:i+window])).astype(float) if i + window <= n else None
            if curr_rank is not None and len(prev_rank) == len(curr_rank):
                change = np.mean(np.abs(curr_rank - prev_rank)) / window
                rank_changes.append(change)
        if not rank_changes:
            return 0.5
        return float(np.mean(rank_changes))

    def _compute_correlation(self, a, b):
        """計算兩個因子值的相關性"""
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        std_a, std_b = np.std(a), np.std(b)
        if std_a < 1e-10 or std_b < 1e-10:
            return 0.0
        return float(np.abs(np.corrcoef(a, b)[0, 1]))

    def _is_redundant(self, new_values, data):
        """檢查新因子是否和現有因子冗餘"""
        for f in self.discovered_factors:
            try:
                existing = f["tree"].evaluate(data)
                corr = self._compute_correlation(new_values, existing)
                if corr > self.max_correlation:
                    return True
            except Exception:
                pass
        return False

    def _get_all_nodes(self, node):
        nodes = [node]
        for child in node.children:
            nodes.extend(self._get_all_nodes(child))
        return nodes

    def _crossover(self, tree1, tree2):
        new_tree = copy.deepcopy(tree1)
        nodes1 = self._get_all_nodes(new_tree)
        nodes2 = self._get_all_nodes(tree2)
        if len(nodes1) < 2 or len(nodes2) < 1:
            return new_tree
        target = random.choice(nodes1[1:]) if len(nodes1) > 1 else nodes1[0]
        source = random.choice(nodes2)
        target.node_type = source.node_type
        target.value = source.value
        target.children = copy.deepcopy(source.children)
        target.window = source.window
        if new_tree.depth() > self.max_depth + 1:
            return copy.deepcopy(tree1)
        return new_tree

    def _mutate(self, tree):
        new_tree = copy.deepcopy(tree)
        nodes = self._get_all_nodes(new_tree)
        if not nodes:
            return new_tree
        target = random.choice(nodes)
        if target.node_type == "factor":
            target.value = random.choice(BASE_FACTORS)
        elif target.node_type == "const":
            target.value = round(target.value + random.gauss(0, 0.5), 2)
        elif target.node_type == "binary":
            target.value = random.choice(BINARY_OPS)
        elif target.node_type == "unary":
            target.value = random.choice(UNARY_OPS)
        elif target.node_type == "window":
            if random.random() < 0.5:
                target.value = random.choice(WINDOW_OPS)
            else:
                target.window = random.choice(WINDOWS)
        return new_tree

    # ═══════════════════════════════════════════════════
    #  v6.0 勝率導向適應度函數
    # ═══════════════════════════════════════════════════
    def _fitness_score(self, ic, ic_ir, size, hit_rate=0.5,
                       cond_hit_rate=0.5, persistence=0.5,
                       turnover=0.5, ic_stability=1.0,
                       direction_consistency=1.0, multi_hz_ic=0.0):
        """v6.0 勝率導向適應度：
        Hit Rate 和條件勝率佔主導地位（60%），IC 降為輔助（20%）
        """
        # ═══ 勝率相關（60% 權重）═══
        # Hit Rate：基礎方向正確率
        hr_score = (hit_rate - 0.50) * 200  # 50%→0, 52%→4, 55%→10, 60%→20
        hr_score = np.clip(hr_score, -20, 30)

        # 條件勝率：強信號時的勝率（更重要）
        chr_score = (cond_hit_rate - 0.50) * 250  # 50%→0, 52%→5, 55%→12.5
        chr_score = np.clip(chr_score, -25, 40)

        # 信號持續性：信號方向穩定
        persist_score = (persistence - 0.5) * 30  # 0.5→0, 0.7→6, 0.9→12
        persist_score = np.clip(persist_score, -10, 15)

        # ═══ IC 相關（20% 權重）═══
        ic_score = abs(ic) * 40
        multi_hz_score = multi_hz_ic * 30

        # ═══ 穩定性（15% 權重）═══
        ir_score = max(0, ic_ir) * 15
        stability_bonus = ic_stability * 8
        direction_bonus = direction_consistency * 6

        # ═══ 懲罰項（5%）═══
        complexity_penalty = max(0, size - 5) * 1.5
        turnover_penalty = max(0, turnover - 0.6) * 15

        # ═══ 硬性門檻 ═══
        # 條件勝率 < 48% 的因子直接大幅扣分
        if cond_hit_rate < 0.48:
            chr_score -= 30

        return (hr_score + chr_score + persist_score
                + ic_score + multi_hz_score
                + ir_score + stability_bonus + direction_bonus
                - complexity_penalty - turnover_penalty)

    def mine(self, data, returns, close_arr=None):
        """v6.0 因子挖掘主流程"""
        print(f"\n  GP 因子挖掘 v6.0 開始（種群={self.pop_size}, 代數={self.n_generations}, "
              f"Hit Rate門檻={self.hit_rate_threshold:.0%}, IC門檻={self.ic_threshold}）")

        # 如果沒有提供 close_arr，嘗試從 data 中獲取
        if close_arr is None:
            close_arr = data.get("close", np.zeros(len(returns)))

        # 初始種群（包含已有因子的變異版本）
        population = []
        for f in self.discovered_factors[:15]:
            try:
                seed = copy.deepcopy(f["tree"])
                mutated = self._mutate(seed)
                population.append(mutated)
                # 再加一個交叉變異版本
                mutated2 = self._mutate(self._mutate(seed))
                population.append(mutated2)
            except Exception:
                pass
        while len(population) < self.pop_size:
            population.append(self._random_tree())

        seen_exprs = set(f["expr"] for f in self.discovered_factors)
        best_factors = []

        for gen in range(self.n_generations):
            scored = []
            for tree in population:
                values = tree.evaluate(data)
                ic = abs(self._compute_ic(values, returns))
                ic_ir = self._compute_ic_ir(values, returns)
                hit_rate = self._compute_hit_rate(values, returns)
                cond_hit_rate = self._compute_conditional_hit_rate(values, returns)
                persistence = self._compute_signal_persistence(values)
                turnover = self._compute_turnover(values)
                ic_stability = self._compute_ic_stability(values, returns)
                dir_consistency = self._compute_direction_consistency(values, returns)
                multi_hz_ic = self._compute_multi_horizon_ic(values, returns, close_arr)

                fitness = self._fitness_score(
                    ic, ic_ir, tree.size(),
                    hit_rate=hit_rate,
                    cond_hit_rate=cond_hit_rate,
                    persistence=persistence,
                    turnover=turnover,
                    ic_stability=ic_stability,
                    direction_consistency=dir_consistency,
                    multi_hz_ic=multi_hz_ic,
                )
                scored.append((tree, ic, ic_ir, fitness, turnover,
                               ic_stability, dir_consistency,
                               hit_rate, cond_hit_rate, persistence, multi_hz_ic))
            scored.sort(key=lambda x: x[3], reverse=True)

            if (gen + 1) % 10 == 0:
                best = scored[0]
                print(f"    第 {gen+1:>3d} 代 | IC={best[1]:.4f} HR={best[7]:.1%} "
                      f"CHR={best[8]:.1%} pers={best[9]:.2f} "
                      f"fit={best[3]:.1f} | {best[0].to_string()[:45]}")

            # 收集好因子（v6.0 勝率導向門檻）
            for (tree, ic, ic_ir, fitness, turnover, stab, dir_c,
                 hr, chr_, pers, mhz_ic) in scored[:20]:
                expr = tree.to_string()
                # v6.0：條件勝率 > 52% 或 (IC > 0.03 且 HR > 50%)
                passes_hr = chr_ >= self.hit_rate_threshold
                passes_ic = ic >= self.ic_threshold and hr >= 0.50
                if (passes_hr or passes_ic) and expr not in seen_exprs:
                    if tree.node_type != "factor" and tree.size() > 1:
                        values = tree.evaluate(data)
                        if not self._is_redundant(values, data):
                            seen_exprs.add(expr)
                            best_factors.append({
                                "tree": tree, "expr": expr,
                                "ic": ic, "ic_ir": ic_ir,
                                "hit_rate": hr,
                                "cond_hit_rate": chr_,
                                "persistence": pers,
                                "fitness": fitness, "turnover": turnover,
                                "ic_stability": stab,
                                "direction_consistency": dir_c,
                                "multi_hz_ic": mhz_ic,
                            })

            # 進化
            new_population = []
            elite_count = max(5, self.pop_size // 12)
            for tree, _, _, _, _, _, _, _, _, _, _ in scored[:elite_count]:
                new_population.append(tree)
            while len(new_population) < self.pop_size:
                candidates = random.sample(scored, min(self.tournament_size, len(scored)))
                parent1 = max(candidates, key=lambda x: x[3])[0]
                candidates = random.sample(scored, min(self.tournament_size, len(scored)))
                parent2 = max(candidates, key=lambda x: x[3])[0]
                child = (self._crossover(parent1, parent2)
                         if random.random() < self.crossover_rate
                         else copy.deepcopy(parent1))
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                new_population.append(child)
            population = new_population

        # 去重並排序（v6.0：以條件勝率為主排序）
        unique = {}
        for f in best_factors:
            if f["expr"] not in unique or f["fitness"] > unique[f["expr"]]["fitness"]:
                unique[f["expr"]] = f
        new_factors = sorted(unique.values(), key=lambda x: x["fitness"],
                             reverse=True)[:self.max_new_factors]

        timestamp = datetime.now().strftime("%m%d%H%M")
        result = []
        for i, f in enumerate(new_factors):
            name = f"gp_{timestamp}_{i}"
            factor = {"name": name, "expr": f["expr"], "tree": f["tree"],
                      "ic": f["ic"], "ic_ir": f.get("ic_ir", 0),
                      "hit_rate": f.get("hit_rate", 0.5),
                      "cond_hit_rate": f.get("cond_hit_rate", 0.5),
                      "persistence": f.get("persistence", 0.5),
                      "turnover": f.get("turnover", 0),
                      "ic_stability": f.get("ic_stability", 0)}
            result.append(factor)
            self.discovered_factors.append(factor)
            print(f"    新因子: {name} | IC={f['ic']:.4f} HR={f.get('hit_rate',0.5):.1%} "
                  f"CHR={f.get('cond_hit_rate',0.5):.1%} pers={f.get('persistence',0.5):.2f} "
                  f"| {f['expr'][:55]}")

        # 淘汰低質量因子（保留品質最高的 20 個）
        self._prune_factors(data, returns, close_arr)
        self._save_factors()
        print(f"  GP 挖掘完成，發現 {len(result)} 個新因子，"
              f"因子庫共 {len(self.discovered_factors)} 個")
        return result

    def _prune_factors(self, data, returns, close_arr, max_factors=20):
        """v6.0 因子庫動態管理：重新評估所有因子，淘汰表現差的"""
        if len(self.discovered_factors) <= max_factors:
            return

        scored = []
        for f in self.discovered_factors:
            try:
                values = f["tree"].evaluate(data)
                ic = abs(self._compute_ic(values, returns))
                hr = self._compute_hit_rate(values, returns)
                chr_ = self._compute_conditional_hit_rate(values, returns)
                pers = self._compute_signal_persistence(values)
                # 更新因子的最新指標
                f["ic"] = ic
                f["hit_rate"] = hr
                f["cond_hit_rate"] = chr_
                f["persistence"] = pers
                score = self._fitness_score(
                    ic, f.get("ic_ir", 0), f["tree"].size(),
                    hit_rate=hr, cond_hit_rate=chr_, persistence=pers,
                    turnover=f.get("turnover", 0.5),
                    ic_stability=f.get("ic_stability", 0.5))
                scored.append((f, score))
            except Exception:
                scored.append((f, -999))

        scored.sort(key=lambda x: x[1], reverse=True)
        self.discovered_factors = [f for f, _ in scored[:max_factors]]

    def compute_gp_factors(self, data):
        gp_data = {}
        for f in self.discovered_factors:
            try:
                values = f["tree"].evaluate(data)
                gp_data[f["name"]] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                pass
        return gp_data

    def get_factor_names(self):
        return [f["name"] for f in self.discovered_factors]

    def get_top_factors(self, n=8):
        sorted_f = sorted(
            self.discovered_factors,
            key=lambda x: x.get("cond_hit_rate", 0.5),
            reverse=True
        )
        return [f["name"] for f in sorted_f[:n]]
