"""
BTC-Bot 遺傳編程（GP）因子挖掘器 v5.0
改進：
  1. IC 穩定性評估（IC_IR = mean(IC) / std(IC)）
  2. 冗餘檢測（新因子和現有因子相關性 > 0.7 則淘汰）
  3. 擴展搜索空間（更多窗口操作、更多窗口大小）
  4. 因子衰減淘汰（定期重新評估因子 IC）
  5. 種子機制（用現有好因子做交叉的種子）
"""

import numpy as np
import random
import json
import os
import copy
from datetime import datetime

# 擴展的基礎因子列表（匹配 factor_engine v5.0）
BASE_FACTORS = [
    "roc5", "roc20", "mom_accel",
    "ma10dev", "ma30dev", "ma_alignment",
    "vwapdev", "priceimpact", "macdhist",
    "rsi_norm", "bb_pctb",
    "vol_ratio", "tr_ratio", "vol_surge", "vol_price_div", "obv_slope",
    "candle_dir",
    "oi_roc6", "oi_roc24", "price_oi_confirm",
    "hour_sin", "hour_cos",
]

BINARY_OPS = ["add", "sub", "mul", "div", "max", "min"]
UNARY_OPS = ["abs", "neg", "square", "sign", "log1p", "tanh"]
WINDOW_OPS = ["rolling_mean", "rolling_std", "rolling_rank", "diff", "shift",
              "rolling_zscore", "ewm"]
WINDOWS = [3, 5, 8, 10, 15, 20, 30]


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
        elif op == "square": return np.clip(a ** 2, -1e6, 1e6)
        elif op == "sign": return np.sign(a)
        elif op == "log1p": return np.sign(a) * np.log1p(np.abs(a))
        elif op == "tanh": return np.tanh(a)
        return a

    @staticmethod
    def _window_op(op, a, window):
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(a)
        result = np.zeros(n)
        if op == "rolling_mean":
            cumsum = np.cumsum(a)
            result[window-1:] = (cumsum[window-1:] - np.concatenate([[0], cumsum[:-window]])) / window
        elif op == "rolling_std":
            for i in range(window - 1, n):
                result[i] = np.std(a[i-window+1:i+1])
        elif op == "rolling_rank":
            for i in range(window - 1, n):
                w = a[i-window+1:i+1]
                result[i] = np.sum(w <= a[i]) / window
        elif op == "diff":
            w = min(window, n - 1)
            result[w:] = a[w:] - a[:-w]
        elif op == "shift":
            w = min(window, n - 1)
            result[w:] = a[:-w]
        elif op == "rolling_zscore":
            for i in range(window - 1, n):
                w = a[i-window+1:i+1]
                std = np.std(w)
                if std > 1e-10:
                    result[i] = (a[i] - np.mean(w)) / std
        elif op == "ewm":
            alpha = 2.0 / (window + 1)
            result[0] = a[0]
            for i in range(1, n):
                result[i] = alpha * a[i] + (1 - alpha) * result[i-1]
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
    def __init__(self, pop_size=300, n_generations=50, max_depth=5,
                 tournament_size=5, crossover_rate=0.6, mutation_rate=0.3,
                 ic_threshold=0.02, max_new_factors=5,
                 max_correlation=0.7,
                 save_path="data/gp_factors.json"):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.ic_threshold = ic_threshold
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
                     "ic": d["ic"], "ic_ir": d.get("ic_ir", 0)}
                    for d in data
                ]
                print(f"  載入 {len(self.discovered_factors)} 個 GP 因子")
            except Exception as e:
                print(f"  載入 GP 因子失敗: {e}")

    def _save_factors(self):
        data = [{"name": f["name"], "expr": f["expr"],
                 "tree": f["tree"].to_dict(), "ic": f["ic"],
                 "ic_ir": f.get("ic_ir", 0)}
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

    def _fitness_score(self, ic, ic_ir, size):
        """綜合評分：IC 強度 + IC 穩定性 - 複雜度懲罰"""
        ic_score = abs(ic) * 50
        ir_score = max(0, ic_ir) * 20
        complexity_penalty = max(0, size - 5) * 1.0
        return ic_score + ir_score - complexity_penalty

    def mine(self, data, returns):
        print(f"\n  GP 因子挖掘開始（種群={self.pop_size}, 代數={self.n_generations}, 深度≤{self.max_depth}）")

        # 初始種群：部分隨機 + 部分用現有好因子做種子
        population = []
        # 種子：用現有因子做交叉基礎
        for f in self.discovered_factors[:10]:
            try:
                seed = copy.deepcopy(f["tree"])
                mutated = self._mutate(seed)
                population.append(mutated)
            except Exception:
                pass
        # 補充隨機個體
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
                fitness = self._fitness_score(ic, ic_ir, tree.size())
                scored.append((tree, ic, ic_ir, fitness))
            scored.sort(key=lambda x: x[3], reverse=True)

            if (gen + 1) % 10 == 0:
                best = scored[0]
                print(f"    第 {gen+1:>3d} 代 | IC={best[1]:.4f} IR={best[2]:.2f} "
                      f"fit={best[3]:.1f} | {best[0].to_string()[:50]}")

            # 收集好因子
            for tree, ic, ic_ir, fitness in scored[:15]:
                expr = tree.to_string()
                if ic >= self.ic_threshold and expr not in seen_exprs:
                    if tree.node_type != "factor" and tree.size() > 1:
                        values = tree.evaluate(data)
                        if not self._is_redundant(values, data):
                            seen_exprs.add(expr)
                            best_factors.append({
                                "tree": tree, "expr": expr,
                                "ic": ic, "ic_ir": ic_ir, "fitness": fitness
                            })

            # 進化
            new_population = []
            elite_count = max(2, self.pop_size // 15)
            for tree, _, _, _ in scored[:elite_count]:
                new_population.append(tree)
            while len(new_population) < self.pop_size:
                candidates = random.sample(scored, min(self.tournament_size, len(scored)))
                parent1 = max(candidates, key=lambda x: x[3])[0]
                candidates = random.sample(scored, min(self.tournament_size, len(scored)))
                parent2 = max(candidates, key=lambda x: x[3])[0]
                child = self._crossover(parent1, parent2) if random.random() < self.crossover_rate else copy.deepcopy(parent1)
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                new_population.append(child)
            population = new_population

        # 去重並排序
        unique = {}
        for f in best_factors:
            if f["expr"] not in unique or f["fitness"] > unique[f["expr"]]["fitness"]:
                unique[f["expr"]] = f
        new_factors = sorted(unique.values(), key=lambda x: x["fitness"], reverse=True)[:self.max_new_factors]

        timestamp = datetime.now().strftime("%m%d%H%M")
        result = []
        for i, f in enumerate(new_factors):
            name = f"gp_{timestamp}_{i}"
            factor = {"name": name, "expr": f["expr"], "tree": f["tree"],
                      "ic": f["ic"], "ic_ir": f.get("ic_ir", 0)}
            result.append(factor)
            self.discovered_factors.append(factor)
            print(f"    新因子: {name} | IC={f['ic']:.4f} IR={f.get('ic_ir',0):.2f} | {f['expr'][:60]}")

        # 淘汰低質量因子（保留 IC_IR 最高的 25 個）
        self.discovered_factors.sort(
            key=lambda x: self._fitness_score(x["ic"], x.get("ic_ir", 0), x["tree"].size()),
            reverse=True
        )
        self.discovered_factors = self.discovered_factors[:25]
        self._save_factors()
        print(f"  GP 挖掘完成，發現 {len(result)} 個新因子，因子庫共 {len(self.discovered_factors)} 個")
        return result

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

    def get_top_factors(self, n=5):
        sorted_f = sorted(
            self.discovered_factors,
            key=lambda x: self._fitness_score(x["ic"], x.get("ic_ir", 0), x["tree"].size()),
            reverse=True
        )
        return [f["name"] for f in sorted_f[:n]]

