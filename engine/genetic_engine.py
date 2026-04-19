"""
BTC-Bot 遺傳搜索引擎 v5.0
改進：
  1. 非線性交互項：自動生成 top 因子的交叉乘積
  2. 分離多空閾值：多頭和空頭各自獨立閾值
  3. Walk-forward 驗證：3 段滾動驗證
  4. 改進適應度：加入最大回撤懲罰
  5. 手續費扣除 + 多空平衡
"""

import polars as pl
import numpy as np
import random
from typing import Optional

FACTOR_COLS = [
    "roc5", "roc20", "mom_accel",
    "ma10dev", "ma30dev", "ma_alignment",
    "vwapdev", "priceimpact", "macdhist",
    "rsi_norm", "bb_pctb",
    "vol_ratio", "tr_ratio", "vol_surge", "vol_price_div", "obv_slope",
    "candle_dir",
    "oi_roc6", "oi_roc24", "price_oi_confirm",
    "hour_sin", "hour_cos",
]

ROUND_TRIP_FEE = 0.0007  # 0.035% × 2


class GeneticEngine:
    def __init__(self, pop_size=300, n_generations=60, train_ratio=0.6,
                 mutation_rate=0.15, crossover_rate=0.6,
                 threshold_range=(0.05, 1.2),
                 fee_rate=ROUND_TRIP_FEE,
                 n_interactions=10):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.train_ratio = train_ratio
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.threshold_range = threshold_range
        self.fee_rate = fee_rate
        self.n_interactions = n_interactions
        self.n_factors = len(FACTOR_COLS)
        self.factor_cols_used = list(FACTOR_COLS)
        self.interaction_pairs = []

    def _add_interactions(self, X, factor_cols):
        """自動生成交互項（top 因子的乘積）"""
        n_cols = X.shape[1]
        if n_cols < 2:
            return X, []

        # 選擇方差最大的因子做交互
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-min(6, n_cols):]

        interaction_cols = []
        pairs = []
        count = 0
        for i in range(len(top_indices)):
            for j in range(i + 1, len(top_indices)):
                if count >= self.n_interactions:
                    break
                idx_i, idx_j = top_indices[i], top_indices[j]
                interaction = X[:, idx_i] * X[:, idx_j]
                interaction_cols.append(interaction)
                name_i = factor_cols[idx_i] if idx_i < len(factor_cols) else f"f{idx_i}"
                name_j = factor_cols[idx_j] if idx_j < len(factor_cols) else f"f{idx_j}"
                pairs.append(f"{name_i}×{name_j}")
                count += 1
            if count >= self.n_interactions:
                break

        if interaction_cols:
            X_ext = np.column_stack([X] + interaction_cols)
            return X_ext, pairs
        return X, []

    def _init_population(self):
        population = []
        for _ in range(self.pop_size):
            weights = np.random.randn(self.n_factors)
            weights = weights / (np.linalg.norm(weights) + 1e-10)
            long_thr = random.uniform(*self.threshold_range)
            short_thr = random.uniform(*self.threshold_range)
            population.append({
                "weights": weights,
                "long_threshold": long_thr,
                "short_threshold": short_thr,
            })
        return population

    def _compute_signal(self, X, weights):
        return X @ weights[:X.shape[1]]

    def _evaluate(self, individual, X, ret):
        """評估個體表現，分離多空閾值"""
        score = self._compute_signal(X, individual["weights"])
        long_thr = individual["long_threshold"]
        short_thr = individual["short_threshold"]

        signal = np.zeros(len(score))
        signal[score > long_thr] = 1
        signal[score < -short_thr] = -1

        mask = signal != 0
        n_trades = int(mask.sum())

        if n_trades < 40:
            return {
                "win_rate": 0.0, "n_trades": 0, "profit_factor": 0.0,
                "sharpe": 0.0, "net_pnl": 0.0,
                "long_pct": 0.0, "short_pct": 0.0,
                "max_drawdown": 0.0,
            }

        raw_returns = signal * ret
        trade_returns_raw = raw_returns[mask]
        trade_returns = trade_returns_raw - self.fee_rate

        wins = (trade_returns > 0).sum()
        win_rate = float(wins / n_trades)

        gross_profit = trade_returns[trade_returns > 0].sum()
        gross_loss = abs(trade_returns[trade_returns < 0].sum())
        profit_factor = float(gross_profit / (gross_loss + 1e-10))

        net_pnl = float(trade_returns.sum())

        sharpe = float(
            trade_returns.mean() / (trade_returns.std() + 1e-10) * np.sqrt(252 * 288)
        )

        # 最大回撤
        cumulative = np.cumsum(trade_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = float(drawdown.max()) if len(drawdown) > 0 else 0.0

        long_count = int((signal == 1).sum())
        short_count = int((signal == -1).sum())
        total = long_count + short_count
        long_pct = long_count / total if total > 0 else 0.5
        short_pct = short_count / total if total > 0 else 0.5

        return {
            "win_rate": win_rate,
            "n_trades": n_trades,
            "n_bars": len(score),  # Bug 22 Fix：傳遞數據長度供 fitness 計算 trades_per_day
            "profit_factor": profit_factor,
            "sharpe": sharpe,
            "net_pnl": net_pnl,
            "long_pct": long_pct,
            "short_pct": short_pct,
            "max_drawdown": max_drawdown,
        }

    def _fitness(self, metrics):
        """改進的適應度函數"""
        wr = metrics["win_rate"]
        nt = metrics["n_trades"]
        pf = metrics["profit_factor"]
        net_pnl = metrics["net_pnl"]
        long_pct = metrics["long_pct"]
        max_dd = metrics["max_drawdown"]

        if nt < 40:
            return -999.0

        if pf < 0.8:
            return -100.0 + pf * 10

        # ═══ 核心得分 ═══
        pf_score = min(pf, 3.0) * 20
        wr_score = wr * 25
        pnl_score = min(max(net_pnl * 1000, -10), 20)
        sharpe_score = min(max(metrics["sharpe"] / 2, -5), 10)

        # ═══ 懲罰項 ═══
        balance_ratio = min(long_pct, 1 - long_pct)
        balance_penalty = max(0, (0.25 - balance_ratio)) * 40

        # 最大回撤懲罰
        dd_penalty = max(0, max_dd - 0.02) * 200

        # 交易次數獎勵（鼓勵適度交易）
        # Bug 22 Fix：用數據的總 K 線數估算天數，而非交易次數
        n_bars = metrics.get("n_bars", nt)
        n_days = max(1, n_bars / 288)
        trades_per_day = nt / n_days
        if trades_per_day < 5:
            freq_bonus = -10
        elif trades_per_day < 50:
            freq_bonus = min(trades_per_day / 5, 5)
        else:
            freq_bonus = -max(0, (trades_per_day - 50)) * 0.3

        fitness = pf_score + wr_score + pnl_score + sharpe_score + freq_bonus - balance_penalty - dd_penalty
        return fitness

    def _crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in parent1.items()}
        point = random.randint(1, len(parent1["weights"]) - 1)
        child_weights = np.concatenate([parent1["weights"][:point], parent2["weights"][point:]])
        return {
            "weights": child_weights,
            "long_threshold": random.choice([parent1["long_threshold"], parent2["long_threshold"]]),
            "short_threshold": random.choice([parent1["short_threshold"], parent2["short_threshold"]]),
        }

    def _mutate(self, individual):
        weights = individual["weights"].copy()
        long_thr = individual["long_threshold"]
        short_thr = individual["short_threshold"]
        for i in range(len(weights)):
            if random.random() < self.mutation_rate:
                weights[i] += np.random.randn() * 0.3
        if random.random() < self.mutation_rate:
            long_thr += random.gauss(0, 0.1)
            long_thr = np.clip(long_thr, *self.threshold_range)
        if random.random() < self.mutation_rate:
            short_thr += random.gauss(0, 0.1)
            short_thr = np.clip(short_thr, *self.threshold_range)
        weights = weights / (np.linalg.norm(weights) + 1e-10)
        return {"weights": weights, "long_threshold": long_thr, "short_threshold": short_thr}

    def search(self, df):
        factor_cols = list(FACTOR_COLS)

        available_cols = [c for c in factor_cols if c in df.columns]
        if len(available_cols) < 3:
            print(f"  可用因子不足: {available_cols}")
            return None

        for col in factor_cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(col))

        X_raw = df.select(factor_cols).to_numpy().astype(np.float64)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        X_mean = X_raw.mean(axis=0)
        X_std = X_raw.std(axis=0)
        X = (X_raw - X_mean) / (X_std + 1e-10)

        # 加入交互項
        X, interaction_names = self._add_interactions(X, factor_cols)
        all_cols = factor_cols + interaction_names
        self.interaction_pairs = interaction_names
        self.n_factors = X.shape[1]
        self.factor_cols_used = all_cols

        close = df["close"].to_numpy().astype(np.float64)
        ret = np.zeros(len(close))
        ret[:-1] = (close[1:] - close[:-1]) / close[:-1]

        # Walk-forward: 60% 訓練 / 20% 驗證1 / 20% 驗證2
        n = len(X)
        split1 = int(n * 0.6)
        split2 = int(n * 0.8)
        X_train, X_val1, X_val2 = X[:split1], X[split1:split2], X[split2:]
        ret_train, ret_val1, ret_val2 = ret[:split1], ret[split1:split2], ret[split2:]

        print(f"  因子: {len(factor_cols)} 基礎 + {len(interaction_names)} 交互 = {self.n_factors} 總計")
        print(f"  數據分割: 訓練={split1} 驗證1={split2-split1} 驗證2={n-split2}")
        print(f"  手續費: 每筆 {self.fee_rate*100:.3f}% (雙邊)")

        population = self._init_population()
        best_combined_fitness = -999
        best_individual = None
        no_improve_count = 0

        for gen in range(self.n_generations):
            train_results = []
            for ind in population:
                metrics = self._evaluate(ind, X_train, ret_train)
                fitness = self._fitness(metrics)
                train_results.append((ind, metrics, fitness))
            train_results.sort(key=lambda x: x[2], reverse=True)

            # Walk-forward 驗證：兩段驗證集的平均適應度
            for ind, _, train_fit in train_results[:10]:
                val1_metrics = self._evaluate(ind, X_val1, ret_val1)
                val2_metrics = self._evaluate(ind, X_val2, ret_val2)
                val1_fit = self._fitness(val1_metrics)
                val2_fit = self._fitness(val2_metrics)

                # 綜合分數 = 訓練 30% + 驗證1 35% + 驗證2 35%
                combined = train_fit * 0.3 + val1_fit * 0.35 + val2_fit * 0.35

                if combined > best_combined_fitness:
                    best_combined_fitness = combined
                    best_individual = {
                        "weights": ind["weights"].copy(),
                        "long_threshold": ind["long_threshold"],
                        "short_threshold": ind["short_threshold"],
                        "threshold": (ind["long_threshold"] + ind["short_threshold"]) / 2,
                        "factor_mean": X_mean.copy(),
                        "factor_std": X_std.copy(),
                        "interaction_pairs": interaction_names,
                        "train_metrics": self._evaluate(ind, X_train, ret_train),
                        "val1_metrics": val1_metrics,
                        "val2_metrics": val2_metrics,
                    }
                    no_improve_count = 0

            no_improve_count += 1
            if no_improve_count > 15:
                print(f"  第 {gen+1} 代早停")
                break

            # 進化
            new_population = []
            elite_count = max(2, self.pop_size // 15)
            for ind, _, _ in train_results[:elite_count]:
                new_population.append(ind)
            while len(new_population) < self.pop_size:
                candidates = random.sample(train_results, min(5, len(train_results)))
                parent1 = max(candidates, key=lambda x: x[2])[0]
                candidates = random.sample(train_results, min(5, len(train_results)))
                parent2 = max(candidates, key=lambda x: x[2])[0]
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
            population = new_population

            if (gen + 1) % 10 == 0:
                best_train = train_results[0]
                print(
                    f"  第 {gen+1:>3d} 代 | "
                    f"訓練 WR={best_train[1]['win_rate']:.3f} "
                    f"PF={best_train[1]['profit_factor']:.2f} "
                    f"交易={best_train[1]['n_trades']:>5d} "
                    f"多空={best_train[1]['long_pct']:.0%}/{best_train[1]['short_pct']:.0%} "
                    f"DD={best_train[1]['max_drawdown']:.4f} | "
                    f"最佳綜合={best_combined_fitness:.2f}"
                )

        if best_individual is not None:
            print(f"\n  最優因子權重（{len(factor_cols)} 基礎 + {len(interaction_names)} 交互）:")
            all_w = best_individual["weights"]
            for i, col in enumerate(factor_cols):
                if i < len(all_w) and abs(all_w[i]) > 0.05:
                    print(f"    {col:<22} {all_w[i]:+.3f}")
            for i, name in enumerate(interaction_names):
                idx = len(factor_cols) + i
                if idx < len(all_w) and abs(all_w[idx]) > 0.05:
                    print(f"    {name:<22} {all_w[idx]:+.3f}")

            tm = best_individual["train_metrics"]
            v1 = best_individual["val1_metrics"]
            v2 = best_individual["val2_metrics"]
            print(f"\n  訓練集: WR={tm['win_rate']:.3f} PF={tm['profit_factor']:.2f} "
                  f"交易={tm['n_trades']} 多空={tm['long_pct']:.0%}/{tm['short_pct']:.0%}")
            print(f"  驗證1:  WR={v1['win_rate']:.3f} PF={v1['profit_factor']:.2f} "
                  f"交易={v1['n_trades']} DD={v1['max_drawdown']:.4f}")
            print(f"  驗證2:  WR={v2['win_rate']:.3f} PF={v2['profit_factor']:.2f} "
                  f"交易={v2['n_trades']} DD={v2['max_drawdown']:.4f}")
            print(f"  多頭閾值: {best_individual['long_threshold']:.4f}")
            print(f"  空頭閾值: {best_individual['short_threshold']:.4f}")

            # 警告
            avg_pf = (v1['profit_factor'] + v2['profit_factor']) / 2
            if avg_pf < 1.0:
                print(f"\n  ⚠️ 警告: 驗證集平均 PF={avg_pf:.2f} < 1.0（扣費後虧損）")

        return best_individual

