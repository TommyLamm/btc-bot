"""
BTC-Bot 遺傳搜索引擎 v3.1 — 高頻版
改進：
  1. 手續費扣除：每筆交易扣除 0.07% 雙邊成本
  2. 多空平衡懲罰
  3. 閾值搜索範圍 (0.1, 1.5)，配合高頻交易策略
"""

import polars as pl
import numpy as np
import random
from typing import Optional

FACTOR_COLS = [
    "roc5", "roc20", "ma10dev", "ma30dev",
    "vwapdev", "priceimpact", "macdhist",
    "oi_roc6", "oi_roc24", "price_oi_confirm",
    "vol_impact_hl", "candle_dir",
]

ROUND_TRIP_FEE = 0.0007


class GeneticEngine:
    def __init__(self, pop_size=300, n_generations=60, train_ratio=0.7,
                 mutation_rate=0.15, crossover_rate=0.6,
                 threshold_range=(0.1, 1.5),
                 fee_rate=ROUND_TRIP_FEE):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.train_ratio = train_ratio
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.threshold_range = threshold_range
        self.fee_rate = fee_rate
        self.n_factors = len(FACTOR_COLS)
        self.factor_cols_used = list(FACTOR_COLS)

    def _init_population(self):
        population = []
        for _ in range(self.pop_size):
            weights = np.random.randn(self.n_factors)
            weights = weights / (np.linalg.norm(weights) + 1e-10)
            threshold = random.uniform(*self.threshold_range)
            population.append({"weights": weights, "threshold": threshold})
        return population

    def _compute_signal(self, X, weights, threshold):
        score = X @ weights
        signal = np.zeros(len(score))
        signal[score > threshold] = 1
        signal[score < -threshold] = -1
        return signal

    def _evaluate(self, individual, X, ret):
        signal = self._compute_signal(X, individual["weights"], individual["threshold"])
        mask = signal != 0
        n_trades = int(mask.sum())

        if n_trades < 50:
            return {
                "win_rate": 0.0, "n_trades": 0, "profit_factor": 0.0,
                "sharpe": 0.0, "net_pnl": 0.0,
                "long_pct": 0.0, "short_pct": 0.0,
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

        long_count = int((signal == 1).sum())
        short_count = int((signal == -1).sum())
        total = long_count + short_count
        long_pct = long_count / total if total > 0 else 0.5
        short_pct = short_count / total if total > 0 else 0.5

        return {
            "win_rate": win_rate,
            "n_trades": n_trades,
            "profit_factor": profit_factor,
            "sharpe": sharpe,
            "net_pnl": net_pnl,
            "long_pct": long_pct,
            "short_pct": short_pct,
        }

    def _fitness(self, metrics):
        wr = metrics["win_rate"]
        nt = metrics["n_trades"]
        pf = metrics["profit_factor"]
        net_pnl = metrics["net_pnl"]
        long_pct = metrics["long_pct"]

        if nt < 50:
            return -999.0

        if pf < 1.0:
            return -100.0 + pf * 10

        pf_score = min(pf, 3.0) * 20
        wr_score = wr * 30
        pnl_score = min(max(net_pnl * 1000, -10), 20)
        sharpe_score = min(max(metrics["sharpe"] / 2, -5), 10)

        balance_ratio = min(long_pct, 1 - long_pct)
        balance_penalty = max(0, (0.3 - balance_ratio)) * 30

        n_days = max(1, nt / 288)
        trades_per_day = nt / n_days
        overtrade_penalty = max(0, (trades_per_day - 20)) * 0.5

        fitness = pf_score + wr_score + pnl_score + sharpe_score - balance_penalty - overtrade_penalty
        return fitness

    def _crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1.copy()
        point = random.randint(1, self.n_factors - 1)
        child_weights = np.concatenate([parent1["weights"][:point], parent2["weights"][point:]])
        return {
            "weights": child_weights,
            "threshold": random.choice([parent1["threshold"], parent2["threshold"]]),
        }

    def _mutate(self, individual):
        weights = individual["weights"].copy()
        threshold = individual["threshold"]
        for i in range(self.n_factors):
            if random.random() < self.mutation_rate:
                weights[i] += np.random.randn() * 0.3
        if random.random() < self.mutation_rate:
            threshold += random.gauss(0, 0.15)
            threshold = np.clip(threshold, *self.threshold_range)
        weights = weights / (np.linalg.norm(weights) + 1e-10)
        return {"weights": weights, "threshold": threshold}

    def search(self, df):
        factor_cols = list(FACTOR_COLS)
        self.n_factors = len(factor_cols)
        self.factor_cols_used = factor_cols

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

        close = df["close"].to_numpy().astype(np.float64)
        ret = np.zeros(len(close))
        ret[:-1] = (close[1:] - close[:-1]) / close[:-1]

        split_idx = int(len(X) * self.train_ratio)
        X_train, X_val = X[:split_idx], X[split_idx:]
        ret_train, ret_val = ret[:split_idx], ret[split_idx:]

        sample_weights = np.random.randn(self.n_factors)
        sample_weights = sample_weights / (np.linalg.norm(sample_weights) + 1e-10)
        sample_signals = X_train @ sample_weights
        print(f"  信號分佈參考: P10={np.percentile(np.abs(sample_signals), 10):.3f} "
              f"P50={np.percentile(np.abs(sample_signals), 50):.3f} "
              f"P90={np.percentile(np.abs(sample_signals), 90):.3f} "
              f"Max={np.abs(sample_signals).max():.3f}")
        print(f"  手續費: 每筆 {self.fee_rate*100:.3f}% (雙邊)")

        population = self._init_population()
        best_val_fitness = -999
        best_individual = None
        no_improve_count = 0

        for gen in range(self.n_generations):
            train_results = []
            for ind in population:
                metrics = self._evaluate(ind, X_train, ret_train)
                fitness = self._fitness(metrics)
                train_results.append((ind, metrics, fitness))
            train_results.sort(key=lambda x: x[2], reverse=True)

            for ind, _, _ in train_results[:10]:
                val_metrics = self._evaluate(ind, X_val, ret_val)
                val_fitness = self._fitness(val_metrics)
                if val_fitness > best_val_fitness:
                    best_val_fitness = val_fitness
                    best_individual = {
                        "weights": ind["weights"].copy(),
                        "threshold": ind["threshold"],
                        "factor_mean": X_mean.copy(),
                        "factor_std": X_std.copy(),
                        "train_metrics": self._evaluate(ind, X_train, ret_train),
                        "val_metrics": val_metrics,
                    }
                    no_improve_count = 0

            no_improve_count += 1
            if no_improve_count > 15:
                print(f"  第 {gen+1} 代早停")
                break

            new_population = []
            elite_count = max(2, self.pop_size // 20)
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
                    f"訓練勝率={best_train[1]['win_rate']:.3f} "
                    f"PF={best_train[1]['profit_factor']:.2f} "
                    f"交易={best_train[1]['n_trades']:>5d} "
                    f"多空={best_train[1]['long_pct']:.0%}/{best_train[1]['short_pct']:.0%} | "
                    f"最佳驗證適應度={best_val_fitness:.2f}"
                )

        if best_individual is not None:
            print(f"\n  最優因子權重（{len(factor_cols)} 個因子）:")
            for col, w in zip(factor_cols, best_individual["weights"]):
                if abs(w) > 0.05:
                    print(f"    {col:<22} {w:+.3f}")

            tm = best_individual["train_metrics"]
            vm = best_individual["val_metrics"]
            print(f"\n  訓練集: 勝率={tm['win_rate']:.3f} PF={tm['profit_factor']:.2f} "
                  f"交易={tm['n_trades']} 淨PnL={tm['net_pnl']:+.4f} "
                  f"多空={tm['long_pct']:.0%}/{tm['short_pct']:.0%}")
            print(f"  驗證集: 勝率={vm['win_rate']:.3f} PF={vm['profit_factor']:.2f} "
                  f"交易={vm['n_trades']} 淨PnL={vm['net_pnl']:+.4f} "
                  f"多空={vm['long_pct']:.0%}/{vm['short_pct']:.0%}")
            print(f"  搜索閾值: {best_individual['threshold']:.4f}")

            if vm['profit_factor'] < 1.0:
                print(f"\n  ⚠️ 警告: 驗證集 PF={vm['profit_factor']:.2f} < 1.0（扣費後虧損）")
                print(f"  建議: 降低交易頻率或等待更好的市場條件")

        return best_individual
