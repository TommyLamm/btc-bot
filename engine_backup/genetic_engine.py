import polars as pl
import numpy as np
from deap import base, creator, tools, algorithms
import random
import warnings
warnings.filterwarnings("ignore")

FACTOR_COLS = [
    "roc_5", "roc_20",
    "ma10_dev", "ma30_dev",
    "vwap_dev", "price_impact",
    "macd_hist",
    "oi_roc_6", "oi_roc_24",
    "price_oi_confirm",
    "vol_impact_hl", "candle_dir",
]
N_FACTORS = len(FACTOR_COLS)

MIN_TRADES = 500   # 验证集至少500笔，才有统计意义


def compute_metrics(weights, threshold, X, y):
    signals    = X @ weights
    long_mask  = signals >  threshold
    short_mask = signals < -threshold
    n_trades   = long_mask.sum() + short_mask.sum()

    if n_trades < MIN_TRADES:
        return 0.0, -99.0, n_trades

    all_ret = np.concatenate([y[long_mask], -y[short_mask]])
    winrate = (all_ret > 0).mean()
    sharpe  = all_ret.mean() / (all_ret.std() + 1e-10) * np.sqrt(105120)
    return winrate, sharpe, n_trades


def evaluate(individual, X_train, y_train, X_val, y_val):
    weights   = np.array(individual[:N_FACTORS])
    # 阈值限制在 0.1~0.8，防止过于稀疏或过于密集
    threshold = np.clip(abs(individual[N_FACTORS]), 0.1, 0.8)

    wr_train, sh_train, n_train = compute_metrics(weights, threshold, X_train, y_train)
    wr_val,   sh_val,   n_val   = compute_metrics(weights, threshold, X_val,   y_val)

    # 交易次数不足直接淘汰
    if n_train < MIN_TRADES or n_val < MIN_TRADES:
        return (0.0, 0.0, 0.0)

    # 过拟合惩罚：训练验证胜率差距
    overfit = max(0.0, wr_train - wr_val - 0.03) * 3.0

    # 复杂度惩罚
    complexity = (np.abs(weights) > 0.05).sum() * 0.005

    score = wr_val * 0.6 + wr_train * 0.4 - overfit - complexity
    return (score, sh_val, float(n_val))


def prepare_data(df, lookback=20000):
    df_recent = df.tail(lookback).drop_nulls()
    X_all = df_recent.select(FACTOR_COLS).to_numpy().astype(np.float32)
    y_all = df_recent["close"].pct_change().shift(-1).to_numpy()

    split = int(len(X_all) * 0.6)
    X_mean = np.nanmean(X_all[:split], axis=0)
    X_std  = np.nanstd(X_all[:split],  axis=0) + 1e-10

    X_norm = (X_all - X_mean) / X_std
    X_norm = np.nan_to_num(X_norm)
    y_all  = np.nan_to_num(y_all)

    return (X_norm[:split], y_all[:split],
            X_norm[split:], y_all[split:],
            X_mean, X_std)


def run_genetic_search(df, lookback=20000, n_gen=60, pop_size=300, verbose=True):

    X_train, y_train, X_val, y_val, X_mean, X_std = prepare_data(df, lookback)

    if verbose:
        print(f"训练集: {len(X_train)} 根 | 验证集: {len(X_val)} 根")
        print(f"最低交易次数要求: {MIN_TRADES} 笔")

    if "FitnessMulti" in creator.__dict__: del creator.FitnessMulti
    if "Individual"   in creator.__dict__: del creator.Individual

    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 0.3, 0.1))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_float, n=N_FACTORS + 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate,
                     X_train=X_train, y_train=y_train,
                     X_val=X_val,     y_val=y_val)
    toolbox.register("mate",   tools.cxBlend, alpha=0.3)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.15, indpb=0.15)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(30)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("wr_avg", lambda x: f"{np.mean([v[0] for v in x]):.3f}")
    stats.register("wr_max", lambda x: f"{np.max([v[0] for v in x]):.3f}")

    if verbose:
        print(f"开始搜索：{pop_size}个体 × {n_gen}代\n")

    pop, log = algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=pop_size, lambda_=pop_size,
        cxpb=0.6, mutpb=0.3,
        ngen=n_gen,
        stats=stats, halloffame=hof,
        verbose=verbose
    )

    # 从候选里找验证集胜率最高 且 交易次数充足的
    best_result = None
    best_val_wr = 0

    for ind in hof:
        weights   = np.array(ind[:N_FACTORS])
        threshold = np.clip(abs(ind[N_FACTORS]), 0.1, 0.8)
        wr_val, sh_val, n_val = compute_metrics(weights, threshold, X_val, y_val)

        if n_val >= MIN_TRADES and wr_val > best_val_wr:
            best_val_wr = wr_val
            best_result = {
                "weights":     weights,
                "threshold":   threshold,
                "factor_mean": X_mean,
                "factor_std":  X_std,
                "val_winrate": wr_val,
                "val_sharpe":  sh_val,
                "val_trades":  int(n_val),
            }

    if best_result and verbose:
        print(f"\n最优（验证集）→ 胜率: {best_result['val_winrate']:.3f}"
              f" | Sharpe: {best_result['val_sharpe']:.2f}"
              f" | 交易次数: {best_result['val_trades']}")
        print("有效因子权重:")
        for col, w in zip(FACTOR_COLS, best_result["weights"]):
            if abs(w) > 0.05:
                print(f"  {col:20s}: {w:+.3f}")
    elif verbose:
        print(f"\n⚠️  未找到满足 {MIN_TRADES} 笔交易要求的组合，尝试增加 lookback 或降低 MIN_TRADES")

    return best_result


if __name__ == "__main__":
    df = pl.read_parquet("data/btc_5m_factors.parquet")
    result = run_genetic_search(df, lookback=20000, n_gen=60, pop_size=300)

    if result:
        wr = result['val_winrate']
        trades = result['val_trades']
        print(f"\n{'✅' if wr >= 0.53 else '⚠️ '} 验证集胜率 {wr:.3f}，交易次数 {trades}")
        if wr > 0.65:
            print("    注意：胜率仍偏高，上线前需做更长周期验证")
    else:
        print("\n❌ 未找到有效组合")
