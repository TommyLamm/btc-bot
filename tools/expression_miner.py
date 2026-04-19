import polars as pl
import numpy as np
import random
import sys
sys.path.insert(0, "/root/btc-bot")

FACTOR_COLS = [
    "roc_5", "roc_20",
    "ma10_dev", "ma30_dev",
    "vwap_dev", "price_impact",
    "macd_hist",
    "oi_roc_6", "oi_roc_24",
    "price_oi_confirm",
]

# ── 算子（全部用 numpy，不用 pandas/polars）────────────────
def roll_mean(x, w=10):
    result = np.full_like(x, np.nan)
    for i in range(w-1, len(x)):
        result[i] = np.mean(x[i-w+1:i+1])
    return result

def roll_std(x, w=20):
    result = np.full_like(x, np.nan)
    for i in range(w-1, len(x)):
        result[i] = np.std(x[i-w+1:i+1])
    return result

def zscore(x, w=20):
    mu  = roll_mean(x, w)
    sig = roll_std(x, w)
    return (x - mu) / (sig + 1e-10)

def delay(x, d=1):
    result = np.full_like(x, np.nan)
    result[d:] = x[:-d]
    return result

def diff(x, d=1):
    result = np.full_like(x, np.nan)
    result[d:] = x[d:] - x[:-d]
    return result

def rank(x):
    valid = ~np.isnan(x)
    r = np.full_like(x, np.nan)
    r[valid] = x[valid].argsort().argsort().astype(float) / valid.sum()
    return r

UNARY_OPS = {
    "log":     lambda x: np.log(np.abs(x) + 1e-10),
    "sqrt":    lambda x: np.sqrt(np.abs(x)),
    "square":  lambda x: np.sign(x) * x**2,
    "zscore":  zscore,
    "rank":    rank,
    "delay1":  lambda x: delay(x, 1),
    "delay3":  lambda x: delay(x, 3),
    "diff1":   lambda x: diff(x, 1),
    "diff3":   lambda x: diff(x, 3),
    "rollmean":lambda x: roll_mean(x, 10),
    "rollstd": lambda x: roll_std(x, 20),
}

BINARY_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / (np.abs(b) + 1e-10),
    "max": lambda a, b: np.maximum(a, b),
    "min": lambda a, b: np.minimum(a, b),
}


def safe_compute(func, *args):
    try:
        with np.errstate(all="ignore"):
            result = func(*args)
            result = np.array(result, dtype=np.float64)
            result[np.isinf(result)] = np.nan
            valid = ~np.isnan(result)
            if valid.sum() < 100:
                return np.full(len(args[0]), np.nan)
            p1, p99 = np.nanpercentile(result, [1, 99])
            return np.clip(result, p1, p99)
    except Exception:
        return np.full(len(args[0]), np.nan)


def compute_monthly_ic(factor, ret, timestamps):
    month_ms = 30 * 24 * 60 * 60 * 1000
    t_start  = timestamps[0]
    monthly_ics = []

    while t_start < timestamps[-1]:
        t_end = t_start + month_ms
        mask  = (timestamps >= t_start) & (timestamps < t_end)
        if mask.sum() > 100:
            f, r  = factor[mask], ret[mask]
            valid = ~(np.isnan(f) | np.isnan(r) | np.isinf(f))
            if valid.sum() > 50:
                ic = np.corrcoef(f[valid], r[valid])[0, 1]
                if not np.isnan(ic):
                    monthly_ics.append(ic)
        t_start = t_end

    if len(monthly_ics) < 3:
        return 0.0, 0.0
    return float(np.mean(monthly_ics)), float(np.mean(np.abs(monthly_ics) > 0.02))


def mine_expressions(df, ret, timestamps, n_trials=800, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    base_data = {col: df[col].to_numpy().astype(np.float64)
                 for col in FACTOR_COLS if col in df.columns}
    available = list(base_data.keys())
    results   = []

    print(f"开始搜索，共 {n_trials} 次试验，基础因子 {len(available)} 个...")

    for i in range(n_trials):
        expr_type = random.choice(["unary", "binary", "binary_unary", "triple"])

        try:
            if expr_type == "unary":
                f1   = random.choice(available)
                uop  = random.choice(list(UNARY_OPS.keys()))
                val  = safe_compute(UNARY_OPS[uop], base_data[f1])
                desc = f"{uop}({f1})"

            elif expr_type == "binary":
                f1, f2 = random.sample(available, 2)
                bop    = random.choice(list(BINARY_OPS.keys()))
                val    = safe_compute(BINARY_OPS[bop], base_data[f1], base_data[f2])
                desc   = f"({f1} {bop} {f2})"

            elif expr_type == "binary_unary":
                f1, f2 = random.sample(available, 2)
                bop    = random.choice(list(BINARY_OPS.keys()))
                uop    = random.choice(list(UNARY_OPS.keys()))
                tmp    = safe_compute(BINARY_OPS[bop], base_data[f1], base_data[f2])
                val    = safe_compute(UNARY_OPS[uop], tmp)
                desc   = f"{uop}({f1} {bop} {f2})"

            else:  # triple
                f1, f2, f3 = random.sample(available, 3)
                bop1 = random.choice(list(BINARY_OPS.keys()))
                bop2 = random.choice(list(BINARY_OPS.keys()))
                tmp  = safe_compute(BINARY_OPS[bop1], base_data[f1], base_data[f2])
                val  = safe_compute(BINARY_OPS[bop2], tmp, base_data[f3])
                desc = f"(({f1} {bop1} {f2}) {bop2} {f3})"

        except Exception:
            continue

        if np.isnan(val).mean() > 0.3:
            continue

        ic_mean, sig_ratio = compute_monthly_ic(val, ret, timestamps)

        if abs(ic_mean) >= 0.015:
            if abs(ic_mean) > 0.025:   grade = "⭐⭐⭐"
            elif abs(ic_mean) > 0.018: grade = "⭐⭐"
            else:                      grade = "⭐"
            results.append((desc, ic_mean, sig_ratio, grade, val.copy()))

        if (i+1) % 200 == 0:
            print(f"  {i+1}/{n_trials}，已发现 {len(results)} 个候选")

    # 去相关筛选
    results.sort(key=lambda x: abs(x[1]), reverse=True)
    kept, kept_vals = [], []

    for desc, ic, sr, grade, val in results:
        redundant = False
        for pv in kept_vals:
            valid = ~(np.isnan(val) | np.isnan(pv))
            if valid.sum() > 100:
                if abs(np.corrcoef(val[valid], pv[valid])[0, 1]) > 0.80:
                    redundant = True
                    break
        if not redundant:
            kept.append((desc, ic, sr, grade))
            kept_vals.append(val)

    return kept


if __name__ == "__main__":
    df = pl.read_parquet("data/btc_5m_factors.parquet").drop_nulls().fill_nan(0)
    df = df.with_columns([
        pl.col("close").pct_change().shift(-1).alias("future_ret")
    ]).drop_nulls()

    ret        = df["future_ret"].to_numpy()
    timestamps = df["timestamp"].to_numpy()

    results = mine_expressions(df, ret, timestamps, n_trials=800)

    print(f"\n{'─'*68}")
    print(f"{'表达式':<42} {'IC均值':>8} {'占比':>6} {'评级':>8}")
    print(f"{'─'*68}")
    for desc, ic, sr, grade in results[:30]:
        print(f"{desc:<42} {ic:>+8.4f} {sr:>6.2f} {grade:>8}")

    print(f"\n✅ 共发现 {len(results)} 个独立有效因子")

    if results:
        print("\n⭐⭐ 以上建议加入因子库：")
        for desc, ic, sr, grade in results:
            if "⭐⭐" in grade:
                print(f"  {desc}  IC={ic:+.4f}")
