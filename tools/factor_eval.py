import polars as pl
import numpy as np
import sys
sys.path.insert(0, "/root/btc-bot")

from engine.genetic_engine import FACTOR_COLS

df = pl.read_parquet("data/btc_5m_factors.parquet").drop_nulls()

# 未来1根K线收益率
df = df.with_columns([
    pl.col("close").pct_change().shift(-1).alias("future_ret")
]).drop_nulls()

print(f"{'因子':<20} {'IC均值':>8} {'IC稳定性':>10} {'|IC|>0.02占比':>14} {'评级':>6}")
print("─" * 65)

results = []
for col in FACTOR_COLS:
    factor = df[col].to_numpy()
    ret    = df["future_ret"].to_numpy()

    # 按月计算 IC
    monthly_ics = []
    timestamps  = df["timestamp"].to_numpy()
    month_ms    = 30 * 24 * 60 * 60 * 1000

    t_start = timestamps[0]
    while t_start < timestamps[-1]:
        t_end = t_start + month_ms
        mask  = (timestamps >= t_start) & (timestamps < t_end)
        if mask.sum() > 100:
            f_m = factor[mask]
            r_m = ret[mask]
            valid = ~(np.isnan(f_m) | np.isnan(r_m))
            if valid.sum() > 50:
                ic = np.corrcoef(f_m[valid], r_m[valid])[0, 1]
                if not np.isnan(ic):
                    monthly_ics.append(ic)
        t_start = t_end

    if not monthly_ics:
        continue

    ic_mean   = np.mean(monthly_ics)
    ic_std    = np.std(monthly_ics)
    ic_ir     = ic_mean / (ic_std + 1e-10)   # IC 信息比率
    sig_ratio = np.mean(np.abs(monthly_ics) > 0.02)

    # 评级
    if abs(ic_mean) > 0.03 and sig_ratio > 0.6:
        grade = "⭐⭐⭐"
    elif abs(ic_mean) > 0.02 and sig_ratio > 0.4:
        grade = "⭐⭐"
    elif abs(ic_mean) > 0.01:
        grade = "⭐"
    else:
        grade = "✗"

    results.append((col, ic_mean, ic_ir, sig_ratio, grade))

results.sort(key=lambda x: abs(x[1]), reverse=True)
for col, ic_mean, ic_ir, sig_ratio, grade in results:
    print(f"{col:<20} {ic_mean:>+8.4f} {ic_ir:>10.3f} {sig_ratio:>14.2f} {grade:>6}")
