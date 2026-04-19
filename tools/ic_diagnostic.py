import polars as pl
import numpy as np
import sys
sys.path.insert(0, "/root/btc-bot")

df = pl.read_parquet("data/btc_5m_factors.parquet").drop_nulls().fill_nan(0)
df = df.with_columns([
    pl.col("close").pct_change().shift(-1).alias("future_ret")
]).drop_nulls()

ret        = df["future_ret"].to_numpy()
timestamps = df["timestamp"].to_numpy()
month_ms   = 30 * 24 * 60 * 60 * 1000

FACTOR_COLS = [
    "roc_5", "roc_20", "ma10_dev", "ma30_dev",
    "vwap_dev", "price_impact", "macd_hist",
    "oi_roc_6", "oi_roc_24", "price_oi_confirm",
]

# 1. 先看原始因子的全样本 IC（不分月）
print("── 全样本 IC（不分月）──")
for col in FACTOR_COLS:
    if col not in df.columns:
        continue
    f = df[col].to_numpy()
    valid = ~(np.isnan(f) | np.isnan(ret) | np.isinf(f))
    ic = np.corrcoef(f[valid], ret[valid])[0, 1]
    print(f"  {col:<22}: {ic:+.5f}")

# 2. 看月度 IC 分布
print("\n── ma10_dev 月度 IC 详情 ──")
f = df["ma10_dev"].to_numpy()
t_start = timestamps[0]
month_ics = []
while t_start < timestamps[-1]:
    t_end = t_start + month_ms
    mask  = (timestamps >= t_start) & (timestamps < t_end)
    if mask.sum() > 100:
        f_m, r_m = f[mask], ret[mask]
        valid = ~(np.isnan(f_m) | np.isnan(r_m))
        if valid.sum() > 50:
            ic = np.corrcoef(f_m[valid], r_m[valid])[0, 1]
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(t_start/1000, tz=timezone.utc)
            print(f"  {dt.strftime('%Y-%m')}: IC={ic:+.4f}  n={mask.sum()}")
            month_ics.append(ic)
    t_start = t_end

print(f"\n月度IC均值: {np.mean(month_ics):+.4f}")
print(f"月度IC标准差: {np.std(month_ics):.4f}")
print(f"|IC|>0.015占比: {np.mean(np.abs(month_ics)>0.015):.2f}")
print(f"|IC|>0.010占比: {np.mean(np.abs(month_ics)>0.010):.2f}")

# 3. 看几个简单组合的全样本 IC
print("\n── 简单组合全样本 IC ──")
combos = {
    "ma10_dev * roc_5":      df["ma10_dev"].to_numpy() * df["roc_5"].to_numpy(),
    "ma10_dev + vwap_dev":   df["ma10_dev"].to_numpy() + df["vwap_dev"].to_numpy(),
    "ma10_dev / price_impact": df["ma10_dev"].to_numpy() / (df["price_impact"].to_numpy() + 1e-10),
    "log(|ma10_dev|)*sign":  np.log(np.abs(df["ma10_dev"].to_numpy())+1e-10) * np.sign(df["ma10_dev"].to_numpy()),
}
for name, vals in combos.items():
    vals = np.clip(vals, *np.nanpercentile(vals, [1,99]))
    valid = ~(np.isnan(vals) | np.isnan(ret) | np.isinf(vals))
    ic = np.corrcoef(vals[valid], ret[valid])[0, 1]
    print(f"  {name:<35}: {ic:+.5f}")
