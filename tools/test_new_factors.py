import polars as pl
import numpy as np
import sys
sys.path.insert(0, "/root/btc-bot")

kline   = pl.read_parquet("data/btc_5m.parquet").sort("timestamp")
funding = pl.read_parquet("data/btc_funding.parquet").sort("timestamp")
oi      = pl.read_parquet("data/btc_oi.parquet").sort("timestamp")

# 对齐资金费率到5m K线
df = kline.join_asof(
    funding.select(["timestamp", "funding_rate"]),
    on="timestamp", strategy="backward"
).join_asof(
    oi.select(["timestamp", "open_interest"]),
    on="timestamp", strategy="backward"
)

# 计算候选因子
df = df.with_columns([
    # 资金费率 z-score（滚动90期≈30天）
    ((pl.col("funding_rate") - pl.col("funding_rate").rolling_mean(90)) /
     (pl.col("funding_rate").rolling_std(90) + 1e-10)).alias("funding_zscore"),

    # 资金费率动量
    (pl.col("funding_rate") - pl.col("funding_rate").shift(3)).alias("funding_mom"),

    # 持仓量变化率
    (pl.col("open_interest") / pl.col("open_interest").shift(6) - 1).alias("oi_roc_6"),
    (pl.col("open_interest") / pl.col("open_interest").shift(24) - 1).alias("oi_roc_24"),

    # 持仓量 z-score
    ((pl.col("open_interest") - pl.col("open_interest").rolling_mean(48)) /
     (pl.col("open_interest").rolling_std(48) + 1e-10)).alias("oi_zscore"),

    # 价格 × 持仓量变化（趋势确认）
    # 价格涨+OI涨 = 真趋势；价格涨+OI跌 = 反弹不可信
    (pl.col("close").pct_change() *
     pl.col("open_interest").pct_change()).alias("price_oi_confirm"),

]).with_columns([
    pl.col("close").pct_change().shift(-1).alias("future_ret")
]).drop_nulls()

# 计算月度 IC
timestamps = df["timestamp"].to_numpy()
ret        = df["future_ret"].to_numpy()
month_ms   = 30 * 24 * 60 * 60 * 1000

test_cols = ["funding_zscore", "funding_mom", "oi_roc_6", "oi_roc_24",
             "oi_zscore", "price_oi_confirm"]

print(f"\n{'因子':<22} {'IC均值':>9} {'|IC|>0.02占比':>14} {'评级':>6}")
print("─" * 58)

for col in test_cols:
    vals = df[col].to_numpy().astype(np.float64)
    p1, p99 = np.nanpercentile(vals, [1, 99])
    vals = np.clip(vals, p1, p99)

    monthly_ics = []
    t_start = timestamps[0]
    while t_start < timestamps[-1]:
        t_end = t_start + month_ms
        mask  = (timestamps >= t_start) & (timestamps < t_end)
        if mask.sum() > 100:
            f_m, r_m = vals[mask], ret[mask]
            valid = ~(np.isnan(f_m) | np.isnan(r_m) | np.isinf(f_m))
            if valid.sum() > 50:
                ic = np.corrcoef(f_m[valid], r_m[valid])[0, 1]
                if not np.isnan(ic):
                    monthly_ics.append(ic)
        t_start = t_end

    if not monthly_ics:
        print(f"{col:<22} {'数据不足':>9}")
        continue

    ic_mean   = np.mean(monthly_ics)
    sig_ratio = np.mean(np.abs(monthly_ics) > 0.02)

    if abs(ic_mean) > 0.025:   grade = "⭐⭐⭐"
    elif abs(ic_mean) > 0.018: grade = "⭐⭐"
    elif abs(ic_mean) > 0.012: grade = "⭐"
    else:                      grade = "✗"

    print(f"{col:<22} {ic_mean:>+9.4f} {sig_ratio:>14.2f} {grade:>6}")
