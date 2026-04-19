import polars as pl
import numpy as np
import sys
sys.path.insert(0, "/root/btc-bot")

# 读取资金费率（每8小时一条）
funding = pl.read_parquet("data/btc_funding.parquet").sort("timestamp")
kline   = pl.read_parquet("data/btc_5m.parquet").sort("timestamp")

print(f"资金费率条数: {len(funding)}")
print(funding.head(3))

# 资金费率每8小时更新一次，用 join_asof 对齐到5m K线
kline_with_funding = kline.join_asof(
    funding.select(["timestamp", "funding_rate"]),
    on="timestamp",
    strategy="backward"   # 用最近一次已知的资金费率
)

# 计算资金费率因子
kline_with_funding = kline_with_funding.with_columns([
    # z-score 标准化（滚动90期 = 最近30天）
    ((pl.col("funding_rate") - pl.col("funding_rate").rolling_mean(90)) /
     (pl.col("funding_rate").rolling_std(90) + 1e-10)).alias("funding_zscore"),

    # 资金费率动量（连续几期方向）
    pl.col("funding_rate").rolling_mean(3).alias("funding_ma3"),
])

# 验证
print(kline_with_funding.select([
    "datetime", "close", "funding_rate", "funding_zscore"
]).tail(5))

# 保存
kline_with_funding.write_parquet("data/btc_5m_with_funding.parquet")
print("\n已保存，现在测试 IC...")

# 快速 IC 测试
df = kline_with_funding.with_columns([
    pl.col("close").pct_change().shift(-1).alias("future_ret")
]).drop_nulls()

for col in ["funding_zscore", "funding_ma3"]:
    f = df[col].to_numpy()
    r = df["future_ret"].to_numpy()
    valid = ~(np.isnan(f) | np.isnan(r) | np.isinf(f))
    ic = np.corrcoef(f[valid], r[valid])[0, 1]
    print(f"{col}: IC = {ic:.4f}")
