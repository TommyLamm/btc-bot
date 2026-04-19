import polars as pl

df = pl.read_parquet("data/btc_5m.parquet")

# 修复 datetime 列：timestamp 是毫秒，直接转就行
df = df.with_columns([
    pl.col("timestamp").cast(pl.Datetime("ms", "UTC")).alias("datetime")
])

df.write_parquet("data/btc_5m.parquet")

print("修复完成，验证：")
print(df.head(3))
print(df.tail(3))
print(f"\n时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
