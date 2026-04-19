import polars as pl

df = pl.read_parquet("data/btc_5m.parquet")
print(df.shape)
print(df.head(3))
print(df.tail(3))

df = df.with_columns([
    pl.col("timestamp").diff().alias("gap")
])
gaps = df.filter(pl.col("gap") > 300_000)
print(f"缺失K线数量: {len(gaps)}")
