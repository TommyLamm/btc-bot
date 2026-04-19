import polars as pl
import numpy as np

def compute_factors(df: pl.DataFrame,
                    oi_df: pl.DataFrame = None) -> pl.DataFrame:
    """
    df:    含 timestamp/open/high/low/close/volume/datetime 的 K线
    oi_df: 含 timestamp/open_interest 的持仓量数据（可选）
    """

    # ── 价格类因子 ────────────────────────────────────────
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(5)  - 1).alias("roc_5"),
        (pl.col("close") / pl.col("close").shift(20) - 1).alias("roc_20"),
        pl.col("close").rolling_mean(10).alias("ma_10"),
        pl.col("close").rolling_mean(30).alias("ma_30"),
        (pl.col("close") * pl.col("volume")).alias("turnover"),
        pl.col("volume").rolling_mean(20).alias("vol_ma_20"),
    ])

    # 均线偏离
    df = df.with_columns([
        (pl.col("close") / pl.col("ma_10") - 1).alias("ma10_dev"),
        (pl.col("close") / pl.col("ma_30") - 1).alias("ma30_dev"),
    ])

    # VWAP 偏离
    df = df.with_columns([
        (pl.col("turnover").rolling_sum(20) /
         (pl.col("volume").rolling_sum(20) + 1e-10)).alias("vwap_20"),
    ]).with_columns([
        (pl.col("close") / pl.col("vwap_20") - 1).alias("vwap_dev"),
    ])

    # 价格冲击
    df = df.with_columns([
        pl.col("close").pct_change().alias("_ret1"),
    ]).with_columns([
        (pl.col("_ret1") / (pl.col("vol_ma_20") /
         (pl.col("volume") + 1e-10) + 1e-10)).alias("price_impact"),
    ])

    # MACD hist
    df = df.with_columns([
        pl.col("close").ewm_mean(span=12).alias("_ema12"),
        pl.col("close").ewm_mean(span=26).alias("_ema26"),
    ]).with_columns([
        (pl.col("_ema12") - pl.col("_ema26")).alias("_macd"),
    ]).with_columns([
        (pl.col("_macd") - pl.col("_macd").ewm_mean(span=9)).alias("macd_hist"),
    ])

    # ── 持仓量因子（如果有数据）────────────────────────────
    if oi_df is not None:
        if "open_interest_oi" in df.columns:
            df = df.drop("open_interest_oi")
        df = df.join_asof(
            oi_df.select(["timestamp", "open_interest"]).sort("timestamp"),
            on="timestamp", strategy="backward", suffix="_oi"
        ).with_columns([
            (pl.col("open_interest") /
             pl.col("open_interest").shift(6)  - 1).alias("oi_roc_6"),
            (pl.col("open_interest") /
             pl.col("open_interest").shift(24) - 1).alias("oi_roc_24"),
        ]).with_columns([
            # 趋势确认因子：价格涨+OI涨=真趋势
            (pl.col("_ret1") *
             pl.col("open_interest").pct_change()).alias("price_oi_confirm"),
        ])

    # ── 清理中间列 ────────────────────────────────────────
    drop_cols = ["_ret1", "_ema12", "_ema26", "_macd"]
    existing  = [c for c in drop_cols if c in df.columns]
    df = df.drop(existing)

    # 微观结构因子
    df = df.with_columns([
        (pl.col("close").pct_change() * pl.col("volume") /
         (pl.col("vol_ma_20") + 1e-10)).alias("vol_impact_hl"),
        ((pl.col("close") - pl.col("open")) /
         (pl.col("high") - pl.col("low") + 1e-10)).alias("candle_dir"),
    ])

    return df


if __name__ == "__main__":
    kline = pl.read_parquet("data/btc_5m.parquet")
    oi    = pl.read_parquet("data/btc_oi.parquet").sort("timestamp")

    df = compute_factors(kline, oi_df=oi)
    df.write_parquet("data/btc_5m_factors.parquet")

    print(f"因子列数: {len(df.columns)}")
    for c in df.columns:
        print(f"  {c}")
    print(df.tail(3))
