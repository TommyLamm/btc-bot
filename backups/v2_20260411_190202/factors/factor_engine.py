"""
BTC-Bot 因子引擎（優化版）
計算所有技術因子，包含 OI 持倉量因子和 Hyperliquid 微觀結構因子。
"""

import polars as pl
import numpy as np


def compute_factors(df: pl.DataFrame, oi_df: pl.DataFrame = None) -> pl.DataFrame:
    """計算所有因子並返回帶因子列的 DataFrame。"""

    # 收益率
    df = df.with_columns([
        pl.col("close").pct_change().alias("ret1"),
    ])

    # 動量因子
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(5) - 1).alias("roc5"),
        (pl.col("close") / pl.col("close").shift(20) - 1).alias("roc20"),
    ])

    # 均線偏離
    df = df.with_columns([
        pl.col("close").rolling_mean(10).alias("_ma10"),
        pl.col("close").rolling_mean(30).alias("_ma30"),
        pl.col("volume").rolling_mean(20).alias("vol_ma20"),
    ]).with_columns([
        ((pl.col("close") - pl.col("_ma10")) / (pl.col("_ma10") + 1e-10)).alias("ma10dev"),
        ((pl.col("close") - pl.col("_ma30")) / (pl.col("_ma30") + 1e-10)).alias("ma30dev"),
    ])

    # VWAP 偏離
    df = df.with_columns([
        (pl.col("close") * pl.col("volume")).rolling_sum(20).alias("_vwap_num"),
        pl.col("volume").rolling_sum(20).alias("_vwap_den"),
    ]).with_columns([
        (pl.col("_vwap_num") / (pl.col("_vwap_den") + 1e-10)).alias("_vwap"),
    ]).with_columns([
        ((pl.col("close") - pl.col("_vwap")) / (pl.col("_vwap") + 1e-10)).alias("vwapdev"),
    ])

    # 價格衝擊
    df = df.with_columns([
        (pl.col("close").pct_change() * pl.col("volume") /
         (pl.col("vol_ma20") + 1e-10)).alias("priceimpact"),
    ])

    # MACD
    df = df.with_columns([
        pl.col("close").ewm_mean(span=12).alias("_ema12"),
        pl.col("close").ewm_mean(span=26).alias("_ema26"),
    ]).with_columns([
        (pl.col("_ema12") - pl.col("_ema26")).alias("_macd_line"),
    ]).with_columns([
        pl.col("_macd_line").ewm_mean(span=9).alias("_macd_signal"),
    ]).with_columns([
        (pl.col("_macd_line") - pl.col("_macd_signal")).alias("macdhist"),
    ])

    # 微觀結構因子
    df = df.with_columns([
        (pl.col("close").pct_change() * pl.col("volume") /
         (pl.col("vol_ma20") + 1e-10)).alias("vol_impact_hl"),
        ((pl.col("close") - pl.col("open")) /
         (pl.col("high") - pl.col("low") + 1e-10)).alias("candle_dir"),
    ])

    # 持倉量因子
    if oi_df is not None:
        oi_cols_to_drop = [c for c in df.columns
                          if c.startswith("open_interest") or
                             c in ("oi_roc6", "oi_roc24", "price_oi_confirm")]
        if oi_cols_to_drop:
            df = df.drop(oi_cols_to_drop)

        df = df.join_asof(
            oi_df.select(["timestamp", "open_interest"]).sort("timestamp"),
            on="timestamp",
            strategy="backward",
        ).with_columns([
            (pl.col("open_interest") /
             pl.col("open_interest").shift(6) - 1).alias("oi_roc6"),
            (pl.col("open_interest") /
             pl.col("open_interest").shift(24) - 1).alias("oi_roc24"),
        ]).with_columns([
            (pl.col("ret1") *
             pl.col("open_interest").pct_change()).alias("price_oi_confirm"),
        ])

    # 清理中間列
    drop_cols = [c for c in df.columns if c.startswith("_")]
    if drop_cols:
        df = df.drop(drop_cols)

    return df
