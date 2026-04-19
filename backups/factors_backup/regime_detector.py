import polars as pl
import numpy as np

def detect_regime(df: pl.DataFrame) -> pl.DataFrame:
    """
    市场状态检测
    0 = 震荡（用反转因子）
    1 = 上升趋势（用趋势跟踪）
    -1 = 下降趋势（用趋势跟踪）
    """

    df = df.with_columns([
        # 方法1：ADX 简化版（趋势强度）
        (pl.col("high") - pl.col("low")).rolling_mean(14).alias("atr_14"),
        pl.col("close").rolling_mean(20).alias("_ma20"),
        pl.col("close").rolling_mean(60).alias("_ma60"),
    ]).with_columns([
        # 趋势强度：价格相对长期均线的偏离 + 均线斜率
        ((pl.col("_ma20") - pl.col("_ma60")) /
         (pl.col("atr_14") + 1e-10)).alias("trend_strength"),

        # 均线斜率（最近10根均线的变化率）
        (pl.col("_ma20") / pl.col("_ma20").shift(10) - 1).alias("ma_slope"),
    ]).with_columns([
        # 市场状态：trend_strength 绝对值大 = 趋势市
        pl.when(pl.col("trend_strength") > 1.0)
          .then(pl.lit(1))
          .when(pl.col("trend_strength") < -1.0)
          .then(pl.lit(-1))
          .otherwise(pl.lit(0))
          .alias("regime")
    ])

    return df.drop(["_ma20", "_ma60"])


if __name__ == "__main__":
    df = pl.read_parquet("data/btc_5m_factors.parquet").drop_nulls()
    df = detect_regime(df)

    regime_counts = df["regime"].value_counts().sort("regime")
    print("市场状态分布:")
    print(regime_counts)

    total = len(df)
    for row in regime_counts.iter_rows():
        pct = row[1] / total * 100
        label = {1: "上升趋势", -1: "下降趋势", 0: "震荡"}[row[0]]
        print(f"  {label}: {pct:.1f}%")

    # 分状态测试 ma10_dev 的 IC
    print("\n── ma10_dev 分状态 IC ──")
    df = df.with_columns([
        pl.col("close").pct_change().shift(-1).alias("future_ret")
    ]).drop_nulls()

    for regime_val, label in [(0, "震荡"), (1, "趋势")]:
        sub = df.filter(pl.col("regime") == regime_val)
        if len(sub) < 100:
            continue
        f = sub["ma10_dev"].to_numpy()
        r = sub["future_ret"].to_numpy()
        valid = ~(np.isnan(f) | np.isnan(r))
        ic = np.corrcoef(f[valid], r[valid])[0, 1]
        print(f"  {label}: IC = {ic:.4f}  (n={len(sub)})")
