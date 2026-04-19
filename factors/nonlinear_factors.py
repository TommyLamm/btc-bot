import polars as pl
import numpy as np
import os
import sys
# Bug 21 Fix：使用動態路徑而非硬編碼
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def add_nonlinear_factors(df: pl.DataFrame) -> pl.DataFrame:
    """
    基于已有因子生成非线性组合
    只对 IC 最高的几个因子做组合，避免维度爆炸
    """

    # Bug 21 Fix：因子名稱對齊 v5.0（移除底線分隔）
    required = ["ma10dev", "roc5", "priceimpact",
                "price_oi_confirm", "vwapdev"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"缺少因子: {missing}")
        return df

    df = df.with_columns([

        # ── 交互项：两个因子同向时信号更强 ──────────────
        # 均线偏离 × 短期动量（都超买时更可能反转）
        (pl.col("ma10dev") * pl.col("roc5")).alias("ma10_x_roc5"),

        # 价格冲击 × 持仓量确认（量价齐升时趋势更可信）
        (pl.col("priceimpact") * pl.col("price_oi_confirm")).alias("impact_x_oi"),

        # VWAP偏离 × 均线偏离（双重偏离信号）
        (pl.col("vwapdev") * pl.col("ma10dev")).alias("vwap_x_ma10"),

        # ── 非线性变换：捕捉极端值 ───────────────────────
        # 偏离程度的平方（惩罚极端偏离）
        (pl.col("ma10dev") ** 2 * pl.col("ma10dev").sign()).alias("ma10_dev_sq"),

        # ── 条件因子：只在特定市场状态下激活 ────────────
        # 高波动时的均线偏离（高波动下反转更快）
        (pl.col("ma10dev") *
         pl.col("roc5").abs()).alias("ma10_dev_vol_weighted"),

    ])

    return df


if __name__ == "__main__":
    df = pl.read_parquet("data/btc_5m_factors.parquet").drop_nulls()
    df = add_nonlinear_factors(df)

    # 测试 IC
    df = df.with_columns([
        pl.col("close").pct_change().shift(-1).alias("future_ret")
    ]).drop_nulls()

    new_cols = ["ma10_x_roc5", "impact_x_oi", "vwap_x_ma10",
                "ma10_dev_sq", "ma10_dev_vol_weighted"]

    timestamps = df["timestamp"].to_numpy()
    ret        = df["future_ret"].to_numpy()
    month_ms   = 30 * 24 * 60 * 60 * 1000

    print(f"\n{'因子':<28} {'IC均值':>9} {'占比':>8} {'评级':>6}")
    print("─" * 58)

    results = []
    for col in new_cols:
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
            continue

        ic_mean   = np.mean(monthly_ics)
        sig_ratio = np.mean(np.abs(monthly_ics) > 0.02)

        if abs(ic_mean) > 0.025:   grade = "⭐⭐⭐"
        elif abs(ic_mean) > 0.018: grade = "⭐⭐"
        elif abs(ic_mean) > 0.012: grade = "⭐"
        else:                      grade = "✗"

        results.append((col, ic_mean, sig_ratio, grade))

    results.sort(key=lambda x: abs(x[1]), reverse=True)
    for col, ic, sr, grade in results:
        print(f"{col:<28} {ic:>+9.4f} {sr:>8.2f} {grade:>6}")
