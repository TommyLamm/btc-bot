"""
BTC-Bot 因子引擎 v5.0 — 全面升級版
新增因子類型：
  1. 波動率因子：ATR ratio、Bollinger %B、真實波幅比
  2. 成交量因子：OBV 斜率、成交量突破、量價背離
  3. 微觀結構因子：OBI 動量、spread 變化率、bid/ask 不平衡滾動
  4. 時間因子：小時效應
  5. 多週期因子：快慢動量差
  6. 改進的原有因子：去除冗餘
"""

import polars as pl
import numpy as np


def compute_factors(df: pl.DataFrame, oi_df: pl.DataFrame = None) -> pl.DataFrame:
    """計算所有因子並返回帶因子列的 DataFrame。"""

    # ═══ 基礎收益率 ═══
    df = df.with_columns([
        pl.col("close").pct_change().alias("ret1"),
    ])

    # ═══ 動量因子 ═══
    df = df.with_columns([
        (pl.col("close") / pl.col("close").shift(5) - 1).alias("roc5"),
        (pl.col("close") / pl.col("close").shift(20) - 1).alias("roc20"),
        # 快慢動量差（動量加速度）
        ((pl.col("close") / pl.col("close").shift(5) - 1)
         - (pl.col("close").shift(5) / pl.col("close").shift(10) - 1)).alias("mom_accel"),
    ])

    # ═══ 均線偏離 ═══
    df = df.with_columns([
        pl.col("close").rolling_mean(10).alias("_ma10"),
        pl.col("close").rolling_mean(30).alias("_ma30"),
        pl.col("close").rolling_mean(60).alias("_ma60"),
        pl.col("volume").rolling_mean(20).alias("vol_ma20"),
    ]).with_columns([
        ((pl.col("close") - pl.col("_ma10")) / (pl.col("_ma10") + 1e-10)).alias("ma10dev"),
        ((pl.col("close") - pl.col("_ma30")) / (pl.col("_ma30") + 1e-10)).alias("ma30dev"),
        # 均線排列（MA10 vs MA30 vs MA60 的相對位置）
        ((pl.col("_ma10") - pl.col("_ma30")) / (pl.col("_ma30") + 1e-10)).alias("ma_alignment"),
    ])

    # ═══ VWAP 偏離 ═══
    df = df.with_columns([
        (pl.col("close") * pl.col("volume")).rolling_sum(20).alias("_vwap_num"),
        pl.col("volume").rolling_sum(20).alias("_vwap_den"),
    ]).with_columns([
        (pl.col("_vwap_num") / (pl.col("_vwap_den") + 1e-10)).alias("_vwap"),
    ]).with_columns([
        ((pl.col("close") - pl.col("_vwap")) / (pl.col("_vwap") + 1e-10)).alias("vwapdev"),
    ])

    # ═══ 波動率因子 ═══
    df = df.with_columns([
        # Bollinger %B
        pl.col("close").rolling_mean(20).alias("_bb_mid"),
        pl.col("close").rolling_std(20).alias("_bb_std"),
    ]).with_columns([
        ((pl.col("close") - (pl.col("_bb_mid") - 2 * pl.col("_bb_std")))
         / (4 * pl.col("_bb_std") + 1e-10)).alias("bb_pctb"),
    ])

    df = df.with_columns([
        # 短期波動 / 長期波動（波動率收縮/擴張）
        (pl.col("close").rolling_std(5) / (pl.col("close").rolling_std(20) + 1e-10)).alias("vol_ratio"),
        # 真實波幅比（當前 K 線振幅 vs 平均振幅）
        ((pl.col("high") - pl.col("low"))
         / ((pl.col("high") - pl.col("low")).rolling_mean(20) + 1e-10)).alias("tr_ratio"),
    ])

    # ═══ 成交量因子 ═══
    df = df.with_columns([
        # 成交量突破（當前成交量 / 20 期均量）
        (pl.col("volume") / (pl.col("vol_ma20") + 1e-10)).alias("vol_surge"),
        # 量價背離：價格上漲但成交量下降（或反之）
        (pl.col("close").pct_change().sign()
         * (1 - pl.col("volume") / (pl.col("vol_ma20") + 1e-10))).alias("vol_price_div"),
    ])

    # OBV 斜率
    df = df.with_columns([
        (pl.col("close").pct_change().sign() * pl.col("volume")).alias("_obv_delta"),
    ]).with_columns([
        pl.col("_obv_delta").cum_sum().alias("_obv"),
    ]).with_columns([
        # OBV 5 期變化率（標準化）
        ((pl.col("_obv") - pl.col("_obv").shift(5))
         / (pl.col("volume").rolling_sum(5) + 1e-10)).alias("obv_slope"),
    ])

    # ═══ 價格衝擊 ═══
    df = df.with_columns([
        (pl.col("close").pct_change() * pl.col("volume")
         / (pl.col("vol_ma20") + 1e-10)).alias("priceimpact"),
    ])

    # ═══ MACD ═══
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

    # ═══ RSI ═══
    df = df.with_columns([
        pl.when(pl.col("ret1") > 0).then(pl.col("ret1")).otherwise(0.0).alias("_gain"),
        pl.when(pl.col("ret1") < 0).then(-pl.col("ret1")).otherwise(0.0).alias("_loss"),
    ]).with_columns([
        pl.col("_gain").rolling_mean(14).alias("_avg_gain"),
        pl.col("_loss").rolling_mean(14).alias("_avg_loss"),
    ]).with_columns([
        # RSI 標準化到 [-1, 1] 範圍（0.5 對應 RSI=50）
        (pl.col("_avg_gain") / (pl.col("_avg_gain") + pl.col("_avg_loss") + 1e-10) - 0.5).alias("rsi_norm"),
    ])

    # ═══ 微觀結構因子 ═══
    df = df.with_columns([
        # K 線方向（實體 / 振幅）
        ((pl.col("close") - pl.col("open"))
         / (pl.col("high") - pl.col("low") + 1e-10)).alias("candle_dir"),
        # 上影線比例
        ((pl.col("high") - pl.col("close").clip(pl.col("open"), None).fill_null(pl.col("close")))
         / (pl.col("high") - pl.col("low") + 1e-10)).alias("upper_shadow"),
    ])

    # OBI 相關因子（如果有 obi 列）
    if "obi" in df.columns:
        df = df.with_columns([
            # OBI 動量（5 期變化）
            (pl.col("obi") - pl.col("obi").shift(5)).alias("obi_momentum"),
            # OBI 滾動均值
            pl.col("obi").rolling_mean(10).alias("obi_ma"),
        ])

    # ═══ 時間因子 ═══
    if "timestamp" in df.columns:
        df = df.with_columns([
            # 小時效應（UTC 小時的正弦/餘弦編碼）
            (2 * 3.14159 * (pl.col("timestamp") / 1000 % 86400) / 86400).alias("_hour_rad"),
        ]).with_columns([
            pl.col("_hour_rad").sin().alias("hour_sin"),
            pl.col("_hour_rad").cos().alias("hour_cos"),
        ])

    # ═══ 持倉量因子 ═══
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
            (pl.col("open_interest")
             / pl.col("open_interest").shift(6) - 1).alias("oi_roc6"),
            (pl.col("open_interest")
             / pl.col("open_interest").shift(24) - 1).alias("oi_roc24"),
        ]).with_columns([
            (pl.col("ret1")
             * pl.col("open_interest").pct_change()).alias("price_oi_confirm"),
        ])

    # ═══ 清理中間列 ═══
    drop_cols = [c for c in df.columns if c.startswith("_")]
    if drop_cols:
        df = df.drop(drop_cols)

    return df

