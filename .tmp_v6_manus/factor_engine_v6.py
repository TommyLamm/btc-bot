"""
BTC-Bot 因子引擎 v6.0 — 勝率導向因子庫
新增因子（基於微觀結構研究和實戰驗證）：
  v6.0 新增：
  1. VPIN（Volume-Synchronized Probability of Informed Trading）
     知情交易概率 — 高 VPIN = 知情交易者活躍 = 大行情即將來臨
  2. Kyle's Lambda（價格衝擊係數）
     衡量每單位成交量對價格的衝擊力度
  3. Amihud 非流動性指標
     |收益率| / 成交量 — 高值 = 流動性差
  4. 多時間框架趨勢對齊（mtf_trend_align）
     5 分鐘和 15 分鐘趨勢方向一致時信號更強
  5. 多時間框架動量確認（mtf_mom_confirm）
     短期動量和中期動量同向確認
  6. 信號持續性（signal_persistence）
     價格動量在最近 3 根 K 線內方向一致
  7. 市場狀態指標（regime_indicator）
     基於 Kaufman ER 的趨勢/震盪分類
  8. 收盤價位置（close_to_high, close_to_low）
     收盤價在 K 線中的相對位置
  9. 量價趨勢（volume_price_trend）
     累積量價趨勢指標
  10. MFI 標準化（mfi_norm）
      資金流量指標
  保留所有 v5.5 因子
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
        # 多時間框架動量
        (pl.col("close") / pl.col("close").shift(3) - 1).alias("roc15m"),
        (pl.col("close") / pl.col("close").shift(12) - 1).alias("roc1h"),
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
        ((pl.col("_ma10") - pl.col("_ma30")) / (pl.col("_ma30") + 1e-10)).alias("ma_alignment"),
    ])

    # ═══ VWAP 偏離 + 斜率 ═══
    df = df.with_columns([
        (pl.col("close") * pl.col("volume")).rolling_sum(20).alias("_vwap_num"),
        pl.col("volume").rolling_sum(20).alias("_vwap_den"),
    ]).with_columns([
        (pl.col("_vwap_num") / (pl.col("_vwap_den") + 1e-10)).alias("_vwap"),
    ]).with_columns([
        ((pl.col("close") - pl.col("_vwap")) / (pl.col("_vwap") + 1e-10)).alias("vwapdev"),
    ])
    df = df.with_columns([
        ((pl.col("_vwap") / pl.col("_vwap").shift(5) - 1)).alias("vwap_slope"),
    ])

    # ═══ 波動率因子 ═══
    df = df.with_columns([
        pl.col("close").rolling_mean(20).alias("_bb_mid"),
        pl.col("close").rolling_std(20).alias("_bb_std"),
    ]).with_columns([
        ((pl.col("close") - (pl.col("_bb_mid") - 2 * pl.col("_bb_std")))
         / (4 * pl.col("_bb_std") + 1e-10)).alias("bb_pctb"),
        (4 * pl.col("_bb_std") / (pl.col("_bb_mid") + 1e-10)).alias("bb_width"),
    ])

    df = df.with_columns([
        (pl.col("close").rolling_std(5) / (pl.col("close").rolling_std(20) + 1e-10)).alias("vol_ratio"),
        ((pl.col("high") - pl.col("low"))
         / ((pl.col("high") - pl.col("low")).rolling_mean(20) + 1e-10)).alias("tr_ratio"),
    ])

    # v5.5：波動率聚類變化
    df = df.with_columns([
        (pl.col("close").rolling_std(5) / (pl.col("close").rolling_std(20) + 1e-10)).alias("_vol_ratio_raw"),
    ]).with_columns([
        (pl.col("_vol_ratio_raw") / (pl.col("_vol_ratio_raw").shift(5) + 1e-10) - 1).alias("vol_regime_change"),
    ])

    # v5.5：振幅擴張
    df = df.with_columns([
        ((pl.col("high") - pl.col("low"))
         / ((pl.col("high") - pl.col("low")).rolling_mean(10) + 1e-10)).alias("range_expansion"),
    ])

    # ═══ 成交量因子 ═══
    df = df.with_columns([
        (pl.col("volume") / (pl.col("vol_ma20") + 1e-10)).alias("vol_surge"),
        (pl.col("close").pct_change().sign()
         * (1 - pl.col("volume") / (pl.col("vol_ma20") + 1e-10))).alias("vol_price_div"),
    ])

    # 成交量加速度
    df = df.with_columns([
        pl.col("volume").rolling_mean(5).alias("_vol_ma5"),
    ]).with_columns([
        ((pl.col("_vol_ma5") / pl.col("_vol_ma5").shift(5) - 1)).alias("vol_accel"),
    ])

    # OBV 斜率
    df = df.with_columns([
        (pl.col("close").pct_change().sign() * pl.col("volume")).alias("_obv_delta"),
    ]).with_columns([
        pl.col("_obv_delta").cum_sum().alias("_obv"),
    ]).with_columns([
        ((pl.col("_obv") - pl.col("_obv").shift(5))
         / (pl.col("volume").rolling_sum(5) + 1e-10)).alias("obv_slope"),
    ])

    # v5.5：動量質量
    df = df.with_columns([
        ((pl.col("close") / pl.col("close").shift(5) - 1)
         * pl.col("volume") / (pl.col("vol_ma20") + 1e-10)).alias("momentum_quality"),
    ])

    # 價格衝擊
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
        (pl.col("_avg_gain") / (pl.col("_avg_gain") + pl.col("_avg_loss") + 1e-10) - 0.5).alias("rsi_norm"),
    ])

    # RSI 背離
    df = df.with_columns([
        pl.col("close").rolling_max(10).alias("_price_high10"),
        pl.col("close").rolling_min(10).alias("_price_low10"),
        (pl.col("_avg_gain") / (pl.col("_avg_gain") + pl.col("_avg_loss") + 1e-10)).alias("_rsi_raw"),
    ]).with_columns([
        pl.col("_rsi_raw").rolling_max(10).alias("_rsi_high10"),
        pl.col("_rsi_raw").rolling_min(10).alias("_rsi_low10"),
    ]).with_columns([
        (((pl.col("close") - pl.col("_price_low10"))
          / (pl.col("_price_high10") - pl.col("_price_low10") + 1e-10))
         - ((pl.col("_rsi_raw") - pl.col("_rsi_low10"))
            / (pl.col("_rsi_high10") - pl.col("_rsi_low10") + 1e-10))).alias("rsi_divergence"),
    ])

    # ═══ 微觀結構因子 ═══
    df = df.with_columns([
        ((pl.col("close") - pl.col("open"))
         / (pl.col("high") - pl.col("low") + 1e-10)).alias("candle_dir"),
    ])

    # v5.5：Kaufman Efficiency Ratio
    df = df.with_columns([
        (pl.col("close") - pl.col("close").shift(10)).abs().alias("_er_direction"),
        pl.col("close").diff().abs().rolling_sum(10).alias("_er_volatility"),
    ]).with_columns([
        (pl.col("_er_direction") / (pl.col("_er_volatility") + 1e-10)).alias("kaufman_er"),
    ])

    # v5.5：連續K線計數
    if len(df) > 0:
        close_arr = df["close"].to_numpy()
        consec = np.zeros(len(close_arr))
        for i in range(1, len(close_arr)):
            if close_arr[i] > close_arr[i - 1]:
                consec[i] = max(consec[i - 1], 0) + 1
            elif close_arr[i] < close_arr[i - 1]:
                consec[i] = min(consec[i - 1], 0) - 1
            else:
                consec[i] = 0
        consec = np.clip(consec / 6.0, -1.0, 1.0)
        df = df.with_columns(pl.Series("consec_candles", consec))

    # v5.5：收益偏度
    if len(df) > 10:
        ret_arr = df["ret1"].to_numpy().astype(np.float64)
        ret_arr = np.nan_to_num(ret_arr, nan=0.0)
        skew_arr = np.zeros(len(ret_arr))
        window = 10
        for i in range(window, len(ret_arr)):
            chunk = ret_arr[i - window:i]
            m = chunk.mean()
            s = chunk.std()
            if s > 1e-10:
                skew_arr[i] = ((chunk - m) ** 3).mean() / (s ** 3)
            else:
                skew_arr[i] = 0.0
        skew_arr = np.clip(skew_arr, -3.0, 3.0) / 3.0
        df = df.with_columns(pl.Series("return_skew", skew_arr))
    else:
        df = df.with_columns(pl.lit(0.0).alias("return_skew"))

    # ═══════════════════════════════════════════════════
    #  v6.0 新增因子
    # ═══════════════════════════════════════════════════

    # ═══ v6.0 新增 1: VPIN（Volume-Synchronized Probability of Informed Trading）═══
    # 簡化版 VPIN：使用 bulk volume classification
    # VPIN = mean(|buy_vol - sell_vol|) / total_vol over window
    if len(df) > 20:
        close_np = df["close"].to_numpy().astype(np.float64)
        vol_np = df["volume"].to_numpy().astype(np.float64)
        vol_np = np.nan_to_num(vol_np, nan=0.0)
        ret_np = np.zeros(len(close_np))
        ret_np[1:] = (close_np[1:] - close_np[:-1]) / (close_np[:-1] + 1e-10)

        # Bulk Volume Classification: buy_pct = Phi(ret / sigma)
        sigma = np.zeros(len(ret_np))
        for i in range(20, len(ret_np)):
            sigma[i] = np.std(ret_np[i-20:i])
        sigma = np.where(sigma < 1e-10, 1e-10, sigma)

        # 使用 tanh 近似 CDF
        z = ret_np / sigma
        buy_pct = 0.5 * (1 + np.tanh(z * 0.7978))  # 近似正態 CDF
        buy_vol = buy_pct * vol_np
        sell_vol = (1 - buy_pct) * vol_np
        abs_imbalance = np.abs(buy_vol - sell_vol)

        # 20 期滾動 VPIN
        vpin = np.zeros(len(close_np))
        bucket_size = 20
        for i in range(bucket_size, len(close_np)):
            total_vol = vol_np[i-bucket_size:i].sum()
            if total_vol > 0:
                vpin[i] = abs_imbalance[i-bucket_size:i].sum() / total_vol
        # 標準化到 [-1, 1]
        vpin_mean = np.mean(vpin[bucket_size:]) if len(vpin) > bucket_size else 0
        vpin_std = np.std(vpin[bucket_size:]) if len(vpin) > bucket_size else 1
        if vpin_std < 1e-10:
            vpin_std = 1.0
        vpin_norm = np.clip((vpin - vpin_mean) / vpin_std, -3, 3) / 3.0
        df = df.with_columns(pl.Series("vpin", vpin_norm))
    else:
        df = df.with_columns(pl.lit(0.0).alias("vpin"))

    # ═══ v6.0 新增 2: Kyle's Lambda（價格衝擊係數）═══
    # lambda = |ret| / sqrt(volume) 的滾動平均
    # 高 lambda = 低流動性 = 價格容易被推動
    df = df.with_columns([
        (pl.col("ret1").abs()
         / (pl.col("volume").sqrt() + 1e-10)).alias("_kyle_raw"),
    ]).with_columns([
        pl.col("_kyle_raw").rolling_mean(10).alias("_kyle_ma"),
    ])
    if "_kyle_ma" in df.columns:
        kyle_arr = df["_kyle_ma"].to_numpy().astype(np.float64)
        kyle_arr = np.nan_to_num(kyle_arr, nan=0.0, posinf=0.0, neginf=0.0)
        k_mean = np.mean(kyle_arr[10:]) if len(kyle_arr) > 10 else 0
        k_std = np.std(kyle_arr[10:]) if len(kyle_arr) > 10 else 1
        if k_std < 1e-10:
            k_std = 1.0
        kyle_norm = np.clip((kyle_arr - k_mean) / k_std, -3, 3) / 3.0
        df = df.with_columns(pl.Series("kyle_lambda", kyle_norm))
    else:
        df = df.with_columns(pl.lit(0.0).alias("kyle_lambda"))

    # ═══ v6.0 新增 3: Amihud 非流動性指標 ═══
    # |ret| / volume 的 10 期滾動平均
    df = df.with_columns([
        (pl.col("ret1").abs()
         / (pl.col("volume") + 1e-10)).alias("_amihud_raw"),
    ]).with_columns([
        pl.col("_amihud_raw").rolling_mean(10).alias("_amihud_ma"),
    ])
    if "_amihud_ma" in df.columns:
        amihud_arr = df["_amihud_ma"].to_numpy().astype(np.float64)
        amihud_arr = np.nan_to_num(amihud_arr, nan=0.0, posinf=0.0, neginf=0.0)
        a_mean = np.mean(amihud_arr[10:]) if len(amihud_arr) > 10 else 0
        a_std = np.std(amihud_arr[10:]) if len(amihud_arr) > 10 else 1
        if a_std < 1e-10:
            a_std = 1.0
        amihud_norm = np.clip((amihud_arr - a_mean) / a_std, -3, 3) / 3.0
        df = df.with_columns(pl.Series("amihud_illiq", amihud_norm))
    else:
        df = df.with_columns(pl.lit(0.0).alias("amihud_illiq"))

    # ═══ v6.0 新增 4: 多時間框架趨勢對齊 ═══
    # 5 分鐘 MA10 方向 × 15 分鐘 MA10 方向（用 shift(3) 模擬 15 分鐘）
    # 同向 = +1，反向 = -1
    df = df.with_columns([
        (pl.col("close") - pl.col("close").shift(1)).sign().alias("_dir_5m"),
        (pl.col("close").rolling_mean(3) - pl.col("close").rolling_mean(3).shift(3)).sign().alias("_dir_15m"),
        (pl.col("close").rolling_mean(12) - pl.col("close").rolling_mean(12).shift(12)).sign().alias("_dir_1h"),
    ]).with_columns([
        # 三個時間框架的趨勢對齊度（-1 到 +1）
        ((pl.col("_dir_5m") + pl.col("_dir_15m") + pl.col("_dir_1h")) / 3.0).alias("mtf_trend_align"),
    ])

    # ═══ v6.0 新增 5: 多時間框架動量確認 ═══
    # ROC5 和 ROC20 同向且都超過閾值時 = 強確認
    df = df.with_columns([
        ((pl.col("close") / pl.col("close").shift(5) - 1).sign()
         * (pl.col("close") / pl.col("close").shift(20) - 1).sign()).alias("mtf_mom_confirm"),
    ])

    # ═══ v6.0 新增 6: 信號持續性 ═══
    # 最近 3 根 K 線的收益方向一致性
    if len(df) > 3:
        close_np = df["close"].to_numpy().astype(np.float64)
        ret_signs = np.zeros(len(close_np))
        ret_signs[1:] = np.sign(close_np[1:] - close_np[:-1])
        persist = np.zeros(len(close_np))
        for i in range(3, len(close_np)):
            # 最近 3 根 K 線方向的平均（-1 到 +1）
            persist[i] = ret_signs[i-2:i+1].mean()
        df = df.with_columns(pl.Series("signal_persistence", persist))
    else:
        df = df.with_columns(pl.lit(0.0).alias("signal_persistence"))

    # ═══ v6.0 新增 7: 市場狀態指標 ═══
    # 基於 Kaufman ER：ER > 0.5 = 趨勢，ER < 0.3 = 震盪
    # 標準化到 [-1, 1]：正值 = 趨勢，負值 = 震盪
    if "kaufman_er" in df.columns:
        df = df.with_columns([
            (pl.col("kaufman_er") * 2 - 1).clip(-1, 1).alias("regime_indicator"),
        ])
    else:
        df = df.with_columns(pl.lit(0.0).alias("regime_indicator"))

    # ═══ v6.0 新增 8: 收盤價位置 ═══
    # close_to_high: 收盤價距離最高價的相對位置（接近 0 = 接近最高）
    # close_to_low: 收盤價距離最低價的相對位置（接近 0 = 接近最低）
    df = df.with_columns([
        ((pl.col("high") - pl.col("close"))
         / (pl.col("high") - pl.col("low") + 1e-10)).alias("close_to_high"),
        ((pl.col("close") - pl.col("low"))
         / (pl.col("high") - pl.col("low") + 1e-10)).alias("close_to_low"),
    ])

    # ═══ v6.0 新增 9: 量價趨勢（Volume Price Trend）═══
    # VPT = cumsum(ret * volume) 的短期斜率
    df = df.with_columns([
        (pl.col("ret1") * pl.col("volume")).alias("_vpt_delta"),
    ]).with_columns([
        pl.col("_vpt_delta").cum_sum().alias("_vpt"),
    ]).with_columns([
        ((pl.col("_vpt") - pl.col("_vpt").shift(5))
         / (pl.col("volume").rolling_sum(5) + 1e-10)).alias("volume_price_trend"),
    ])

    # ═══ v6.0 新增 10: MFI 標準化（Money Flow Index）═══
    # 典型價格 = (H+L+C)/3
    # 正資金流 = TP > TP[-1] 時的 TP * Volume
    # MFI = 100 - 100/(1 + positive_flow/negative_flow)
    if len(df) > 14:
        h = df["high"].to_numpy().astype(np.float64)
        l = df["low"].to_numpy().astype(np.float64)
        c = df["close"].to_numpy().astype(np.float64)
        v = df["volume"].to_numpy().astype(np.float64)
        v = np.nan_to_num(v, nan=0.0)

        tp = (h + l + c) / 3.0
        mf = tp * v  # money flow

        mfi = np.zeros(len(c))
        period = 14
        for i in range(period + 1, len(c)):
            pos_flow = 0.0
            neg_flow = 0.0
            for j in range(i - period, i):
                if tp[j + 1] > tp[j]:
                    pos_flow += mf[j + 1]
                else:
                    neg_flow += mf[j + 1]
            if neg_flow > 0:
                mfi[i] = 100.0 - 100.0 / (1.0 + pos_flow / (neg_flow + 1e-10))
            else:
                mfi[i] = 100.0
        # 標準化到 [-0.5, 0.5]
        mfi_norm = (mfi / 100.0) - 0.5
        df = df.with_columns(pl.Series("mfi_norm", mfi_norm))
    else:
        df = df.with_columns(pl.lit(0.0).alias("mfi_norm"))

    # ═══ 訂單簿微觀結構因子 ═══
    if "obi" in df.columns:
        df = df.with_columns([
            (pl.col("obi") - pl.col("obi").shift(5)).alias("obi_momentum"),
            pl.col("obi").rolling_mean(10).alias("obi_ma"),
        ])
        df = df.with_columns([
            ((pl.col("obi") - pl.col("obi").shift(3))
             - (pl.col("obi").shift(3) - pl.col("obi").shift(6))).alias("obi_accel"),
        ])
        df = df.with_columns([
            (pl.col("obi").sign() * pl.col("close").pct_change().sign()).alias("obi_price_confirm"),
        ])

    if "spread" in df.columns:
        df = df.with_columns([
            pl.col("spread").rolling_mean(20).alias("_spread_ma"),
        ]).with_columns([
            ((pl.col("spread") - pl.col("_spread_ma"))
             / (pl.col("_spread_ma") + 1e-10)).alias("spread_norm"),
        ])

    if "bid_vol" in df.columns and "ask_vol" in df.columns:
        df = df.with_columns([
            ((pl.col("bid_vol") - pl.col("ask_vol"))
             / (pl.col("bid_vol") + pl.col("ask_vol") + 1e-10)).alias("bid_ask_imbalance"),
        ])

    # ═══ 時間因子 ═══
    if "timestamp" in df.columns:
        df = df.with_columns([
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
