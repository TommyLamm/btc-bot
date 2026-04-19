import polars as pl
import numpy as np
import sys
sys.path.insert(0, "/root/btc-bot")

df = pl.read_parquet("data/btc_5m_factors.parquet").drop_nulls()

# 未来收益
df = df.with_columns([
    pl.col("close").pct_change().shift(-1).alias("future_ret")
]).drop_nulls()

ret       = df["future_ret"].to_numpy()
timestamps = df["timestamp"].to_numpy()
month_ms  = 30 * 24 * 60 * 60 * 1000


def calc_ic(factor_vals):
    """计算月度 IC 均值和稳定性"""
    monthly_ics = []
    t_start = timestamps[0]
    while t_start < timestamps[-1]:
        t_end = t_start + month_ms
        mask  = (timestamps >= t_start) & (timestamps < t_end)
        if mask.sum() > 100:
            f_m = factor_vals[mask]
            r_m = ret[mask]
            valid = ~(np.isnan(f_m) | np.isnan(r_m) |
                      np.isinf(f_m) | np.isinf(r_m))
            if valid.sum() > 50:
                ic = np.corrcoef(f_m[valid], r_m[valid])[0, 1]
                if not np.isnan(ic):
                    monthly_ics.append(ic)
        t_start = t_end

    if len(monthly_ics) < 3:
        return 0.0, 0.0

    return np.mean(monthly_ics), np.mean(np.abs(monthly_ics) > 0.02)


# ── 候选因子生成 ──────────────────────────────────────────
close  = df["close"].to_numpy()
high   = df["high"].to_numpy()
low    = df["low"].to_numpy()
volume = df["volume"].to_numpy()

candidates = {}

# 1. 不同窗口的均线偏离
for w in [5, 8, 15, 20, 40, 60]:
    ma = pl.Series(close).rolling_mean(w).to_numpy()
    candidates[f"ma{w}_dev"] = close / (ma + 1e-10) - 1

# 2. 不同窗口的动量
for w in [3, 8, 12, 30, 45]:
    candidates[f"roc_{w}"] = np.roll(close, w) 
    candidates[f"roc_{w}"] = close / (np.concatenate([[np.nan]*w, close[:-w]]) + 1e-10) - 1

# 3. 区间位置（价格在近期高低点区间的位置）
for w in [10, 20, 40]:
    roll_min = pl.Series(low).rolling_min(w).to_numpy()
    roll_max = pl.Series(high).rolling_max(w).to_numpy()
    candidates[f"range_pos_{w}"] = (close - roll_min) / (roll_max - roll_min + 1e-10)

# 4. ATR 标准化收益
for w in [10, 20]:
    atr = pl.Series(high - low).rolling_mean(w).to_numpy()
    ret1 = np.concatenate([[np.nan], np.diff(close)])
    candidates[f"atr_ret_{w}"] = ret1 / (atr + 1e-10)

# 5. 成交量标准化价格冲击
vol_ma20 = pl.Series(volume).rolling_mean(20).to_numpy()
ret1 = np.concatenate([[np.nan], np.diff(close)])
candidates["price_impact"] = ret1 / (vol_ma20 / (volume + 1e-10) + 1e-10)

# 6. 偏离加速度（偏离在变大还是变小）
ma10 = pl.Series(close).rolling_mean(10).to_numpy()
ma10_dev = close / (ma10 + 1e-10) - 1
candidates["ma10_dev_accel"] = ma10_dev - np.concatenate([[np.nan]*3, ma10_dev[:-3]])

# 7. VWAP 不同窗口
for w in [10, 30, 60]:
    turnover = close * volume
    vwap = (pl.Series(turnover).rolling_sum(w).to_numpy() /
            (pl.Series(volume).rolling_sum(w).to_numpy() + 1e-10))
    candidates[f"vwap{w}_dev"] = close / (vwap + 1e-10) - 1

# ── 评估所有候选 ─────────────────────────────────────────
print(f"\n{'因子候选':<20} {'IC均值':>9} {'|IC|>0.02占比':>14} {'评级':>6}")
print("─" * 55)

found = []
for name, vals in candidates.items():
    vals = np.array(vals, dtype=np.float64)
    # 去极值（winsorize 1%~99%）
    p1, p99 = np.nanpercentile(vals, [1, 99])
    vals = np.clip(vals, p1, p99)

    ic_mean, sig_ratio = calc_ic(vals)

    if abs(ic_mean) >= 0.015:
        if abs(ic_mean) > 0.025:   grade = "⭐⭐⭐"
        elif abs(ic_mean) > 0.018: grade = "⭐⭐"
        else:                      grade = "⭐"
        found.append((name, ic_mean, sig_ratio, grade))

found.sort(key=lambda x: abs(x[1]), reverse=True)
for name, ic, sr, grade in found:
    print(f"{name:<20} {ic:>+9.4f} {sr:>14.2f} {grade:>6}")

if not found:
    print("未发现 IC > 0.015 的新因子")
else:
    print(f"\n共发现 {len(found)} 个有效候选因子")
    print("建议将 ⭐⭐ 以上的加入 FACTOR_COLS")
