
import requests
import polars as pl
import numpy as np
import sys
sys.path.insert(0, "/root/btc-bot")

HL_API = "[api.hyperliquid.xyz](https://api.hyperliquid.xyz/info)"

def fetch_hl_obi_history(symbol="BTC", interval="5m"):
    from datetime import datetime, timezone
    start = int((datetime.now(timezone.utc).timestamp() - 86400*30)*1000)
    end   = int(datetime.now(timezone.utc).timestamp()*1000)
    payload = {"type": "candleSnapshot", "req": {"coin": symbol, "interval": interval, "startTime": start, "endTime": end}}
    resp = requests.post(HL_API, json=payload, timeout=15)
    candles = resp.json()
    rows = []
    for c in candles:
        rows.append({"timestamp": int(c["t"]), "open": float(c["o"]), "high": float(c["h"]), "low": float(c["l"]), "close": float(c["c"]), "volume": float(c["v"]), "n_trades": int(c["n"])})
    df = pl.DataFrame(rows)
    df = df.with_columns([
        ((pl.col("close") - pl.col("open")) / (pl.col("high") - pl.col("low") + 1e-10)).alias("candle_dir"),
        (pl.col("close").pct_change() * pl.col("volume") / (pl.col("volume").rolling_mean(20) + 1e-10)).alias("vol_impact"),
        ((pl.col("high") - pl.col("close")) / (pl.col("high") - pl.col("low") + 1e-10)).alias("upper_shadow"),
        ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-10)).alias("lower_shadow"),
        (pl.col("n_trades") / pl.col("n_trades").rolling_mean(20)).alias("trade_intensity"),
        pl.col("close").pct_change().shift(-1).alias("future_ret"),
    ]).drop_nulls()
    return df

if __name__ == "__main__":
    print("拉取 Hyperliquid 5m K线（30天）...")
    df = fetch_hl_obi_history()
    print(f"数据量: {len(df)} 根K线")

    factor_cols = ["candle_dir", "vol_impact", "upper_shadow", "lower_shadow", "trade_intensity"]
    ret = df["future_ret"].to_numpy()

    print()
    print(f"{'因子':<22} {'全样本IC':>10} {'评级':>6}")
    print("-" * 42)
    for col in factor_cols:
        f = df[col].to_numpy().astype(float)
        p1, p99 = np.nanpercentile(f, [1, 99])
        f = np.clip(f, p1, p99)
        valid = ~(np.isnan(f) | np.isnan(ret) | np.isinf(f))
        if valid.sum() < 100:
            continue
        ic = np.corrcoef(f[valid], ret[valid])[0, 1]
        grade = "3star" if abs(ic)>0.025 else "2star" if abs(ic)>0.018 else "1star" if abs(ic)>0.012 else "x"
        print(f"{col:<22} {ic:>+10.4f} {grade:>6}")

    print()
    print("对比 Binance ma10_dev（最近30天）:")
    btc = pl.read_parquet("data/btc_5m_factors.parquet").tail(8640).drop_nulls()
    btc = btc.with_columns([pl.col("close").pct_change().shift(-1).alias("future_ret")]).drop_nulls()
    f = btc["ma10_dev"].to_numpy()
    r = btc["future_ret"].to_numpy()
    valid = ~(np.isnan(f)|np.isnan(r))
    ic = np.corrcoef(f[valid], r[valid])[0, 1]
    print(f"  ma10_dev IC = {ic:+.4f}")
