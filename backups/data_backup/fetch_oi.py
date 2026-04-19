import ccxt
import polars as pl
import time
from datetime import datetime, timezone, timedelta

def fetch_open_interest(save_path="data/btc_oi.parquet"):
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"}
    })

    # Binance 持仓量历史只支持最近30天
    since = int((datetime.now(timezone.utc) - timedelta(days=29))
                .timestamp() * 1000)
    all_records = []

    print("开始拉取持仓量历史（最近30天）...")
    while True:
        try:
            records = exchange.fetch_open_interest_history(
                "BTC/USDT", "5m", since=since, limit=500
            )
        except Exception as e:
            print(f"错误: {e}")
            break

        if not records:
            break

        all_records.extend(records)
        since = records[-1]["timestamp"] + 1
        dt = datetime.fromtimestamp(records[-1]["timestamp"]/1000, tz=timezone.utc)
        print(f"  已拉取至 {dt.strftime('%Y-%m-%d %H:%M')}，共 {len(all_records)} 条")

        if records[-1]["timestamp"] >= exchange.milliseconds() - 5*60*1000:
            break
        time.sleep(0.5)

    if not all_records:
        print("无数据")
        return None

    df = pl.DataFrame([{
        "timestamp":     r["timestamp"],
        "open_interest": r["openInterestAmount"],
    } for r in all_records]).with_columns([
        pl.col("timestamp").cast(pl.Datetime("ms", "UTC")).alias("datetime")
    ])

    df.write_parquet(save_path)
    print(f"完成，共 {len(df)} 条持仓量数据")
    return df

if __name__ == "__main__":
    df = fetch_open_interest()
    if df is not None:
        print(df.tail(3))
