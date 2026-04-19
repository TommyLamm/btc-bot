import ccxt
import polars as pl
import time
from datetime import datetime, timezone

def fetch_funding_history(
    symbol="BTC/USDT",
    start_date="2022-01-01",
    save_path="data/btc_funding.parquet"
):
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"}
    })

    since = int(datetime.strptime(start_date, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc).timestamp() * 1000)

    all_records = []
    print(f"开始拉取资金费率历史...")

    while True:
        records = exchange.fetch_funding_rate_history(
            symbol, since=since, limit=1000
        )
        if not records:
            break

        all_records.extend(records)
        since = records[-1]["timestamp"] + 1
        dt = datetime.fromtimestamp(records[-1]["timestamp"]/1000, tz=timezone.utc)
        print(f"  已拉取至 {dt.strftime('%Y-%m-%d %H:%M')}，共 {len(all_records)} 条")

        if records[-1]["timestamp"] >= exchange.milliseconds() - 8*3600*1000:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pl.DataFrame([{
        "timestamp":    r["timestamp"],
        "funding_rate": r["fundingRate"],
    } for r in all_records]).with_columns([
        pl.col("timestamp").cast(pl.Datetime("ms", "UTC")).alias("datetime")
    ])

    df.write_parquet(save_path)
    print(f"完成，共 {len(df)} 条资金费率记录")
    return df

if __name__ == "__main__":
    df = fetch_funding_history(start_date="2022-01-01")
    print(df.tail(5))
