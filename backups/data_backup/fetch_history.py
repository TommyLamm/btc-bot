# data/fetch_history.py
import ccxt
import polars as pl
from datetime import datetime, timezone
import time
import os

def fetch_ohlcv_history(
    symbol: str = "BTC/USDT",
    timeframe: str = "5m",
    start_date: str = "2023-01-01",
    save_path: str = "data/btc_5m.parquet"
):
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"}  # 用永续合约，有资金费率
    })

    since = int(datetime.strptime(start_date, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc).timestamp() * 1000)

    all_candles = []
    print(f"开始拉取 {symbol} {timeframe} 历史数据...")

    while True:
        candles = exchange.fetch_ohlcv(
            symbol, timeframe,
            since=since,
            limit=1000
        )

        if not candles:
            break

        all_candles.extend(candles)
        since = candles[-1][0] + 1  # 下一批从最后一根的下一毫秒开始

        latest = datetime.fromtimestamp(candles[-1][0] / 1000, tz=timezone.utc)
        print(f"  已拉取至 {latest.strftime('%Y-%m-%d %H:%M')}，共 {len(all_candles)} 根")

        # 拉到当前时间就停
        if candles[-1][0] >= exchange.milliseconds() - 5 * 60 * 1000:
            break

        time.sleep(exchange.rateLimit / 1000)

    # 转成 Polars DataFrame
    df = pl.DataFrame(
        all_candles,
        schema=["timestamp", "open", "high", "low", "close", "volume"],
        orient="row"
    ).with_columns([
        (pl.col("timestamp") / 1000).cast(pl.Int64)
          .cast(pl.Datetime("ms", "UTC")).alias("datetime"),
    ])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.write_parquet(save_path)
    print(f"\n完成，共 {len(df)} 根K线，已保存至 {save_path}")
    return df


if __name__ == "__main__":
    df = fetch_ohlcv_history(start_date="2022-01-01")
    print(df.tail(5))
