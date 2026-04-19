#!/bin/bash
cd /root/btc-bot
source /root/btc-bot-env/bin/activate

echo "$(date) 开始每日数据更新..."

# 增量更新K线
python -c "
import ccxt, polars as pl

exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
df = pl.read_parquet('data/btc_5m.parquet')
since = int(df['timestamp'].max()) + 1

candles = []
while True:
    batch = exchange.fetch_ohlcv('BTC/USDT', '5m', since=since, limit=1000)
    if not batch: break
    candles.extend(batch)
    since = batch[-1][0] + 1
    if batch[-1][0] >= exchange.milliseconds() - 300000: break

if candles:
    new = pl.DataFrame(candles,
        schema=['timestamp','open','high','low','close','volume'], orient='row')
    df = pl.concat([df, new], how='diagonal')
    df.write_parquet('data/btc_5m.parquet')
    print(f'K线新增 {len(candles)} 根，总计 {len(df)} 根')
else:
    print('K线无新数据')
"

# 更新持仓量
python data/fetch_oi.py

# 重新计算因子
python factors/factor_engine.py

echo "$(date) 更新完成"
