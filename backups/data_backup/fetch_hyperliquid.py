import requests
from datetime import datetime, timezone

HL_API = "https://api.hyperliquid.xyz/info"

def fetch_hl_klines(symbol="BTC", interval="5m"):
    start = int((datetime.now(timezone.utc).timestamp() - 86400*30)*1000)
    end   = int(datetime.now(timezone.utc).timestamp()*1000)
    payload = {
        "type": "candleSnapshot",
        "req": {"coin": symbol, "interval": interval,
                "startTime": start, "endTime": end}
    }
    resp = requests.post(HL_API, json=payload, timeout=15)
    return resp.json()

def fetch_hl_orderbook(symbol="BTC"):
    payload = {"type": "l2Book", "coin": symbol}
    resp    = requests.post(HL_API, json=payload, timeout=15)
    data    = resp.json()
    levels  = data.get("levels", [[], []])
    bids    = levels[0][:10] if len(levels) > 0 else []
    asks    = levels[1][:10] if len(levels) > 1 else []
    bid_vol = sum(float(b["sz"]) for b in bids)
    ask_vol = sum(float(a["sz"]) for a in asks)
    spread  = float(asks[0]["px"]) - float(bids[0]["px"]) if bids and asks else 0
    obi     = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)
    return {"bid1": float(bids[0]["px"]) if bids else 0,
            "ask1": float(asks[0]["px"]) if asks else 0,
            "spread": spread, "bid_vol": bid_vol,
            "ask_vol": ask_vol, "obi": obi}

def fetch_hl_funding(symbol="BTC"):
    start   = int((datetime.now(timezone.utc).timestamp() - 86400*30)*1000)
    payload = {"type": "fundingHistory", "coin": symbol, "startTime": start}
    resp    = requests.post(HL_API, json=payload, timeout=15)
    return resp.json()

if __name__ == "__main__":
    print("测试 Hyperliquid API...")

    print("\n1. K线数据:")
    try:
        klines = fetch_hl_klines()
        print(f"   返回 {len(klines)} 根K线")
        print(f"   最新: {klines[-1]}")
    except Exception as e:
        print(f"   错误: {e}")

    print("\n2. 订单簿:")
    try:
        ob = fetch_hl_orderbook()
        print(f"   Bid1={ob['bid1']:.1f}  Ask1={ob['ask1']:.1f}")
        print(f"   Spread={ob['spread']:.2f}  OBI={ob['obi']:.4f}")
        print(f"   买方深度={ob['bid_vol']:.3f}  卖方深度={ob['ask_vol']:.3f}")
    except Exception as e:
        print(f"   错误: {e}")

    print("\n3. 资金费率:")
    try:
        funding = fetch_hl_funding()
        print(f"   记录数: {len(funding)}")
        if funding:
            print(f"   最新: {funding[-1]}")
    except Exception as e:
        print(f"   错误: {e}")
