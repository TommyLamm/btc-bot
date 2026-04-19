import requests
import json

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

def get_btc_markets():
    url    = GAMMA_API + "/markets"
    params = {"limit": 100, "active": "true"}

    resp    = requests.get(url, params=params, timeout=15)
    markets = resp.json()

    btc_markets = []
    for m in markets:
        question = m.get("question", "").lower()
        if "bitcoin" in question or "btc" in question:
            btc_markets.append({
                "condition_id": m.get("conditionId"),
                "question":     m.get("question"),
                "end_date":     m.get("endDate"),
                "volume":       m.get("volume"),
                "liquidity":    m.get("liquidity"),
                "tokens":       m.get("tokens", []),
            })
    return btc_markets


def get_price_history(token_id: str):
    url    = CLOB_API + "/prices-history"
    params = {"market": token_id, "interval": "1m", "fidelity": 5}
    resp   = requests.get(url, params=params, timeout=15)
    return resp.json()


if __name__ == "__main__":
    print("搜索 BTC 相关市场...")
    markets = get_btc_markets()
    print(f"找到 {len(markets)} 个 BTC 市场\n")

    for m in markets[:10]:
        vol = float(m["volume"] or 0)
        liq = float(m["liquidity"] or 0)
        prices = [t.get("price") for t in m["tokens"]]
        token_ids = [t.get("token_id") for t in m["tokens"]]

        print(f"问题:   {m['question']}")
        print(f"  到期:   {m['end_date']}")
        print(f"  成交量: ${vol:,.0f}")
        print(f"  流动性: ${liq:,.0f}")
        print(f"  价格:   {prices}")
        print(f"  TokenID:{[t[:16]+'...' for t in token_ids if t]}")
        print()

    with open("data/poly_markets.json", "w") as f:
        json.dump(markets, f, indent=2, ensure_ascii=False)
    print("已保存至 data/poly_markets.json")

    # 拉取流动性最高的市场的价格历史
    best = max(markets, key=lambda m: float(m["liquidity"] or 0))
    print(f"\n流动性最高: {best['question']}")
    if best["tokens"]:
        token_id = best["tokens"][0].get("token_id")
        if token_id:
            print(f"拉取价格历史 token: {token_id[:20]}...")
            history = get_price_history(token_id)
            print(json.dumps(history, indent=2)[:800])
