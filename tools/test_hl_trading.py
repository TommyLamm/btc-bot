"""
Hyperliquid 交易測試腳本（統一帳戶版）
測試流程：連接 → 查餘額 → 設槓桿 → 開多 → 等 5 秒 → 平倉
已驗證：2026-04-11 成功
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from executor.hyperliquid_executor import HyperliquidExecutor

# ═══ 配置 ═══
SECRET_KEY = os.environ.get(
    "HL_SECRET_KEY",
    "0xf3ddcc9a128d6251033058f671450a929b32d47389886c664f4497fe1c100e99",
)
ACCOUNT_ADDRESS = os.environ.get(
    "HL_ACCOUNT_ADDRESS",
    "0x73531FF0bAf770EaFDc25003fAD6E33F7dB682EB",
)
LEVERAGE = 2
POSITION_SIZE_USD = 10.0
COIN = "BTC"


def main():
    print("=" * 60)
    print("Hyperliquid 交易測試（統一帳戶版）")
    print("=" * 60)

    print("\n[1/6] 連接帳戶...")
    try:
        executor = HyperliquidExecutor(
            secret_key=SECRET_KEY,
            account_address=ACCOUNT_ADDRESS,
            leverage=LEVERAGE,
            position_size_usd=POSITION_SIZE_USD,
            coin=COIN,
        )
    except Exception as e:
        print(f"  連接失敗: {e}")
        return

    print("\n[2/6] 查詢餘額...")
    balance = executor.get_balance()
    print(f"  可用餘額: ${balance:.2f}")

    if balance < 5:
        print("\n帳戶餘額不足！至少需要 $5 USDC")
        return

    print("\n[3/6] 查詢 BTC 價格...")
    price = executor._get_current_price()
    print(f"  當前價格: ${price:,.1f}")
    size = executor._calc_size(price)
    print(f"  計劃下單: {size} BTC (名義 ${size * price:,.2f})")

    print("\n[4/6] 開多倉測試...")
    result = executor.open_long("測試下單")
    if result["status"] == "ok":
        print(f"  成交: {result['size']} BTC @ ${result['price']:,.1f}")
    else:
        print(f"  失敗: {result.get('msg', result)}")
        return

    print("\n[5/6] 等待 5 秒...")
    for i in range(5, 0, -1):
        print(f"  {i}...", end=" ", flush=True)
        time.sleep(1)
    print()

    print("\n[6/6] 平倉測試...")
    result = executor.close_position("測試平倉")
    if result["status"] == "ok":
        print(f"  平倉成交: {result['size']} BTC @ ${result['price']:,.1f}")
        print(f"  測試損益: ${result['pnl']:+.4f}")
    else:
        print(f"  平倉失敗: {result.get('msg', result)}")
        return

    print("\n" + "=" * 60)
    print("測試完成！")
    print(f"  {executor.status_summary()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
