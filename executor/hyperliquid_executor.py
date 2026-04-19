"""
Hyperliquid 永續合約交易執行器 v3 — 動態倉位版
改進：
  1. 支援動態倉位：open_long/open_short 接受 position_size_usd 參數
  2. 平倉前同步 HL 實際倉位，避免 "asset=0" 錯誤
  3. 開倉前也同步倉位，避免重複開倉
  4. 更好的錯誤處理和日誌
"""

import time
import json
import os
import math
from datetime import datetime

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account


class HyperliquidExecutor:
    """Hyperliquid 實盤交易執行器（統一帳戶版，支援動態倉位）"""

    def __init__(
        self,
        secret_key: str,
        account_address: str,
        leverage: int = 3,
        default_position_size_usd: float = 237.0,
        coin: str = "BTC",
        log_path: str = "logs/trades.jsonl",
    ):
        self.account_address = account_address
        self.leverage = leverage
        self.default_position_size_usd = default_position_size_usd
        self.coin = coin
        self.log_path = log_path

        # 初始化 SDK（統一帳戶：用 account_address，不用 vault_address）
        wallet = Account.from_key(secret_key)
        self.info = Info(constants.MAINNET_API_URL)
        self.exchange = Exchange(
            wallet,
            constants.MAINNET_API_URL,
            account_address=account_address,
        )

        # 查詢資產精度
        self.sz_decimals = 5  # BTC default
        self._load_meta()

        # 狀態
        self.current_position = None  # "long" / "short" / None
        self.entry_price = 0.0
        self.position_size = 0.0  # BTC 數量
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

        # 確保日誌目錄存在
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # 設置槓桿
        self._set_leverage()

        # 同步持倉狀態
        self._sync_position()

    def _load_meta(self):
        """從 API 載入資產精度資訊"""
        try:
            meta = self.info.meta()
            for asset in meta.get("universe", []):
                if asset.get("name") == self.coin:
                    self.sz_decimals = asset.get("szDecimals", 5)
                    break
        except Exception:
            pass

    def _set_leverage(self):
        """設置槓桿倍數"""
        try:
            result = self.exchange.update_leverage(
                self.leverage, self.coin, is_cross=True
            )
            print(f"  [HL] 槓桿已設置: {self.leverage}x ({self.coin})")
            return result
        except Exception as e:
            print(f"  [HL] 設置槓桿失敗: {e}")
            return None

    def get_balance(self) -> float:
        """查詢可用餘額（支援統一帳戶）"""
        try:
            state = self.info.user_state(self.account_address)
            perps_val = float(state.get("marginSummary", {}).get("accountValue", "0"))
            if perps_val > 0:
                return perps_val
            spot = self.info.spot_user_state(self.account_address)
            for b in spot.get("balances", []):
                if b["coin"] == "USDC":
                    total = float(b.get("total", "0"))
                    hold = float(b.get("hold", "0"))
                    return total - hold
            return 0.0
        except Exception as e:
            print(f"  [HL] 查詢餘額失敗: {e}")
            return 0.0

    def get_account_info(self) -> dict:
        """查詢帳戶完整資訊"""
        balance = self.get_balance()
        return {
            "balance": balance,
            "position": self.current_position,
            "entry_price": self.entry_price,
            "position_size": self.position_size,
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
        }

    def _sync_position(self):
        """同步當前持倉狀態（從 HL API 讀取實際倉位）"""
        try:
            state = self.info.user_state(self.account_address)
            positions = state.get("assetPositions", [])
            for pos in positions:
                p = pos.get("position", {})
                if p.get("coin") == self.coin:
                    szi = float(p.get("szi", "0"))
                    if abs(szi) > 1e-10:
                        self.current_position = "long" if szi > 0 else "short"
                        self.entry_price = float(p.get("entryPx", "0"))
                        self.position_size = abs(szi)
                        return
            self.current_position = None
            self.entry_price = 0.0
            self.position_size = 0.0
        except Exception as e:
            print(f"  [HL] 同步持倉失敗: {e}")

    def _get_current_price(self) -> float:
        """取得當前 BTC 價格"""
        try:
            mids = self.info.all_mids()
            return float(mids.get(self.coin, "0"))
        except Exception:
            return 0.0

    @staticmethod
    def _round_price_hl(price: float, sz_decimals: int = 5, is_buy: bool = True) -> float:
        """將價格調整為 Hyperliquid 接受的格式"""
        if price <= 0:
            return 0
        max_decimals = max(0, 6 - sz_decimals)
        factor = 10 ** max_decimals
        if is_buy:
            rounded = math.ceil(price * factor) / factor
        else:
            rounded = math.floor(price * factor) / factor
        if rounded >= 10 and rounded != int(rounded):
            magnitude = math.floor(math.log10(abs(rounded)))
            if magnitude >= 4:
                step = 10 ** max(0, magnitude - 4)
                if is_buy:
                    rounded = math.ceil(price / step) * step
                else:
                    rounded = math.floor(price / step) * step
        elif rounded >= 100000:
            magnitude = math.floor(math.log10(abs(rounded)))
            step = 10 ** max(0, magnitude - 4)
            if is_buy:
                rounded = math.ceil(price / step) * step
            else:
                rounded = math.floor(price / step) * step
        return rounded

    def _calc_size(self, price: float, position_size_usd: float = None) -> float:
        """計算下單數量（BTC），支援動態倉位"""
        if price <= 0:
            return 0.0
        usd = position_size_usd if position_size_usd else self.default_position_size_usd
        notional = usd * self.leverage
        raw_size = notional / price
        size = round(raw_size, self.sz_decimals)
        # 確保最小名義價值 >= $10（Hyperliquid 最低要求）
        if size * price < 10:
            size = round(10.0 / price + 10 ** (-self.sz_decimals), self.sz_decimals)
        return size

    def open_long(self, reason: str = "", position_size_usd: float = None) -> dict:
        """開多倉（支援動態倉位）"""
        return self._open_position(True, reason, position_size_usd)

    def open_short(self, reason: str = "", position_size_usd: float = None) -> dict:
        """開空倉（支援動態倉位）"""
        return self._open_position(False, reason, position_size_usd)

    def _open_position(self, is_buy: bool, reason: str = "", position_size_usd: float = None) -> dict:
        """開倉（市價單，用 IOC 限價單模擬）"""
        direction = "long" if is_buy else "short"

        self._sync_position()

        if self.current_position == direction:
            return {"status": "skip", "msg": f"已有 {direction} 持倉"}

        if self.current_position is not None:
            close_result = self.close_position(f"反向開倉: {direction}")
            if close_result["status"] == "error":
                print(f"  [HL] 反向平倉失敗，取消開倉: {close_result.get('msg', '')}")
                return {"status": "error", "msg": f"反向平倉失敗: {close_result.get('msg', '')}"}

        price = self._get_current_price()
        if price <= 0:
            return {"status": "error", "msg": "無法取得價格"}

        size = self._calc_size(price, position_size_usd)
        if size <= 0:
            return {"status": "error", "msg": "計算數量失敗"}

        usd_used = position_size_usd or self.default_position_size_usd

        try:
            slippage = 0.005
            if is_buy:
                limit_price = self._round_price_hl(price * (1 + slippage), self.sz_decimals, is_buy=True)
            else:
                limit_price = self._round_price_hl(price * (1 - slippage), self.sz_decimals, is_buy=False)

            result = self.exchange.order(
                self.coin, is_buy, size, limit_price,
                {"limit": {"tif": "Ioc"}},
            )

            status = result.get("status", "")
            response = result.get("response", {})

            if status == "ok":
                statuses = response.get("data", {}).get("statuses", [])
                if statuses and "filled" in statuses[0]:
                    filled = statuses[0]["filled"]
                    fill_price = float(filled.get("avgPx", price))
                    fill_size = float(filled.get("totalSz", size))

                    self.current_position = direction
                    self.entry_price = fill_price
                    self.position_size = fill_size
                    self.trade_count += 1

                    self._log_trade("open", direction, fill_price, fill_size, reason,
                                    position_usd=usd_used)

                    return {
                        "status": "ok",
                        "direction": direction,
                        "price": fill_price,
                        "size": fill_size,
                        "position_usd": usd_used,
                        "oid": filled.get("oid", ""),
                    }
                elif statuses and "error" in statuses[0]:
                    err = statuses[0]["error"]
                    print(f"  [HL] 開倉錯誤: {err}")
                    return {"status": "error", "msg": err}
                else:
                    print(f"  [HL] 開倉未成交（IOC expired）: {statuses}")
                    return {"status": "error", "msg": f"IOC 未成交: {statuses}"}
            else:
                return {"status": "error", "msg": f"下單失敗: {result}"}

        except Exception as e:
            return {"status": "error", "msg": f"下單異常: {e}"}

    def close_position(self, reason: str = "") -> dict:
        """平倉（先同步 HL 實際倉位）"""
        self._sync_position()

        if self.current_position is None:
            return {"status": "skip", "msg": "無持倉（HL 確認）"}

        is_buy = self.current_position == "short"
        price = self._get_current_price()
        if price <= 0:
            return {"status": "error", "msg": "無法取得價格"}

        size = self.position_size
        if size <= 0:
            self.current_position = None
            return {"status": "skip", "msg": "持倉數量為 0"}

        try:
            slippage = 0.005
            if is_buy:
                limit_price = self._round_price_hl(price * (1 + slippage), self.sz_decimals, is_buy=True)
            else:
                limit_price = self._round_price_hl(price * (1 - slippage), self.sz_decimals, is_buy=False)

            result = self.exchange.order(
                self.coin, is_buy, size, limit_price,
                {"limit": {"tif": "Ioc"}},
                reduce_only=True,
            )

            status = result.get("status", "")
            response = result.get("response", {})

            if status == "ok":
                statuses = response.get("data", {}).get("statuses", [])
                if statuses and "filled" in statuses[0]:
                    filled = statuses[0]["filled"]
                    fill_price = float(filled.get("avgPx", price))

                    if self.current_position == "long":
                        pnl = (fill_price - self.entry_price) * self.position_size
                    else:
                        pnl = (self.entry_price - fill_price) * self.position_size

                    self.total_pnl += pnl
                    if pnl > 0:
                        self.win_count += 1
                    direction = self.current_position

                    self._log_trade("close", direction, fill_price, size, reason, pnl)

                    self.current_position = None
                    self.entry_price = 0.0
                    self.position_size = 0.0

                    return {
                        "status": "ok",
                        "direction": direction,
                        "price": fill_price,
                        "size": size,
                        "pnl": pnl,
                    }
                elif statuses and "error" in statuses[0]:
                    err = statuses[0]["error"]
                    if "Reduce only" in err or "asset=0" in str(err):
                        print(f"  [HL] HL 上已無倉位，重置本地狀態")
                        self.current_position = None
                        self.entry_price = 0.0
                        self.position_size = 0.0
                        return {"status": "skip", "msg": "HL 上已無倉位"}
                    print(f"  [HL] 平倉錯誤: {err}")
                    return {"status": "error", "msg": err}
                else:
                    print(f"  [HL] 平倉未成交（IOC expired）: {statuses}")
                    return {"status": "error", "msg": f"IOC 未成交: {statuses}"}
            else:
                return {"status": "error", "msg": f"平倉失敗: {result}"}

        except Exception as e:
            return {"status": "error", "msg": f"平倉異常: {e}"}

    def _log_trade(self, action, direction, price, size, reason="", pnl=None, position_usd=None):
        """記錄交易日誌"""
        record = {
            "time": datetime.utcnow().isoformat(),
            "action": action,
            "direction": direction,
            "coin": self.coin,
            "price": price,
            "size": size,
            "position_usd": position_usd,
            "reason": reason,
            "pnl": pnl,
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
        }
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def status_summary(self) -> str:
        """一行狀態摘要"""
        balance = self.get_balance()
        pos_str = (
            f"{self.current_position} {self.position_size:.5f}@{self.entry_price:.1f}"
            if self.current_position
            else "空倉"
        )
        wr = f"{self.win_count}/{self.trade_count}" if self.trade_count > 0 else "0/0"
        return (
            f"[HL] 餘額=${balance:.2f} | {pos_str} | "
            f"累計PnL=${self.total_pnl:+.2f} | 勝={wr} | 交易={self.trade_count}筆"
        )

