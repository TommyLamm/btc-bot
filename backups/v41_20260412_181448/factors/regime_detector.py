"""
BTC-Bot 市場狀態分類器（Regime Detector）
自動識別趨勢市/震盪市/高波動，切換因子組合。
"""

import numpy as np


class RegimeDetector:
    def __init__(
        self,
        adx_period: int = 14,
        adx_trending_threshold: float = 25.0,
        adx_ranging_threshold: float = 18.0,
        vol_ratio_period_short: int = 10,
        vol_ratio_period_long: int = 50,
        vol_expansion_threshold: float = 1.5,
    ):
        self.adx_period = adx_period
        self.adx_trending_threshold = adx_trending_threshold
        self.adx_ranging_threshold = adx_ranging_threshold
        self.vol_ratio_period_short = vol_ratio_period_short
        self.vol_ratio_period_long = vol_ratio_period_long
        self.vol_expansion_threshold = vol_expansion_threshold
        self.current_regime = "unknown"
        self.regime_history = []
        self.regime_confidence = 0.0

    def _compute_adx(self, high, low, close):
        n = len(close)
        period = self.adx_period
        if n < period * 3:
            return 20.0
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up = high[i] - high[i-1]
            down = low[i-1] - low[i]
            if up > down and up > 0:
                plus_dm[i] = up
            if down > up and down > 0:
                minus_dm[i] = down
        atr = np.zeros(n)
        s_plus_dm = np.zeros(n)
        s_minus_dm = np.zeros(n)
        atr[period] = np.mean(tr[1:period+1])
        s_plus_dm[period] = np.mean(plus_dm[1:period+1])
        s_minus_dm[period] = np.mean(minus_dm[1:period+1])
        for i in range(period + 1, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            s_plus_dm[i] = (s_plus_dm[i-1] * (period - 1) + plus_dm[i]) / period
            s_minus_dm[i] = (s_minus_dm[i-1] * (period - 1) + minus_dm[i]) / period
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        for i in range(period, n):
            if atr[i] > 0:
                plus_di[i] = 100 * s_plus_dm[i] / atr[i]
                minus_di[i] = 100 * s_minus_dm[i] / atr[i]
        dx = np.zeros(n)
        for i in range(period, n):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        start = period * 2
        if start >= n:
            return 20.0
        adx = np.zeros(n)
        adx[start] = np.mean(dx[period:start+1])
        for i in range(start + 1, n):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        return float(adx[-1])

    def _compute_vol_ratio(self, close):
        n = len(close)
        if n < self.vol_ratio_period_long + 1:
            return 1.0
        returns = np.diff(close) / close[:-1]
        returns = np.nan_to_num(returns, nan=0.0)
        short_vol = np.std(returns[-self.vol_ratio_period_short:])
        long_vol = np.std(returns[-self.vol_ratio_period_long:])
        if long_vol < 1e-10:
            return 1.0
        return float(short_vol / long_vol)

    def _compute_trend_consistency(self, close):
        n = len(close)
        if n < 20:
            return 0.0
        returns = np.diff(close[-21:]) / close[-21:-1]
        returns = np.nan_to_num(returns, nan=0.0)
        if len(returns) == 0:
            return 0.0
        positive = np.sum(returns > 0)
        negative = np.sum(returns < 0)
        return float(abs(positive - negative) / len(returns))

    def detect(self, high, low, close):
        adx = self._compute_adx(high, low, close)
        vol_ratio = self._compute_vol_ratio(close)
        trend_consistency = self._compute_trend_consistency(close)
        trend_score = 0.0
        range_score = 0.0
        volatile_score = 0.0
        if adx > self.adx_trending_threshold:
            trend_score += (adx - self.adx_trending_threshold) / 20.0
        elif adx < self.adx_ranging_threshold:
            range_score += (self.adx_ranging_threshold - adx) / 10.0
        if vol_ratio > self.vol_expansion_threshold:
            volatile_score += (vol_ratio - self.vol_expansion_threshold)
            trend_score += 0.3
        elif vol_ratio < 0.8:
            range_score += (0.8 - vol_ratio) * 2
        if trend_consistency > 0.4:
            trend_score += trend_consistency
        else:
            range_score += (0.4 - trend_consistency)
        scores = {"trending": trend_score, "ranging": range_score, "volatile": volatile_score}
        regime = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[regime] / (total_score + 1e-10)
        factor_weights = self._get_factor_weights(regime, confidence)
        old_regime = self.current_regime
        self.current_regime = regime
        self.regime_confidence = confidence
        self.regime_history.append(regime)
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-500:]
        if regime != old_regime and old_regime != "unknown":
            print(f"  市場狀態切換: {old_regime} -> {regime} (置信度={confidence:.2f})")
        return {
            "regime": regime, "confidence": confidence,
            "adx": adx, "vol_ratio": vol_ratio,
            "trend_consistency": trend_consistency,
            "factor_weights": factor_weights,
        }

    def _get_factor_weights(self, regime, confidence):
        weights = {
            "roc5": 1.0, "roc20": 1.0, "ma10dev": 1.0, "ma30dev": 1.0,
            "vwapdev": 1.0, "priceimpact": 1.0, "macdhist": 1.0,
            "oi_roc6": 1.0, "oi_roc24": 1.0, "price_oi_confirm": 1.0,
            "vol_impact_hl": 1.0, "candle_dir": 1.0,
        }
        strength = min(confidence, 0.8)
        if regime == "trending":
            for f in ["roc20", "macdhist", "oi_roc24", "price_oi_confirm"]:
                weights[f] = 1.0 + strength * 0.5
            for f in ["ma10dev", "ma30dev", "vwapdev", "candle_dir"]:
                weights[f] = 1.0 - strength * 0.3
        elif regime == "ranging":
            for f in ["ma10dev", "ma30dev", "vwapdev", "candle_dir"]:
                weights[f] = 1.0 + strength * 0.5
            for f in ["roc20", "macdhist", "oi_roc24"]:
                weights[f] = 1.0 - strength * 0.3
        elif regime == "volatile":
            for f in weights:
                weights[f] = 1.0 - strength * 0.4
        return weights

    def get_regime_stats(self):
        if not self.regime_history:
            return {"current": "unknown", "trending_pct": 0, "ranging_pct": 0, "volatile_pct": 0}
        recent = self.regime_history[-100:]
        total = len(recent)
        return {
            "current": self.current_regime,
            "confidence": self.regime_confidence,
            "trending_pct": sum(1 for r in recent if r == "trending") / total * 100,
            "ranging_pct": sum(1 for r in recent if r == "ranging") / total * 100,
            "volatile_pct": sum(1 for r in recent if r == "volatile") / total * 100,
        }
