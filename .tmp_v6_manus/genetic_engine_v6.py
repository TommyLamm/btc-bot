"""
BTC-Bot 遺傳搜索引擎 v6.0 — 勝率極致優化版
核心改進：
  1. 適應度以「模擬勝率」為第一優先（WR < 48% 直接淘汰）
  2. 信號確認機制：連續 2 根 K 線信號同向才開倉
  3. 信號置信度過濾：|signal| 必須超過動態閾值的 1.2 倍
  4. 滑點模擬：每筆交易額外扣除 0.01%
  5. 更嚴格的 Walk-forward：6 段驗證
  6. 勝率加權 + 盈虧比聯合優化
  7. 因子 IC 篩選 + 月度穩定性 + Hit Rate 三重篩選
  8. 動態止損：根據信號強度調整止損距離
  9. 保留 v5.5 的交易模擬、交互項、連續虧損懲罰
"""

import polars as pl
import numpy as np
import random
from typing import Optional

FACTOR_COLS = [
    # 動量
    "roc5", "roc20", "mom_accel", "roc15m", "roc1h",
    # 均線
    "ma10dev", "ma30dev", "ma_alignment",
    # VWAP
    "vwapdev", "vwap_slope",
    # 波動率
    "bb_pctb", "bb_width", "vol_ratio", "tr_ratio",
    # 成交量
    "vol_surge", "vol_price_div", "obv_slope", "vol_accel",
    # 技術指標
    "priceimpact", "macdhist", "rsi_norm", "rsi_divergence",
    # 微觀結構
    "candle_dir",
    # 持倉量
    "oi_roc6", "oi_roc24", "price_oi_confirm",
    # 時間
    "hour_sin", "hour_cos",
    # v5.5 因子
    "momentum_quality", "kaufman_er", "vol_regime_change",
    "range_expansion", "consec_candles", "return_skew",
    "obi_accel", "obi_price_confirm", "spread_norm", "bid_ask_imbalance",
    # v6.0 新增因子
    "vpin", "kyle_lambda", "amihud_illiq",
    "mtf_trend_align", "mtf_mom_confirm",
    "signal_persistence", "regime_indicator",
    "close_to_high", "close_to_low",
    "volume_price_trend", "mfi_norm",
]

ROUND_TRIP_FEE = 0.0007  # 0.035% x 2
SLIPPAGE = 0.0001  # 0.01% 滑點


def _rank_ic(factor_values, future_returns):
    """計算 Rank IC（Spearman 相關）"""
    fv = np.nan_to_num(factor_values, nan=0.0, posinf=0.0, neginf=0.0)
    fr = np.nan_to_num(future_returns, nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(fv) < 1e-10 or np.std(fr) < 1e-10:
        return 0.0
    rank_f = np.argsort(np.argsort(fv)).astype(float)
    rank_r = np.argsort(np.argsort(fr)).astype(float)
    mf, mr = rank_f.mean(), rank_r.mean()
    sf, sr = rank_f.std(), rank_r.std()
    if sf < 1e-10 or sr < 1e-10:
        return 0.0
    return float(np.mean((rank_f - mf) * (rank_r - mr)) / (sf * sr))


def _hit_rate(factor_values, future_returns):
    """v6.0 新增：計算因子的方向正確率"""
    fv = np.nan_to_num(factor_values, nan=0.0, posinf=0.0, neginf=0.0)
    fr = np.nan_to_num(future_returns, nan=0.0, posinf=0.0, neginf=0.0)
    mask = (np.abs(fv) > 1e-10) & (np.abs(fr) > 1e-10)
    if mask.sum() < 50:
        return 0.5
    correct = np.sign(fv[mask]) == np.sign(fr[mask])
    return float(correct.mean())


def _monthly_ic_stability(factor_values, future_returns, timestamps, min_months=3):
    """計算因子的月度 IC 穩定性"""
    if timestamps is None or len(timestamps) < 100:
        return 1.0
    ts = np.array(timestamps)
    months = (ts // (30 * 24 * 3600 * 1000)).astype(int)
    unique_months = np.unique(months)
    if len(unique_months) < min_months:
        return 1.0
    monthly_ics = []
    for m in unique_months:
        mask = months == m
        if mask.sum() < 50:
            continue
        ic = abs(_rank_ic(factor_values[mask], future_returns[mask]))
        monthly_ics.append(ic)
    if len(monthly_ics) < min_months:
        return 1.0
    hit_rate = sum(1 for ic in monthly_ics if ic > 0.01) / len(monthly_ics)
    return hit_rate


class GeneticEngine:
    def __init__(self, pop_size=600, n_generations=120, train_ratio=0.5,
                 mutation_rate=0.12, crossover_rate=0.65,
                 threshold_range=(0.05, 1.5),
                 fee_rate=ROUND_TRIP_FEE,
                 n_interactions=8,
                 sl_atr=1.5, tp_atr=2.0, max_hold=18,
                 top_k_factors=20,
                 signal_confirm_bars=2,
                 confidence_multiplier=1.15):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.train_ratio = train_ratio
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.threshold_range = threshold_range
        self.fee_rate = fee_rate
        self.n_interactions = n_interactions
        self.sl_atr = sl_atr
        self.tp_atr = tp_atr
        self.max_hold = max_hold
        self.top_k_factors = top_k_factors
        self.signal_confirm_bars = signal_confirm_bars
        self.confidence_multiplier = confidence_multiplier
        self.n_factors = len(FACTOR_COLS)
        self.factor_cols_used = list(FACTOR_COLS)
        self.interaction_pairs = []
        self.selected_factor_indices = None

    # ═══════════════════════════════════════════════════
    #  v6.0 三重篩選：IC + 穩定性 + Hit Rate
    # ═══════════════════════════════════════════════════
    def _ic_screen(self, X, ret, factor_cols, timestamps=None, top_k=None):
        """v6.0 三重篩選：IC + 月度穩定性 + Hit Rate"""
        if top_k is None:
            top_k = self.top_k_factors
        n_cols = X.shape[1]
        if n_cols <= top_k:
            return X, factor_cols, list(range(n_cols))

        scores = []
        for j in range(n_cols):
            ic = abs(_rank_ic(X[:, j], ret))
            stability = _monthly_ic_stability(X[:, j], ret, timestamps)
            hr = _hit_rate(X[:, j], ret)

            # v6.0 綜合分數 = IC × stability × hit_rate_bonus
            # Hit Rate > 52% 的因子得到額外加分
            hr_bonus = 1.0 + max(0, (hr - 0.50)) * 5.0  # 52%→1.1, 55%→1.25
            score = ic * (0.4 + 0.6 * stability) * hr_bonus
            scores.append((score, ic, hr, stability))

        score_vals = np.array([s[0] for s in scores])
        top_indices = np.argsort(score_vals)[-top_k:]
        top_indices = sorted(top_indices)

        X_screened = X[:, top_indices]
        cols_screened = [factor_cols[i] for i in top_indices]

        print(f"  三重篩選: {n_cols} → {len(top_indices)} 因子 (IC+穩定性+HitRate)")
        for idx in top_indices:
            s, ic, hr, stab = scores[idx]
            if s > 0.003:
                print(f"    {factor_cols[idx]:<25} IC={ic:.4f} HR={hr:.1%} stab={stab:.2f} score={s:.4f}")

        return X_screened, cols_screened, top_indices

    # ═══════════════════════════════════════════════════
    #  交互項：基於 IC + Hit Rate 選擇
    # ═══════════════════════════════════════════════════
    def _add_interactions_ic(self, X, ret, factor_cols):
        """基於 IC 和 Hit Rate 選擇 top 因子做交互項"""
        n_cols = X.shape[1]
        if n_cols < 2:
            return X, []

        # 計算每個因子的綜合分數
        factor_scores = []
        for j in range(n_cols):
            ic = abs(_rank_ic(X[:, j], ret))
            hr = _hit_rate(X[:, j], ret)
            factor_scores.append(ic * (1 + max(0, hr - 0.5) * 3))
        factor_scores = np.array(factor_scores)

        n_top = min(6, n_cols)
        top_indices = np.argsort(factor_scores)[-n_top:]

        interaction_cols = []
        pairs = []
        count = 0
        for i in range(len(top_indices)):
            for j in range(i + 1, len(top_indices)):
                if count >= self.n_interactions:
                    break
                idx_i, idx_j = top_indices[i], top_indices[j]
                interaction = X[:, idx_i] * X[:, idx_j]
                inter_ic = abs(_rank_ic(interaction, ret))
                inter_hr = _hit_rate(interaction, ret)
                # v6.0：交互項也要求 Hit Rate > 50%
                if inter_ic > 0.01 and inter_hr > 0.50:
                    interaction_cols.append(interaction)
                    name_i = factor_cols[idx_i] if idx_i < len(factor_cols) else f"f{idx_i}"
                    name_j = factor_cols[idx_j] if idx_j < len(factor_cols) else f"f{idx_j}"
                    pairs.append(f"{name_i}\u00d7{name_j}")
                    count += 1
            if count >= self.n_interactions:
                break

        if interaction_cols:
            X_ext = np.column_stack([X] + interaction_cols)
            return X_ext, pairs
        return X, []

    # ═══════════════════════════════════════════════════
    #  初始化種群
    # ═══════════════════════════════════════════════════
    def _init_population(self):
        population = []
        for _ in range(self.pop_size):
            weights = np.random.randn(self.n_factors)
            mask = np.random.random(self.n_factors) < 0.5
            weights *= mask
            norm = np.linalg.norm(weights)
            if norm > 1e-10:
                weights = weights / norm
            long_thr = random.uniform(*self.threshold_range)
            short_thr = random.uniform(*self.threshold_range)
            population.append({
                "weights": weights,
                "long_threshold": long_thr,
                "short_threshold": short_thr,
            })
        return population

    def _compute_signal(self, X, weights):
        n = min(X.shape[1], len(weights))
        return X[:, :n] @ weights[:n]

    # ═══════════════════════════════════════════════════
    #  v6.0 交易模擬（信號確認 + 置信度過濾 + 滑點）
    # ═══════════════════════════════════════════════════
    def _evaluate_trades(self, individual, X, close, high, low, atr):
        """
        v6.0 模擬實際交易流程：
        1. 信號超過閾值 × confidence_multiplier → 候選
        2. 連續 signal_confirm_bars 根同向 → 確認開倉
        3. 持倉期間檢查止損/止盈/超時/信號反轉
        4. 扣除手續費 + 滑點
        """
        score = self._compute_signal(X, individual["weights"])
        long_thr = individual["long_threshold"]
        short_thr = individual["short_threshold"]
        n = len(score)

        # v6.0：置信度閾值（信號必須超過閾值的 confidence_multiplier 倍）
        conf_long_thr = long_thr * self.confidence_multiplier
        conf_short_thr = short_thr * self.confidence_multiplier

        trades = []
        position = 0
        entry_price = 0.0
        entry_idx = 0
        entry_atr = 0.0
        cooldown = 0

        # v6.0：信號確認計數器
        long_confirm_count = 0
        short_confirm_count = 0

        for i in range(1, n):
            if cooldown > 0:
                cooldown -= 1
                long_confirm_count = 0
                short_confirm_count = 0
                continue

            if position == 0:
                # v6.0：信號確認機制
                if score[i] > conf_long_thr:
                    long_confirm_count += 1
                    short_confirm_count = 0
                elif score[i] < -conf_short_thr:
                    short_confirm_count += 1
                    long_confirm_count = 0
                else:
                    long_confirm_count = 0
                    short_confirm_count = 0

                # 連續確認後才開倉
                if long_confirm_count >= self.signal_confirm_bars:
                    position = 1
                    entry_price = close[i]
                    entry_idx = i
                    entry_atr = atr[i] if atr[i] > 0 else close[i] * 0.005
                    long_confirm_count = 0
                elif short_confirm_count >= self.signal_confirm_bars:
                    position = -1
                    entry_price = close[i]
                    entry_idx = i
                    entry_atr = atr[i] if atr[i] > 0 else close[i] * 0.005
                    short_confirm_count = 0
            else:
                bars_held = i - entry_idx
                if position == 1:
                    pnl_pct = (close[i] - entry_price) / entry_price
                    worst = (low[i] - entry_price) / entry_price
                    best = (high[i] - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - close[i]) / entry_price
                    worst = (entry_price - high[i]) / entry_price
                    best = (entry_price - low[i]) / entry_price

                sl_pct = entry_atr * self.sl_atr / entry_price
                tp_pct = entry_atr * self.tp_atr / entry_price

                close_trade = False
                exit_pnl = pnl_pct

                # 止損
                if worst <= -sl_pct:
                    exit_pnl = -sl_pct
                    close_trade = True
                # 止盈
                elif best >= tp_pct:
                    exit_pnl = tp_pct
                    close_trade = True
                # 超時
                elif bars_held >= self.max_hold:
                    exit_pnl = pnl_pct
                    close_trade = True
                # 信號反轉（使用基礎閾值，不用確認）
                elif position == 1 and score[i] < -short_thr:
                    exit_pnl = pnl_pct
                    close_trade = True
                elif position == -1 and score[i] > long_thr:
                    exit_pnl = pnl_pct
                    close_trade = True

                if close_trade:
                    # v6.0：手續費 + 滑點
                    net_pnl = exit_pnl - self.fee_rate - SLIPPAGE
                    trades.append((position, entry_idx, i, net_pnl))
                    position = 0
                    cooldown = 3  # 冷卻期

        if len(trades) < 12:
            return {
                "win_rate": 0.0, "n_trades": 0, "profit_factor": 0.0,
                "sharpe": 0.0, "net_pnl": 0.0,
                "long_pct": 0.0, "short_pct": 0.0,
                "max_drawdown": 0.0, "calmar": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0,
                "avg_bars_held": 0.0,
                "max_consec_loss": 0, "win_loss_ratio": 0.0,
            }

        pnls = np.array([t[3] for t in trades])
        directions = np.array([t[0] for t in trades])
        bars_held_arr = np.array([t[2] - t[1] for t in trades])

        wins = (pnls > 0).sum()
        win_rate = float(wins / len(trades))

        gross_profit = pnls[pnls > 0].sum() if (pnls > 0).any() else 0.0
        gross_loss = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 0.0
        profit_factor = float(gross_profit / (gross_loss + 1e-10))

        net_pnl = float(pnls.sum())

        if pnls.std() > 1e-10:
            sharpe = float(pnls.mean() / pnls.std() * np.sqrt(len(trades)))
        else:
            sharpe = 0.0

        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = float(drawdown.max()) if len(drawdown) > 0 else 0.0

        if max_drawdown > 1e-10:
            calmar = float(net_pnl / max_drawdown)
        else:
            calmar = float(net_pnl * 100) if net_pnl > 0 else 0.0

        long_count = int((directions == 1).sum())
        short_count = int((directions == -1).sum())
        total = long_count + short_count
        long_pct = long_count / total if total > 0 else 0.5

        avg_win = float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0.0
        avg_loss = float(abs(pnls[pnls < 0].mean())) if (pnls < 0).any() else 0.0

        # 最大連續虧損
        max_consec_loss = 0
        current_consec = 0
        for p in pnls:
            if p < 0:
                current_consec += 1
                max_consec_loss = max(max_consec_loss, current_consec)
            else:
                current_consec = 0

        # 盈虧比
        win_loss_ratio = avg_win / avg_loss if avg_loss > 1e-10 else 0.0

        return {
            "win_rate": win_rate,
            "n_trades": len(trades),
            "profit_factor": profit_factor,
            "sharpe": sharpe,
            "net_pnl": net_pnl,
            "long_pct": long_pct,
            "short_pct": 1.0 - long_pct,
            "max_drawdown": max_drawdown,
            "calmar": calmar,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_bars_held": float(bars_held_arr.mean()),
            "max_consec_loss": max_consec_loss,
            "win_loss_ratio": win_loss_ratio,
        }

    # ═══════════════════════════════════════════════════
    #  v6.0 勝率極致優化適應度函數
    # ═══════════════════════════════════════════════════
    def _fitness(self, metrics):
        """
        v6.0 勝率極致優化適應度：
        - WR < 48% → 直接大幅負分（比 v5.5 的 42% 更嚴格）
        - WR 權重佔 50%，PF 佔 20%，其他佔 30%
        - 勝率和盈虧比的乘積作為核心指標
        """
        wr = metrics["win_rate"]
        nt = metrics["n_trades"]
        pf = metrics["profit_factor"]
        net_pnl = metrics["net_pnl"]
        long_pct = metrics["long_pct"]
        max_dd = metrics["max_drawdown"]
        calmar = metrics["calmar"]
        max_consec_loss = metrics.get("max_consec_loss", 0)
        win_loss_ratio = metrics.get("win_loss_ratio", 0.0)

        if nt < 12:
            return -999.0

        # ═══ 硬性門檻（v6.0 更嚴格）═══
        if pf < 1.0:
            return -200.0 + pf * 100

        # v6.0：WR < 48% 直接淘汰
        if wr < 0.48:
            return -150.0 + wr * 200

        # ═══ 核心得分：勝率主導（50%）═══
        # WR=48%→0, WR=50%→8, WR=55%→28, WR=60%→48
        wr_score = (wr - 0.48) * 400
        wr_score = np.clip(wr_score, -40, 80)

        # v6.0：勝率 × 盈虧比的聯合指標
        # 這是最重要的指標：高勝率 + 高盈虧比 = 穩定盈利
        wr_wl_score = 0.0
        if wr > 0.50 and win_loss_ratio > 1.0:
            wr_wl_score = (wr - 0.50) * (win_loss_ratio - 1.0) * 500
            wr_wl_score = min(wr_wl_score, 40)

        # ═══ PF 得分（20%）═══
        pf_score = (pf - 1.0) * 25
        pf_score = min(pf_score, 40)

        # ═══ 風險調整（15%）═══
        calmar_score = min(max(calmar, -5), 10) * 2
        sharpe_score = min(max(metrics["sharpe"] / 2, -5), 8)

        # ═══ 盈虧比獎勵（10%）═══
        wl_bonus = 0.0
        if win_loss_ratio > 1.5:
            wl_bonus = min((win_loss_ratio - 1.0) * 8, 20)
        elif win_loss_ratio > 1.0:
            wl_bonus = (win_loss_ratio - 1.0) * 8
        elif win_loss_ratio > 0:
            wl_bonus = (win_loss_ratio - 1.0) * 15  # 更重的懲罰

        # ═══ 懲罰項（5%）═══
        # 多空平衡
        balance_ratio = min(long_pct, 1 - long_pct)
        balance_penalty = max(0, (0.30 - balance_ratio)) * 60

        # 最大回撤
        dd_penalty = max(0, max_dd - 0.012) * 400

        # 連續虧損懲罰（v6.0 更嚴格）
        consec_penalty = 0.0
        if max_consec_loss > 6:
            consec_penalty = (max_consec_loss - 6) * 8
        elif max_consec_loss > 4:
            consec_penalty = (max_consec_loss - 4) * 3

        # 交易頻率
        freq_bonus = 0.0
        if nt < 20:
            freq_bonus = -20
        elif nt < 35:
            freq_bonus = -8
        elif nt < 200:
            freq_bonus = min((nt - 35) / 40, 5)
        else:
            freq_bonus = 5 - max(0, (nt - 200)) * 0.05

        # 持倉時間
        avg_bars = metrics.get("avg_bars_held", 0)
        if 3 <= avg_bars <= 12:
            hold_bonus = 3
        elif avg_bars < 2:
            hold_bonus = -10  # 太短 = 噪音
        else:
            hold_bonus = 0

        fitness = (wr_score + wr_wl_score + pf_score
                   + calmar_score + sharpe_score
                   + wl_bonus + freq_bonus + hold_bonus
                   - balance_penalty - dd_penalty - consec_penalty)
        return fitness

    # ═══════════════════════════════════════════════════
    #  進化操作
    # ═══════════════════════════════════════════════════
    def _crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in parent1.items()}
        n = len(parent1["weights"])
        mask = np.random.random(n) < 0.5
        child_weights = np.where(mask, parent1["weights"], parent2["weights"])
        return {
            "weights": child_weights,
            "long_threshold": random.choice([parent1["long_threshold"], parent2["long_threshold"]]),
            "short_threshold": random.choice([parent1["short_threshold"], parent2["short_threshold"]]),
        }

    def _mutate(self, individual, gen_ratio=0.0):
        """自適應變異"""
        weights = individual["weights"].copy()
        long_thr = individual["long_threshold"]
        short_thr = individual["short_threshold"]

        base_sigma = 0.4 * (1.0 - gen_ratio * 0.6)
        mut_rate = self.mutation_rate * (1.0 + (1.0 - gen_ratio) * 0.5)

        for i in range(len(weights)):
            if random.random() < mut_rate:
                weights[i] += np.random.randn() * base_sigma

        if random.random() < mut_rate:
            long_thr += np.random.randn() * 0.08
            long_thr = max(self.threshold_range[0], min(self.threshold_range[1], long_thr))
        if random.random() < mut_rate:
            short_thr += np.random.randn() * 0.08
            short_thr = max(self.threshold_range[0], min(self.threshold_range[1], short_thr))

        # 隨機歸零（稀疏化）
        if random.random() < 0.15:
            idx = random.randint(0, len(weights) - 1)
            weights[idx] = 0.0

        norm = np.linalg.norm(weights)
        if norm > 1e-10:
            weights = weights / norm

        return {
            "weights": weights,
            "long_threshold": long_thr,
            "short_threshold": short_thr,
        }

    # ═══════════════════════════════════════════════════
    #  主搜索流程
    # ═══════════════════════════════════════════════════
    def search(self, df):
        """v6.0 遺傳搜索主流程"""
        factor_cols = list(self.factor_cols_used) if hasattr(self, 'factor_cols_used') else list(FACTOR_COLS)

        available_cols = [c for c in factor_cols if c in df.columns]
        if len(available_cols) < 5:
            print(f"  可用因子不足: {available_cols}")
            return None

        for col in factor_cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(col))

        X_raw = df.select(factor_cols).to_numpy().astype(np.float64)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        X_mean = X_raw.mean(axis=0)
        X_std = X_raw.std(axis=0)

        safe_std = np.where(X_std < 1e-10, 1.0, X_std)
        X = (X_raw - X_mean) / safe_std

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64) if "high" in df.columns else close.copy()
        low = df["low"].to_numpy().astype(np.float64) if "low" in df.columns else close.copy()

        timestamps = None
        if "timestamp" in df.columns:
            timestamps = df["timestamp"].to_numpy()

        # ATR
        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        atr = np.zeros(len(close))
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14

        # 未來收益
        ret = np.zeros(len(close))
        ret[:-1] = (close[1:] - close[:-1]) / close[:-1]

        # ═══ v6.0 三重篩選 ═══
        X, factor_cols_screened, selected_indices = self._ic_screen(
            X, ret, factor_cols, timestamps=timestamps, top_k=self.top_k_factors
        )

        # ═══ 交互項 ═══
        X, interaction_names = self._add_interactions_ic(X, ret, factor_cols_screened)
        all_cols = factor_cols_screened + interaction_names
        self.interaction_pairs = interaction_names
        self.n_factors = X.shape[1]
        self.factor_cols_used = all_cols
        self.selected_factor_indices = selected_indices

        X_mean_screened = X_mean[selected_indices]
        X_std_screened = X_std[selected_indices]

        # ═══ v6.0 Walk-forward: 6 段（更嚴格驗證）═══
        n = len(X)
        seg_size = n // 6
        segments = []
        for s in range(6):
            start = s * seg_size
            end = (s + 1) * seg_size if s < 5 else n
            segments.append((start, end))

        # 訓練：前 3 段，驗證1：第 4 段，驗證2：第 5 段，驗證3：第 6 段
        train_end = segments[2][1]
        val1_start, val1_end = segments[3]
        val2_start, val2_end = segments[4]
        val3_start, val3_end = segments[5]

        X_train = X[:train_end]
        close_train = close[:train_end]
        high_train = high[:train_end]
        low_train = low[:train_end]
        atr_train = atr[:train_end]

        X_val1 = X[val1_start:val1_end]
        close_val1 = close[val1_start:val1_end]
        high_val1 = high[val1_start:val1_end]
        low_val1 = low[val1_start:val1_end]
        atr_val1 = atr[val1_start:val1_end]

        X_val2 = X[val2_start:val2_end]
        close_val2 = close[val2_start:val2_end]
        high_val2 = high[val2_start:val2_end]
        low_val2 = low[val2_start:val2_end]
        atr_val2 = atr[val2_start:val2_end]

        X_val3 = X[val3_start:val3_end]
        close_val3 = close[val3_start:val3_end]
        high_val3 = high[val3_start:val3_end]
        low_val3 = low[val3_start:val3_end]
        atr_val3 = atr[val3_start:val3_end]

        print(f"  因子: {len(factor_cols_screened)} 篩選 + {len(interaction_names)} 交互 = {self.n_factors} 總計")
        print(f"  數據分割: 訓練={train_end} 驗證1={val1_end-val1_start} 驗證2={val2_end-val2_start} 驗證3={val3_end-val3_start}")
        print(f"  交易模擬: SL={self.sl_atr}xATR TP={self.tp_atr}xATR 最大持倉={self.max_hold}根")
        print(f"  v6.0: 信號確認={self.signal_confirm_bars}根 置信度={self.confidence_multiplier:.2f}x "
              f"滑點={SLIPPAGE*100:.3f}% WR門檻=48%")

        population = self._init_population()
        best_combined_fitness = -999
        best_individual = None
        no_improve_count = 0

        for gen in range(self.n_generations):
            gen_ratio = gen / max(1, self.n_generations - 1)

            train_results = []
            for ind in population:
                metrics = self._evaluate_trades(ind, X_train, close_train,
                                                high_train, low_train, atr_train)
                fitness = self._fitness(metrics)
                train_results.append((ind, metrics, fitness))
            train_results.sort(key=lambda x: x[2], reverse=True)

            # v6.0 Walk-forward 驗證（3 個驗證集）
            for ind, _, train_fit in train_results[:20]:
                val1_metrics = self._evaluate_trades(ind, X_val1, close_val1,
                                                     high_val1, low_val1, atr_val1)
                val2_metrics = self._evaluate_trades(ind, X_val2, close_val2,
                                                     high_val2, low_val2, atr_val2)
                val3_metrics = self._evaluate_trades(ind, X_val3, close_val3,
                                                     high_val3, low_val3, atr_val3)
                val1_fit = self._fitness(val1_metrics)
                val2_fit = self._fitness(val2_metrics)
                val3_fit = self._fitness(val3_metrics)

                # v6.0 綜合分數 = 訓練 15% + 驗證1 25% + 驗證2 30% + 驗證3 30%
                # 越近期的驗證集權重越高
                combined = (train_fit * 0.15 + val1_fit * 0.25
                            + val2_fit * 0.30 + val3_fit * 0.30)

                # v6.0：所有驗證集 PF > 1.0
                all_pf_ok = (val1_metrics["profit_factor"] > 1.0 and
                             val2_metrics["profit_factor"] > 1.0 and
                             val3_metrics["profit_factor"] > 1.0)
                if not all_pf_ok:
                    combined -= 50

                # v6.0：所有驗證集 WR > 48%
                all_wr_ok = (val1_metrics["win_rate"] > 0.48 and
                             val2_metrics["win_rate"] > 0.48 and
                             val3_metrics["win_rate"] > 0.48)
                if not all_wr_ok:
                    combined -= 40

                # v6.0 新增：驗證集勝率一致性（標準差不能太大）
                val_wrs = [val1_metrics["win_rate"], val2_metrics["win_rate"],
                           val3_metrics["win_rate"]]
                wr_std = np.std(val_wrs)
                if wr_std > 0.10:
                    combined -= wr_std * 100  # 勝率波動太大 = 不穩定

                if combined > best_combined_fitness:
                    best_combined_fitness = combined
                    no_improve_count = 0
                    best_individual = {
                        "weights": ind["weights"].copy(),
                        "long_threshold": ind["long_threshold"],
                        "short_threshold": ind["short_threshold"],
                        "threshold": (ind["long_threshold"] + ind["short_threshold"]) / 2,
                        "factor_mean": X_mean_screened.copy(),
                        "factor_std": X_std_screened.copy(),
                        "screened_factor_cols": list(factor_cols_screened),
                        "interaction_pairs": interaction_names,
                        "selected_factor_indices": selected_indices,
                        "signal_confirm_bars": self.signal_confirm_bars,
                        "confidence_multiplier": self.confidence_multiplier,
                        "train_metrics": self._evaluate_trades(
                            ind, X_train, close_train, high_train, low_train, atr_train),
                        "val1_metrics": val1_metrics,
                        "val2_metrics": val2_metrics,
                        "val3_metrics": val3_metrics,
                    }

            no_improve_count += 1
            if no_improve_count > 30:
                print(f"  第 {gen+1} 代早停（連續 30 代無改進）")
                break

            # ═══ 進化 ═══
            new_population = []
            elite_count = max(5, self.pop_size // 15)
            for ind, _, _ in train_results[:elite_count]:
                new_population.append({
                    "weights": ind["weights"].copy(),
                    "long_threshold": ind["long_threshold"],
                    "short_threshold": ind["short_threshold"],
                })

            while len(new_population) < self.pop_size:
                candidates = random.sample(train_results, min(7, len(train_results)))
                parent1 = max(candidates, key=lambda x: x[2])[0]
                candidates = random.sample(train_results, min(7, len(train_results)))
                parent2 = max(candidates, key=lambda x: x[2])[0]
                child = self._crossover(parent1, parent2)
                child = self._mutate(child, gen_ratio)
                new_population.append(child)
            population = new_population

            if (gen + 1) % 10 == 0:
                best_train = train_results[0]
                print(
                    f"  第 {gen+1:>3d} 代 | "
                    f"訓練 WR={best_train[1]['win_rate']:.3f} "
                    f"PF={best_train[1]['profit_factor']:.2f} "
                    f"交易={best_train[1]['n_trades']:>4d} "
                    f"盈虧比={best_train[1].get('win_loss_ratio', 0):.2f} "
                    f"連虧={best_train[1].get('max_consec_loss', 0)} | "
                    f"最佳綜合={best_combined_fitness:.2f}"
                )

        if best_individual is not None:
            print(f"\n  最優因子權重（{len(factor_cols_screened)} 篩選 + {len(interaction_names)} 交互）:")
            all_w = best_individual["weights"]
            for i, col in enumerate(factor_cols_screened):
                if i < len(all_w) and abs(all_w[i]) > 0.03:
                    print(f"    {col:<25} {all_w[i]:+.3f}")
            for i, name in enumerate(interaction_names):
                idx = len(factor_cols_screened) + i
                if idx < len(all_w) and abs(all_w[idx]) > 0.03:
                    print(f"    {name:<25} {all_w[idx]:+.3f}")

            tm = best_individual["train_metrics"]
            v1 = best_individual["val1_metrics"]
            v2 = best_individual["val2_metrics"]
            v3 = best_individual["val3_metrics"]
            print(f"\n  訓練集: WR={tm['win_rate']:.3f} PF={tm['profit_factor']:.2f} "
                  f"交易={tm['n_trades']} 盈虧比={tm.get('win_loss_ratio', 0):.2f} "
                  f"連虧={tm.get('max_consec_loss', 0)}")
            print(f"  驗證1:  WR={v1['win_rate']:.3f} PF={v1['profit_factor']:.2f} "
                  f"交易={v1['n_trades']} DD={v1['max_drawdown']:.4f} "
                  f"盈虧比={v1.get('win_loss_ratio', 0):.2f}")
            print(f"  驗證2:  WR={v2['win_rate']:.3f} PF={v2['profit_factor']:.2f} "
                  f"交易={v2['n_trades']} DD={v2['max_drawdown']:.4f} "
                  f"盈虧比={v2.get('win_loss_ratio', 0):.2f}")
            print(f"  驗證3:  WR={v3['win_rate']:.3f} PF={v3['profit_factor']:.2f} "
                  f"交易={v3['n_trades']} DD={v3['max_drawdown']:.4f} "
                  f"盈虧比={v3.get('win_loss_ratio', 0):.2f}")
            print(f"  多頭閾值: {best_individual['long_threshold']:.4f}")
            print(f"  空頭閾值: {best_individual['short_threshold']:.4f}")
            print(f"  信號確認: {self.signal_confirm_bars}根 置信度: {self.confidence_multiplier:.2f}x")

            avg_pf = (v1['profit_factor'] + v2['profit_factor'] + v3['profit_factor']) / 3
            avg_wr = (v1['win_rate'] + v2['win_rate'] + v3['win_rate']) / 3
            if avg_pf < 1.0:
                print(f"\n  !! 警告: 驗證集平均 PF={avg_pf:.2f} < 1.0（扣費後虧損）")
            elif avg_wr < 0.48:
                print(f"\n  ! 注意: 驗證集平均 WR={avg_wr:.1%}，勝率偏低")
            elif avg_pf < 1.2:
                print(f"\n  ! 注意: 驗證集平均 PF={avg_pf:.2f}，邊際盈利")
            else:
                print(f"\n  OK 驗證集平均 PF={avg_pf:.2f} WR={avg_wr:.1%}")

        return best_individual
