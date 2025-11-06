from __future__ import annotations
import os
import time
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import ccxt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler

# ==============================
# 0) CONFIG
# ==============================
@dataclass
class Cfg:
    exchange_id: str = "binance"
    symbol_ohlcv: str = "ETH/USDT"
    symbol_oi: str = "ETHUSDT"
    tf: str = "15m"
    lookback_days: int = 120

    # Markov
    k_states: int = 3
    w_window: int = 2000
    h_horizon: int = 4
    laplace_alpha: float = 1.0
    identity_lambda: float = 0.05
    edge_tau: float = 0.02  # sizing divisor (pos = clip(edge/tau, -1, 1))
    rv_gate_percentile: float = 0.80  # trade only if rv < percentile(train)

    # HMM
    hmm_components: int = 3
    test_split_ratio: float = 0.30

    # Costs & limits
    fee_per_side: float = 0.0004  # 4 bps per side
    request_timeout: int = 20

    # Seeds/limits
    seed: int = 42
    max_retries: int = 6


cfg = Cfg()

# Determinism
random.seed(cfg.seed)
np.random.seed(cfg.seed)

# Annualization factor for 15m bars
ANNUAL_FACTOR = int((60 / 15) * 24 * 365)  # 35040

# ==============================
# 1) EXCHANGE INIT
# ==============================

def init_exchange(exchange_id: str) -> ccxt.Exchange:
    try:
        ex = getattr(ccxt, exchange_id)({
            "enableRateLimit": True,
            "rateLimit": 500,
        })
        return ex
    except AttributeError as e:
        raise SystemExit(f"Unsupported exchange: {exchange_id}") from e


EXCHANGE = init_exchange(cfg.exchange_id)

# ==============================
# 2) DATA FETCHING
# ==============================

def fetch_ohlcv(symbol: str, timeframe: str, start_dt: datetime, limit: int = 1000) -> pd.DataFrame:
    """Fetches historical OHLCV using CCXT public endpoints."""
    since_ms = int(start_dt.timestamp() * 1000)
    out = []
    while True:
        try:
            batch = EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            time.sleep(5)
            continue
        if not batch:
            break
        out.extend(batch)
        since_ms = batch[-1][0] + 1
        if len(batch) < limit:
            break
    df = pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "volume"])\
            .assign(ts=lambda d: pd.to_datetime(d["ts"], unit="ms", utc=True))\
            .drop_duplicates().sort_values("ts").set_index("ts")
    return df


def fetch_open_interest(symbol: str, interval: str, start_dt: datetime) -> pd.DataFrame:
    """Paginate Binance futures OI (max 500 per call). Reverse from now to start."""
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int(start_dt.timestamp() * 1000)
    all_records: list[dict] = []
    seen = set()
    retries = 0
    while end_ms > start_ms and retries <= cfg.max_retries:
        params = {"symbol": symbol, "period": interval, "limit": 500, "endTime": end_ms}
        try:
            r = requests.get(url, params=params, timeout=cfg.request_timeout)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            # dedupe
            added = 0
            for it in batch:
                ts = it["timestamp"]
                if ts not in seen:
                    seen.add(ts)
                    all_records.append(it)
                    added += 1
            # step older
            end_ms = batch[-1]["timestamp"] - 1
            # gentle rate limit
            time.sleep(max(0.001, EXCHANGE.rateLimit / 1000))
            retries = 0
        except requests.exceptions.HTTPError as e:
            status = r.status_code if "r" in locals() else None
            if status == 429:
                delay = 2 ** retries
                time.sleep(delay)
                retries += 1
                continue
            else:
                break
        except requests.exceptions.RequestException:
            delay = min(2 ** retries, 60)
            time.sleep(delay)
            retries += 1
            continue
    if not all_records:
        return pd.DataFrame(columns=["sumOpenInterest"]).astype({"sumOpenInterest": float})
    d = pd.DataFrame(all_records)
    d["timestamp"] = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
    d = d[["timestamp", "sumOpenInterest"]]
    d["sumOpenInterest"] = d["sumOpenInterest"].astype(float)
    d = d.drop_duplicates().sort_values("timestamp").set_index("timestamp")
    return d

# ==============================
# 3) FEATURE ENGINEERING
# ==============================

def build_features(df_ohlcv: pd.DataFrame, df_oi: pd.DataFrame) -> pd.DataFrame:
    # strict alignment
    df = df_ohlcv.join(df_oi, how="inner")
    if len(df) < 1000:
        raise SystemExit("Insufficient aligned bars after INNER JOIN.")

    # returns & volatility
    df["ret"] = np.log(df["close"]).diff()
    df["rv"] = df["ret"].rolling(20).std() * math.sqrt(ANNUAL_FACTOR)

    # OI change (clipped)
    df["dOI"] = df["sumOpenInterest"].pct_change().clip(-0.2, 0.2)

    # correct forward H-step return: sum of next H bars
    df["fwd_ret_H"] = df["ret"].shift(-1).rolling(cfg.h_horizon).sum()

    return df.dropna()

# ==============================
# 4) MARKOV CHAIN
# ==============================

def trans_mat_from_states(states: np.ndarray, K: int, alpha: float, lam: float) -> np.ndarray:
    """Laplace-smoothed transition matrix with identity mixing and numeric floor."""
    C = np.full((K, K), alpha, dtype=float)
    # counts
    for a, b in zip(states[:-1], states[1:]):
        C[a, b] += 1.0
    T = C / C.sum(1, keepdims=True)
    T = (1 - lam) * T + lam * np.eye(K)
    T = np.maximum(T, 1e-12)
    T = T / T.sum(1, keepdims=True)
    return T


def markov_walk_forward(df_feat: pd.DataFrame, idx_train: pd.DatetimeIndex, idx_test: pd.DatetimeIndex) -> dict:
    K = cfg.k_states
    W = cfg.w_window
    H = cfg.h_horizon

    signals = []
    edges = []
    p_up_list = []
    ts_exec = []

    # rv gate learned on train
    rv_cap = df_feat.loc[idx_train, "rv"].quantile(cfg.rv_gate_percentile)

    # stride test by H for non-overlap
    test_ts_strided = idx_test[:-H: H].copy()

    for t in test_ts_strided:
        # ensure we have W bars of history ending at t (exclusive)
        pos = df_feat.index.get_indexer_for([t])[0]
        if pos - W < 0:
            continue
        win_slice = slice(pos - W, pos)
        r_win = df_feat["ret"].iloc[win_slice]

        # in-window rolling quantile bins (no leakage)
        qcuts = r_win.quantile([1 / K, 2 / K]).values
        s_win = np.clip(np.digitize(r_win.values, qcuts), 0, K - 1)

        # current state at time t using same window cutoffs
        r_t = df_feat.at[t, "ret"]
        cur_state = int(np.clip(np.digitize([r_t], qcuts)[0], 0, K - 1))

        T = trans_mat_from_states(s_win, K, cfg.laplace_alpha, cfg.identity_lambda)
        Th = np.linalg.matrix_power(T, H)
        e = np.eye(K)[cur_state]
        probs = e @ Th
        p_up, p_down = probs[K - 1], probs[0]
        edge = float(p_up - p_down)

        # rv gate
        if df_feat.at[t, "rv"] >= rv_cap:
            sig = 0.0
        else:
            sig = float(np.clip(edge / cfg.edge_tau, -1.0, 1.0))

        signals.append(sig)
        edges.append(edge)
        p_up_list.append(float(p_up))
        ts_exec.append(t)

    sig = pd.Series(signals, index=pd.DatetimeIndex(ts_exec), name="sig_mc").astype(float)
    edge_series = pd.Series(edges, index=sig.index, name="edge")
    p_up_series = pd.Series(p_up_list, index=sig.index, name="p_up")

    # Non-overlapping H-step returns
    fwd = df_feat["fwd_ret_H"].reindex(sig.index)
    strat = (sig * fwd).dropna()

    # Fees via turnover on rebalancing (per stride)
    turnover = sig.diff().abs().fillna(0).sum()
    net = strat - cfg.fee_per_side * turnover / max(len(strat), 1)

    sharpe = np.nan
    if strat.std() > 0:
        sharpe = strat.mean() / strat.std() * math.sqrt(ANNUAL_FACTOR / H)

    hit = (np.sign(strat) == np.sign(fwd.reindex(strat.index))).mean() if len(strat) else np.nan

    return {
        "signals": sig,
        "edge": edge_series,
        "p_up": p_up_series,
        "strat": strat,
        "strat_net": net,
        "turnover": float(turnover),
        "hit_rate": float(hit) if not math.isnan(hit) else np.nan,
        "sharpe": float(sharpe) if not math.isnan(sharpe) else np.nan,
    }

# ==============================
# 5) HMM REGIME MODEL
# ==============================

def hmm_regime(df_feat: pd.DataFrame, idx_train: pd.DatetimeIndex, idx_test: pd.DatetimeIndex) -> dict:
    H = cfg.h_horizon
    X_all = df_feat[["ret", "rv", "dOI"]].copy()
    Y_all = df_feat["fwd_ret_H"].copy()

    X_tr, X_te = X_all.loc[idx_train], X_all.loc[idx_test]
    Y_tr, Y_te = Y_all.loc[idx_train], Y_all.loc[idx_test]

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    hmm = GaussianHMM(
        n_components=cfg.hmm_components,
        covariance_type="full",
        n_iter=200,
        random_state=cfg.seed,
        min_covar=1e-6,
    )
    hmm.fit(X_tr_s)

    # Regimes
    reg_tr = hmm.predict(X_tr_s)
    reg_te = hmm.predict(X_te_s)

    mu_by_reg = pd.Series(Y_tr.values).groupby(reg_tr).mean()
    mu_hat = pd.Series(reg_te, index=idx_test).map(mu_by_reg).rename("mu_hat")

    # Strided evaluation to avoid overlap
    te_strided = idx_test[:-H: H]
    sig = np.sign(mu_hat.reindex(te_strided)).fillna(0.0)
    fwd = Y_all.reindex(te_strided)
    strat = (sig * fwd).dropna()

    sharpe = np.nan
    if strat.std() > 0:
        sharpe = strat.mean() / strat.std() * math.sqrt(ANNUAL_FACTOR / H)
    hit = (np.sign(strat) == np.sign(fwd.reindex(strat.index))).mean() if len(strat) else np.nan

    return {
        "signals": sig,
        "strat": strat,
        "hit_rate": float(hit) if not math.isnan(hit) else np.nan,
        "sharpe": float(sharpe) if not math.isnan(sharpe) else np.nan,
    }

# ==============================
# 6) MAIN
# ==============================

def main() -> None:
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=cfg.lookback_days)

    df_ohlcv = fetch_ohlcv(cfg.symbol_ohlcv, cfg.tf, start_dt)
    df_oi = fetch_open_interest(cfg.symbol_oi, cfg.tf, start_dt)

    # Coverage diagnostic
    inter = df_ohlcv.index.intersection(df_oi.index)
    coverage = len(inter) / max(1, len(df_ohlcv))
    if coverage < 0.8:
        raise SystemExit(f"Low OI coverage vs OHLCV: {coverage:.1%}. Shorten lookback or improve pagination.")

    df_feat = build_features(df_ohlcv, df_oi)

    # Train/test split AFTER feature dropna so indices align
    n = len(df_feat)
    n_test = int(n * cfg.test_split_ratio)
    n_train = n - n_test

    idx_all = df_feat.index
    idx_train = idx_all[:n_train]
    idx_test = idx_all[n_train:]

    if n_train < cfg.w_window + 10:
        raise SystemExit("Training set too small relative to Markov window. Adjust lookback/W.")

    # Report ranges
    print(f"Train range: {idx_train[0].date()} -> {idx_train[-1].date()}  (n={n_train})")
    print(f"Test  range: {idx_test[0].date()} -> {idx_test[-1].date()}  (n={n_test})")

    # Markov
    mc = markov_walk_forward(df_feat, idx_train, idx_test)
    print("\n[Markov] hit=%.4f sharpe=%.4f turnover=%.2f bars=%d" % (
        mc["hit_rate"], mc["sharpe"], mc["turnover"], len(mc["strat"]))
    )

    # HMM
    hmm = hmm_regime(df_feat, idx_train, idx_test)
    print("[HMM]    hit=%.4f sharpe=%.4f bars=%d" % (
        hmm["hit_rate"], hmm["sharpe"], len(hmm["strat"]))
    )

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        (mc["strat_net"].cumsum().plot(label="Markov"))
        (hmm["strat"].cumsum().plot(label="HMM"))
        plt.legend(); plt.title("Equity Curve (Test, Strided)")
        plt.show()
    except ImportError:
        pass

if __name__ == "__main__":
    main()
