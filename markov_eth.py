from __future__ import annotations
import os
import time
import math
import random
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import ccxt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress HMM convergence warnings (we'll check manually)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='hmmlearn')

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
    min_required_bars: int = 1500  # Minimum bars needed after alignment

    # Markov
    k_states: int = 3
    w_window: int = 2000
    h_horizon: int = 4
    laplace_alpha: float = 1.0
    identity_lambda: float = 0.05
    edge_tau: float = 0.02
    rv_gate_percentile: float = 0.80

    # HMM
    hmm_components: int = 3
    hmm_n_iter: int = 300  # Increased from 200
    hmm_tol: float = 1e-4  # Convergence tolerance
    test_split_ratio: float = 0.30

    # Costs & limits
    fee_per_side: float = 0.0004
    request_timeout: int = 20

    # Seeds/limits
    seed: int = 42
    max_retries: int = 6

    # Safety thresholds
    min_coverage_ratio: float = 0.95  # Changed from 0.8
    numeric_floor: float = 1e-14  # For stability


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
    """Initialize CCXT exchange with rate limiting."""
    try:
        ex = getattr(ccxt, exchange_id)({
            "enableRateLimit": True,
            "rateLimit": 500,
            "timeout": cfg.request_timeout * 1000,
        })
        logger.info(f"Initialized {exchange_id} exchange")
        return ex
    except AttributeError as e:
        raise SystemExit(f"Unsupported exchange: {exchange_id}") from e


EXCHANGE = init_exchange(cfg.exchange_id)

# ==============================
# 2) DATA FETCHING
# ==============================

def fetch_ohlcv(symbol: str, timeframe: str, start_dt: datetime, limit: int = 1000) -> pd.DataFrame:
    """Fetches historical OHLCV using CCXT public endpoints with retry logic."""
    since_ms = int(start_dt.timestamp() * 1000)
    out = []
    retries = 0
    
    logger.info(f"Fetching OHLCV for {symbol} ({timeframe}) from {start_dt.date()}")
    
    while True:
        try:
            batch = EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
            retries = 0  # Reset on success
        except ccxt.NetworkError as e:
            logger.warning(f"Network error fetching OHLCV: {e}. Retrying...")
            time.sleep(5)
            retries += 1
            if retries > cfg.max_retries:
                raise SystemExit("Max retries exceeded for OHLCV fetch") from e
            continue
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            time.sleep(5)
            retries += 1
            if retries > cfg.max_retries:
                raise SystemExit("Max retries exceeded for OHLCV fetch") from e
            continue
            
        if not batch:
            break
        out.extend(batch)
        since_ms = batch[-1][0] + 1
        if len(batch) < limit:
            break
    
    if not out:
        raise SystemExit("No OHLCV data retrieved")
    
    df = pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "volume"])\
            .assign(ts=lambda d: pd.to_datetime(d["ts"], unit="ms", utc=True))\
            .drop_duplicates(subset=['ts']).sort_values("ts").set_index("ts")
    
    logger.info(f"OHLCV fetch complete: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
    return df


def fetch_open_interest(symbol: str, interval: str, start_dt: datetime) -> pd.DataFrame:
    """Paginate Binance futures OI with improved error handling."""
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int(start_dt.timestamp() * 1000)
    all_records: list[dict] = []
    seen = set()
    retries = 0
    
    logger.info(f"Fetching Open Interest for {symbol} ({interval}) from {start_dt.date()}")
    
    while end_ms > start_ms and retries <= cfg.max_retries:
        params = {"symbol": symbol, "period": interval, "limit": 500, "endTime": end_ms}
        try:
            r = requests.get(url, params=params, timeout=cfg.request_timeout)
            r.raise_for_status()
            batch = r.json()
            
            if not batch:
                break
            
            # Dedupe and add
            for it in batch:
                ts = it["timestamp"]
                if ts not in seen:
                    seen.add(ts)
                    all_records.append(it)
            
            # Step to older data
            end_ms = batch[-1]["timestamp"] - 1
            time.sleep(max(0.001, EXCHANGE.rateLimit / 1000))
            retries = 0  # Reset on success
            
        except requests.exceptions.HTTPError as e:
            status = getattr(r, 'status_code', None)
            if status == 429:
                delay = min(2 ** retries, 60)
                logger.warning(f"Rate limited (429). Waiting {delay}s...")
                time.sleep(delay)
                retries += 1
            else:
                logger.error(f"HTTP error {status}: {e}")
                break
        except requests.exceptions.RequestException as e:
            delay = min(2 ** retries, 60)
            logger.warning(f"Request error: {e}. Waiting {delay}s...")
            time.sleep(delay)
            retries += 1
    
    if not all_records:
        logger.warning("No OI data retrieved - returning empty DataFrame")
        return pd.DataFrame(columns=["sumOpenInterest"]).astype({"sumOpenInterest": float})
    
    d = pd.DataFrame(all_records)
    d["timestamp"] = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
    d = d[["timestamp", "sumOpenInterest"]]
    d["sumOpenInterest"] = d["sumOpenInterest"].astype(float)
    d = d.drop_duplicates(subset=['timestamp']).sort_values("timestamp").set_index("timestamp")
    
    logger.info(f"OI fetch complete: {len(d)} records from {d.index[0].date()} to {d.index[-1].date()}")
    return d

# ==============================
# 3) FEATURE ENGINEERING
# ==============================

def build_features(df_ohlcv: pd.DataFrame, df_oi: pd.DataFrame) -> pd.DataFrame:
    """Build features with strict alignment and validation."""
    # Strict alignment
    df = df_ohlcv.join(df_oi, how="inner")
    
    if len(df) < cfg.min_required_bars:
        raise SystemExit(
            f"Insufficient aligned bars: {len(df)} < {cfg.min_required_bars}. "
            "Reduce lookback_days or adjust min_required_bars."
        )

    # Returns & volatility
    df["ret"] = np.log(df["close"]).diff()
    df["rv"] = df["ret"].rolling(20).std() * math.sqrt(ANNUAL_FACTOR)

    # OI change (clipped for outliers)
    df["dOI"] = df["sumOpenInterest"].pct_change().clip(-0.2, 0.2)

    # Forward H-step return (no lookahead)
    df["fwd_ret_H"] = df["ret"].shift(-1).rolling(cfg.h_horizon).sum()

    df_clean = df.dropna()
    logger.info(f"Features built: {len(df_clean)} bars after dropna")
    
    return df_clean

# ==============================
# 4) MARKOV CHAIN
# ==============================

def trans_mat_from_states(states: np.ndarray, K: int, alpha: float, lam: float) -> np.ndarray:
    """
    Laplace-smoothed transition matrix with identity mixing.
    Improved numeric stability.
    """
    C = np.full((K, K), alpha, dtype=float)
    
    # Count transitions
    for a, b in zip(states[:-1], states[1:]):
        C[a, b] += 1.0
    
    # Normalize
    row_sums = C.sum(axis=1, keepdims=True)
    T = C / np.maximum(row_sums, cfg.numeric_floor)
    
    # Identity mixing for stability
    T = (1 - lam) * T + lam * np.eye(K)
    
    # Floor and renormalize
    T = np.maximum(T, cfg.numeric_floor)
    T = T / T.sum(axis=1, keepdims=True)
    
    return T


def markov_walk_forward(
    df_feat: pd.DataFrame, 
    idx_train: pd.DatetimeIndex, 
    idx_test: pd.DatetimeIndex
) -> dict:
    """Walk-forward Markov chain prediction with improved edge calculation."""
    K = cfg.k_states
    W = cfg.w_window
    H = cfg.h_horizon

    signals = []
    edges = []
    p_up_list = []
    ts_exec = []

    # RV gate learned on train
    rv_cap = df_feat.loc[idx_train, "rv"].quantile(cfg.rv_gate_percentile)
    logger.info(f"Markov RV gate threshold: {rv_cap:.4f}")

    # Stride test by H for non-overlap
    test_ts_strided = idx_test[:-H: H].copy()
    logger.info(f"Markov evaluation on {len(test_ts_strided)} strided points")

    for t in test_ts_strided:
        # Ensure sufficient history
        pos = df_feat.index.get_indexer([t])[0]
        if pos - W < 0:
            continue
            
        win_slice = slice(pos - W, pos)
        r_win = df_feat["ret"].iloc[win_slice]

        # In-window quantile bins (no leakage)
        qcuts = r_win.quantile([1 / K, 2 / K]).values
        s_win = np.digitize(r_win.values, qcuts, right=False)
        s_win = np.clip(s_win, 0, K - 1)

        # Current state at time t
        r_t = df_feat.at[t, "ret"]
        cur_state = int(np.clip(np.digitize([r_t], qcuts, right=False)[0], 0, K - 1))

        # Transition matrix and H-step forecast
        T = trans_mat_from_states(s_win, K, cfg.laplace_alpha, cfg.identity_lambda)
        
        # Use safe matrix power with error handling
        try:
            Th = np.linalg.matrix_power(T, H)
        except np.linalg.LinAlgError:
            logger.warning(f"Matrix power failed at {t}, using T^1")
            Th = T
        
        e = np.eye(K)[cur_state]
        probs = e @ Th
        p_up, p_down = probs[K - 1], probs[0]
        edge = float(p_up - p_down)

        # RV gate: only trade in low-vol regimes
        current_rv = df_feat.at[t, "rv"]
        if current_rv >= rv_cap:
            sig = 0.0
        else:
            sig = float(np.clip(edge / cfg.edge_tau, -1.0, 1.0))

        signals.append(sig)
        edges.append(edge)
        p_up_list.append(float(p_up))
        ts_exec.append(t)

    # Convert to series
    sig = pd.Series(signals, index=pd.DatetimeIndex(ts_exec), name="sig_mc").astype(float)
    edge_series = pd.Series(edges, index=sig.index, name="edge")
    p_up_series = pd.Series(p_up_list, index=sig.index, name="p_up")

    # Non-overlapping H-step returns
    fwd = df_feat["fwd_ret_H"].reindex(sig.index)
    strat = (sig * fwd).dropna()

    # Fees via turnover on rebalancing
    turnover = sig.diff().abs().fillna(0).sum()
    total_fee = cfg.fee_per_side * turnover
    net = strat - total_fee / max(len(strat), 1)

    # Performance metrics
    sharpe = np.nan
    if len(strat) > 0 and strat.std() > 0:
        sharpe = strat.mean() / strat.std() * math.sqrt(ANNUAL_FACTOR / H)

    hit = (np.sign(strat) == np.sign(fwd.reindex(strat.index))).mean() if len(strat) else np.nan

    return {
        "signals": sig,
        "edge": edge_series,
        "p_up": p_up_series,
        "strat": strat,
        "strat_net": net,
        "turnover": float(turnover),
        "total_fee": float(total_fee),
        "hit_rate": float(hit) if not np.isnan(hit) else np.nan,
        "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
    }

# ==============================
# 5) HMM REGIME MODEL
# ==============================

def hmm_regime(
    df_feat: pd.DataFrame, 
    idx_train: pd.DatetimeIndex, 
    idx_test: pd.DatetimeIndex
) -> dict:
    """HMM regime detection with improved convergence checking."""
    H = cfg.h_horizon
    X_all = df_feat[["ret", "rv", "dOI"]].copy()
    Y_all = df_feat["fwd_ret_H"].copy()

    X_tr, X_te = X_all.loc[idx_train], X_all.loc[idx_test]
    Y_tr, Y_te = Y_all.loc[idx_train], Y_all.loc[idx_test]

    # Robust scaling
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Fit HMM with convergence monitoring
    hmm = GaussianHMM(
        n_components=cfg.hmm_components,
        covariance_type="full",
        n_iter=cfg.hmm_n_iter,
        tol=cfg.hmm_tol,
        random_state=cfg.seed,
        min_covar=1e-6,
        verbose=False,
    )
    
    logger.info(f"Fitting HMM with {cfg.hmm_components} components...")
    hmm.fit(X_tr_s)
    
    # Check convergence
    if hasattr(hmm, 'monitor_') and hasattr(hmm.monitor_, 'converged'):
        if not hmm.monitor_.converged:
            logger.warning("HMM did not converge! Results may be unstable.")
        else:
            logger.info(f"HMM converged in {hmm.monitor_.iter} iterations")

    # Predict regimes
    reg_tr = hmm.predict(X_tr_s)
    reg_te = hmm.predict(X_te_s)

    # Regime-conditional expected returns
    mu_by_reg = pd.Series(Y_tr.values).groupby(reg_tr).mean()
    
    # Log regime statistics
    for reg in range(cfg.hmm_components):
        count = (reg_tr == reg).sum()
        mu = mu_by_reg.get(reg, np.nan)
        logger.info(f"  Regime {reg}: n={count}, E[ret]={mu:.6f}")
    
    mu_hat = pd.Series(reg_te, index=idx_test).map(mu_by_reg).rename("mu_hat")

    # Strided evaluation to avoid overlap
    te_strided = idx_test[:-H: H]
    sig = np.sign(mu_hat.reindex(te_strided)).fillna(0.0)
    fwd = Y_all.reindex(te_strided)
    strat = (sig * fwd).dropna()

    # Performance metrics
    sharpe = np.nan
    if len(strat) > 0 and strat.std() > 0:
        sharpe = strat.mean() / strat.std() * math.sqrt(ANNUAL_FACTOR / H)
    
    hit = (np.sign(strat) == np.sign(fwd.reindex(strat.index))).mean() if len(strat) else np.nan

    return {
        "signals": sig,
        "strat": strat,
        "mu_by_reg": mu_by_reg,
        "regimes_train": reg_tr,
        "regimes_test": reg_te,
        "hit_rate": float(hit) if not np.isnan(hit) else np.nan,
        "sharpe": float(sharpe) if not np.isnan(sharpe) else np.nan,
    }

# ==============================
# 6) MAIN
# ==============================

def main() -> None:
    """Main execution flow with comprehensive error handling."""
    logger.info("=" * 70)
    logger.info(f"Crypto Trading System - {cfg.symbol_ohlcv}")
    logger.info("=" * 70)
    
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=cfg.lookback_days)
    logger.info(f"Target date range: {start_dt.date()} to {end_dt.date()}")

    # Fetch data
    df_ohlcv = fetch_ohlcv(cfg.symbol_ohlcv, cfg.tf, start_dt)
    df_oi = fetch_open_interest(cfg.symbol_oi, cfg.tf, start_dt)

    # Adjust OHLCV to OI availability
    if not df_oi.empty:
        oi_start = df_oi.index.min()
        oi_end = df_oi.index.max()
        
        # Trim OHLCV to OI range
        df_ohlcv = df_ohlcv[(df_ohlcv.index >= oi_start) & (df_ohlcv.index <= oi_end)]
        
        actual_days = (df_ohlcv.index[-1] - df_ohlcv.index[0]).days
        logger.info(f"Aligned OHLCV to OI range: {oi_start.date()} to {oi_end.date()} ({actual_days} days)")
        
        # Safety check for minimum data
        if len(df_ohlcv) < cfg.min_required_bars:
            raise SystemExit(
                f"Insufficient data after alignment: {len(df_ohlcv)} bars < {cfg.min_required_bars}. "
                f"Reduce lookback_days or min_required_bars."
            )
    else:
        raise SystemExit("No Open Interest data available!")
    
    # Coverage diagnostic
    inter = df_ohlcv.index.intersection(df_oi.index)
    coverage = len(inter) / max(1, len(df_ohlcv))
    logger.info(f"Coverage: {coverage:.2%} ({len(inter)}/{len(df_ohlcv)} bars)")
    
    if coverage < cfg.min_coverage_ratio:
        raise SystemExit(
            f"Low OI coverage: {coverage:.1%} < {cfg.min_coverage_ratio:.1%}. "
            "Data quality insufficient."
        )

    # Build features
    df_feat = build_features(df_ohlcv, df_oi)

    # Train/test split AFTER feature dropna
    n = len(df_feat)
    n_test = int(n * cfg.test_split_ratio)
    n_train = n - n_test

    idx_all = df_feat.index
    idx_train = idx_all[:n_train]
    idx_test = idx_all[n_train:]

    # Validate training size
    min_train = cfg.w_window + cfg.h_horizon + 100
    if n_train < min_train:
        raise SystemExit(
            f"Training set too small: {n_train} < {min_train}. "
            f"Reduce w_window or increase lookback_days."
        )

    # Report ranges
    logger.info("\n" + "=" * 70)
    logger.info(f"Train: {idx_train[0].date()} → {idx_train[-1].date()}  (n={n_train})")
    logger.info(f"Test:  {idx_test[0].date()} → {idx_test[-1].date()}   (n={n_test})")
    logger.info("=" * 70)

    # Run models
    logger.info("\nRunning Markov Chain model...")
    mc = markov_walk_forward(df_feat, idx_train, idx_test)
    
    logger.info("\nRunning HMM Regime model...")
    hmm = hmm_regime(df_feat, idx_train, idx_test)

    # Results
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(
        f"[Markov] Hit Rate: {mc['hit_rate']:.4f}  |  Sharpe: {mc['sharpe']:.4f}  |  "
        f"Turnover: {mc['turnover']:.2f}  |  Bars: {len(mc['strat'])}"
    )
    logger.info(
        f"[HMM]    Hit Rate: {hmm['hit_rate']:.4f}  |  Sharpe: {hmm['sharpe']:.4f}  |  "
        f"Bars: {len(hmm['strat'])}"
    )
    logger.info("=" * 70)

    # Visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity curves
        mc["strat_net"].cumsum().plot(ax=axes[0], label="Markov (net)", linewidth=2)
        hmm["strat"].cumsum().plot(ax=axes[0], label="HMM", linewidth=2)
        axes[0].set_title("Cumulative Returns (Test Period)", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("Cumulative Return")
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Signal strength
        mc["edge"].plot(ax=axes[1], label="Markov Edge", alpha=0.7)
        axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1].set_title("Markov Edge Signal", fontsize=14, fontweight='bold')
        axes[1].set_ylabel("Edge (p_up - p_down)")
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        logger.warning("Matplotlib not available - skipping visualization")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nExecution interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
