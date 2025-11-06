import ccxt
import pandas as pd
import numpy as np
import requests
import time
import json
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import warnings
import matplotlib.pyplot as plt # Included for diagnostic plotting

# Suppress sklearn convergence warnings for HMM
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 0. CONFIGURATION & PARAMETERS
# ==============================================================================

@dataclass
class Config:
    # Data Acquisition
    exchange_id: str = 'binance'
    symbol_ohlcv: str = 'ETH/USDT'
    symbol_oi: str = 'ETHUSDT'       # Binance Futures symbol format
    tf: str = '15m'                  # Timeframe
    lookback_days: int = 180         # Total data to fetch (increased for robustness)
    oi_coverage_min: float = 0.8     # Minimum required Open Interest coverage ratio (e.g., 80%)

    # Markov Chain Parameters
    k_states: int = 3                # Number of states for return discretization (e.g., Down, Flat, Up)
    w_window: int = 2000             # Rolling window size in bars for MC training
    h_horizon: int = 4               # Prediction horizon (e.g., 4x15m = 1 hour). Used as stride for backtest.
    smoothing_alpha: float = 1.0     # Laplace smoothing alpha
    smoothing_lambda: float = 0.05   # Mixing with Identity factor for T^H stability (e.g., 0.01 to 0.1)
    edge_tau: float = 0.02           # Divisor (tau) for continuous signal sizing
    rv_cap_percentile: int = 90      # Volatility cap: do not trade if RV is above this percentile (from training data)

    # HMM Parameters
    hmm_n_components: int = 3        # Number of hidden states
    test_split_ratio: float = 0.3    # Ratio of data to reserve for out-of-sample testing
    seed: int = 42                   # Global random seed for reproducibility

cfg = Config()

# Set seeds for reproducibility
np.random.seed(cfg.seed)

# Calculate derived constants
BARS_PER_HOUR = 60 / int(cfg.tf.replace('m', ''))
BARS_PER_DAY = BARS_PER_HOUR * 24
BARS_PER_YEAR = int(365.25 * BARS_PER_DAY)
# Factor to annualize Sharpe ratio from H-step non-overlapping returns
SHARPE_ANNUAL_FACTOR = np.sqrt(BARS_PER_YEAR / cfg.h_horizon)
print(f"Annualization Factor (sqrt(BarsPerYear/H)): {SHARPE_ANNUAL_FACTOR:.2f}")


# ==============================================================================
# 1. DATA FETCHING UTILITIES
# ==============================================================================

def initialize_exchange(exchange_id):
    """Initializes the exchange client."""
    try:
        # Use low default rate limit as a precaution
        exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True, 'rateLimit': 500})
        print(f"Initialized {exchange_id} exchange.")
        return exchange
    except AttributeError:
        print(f"Error: Exchange ID '{exchange_id}' not supported by CCXT.")
        return None

EXCHANGE = initialize_exchange(cfg.exchange_id)
if not EXCHANGE:
    raise SystemExit("Exiting due to failed exchange initialization.")


def fetch_ohlcv_historical(symbol, timeframe, start_dt, limit=1000):
    """Fetches OHLCV data spanning the required lookback period."""
    print(f"Fetching OHLCV data for {symbol}...")
    since_ms = int(start_dt.timestamp() * 1000)
    all_rows = []
    
    while True:
        try:
            batch = EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            print(f"Warning: Failed to fetch batch. Retrying in 5s. Error: {e}")
            time.sleep(5)
            continue
        
        if not batch: break
        
        # Avoid duplicate data points and ensure progression
        since_ms = batch[-1][0] + 1 
        all_rows += batch
        
        if len(batch) < limit: break
        
    df = pd.DataFrame(all_rows, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df.set_index('ts', inplace=True)
    df = df.drop_duplicates().sort_index()
    print(f"Fetched {len(df)} OHLCV bars.")
    return df


def fetch_open_interest_historical(symbol, interval, start_dt, limit=500):
    """Fetches historical Binance Futures Open Interest data using robust pagination (backward traversal)."""
    print(f"Fetching Open Interest data for {symbol}...")
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    
    all_rows = []
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int(start_dt.timestamp() * 1000)
    retries = 0

    while True:
        params = {"symbol": symbol, "period": interval, "limit": limit, "endTime": end_ms}
        
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            retries = 0 # Reset retries on success

            if not data:
                break
            
            # Filter batch to keep only records newer than the start time
            batch = [row for row in data if row['timestamp'] >= start_ms]
            
            if not batch:
                # If the entire batch is older than the start time, we're done
                break

            all_rows.extend(batch)
            
            # Move cursor to the timestamp immediately before the oldest record in this batch
            end_ms = data[-1]['timestamp'] - 1
            
            # Respect Binance rate limit (1200 req/min = 50ms/req, use 120ms to be safe)
            time.sleep(0.12)

        except requests.exceptions.HTTPError as e:
            if r.status_code == 429 and retries < 5:
                print(f"Rate limit hit (429). Exponential backoff: {2**retries}s")
                time.sleep(2**retries)
                retries += 1
            else:
                print(f"HTTP Error or too many retries: {e}. Stopping OI fetch.")
                break
        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to fetch OI batch. Error: {e}. Stopping OI fetch.")
            break
            
    if not all_rows:
        print("Warning: Could not fetch any Open Interest data.")
        return pd.DataFrame({'sumOpenInterest': []})

    df = pd.DataFrame(all_rows)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['sumOpenInterest'] = df['sumOpenInterest'].astype(float)
    df = df.set_index('timestamp')['sumOpenInterest']
    
    df = df.sort_index().drop_duplicates()
    print(f"Fetched {len(df)} Open Interest bars.")
    return df


# ==============================================================================
# 2. DATA ACQUISITION & FEATURE ENGINEERING
# ==============================================================================

END_DT = datetime.now(timezone.utc)
START_DT = END_DT - timedelta(days=cfg.lookback_days)

# 2.1. Fetch data
df_ohlcv = fetch_ohlcv_historical(cfg.symbol_ohlcv, cfg.tf, START_DT)
df_oi = fetch_open_interest_historical(cfg.symbol_oi, cfg.tf, START_DT)

# 2.2. Sanity check OI coverage
oi_coverage = len(df_oi.index.intersection(df_ohlcv.index)) / len(df_ohlcv)
if oi_coverage < cfg.oi_coverage_min:
    raise SystemExit(f"Low OI coverage: {oi_coverage:.1%}. Must be > {cfg.oi_coverage_min:.1%}. Aborting.")
else:
    print(f"OI coverage sufficient: {oi_coverage:.1%}.")

# 2.3. Join and Feature Engineer
df = df_ohlcv.copy()
# Strict inner join ensures OHLCV and OI data are present for the bar
df = df.join(df_oi, how='inner') 

# Log returns
df['ret'] = np.log(df['close']).diff()
# Annualization factor for 15m
df['rv'] = df['ret'].rolling(20).std() * np.sqrt(BARS_PER_YEAR) 

# Open Interest feature: clipped percentage change
df['dOI'] = df['sumOpenInterest'].pct_change().clip(-0.2, 0.2).fillna(0)

# CORRECT H-step forward log-return starting at t (next H bars)
df['fwd_ret_H'] = df['ret'].rolling(cfg.h_horizon).sum().shift(-cfg.h_horizon)

# Final cleanup after all feature calculations (removes NaNs from diff/rolling/shift)
df_features = df[['ret', 'rv', 'dOI', 'fwd_ret_H']].dropna() 
print(f"\nTotal bars available for modeling: {len(df_features)}")

# Define train/test splits based on the cleaned feature dataframe
N_TOTAL = len(df_features)
N_TEST = int(N_TOTAL * cfg.test_split_ratio)
N_TRAIN = N_TOTAL - N_TEST

# Indices for splitting
IDX_ALL = df_features.index
IDX_TRAIN = IDX_ALL[:N_TRAIN]
IDX_TEST = IDX_ALL[N_TRAIN:]

# === Guard Clause: Check if enough training data exists for MC window ===
if N_TRAIN < cfg.w_window:
    raise SystemExit(f"Training bars ({N_TRAIN}) is less than Markov window ({cfg.w_window}). Increase LOOKBACK_DAYS or decrease W_WINDOW.")

print(f"Training bars: {N_TRAIN}, Testing bars: {N_TEST}")
print(f"Train Date Range: {IDX_TRAIN[0].date()} to {IDX_TRAIN[-1].date()}")
print(f"Test Date Range: {IDX_TEST[0].date()} to {IDX_TEST[-1].date()}")


# ==============================================================================
# 3. MARKOV CHAIN (MC) STRATEGY
# ==============================================================================

def trans_mat(s, K_states, alpha, lambd):
    """
    Calculates the KxK Markov transition matrix with Laplace smoothing 
    and mixing with the Identity matrix (I) for stability.
    """
    I = np.eye(K_states)
    
    # 1. Laplace Smoothing (Dirichlet Prior)
    T = np.full((K_states, K_states), alpha)
    for a,b in zip(s[:-1], s[1:]): 
        T[a,b] += 1
        
    T_prior = T / np.clip(T.sum(axis=1, keepdims=True), 1, None)
    
    # 2. Mixing with Identity Matrix
    T_mixed = (1 - lambd) * T_prior + lambd * I
    
    # 3. Final normalization 
    T_final = T_mixed / T_mixed.sum(axis=1, keepdims=True)
    return T_final


def run_markov_chain_strategy(df_features, states_all, q_cutoffs, rv_cap, cfg):
    """Runs the walk-forward Markov Chain strategy."""

    print(f"\n--- Running Markov Chain Strategy (W={cfg.w_window}, H={cfg.h_horizon}) ---")
    
    signals_mc = []
    start_bar_index = N_TRAIN
    
    # Loop over the test set, using a stride of H_HORIZON for non-overlapping trades
    # Start loop at the beginning of the test set, step by H_HORIZON
    loop_range = range(start_bar_index, N_TOTAL - cfg.h_horizon, cfg.h_horizon)
    for j, i in enumerate(loop_range):
        # Training window: W bars *before* bar i
        s_win = states_all[i - cfg.w_window : i]
        
        # Assert W_WINDOW length
        assert len(s_win) == cfg.w_window, "MC window length mismatch."
        
        # Calculate and stabilize transition matrix T
        T = trans_mat(s_win, cfg.k_states, cfg.smoothing_alpha, cfg.smoothing_lambda)
        
        # Apply numeric floor and re-normalize before powering (stability)
        T = np.maximum(T, 1e-9)
        T = T / T.sum(axis=1, keepdims=True)
        
        # Calculate the H-step ahead transition matrix: T^H
        Th = np.linalg.matrix_power(T, cfg.h_horizon)
        
        # Current state is the state at bar i
        cur = states_all[i]
        e = np.eye(cfg.k_states)[cur] 
        probs = e @ Th 
        
        p_up = probs[cfg.k_states-1]
        p_down = probs[0]

        # Continuous trading rule: size position based on edge/tau
        edge = p_up - p_down
        sig = np.clip(edge / cfg.edge_tau, -1, 1) # Position size between -1 and +1

        # Volatility Gating: Flat signal if realized volatility is too high
        current_rv = df_features['rv'].iloc[i]
        if current_rv > rv_cap:
            sig = 0 # Gate the trade
        
        signals_mc.append(sig)

    # Filter the index to match the strided walk-forward loop
    idx_mc_test_strided = IDX_ALL[start_bar_index : N_TOTAL - cfg.h_horizon : cfg.h_horizon]
    sig_mc = pd.Series(signals_mc, index=idx_mc_test_strided, name='sig_mc')

    # Next H-step return *already compounded* from the 'fwd_ret_H' column
    fwd_ret_mc = df_features['fwd_ret_H'].reindex(idx_mc_test_strided)
    strat_ret_mc = sig_mc * fwd_ret_mc

    perf_mc = pd.DataFrame({'ret': fwd_ret_mc, 'strat': strat_ret_mc}).dropna()
    
    return perf_mc


# --- 3.1 Discretize returns using only Training Set quantiles (Frozen Bins) ---
ret_train = df_features['ret'].loc[IDX_TRAIN]
quantiles = [i/cfg.k_states for i in range(1, cfg.k_states)] 
q_cutoffs = ret_train.quantile(quantiles).values 

print(f"MC Quantile Cutoffs (used to define states): {q_cutoffs}")

# Redefine states using frozen quantiles and clipping for all data
states_all = np.clip(
    np.digitize(df_features['ret'].values, q_cutoffs), 
    0, cfg.k_states - 1
)

# --- 3.2 Volatility Gating Threshold ---
rv_train = df_features['rv'].loc[IDX_TRAIN]
rv_cap = np.percentile(rv_train.dropna(), cfg.rv_cap_percentile)
print(f"Volatility Cap (RV > {rv_cap:.4f} annualized) will result in a flat signal (0).")


# --- 3.3 Run and Evaluate MC Performance ---
perf_mc = run_markov_chain_strategy(df_features, states_all, q_cutoffs, rv_cap, cfg)


# ==============================================================================
# 4. HIDDEN MARKOV MODEL (HMM) STRATEGY
# ==============================================================================

def run_hmm_strategy(df_features, cfg):
    """Fits HMM on training data and evaluates on test data (strided)."""

    X_all = df_features[['ret', 'rv', 'dOI']].copy()
    Y_all = df_features['fwd_ret_H'].copy() 

    # Split and Scale
    X_tr = X_all.loc[IDX_TRAIN]
    X_te = X_all.loc[IDX_TEST]
    Y_tr = Y_all.loc[IDX_TRAIN]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    # Fit HMM
    print(f"\n--- Fitting HMM with {cfg.hmm_n_components} Components on Training Data ---")
    hmm = GaussianHMM(n_components=cfg.hmm_n_components, 
                      covariance_type='full', 
                      n_iter=200, 
                      random_state=cfg.seed, 
                      min_covar=1e-6)
    hmm.fit(X_tr_scaled)

    if hmm.monitor_.converged:
        print("HMM converged successfully.")
    else:
        print("HMM did NOT converge after max iterations.")

    # Decode Regimes (Test Set)
    reg_te = hmm.predict(X_te_scaled)

    # Map Regime to Expected Return (H-step forward target from training data)
    reg_tr = hmm.predict(X_tr_scaled)
    mu_by_reg = pd.Series(Y_tr.values).groupby(reg_tr).mean() 
    mu_hat = pd.Series(reg_te, index=IDX_TEST).map(mu_by_reg).rename('mu_hat')
    
    # Signal is long/short based on the expected H-step return sign
    signal_hmm = np.sign(mu_hat).fillna(0)

    # Evaluate HMM Performance on Test Set (Strided)
    idx_hmm_test_strided = IDX_TEST[::cfg.h_horizon]

    signal_hmm_strided = signal_hmm.reindex(idx_hmm_test_strided)
    fwd_ret_te_strided = Y_all.reindex(idx_hmm_test_strided)

    strat_ret_hmm = signal_hmm_strided * fwd_ret_te_strided
    perf_hmm = pd.DataFrame({'ret': fwd_ret_te_strided, 'strat': strat_ret_hmm}).dropna()
    
    return perf_hmm


perf_hmm = run_hmm_strategy(df_features, cfg)


# ==============================================================================
# 5. DIAGNOSTICS AND RESULTS
# ==============================================================================

def calculate_results(perf_df, strategy_name):
    """Calculates and prints final backtest metrics."""
    results = {}
    
    if perf_df.empty:
        print(f"\n--- {strategy_name} Performance: No trading observations found. ---")
        return results

    sharpe = perf_df['strat'].mean() / perf_df['strat'].std() * SHARPE_ANNUAL_FACTOR
    hit_rate = (np.sign(perf_df['strat']) == np.sign(perf_df['ret'])).mean()
    
    print(f"\n--- {strategy_name} Out-of-Sample Performance (Test Set, Strided H={cfg.h_horizon}) ---")
    print(f"Test Period: {perf_df.index[0].date()} to {perf_df.index[-1].date()}")
    print(f"Hit-rate: {hit_rate:.4f}")
    print(f"Sharpe (Annualized): {sharpe:.4f}")

    results['name'] = strategy_name
    results['sharpe'] = sharpe
    results['hit_rate'] = hit_rate
    results['test_start'] = str(perf_df.index[0].date())
    results['test_end'] = str(perf_df.index[-1].date())
    
    # Plotting for quick sanity check (requires interactive environment)
    # try:
    #     perf_df[['ret', 'strat']].cumsum().plot(
    #         title=f'{strategy_name} Equity Curve (Test, Strided)',
    #         figsize=(10, 6)
    #     )
    #     plt.show()
    # except Exception as e:
    #     # Print exception if plotting fails
    #     # print(f"Plotting failed: {e}")
    #     pass

    return results


mc_results = calculate_results(perf_mc, "Markov Chain")
hmm_results = calculate_results(perf_hmm, "Hidden Markov Model")

final_summary = {
    'config': cfg.__dict__,
    'markov_chain': mc_results,
    'hmm': hmm_results
}

# Print full results dictionary (for consumption by environment)
print("\n--- FINAL SUMMARY (JSON) ---")
print(json.dumps(final_summary, indent=4))
