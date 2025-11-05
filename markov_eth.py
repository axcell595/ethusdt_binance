# markov_eth.py
import ccxt, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone

EXCHANGE = ccxt.binance()   # public endpoints only
SYMBOL = 'ETH/USDT'
TF = '15m'  # change preferred TF here
LOOKBACK_DAYS = 120


def fetch_ohlcv(symbol, timeframe, since_ms, limit=1000):
    all_rows = []
    while True:
        batch = EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        if not batch: break
        all_rows += batch
        since_ms = batch[-1][0] + 1
        if len(batch) < limit: break
    return all_rows


end = datetime.now(timezone.utc)
start = end - timedelta(days=LOOKBACK_DAYS)
since_ms = int(start.timestamp() * 1000)

rows = fetch_ohlcv(SYMBOL, TF, since_ms)
df = pd.DataFrame(rows, columns=['ts','open','high','low','close','volume'])
df['ts'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
df.set_index('ts', inplace=True)
df = df.drop_duplicates().sort_index()

# log returns and simple realized volatility feature
df['ret'] = np.log(df['close']).diff()
df['rv'] = df['ret'].rolling(20).std() * np.sqrt(60*24*365)  # annualized-ish on 15m
df = df.dropna()

print(df.tail())


# === discretize return into states
K = 3
q = df['ret'].quantile([0, .33, .66, 1]).values
states = np.digitize(df['ret'].values, q[1:-1])  # 0,1,2
df['state'] = states

# === rolling transition matrices and signals
W = 2000  # rolling windows in bars
H = 4  # horizon in steps (e.g. 4x15m = 1h)
signals = []  # +1 long, -1 short, 0 flat
p_up_list = []


def trans_mat(s):
    T = np.zeros((K, K))
    for a,b in zip(s[:-1], s[1:]): T[a,b] += 1
    T = T / np.clip(T.sum(1, keepdims=True), 1, None)
    return np.nan_to_num(T)


for i in range(W, len(df)-H):
    s_win = states[i-W:i]
    T = trans_mat(s_win)
    cur = states[i]
    e = np.eye(K)[cur]
    Th = np.linalg.matrix_power(T, H)
    probs = e @ Th
    p_up = probs[2]; p_down = probs[0]
    p_up_list.append(p_up)

    # simple rule: go long if edge > threshold; short if negative edge
    edge = p_up - p_down
    if edge > 0.10: sig = +1
    elif edge < -0.10: sig = -1
    else: sig = 0
    signals.append(sig)

# align series
idx = df.index[W:len(df)-H]
sig = pd.Series(signals, index=idx, name='sig')
pup = pd.Series(p_up_list, index=idx, name='p_up')

# naive execution: hold signal for next H steps
fwd_ret = df['ret'].shift(-H).reindex(idx)
strat_ret = sig * fwd_ret

perf = pd.DataFrame({'ret': fwd_ret, 'strat':strat_ret}).dropna()
print('Hit-rate:', (np.sign(strat_ret)==np.sign(fwd_ret)).mean())
print('Sharpe (naive, no fees):', perf['strat'].mean()/perf['strat'].std()*np.sqrt(365*24*60/(15)))

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# prepare features
# X = df[['ret','rv']].dropna().copy()
X = df[['ret','rv', 'dOI']].dropna().copy()  # with Binance Open Interest feature
X_index = X.index
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# walk forward split
split = int(len(X)*0.7)
X_tr, X_te = X_scaled[:split], X_scaled[split:]
idx_tr, idx_te = X_index[:split], X_index[split:]

# fit HMM (2-4 components is typical)
hmm = GaussianHMM(n_components=3, covariance_type='full', n_iter=200, random_state=0)
hmm.fit(X_tr)

# decode regimes on test and get next step probabilities
post = hmm.predict_proba(X_te)  # P(Z_t = k | x_{1:t})
reg = post.argmax(1)  # most likely regime

# regime conditional expected next return (on training)
# compute E[ret | regime] using training set alignment
reg_tr = hmm.predict(X_tr)
mu_by_reg = pd.Series(df['ret'].reindex(idx_tr).values).groupby(reg_tr).mean()

# map regime to expectations on test
mu_hat = pd.Series(reg, index=idx_te).map(mu_by_reg).rename('mu_hat')
signal_hmm = np.sign(mu_hat).fillna(0)  # long if expected positive

# next-H step return on test
H = 4
fwd_ret_te = df['ret'].shift(-H).reindex(idx_te)
strat_ret_hmm = signal_hmm * fwd_ret_te
print('HMM hit-rate:', (np.sign(strat_ret_hmm)==np.sign(fwd_ret_te)).mean())


# == Binance derivatives Open Interest
import requests, time


def binance_oi(symbol='ETHUSDT', interval='15m', limit=500):
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=20); r.raise_for_status()
    j = r.json()
    d = pd.DataFrame(j)
    d['timestamp'] = pd.to_datetime(d['timestamp'], unit='ms', utc=True)
    d = d[['timestamp', 'sumOpenInterest']]
    d['sumOpenInterest'] = d['sumOpenInterest'].astype(float)
    d.set_index('timestamp', inplace=True)
    return d


oi = binance_oi('ETHUSDT', '15m', 500)
df = df.join(oi, how='left').ffill()
df['dOI'] = df['sumOpenInterest'].pct_change().clip(-0.2,0.2).fillna(0)
