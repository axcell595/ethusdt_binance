#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production-grade Markov + HMM ETH/USDT research script (Binance).
Implements:
- Deterministic seeding
- Robust OHLCV + OI fetching with pagination/backoff
- Strict timestamp alignment (INNER JOIN)
- Features: ret, rv, dOI
- Correct H-step forward return alignment (t -> sum of t+1..t+H)
- Markov chain with rolling in-window quantile bins (no leakage), Laplace prior + identity mix, numeric floors
- Non-overlapping, strided backtest with fee/turnover
- HMM on robust-scaled features; regime -> E[fwd_ret_H|regime] learned on train
- Diagnostics/guards
"""
for later upgrades:
-Add live execution
-Export signals to CSV
-Run Monte Carlo or walk-forward optimization
