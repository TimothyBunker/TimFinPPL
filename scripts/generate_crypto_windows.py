#!/usr/bin/env python3
"""
Generate sliding‑window tensors for crypto RL agents.

Reads a Parquet file produced by `fetch_crypto_data.py` that contains columns:
    Date, Ticker, spot_price, perp_price, fundingRate, basis

Outputs a NumPy `.npy` file of shape:
    (T_windows, n_assets, lookback, n_features)
where each slice is ordered [assets, time‑within‑window, features].

Fixes vs. old version
---------------------
* **Axis order bug** – source panel is (L, A, F); we now transpose to (A, L, F)
  before assigning into the pre‑allocated `windows` array.
* **Ticker mismatch guard** – throws a clear error if requested `--tickers` are
  absent from the frame.
* **Verbose summary** – prints final tensor shape and date coverage.
"""
from __future__ import annotations

import argparse, os, sys
from typing import List

import numpy as np
import pandas as pd

# ────────────────────────── CLI ──────────────────────────

def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("Generate crypto sliding windows")
    p.add_argument("--parquet", required=True,
                   help="Input Parquet with Date, Ticker, and feature columns")
    p.add_argument("--lookback", type=int, required=True,
                   help="Length of each sliding window")
    p.add_argument("--features", required=True,
                   help="Comma‑separated list of feature column names")
    p.add_argument("--tickers", required=True,
                   help="Comma‑separated list of tickers (asset order in window)")
    p.add_argument("--output", required=True,
                   help="Path to output .npy file")
    return p.parse_args()

# ────────────────────────── Main ─────────────────────────

def main() -> None:
    a = cli()
    feats: List[str] = a.features.split(',')
    tickers: List[str] = a.tickers.split(',')

    df = pd.read_parquet(a.parquet)
    missing_cols = set(['Date','Ticker'] + feats) - set(df.columns)
    if missing_cols:
        sys.exit(f"Parquet missing columns: {missing_cols}")

    missing_tickers = set(tickers) - set(df['Ticker'].unique())
    if missing_tickers:
        sys.exit(f"Tickers not in data: {missing_tickers}")

    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values('Date')
    dates = df['Date'].unique()

    T_total      = len(dates)
    lookback     = a.lookback
    if lookback > T_total:
        sys.exit(f"Lookback {lookback} > data length {T_total}")
    T_windows    = T_total - lookback + 1
    n_assets     = len(tickers)
    n_features   = len(feats)

    # Build (T, A, F) panel
    panels = []
    for feat in feats:
        pivot = df.pivot(index='Date', columns='Ticker', values=feat)
        pivot = pivot.reindex(index=dates, columns=tickers)
        panels.append(pivot.to_numpy(dtype=np.float32)[..., None])
    data = np.concatenate(panels, axis=2)  # shape (T, A, F)

    # Allocate output
    windows = np.empty((T_windows, n_assets, lookback, n_features), dtype=np.float32)
    for i in range(T_windows):
        # slice (lookback, A, F) → transpose to (A, lookback, F)
        windows[i] = data[i:i+lookback].transpose(1, 0, 2)

    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    np.save(a.output, windows)

    print(f"✅ Wrote {a.output}  shape {windows.shape}")
    print(f"Dates covered: {str(dates[0])[:10]} → {str(dates[-1])[:10]}")

if __name__ == '__main__':
    main()
