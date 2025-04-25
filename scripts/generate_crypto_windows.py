#!/usr/bin/env python3
"""
Generate sliding window arrays for crypto data.

This script loads a Parquet with columns [Date, Ticker, spot_price, perp_price, fundingRate, basis],
pivots into wide format, and produces a .npy array of shape
(T-lookback+1, n_assets, lookback, n_features) for use in CryptoTradingEnv.

Usage:
  python scripts/generate_crypto_windows.py \
    --parquet data/crypto/crypto_data.parquet \
    --lookback 50 \
    --features spot_price,perp_price,fundingRate,basis \
    --tickers BTC/USDT,ETH/USDT \
    --output data/processed/crypto_windows_50.npy
"""
import argparse
import os
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Generate crypto sliding windows')
    parser.add_argument('--parquet', type=str, required=True,
                        help='Input Parquet file with Date, Ticker, and feature columns')
    parser.add_argument('--lookback', type=int, required=True,
                        help='Length of each sliding window')
    parser.add_argument('--features', type=str, required=True,
                        help='Comma-separated list of feature column names to include')
    parser.add_argument('--tickers', type=str, required=True,
                        help='Comma-separated list of tickers (column order in window)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output .npy file for windows')
    return parser.parse_args()

def main():
    args = parse_args()
    # Load DataFrame
    df = pd.read_parquet(args.parquet)
    # Ensure proper columns
    feats = args.features.split(',')
    required = set(['Date', 'Ticker'] + feats)
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f'Missing columns in parquet: {missing}')
    # Convert Date to datetime and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    # Unique dates and assets
    dates = sorted(df['Date'].unique())
    tickers = args.tickers.split(',')
    T_total = len(dates)
    n_assets = len(tickers)
    n_features = len(feats)
    # Build wide feature panel: shape (T_total, n_assets, n_features)
    panels = []
    for feat in feats:
        pivot = df.pivot(index='Date', columns='Ticker', values=feat)
        pivot = pivot.reindex(index=dates, columns=tickers)
        arr = pivot.values.astype(np.float32)
        panels.append(arr[..., None])
    data = np.concatenate(panels, axis=2)
    # Sliding windows
    lookback = args.lookback
    if lookback > T_total:
        raise RuntimeError(f'Lookback {lookback} > data length {T_total}')
    T_windows = T_total - lookback + 1
    windows = np.zeros((T_windows, n_assets, lookback, n_features), dtype=np.float32)
    for i in range(T_windows):
        windows[i] = data[i:i+lookback]
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, windows)
    print(f'Wrote crypto windows to {args.output}, shape {windows.shape}')

if __name__ == '__main__':
    main()