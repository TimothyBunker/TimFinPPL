#!/usr/bin/env python3
"""
Generate sentiment feature windows for the SentimentAgent.

This script reads a parquet of time-series data with a sentiment column,
aligns it with a precomputed market window file, and outputs a .npy
array of shape (T, n_assets, n_sent_features) suitable for
use with agents/train_specialized.py --agent sentiment.

Usage:
  python scripts/generate_sentiment_windows.py \
      --parquet data/processed/train.parquet \
      --windows data/processed/train_windows_50.npy \
      --sentiment_col sentiment_pred \
      --output data/processed/sentiment_windows_50.npy
"""
import os
import argparse
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sentiment windows from parquet and market windows"
    )
    parser.add_argument(
        '--parquet', type=str, required=True,
        help='Path to input parquet with Date, Ticker, and sentiment column'
    )
    parser.add_argument(
        '--windows', type=str, required=True,
        help='Path to market windows .npy of shape (T, n_assets, lookback, n_features)'
    )
    parser.add_argument(
        '--sentiment_col', type=str, default='sentiment_pred',
        help='Column name for sentiment feature in parquet'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output .npy file path for sentiment windows'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    # Load market windows to get dimensions
    market_win = np.load(args.windows)
    T_dim, n_assets, lookback, n_features = market_win.shape

    # Load parquet and pivot sentiment
    df = pd.read_parquet(args.parquet)
    if 'Date' not in df.columns or 'Ticker' not in df.columns:
        raise RuntimeError('Input parquet must have Date and Ticker columns')
    # ensure sorted dates
    df = df.sort_values('Date')
    # pivot to (n_dates, n_assets)
    pivot = df.pivot(index='Date', columns='Ticker', values=args.sentiment_col)
    # enforce asset order consistent with windows (assumed alphabetical)
    tickers = sorted(pivot.columns.tolist())
    pivot = pivot[tickers]
    arr = pivot.values.astype(np.float32)
    total_time, n_assets_df = arr.shape
    if n_assets_df != n_assets:
        raise RuntimeError(f"Asset count mismatch: windows n_assets={n_assets}, parquet has {n_assets_df}")

    # slice sentiment series to align with market windows
    start = lookback - 1
    end = start + T_dim
    if end > total_time:
        raise RuntimeError(
            f"Not enough sentiment data: need at least {end} time steps, got {total_time}"
        )
    sent_win = arr[start:end, :]
    # reshape to (T_dim, n_assets, 1)
    sent_win = sent_win.reshape(T_dim, n_assets, 1)

    # Save
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.output, sent_win)
    print(f"Wrote sentiment windows to {args.output}, shape {sent_win.shape}")

if __name__ == '__main__':
    main()