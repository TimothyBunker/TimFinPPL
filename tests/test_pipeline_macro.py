import pandas as pd
import numpy as np
import tempfile
import os

import pytest

from Data.pipeline import DataPipeline

def make_price_df():
    # 4 days of data for 1 ticker, raw price columns required by processor
    dates = pd.date_range('2021-01-01', periods=4)
    rows = []
    for date in dates:
        # Dummy price and volume
        price = float(date.day)
        rows.append({
            'Date': date,
            'Ticker': 'X',
            'Open': price,
            'High': price,
            'Low': price,
            'Close': price,
            'Volume': 100.0
        })
    return pd.DataFrame(rows)

def make_macro_df():
    # 4 days macro data with fields m1, m2
    dates = pd.date_range('2021-01-01', periods=4)
    rows = []
    for i, date in enumerate(dates, start=1):
        rows.append({
            'Date': date,
            'm1': float(i * 10),
            'm2': float(i * -1)
        })
    return pd.DataFrame(rows)

def test_macro_merge(tmp_path):
    # Create synthetic price data
    price_df = make_price_df()
    # Save price_df to a temporary parquet via DataLoader mockups
    # Monkeypatch DataLoader.fetch_data
    class DummyLoader:
        def __init__(self, df): self.df = df
        def fetch_data(self): return self.df

    # Create macro_file
    macro_df = make_macro_df()
    macro_file = tmp_path / 'macro.parquet'
    macro_df.to_parquet(macro_file, index=False)

    # Initialize pipeline with dummy loader and test settings
    dp = DataPipeline(
        tickers=['X'],
        start_date='2021-01-01',
        end_date=None,
        interval='1d',
        test_start_date='2021-01-03',
        exclude_columns=['f1'],
        output_dir=str(tmp_path),
        sentiment_file=None,
        macro_file=str(macro_file),
        lookback_window=None,
        skip_indicators=True
    )
    # Replace loader
    dp.loader = DummyLoader(price_df)

    train_df, test_df = dp.run()
    # Check macro columns exist and values match input macro_df
    for df in [train_df, test_df]:
        assert 'm1' in df.columns and 'm2' in df.columns
        # For each row, m1 and m2 should match macro_df by Date
        for _, row in df.iterrows():
            date = row['Date']
            macro_row = macro_df[macro_df['Date'] == date].iloc[0]
            assert row['m1'] == macro_row['m1']
            assert row['m2'] == macro_row['m2']