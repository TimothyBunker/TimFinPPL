"""
Data pipeline for fetching, processing, splitting, and scaling financial time-series data.
"""
import os, sys
# Ensure project root is on path for imports when running as script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from Data.DataAggregator import DataLoader, DataProcessor


class DataPipeline:
    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str],
        interval: str,
        test_start_date: str,
        exclude_columns: List[str],
        output_dir: str = "data/processed",
        sentiment_file: Optional[str] = None,
        auto_sentiment: bool = False,
        macro_file: Optional[str] = None,
        lookback_window: Optional[int] = None,
        skip_indicators: bool = False
    ):
        self.loader = DataLoader(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        self.processor = DataProcessor(tickers=tickers)
        self.test_start_date = pd.to_datetime(test_start_date)
        self.exclude_columns = exclude_columns
        self.output_dir = output_dir
        self.sentiment_file = sentiment_file
        self.auto_sentiment = auto_sentiment
        self.macro_file = macro_file
        os.makedirs(self.output_dir, exist_ok=True)
        self.lookback_window = lookback_window
        self.skip_indicators = skip_indicators
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def generate_windows(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate sliding windows of shape (n_windows, n_assets, lookback_window, n_features).
        Requires DataFrame with columns [Date, Ticker, feature1...featureN].
        """
        if self.lookback_window is None or self.lookback_window <= 0:
            raise ValueError("lookback_window must be a positive integer to generate windows")
        # Pivot to 3D array: (T, n_assets, n_features)
        feature_cols = [c for c in df.columns if c not in ['Date', 'Ticker']]
        dates = sorted(df['Date'].unique())
        tickers = sorted(df['Ticker'].unique())
        # Create a 3D array of shape (T, n_assets, n_features)
        cubes = []
        for date in dates:
            day_df = df[df['Date'] == date].set_index('Ticker')
            # ensure consistent ticker order
            arr = day_df.loc[tickers, feature_cols].values
            cubes.append(arr)
        data_array = np.stack(cubes, axis=0)  # shape: (T, n_assets, n_features)
        # Generate windows
        T_len = data_array.shape[0]
        windows = []
        for i in range(self.lookback_window, T_len):
            win = data_array[i-self.lookback_window:i]
            # win shape: (lookback_window, n_assets, n_features)
            # transpose to (n_assets, lookback_window, n_features)
            win = np.transpose(win, (1, 0, 2))
            windows.append(win)
        return np.stack(windows, axis=0)

    def run(self):
        # Fetch raw data
        raw_df = self.loader.fetch_data()
        # Compute indicators if not skipped
        if self.skip_indicators:
            proc_df = raw_df.copy()
        else:
            proc_df = self.processor.compute_indicators(raw_df)
        # Ensure Date is datetime
        proc_df["Date"] = pd.to_datetime(proc_df["Date"])

        # Merge sentiment data: either auto-fetch or from a file
        if self.auto_sentiment:
            sentiment_df = self.loader.fetch_sentiment()
            if 'Date' not in sentiment_df.columns:
                raise KeyError("Auto sentiment DataFrame missing 'Date' column")
            sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
            proc_df = proc_df.merge(
                sentiment_df, on=["Date", "Ticker"], how="left"
            )
        elif self.sentiment_file:
            sentiment_df = pd.read_parquet(self.sentiment_file)
            # if Date is index, bring it into a column
            if 'Date' not in sentiment_df.columns and sentiment_df.index.name == 'Date':
                sentiment_df = sentiment_df.reset_index()
            if 'Date' in sentiment_df.columns:
                sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
            else:
                raise KeyError(f"Sentiment file {self.sentiment_file} has no 'Date' column or index")
            # Merge on Date and Ticker
            proc_df = proc_df.merge(
                sentiment_df, on=["Date", "Ticker"], how="left"
            )
        # Merge additional macroeconomic data if provided
        if self.macro_file:
            # Only merge if file exists
            if not os.path.exists(self.macro_file):
                print(f"Warning: macro_file {self.macro_file} not found, skipping macro merge")
            else:
                macro_df = pd.read_parquet(self.macro_file)
                # if Date is index, bring it into a column
                if 'Date' not in macro_df.columns and macro_df.index.name == 'Date':
                    macro_df = macro_df.reset_index()
                if 'Date' in macro_df.columns:
                    macro_df['Date'] = pd.to_datetime(macro_df['Date'])
                else:
                    raise KeyError(f"Macro file {self.macro_file} has no 'Date' column or index")
                # Merge on Date only (macros apply to all tickers)
                proc_df = proc_df.merge(
                    macro_df, on=["Date"], how="left"
                )

        # Split into train/test by date
        train_df = proc_df[proc_df["Date"] < self.test_start_date].copy()
        test_df = proc_df[proc_df["Date"] >= self.test_start_date].copy()

        # Scale numeric columns if not skipping indicators (scaling)
        if not self.skip_indicators:
            numeric_cols = train_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
            scale_cols = [c for c in numeric_cols if c not in self.exclude_columns]
            # Fit scaler on train
            train_df[scale_cols] = self.scaler.fit_transform(train_df[scale_cols])
            # Apply scaler to test
            test_df[scale_cols] = self.scaler.transform(test_df[scale_cols])

        # Save processed datasets
        train_file = os.path.join(self.output_dir, "train.parquet")
        test_file = os.path.join(self.output_dir, "test.parquet")
        train_df.to_parquet(train_file, index=False)
        test_df.to_parquet(test_file, index=False)
        print(f"Saved train data to {train_file}, test data to {test_file}")
        # Optionally generate sliding windows
        if self.lookback_window:
            train_win = self.generate_windows(train_df)
            test_win = self.generate_windows(test_df)
            train_win_file = os.path.join(self.output_dir, f"train_windows_{self.lookback_window}.npy")
            test_win_file = os.path.join(self.output_dir, f"test_windows_{self.lookback_window}.npy")
            np.save(train_win_file, train_win)
            np.save(test_win_file, test_win)
            print(f"Saved train windows to {train_win_file}, test windows to {test_win_file}")
        return train_df, test_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Data pipeline: fetch, process, split, scale.")
    parser.add_argument(
        '--tickers', type=str, required=True,
        help="Comma-separated ticker list, e.g. 'AAPL,MSFT,GOOGL'"
    )
    parser.add_argument(
        '--start_date', type=str, default="2000-01-01",
        help="Start date for data fetch (YYYY-MM-DD)"
    )
    parser.add_argument(
        '--end_date', type=str, default=None,
        help="End date for data fetch (YYYY-MM-DD)"
    )
    parser.add_argument(
        '--interval', type=str, default="1d",
        help="Data interval, e.g. '1d', '1h'"
    )
    parser.add_argument(
        '--test_start_date', type=str, required=True,
        help="Date to split train/test (YYYY-MM-DD)"
    )
    parser.add_argument(
        '--exclude_columns', type=str,
        default="Open,High,Low,Close,Volume",
        help="Comma-separated columns to exclude from scaling"
    )
    parser.add_argument(
        '--output_dir', type=str, default="data/processed",
        help="Directory to save processed data"
    )
    parser.add_argument(
        '--sentiment_file', type=str, default=None,
        help="Optional Parquet file with sentiment features to merge on Date and Ticker"
    )
    parser.add_argument(
        '--auto_sentiment', action='store_true',
        help="Automatically fetch news-based sentiment via yfinance"
    )
    parser.add_argument(
        '--macro_file', type=str, default=None,
        help="Optional Parquet file with macroeconomic features to merge on Date"
    )
    parser.add_argument(
        '--lookback_window', type=int, default=None,
        help="Integer lookback window length to generate sliding windows"
    )
    parser.add_argument(
        '--skip_indicators', action='store_true',
        help="Skip computing technical indicators; use raw data directly"
    )
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(',')]
    exclude_cols = [c.strip() for c in args.exclude_columns.split(',')]
    pipeline = DataPipeline(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
        test_start_date=args.test_start_date,
        exclude_columns=exclude_cols,
        output_dir=args.output_dir,
        sentiment_file=args.sentiment_file,
        macro_file=args.macro_file,
        lookback_window=args.lookback_window,
        skip_indicators=args.skip_indicators
    )
    pipeline.run()