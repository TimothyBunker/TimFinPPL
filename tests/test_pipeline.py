import pandas as pd
import numpy as np
import pytest

from Data.pipeline import DataPipeline

def make_synthetic_df():
    # 5 days of data for 2 tickers, features f1 (day of month) and f2 (ticker length)
    dates = pd.date_range('2020-01-01', periods=5)
    tickers = ['A', 'BB']
    rows = []
    for date in dates:
        for t in tickers:
            rows.append({
                'Date': date,
                'Ticker': t,
                'f1': float(date.day),
                'f2': float(len(t))
            })
    return pd.DataFrame(rows)

@pytest.mark.parametrize("window_size,expected_shape", [
    (3, (2, 2, 3, 2)),  # 5 days -> 5-3=2 windows
    (1, (4, 2, 1, 2)),  # 5-1=4 windows
])
def test_generate_windows_shapes(window_size, expected_shape):
    df = make_synthetic_df()
    dp = DataPipeline(
        tickers=['A', 'BB'],
        start_date='2020-01-01',
        end_date=None,
        interval='1d',
        test_start_date='2020-01-03',
        exclude_columns=['f1', 'f2'],
        output_dir='data/processed',
        sentiment_file=None,
        lookback_window=window_size
    )
    windows = dp.generate_windows(df)
    assert windows.shape == expected_shape

def test_generate_windows_content():
    df = make_synthetic_df()
    window_size = 3
    dp = DataPipeline(
        tickers=['A', 'BB'],
        start_date='2020-01-01',
        end_date=None,
        interval='1d',
        test_start_date='2020-01-03',
        exclude_columns=['f1', 'f2'],
        output_dir='data/processed',
        sentiment_file=None,
        lookback_window=window_size
    )
    windows = dp.generate_windows(df)
    # First window (days 1,2,3) for ticker 'A': f1 [1,2,3], f2=1.0 each
    expected_A = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    np.testing.assert_array_equal(windows[0][0], expected_A)
    # First window for ticker 'BB': f1 [1,2,3], f2=2.0
    expected_BB = np.array([[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]])
    np.testing.assert_array_equal(windows[0][1], expected_BB)