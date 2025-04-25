import yfinance as yf
import pandas as pd
import ta
from ta.volume import OnBalanceVolumeIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

# from FinEnv import scaled_features  # removed unused import (legacy)

# Mapping from textual analyst rating to numeric.
RATING_MAP = {
    "Strong Buy":    2,
    "Buy":           1,
    "Hold":          0,
    "Unknown":       0,  # or None, depending on preference
    "Underperform": -1,
    "Sell":         -2
}

class DataLoader:
    """
    Fetches data from Yahoo Finance for given tickers and date range.
    """
    def __init__(self, tickers, start_date="2000-01-01", end_date=None,interval="1d"):
        self.tickers = tickers
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        """
        Download the data from Yahoo, stack it into a flat DataFrame.
        """
        data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            group_by='ticker',
            threads=True
        )
        print("Initial Data Head:")
        print(data.head().to_string())

        # Stack the data from wide to long format (Date/Ticker index)
        # Use level=0 to pivot tickers into the index
        stacked = (
            data.stack(level=0)
                .rename_axis(['Date', 'Ticker'])
                .reset_index(level=1)
        )

        print("Stacked Data Head:")
        print(stacked.head())

        return pd.DataFrame(stacked)
    
    def fetch_sentiment(self):
        """
        Fetch news headlines for each ticker and compute a simple sentiment score per date.
        Uses a lexicon-based approach on titles.
        Returns a DataFrame with columns ['Date', 'Ticker', 'sentiment_score'].
        """
        # Simple word lists for sentiment
        POS_WORDS = ['good','great','positive','beat','bullish','up','strong','profit','gain']
        NEG_WORDS = ['bad','poor','negative','miss','bearish','down','weak','loss','drop','risk']
        records = []
        # Iterate tickers
        for ticker in self.tickers:
            try:
                tk = yf.Ticker(ticker)
                news_items = tk.news
            except Exception:
                news_items = []
            for item in news_items:
                ts = item.get('providerPublishTime')
                title = item.get('title', '')
                if ts is None or not title:
                    continue
                # Convert timestamp to date
                try:
                    dt = pd.to_datetime(ts, unit='s').normalize()
                except Exception:
                    continue
                text = title.lower()
                pos = sum(text.count(w) for w in POS_WORDS)
                neg = sum(text.count(w) for w in NEG_WORDS)
                score = (pos - neg) / (pos + neg + 1e-8)
                records.append({'Date': dt, 'Ticker': ticker, 'sentiment_score': score})
        if not records:
            return pd.DataFrame(columns=['Date','Ticker','sentiment_score'])
        df_sent = pd.DataFrame(records)
        # Average scores per date & ticker
        df_sent = df_sent.groupby(['Date','Ticker'], as_index=False)['sentiment_score'].mean()
        return df_sent


class DataProcessor:
    """
    Cleans raw price data, computes technical indicators, sentiment,
    and scales the numeric columns.
    """
    def __init__(self, tickers):
        self.tickers = tickers

    def calculate_recommendation_score(self, row):
        """
        Calculate a weighted recommendation score from analyst recommendation counts.
        Higher score indicates more buy recommendations, lower indicates more sell.
        """
        score = (
                2 * row.get("strongBuy", 0) +
                1 * row.get("buy", 0) +
                0 * row.get("hold", 0) +
                -1 * row.get("sell", 0) +
                -2 * row.get("strongSell", 0)
        )
        return score

    def scale_features(self, df, exclude_columns):
        scaler = MinMaxScaler(feature_range=(0, 1))
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        columns_to_scale = [col for col in numeric_cols if col not in exclude_columns]
        if columns_to_scale:
            df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        return df

    def compute_indicators(self, df):
        """
        Add various TA indicators, sentiment data (analyst rating, earnings surprise),
        and handle missing values + scaling. Returns an enriched DataFrame.
        """
        # ----------------
        # Moving Averages, RSI, MACD, Bollinger Bands
        # ----------------
        df['MA50'] = df.groupby('Ticker')['Close'].transform(
            lambda x: ta.trend.sma_indicator(x, window=50)
        )
        df['RSI'] = df.groupby('Ticker')['Close'].transform(
            lambda x: ta.momentum.rsi(x, window=14)
        )
        df['MACD'] = df.groupby('Ticker')['Close'].transform(ta.trend.macd)
        df['Bollinger_High'] = df.groupby('Ticker')['Close'].transform(
            lambda x: ta.volatility.bollinger_hband(x, window=20)
        )
        df['Bollinger_Low'] = df.groupby('Ticker')['Close'].transform(
            lambda x: ta.volatility.bollinger_lband(x, window=20)
        )

        # ----------------
        # OBV, VWAP, ATR, Stochastic, Williams %R, EMA, ADX
        # ----------------
        df['OBV'] = float('nan')
        for ticker, subdf in df.groupby('Ticker'):
            obv_indicator = OnBalanceVolumeIndicator(close=subdf['Close'], volume=subdf['Volume'])
            df.loc[subdf.index, 'OBV'] = obv_indicator.on_balance_volume()

        df['VWAP'] = float('nan')
        for ticker, subdf in df.groupby('Ticker'):
            vwap_ind = ta.volume.VolumeWeightedAveragePrice(
                high=subdf['High'],
                low=subdf['Low'],
                close=subdf['Close'],
                volume=subdf['Volume']
            )
            df.loc[subdf.index, 'VWAP'] = vwap_ind.volume_weighted_average_price()

        df['ATR'] = float('nan')
        for ticker, subdf in df.groupby('Ticker'):
            atr_ind = ta.volatility.AverageTrueRange(
                high=subdf['High'],
                low=subdf['Low'],
                close=subdf['Close']
            )
            df.loc[subdf.index, 'ATR'] = atr_ind.average_true_range()

        df['Stochastic_%K'] = float('nan')
        for ticker, subdf in df.groupby('Ticker'):
            stoch_ind = ta.momentum.StochasticOscillator(
                high=subdf['High'],
                low=subdf['Low'],
                close=subdf['Close']
            )
            df.loc[subdf.index, 'Stochastic_%K'] = stoch_ind.stoch()

        df['Williams_%R'] = float('nan')
        for ticker, subdf in df.groupby('Ticker'):
            w_ind = ta.momentum.WilliamsRIndicator(
                high=subdf['High'],
                low=subdf['Low'],
                close=subdf['Close']
            )
            df.loc[subdf.index, 'Williams_%R'] = w_ind.williams_r()

        df['EMA50'] = df.groupby('Ticker')['Close'].transform(
            lambda x: ta.trend.ema_indicator(x, window=50)
        )

        df['ADX'] = float('nan')
        for ticker, subdf in df.groupby('Ticker'):
            adx_ind = ta.trend.ADXIndicator(
                high=subdf['High'],
                low=subdf['Low'],
                close=subdf['Close']
            )
            df.loc[subdf.index, 'ADX'] = adx_ind.adx()

        df['Log_Returns'] = df.groupby('Ticker')['Close'].transform(lambda x: np.log(x / x.shift(1)))
        df['Pct_Change'] = df.groupby('Ticker')['Close'].transform(lambda x: x.pct_change())

        # ----------------
        # Handle Missing Data
        # ----------------
        # Fill missing values per ticker and interpolate
        df_filled = df.groupby('Ticker', group_keys=False).apply(
            lambda group: group.ffill().bfill().infer_objects().interpolate(method='linear')
        )
        # Reset index to avoid ambiguity between index and columns
        df_filled = df_filled.reset_index()

        # ----------------
        # Add sentiment features from Analyst_Rating and Earnings_Surprise if available
        if 'Analyst_Rating' in df_filled.columns:
            df_filled['sentiment_pred'] = df_filled['Analyst_Rating'].astype(float)
        else:
            df_filled['sentiment_pred'] = 0.0
        # Include earnings surprise as a separate feature if present
        if 'Earnings_Surprise' in df_filled.columns:
            df_filled['earnings_surprise'] = df_filled['Earnings_Surprise'].astype(float)
        else:
            df_filled['earnings_surprise'] = 0.0
        # Add placeholder columns for specialized agent outputs (to be filled by ensemble)
        df_filled['anomaly_score'] = 0.0
        df_filled['short_term_pred'] = 0.0
        df_filled['long_term_pred'] = 0.0
        # Return enriched DataFrame (scaling handled by external pipeline)
        return df_filled


class DataVisualizer:
    """
    Provides plotting methods to visualize DataFrames with
    time-series plots, distributions, and correlation heatmaps.
    """
    def plot_time_series(self, df, tickers, col1='Close', col2='RSI'):
        """
        Plots two columns side by side (e.g., Close and RSI)
        for each ticker in df.
        """
        for ticker in tickers:
            subdf = df[df['Ticker'] == ticker]

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 4), sharex=True)

            # Plot col1
            axes[0].plot(subdf.index, subdf[col1], label=col1, color='blue')
            axes[0].set_title(f"{ticker} - {col1}")
            axes[0].set_xlabel("Date")
            axes[0].set_ylabel(col1)

            # Plot col2
            axes[1].plot(subdf.index, subdf[col2], label=col2, color='red')
            axes[1].set_title(f"{ticker} - {col2}")
            axes[1].set_xlabel("Date")
            axes[1].set_ylabel(col2)

            plt.tight_layout()
            plt.show()

    def plot_distributions(self, df, columns=None):
        """
        Plots a histogram + KDE for the specified numeric columns.
        If no columns specified, selects all numeric columns.
        """
        if columns is None:
            columns = df.select_dtypes(include=['float64', 'int64']).columns

        for col in columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()

    def plot_correlation_heatmap(self, df):
        """
        Plots a correlation heatmap of numeric columns in df.
        """
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = df[numeric_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
        plt.title("Correlation Heatmap")
        plt.show()


# --------------------------------------------
# Example of usage
# --------------------------------------------
# if __name__ == "__main__":
#     # 1) Load data
#     tickers = ["AAPL", "MSFT", "GOOGL", "ABNB", "ANET", "BB", "DJT", "LMT", "KO"]
#
#     # data = yf.download(tickers, period="max", interval="1d")
#     # print(data.shape)
#
#     loader = DataLoader(tickers, start_date="2006-01-01", interval="1d")
#     df_raw = loader.fetch_data()
#
#     # 2) Process data
#     processor = DataProcessor(tickers)
#     df_enriched = processor.compute_indicators(df_raw)
#
#     # 3) Save Parquet
#     df_enriched.to_parquet("enriched_stock_data_with_sentiment.parquet")
#     print("Final Enriched Data Head:")
#     print(df_enriched.tail())
#
#     # 4) Visualizations on unscaled data (if you want unscaled,
#     #    you can pull it from somewhere in the pipeline or
#     #    just use df_raw + partial features, but here let's show scaled for brevity)
#     #    Typically you'd keep an unscaled version if you wanted to do time-series plots.
#     visualizer = DataVisualizer()
#     # For demonstration, we'll pass the processed (scaled) DataFrame.
#     # Ideally, you'd keep a copy of the "filled but unscaled" DF if you want
#     # the actual prices. We'll do it anyway:
#     unique_tickers = df_enriched['Ticker'].unique()
#     # Time Series Example: might not look "price-like" due to scaling,
#     # but good for demonstration.
#     visualizer.plot_time_series(df_enriched, unique_tickers, col1='Close', col2='RSI')
#
#     # Distribution
#     numeric_columns_to_plot = ['Close', 'OBV', 'ATR', 'MACD', 'RSI']
#     visualizer.plot_distributions(df_enriched, columns=numeric_columns_to_plot)
#
#     # Correlation Heatmap
#     visualizer.plot_correlation_heatmap(df_enriched)
