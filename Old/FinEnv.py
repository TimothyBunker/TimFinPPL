import yfinance as yf
import pandas as pd
import ta
from ta.volume import OnBalanceVolumeIndicator
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


rating_map = {
    "Strong Buy":  2,
    "Buy":         1,
    "Hold":        0,
    "Unknown":     0,
    "Underperform": -1,
    "Sell":        -2
}

# --------------------------------------------
# 1) Download & Prepare Initial Data
# --------------------------------------------
tickers = ['AAPL', 'MSFT', 'GOOGL']
data = yf.download(tickers, start='2020-01-01', end='2021-01-01', group_by='ticker', threads=True)

print("Initial Data Head:")
print(data.head())

# Stack the data to have a MultiIndex with ['Date', 'Ticker']
data = (
    data.stack(level=0, future_stack=True)
    .rename_axis(['Date', 'Ticker'])
    .reset_index(level=1)
)

print("Stacked Data Head:")
print(data.head())

# Convert to a standard DataFrame
df = pd.DataFrame(data)

# --------------------------------------------
# 2) Compute Technical Indicators
# --------------------------------------------

# ----------------
# Moving Averages
# ----------------
df['MA50'] = df.groupby('Ticker')['Close'].transform(
    lambda x: ta.trend.sma_indicator(x, window=50)
)

# RSI
df['RSI'] = df.groupby('Ticker')['Close'].transform(
    lambda x: ta.momentum.rsi(x, window=14)
)

# MACD
df['MACD'] = df.groupby('Ticker')['Close'].transform(ta.trend.macd)

# Bollinger Bands
df['Bollinger_High'] = df.groupby('Ticker')['Close'].transform(
    lambda x: ta.volatility.bollinger_hband(x, window=20)
)
df['Bollinger_Low'] = df.groupby('Ticker')['Close'].transform(
    lambda x: ta.volatility.bollinger_lband(x, window=20)
)

print("Missing Values Summary:")
print(
    df[['MA50', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low']]
    .isna()
    .sum()
)

# ----------------
# **OBV** via a loop (avoids groupby-apply issues!)
# ----------------
df['OBV'] = float('nan')  # Initialize column
for ticker, subdf in df.groupby('Ticker'):
    obv_indicator = OnBalanceVolumeIndicator(
        close=subdf['Close'],
        volume=subdf['Volume']
    )
    # Assign OBV values back to the main df using the subdf's index
    df.loc[subdf.index, 'OBV'] = obv_indicator.on_balance_volume()

# ----------------
# VWAP
# ----------------
def calculate_vwap(group):
    vwap_indicator = ta.volume.VolumeWeightedAveragePrice(
        high=group['High'],
        low=group['Low'],
        close=group['Close'],
        volume=group['Volume']
    )
    # This is a single Series
    return vwap_indicator.volume_weighted_average_price()

df['VWAP'] = float('nan')

for ticker, subdf in df.groupby('Ticker'):
    # subdf has only the rows for one ticker
    vwap_indicator = ta.volume.VolumeWeightedAveragePrice(
        high=subdf['High'],
        low=subdf['Low'],
        close=subdf['Close'],
        volume=subdf['Volume']
    )
    # Assign result back to the full df using subdf’s index
    df.loc[subdf.index, 'VWAP'] = vwap_indicator.volume_weighted_average_price()
# ----------------
# ATR
# ----------------
def calculate_atr(group):
    atr_indicator = ta.volatility.AverageTrueRange(
        high=group['High'],
        low=group['Low'],
        close=group['Close']
    )
    return atr_indicator.average_true_range()

df['ATR'] = float('nan')  # Initialize an empty column

for ticker, subdf in df.groupby('Ticker'):
    atr_indicator = ta.volatility.AverageTrueRange(
        high=subdf['High'],
        low=subdf['Low'],
        close=subdf['Close']
    )
    # Assign directly by the subdf index:
    df.loc[subdf.index, 'ATR'] = atr_indicator.average_true_range()

# ----------------
# Stochastic %K
# ----------------
def calculate_stochastic_k(group):
    stoch_indicator = ta.momentum.StochasticOscillator(
        high=group['High'],
        low=group['Low'],
        close=group['Close']
    )
    return stoch_indicator.stoch()

df['Stochastic_%K'] = float('nan')
for ticker, subdf in df.groupby('Ticker'):
    stoch_indicator = ta.momentum.StochasticOscillator(
        high=subdf['High'],
        low=subdf['Low'],
        close=subdf['Close']
    )
    df.loc[subdf.index, 'Stochastic_%K'] = stoch_indicator.stoch()
# ----------------
# Williams %R
# ----------------
def calculate_williams_r(group):
    williams_indicator = ta.momentum.WilliamsRIndicator(
        high=group['High'],
        low=group['Low'],
        close=group['Close']
    )
    return williams_indicator.williams_r()

df['Williams_%R'] = float('nan')
for ticker, subdf in df.groupby('Ticker'):
    w_indicator = ta.momentum.WilliamsRIndicator(
        high=subdf['High'],
        low=subdf['Low'],
        close=subdf['Close']
    )
    df.loc[subdf.index, 'Williams_%R'] = w_indicator.williams_r()

# ----------------
# EMA (50-day)
# ----------------
df['EMA50'] = df.groupby('Ticker')['Close'].transform(
    lambda x: ta.trend.ema_indicator(x, window=50)
)

# ----------------
# ADX
# ----------------
def calculate_adx(group):
    adx_indicator = ta.trend.ADXIndicator(
        high=group['High'],
        low=group['Low'],
        close=group['Close']
    )
    return adx_indicator.adx()

df['ADX'] = float('nan')
for ticker, subdf in df.groupby('Ticker'):
    adx_indicator = ta.trend.ADXIndicator(
        high=subdf['High'],
        low=subdf['Low'],
        close=subdf['Close']
    )
    df.loc[subdf.index, 'ADX'] = adx_indicator.adx()
# --------------------------------------------
# 3) Analyst Data / Sentiment
# --------------------------------------------
analyst_ratings = []
earnings_surprises = []

for ticker in tickers:
    stock = yf.Ticker(ticker)

    # Analyst Recommendations
    try:
        rec = stock.recommendations
        latest_rec = rec.iloc[-1]['To Grade'] if (rec is not None and not rec.empty) else 'Hold'
        analyst_ratings.append(latest_rec)
    except Exception:
        analyst_ratings.append('Unknown')

    # Earnings Surprises
    try:
        earnings_hist = stock.earnings_history
        if earnings_hist:
            latest_earnings = earnings_hist[-1]
            earnings_surprise = latest_earnings['actual'] - latest_earnings['estimate']
            earnings_surprises.append(earnings_surprise)
        else:
            earnings_surprises.append(0.0)
    except Exception:
        earnings_surprises.append(0.0)

df['Analyst_Rating'] = df['Ticker'].map(dict(zip(tickers, analyst_ratings))).map(rating_map).fillna(0)
df['Earnings_Surprise'] = df['Ticker'].map(dict(zip(tickers, earnings_surprises)))

# --------------------------------------------
# 4) Handle Missing Values & Scale
# --------------------------------------------
df_filled = df.ffill().bfill()
df_filled = df_filled.infer_objects(copy=False).interpolate(method='linear')

scaler = StandardScaler()
numeric_cols = df_filled.select_dtypes(include=['float64', 'int64']).columns
scaled_features = scaler.fit_transform(df_filled[numeric_cols])
df_scaled = pd.DataFrame(scaled_features, index=df_filled.index, columns=numeric_cols)

df_final = pd.concat([
    df_filled.select_dtypes(exclude=['float64', 'int64']),
    df_scaled
], axis=1)

# --------------------------------------------
# 5) Save & Check Final Data
# --------------------------------------------
df_final.to_parquet('enriched_stock_data_with_sentiment.parquet')

print("Final Enriched Data Head:")
print(df_final.head())

# --------------------------------------------
# 6) Visualization Examples
# --------------------------------------------
# Let's create a few quick plots using df_filled (unscaled)
# so it's more recognizable in line plots.

## (A) Time-Series Plot of Close Price (and maybe RSI) by Ticker

# We'll loop over each Ticker and plot its Close price & RSI in one figure
unique_tickers = df_filled['Ticker'].unique()

for ticker in unique_tickers:
    subdf = df_filled[df_filled['Ticker'] == ticker]

    # Create a 1x2 subplot: left for Close, right for RSI
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 4), sharex=True)

    # Plot Close
    axes[0].plot(subdf.index, subdf['Close'], label='Close Price', color='blue')
    axes[0].set_title(f"{ticker} - Close Price")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Price")

    # Plot RSI
    axes[1].plot(subdf.index, subdf['RSI'], label='RSI', color='red')
    axes[1].set_title(f"{ticker} - RSI")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("RSI (0-100)")

    plt.tight_layout()
    plt.show()

## (B) Distribution Plots for Key Numeric Columns
# Let's pick some columns: Close, OBV, ATR, MACD, etc.

numeric_columns_to_plot = ['Close', 'OBV', 'ATR', 'MACD', 'RSI']
for col in numeric_columns_to_plot:
    plt.figure(figsize=(6, 4))
    sns.histplot(df_filled[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

## (C) Correlation Heatmap of Numeric Features in df_filled
numeric_cols_for_corr = df_filled.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df_filled[numeric_cols_for_corr].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap (Unscaled Data)")
plt.show()
