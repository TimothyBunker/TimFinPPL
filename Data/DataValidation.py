import pandas as pd
from TFinEnv import *


df = pd.read_parquet('../enriched_stock_data_with_sentiment_training.parquet')
print("Loaded DataFrame shape:", df.shape)
print(df.head())  # Inspect if the data matches expected format
# print(f"Number of NaNs before passing to environment: {df.isna().sum().sum()}")
#
# env = CustomTradingEnv(data=df, initial_balance=1000., lookback_window=50)
# # Check after environment reset
# observation, edge_index = env.reset()
# print(f"Number of NaNs in observation after reset: {np.isnan(observation).sum()}")