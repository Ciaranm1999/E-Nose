import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('Data/batch_one/batch_one_data_converted.csv')

# Columns to exclude from moving average/trend calculation
exclude_cols = ['Timestamp', 'time', 'Time_to_Spoilage_Minutes', 'Time_from_Start_Minutes' ]

# Cap PPM values at 300 (set anything above 300 to NaN)
for col in ['MQ3_Bottom_PPM', 'MQ3_Top_PPM']:
    df.loc[df[col] > 300, col] = np.nan

# Columns to apply moving average/trend calculation
columns_to_trend = [col for col in df.columns if col not in exclude_cols]

# Apply moving average with window size 150 (centered), overwrite original columns
window_size = 360
for col in columns_to_trend:
    temp_series = df[col].fillna(method='ffill').fillna(method='bfill')
    df[col] = temp_series.rolling(window=window_size, center=True, min_periods=1).mean()

# Save to new CSV (overwriting the original values with trended values)
df.to_csv('Data/batch_one/batch_one_complete.csv', index=False)

print("Original data overwritten with trended values (excluding Timestamp, time, Time_to_Spoilage_Minutes). Saved to Data\\batch_one_filtered.csv")