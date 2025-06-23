import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('Data/batch_two/converted_data.csv')

# Convert Timestamp to datetime if not already
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Columns to exclude from moving average/trend calculation
exclude_cols = ['Timestamp', 'time', 'Time_to_Spoilage_Minutes', 'Time_from_Start_Minutes']

# Environmental filtering
df.loc[df['BME_VOC_Ohm'] < 2.5, 'BME_VOC_Ohm'] = np.nan
df.loc[df['BME_Temp'] < 24, 'BME_Temp'] = np.nan
df.loc[df['BME_Humidity'] < 70, 'BME_Humidity'] = np.nan

# Cap PPM values at 300 (set anything above 300 to NaN)
for col in ['MQ3_Bottom_PPM', 'MQ3_Top_PPM']:
    df.loc[df[col] > 300, col] = np.nan

# Columns to apply moving average/trend calculation
columns_to_trend = [col for col in df.columns if col not in exclude_cols]

# Apply moving average with window size 360 (centered), overwrite original columns
window_size = 360
for col in columns_to_trend:
    temp_series = df[col].fillna(method='ffill').fillna(method='bfill')
    df[col] = temp_series.rolling(window=window_size, center=True, min_periods=1).mean()

# After rolling average, set MQ3 data to NaN after cutoff
cutoff_time = pd.Timestamp('2025-06-20 05:40')
cols_to_clear = [
    'MQ3_Bottom_PPM', 'MQ3_Top_PPM',
    'MQ3_Bottom_Analog', 'MQ3_Top_Analog',
    'MQ3_Bottom_PPM_trend', 'MQ3_Top_PPM_trend',
    'MQ3_Bottom_Analog_trend', 'MQ3_Top_Analog_trend'
]
for col in cols_to_clear:
    if col in df.columns:
        df.loc[df['Timestamp'] > cutoff_time, col] = np.nan

# Save to new CSV (overwriting the original values with trended values)
df.to_csv('Data/batch_two/filtered_data.csv', index=False)

print("Original data overwritten with trended values (excluding MQ3 data after cutoff). Saved to Data\\batch_two\\filtered_data.csv")
