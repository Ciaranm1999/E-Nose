import pandas as pd

def engineer_features(df, diff_window=1):
    rate_cols = [
        'BME_Temp',
        'BME_Humidity',
        'BME_VOC_Ohm',
        'MQ3_Bottom_Analog',
        'MQ3_Bottom_PPM',
        'MQ3_Top_Analog',
        'MQ3_Top_PPM'
    ]
    for col in rate_cols:
        df[f'{col}_diff'] = df[col].diff(periods=diff_window).fillna(0)
    if 'time_seconds' not in df.columns:
        df['time_seconds'] = pd.to_numeric(df['time'], errors='coerce') - pd.to_numeric(df['time'].iloc[0], errors='coerce')
    return df