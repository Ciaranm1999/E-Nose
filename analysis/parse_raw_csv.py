import pandas as pd
import json
import re

# Load the raw CSV file
df = pd.read_csv('/mnt/data/raw_data2.csv')

# Extract timestamp and sensor JSON string
df[['timestamp', 'sensor_data']] = df.iloc[:, 0].str.extract(r'([^,]+),"(.*)"')

# Clean JSON string: fix escaped quotes
df['sensor_data'] = df['sensor_data'].str.replace('""', '"').str.strip('"')
df['sensor_data'] = df['sensor_data'].apply(lambda x: re.sub(r'\\', '', x))

# Parse JSON strings safely, skipping malformed ones
valid_entries = []
timestamps = []
for idx, row in df.iterrows():
    try:
        parsed_json = json.loads(row['sensor_data'])
        valid_entries.append(parsed_json)
        timestamps.append(row['timestamp'])
    except json.JSONDecodeError:
        continue  # Skip corrupted rows

# Convert to structured DataFrame
clean_df = pd.DataFrame(valid_entries)
clean_df['timestamp'] = pd.to_datetime(timestamps)
clean_df = clean_df[['timestamp'] + [col for col in clean_df.columns if col != 'timestamp']]
