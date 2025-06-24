import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load your data
df = pd.read_csv('Data Processing/Data/batch_one/complete_data.csv', parse_dates=['Timestamp'])

# --- Toggle for truncating at first NaN in MQ3 data ---
truncate_at_nan = True  # Set to False to use all data

mq3_cols = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'MQ3_Top_Analog', 'MQ3_Bottom_Analog']

if truncate_at_nan:
    # Find the first index where any MQ3 column is NaN
    nan_mask = df[mq3_cols].isnull().any(axis=1)
    if nan_mask.any():
        first_nan_idx = nan_mask.idxmax()
        df = df.loc[:first_nan_idx - 1].reset_index(drop=True)

lag = 180  # Change this to your preferred lag

for col in mq3_cols:
    df[f'{col}_diff'] = df[col] - df[col].shift(lag)
    df[f'{col}_diff'] = df[f'{col}_diff'].fillna(0)

features = mq3_cols  # or add diff columns if you want
X = df[features].values

# --- Normalise features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Unsupervised Clustering ---
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Sort clusters by mean MQ3_Top_PPM for consistent coloring
cluster_order = df.groupby('Cluster')['MQ3_Bottom_PPM'].mean().sort_values().index
cluster_labels = {cluster: label for cluster, label in zip(cluster_order, ['fresh', 'spoiling', 'spoiled'])}
df['Cluster_Label'] = df['Cluster'].map(cluster_labels)

# --- Manual Thresholds (lines only, no manual points) ---
fresh_time = pd.Timestamp('2025-06-14 16:00')
spoiling_time = pd.Timestamp('2025-06-15 20:00')
definite_spoilage_time = pd.Timestamp('2025-06-16 10:00')

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Timestamp'], df['MQ3_Bottom_PPM'], 'b-', label='MQ3 Top PPM')

colors = {'fresh': 'green', 'spoiling': 'orange', 'spoiled': 'red'}
for label, color in colors.items():
    mask = df['Cluster_Label'] == label
    ax.scatter(df.loc[mask, 'Timestamp'], df.loc[mask, 'MQ3_Bottom_PPM'], color=color, marker='x', s=30, label=f'Cluster: {label}')

ax.axvline(fresh_time, color='green', linestyle='--', label='Fresh/Spoiling Threshold')
ax.axvline(spoiling_time, color='red', linestyle='--', label='Spoiling/Suspected Spoiled Threshold')
ax.axvline(definite_spoilage_time, color='black', linestyle='-', label='Definite Spoilage')

date_format = DateFormatter('%m-%d %H:%M')
ax.xaxis.set_major_formatter(date_format)
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
ax.set_xlabel('Timestamp')
ax.set_ylabel('MQ3 Top PPM')
ax.set_title('MQ3 Top PPM vs. Time: Unsupervised Clustering (All Sensors, Normalised) with Manual Thresholds')
ax.legend(loc='upper right', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()