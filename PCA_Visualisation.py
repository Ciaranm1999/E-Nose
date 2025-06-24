import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# --- User Settings ---
DATA_PATH = 'Data Processing/Data/batch_two/complete_data.csv'  # Change to batch_one if needed
truncate_at_nan = True   # True to truncate at first NaN in MQ3 columns, False to use all data
use_diff_features = True # True to include diff features, False for only original features
lags = [0, 60, 120, 180, 240, 300, 360]  # 0 means no diff features; others are lag in minutes/rows

mq3_cols = ['MQ3_Top_PPM', 'MQ3_Bottom_PPM', 'MQ3_Top_Analog', 'MQ3_Bottom_Analog', 'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm']
# --- Load Data ---
df_orig = pd.read_csv(DATA_PATH, parse_dates=['Timestamp'])

results = []

# --- Grid Search over feature combinations and lags ---
for lag in lags if use_diff_features else [0]:
    for r in range(1, len(mq3_cols) + 1):
        for feature_combo in combinations(mq3_cols, r):
            df = df_orig.copy()
            # Truncate at first NaN in MQ3 data if enabled
            if truncate_at_nan:
                nan_mask = df[list(feature_combo)].isnull().any(axis=1)
                if nan_mask.any():
                    first_nan_idx = nan_mask.idxmax()
                    df = df.loc[:first_nan_idx - 1].reset_index(drop=True)
            feature_list = list(feature_combo)
            # Add diff features if enabled and lag > 0
            if use_diff_features and lag > 0:
                for col in feature_combo:
                    df[f'{col}_diff'] = df[col] - df[col].shift(lag)
                    df[f'{col}_diff'] = df[f'{col}_diff'].fillna(0)
                feature_list += [f'{col}_diff' for col in feature_combo]
            X = df[feature_list].values
            # Skip if not enough data or features
            if len(df) < 10 or X.shape[1] == 0:
                continue
            # Normalise
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # Clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            # Metrics
            sil = silhouette_score(X_scaled, clusters)
            db = davies_bouldin_score(X_scaled, clusters)
            results.append({
                'lag': lag,
                'features': feature_combo,
                'silhouette': sil,
                'davies_bouldin': db,
                'feature_list': feature_list
            })

# --- Find the best by silhouette score ---
best = max(results, key=lambda x: x['silhouette'])
print("\nBest combination:")
print(f"Lag: {best['lag']}")
print(f"Features: {best['features']}")
print(f"Silhouette Score: {best['silhouette']:.3f}")
print(f"Davies-Bouldin Index: {best['davies_bouldin']:.3f}")

# --- Visualize the best combination ---
df = df_orig.copy()
if truncate_at_nan:
    nan_mask = df[list(best['features'])].isnull().any(axis=1)
    if nan_mask.any():
        first_nan_idx = nan_mask.idxmax()
        df = df.loc[:first_nan_idx - 1].reset_index(drop=True)
feature_list = list(best['features'])
if use_diff_features and best['lag'] > 0:
    for col in best['features']:
        df[f'{col}_diff'] = df[col] - df[col].shift(best['lag'])
        df[f'{col}_diff'] = df[f'{col}_diff'].fillna(0)
    feature_list += [f'{col}_diff' for col in best['features']]
X = df[feature_list].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
# Sort clusters by mean of first feature for coloring
cluster_order = df.groupby('Cluster')[feature_list[0]].mean().sort_values().index
cluster_labels = {cluster: label for cluster, label in zip(cluster_order, ['fresh', 'spoiling', 'spoiled'])}
df['Cluster_Label'] = df['Cluster'].map(cluster_labels)
colors = {'fresh': 'green', 'spoiling': 'orange', 'spoiled': 'red'}

if len(feature_list) >= 2:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]

    plt.figure(figsize=(8, 6))
    for label, color in colors.items():
        mask = df['Cluster_Label'] == label
        plt.scatter(df.loc[mask, 'PCA1'], df.loc[mask, 'PCA2'], color=color, label=f'Cluster: {label}', alpha=0.7, s=30)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f"Best PCA Clustering\nFeatures: {best['features']}, Lag: {best['lag']}, Truncate: {truncate_at_nan}, Diff: {use_diff_features}")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    # 1D plot for single feature
    plt.figure(figsize=(8, 4))
    for label, color in colors.items():
        mask = df['Cluster_Label'] == label
        plt.scatter(df.loc[mask, 'Timestamp'], df.loc[mask, feature_list[0]], color=color, label=f'Cluster: {label}', alpha=0.7, s=30)
    plt.xlabel('Timestamp')
    plt.ylabel(feature_list[0])
    plt.title(f"Best 1D Clustering\nFeature: {feature_list[0]}, Lag: {best['lag']}, Truncate: {truncate_at_nan}, Diff: {use_diff_features}")
    plt.legend()
    plt.tight_layout()
    plt.show()