from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

# Load your data
df = pd.read_csv('Data/batch_one_filtered.csv')

# 1. Run K-Means
X = df[['MQ3_Bottom_Analog']].values
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# 2. Sort cluster centers and get thresholds
centers = np.sort(kmeans.cluster_centers_.flatten())
thresholds = [(centers[i] + centers[i+1]) / 2 for i in range(len(centers)-1)]

# 3. Label data using K-Means thresholds
def spoilage_label_kmeans(row):
    if row['MQ3_Bottom_Analog'] < thresholds[0]:
        return 'fresh'
    elif row['MQ3_Bottom_Analog'] < thresholds[1]:
        return 'spoiling'
    else:
        return 'spoiled'

df['spoilage_class'] = df.apply(spoilage_label_kmeans, axis=1)

# 4. Now you can use df['spoilage_class'] as your label for Random Forest