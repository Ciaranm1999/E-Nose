import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from Feature_Extraction import engineer_features

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Load the filtered data
df = pd.read_csv('Data/batch_one_filtered.csv')

# Feature engineering
df = engineer_features(df, diff_window=5)

# --- K-Means to determine thresholds ---
X = df[['MQ3_Bottom_Analog']].values
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Sort cluster centers and get thresholds
centers = np.sort(kmeans.cluster_centers_.flatten())
thresholds = [(centers[i] + centers[i+1]) / 2 for i in range(len(centers)-1)]
print("K-Means thresholds (ethanol analog):", thresholds)

# Label data using K-Means thresholds
def spoilage_label_kmeans(row):
    if row['MQ3_Bottom_Analog'] < thresholds[0]:
        return 'fresh'
    elif row['MQ3_Bottom_Analog'] < thresholds[1]:
        return 'spoiling'
    else:
        return 'spoiled'

df['spoilage_class'] = df.apply(spoilage_label_kmeans, axis=1)

print("Class distribution (K-Means labels):\n", df['spoilage_class'].value_counts())

# --- Random Forest Training ---
feature_cols = [
    'BME_Temp', 'BME_Humidity', 'BME_VOC_Ohm',
    'MQ3_Bottom_Analog', 'MQ3_Bottom_PPM', 'MQ3_Top_Analog', 'MQ3_Top_PPM',
    'BME_Temp_diff', 'BME_Humidity_diff', 'BME_VOC_Ohm_diff',
    'MQ3_Bottom_Analog_diff', 'MQ3_Bottom_PPM_diff', 'MQ3_Top_Analog_diff', 'MQ3_Top_PPM_diff',
    'time_seconds'
]
features = df[feature_cols]
target_clf = df['spoilage_class']

X_train, X_test, y_train, y_test = train_test_split(
    features, target_clf, test_size=0.2, random_state=42, stratify=target_clf
)

clf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf_model.fit(X_train, y_train)
clf_predictions = clf_model.predict(X_test)
print(classification_report(y_test, clf_predictions))

# Save model
joblib.dump(clf_model, 'models/spoilage_classifier_kmeans.pkl')

# --- Predict on ALL samples ---
df['predicted_class'] = clf_model.predict(df[feature_cols])

# --- Plotting ---
plt.figure(figsize=(14, 6))
colors = {'fresh': 'green', 'spoiling': 'orange', 'spoiled': 'red'}

# Plot the raw ethanol analog signal as a line (sorted by Timestamp)
df_sorted = df.sort_values('Timestamp')
plt.plot(df_sorted['Timestamp'], df_sorted['MQ3_Bottom_Analog'], color='black', linewidth=1, alpha=0.5, label='Raw Ethanol Analog')

# Overlay: scatter plot colored by predicted class
for spoilage in ['fresh', 'spoiling', 'spoiled']:
    mask = df_sorted['predicted_class'] == spoilage
    plt.scatter(df_sorted.loc[mask, 'Timestamp'], df_sorted.loc[mask, 'MQ3_Bottom_Analog'],
                label=f"Predicted: {spoilage}", color=colors[spoilage], s=15, alpha=0.8)

plt.xlabel('Timestamp')
plt.ylabel('Ethanol (MQ3_Bottom_Analog)')
plt.title('Ethanol Analog vs Time: Random Forest Prediction (K-Means Threshold Labels)')
plt.legend()
#plt.xlim()
#plt.ylim()
plt.tight_layout()
plt.show()