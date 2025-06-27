
"""
E-Nose Spoilage Prediction using XGBoost (Reproduced Setup)

This script:
- Loads Batch 1 (training) and Batch 2 (testing) datasets
- Processes them into 3-minute windowed features
- Extracts statistical and delta-based features
- Labels each window as Fresh, Spoiling, or Late_Spoilage
- Trains an XGBoost Classifier on Batch 1
- Predicts and evaluates on Batch 2
- Saves processed features and prediction outputs for inspection
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
# --------------------------- CONFIGURATION ---------------------------
# Input data files (ensure these paths are correct)
batch1_path = "batch_one_complete_data.csv"
batch2_path = "Fully_complete_batch_two_data.csv"

# Output directory for processed files and predictions
output_dir = "reproduced_model_output"
os.makedirs(output_dir, exist_ok=True)

# --------------------------- FUNCTION: PREPROCESS DATA ---------------------------
def preprocess_data(df, batch_name):
    """
    Process sensor data into 3-minute windowed features.
    Adds statistical metrics, delta changes, and ratio features.
    Labels spoilage stage based on time-to-spoilage mean.
    Saves the features to CSV.

    Parameters:
        df (DataFrame): Raw input data
        batch_name (str): Used in naming the saved feature CSV

    Returns:
        DataFrame: Feature-engineered dataset
    """
    # Drop rows with missing sensor readings
    df = df.dropna(subset=[
        "MQ3_Top_PPM", "MQ3_Bottom_PPM", "BME_VOC_Ohm", "BME_Temp", "BME_Humidity"
    ]).copy()

    # Convert timestamp to datetime and sort
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Extract features in 3-minute (non-overlapping) windows
    window_size = 3
    features = []

    for i in range(0, len(df), window_size):
        window = df.iloc[i:i+window_size]
        if len(window) < window_size:
            continue  # Skip incomplete windows

        # Extract key statistical and dynamic features
        f = {
            "mean_BME_Temp": window["BME_Temp"].mean(),
            "std_BME_Temp": window["BME_Temp"].std(),
            "mean_BME_Humidity": window["BME_Humidity"].mean(),
            "std_BME_Humidity": window["BME_Humidity"].std(),
            "mean_BME_VOC_Ohm": window["BME_VOC_Ohm"].mean(),
            "std_BME_VOC_Ohm": window["BME_VOC_Ohm"].std(),
            "mean_MQ3_Top_PPM": window["MQ3_Top_PPM"].mean(),
            "std_MQ3_Top_PPM": window["MQ3_Top_PPM"].std(),
            "mean_MQ3_Bottom_PPM": window["MQ3_Bottom_PPM"].mean(),
            "std_MQ3_Bottom_PPM": window["MQ3_Bottom_PPM"].std(),
            "VOC_to_MQ3_Top_Ratio": window["BME_VOC_Ohm"].mean() / (window["MQ3_Top_PPM"].mean() + 1e-5),
            "VOC_to_MQ3_Bottom_Ratio": window["BME_VOC_Ohm"].mean() / (window["MQ3_Bottom_PPM"].mean() + 1e-5),
            "delta_VOC_Ohm": window["BME_VOC_Ohm"].iloc[-1] - window["BME_VOC_Ohm"].iloc[0],
            "delta_MQ3_Top_PPM": window["MQ3_Top_PPM"].iloc[-1] - window["MQ3_Top_PPM"].iloc[0],
            "delta_MQ3_Bottom_PPM": window["MQ3_Bottom_PPM"].iloc[-1] - window["MQ3_Bottom_PPM"].iloc[0],
            "Time_to_Spoilage_Minutes_mean": window["Time_to_Spoilage_Minutes"].mean()
        }

        # Label spoilage stage
        if f["Time_to_Spoilage_Minutes_mean"] >= 2400:
            f["Spoilage_Stage"] = "Fresh"
        elif f["Time_to_Spoilage_Minutes_mean"] >= 600:
            f["Spoilage_Stage"] = "Spoiling"
        else:
            f["Spoilage_Stage"] = "Late_Spoilage"

        features.append(f)

    # Convert to DataFrame and save
    df_features = pd.DataFrame(features)
    df_features.to_csv(os.path.join(output_dir, f"{batch_name}_features.csv"), index=False)
    return df_features

# --------------------------- LOAD & PROCESS DATA ---------------------------
# Load raw batch files
df1 = pd.read_csv(batch1_path)
df2 = pd.read_csv(batch2_path)

# Preprocess and extract features
df_train = preprocess_data(df1, "batch1")
df_test = preprocess_data(df2, "batch2")

# Select input features (exclude labels)
feature_cols = [col for col in df_train.columns if col not in ["Spoilage_Stage", "Time_to_Spoilage_Minutes_mean"]]
X_train = df_train[feature_cols]
y_train = df_train["Spoilage_Stage"]

X_test = df_test[feature_cols]
y_test = df_test["Spoilage_Stage"]

# Encode class labels as integers for XGBoost
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

# --------------------------- TRAINING ---------------------------
# Initialize and train the XGBoost Classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train_enc)

# --------------------------- PREDICTION ---------------------------
# Predict on the test batch
y_pred_enc = model.predict(X_test)
y_pred = encoder.inverse_transform(y_pred_enc)

# Save predictions alongside original test data
df_test_out = df_test.copy()
df_test_out["Predicted_Stage"] = y_pred
df_test_out.to_csv(os.path.join(output_dir, "batch2_predictions.csv"), index=False)

# --------------------------- EVALUATION ---------------------------
# Print evaluation report
print(" Classification Report:")
print(classification_report(y_test, y_pred))
print(f" Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f" Outputs saved in folder: {output_dir}")


# --------------------------- ROC CURVE (Multiclass) ---------------------------
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

# Binarize the output for ROC curve computation
y_test_bin = label_binarize(y_test_enc, classes=[0, 1, 2])
y_score = model.predict_proba(X_test)

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_bin.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
colors = cycle(["blue", "orange", "green"])
class_names = encoder.inverse_transform([0, 1, 2])

plt.figure(figsize=(8, 6))
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"ROC curve (class {class_names[i]}) - AUC = {roc_auc[i]:0.2f}")

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "multiclass_roc_curve.png"))
plt.close()


# --------------------------- CONFUSION MATRIX ---------------------------


# Generate confusion matrix
labels = ["Fresh", "Spoiling", "Late_Spoilage"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

# Plot and save confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()


# --------------------------- SAVE TRAINED MODEL ---------------------------


# Save the trained XGBoost model to a file
model_path = os.path.join(output_dir, "xgboost_spoilage_model.pkl")
joblib.dump(model, model_path)

print(f" Trained model saved to: {model_path}")
# Save the trained LabelEncoder
encoder_path = os.path.join(output_dir, "label_encoder.joblib")
joblib.dump(encoder, encoder_path)

print(f" LabelEncoder saved to: {encoder_path}")
