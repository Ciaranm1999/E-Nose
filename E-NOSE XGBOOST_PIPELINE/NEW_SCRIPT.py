import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from itertools import cycle

# --------------------------- CONFIGURATION ---------------------------
batch1_path = "batch_one_complete_data.csv"
batch2_path = "Fully_complete_batch_two_data.csv"
output_dir = "reproduced_model_output_balanced"
os.makedirs(output_dir, exist_ok=True)

# --------------------------- FUNCTION: PREPROCESS DATA ---------------------------
def preprocess_data(df, batch_name):
    df = df.dropna(subset=[
        "MQ3_Top_PPM", "MQ3_Bottom_PPM", "BME_VOC_Ohm", "BME_Temp", "BME_Humidity"
    ]).copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    window_size = 3
    features = []
    for i in range(0, len(df), window_size):
        window = df.iloc[i:i+window_size]
        if len(window) < window_size:
            continue
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
        if f["Time_to_Spoilage_Minutes_mean"] >= 2400:
            f["Spoilage_Stage"] = "Fresh"
        elif f["Time_to_Spoilage_Minutes_mean"] >= 600:
            f["Spoilage_Stage"] = "Spoiling"
        else:
            f["Spoilage_Stage"] = "Late_Spoilage"
        features.append(f)

    df_features = pd.DataFrame(features)
    df_features.to_csv(os.path.join(output_dir, f"{batch_name}_features.csv"), index=False)
    return df_features

# --------------------------- LOAD & PROCESS DATA ---------------------------
df1 = pd.read_csv(batch1_path)
df2 = pd.read_csv(batch2_path)

df_train = preprocess_data(df1, "batch1")
df_test = preprocess_data(df2, "batch2")

feature_cols = [col for col in df_train.columns if col not in ["Spoilage_Stage", "Time_to_Spoilage_Minutes_mean"]]
X_train = df_train[feature_cols]
y_train = df_train["Spoilage_Stage"]
X_test = df_test[feature_cols]
y_test = df_test["Spoilage_Stage"]

# --------------------------- ENCODING & WEIGHTING ---------------------------
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.transform(y_test)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_enc), y=y_train_enc)
sample_weights = np.array([class_weights[label] for label in y_train_enc])

# --------------------------- TRAINING ---------------------------
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train_enc, sample_weight=sample_weights)

# --------------------------- PREDICTION ---------------------------
y_pred_enc = model.predict(X_test)
y_pred = encoder.inverse_transform(y_pred_enc)
df_test_out = df_test.copy()
df_test_out["Predicted_Stage"] = y_pred
df_test_out.to_csv(os.path.join(output_dir, "batch2_predictions.csv"), index=False)

# --------------------------- EVALUATION ---------------------------
report = classification_report(y_test, y_pred, output_dict=True)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=encoder.classes_)

# Save classification report
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(output_dir, "classification_report.csv"))

# Save confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# --------------------------- ROC CURVE ---------------------------
y_test_bin = label_binarize(y_test_enc, classes=[0, 1, 2])
y_score = model.predict_proba(X_test)

fpr, tpr, roc_auc = {}, {}, {}
n_classes = y_test_bin.shape[1]
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
colors = cycle(["blue", "orange", "green"])
class_names = encoder.inverse_transform([0, 1, 2])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"ROC curve (class {class_names[i]}) - AUC = {roc_auc[i]:0.2f}")
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

# --------------------------- SAVE MODEL ---------------------------
joblib.dump(model, os.path.join(output_dir, "xgboost_spoilage_model.pkl"))
joblib.dump(encoder, os.path.join(output_dir, "label_encoder.joblib"))

# Generate and print classification report as text
classification_text = classification_report(y_test, y_pred, digits=3)
accuracy_value = accuracy_score(y_test, y_pred)

# Print to console
print(" Classification Report (Balanced Model):\n")
print(classification_text)
print(f" Accuracy: {accuracy_value:.4f}")

