import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Simulate a 3-minute live test window
live_data = pd.DataFrame({
    "Timestamp": pd.date_range(start="2025-06-24 10:00:00", periods=3, freq="min"),
    "BME_Temp": [24.0, 23.8, 23.6],
    "BME_Humidity": [75.0, 76.2, 77.1],
    "BME_VOC_Ohm": [20000, 30000, 28000],
    "MQ3_Top_PPM": [20, 40, 60],
    "MQ3_Bottom_PPM": [18, 35, 55]
})

# Extract features from live data (mirroring preprocessing)
def extract_live_features(df):
    f = {
        "mean_BME_Temp": df["BME_Temp"].mean(),
        "std_BME_Temp": df["BME_Temp"].std(),
        "mean_BME_Humidity": df["BME_Humidity"].mean(),
        "std_BME_Humidity": df["BME_Humidity"].std(),
        "mean_BME_VOC_Ohm": df["BME_VOC_Ohm"].mean(),
        "std_BME_VOC_Ohm": df["BME_VOC_Ohm"].std(),
        "mean_MQ3_Top_PPM": df["MQ3_Top_PPM"].mean(),
        "std_MQ3_Top_PPM": df["MQ3_Top_PPM"].std(),
        "mean_MQ3_Bottom_PPM": df["MQ3_Bottom_PPM"].mean(),
        "std_MQ3_Bottom_PPM": df["MQ3_Bottom_PPM"].std(),
        "VOC_to_MQ3_Top_Ratio": df["BME_VOC_Ohm"].mean() / (df["MQ3_Top_PPM"].mean() + 1e-5),
        "VOC_to_MQ3_Bottom_Ratio": df["BME_VOC_Ohm"].mean() / (df["MQ3_Bottom_PPM"].mean() + 1e-5),
        "delta_VOC_Ohm": df["BME_VOC_Ohm"].iloc[-1] - df["BME_VOC_Ohm"].iloc[0],
        "delta_MQ3_Top_PPM": df["MQ3_Top_PPM"].iloc[-1] - df["MQ3_Top_PPM"].iloc[0],
        "delta_MQ3_Bottom_PPM": df["MQ3_Bottom_PPM"].iloc[-1] - df["MQ3_Bottom_PPM"].iloc[0]
    }
    return pd.DataFrame([f])

# --------------------------- PREDICTION & INTERPRETATION ---------------------------

# Extract features from live data
live_features = extract_live_features(live_data)

# Load trained model and label encoder
model = joblib.load("reproduced_model_output/xgboost_spoilage_model.pkl")
encoder = joblib.load("reproduced_model_output/label_encoder.joblib")

# Predict spoilage stage
live_pred_enc = model.predict(live_features)
live_pred_label = encoder.inverse_transform(live_pred_enc)[0]

# Generate and print interpretation
if live_pred_label == "Fresh":
    interpretation = "🟢 Sample is in fresh stage — expected time to spoilage: > 40 hours."
elif live_pred_label == "Spoiling":
    interpretation = "🟠 Sample is in spoiling stage — expected time to spoilage: 10–40 hours."
elif live_pred_label == "Late_Spoilage":
    interpretation = "🔴 Sample is in late spoilage — expected time to spoilage: < 10 hours."
else:
    interpretation = " Unable to interpret result — unrecognized prediction."

# Print the results
print(f"\n🔍 Predicted Spoilage Stage: {live_pred_label}")
print(f"{interpretation}")
