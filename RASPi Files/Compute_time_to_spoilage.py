import pandas as pd
import json
from datetime import datetime

# === CONFIGURATION ===
input_file = "raw_data4.csv"
output_file = "structured_with_spoilage_time.xlsx"
time_format = "%Y-%m-%d %H:%M:%S"  # Update this if your timestamp format differs

# === READ CSV ===
df = pd.read_csv(input_file, header=None, names=["Timestamp", "JSON_Data"])

# === CLEAN AND PARSE JSON ===
structured_data = []

for index, row in df.iterrows():
    try:
        timestamp = row["Timestamp"]
        sensor_json = json.loads(row["JSON_Data"])
        
        structured_data.append({
            "Timestamp": timestamp,
            "time": sensor_json.get("time"),
            "BME_Temp": sensor_json.get("BME_Temp"),
            "BME_Humidity": sensor_json.get("BME_Humidity"),
            "BME_VOC_Ohm": sensor_json.get("BME_VOC_Ohm"),
            "MQ3_Bottom_Analog": sensor_json.get("MQ3_Bottom_Analog"),
            "MQ3_Bottom_PPM": sensor_json.get("MQ3_Bottom_PPM"),
            "MQ3_Top_Analog": sensor_json.get("MQ3_Top_Analog"),
            "MQ3_Top_PPM": sensor_json.get("MQ3_Top_PPM"),
        })
    except Exception as e:
        print(f"Row {index} ❌ Error: {e}")

# === CONVERT TO DATAFRAME ===
df_structured = pd.DataFrame(structured_data)

# === CONVERT TIMESTAMP TO DATETIME ===
df_structured["Timestamp"] = pd.to_datetime(df_structured["Timestamp"], errors="coerce")

# === CALCULATE TIME TO SPOILAGE ===
last_time = df_structured["Timestamp"].max()
df_structured["Time_to_Spoilage_Minutes"] = (last_time - df_structured["Timestamp"]).dt.total_seconds() / 60

# === EXPORT TO EXCEL ===
df_structured.to_excel(output_file, index=False)

print(f"✅ Saved to {output_file}")
