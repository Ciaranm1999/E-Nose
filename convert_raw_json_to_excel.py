import pandas as pd
import json

input_csv = "raw_data2.csv"
output_excel = "structured_data.xlsx"
error_log = "json_errors.txt"

# Store processed rows
structured_rows = []
error_lines = []

# Read the CSV
df = pd.read_csv(input_csv, header=None, names=["Timestamp", "RawJSON"])

# Loop through rows
for idx, row in df.iterrows():
    try:
        timestamp = row["Timestamp"]
        json_data = json.loads(row["RawJSON"])  # Try to parse
        flat = {
            "Timestamp": timestamp,
            "time": json_data.get("time"),
            "BME_Temp": json_data.get("BME_Temp"),
            "BME_Humidity": json_data.get("BME_Humidity"),
            "BME_VOC_Ohm": json_data.get("BME_VOC_Ohm"),
            "MQ3_Bottom_Analog": json_data.get("MQ3_Bottom_Analog"),
            "MQ3_Bottom_PPM": json_data.get("MQ3_Bottom_PPM"),
            "MQ3_Top_Analog": json_data.get("MQ3_Top_Analog"),
            "MQ3_Top_PPM": json_data.get("MQ3_Top_PPM")
        }
        structured_rows.append(flat)
    except Exception as e:
        error_lines.append(f"Row {idx + 1} ERROR: {str(e)}")

# Save to Excel
pd.DataFrame(structured_rows).to_excel(output_excel, index=False)


with open(error_log, "w", encoding="utf-8") as f:
    f.write("\n".join(error_lines))

print(f"Conversion complete. {len(structured_rows)} rows written to Excel.")
print(f" {len(error_lines)} rows skipped due to JSON errors. See 'json_errors.txt'")
