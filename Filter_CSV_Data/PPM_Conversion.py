import pandas as pd
import numpy as np

# --- Parameters ---
INPUT_CSV = 'Data/batch_two/filtered_data.csv'
OUTPUT_CSV = 'Data/batch_two/complete_data.csv'

# Set your calibration constants here
R0_MQ3_Bottom = 6.0
R0_MQ3_Top = 5.0
RL_VALUE = 10.0  # kOhms
Vc = 5.0         # Supply voltage

def analog_to_ppm(analog_value, r0, RL_VALUE=10.0, Vc=5.0):
    analog_value = np.maximum(analog_value, 1)
    sensor_voltage = analog_value * (Vc / 4095.0)
    sensor_voltage = np.maximum(sensor_voltage, 0.1)
    rs = ((Vc - sensor_voltage) / sensor_voltage) * RL_VALUE
    ratio = rs / r0
    ppm = 607.9 * np.power(ratio, -2.868)
    return ppm

# --- Load data ---
df = pd.read_csv(INPUT_CSV)

# --- Convert analog to PPM and overwrite columns ---
df['MQ3_Bottom_PPM'] = analog_to_ppm(df['MQ3_Bottom_Analog'], R0_MQ3_Bottom, RL_VALUE, Vc)
df['MQ3_Top_PPM'] = analog_to_ppm(df['MQ3_Top_Analog'], R0_MQ3_Top, RL_VALUE, Vc)

# --- Save to new CSV ---
df.to_csv(OUTPUT_CSV, index=False)
print(f"PPM conversion complete. Output saved to: {OUTPUT_CSV}")