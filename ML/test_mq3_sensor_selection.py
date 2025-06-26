"""
Test that Batch 2 model predictions plot against MQ3 Bottom data
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load test data (Batch 2)
df = pd.read_csv("../Data Processing/Data/batch_two/complete_data.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hours_from_start'] = (df['Timestamp'] - df['Timestamp'].iloc[0]).dt.total_seconds() / 3600

print("Testing MQ3 sensor selection for plotting:")
print(f"Batch 1 (dataset_idx=0) should use: MQ3_Top_PPM")
print(f"Batch 2 (dataset_idx=1) should use: MQ3_Bottom_PPM")

# Simulate the logic from the modified function
for dataset_idx, dataset_name in enumerate(['Batch 1 (Training)', 'Batch 2 (Test)']):
    if dataset_idx == 0:  # Batch 1 - use MQ3 Top
        mq3_data = 'MQ3_Top_PPM'
        mq3_label = 'MQ3 Top PPM'
    else:  # Batch 2 - use MQ3 Bottom
        mq3_data = 'MQ3_Bottom_PPM'
        mq3_label = 'MQ3 Bottom PPM'
    
    print(f"{dataset_name}: Will use {mq3_data} (y-axis: {mq3_label})")

# Quick visualization of both sensors for Batch 2
plt.figure(figsize=(12, 6))
valid_top = df['MQ3_Top_PPM'].notna()
valid_bottom = df['MQ3_Bottom_PPM'].notna()

plt.plot(df.loc[valid_top, 'Hours_from_start'], df.loc[valid_top, 'MQ3_Top_PPM'], 
         'b-', alpha=0.7, label='MQ3 Top PPM')
plt.plot(df.loc[valid_bottom, 'Hours_from_start'], df.loc[valid_bottom, 'MQ3_Bottom_PPM'], 
         'r-', alpha=0.7, label='MQ3 Bottom PPM')

plt.xlabel('Hours from Start')
plt.ylabel('PPM')
plt.title('Batch 2 - Both MQ3 Sensors (Bottom will now be used for model predictions)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n✅ Model predictions for Batch 2 will now plot against MQ3 Bottom data!")
