import matplotlib.pyplot as plt

# Calculate 60-minute rolling averages
rolling_df = clean_df.set_index('timestamp').rolling('60T').mean().reset_index()

# Plot MQ3_Top_Analog
plt.figure(figsize=(12, 5))
plt.plot(clean_df['timestamp'], clean_df['MQ3_Top_Analog'], alpha=0.4, label='MQ3_Top_Analog (Raw)', linewidth=1)
plt.plot(rolling_df['timestamp'], rolling_df['MQ3_Top_Analog'], label='MQ3_Top_Analog (Rolling Avg)', linewidth=2)
plt.title('Time Series of MQ3_Top_Analog')
plt.xlabel('Timestamp')
plt.ylabel('Analog Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot MQ3_Bottom_Analog
plt.figure(figsize=(12, 5))
plt.plot(clean_df['timestamp'], clean_df['MQ3_Bottom_Analog'], alpha=0.4, label='MQ3_Bottom_Analog (Raw)', linewidth=1)
plt.plot(rolling_df['timestamp'], rolling_df['MQ3_Bottom_Analog'], label='MQ3_Bottom_Analog (Rolling Avg)', linewidth=2)
plt.title('Time Series of MQ3_Bottom_Analog')
plt.xlabel('Timestamp')
plt.ylabel('Analog Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Combined Top vs Bottom
plt.figure(figsize=(12, 5))
plt.plot(rolling_df['timestamp'], rolling_df['MQ3_Top_Analog'], label='Top Analog (Rolling Avg)', linewidth=2)
plt.plot(rolling_df['timestamp'], rolling_df['MQ3_Bottom_Analog'], label='Bottom Analog (Rolling Avg)', linewidth=2)
plt.title('Comparison of MQ3 Top vs Bottom Analog Readings')
plt.xlabel('Timestamp')
plt.ylabel('Analog Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
