import pandas as pd
import numpy as np

df = pd.read_csv("D:\\Most Recent\\Talias_processed_running_data_summary.csv")

# Extract the average pace values
flat_pace = df[df['terrain_type'] == 'flat']['avg_pace_min_per_km'].iloc[0]
uphill_pace = df[df['terrain_type'] == 'uphill']['avg_pace_min_per_km'].iloc[0]
downhill_pace = df[df['terrain_type'] == 'downhill']['avg_pace_min_per_km'].iloc[0]

print("Terrain Analysis Data:")
print(f"Flat average pace: {flat_pace} min/km")
print(f"Uphill average pace: {uphill_pace} min/km")
print(f"Downhill average pace: {downhill_pace} min/km")
print()

# Calculate uphill offset: ((uphill average pace - flat average pace) / flat average pace) * 100
uphill_offset = ((uphill_pace - flat_pace) / flat_pace) * 100

# Calculate downhill offset: ((downhill average pace - flat average pace) / flat average pace) * 100
downhill_offset = ((downhill_pace - flat_pace) / flat_pace) * 100

print("Calculated Offsets:")
print(f"Uphill offset: {uphill_offset:.2f}%")
print(f"Downhill offset: {downhill_offset:.2f}%")
print()

# Additional insights
# Create a DataFrame with the calculated offsets
results_df = pd.DataFrame({
    'terrain_type': ['uphill', 'downhill'],
    'offset_percentage': [uphill_offset, downhill_offset]
})

# Save the results to a CSV file
results_df.to_csv('Talia_terrain_offsets.csv', index=False)

print("Results saved to 'terrain_offsets.csv'")
print("\nFile contents:")
print(results_df)

print("\nInterpretation:")
if uphill_offset > 0:
    print(f"Running uphill is {uphill_offset:.2f}% slower than running on flat terrain")
else:
    print(f"Running uphill is {abs(uphill_offset):.2f}% faster than running on flat terrain")

if downhill_offset < 0:
    print(f"Running downhill is {abs(downhill_offset):.2f}% faster than running on flat terrain")
else:
    print(f"Running downhill is {downhill_offset:.2f}% slower than running on flat terrain")