################################
### PROCESS
################################

import pandas as pd
from datetime import datetime

# File location
csv_file_path = "C:/my_csv.csv"

# Date format in the CSV
date_format = "%Y-%B-%d %H:%M:%S"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Function to calculate the time difference in seconds
def calculate_work_time(start_date, end_date):
    start_datetime = datetime.strptime(start_date, date_format)
    end_datetime = datetime.strptime(end_date, date_format)
    time_diff = end_datetime - start_datetime
    # Convert the difference to seconds
    return time_diff.total_seconds()

# Iterate over the rows to update MODEL_WORK_TIME where it's 0
for index, row in df.iterrows():
    if row['MODEL_WORK_TIME'] == 0:
        df.at[index, 'MODEL_WORK_TIME'] = calculate_work_time(row['EXPERIMENT_START_DATE'], row['EXPERIMENT_END_DATE'])

# Save the modified DataFrame to a new CSV file
new_csv_file_path = csv_file_path.replace(".csv", "_PROCESSED.csv")
df.to_csv(new_csv_file_path, index=False)

print(f"Processed CSV file saved as: {new_csv_file_path}")

######################################
### STATISTICS
######################################

import pandas as pd
import numpy as np

# File location
csv_file_path = "C:/my_csv.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Convert the 'CORRECTNESS' column from percentage string to float
df['CORRECTNESS'] = df['CORRECTNESS'].str.rstrip('%').astype('float') / 100.0

# Rename ROWS_PROCESSED to DATA_INTENSITY
df = df.rename(columns={'ROWS_PROCESSED': 'DATA_INTENSITY'})

# Define a function to determine TASK_DIFFICULTY based on DATA_INTENSITY
def determine_task_difficulty(data_intensity):
    if data_intensity == 5:
        return 1
    elif data_intensity == 10:
        return 2
    elif data_intensity == 20:
        return 3
    elif data_intensity == 50:
        return 4
    elif data_intensity == 100:
        return 5
    else:
        return np.nan  # Handle unexpected values

# Add TASK_DIFFICULTY column
df['TASK_DIFFICULTY'] = df['DATA_INTENSITY'].apply(determine_task_difficulty)

# Initialize lists to store results for each group
group_stats = []

# Group by the 'MODEL' and 'DATA_INTENSITY' fields
grouped = df.groupby(['MODEL', 'DATA_INTENSITY'])

# Function to calculate statistics for each group with 2 decimal places
def calculate_group_stats(group, model_name, data_intensity):
    stats = {}
    stats['MODEL'] = model_name
    stats['DATA_INTENSITY'] = data_intensity
    
    # Calculate total experiments
    stats['TOTAL_EXPERIMENTS'] = group.shape[0]
    
    # Calculate mean values
    stats['MODEL_WORK_TIME'] = round(group['MODEL_WORK_TIME'].mean(), 2)
    stats['SAVE_DATA_TO_SNOWFLAKE_TIME'] = round(group['SAVE_DATA_TO_SNOWFLAKE_TIME'].mean(), 2)
    stats['EXPERIMENT_TIME_TOTAL'] = round(group['EXPERIMENT_TIME_TOTAL'].mean(), 2)
    stats['DATA_INTENSITY_MEAN'] = round(group['DATA_INTENSITY'].mean(), 2)
    stats['PROMPT_VOLUME_MEAN'] = round(group['PROMPT_VOLUME'].mean(), 2)
    stats['LOAD_TO_STAGING_TIME_MEAN'] = round(group['LOAD_TO_STAGING_TIME'].mean(), 2)
    
    # Calculate statistics for successful experiments (CORRECTNESS != 0)
    success_group = group[group['CORRECTNESS'] != 0]
    stats['MODEL_WORK_TIME_SUCCESS'] = round(success_group['MODEL_WORK_TIME'].mean(), 2)
    stats['EXPERIMENT_TIME_TOTAL_SUCCESS'] = round(success_group['EXPERIMENT_TIME_TOTAL'].mean(), 2)
    stats['CORRECTNESS_IN_SUCCESS'] = round(success_group['CORRECTNESS'].mean(), 2) * 100
    stats['SUCCESS_OUTPUT_VOLUME_MEAN'] = round(success_group['OUTPUT_VOLUME'].mean(), 2)

    # Count successes and failures
    stats['SUCCESS_EXPERIMENTS'] = success_group.shape[0]
    stats['FAILED_EXPERIMENTS'] = group[group['CORRECTNESS'] == 0].shape[0]
    
    # Calculate success rate as a percentage
    stats['SUCCESS_RATE'] = round((stats['SUCCESS_EXPERIMENTS'] / stats['TOTAL_EXPERIMENTS']) * 100, 2)
    
    # Add TASK_DIFFICULTY
    stats['TASK_DIFFICULTY'] = determine_task_difficulty(data_intensity)
    
    return stats

# Apply the function to each group and collect the results
for (model_name, data_intensity), group in grouped:
    group_stats.append(calculate_group_stats(group, model_name, data_intensity))

# Calculate total statistics for each unique DATA_INTENSITY value across all MODEL groups
data_intensity_groups = df.groupby('DATA_INTENSITY')
for data_intensity_value, group in data_intensity_groups:
    group_stats.append(calculate_group_stats(group, f'TOTAL_PER_DATA_INTENSITY_{data_intensity_value}', data_intensity_value))

# Calculate total statistics for the entire dataset
total_stats = calculate_group_stats(df, 'TOTAL', 'TOTAL')

# Append the total statistics to the group stats
group_stats.append(total_stats)

# Create a DataFrame from the statistics
final_df = pd.DataFrame(group_stats)

# Save the final DataFrame to a new CSV file
new_csv_file_path = csv_file_path.replace(".csv", "_STATISTICS.csv")
final_df.to_csv(new_csv_file_path, index=False)

print(f"Processed CSV file saved as: {new_csv_file_path}")

final_df.to_csv(new_csv_file_path, index=False)

print(f"Processed CSV file saved as: {new_csv_file_path}")
