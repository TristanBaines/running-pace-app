import pandas as pd
import numpy as np

dataset = pd.read_csv("D:\\Most Recent\\TaliaStravaData\\fit_files_sorted\\running\\TaliaRunningDataset.csv")

dataset['run_id'] = (dataset['segment_km'] == 1.0).cumsum()


def replace_outliers(group): # function to replace outliers within each run
    run_id = group['run_id'].iloc[0]
    mean = group['avg_pace_min/km'].mean()
    std = group['avg_pace_min/km'].std()

    fast_threshold = mean - 1.5 * std
    slow_threshold = mean + 2 * std

    
    outliers_mask = (group['avg_pace_min/km'] < fast_threshold) | (group['avg_pace_min/km'] > slow_threshold) # mask for outliers
    outliers = group[outliers_mask]

    print(f"\nRun ID: {run_id}")
    print(f"  Mean pace: {mean:.2f}, Std dev: {std:.2f}")
    print(f"  Fast threshold (<): {fast_threshold:.2f}")
    print(f"  Slow threshold (>): {slow_threshold:.2f}")

    if not outliers.empty:
        print("  Outliers found:")
        print(outliers[['segment_km', 'avg_pace_min/km']])
    else:
        print("  No outliers found.")

    
    group.loc[outliers_mask, 'avg_pace_min/km'] = mean # replace outliers with the run mean

    if not outliers.empty:
        print("  After replacement:")
        print(group.loc[outliers_mask, ['segment_km', 'avg_pace_min/km']])

    return group


dataset = dataset.groupby('run_id', group_keys=False).apply(replace_outliers) # apply per run

runs_to_remove = [5, 6, 28, 43]

dataset = dataset[~dataset['run_id'].isin(runs_to_remove)]

dataset = dataset.reset_index(drop=True)


cols = dataset.columns.tolist() # move run_id to be the first column
cols.insert(0, cols.pop(cols.index('run_id')))
dataset = dataset[cols]


dataset.to_csv("D:\\Most Recent\\TaliaStravaData\\TaliasFinalCleanedDataset.csv", index=False) # save cleaned dataset