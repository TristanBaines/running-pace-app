import pandas as pd
import numpy as np
from pathlib import Path
import math

try:
    from fitparse import FitFile
except ImportError:
    print("install fitparse")
    exit(1)

def haversine_distance(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2]) # decimal degrees to radians
    
    
    dlat = lat2 - lat1 # Haversine formula
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    
    r = 6371000 # radius of earth in m
    return c * r

def parse_fit_file(file_path):

    fit_file = FitFile(str(file_path))
    records = []
    
    for record in fit_file.get_messages('record'):
        data = {}
        for field in record:
            
            if field.name in ['position_lat', 'position_long', 'altitude', 'enhanced_altitude', 'timestamp']: # check for multiple altitude field names
                if field.name in ['position_lat', 'position_long'] and field.value is not None:
                    
                    data[field.name] = field.value * (180 / 2**31) # convert semicircles to degrees
                elif field.name in ['altitude', 'enhanced_altitude'] and field.value is not None:
                    if 'altitude' not in data or field.name == 'enhanced_altitude':
                        data['altitude'] = field.value
                else:
                    data[field.name] = field.value
        
        
        if 'position_lat' in data and 'position_long' in data and data['position_lat'] is not None: # only include records with GPS data
            records.append(data)
    
    df = pd.DataFrame(records)
    

    if not df.empty: # Debug altitude info
        if 'altitude' in df.columns:
            non_null = df['altitude'].notna().sum()
            if non_null > 0:
                print(f"Altitude data: {non_null} points, range: {df['altitude'].min():.1f}m to {df['altitude'].max():.1f}m")
            else:
                print(f"WARNING: altitude column exists but all values are null")
        else:
            print(f"WARNING: No altitude data found in this file")
    
    return df

def calculate_cumulative_distance(df):
    if len(df) < 2:
        df['cumulative_distance_m'] = 0
        return df
    
    distances = [0]  # first point has 0 distance
    
    for i in range(1, len(df)):
        if (pd.notna(df.iloc[i]['position_lat']) and pd.notna(df.iloc[i]['position_long']) and
            pd.notna(df.iloc[i-1]['position_lat']) and pd.notna(df.iloc[i-1]['position_long'])):
            
            dist = haversine_distance(
                df.iloc[i-1]['position_lat'], df.iloc[i-1]['position_long'],
                df.iloc[i]['position_lat'], df.iloc[i]['position_long']
            )
            distances.append(distances[-1] + dist)
        else:
            distances.append(distances[-1])
    
    df['cumulative_distance_m'] = distances
    return df

def create_1km_segments(df, run_id):
    if df.empty:
        return pd.DataFrame()
    
    # Calculate cumulative distance if not already present
    if 'cumulative_distance_m' not in df.columns:
        df = calculate_cumulative_distance(df)
    
    # Convert to km
    df['cum_distance_km'] = df['cumulative_distance_m'] / 1000

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    segments = []
    max_distance_km = df['cum_distance_km'].max()
    
    # Create segments for each km
    for km in range(1, int(max_distance_km) + 1):
        # Get data for this km segment
        start_km = km - 1
        end_km = km
        
        segment_data = df[(df['cum_distance_km'] >= start_km) & 
                         (df['cum_distance_km'] <= end_km)].copy()
        
        if segment_data.empty:
            continue

        # Calculate actual measured distance for this segment
        segment_start_distance = segment_data['cum_distance_km'].iloc[0]
        segment_end_distance = segment_data['cum_distance_km'].iloc[-1]
        actual_segment_distance = segment_end_distance - segment_start_distance

        # If this is the first segment, include distance from 0
        if km == 1:
            actual_segment_distance = segment_end_distance

        # Time calculations for pace (TARGET VARIABLE)
        if 'timestamp' in segment_data.columns and not segment_data['timestamp'].isna().all():
            segment_start_time = segment_data['timestamp'].iloc[0]
            segment_end_time = segment_data['timestamp'].iloc[-1]
            elapsed_time = (segment_end_time - segment_start_time).total_seconds() / 60.0  # minutes
            
            if actual_segment_distance > 0:
                avg_pace = elapsed_time / actual_segment_distance  # min/km
            else:
                avg_pace = None
        else:
            avg_pace = None

        # Calculate segment features in the specified order
        segment = {
            'run_id': run_id,
            'segment_km': float(km),
            'segment_distance_km': actual_segment_distance,
            'cum_distance_km': round(float(km), 1)
        }
        
        # Elevation calculations
        if 'altitude' in segment_data.columns and not segment_data['altitude'].isna().all():
            elevations = segment_data['altitude'].dropna()
            
            if len(elevations) > 1:
                # Calculate elevation gain and loss for this segment
                elevation_changes = elevations.diff().dropna()
                
                elevation_gain = elevation_changes[elevation_changes > 0].sum()
                elevation_loss = abs(elevation_changes[elevation_changes < 0].sum())
                
                segment['elevation_gain_m'] = float(elevation_gain) if not pd.isna(elevation_gain) else 0.0
                segment['elevation_loss_m'] = float(elevation_loss) if not pd.isna(elevation_loss) else 0.0
            else:
                segment['elevation_gain_m'] = 0.0
                segment['elevation_loss_m'] = 0.0
        else:
            segment['elevation_gain_m'] = 0.0
            segment['elevation_loss_m'] = 0.0
        
        segments.append(segment)
    
    # Convert to DataFrame and calculate cumulative and derived features
    segments_df = pd.DataFrame(segments)
    
    if not segments_df.empty:
        # Calculate cumulative elevation features
        segments_df['cum_elevation_gain_m'] = segments_df['elevation_gain_m'].cumsum()
        segments_df['cum_elevation_loss_m'] = segments_df['elevation_loss_m'].cumsum()
        
        # Calculate previous km elevation gain
        segments_df['prev_km_elevation_gain'] = segments_df['elevation_gain_m'].shift(1).fillna(0)
        segments_df['prev_km_elevation_loss'] = segments_df['elevation_loss_m'].shift(1).fillna(0)

        # Engineered features
        segments_df['uphill_gradient'] = segments_df['elevation_gain_m'] / segments_df['segment_distance_km'].replace(0, np.nan)
        segments_df['downhill_gradient'] = segments_df['elevation_loss_m'] / segments_df['segment_distance_km'].replace(0, np.nan)
        segments_df['cum_dist_elev_gain'] = segments_df['cum_distance_km'] * segments_df['elevation_gain_m']
        segments_df['cum_dist_prev_elev_gain'] = segments_df['cum_distance_km'] * segments_df['prev_km_elevation_gain']
        segments_df['cum_dist_up_grad'] = segments_df['cum_distance_km'] * segments_df['uphill_gradient']
        segments_df['cum_dist_elev_loss'] = segments_df['cum_distance_km'] * segments_df['elevation_loss_m']
        segments_df['cum_dist_prev_elev_loss'] = segments_df['cum_distance_km'] * segments_df['prev_km_elevation_loss']
        segments_df['cum_dist_down_grad'] = segments_df['cum_distance_km'] * segments_df['downhill_gradient']
        
        # Reorder columns to match exact specification
        column_order = [
            'run_id', 'segment_km', 'segment_distance_km', 'cum_distance_km',
            'elevation_gain_m', 'prev_km_elevation_gain', 'cum_elevation_gain_m',
            'elevation_loss_m', 'prev_km_elevation_loss', 'cum_elevation_loss_m', 'uphill_gradient', 
            'downhill_gradient', 'cum_dist_elev_gain', 'cum_dist_elev_loss', 'cum_dist_prev_elev_gain', 'cum_dist_prev_elev_loss', 
            'cum_dist_up_grad', 'cum_dist_down_grad'
        ]
        
        # Reorder existing columns
        segments_df = segments_df[column_order]
        
        # Add target variable as the last column
        segments_df['avg_pace_min/km'] = None  # Will be filled per segment
        
        # Now fill in the pace values for each segment
        for idx, row in segments_df.iterrows():
            km = int(row['segment_km'])
            start_km = km - 1
            end_km = km
            
            segment_data = df[(df['cum_distance_km'] >= start_km) & 
                             (df['cum_distance_km'] <= end_km)].copy()
            
            if not segment_data.empty and 'timestamp' in segment_data.columns:
                segment_start_time = segment_data['timestamp'].iloc[0]
                segment_end_time = segment_data['timestamp'].iloc[-1]
                elapsed_time = (segment_end_time - segment_start_time).total_seconds() / 60.0
                
                if row['segment_distance_km'] > 0:
                    segments_df.at[idx, 'avg_pace_min/km'] = elapsed_time / row['segment_distance_km']
    
    return segments_df

def process_fit_files_to_dataset(folder_path, output_file="run_segments_dataset.csv"):
    """
    Process all .fit files in a folder and create a dataset of 1km segments
    Each run gets a unique run_id (integer) and segments are grouped by run
    """
    folder = Path(folder_path)
    fit_files = list(folder.glob("*.fit"))
    
    if not fit_files:
        print(f"No .fit files found in {folder_path}")
        return None
    
    print(f"Found {len(fit_files)} .fit files to process")
    
    all_segments = []
    run_id = 0  # Start at 0, increment only when segments are successfully created
    
    for file_num, fit_file in enumerate(fit_files, 1):
        print(f"Processing {fit_file.name} ({file_num}/{len(fit_files)})...")
        
        try:
            # Parse the .fit file
            df = parse_fit_file(fit_file)
            
            if df.empty:
                print(f"  Warning: No valid GPS data found in {fit_file.name} - SKIPPED")
                continue
            
            # Increment run_id only for successful processing
            run_id += 1
            
            # Create 1km segments with integer run_id
            segments = create_1km_segments(df, run_id)
            
            if not segments.empty:
                all_segments.append(segments)
                print(f"Created {len(segments)} segments from {fit_file.name} (run_id: {run_id})")
            else:
                print(f"Warning: No segments created from {fit_file.name} - SKIPPED")
                run_id -= 1  # Decrement since we didn't actually add segments
                
        except Exception as e:
            print(f"Error processing {fit_file.name}: {e} - SKIPPED")
            continue
    
    if not all_segments:
        print("No segments were created from any files")
        return None
    
    # Combine all segments into one dataset
    final_dataset = pd.concat(all_segments, ignore_index=True)
    
    # Save to CSV
    output_path = folder / output_file
    final_dataset.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Dataset created successfully!")
    print(f"{'='*60}")
    print(f"Total segments: {len(final_dataset)}")
    print(f"Unique runs: {final_dataset['run_id'].nunique()}")
    print(f"Output file: {output_path}")
    
    # Show sample of the data
    print(f"\nFirst 10 rows of dataset:")
    print(final_dataset.head(10).to_string())
    
    # Show summary by run
    print(f"\n{'='*60}")
    print(f"Summary by run_id:")
    print(f"{'='*60}")
    summary = final_dataset.groupby('run_id').agg({
        'segment_km': 'count',
        'cum_distance_km': 'max',
        'elevation_gain_m': 'sum',
        'elevation_loss_m': 'sum',
        'avg_pace_min/km': 'mean'
    }).round(2)
    summary.columns = ['num_segments', 'total_distance_km', 'total_elevation_gain_m', 
                       'total_elevation_loss_m', 'avg_pace_min/km']
    print(summary.to_string())
    
    return final_dataset

# Example usage
if __name__ == "__main__":
    
    folder_path = "D:\\Most Recent\\TaliaStravaData\\fit_files_sorted\\running"
    
    # Process all .fit files and create dataset
    dataset = process_fit_files_to_dataset(folder_path, "TaliaRunningDataset.csv")
    
    if dataset is not None:
        print(f"\n{'='*60}")
        print("Dataset column order:")
        print(f"{'='*60}")
        for i, col in enumerate(dataset.columns, 1):
            marker = "<-- TARGET" if col == "avg_pace_min/km" else ""
            print(f"{i}. {col} {marker}")
        print(f"{'='*60}")



    