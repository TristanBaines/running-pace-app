import pandas as pd
import numpy as np

def process_running_data(csv_file_path):
    
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} segments from {csv_file_path}")
    except FileNotFoundError:
        print(f"Error: Could not find file {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    required_columns = ['run_id', 'segment_distance_km', 'elevation_gain_m', 'elevation_loss_m', 'avg_pace_min/km'] # required columns 
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None
    
    df['net_elevation_m'] = df['elevation_gain_m'] - df['elevation_loss_m'] # net elevation
    
    df['segment_distance_m'] = df['segment_distance_km'] * 1000 # segment distance from km to m
    
    df['slope'] = np.where(df['segment_distance_m'] != 0, # slope, net elevation / distance in m
                          df['net_elevation_m'] / df['segment_distance_m'], 
                          0)
    
    
    def classify_slope(slope, flat_threshold=0.01): # classify slopes
        if slope > flat_threshold:
            return 'uphill'
        elif slope < -flat_threshold:
            return 'downhill'
        else:
            return 'flat'
    
    df['terrain_type'] = df['slope'].apply(classify_slope)
    
    avg_pace_by_terrain = df.groupby('terrain_type')['avg_pace_min/km'].agg([ # avg pace per class
        'mean', 'count', 'std'
    ]).round(3)
    
    avg_pace_by_terrain.columns = ['avg_pace_min_per_km', 'segment_count', 'std_dev']
    
    terrain_distribution = df['terrain_type'].value_counts(normalize=True).round(3)
    
    slope_stats = df['slope'].describe()  

    print("RUNNING DATA ANALYSIS RESULTS:")
    
    print(f"\nDataset Overview:")
    print(f"- Total segments: {len(df)}")
    print(f"- Unique runs: {df['run_id'].nunique()}")
    print(f"- Total distance: {df['segment_distance_km'].sum():.2f} km")
    
    print(f"\nSlope Statistics:")
    print(f"- Mean slope: {slope_stats['mean']:.4f}")
    print(f"- Min slope: {slope_stats['min']:.4f}")
    print(f"- Max slope: {slope_stats['max']:.4f}")
    print(f"- Std deviation: {slope_stats['std']:.4f}")
    
    print(f"\nTerrain Distribution:")
    for terrain, percentage in terrain_distribution.items():
        print(f"- {terrain.capitalize()}: {percentage:.1%}")
    
    print(f"\nAverage Pace by Terrain Type:")
    for terrain in avg_pace_by_terrain.index:
        avg_pace = avg_pace_by_terrain.loc[terrain, 'avg_pace_min_per_km']
        count = avg_pace_by_terrain.loc[terrain, 'segment_count']
        std = avg_pace_by_terrain.loc[terrain, 'std_dev']
        print(f"- {terrain.capitalize()}: {avg_pace:.3f} min/km (n={count}, std={std:.3f})")
    
    results = {
        'processed_dataframe': df,
        'avg_pace_by_terrain': avg_pace_by_terrain,
        'terrain_distribution': terrain_distribution,
        'slope_statistics': slope_stats,
        'summary': {
            'total_segments': len(df),
            'unique_runs': df['run_id'].nunique(),
            'total_distance_km': df['segment_distance_km'].sum()
        }
    }
    
    return results

def save_results_to_csv(results, output_file_path):
    if results is None:
        print("No results to save")
        return
    
    df = results['processed_dataframe']
    df.to_csv(output_file_path, index=False)
    print(f"\nProcessed data saved to: {output_file_path}")
    
    summary_file = output_file_path.replace('.csv', '_summary.csv')
    
    summary_df = results['avg_pace_by_terrain'].copy()
    
    if 'pace_offsets' in results:
        print(f"Debug: Found pace_offsets in results: {results['pace_offsets']}")
        
        summary_df['uphill_offset_percent'] = None
        summary_df['downhill_offset_percent'] = None
        
        if 'uphill' in summary_df.index and results['pace_offsets']['uphill_offset_percent'] is not None:
            summary_df.loc['uphill', 'uphill_offset_percent'] = results['pace_offsets']['uphill_offset_percent']
            print(f"Debug: Added uphill offset: {results['pace_offsets']['uphill_offset_percent']}")
        
        if 'downhill' in summary_df.index and results['pace_offsets']['downhill_offset_percent'] is not None:
            summary_df.loc['downhill', 'downhill_offset_percent'] = results['pace_offsets']['downhill_offset_percent']
            print(f"Debug: Added downhill offset: {results['pace_offsets']['downhill_offset_percent']}")
    else:
        print("Debug: pace_offsets not found in results")
    
    print(f"Debug: Summary dataframe before saving:")
    print(summary_df)
    
    summary_df.to_csv(summary_file)
    print(f"Summary statistics with offsets saved to: {summary_file}")

if __name__ == "__main__":
    input_file = "D:\\Most Recent\\TaliaStravaData\\TaliasFinalCleanedDataset.csv"
    output_file = "Talias_processed_running_data.csv"
    
    results = process_running_data(input_file)
    
    if results is not None:
        save_results_to_csv(results, output_file)
        
        print("Sample processed data:")
        print(results['processed_dataframe'][['run_id', 'segment_distance_km', 'net_elevation_m', 'slope', 'terrain_type', 'avg_pace_min/km']].head(10))