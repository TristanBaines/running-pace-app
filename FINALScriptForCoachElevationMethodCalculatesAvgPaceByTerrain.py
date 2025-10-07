import pandas as pd
import numpy as np

def process_running_data(csv_file_path):
    """
    Process running data CSV to calculate slopes and analyze pacing by terrain type.
    
    Parameters:
    csv_file_path (str): Path to the CSV file containing running data
    
    Returns:
    dict: Dictionary containing processed data and analysis results
    """
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} segments from {csv_file_path}")
    except FileNotFoundError:
        print(f"Error: Could not find file {csv_file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    # Check required columns
    required_columns = ['run_id', 'segment_distance_km', 'elevation_gain_m', 'elevation_loss_m', 'avg_pace_min/km']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None
    
    # Calculate net elevation for each segment
    df['net_elevation_m'] = df['elevation_gain_m'] - df['elevation_loss_m']
    
    # Convert segment distance from km to meters
    df['segment_distance_m'] = df['segment_distance_km'] * 1000
    
    # Calculate slope (net elevation / distance in meters)
    # Handle division by zero
    df['slope'] = np.where(df['segment_distance_m'] != 0, 
                          df['net_elevation_m'] / df['segment_distance_m'], 
                          0)
    
    # Classify slopes as flat, uphill, or downhill
    def classify_slope(slope, flat_threshold=0.01):
        """
        Classify slope as flat, uphill, or downhill
        flat_threshold: threshold for considering terrain as flat (1% grade = 0.01)
        """
        if slope > flat_threshold:
            return 'uphill'
        elif slope < -flat_threshold:
            return 'downhill'
        else:
            return 'flat'
    
    df['terrain_type'] = df['slope'].apply(classify_slope)
    
    # Calculate average pace per terrain class
    avg_pace_by_terrain = df.groupby('terrain_type')['avg_pace_min/km'].agg([
        'mean', 'count', 'std'
    ]).round(3)
    
    # Add column names for clarity
    avg_pace_by_terrain.columns = ['avg_pace_min_per_km', 'segment_count', 'std_dev']
    
    # Calculate additional statistics
    terrain_distribution = df['terrain_type'].value_counts(normalize=True).round(3)
    
    # Summary statistics
    slope_stats = df['slope'].describe()
    
    print("\n" + "="*50)
    print("RUNNING DATA ANALYSIS RESULTS")
    print("="*50)
    
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
    
    # Create results dictionary
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
    """
    Save the processed data to a new CSV file
    """
    if results is None:
        print("No results to save")
        return
    
    df = results['processed_dataframe']
    df.to_csv(output_file_path, index=False)
    print(f"\nProcessed data saved to: {output_file_path}")
    
    # Also save summary statistics with offsets
    summary_file = output_file_path.replace('.csv', '_summary.csv')
    
    # Create a comprehensive summary including offsets
    summary_df = results['avg_pace_by_terrain'].copy()
    
    # Add offsets to the summary dataframe if they exist
    if 'pace_offsets' in results:
        print(f"Debug: Found pace_offsets in results: {results['pace_offsets']}")
        
        # Add offset columns to the summary dataframe
        summary_df['uphill_offset_percent'] = None
        summary_df['downhill_offset_percent'] = None
        
        # Fill in the offset values for the corresponding terrain types
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
    
    # Save the complete summary with pace data and offsets
    summary_df.to_csv(summary_file)
    print(f"Summary statistics with offsets saved to: {summary_file}")

# Example usage
if __name__ == "__main__":
    # Replace with your CSV file path
    input_file = "D:\\Most Recent\\TaliaStravaData\\TaliasFinalCleanedDataset.csv"
    output_file = "Talias_processed_running_data.csv"
    
    # Process the data
    results = process_running_data(input_file)
    
    if results is not None:
        # Save results
        save_results_to_csv(results, output_file)
        
        # Access specific results
        print("\n" + "="*30)
        print("Sample processed data:")
        print("="*30)
        print(results['processed_dataframe'][['run_id', 'segment_distance_km', 'net_elevation_m', 'slope', 'terrain_type', 'avg_pace_min/km']].head(10))