import gpxpy
from geopy.distance import geodesic
import pandas as pd
import numpy as np

def segment_route_by_distance(points, segment_length_km=1):
    segments = []
    current_segment = []
    accumulated_distance = 0.0

    for i in range(len(points)):
        if i == 0:
            current_segment.append(points[i])
            continue

        prev_point = points[i-1]
        curr_point = points[i]
        dist = geodesic(
            (prev_point['latitude'], prev_point['longitude']),
            (curr_point['latitude'], curr_point['longitude'])
        ).km

        accumulated_distance += dist
        current_segment.append(curr_point)

        if accumulated_distance >= segment_length_km:
            segments.append(current_segment)
            current_segment = []
            accumulated_distance = 0.0

    if current_segment:
        segments.append(current_segment)

    return segments

def calculate_elevation_gain_loss(segment):
    elevation_gain = 0.0
    elevation_loss = 0.0

    for i in range(1, len(segment)):
        if segment[i]['elevation'] is None or segment[i-1]['elevation'] is None:
            continue
        elev_diff = segment[i]['elevation'] - segment[i-1]['elevation']
        if elev_diff > 0:
            elevation_gain += elev_diff
        elif elev_diff < 0:
            elevation_loss += abs(elev_diff)

    return elevation_gain, elevation_loss

def calculate_segment_distance(segment):
    distance = 0.0
    for i in range(1, len(segment)):
        prev = segment[i-1]
        curr = segment[i]
        distance += geodesic(
            (prev['latitude'], prev['longitude']),
            (curr['latitude'], curr['longitude'])
        ).km
    return distance

def process_gpx_route_with_enhanced_features(gpx_path, output_path=None, segment_length_km=1.0):
    """
    Process GPX route and create dataset with all enhanced features matching the training dataset.
    """

    run_id = 1

    print(f"Processing route with run_id: {run_id}")
    
    # Load your GPX file
    print(f"Loading GPX file: {gpx_path}")
    with open(gpx_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    # Extract all points from the GPX file
    route_points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                route_points.append({
                    'latitude': point.latitude,
                    'longitude': point.longitude,
                    'elevation': point.elevation,
                    'time': point.time.isoformat() if point.time else None
                })

    print(f"Extracted {len(route_points)} route points")

    # Split the route into segments
    route_segments = segment_route_by_distance(route_points, segment_length_km)
    print(f"Created {len(route_segments)} segments")

    # Collect data for each segment
    data = []

    cum_distance_km = 0.0
    cum_elevation_gain_m = 0.0
    cum_elevation_loss_m = 0.0
    prev_elevation_gain = 0.0

    for idx, seg in enumerate(route_segments):
        seg_distance = calculate_segment_distance(seg)
        elev_gain, elev_loss = calculate_elevation_gain_loss(seg)

        # Update cumulative values
        cum_distance_km += seg_distance
        cum_elevation_gain_m += elev_gain
        cum_elevation_loss_m += elev_loss

        # Calculate basic gradient features
        uphill_gradient = elev_gain / seg_distance if seg_distance > 0 else 0
        downhill_gradient = elev_loss / seg_distance if seg_distance > 0 else 0

        # Calculate interaction features
        cum_dist_elev_gain = cum_distance_km * elev_gain
        cum_dist_prev_elev_gain = cum_distance_km * prev_elevation_gain
        cum_dist_up_grad = cum_distance_km * uphill_gradient

        # Create segment record with all features
        segment_record = {

            'run_id': run_id,
            # Basic segment info
            'segment_km': idx + 1,  # Changed to match training data format
            'segment_distance_km': seg_distance,
            'elevation_gain_m': elev_gain,
            'elevation_loss_m': elev_loss,
            
            # Sequential features - previous km (set to 0 for route data since no historical pace/HR)
            'prev_km_avg_pace': 0,  # No previous pace data for route
            'prev_km_avg_hr': 0,    # No previous HR data for route
            'prev_km_avg_cadence': 0,  # No previous cadence data for route
            'prev_km_elevation_gain': prev_elevation_gain,
            
            # Cumulative features
            'cum_distance_km': cum_distance_km,
            'cum_elevation_gain_m': cum_elevation_gain_m,
            'cum_elevation_loss_m': cum_elevation_loss_m,
            
            # Physiological features (set to 0 for route data)
            'avg_cadence': 0,        # No cadence data for route
            'avg_heart_rate': 0,     # No HR data for route
            'recent_avg_pace': 0,    # No recent pace data for route
            'recent_avg_hr': 0,      # No recent HR data for route
            'cum_time_sec': 0,       # No time data for route
            'avg_pace_so_far': 0,    # No pace data for route
            'avg_hr_so_far': 0,      # No HR data for route
            'avg_cadence_so_far': 0, # No cadence data for route
            
            # NEW ENHANCED FEATURES
            'uphill_gradient': uphill_gradient,
            'downhill_gradient': downhill_gradient,
            'cum_dist_elev_gain': cum_dist_elev_gain,
            'cum_dist_prev_elev_gain': cum_dist_prev_elev_gain,
            'cum_dist_up_grad': cum_dist_up_grad,
        }
        
        data.append(segment_record)
        prev_elevation_gain = elev_gain

    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns to match training dataset structure
    # Put basic features first, then sequential, then cumulative, then enhanced features
    column_order = [
        'run_id', 'segment_km', 'segment_distance_km', 'cum_distance_km', 'elevation_gain_m', 'prev_km_elevation_gain', 'cum_elevation_gain_m', 'elevation_loss_m', 'cum_elevation_loss_m', 'uphill_gradient', 'downhill_gradient', 'cum_dist_elev_gain', 'cum_dist_prev_elev_gain', 'cum_dist_up_grad'
    ]
    
    # Only include columns that exist in the dataframe
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    print(f"\nRoute dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Display summary of enhanced features
    enhanced_features = ['uphill_gradient', 'downhill_gradient', 'cum_dist_elev_gain', 
                        'cum_dist_prev_elev_gain', 'cum_dist_up_grad']
    
    print(f"\nEnhanced features summary:")
    for feature in enhanced_features:
        print(f"  {feature}:")
        print(f"    Range: {df[feature].min():.4f} to {df[feature].max():.4f}")
        print(f"    Mean: {df[feature].mean():.4f}, Std: {df[feature].std():.4f}")
    
    # Save to CSV
    if output_path is None:
        output_path = gpx_path.replace('.gpx', '_route_segments_enhanced.csv')
    
    df.to_csv(output_path, index=False)
    print(f"\nEnhanced route dataset saved to: {output_path}")
    
    return df

if __name__ == '__main__':
    # Process your GPX route
    gpx_path = 'D:\\JonkershoekGate.gpx'
    output_path = 'D:\\Most Recent\\NewRoute.csv'
    
    # Create enhanced route dataset
    route_df = process_gpx_route_with_enhanced_features(gpx_path, output_path)
    
    print(f"\nRoute processing complete!")
    print(f"Run ID: {route_df['run_id'].iloc[0]}")
    print(f"Total segments: {len(route_df)}")
    print(f"Total route distance: {route_df['cum_distance_km'].max():.2f} km")
    print(f"Total elevation gain: {route_df['cum_elevation_gain_m'].max():.1f} m")
    print(f"Average uphill gradient: {route_df['uphill_gradient'].mean():.4f}")
    
    # Display first few rows
    print(f"\nFirst 5 segments:")
    key_columns = ['run_id', 'segment_km', 'elevation_gain_m', 'uphill_gradient', 'cum_dist_elev_gain', 'cum_dist_up_grad']
    print(route_df[key_columns].head())