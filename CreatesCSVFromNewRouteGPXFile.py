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
    prev_elevation_loss = 0.0  # NEW: Track previous elevation loss

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

        # Calculate interaction features for GAIN
        cum_dist_elev_gain = cum_distance_km * elev_gain
        cum_dist_prev_elev_gain = cum_distance_km * prev_elevation_gain
        cum_dist_up_grad = cum_distance_km * uphill_gradient

        # NEW: Calculate interaction features for LOSS
        cum_dist_elev_loss = cum_distance_km * elev_loss
        cum_dist_prev_elev_loss = cum_distance_km * prev_elevation_loss
        cum_dist_down_grad = cum_distance_km * downhill_gradient

        # Create segment record with all features in the correct order
        segment_record = {
            'run_id': run_id,
            'segment_km': idx + 1,
            'segment_distance_km': seg_distance,
            'cum_distance_km': cum_distance_km,
            'elevation_gain_m': elev_gain,
            'prev_km_elevation_gain': prev_elevation_gain,
            'cum_elevation_gain_m': cum_elevation_gain_m,
            'elevation_loss_m': elev_loss,
            'prev_km_elevation_loss': prev_elevation_loss,  # NEW
            'cum_elevation_loss_m': cum_elevation_loss_m,
            'uphill_gradient': uphill_gradient,
            'downhill_gradient': downhill_gradient,
            'cum_dist_elev_gain': cum_dist_elev_gain,
            'cum_dist_elev_loss': cum_dist_elev_loss,  # NEW
            'cum_dist_prev_elev_gain': cum_dist_prev_elev_gain,
            'cum_dist_prev_elev_loss': cum_dist_prev_elev_loss,  # NEW
            'cum_dist_up_grad': cum_dist_up_grad,
            'cum_dist_down_grad': cum_dist_down_grad,  # NEW
        }
        
        data.append(segment_record)
        
        # Update previous values for next iteration
        prev_elevation_gain = elev_gain
        prev_elevation_loss = elev_loss  # NEW

    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Define exact column order (matches your training dataset)
    column_order = [
        'run_id',
        'segment_km',
        'segment_distance_km',
        'cum_distance_km',
        'elevation_gain_m',
        'prev_km_elevation_gain',
        'cum_elevation_gain_m',
        'elevation_loss_m',
        'prev_km_elevation_loss',
        'cum_elevation_loss_m',
        'uphill_gradient',
        'downhill_gradient',
        'cum_dist_elev_gain',
        'cum_dist_elev_loss',
        'cum_dist_prev_elev_gain',
        'cum_dist_prev_elev_loss',
        'cum_dist_up_grad',
        'cum_dist_down_grad'
    ]
    
    # Reorder columns
    df = df[column_order]
    
    print(f"\nRoute dataset shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    
    # Display summary of all enhanced features
    enhanced_features = [
        'uphill_gradient', 'downhill_gradient', 
        'cum_dist_elev_gain', 'cum_dist_elev_loss',
        'cum_dist_prev_elev_gain', 'cum_dist_prev_elev_loss',
        'cum_dist_up_grad', 'cum_dist_down_grad'
    ]
    
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
    output_path = 'D:\\Most Recent\\JonkershoekGate.csv'
    
    # Create enhanced route dataset
    route_df = process_gpx_route_with_enhanced_features(gpx_path, output_path)
    
    print(f"\nRoute processing complete!")
    print(f"Run ID: {route_df['run_id'].iloc[0]}")
    print(f"Total segments: {len(route_df)}")
    print(f"Total route distance: {route_df['cum_distance_km'].max():.2f} km")
    print(f"Total elevation gain: {route_df['cum_elevation_gain_m'].max():.1f} m")
    print(f"Total elevation loss: {route_df['cum_elevation_loss_m'].max():.1f} m")
    print(f"Average uphill gradient: {route_df['uphill_gradient'].mean():.4f}")
    print(f"Average downhill gradient: {route_df['downhill_gradient'].mean():.4f}")
    
    # Display first few rows with key columns
    print(f"\nFirst 5 segments:")
    key_columns = [
        'run_id', 'segment_km', 
        'elevation_gain_m', 'prev_km_elevation_gain',
        'elevation_loss_m', 'prev_km_elevation_loss',
        'uphill_gradient', 'downhill_gradient'
    ]
    print(route_df[key_columns].head())
    
    # Verify column order
    print(f"\nColumn order verification:")
    for i, col in enumerate(route_df.columns, 1):
        print(f"  {i}. {col}")