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
    run_id = 1

    print(f"Processing route with run_id: {run_id}")
    
    print(f"Loading GPX file: {gpx_path}")
    with open(gpx_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

    
    route_points = [] # extract all points from the GPX file
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

    
    route_segments = segment_route_by_distance(route_points, segment_length_km) # segment route
    print(f"Created {len(route_segments)} segments")

    data = []

    cum_distance_km = 0.0
    cum_elevation_gain_m = 0.0
    cum_elevation_loss_m = 0.0
    prev_elevation_gain = 0.0
    prev_elevation_loss = 0.0

    for idx, seg in enumerate(route_segments):
        seg_distance = calculate_segment_distance(seg)
        elev_gain, elev_loss = calculate_elevation_gain_loss(seg)

        
        cum_distance_km += seg_distance # update cumulative values
        cum_elevation_gain_m += elev_gain
        cum_elevation_loss_m += elev_loss

        
        uphill_gradient = elev_gain / seg_distance if seg_distance > 0 else 0 # calculate gradient features
        downhill_gradient = elev_loss / seg_distance if seg_distance > 0 else 0

        
        cum_dist_elev_gain = cum_distance_km * elev_gain # interaction features for gain
        cum_dist_prev_elev_gain = cum_distance_km * prev_elevation_gain
        cum_dist_up_grad = cum_distance_km * uphill_gradient

        
        cum_dist_elev_loss = cum_distance_km * elev_loss # interaction features for loss
        cum_dist_prev_elev_loss = cum_distance_km * prev_elevation_loss
        cum_dist_down_grad = cum_distance_km * downhill_gradient

        
        segment_record = { # all features in the correct order
            'run_id': run_id,
            'segment_km': idx + 1,
            'segment_distance_km': seg_distance,
            'cum_distance_km': cum_distance_km,
            'elevation_gain_m': elev_gain,
            'prev_km_elevation_gain': prev_elevation_gain,
            'cum_elevation_gain_m': cum_elevation_gain_m,
            'elevation_loss_m': elev_loss,
            'prev_km_elevation_loss': prev_elevation_loss,  
            'cum_elevation_loss_m': cum_elevation_loss_m,
            'uphill_gradient': uphill_gradient,
            'downhill_gradient': downhill_gradient,
            'cum_dist_elev_gain': cum_dist_elev_gain,
            'cum_dist_elev_loss': cum_dist_elev_loss,  
            'cum_dist_prev_elev_gain': cum_dist_prev_elev_gain,
            'cum_dist_prev_elev_loss': cum_dist_prev_elev_loss,  
            'cum_dist_up_grad': cum_dist_up_grad,
            'cum_dist_down_grad': cum_dist_down_grad,  
        }
        
        data.append(segment_record)
        
        
        prev_elevation_gain = elev_gain # update previous values for next iteration
        prev_elevation_loss = elev_loss

    df = pd.DataFrame(data)
    
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
    

    df = df[column_order]
    
    print(f"\nRoute dataset shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    
    if output_path is None:
        output_path = gpx_path.replace('.gpx', '_route_segments_enhanced.csv')
    
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    
    return df

if __name__ == '__main__': 

    gpx_path = 'C:\\Users\\User\\Desktop\\FirstTestRunRoute.gpx'
    output_path = 'C:\\Users\\User\\Desktop\\TestingIncompleteSegment.csv'
    

    route_df = process_gpx_route_with_enhanced_features(gpx_path, output_path)
    
    print(f"\nFirst 5 segments:")
    key_columns = [
        'run_id', 'segment_km', 
        'elevation_gain_m', 'prev_km_elevation_gain',
        'elevation_loss_m', 'prev_km_elevation_loss',
        'uphill_gradient', 'downhill_gradient'
    ]
    print(route_df[key_columns].head())
    
    
    print(f"\nColumn order verification:") # verify column order
    for i, col in enumerate(route_df.columns, 1):
        print(f"  {i}. {col}")