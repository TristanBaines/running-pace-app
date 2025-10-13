"""
Partial Segment Handler
Handles predictions for partial segments by normalizing features to 1km equivalents
"""

import pandas as pd
import numpy as np

class PartialSegmentHandler:
    def __init__(self, min_full_segment_distance=0.9):
        """
        Initialize handler
        
        Args:
            min_full_segment_distance: Threshold below which segments are considered partial (default 0.9 km)
        """
        self.min_full_segment = min_full_segment_distance
        
    def identify_partial_segments(self, route_df):
        """
        Identify which segments are partial
        
        Args:
            route_df: DataFrame with route segments
            
        Returns:
            Boolean mask indicating partial segments
        """
        return route_df['segment_distance_km'] < self.min_full_segment
    
    def normalize_features_for_prediction(self, route_df):
        """
        Normalize features in partial segments to 1km equivalents
        This allows the model (trained on 1km segments) to make better predictions
        
        Args:
            route_df: DataFrame with route segments
            
        Returns:
            DataFrame with normalized features for partial segments
        """
        df_normalized = route_df.copy()
        partial_mask = self.identify_partial_segments(df_normalized)
        
        if not partial_mask.any():
            print("No partial segments detected")
            return df_normalized
        
        print(f"\nNormalizing {partial_mask.sum()} partial segment(s) for prediction:")
        
        for idx in df_normalized[partial_mask].index:
            actual_distance = df_normalized.loc[idx, 'segment_distance_km']
            scale_factor = 1.0 / actual_distance  # How much to scale up to 1km
            
            print(f"  Segment {df_normalized.loc[idx, 'segment_km']}: " 
                  f"{actual_distance:.3f} km → normalizing to 1.0 km (scale: {scale_factor:.3f}x)")
            
            # Normalize distance-dependent features
            # These features should be scaled to represent what they would be for 1km
            
            # 1. Elevation features (total gain/loss would be higher over 1km)
            if 'elevation_gain_m' in df_normalized.columns:
                original_gain = df_normalized.loc[idx, 'elevation_gain_m']
                df_normalized.loc[idx, 'elevation_gain_m'] = original_gain * scale_factor
            
            if 'elevation_loss_m' in df_normalized.columns:
                original_loss = df_normalized.loc[idx, 'elevation_loss_m']
                df_normalized.loc[idx, 'elevation_loss_m'] = original_loss * scale_factor
            
            # 2. Gradient features (these should stay the same - gradient is rise/run)
            # uphill_gradient and downhill_gradient are already normalized (per km)
            # so we DON'T change them
            
            # 3. Interaction features that include segment distance
            # These need to be recalculated with normalized elevation
            if 'cum_distance_km' in df_normalized.columns:
                cum_dist = df_normalized.loc[idx, 'cum_distance_km']
                
                if 'cum_dist_elev_gain' in df_normalized.columns:
                    normalized_gain = df_normalized.loc[idx, 'elevation_gain_m']
                    df_normalized.loc[idx, 'cum_dist_elev_gain'] = cum_dist * normalized_gain
                
                if 'cum_dist_elev_loss' in df_normalized.columns:
                    normalized_loss = df_normalized.loc[idx, 'elevation_loss_m']
                    df_normalized.loc[idx, 'cum_dist_elev_loss'] = cum_dist * normalized_loss
                
                if 'cum_dist_up_grad' in df_normalized.columns:
                    uphill_grad = df_normalized.loc[idx, 'uphill_gradient']
                    df_normalized.loc[idx, 'cum_dist_up_grad'] = cum_dist * uphill_grad
                
                if 'cum_dist_down_grad' in df_normalized.columns:
                    downhill_grad = df_normalized.loc[idx, 'downhill_gradient']
                    df_normalized.loc[idx, 'cum_dist_down_grad'] = cum_dist * downhill_grad
        
        return df_normalized
    
    def denormalize_predictions(self, predictions, route_df):
        """
        Convert predictions back to actual pace for partial segments
        
        Since we normalized to 1km for prediction, we need to convert back:
        - Predicted pace is for 1km
        - Time for actual distance = predicted_pace × actual_distance
        - Actual pace per km = time / actual_distance = predicted_pace
        
        Actually, the pace per km should remain the same! 
        But we need to adjust the TOTAL TIME for the segment.
        
        Args:
            predictions: Array of predicted paces (min/km)
            route_df: Original DataFrame with actual segment distances
            
        Returns:
            Dictionary with adjusted predictions
        """
        partial_mask = self.identify_partial_segments(route_df)
        
        adjusted_predictions = predictions.copy()
        segment_times = predictions * route_df['segment_distance_km'].values
        
        results = {
            'predicted_paces': adjusted_predictions,  # Pace per km stays the same
            'segment_times': segment_times,  # Time for each segment
            'partial_segments': partial_mask.values
        }
        
        # Print information about partial segments
        if partial_mask.any():
            print(f"\nPartial segment time adjustments:")
            for idx in route_df[partial_mask].index:
                seg_num = route_df.loc[idx, 'segment_km']
                actual_dist = route_df.loc[idx, 'segment_distance_km']
                predicted_pace = predictions[idx]
                actual_time = predicted_pace * actual_dist
                
                print(f"  Segment {int(seg_num)}: {actual_dist:.3f} km at {predicted_pace:.2f} min/km "
                      f"= {actual_time:.2f} minutes")
        
        return results
    
    def get_summary_stats(self, route_df, predictions):
        """
        Get summary statistics accounting for partial segments
        
        Args:
            route_df: DataFrame with route data
            predictions: Array of predicted paces
            
        Returns:
            Dictionary with summary statistics
        """
        partial_mask = self.identify_partial_segments(route_df)
        
        # Calculate times for each segment
        segment_times = predictions * route_df['segment_distance_km'].values
        total_time = segment_times.sum()
        
        # Calculate average pace weighted by distance
        total_distance = route_df['segment_distance_km'].sum()
        weighted_avg_pace = total_time / total_distance if total_distance > 0 else 0
        
        # Full segments only
        full_segments_mask = ~partial_mask
        if full_segments_mask.any():
            full_segment_avg_pace = predictions[full_segments_mask].mean()
        else:
            full_segment_avg_pace = None
        
        return {
            'total_distance_km': total_distance,
            'total_time_min': total_time,
            'weighted_avg_pace_min_per_km': weighted_avg_pace,
            'full_segments_avg_pace': full_segment_avg_pace,
            'num_full_segments': full_segments_mask.sum(),
            'num_partial_segments': partial_mask.sum(),
            'partial_segment_distances': route_df.loc[partial_mask, 'segment_distance_km'].tolist()
        }


def predict_route_with_partial_handling(predictor, route_csv_path, output_csv_path=None):
    """
    Wrapper function to predict with proper partial segment handling
    
    Args:
        predictor: Trained PacePredictor instance
        route_csv_path: Path to route CSV
        output_csv_path: Path to save predictions
        
    Returns:
        Predictions with partial segment adjustments
    """
    # Load route data
    route_df = pd.read_csv(route_csv_path)
    
    # Initialize handler
    handler = PartialSegmentHandler(min_full_segment_distance=0.9)
    
    # Step 1: Normalize features for partial segments
    route_df_normalized = handler.normalize_features_for_prediction(route_df)
    
    # Step 2: Save normalized features temporarily
    temp_normalized_path = route_csv_path.replace('.csv', '_normalized_temp.csv')
    route_df_normalized.to_csv(temp_normalized_path, index=False)
    
    # Step 3: Make predictions using normalized features
    print("\nMaking predictions with normalized features...")
    predictions, _ = predictor.predict_route(temp_normalized_path, create_plots=False)
    
    # Step 4: Denormalize predictions (adjust for actual distances)
    results = handler.denormalize_predictions(predictions, route_df)
    
    # Step 5: Get summary statistics
    summary = handler.get_summary_stats(route_df, results['predicted_paces'])
    
    print("\n" + "="*60)
    print("PREDICTION SUMMARY (with partial segment handling)")
    print("="*60)
    print(f"Total distance: {summary['total_distance_km']:.2f} km")
    print(f"Total time: {summary['total_time_min']:.1f} minutes "
          f"({int(summary['total_time_min']//60)}:{int(summary['total_time_min']%60):02d})")
    print(f"Distance-weighted average pace: {summary['weighted_avg_pace_min_per_km']:.2f} min/km")
    
    if summary['num_partial_segments'] > 0:
        print(f"\n⚠️  Route contains {summary['num_partial_segments']} partial segment(s):")
        for i, dist in enumerate(summary['partial_segment_distances'], 1):
            print(f"   Partial segment {i}: {dist:.3f} km")
        print(f"   These segments were normalized for prediction accuracy")
    
    if summary['full_segments_avg_pace']:
        print(f"\nFull segments average pace: {summary['full_segments_avg_pace']:.2f} min/km")
        print(f"Full segments: {summary['num_full_segments']}")
    
    # Step 6: Save results
    if output_csv_path:
        output_df = route_df.copy()
        output_df['predicted_pace'] = results['predicted_paces']
        output_df['segment_time_min'] = results['segment_times']
        output_df['is_partial_segment'] = results['partial_segments']
        output_df.to_csv(output_csv_path, index=False)
        print(f"\nPredictions saved to: {output_csv_path}")
    
    # Clean up temp file
    import os
    if os.path.exists(temp_normalized_path):
        os.remove(temp_normalized_path)
    
    return results, summary


# Example usage
if __name__ == "__main__":
    print("=== Partial Segment Handler Example ===\n")
    
    # Example route data
    example_route = pd.DataFrame({
        'segment_km': [1, 2, 3, 4],
        'segment_distance_km': [1.0, 1.0, 1.0, 0.6],  # Last segment is partial
        'elevation_gain_m': [20, 30, 25, 15],
        'elevation_loss_m': [10, 15, 20, 8],
        'cum_distance_km': [1.0, 2.0, 3.0, 3.6],
        'uphill_gradient': [20, 30, 25, 25],  # 25 for partial (15/0.6)
        'downhill_gradient': [10, 15, 20, 13.33],
        'cum_dist_elev_gain': [20, 60, 75, 54],
        'cum_dist_up_grad': [20, 60, 75, 90]
    })
    
    print("Original route data:")
    print(example_route[['segment_km', 'segment_distance_km', 'elevation_gain_m']].to_string(index=False))
    
    # Initialize handler
    handler = PartialSegmentHandler()
    
    # Normalize
    normalized = handler.normalize_features_for_prediction(example_route)
    
    print("\nNormalized features (partial segment scaled to 1km):")
    print(normalized[['segment_km', 'segment_distance_km', 'elevation_gain_m']].to_string(index=False))
    
    # Simulate predictions
    example_predictions = np.array([5.5, 5.8, 6.0, 6.2])
    
    # Denormalize
    results = handler.denormalize_predictions(example_predictions, example_route)
    
    print("\nFinal predictions:")
    for i in range(len(example_route)):
        print(f"Segment {i+1}: {example_route.iloc[i]['segment_distance_km']:.2f} km "
              f"at {results['predicted_paces'][i]:.2f} min/km "
              f"= {results['segment_times'][i]:.2f} min")