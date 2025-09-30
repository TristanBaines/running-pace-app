"""
Simple Pace Coaching Script
Takes predicted paces and applies selected coaching methods.
Easily customizable with your own coaching logic.
"""

import pandas as pd
import numpy as np
from typing import List, Dict

class SimplePaceCoaching:
    def __init__(self):
        """Initialize with your coaching methods."""
        # Define your 5 coaching methods here
        self.available_methods = {
            'Slower Uphills': 'Slower Uphills',
            'Faster Downhills': 'Faster Downhills', 
            'Push Flats': 'Push Flats',
            'Negative Splits': 'Negative Splits',
            'Chosen Time': 'Chosen Time'
        }
    
    def apply_coaching(self, route_data: pd.DataFrame, selected_methods: List[str], extra_params: Dict[str, dict] = None) -> Dict[str, np.ndarray]:

        # Extract predicted paces from the DataFrame
        if 'predicted_pace' not in route_data.columns:
            raise ValueError("DataFrame must contain 'predicted_pace' column")
    
        predicted_paces = route_data['predicted_pace'].values
        results = {}
        results["Uncoached Pace"] = {
            "paces": predicted_paces.copy(),
            "total_time": predicted_paces.sum()
        }
        
        cumulative_paces = predicted_paces.copy()

        # Apply each selected method
        for method in selected_methods:
            friendly_name = self.available_methods.get(method, method)

            if method == 'Slower Uphills':
                adjusted = self._method_1(route_data, paces=cumulative_paces)
            elif method == 'Faster Downhills':
                adjusted = self._method_2(route_data, paces=cumulative_paces)
            elif method == 'Push Flats':
                adjusted = self._method_3(route_data, paces=cumulative_paces)
            elif method == 'Negative Splits':
                adjusted = self._method_4(route_data, paces=cumulative_paces)
            elif method == 'Chosen Time':
                params = extra_params.get("Chosen Time", {}) if extra_params else {}
                adjusted = self._method_5(route_data, paces=cumulative_paces, **params)
            else:
                print(f"Warning: Unknown method '{method}' - skipping")
                continue
        
            results[friendly_name] = {
                "paces": adjusted.copy(),
                "total_time": adjusted.sum()
            }

            cumulative_paces = adjusted  # feed into next method
            
        if selected_methods:
            results['Final Plan'] = {
                "paces": cumulative_paces,
                "total_time": cumulative_paces.sum()
            }
                      
        return results
    
    def _method_1(self, route_data: pd.DataFrame, paces = None) -> np.ndarray:

        """Slow Uphills - increase pace by 4.912344777209644% on uphills."""

        predicted_paces = paces if paces is not None else route_data['predicted_pace'].values
        adjusted_paces = predicted_paces.copy()
        
        # Calculate net elevation for each segment
        net_elevation_m = route_data['elevation_gain_m'] - route_data['elevation_loss_m']
        
        # Convert segment distance from km to meters
        segment_distance_m = route_data['segment_distance_km'] * 1000
        
        # Calculate gradient as percentage (rise/run * 100)
        gradient_percent = (net_elevation_m / segment_distance_m) * 100
        
        # Classify terrain based on gradient thresholds
        # Uphill: > 1%, Downhill: < -1%, Flat: between -1% and 1%
        is_uphill = gradient_percent > 1.0
        
        # Apply 4.912344777209644% pace increase to uphill segments
        # (slower pace = higher time per km)
        pace_increase_factor = 1.04912344777209644
        adjusted_paces[is_uphill] *= pace_increase_factor
                
        return adjusted_paces
    
    def _method_2(self, route_data: pd.DataFrame, paces = None) -> np.ndarray:

        """Fast Downhills - decrease pace by 4.419284149013879% on downhills."""

        predicted_paces = paces if paces is not None else route_data['predicted_pace'].values
        adjusted_paces = predicted_paces.copy()
        
        # Calculate net elevation for each segment
        net_elevation_m = route_data['elevation_gain_m'] - route_data['elevation_loss_m']
        
        # Convert segment distance from km to meters
        segment_distance_m = route_data['segment_distance_km'] * 1000
        
        # Calculate gradient as percentage (rise/run * 100)
        gradient_percent = (net_elevation_m / segment_distance_m) * 100
        
        # Classify terrain based on gradient thresholds
        # Downhill: < -1%, Uphill: > 1%, Flat: between -1% and 1%
        is_downhill = gradient_percent < -1.0
        
        # Apply 4.419284149013879% pace decrease to downhill segments
        # (faster pace = lower time per km)
        pace_decrease_factor = 0.95580715850986121  # This is 1 - 0.04419284149013879
        adjusted_paces[is_downhill] *= pace_decrease_factor
                
        return adjusted_paces
    
    def _method_3(self, route_data: pd.DataFrame, paces = None) -> np.ndarray:

        """Your third coaching method - replace with your logic."""

        predicted_paces = paces if paces is not None else route_data['predicted_pace'].values
        adjusted_paces = predicted_paces.copy()

        baseline_pace = 5.476 # calculated as the average pace on flats
        decrease_factor = 0.95
        
        mask_faster = adjusted_paces <= baseline_pace # Case 1: faster than or equal to baseline → leave unchanged

        mask_within  = (adjusted_paces > baseline_pace) & (adjusted_paces <= baseline_pace * 1.05) # Case 2: within 5% slower than baseline → set to baseline
        adjusted_paces[mask_within ] = baseline_pace

        # Case 3: more than 5% slower than baseline → speed up by 5%
        mask_slower = adjusted_paces > baseline_pace * 1.05
        adjusted_paces[mask_slower] *= decrease_factor
                
        return adjusted_paces
    
    def _method_4(self, route_data: pd.DataFrame, paces = None) -> np.ndarray:

        """Your fourth coaching method - replace with your logic."""

        predicted_paces = paces if paces is not None else route_data['predicted_pace'].values
        adjusted_paces = predicted_paces.copy()
        
        n = len(adjusted_paces)
        half_index = n // 2 # start of the second half
        decrease_factor = 0.95

        adjusted_paces[half_index:] *= decrease_factor  # 5% faster pace
                
        return adjusted_paces
    
    def _method_5(self, route_data: pd.DataFrame, paces = None, target_time: float = None, time_reduction: float = None) -> np.ndarray:
        
        """
        Fast_Time:
        Adjust predicted paces so that the total run time matches the
        runner's goal, either:
        - run X minutes faster than predicted, OR
        - finish in a specified total time.

        Args:
            route_data: DataFrame with predicted_pace column
            target_time: desired total time in minutes (optional)
            time_reduction: number of minutes faster than predicted (optional)

        Returns:
            np.ndarray of adjusted paces
        """
        predicted_paces = paces if paces is not None else route_data['predicted_pace'].values
        adjusted_paces = predicted_paces.copy()

        num_segments = len(adjusted_paces)
        predicted_total_time = adjusted_paces.sum()

        # Figure out target total time
        if target_time is not None:
            desired_total_time = target_time
        elif time_reduction is not None:
            desired_total_time = predicted_total_time - time_reduction
        else:
            # If no goal given, just return baseline
            return adjusted_paces

        # Safety check
        if desired_total_time <= 0:
            raise ValueError("Desired total time must be positive")

        # Compute adjustment per km
        time_diff = predicted_total_time - desired_total_time
        per_km_adjustment = time_diff / num_segments

        # Apply adjustment equally to all segments
        adjusted_paces -= per_km_adjustment

        return adjusted_paces
    
    def get_available_methods(self) -> Dict[str, str]:
        """Return available coaching methods and their descriptions."""
        return self.available_methods.copy()
    
    def export_coaching_plan(self, results: Dict[str, dict], route_data: pd.DataFrame, output_path: str) -> str:
        """
        Export the coaching results to CSV.
        
        Args:
            results: Dictionary from apply_coaching method
            route_data: Original route data
            output_path: Where to save the CSV
            
        Returns:
            Path to saved file
        """
        # Create output dataframe
        output_df = route_data.copy()
        
        # Add all pace plans
        for method_name, data in results.items():
            paces = data["paces"]
            output_df[f'{method_name}_pace_min_per_km'] = np.round(paces, 2)
            output_df[f'{method_name}_total_time_min'] = paces.cumsum()
        
        # Save to CSV
        output_df.to_csv(output_path, index=False)
        print(f"Coaching plan saved to: {output_path}")
        
        return output_path

    def format_results_for_display(self, results: Dict[str, dict]) -> Dict[str, dict]:
        """Convert results to runner-friendly display format"""
        formatted_results = {}
        
        for method, data in results.items():
            if isinstance(data, dict) and "paces" in data:
                paces = data["paces"]
                total_time = data["total_time"]
                
                formatted_results[method] = {
                    "avg_pace_display": decimal_minutes_to_pace_format(paces.mean()),
                    "total_time_display": decimal_minutes_to_time_format(total_time),
                    "paces_display": [decimal_minutes_to_pace_format(p) for p in paces],
                    "raw_paces": paces,
                    "raw_total_time": total_time
                }
        
        return formatted_results

def decimal_minutes_to_pace_format(decimal_minutes):
    """Convert decimal minutes (5.76) to pace format (5:46)"""
    minutes = int(decimal_minutes)
    seconds = (decimal_minutes - minutes) * 60
    return f"{minutes}:{int(seconds):02d}"

def decimal_minutes_to_time_format(decimal_minutes):
    """Convert decimal minutes (55.81) to time format (0 h 55 min 49 sec)"""
    total_seconds = int(decimal_minutes * 60)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours} h {minutes} min {seconds} sec"

# Simple function for web app integration
def get_coached_paces(route_data: pd.DataFrame, selected_methods: List[str], output_csv_path: str = None, extra_params: Dict[str, dict] = None) -> Dict[str, dict]:
    """
    Simple function to get coached paces.

    Args:
        route_data: Route CSV data as DataFrame
        selected_methods: List of coaching methods chosen by user
        output_csv_path: Optional path to save results
        extra_params: Optional dict of extra parameters for methods
                      Example: {"Fast_Time": {"target_time": 105}}

    Returns:
        Dictionary with coaching results
    """
    coach = SimplePaceCoaching()

    results = coach.apply_coaching(route_data, selected_methods, extra_params)

    if output_csv_path:
        coach.export_coaching_plan(results, route_data, output_csv_path)

    return results

# Example usage
if __name__ == "__main__":
    print("=== Simple Pace Coaching ===")
    
    # Show available methods
    coach = SimplePaceCoaching()
    print("Available coaching methods:")
    for method, description in coach.get_available_methods().items():
        print(f"  {method}: {description}")
    
    # Example usage with actual file path
    CSV_FILE_PATH = "D:\\Most Recent\\PredictedPacePlan.csv"  # This CSV contains route features + predicted_pace_min_per_km column
    OUTPUT_PATH = "D:\\Most Recent\\CoachedPacePlan.csv"
    
    route_data_with_paces = pd.read_csv(CSV_FILE_PATH)
    selected_methods = ['Slower Uphills', 'Faster Downhills']  # User selection from web app
     
    results = get_coached_paces(route_data_with_paces, selected_methods, OUTPUT_PATH)
    
    print("Results:")
    for method, data in results.items():
        if isinstance(data, dict):  # Handle new data structure
            paces = data["paces"]
            total_time = data["total_time"]
            avg_pace = paces.mean()

            # Convert to runner-friendly formats
            avg_pace_formatted = decimal_minutes_to_pace_format(avg_pace)
            total_time_formatted = decimal_minutes_to_time_format(total_time)

            print(f"{method}: Avg pace = {avg_pace_formatted} min/km, Total time = {total_time_formatted}")
        else:
            print(f"{method}: {data}")

    print(f"\nCoaching plan saved to: {OUTPUT_PATH}")
    
    # print("\nTo use this script:")
    # print("1. Replace the method descriptions with your actual method names")
    # print("2. Replace the _method_X functions with your coaching logic")
    # print("3. Use get_coached_paces() in your web app")