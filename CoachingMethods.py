import pandas as pd
import numpy as np
from typing import List, Dict

class SimplePaceCoaching:
    def __init__(self, model_name='Tristan'):
        from coaching_config import get_coaching_params

        self.model_name = model_name
        self.params = get_coaching_params(model_name)

        
        self.available_methods = { # defined coaching methods here
            'Push Uphills': 'Push Uphills',
            'Push Downhills': 'Push Downhills', 
            'Push Flats': 'Push Flats',
            'Negative Splits': 'Negative Splits',
            'Chosen Time': 'Chosen Time'
        }
    
    def apply_coaching(self, route_data: pd.DataFrame, selected_methods: List[str], extra_params: Dict[str, dict] = None) -> Dict[str, np.ndarray]:

        if 'predicted_pace' not in route_data.columns: # extract predicted paces from csv
            raise ValueError("DataFrame must contain 'predicted_pace' column")
    
        predicted_paces = route_data['predicted_pace'].values
        results = {}
        results["Uncoached Pace"] = {
            "paces": predicted_paces.copy(),
            "total_time": predicted_paces.sum()
        }
        
        cumulative_paces = predicted_paces.copy()

        
        for method in selected_methods: # apply each selected method
            friendly_name = self.available_methods.get(method, method)

            if method == 'Push Uphills':
                adjusted = self._method_1(route_data, paces=cumulative_paces)
            elif method == 'Push Downhills':
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

        predicted_paces = paces if paces is not None else route_data['predicted_pace'].values
        adjusted_paces = predicted_paces.copy()
        
        net_elevation_m = route_data['elevation_gain_m'] - route_data['elevation_loss_m'] # net elevation
        
        
        segment_distance_m = route_data['segment_distance_km'] * 1000 # segment distance from km to m
        
        gradient_percent = (net_elevation_m / segment_distance_m) * 100 # gradient as percentage
        
        is_uphill = gradient_percent > 1.0
        
        pace_decrease_factor = self.params['push_uphills_decrease_factor'] # 0.95087655222790356 # this is 1 - 0.04912344777209644
        adjusted_paces[is_uphill] *= pace_decrease_factor
                
        return adjusted_paces
    
    def _method_2(self, route_data: pd.DataFrame, paces = None) -> np.ndarray:

        predicted_paces = paces if paces is not None else route_data['predicted_pace'].values
        adjusted_paces = predicted_paces.copy()
        
        net_elevation_m = route_data['elevation_gain_m'] - route_data['elevation_loss_m'] 
        
        segment_distance_m = route_data['segment_distance_km'] * 1000 
    
        gradient_percent = (net_elevation_m / segment_distance_m) * 100 

        is_downhill = gradient_percent < -1.0

        pace_decrease_factor = self.params['push_downhills_decrease_factor'] # 0.95580715850986121  # This is 1 - 0.04419284149013879
        adjusted_paces[is_downhill] *= pace_decrease_factor
                
        return adjusted_paces
    
    def _method_3(self, route_data: pd.DataFrame, paces = None) -> np.ndarray:

        predicted_paces = paces if paces is not None else route_data['predicted_pace'].values
        adjusted_paces = predicted_paces.copy()

        net_elevation_m = route_data['elevation_gain_m'] - route_data['elevation_loss_m']

        segment_distance_m = route_data['segment_distance_km'] * 1000
        
        gradient_percent = (net_elevation_m / segment_distance_m) * 100
        
        is_flat = (gradient_percent >= -1.0) & (gradient_percent <= 1.0)

        baseline_pace = self.params['push_flats_baseline_pace'] # 5.476, the average pace on flats
        decrease_factor = 0.95
        
        mask_faster = is_flat & (adjusted_paces <= baseline_pace) # faster than or equal to baseline, leave unchanged

        mask_within  = is_flat & (adjusted_paces > baseline_pace) & (adjusted_paces <= baseline_pace * 1.05) # within 5% slower than baseline, set to baseline
        adjusted_paces[mask_within ] = baseline_pace

        
        mask_slower = is_flat & (adjusted_paces > baseline_pace * 1.05) # more than 5% slower than baseline, speed up by 5%
        adjusted_paces[mask_slower] *= decrease_factor
                
        return adjusted_paces
    
    def _method_4(self, route_data: pd.DataFrame, paces = None) -> np.ndarray:

        predicted_paces = paces if paces is not None else route_data['predicted_pace'].values
        adjusted_paces = predicted_paces.copy()
        
        n = len(adjusted_paces)
        half_index = n // 2 # start of the second half
        decrease_factor = 0.95

        adjusted_paces[half_index:] *= decrease_factor  # 5% faster pace
                
        return adjusted_paces
    
    def _method_5(self, route_data: pd.DataFrame, paces = None, target_time: float = None, time_reduction: float = None) -> np.ndarray:
        
        predicted_paces = paces if paces is not None else route_data['predicted_pace'].values
        adjusted_paces = predicted_paces.copy()

        num_segments = len(adjusted_paces)
        predicted_total_time = adjusted_paces.sum()

        
        if target_time is not None: # target total time
            desired_total_time = target_time
        elif time_reduction is not None:
            desired_total_time = predicted_total_time - time_reduction
        else:
            return adjusted_paces

        if desired_total_time <= 0: # check
            raise ValueError("Desired total time must be positive")

        time_diff = predicted_total_time - desired_total_time # compute adjustment per km
        per_km_adjustment = time_diff / num_segments

        adjusted_paces -= per_km_adjustment # appliy adjustment equally to all segments

        return adjusted_paces

    def get_available_methods(self) -> Dict[str, str]:
        return self.available_methods.copy()
    
    def export_coaching_plan(self, results: Dict[str, dict], route_data: pd.DataFrame, output_path: str) -> str:

        output_df = route_data.copy()
        
        
        for method_name, data in results.items(): # add all pace plans
            paces = data["paces"]
            output_df[f'{method_name}_pace_min_per_km'] = np.round(paces, 2)
            output_df[f'{method_name}_total_time_min'] = paces.cumsum()
        
        output_df.to_csv(output_path, index=False)
        print(f"Coaching plan saved to: {output_path}")
        
        return output_path

    def format_results_for_display(self, results: Dict[str, dict]) -> Dict[str, dict]:
        formatted_results = {}
        
        for method, data in results.items():
            if isinstance(data, dict) and "paces" in data:
                paces = data["paces"]
                total_time = data["total_time"]
                total_time_sec = total_time * 60
                
                formatted_results[method] = {
                    "avg_pace_display": decimal_minutes_to_pace_format(paces.mean()),
                    "total_time_display": decimal_minutes_to_time_format(total_time),
                    "paces_display": [decimal_minutes_to_pace_format(p) for p in paces],
                    "raw_paces": paces,
                    "raw_total_time": total_time,
                    "total_time_seconds": total_time_sec
                }
        
        return formatted_results

def decimal_minutes_to_pace_format(decimal_minutes):
    minutes = int(decimal_minutes)
    seconds = (decimal_minutes - minutes) * 60
    return f"{minutes}:{int(seconds):02d}"

def decimal_minutes_to_time_format(decimal_minutes):
    total_seconds = int(decimal_minutes * 60)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours} h {minutes} min {seconds} sec"


def get_coached_paces(route_data: pd.DataFrame, selected_methods: List[str], output_csv_path: str = None, extra_params: Dict[str, dict] = None, model_name: str = 'Tristan') -> Dict[str, dict]:
    coach = SimplePaceCoaching(model_name=model_name)

    results = coach.apply_coaching(route_data, selected_methods, extra_params)

    if output_csv_path:
        coach.export_coaching_plan(results, route_data, output_csv_path)

    return results

if __name__ == "__main__":
    print("=== Simple Pace Coaching ===")
    
    coach = SimplePaceCoaching()
    print("Available coaching methods:")
    for method, description in coach.get_available_methods().items():
        print(f"  {method}: {description}")
    
    CSV_FILE_PATH = "D:\\Most Recent\\PredictedPacePlan.csv"  # CSV contains route features and predicted paces column
    OUTPUT_PATH = "D:\\Most Recent\\CoachedPacePlan.csv"
    
    route_data_with_paces = pd.read_csv(CSV_FILE_PATH)
    selected_methods = ['Slower Uphills', 'Faster Downhills']  # example selection from web app
     
    results = get_coached_paces(route_data_with_paces, selected_methods, OUTPUT_PATH)
    
    print("Results:")
    for method, data in results.items():
        if isinstance(data, dict):  # handle new data structure
            paces = data["paces"]
            total_time = data["total_time"]
            avg_pace = paces.mean()

            avg_pace_formatted = decimal_minutes_to_pace_format(avg_pace)
            total_time_formatted = decimal_minutes_to_time_format(total_time)

            print(f"{method}: Avg pace = {avg_pace_formatted} min/km, Total time = {total_time_formatted}")
        else:
            print(f"{method}: {data}")

    print(f"\nCoaching plan saved to: {OUTPUT_PATH}")