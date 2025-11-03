import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

class PacePredictor:
    def __init__(self, model_dir):

        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.metadata = None
        
        self._load_model_components()
    
    def _load_model_components(self):

        try:
            
            metadata_files = [f for f in os.listdir(self.model_dir) if f.endswith('_metadata.json')] #detect model type by finding metadata file
            
            if not metadata_files:
                raise FileNotFoundError(f"No metadata file found in {self.model_dir}")
            
            
            metadata_filename = metadata_files[0] # use the first metadata file found
            model_type = metadata_filename.replace('_metadata.json', '')
            
            
            metadata_path = os.path.join(self.model_dir, metadata_filename) # load metadata
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            
            model_path = os.path.join(self.model_dir, f'{model_type}_model.pkl') # load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            
            scaler_path = os.path.join(self.model_dir, f'{model_type}_scaler.pkl') # load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            
            self.feature_cols = self.metadata['feature_cols'] # extract feature columns
            
            print("Model loaded")
            print(f"Model: {self.metadata['model_name']}")
            print(f"Test MAE: {self.metadata['test_mae']:.4f} min/km")
            print(f"Test RMSE: {self.metadata['test_rmse']:.4f} min/km")
            print(f"Test R2: {self.metadata['test_r2']:.4f}")
            print(f"Features: {len(self.feature_cols)}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find model files in {self.model_dir}. "
                                f"Error: {e}")
        except Exception as e:
            raise Exception(f"Error loading model components: {e}")
    
    def predict_route(self, route_csv_path, output_csv_path=None, create_plots=True):

        print(f"PACE PREDICTION")
        
        route_df = pd.read_csv(route_csv_path) # load route data
        
        
        missing_features = set(self.feature_cols) - set(route_df.columns) # validate features
        if missing_features:
            raise ValueError(f"Missing required features in route data: {missing_features}")
        
        
        X_route = route_df[self.feature_cols]
        X_route_scaled = self.scaler.transform(X_route)
        
        
        predictions = self.model.predict(X_route_scaled) # make predictions
        
        avg_pace = predictions.mean()
        total_time_min = predictions.sum()
        total_hours = int(total_time_min // 60)
        total_minutes = int(total_time_min % 60)
        pace_range = (predictions.min(), predictions.max())
        
        print(f"\nSUMMARY:")
        print(f"Average pace: {avg_pace:.2f} min/km")
        print(f"Total estimated time: {total_time_min:.1f} minutes ({total_hours}:{total_minutes:02d})")
        print(f"Pace range: {pace_range[0]:.2f} to {pace_range[1]:.2f} min/km")
        print(f"Model accuracy (Test MAE): +-{self.metadata['test_mae']:.4f} min/km")
        
        output_df = route_df.copy() # output dataframe with predictions
        output_df['predicted_pace'] = predictions
        
        if output_csv_path is None:
            raise ValueError("You must provide an output_csv_path for saving results.")
        
        
        output_df.to_csv(output_csv_path, index=False)# save results to csv 
        print(f"\nPredictions saved to: {output_csv_path}")
        
        preview_cols = ['segment_km', 'predicted_pace'] \
        if 'segment_km' in output_df.columns else ['predicted_pace']
        print(f"\nPreview of predictions:")
        print(output_df[preview_cols].head(10).round(3))
        
        if create_plots:
            self._create_prediction_plots(route_df, predictions)
        
        return predictions, output_csv_path
    
    def _create_prediction_plots(self, route_df, predictions):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        
        if 'segment_km' in route_df.columns: # segment_km if available, otherwise use index
            segments = route_df['segment_km']
            xlabel = 'Distance (km)'
        else:
            segments = range(len(route_df))
            xlabel = 'Segment Index'
        
        mae = self.metadata['test_mae']
        
        ax1.plot(segments, predictions, marker='o', linewidth=2, color='blue', 
                label=f'Predicted Pace', markersize=6)
        
        ax1.fill_between(segments, 
                        predictions - mae, 
                        predictions + mae, 
                        alpha=0.3, color='blue', 
                        label=f'±{mae:.3f} min/km')
        
        ax1.axhspan(3, 8, alpha=0.1, color='green', label='Typical pace range')
        
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Predicted Pace (min/km)')
        ax1.set_title('Pace Prediction Plan')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if 'elevation_gain_m' in route_df.columns and 'elevation_loss_m' in route_df.columns:
            net_elevation = route_df['elevation_gain_m'] - route_df['elevation_loss_m']
            ax2.bar(segments, net_elevation, alpha=0.7, color='green', label='Net Elevation')
            ax2.set_ylabel('Net Elevation Change (m)')
            ax2.set_title('Route Elevation Profile')
        elif 'elevation_gain_m' in route_df.columns:
            ax2.bar(segments, route_df['elevation_gain_m'], alpha=0.7, color='green', 
                   label='Elevation Gain')
            ax2.set_ylabel('Elevation Gain (m)')
            ax2.set_title('Route Elevation Profile')
        else:
            
            cumulative_time = predictions.cumsum()
            ax2.plot(segments, cumulative_time, color='orange', marker='s',
                    label='Cumulative Time')
            ax2.set_ylabel('Cumulative Time (minutes)')
            ax2.set_title('Cumulative Time Profile')
        
        ax2.set_xlabel(xlabel)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        
        self._print_detailed_breakdown(route_df, predictions) # detailed segment breakdown
    
    def _print_detailed_breakdown(self, route_df, predictions):

        print(f"DETAILED SEGMENT BREAKDOWN")

        
        mae = self.metadata['test_mae']
        cumulative_time = predictions.cumsum()
        
        
        header = "Seg"
        if 'elevation_gain_m' in route_df.columns and 'elevation_loss_m' in route_df.columns: # header
            header += " | Net Elev | Pace | Range      | Cum Time"
        elif 'elevation_gain_m' in route_df.columns:
            header += " | Elev Gain| Pace | Range      | Cum Time"
        else:
            header += " | Pace | Range      | Cum Time"
            
        print(header)
        print("-" * len(header))
        
        for i in range(len(route_df)):
            seg = route_df.iloc[i]
            
            if 'segment_km' in route_df.columns:
                line = f" {int(seg['segment_km']):2d} "
            else:
                line = f" {i:2d} "
            
            
            if 'elevation_gain_m' in route_df.columns and 'elevation_loss_m' in route_df.columns: # elevation info
                net_elev = seg['elevation_gain_m'] - seg['elevation_loss_m']
                line += f"| {net_elev:+5.0f}m  "
            elif 'elevation_gain_m' in route_df.columns:
                line += f"| {seg['elevation_gain_m']:5.0f}m  "
            

            pace = predictions[i] #  pace and range
            pace_min = pace - mae
            pace_max = pace + mae
            line += f"| {pace:4.2f} | {pace_min:4.2f}-{pace_max:4.2f} "
            
            
            cum_time = cumulative_time[i] # cumulative time
            hours = int(cum_time // 60)
            minutes = int(cum_time % 60)
            line += f"| {hours:2d}:{minutes:02d}"
            
            print(line)
    
    def get_model_info(self):
        return {
            'model_name': self.metadata['model_name'],
            'test_mae': self.metadata['test_mae'],
            'test_rmse': self.metadata['test_rmse'],
            'test_r2': self.metadata['test_r2'],
            'features': self.feature_cols,
            'num_features': len(self.feature_cols)
        }

def main():
    parser = argparse.ArgumentParser(description='Predict running pace for a new route')
    parser.add_argument('route_csv', help='Path to CSV file containing route data')
    parser.add_argument('--model-dir', default='./saved_models', 
                       help='Directory containing saved model files')
    parser.add_argument('--output', help='Output CSV file path (optional)')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Skip creating visualization plots')
    
    args = parser.parse_args()
    
    try:
        
        predictor = PacePredictor(args.model_dir)
        
        
        predictions, output_path = predictor.predict_route( # make predictions
            args.route_csv, 
            args.output, 
            create_plots=not args.no_plots
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":

    import sys
    
    if len(sys.argv) > 1:
 
        main()
    else:

        MODEL_DIR = r'D:\Most Recent\Skripsie\FINAL WEB APP PIPELINE\Saved_Models'
        ROUTE_CSV = r'D:\Most Recent\NewRoute.csv'
        
        try:

            predictor = PacePredictor(MODEL_DIR)
            

            info = predictor.get_model_info()
            print(f"\nModel Information:")
            for key, value in info.items():
                if key != 'features':
                    print(f"  {key}: {value}")
            

            predictions, output_path = predictor.predict_route(ROUTE_CSV, output_csv_path = "D:\\Most Recent\\PredictedPacePlan.csv")
            
        except Exception as e:
            print(f"Error: {e}")