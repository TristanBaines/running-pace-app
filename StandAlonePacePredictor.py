"""
Standalone Pace Predictor
Uses a pre-trained Linear Regression model to predict running pace for new routes.
"""

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
        """
        Initialize the pace predictor by loading the saved model components.
        
        Args:
            model_dir (str): Directory containing the saved model files
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.metadata = None
        
        self._load_model_components()
    
    def _load_model_components(self):
        """Load the trained model, scaler, and metadata."""
        try:
            # Auto-detect model type by finding metadata file
            metadata_files = [f for f in os.listdir(self.model_dir) if f.endswith('_metadata.json')]
            
            if not metadata_files:
                raise FileNotFoundError(f"No metadata file found in {self.model_dir}")
            
            # Use the first metadata file found
            metadata_filename = metadata_files[0]
            model_type = metadata_filename.replace('_metadata.json', '')
            
            # Load metadata
            metadata_path = os.path.join(self.model_dir, metadata_filename)
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Load model
            model_path = os.path.join(self.model_dir, f'{model_type}_model.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, f'{model_type}_scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Extract feature columns
            self.feature_cols = self.metadata['feature_cols']
            
            print("Model loaded successfully!")
            print(f"Model: {self.metadata['model_name']}")
            print(f"Test MAE: {self.metadata['test_mae']:.4f} min/km")
            print(f"Test RMSE: {self.metadata['test_rmse']:.4f} min/km")
            print(f"Test R²: {self.metadata['test_r2']:.4f}")
            print(f"Features: {len(self.feature_cols)}")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find model files in {self.model_dir}. "
                                f"Make sure you've saved the model first. Error: {e}")
        except Exception as e:
            raise Exception(f"Error loading model components: {e}")
    
    def predict_route(self, route_csv_path, output_csv_path=None, create_plots=True):
        """
        Predict pace for a new route and save results.
        
        Args:
            route_csv_path (str): Path to CSV file containing route data
            output_csv_path (str): Optional path for output CSV
            create_plots (bool): Whether to create visualization plots
            
        Returns:
            tuple: (predictions_array, output_csv_path)
        """
        print(f"\n{'='*60}")
        print(f"PACE PREDICTION FOR NEW ROUTE")
        print(f"{'='*60}")
        
        # Load route data
        print(f"Loading route data from: {route_csv_path}")
        route_df = pd.read_csv(route_csv_path)
        print(f"Route segments: {len(route_df)}")
        
        # Validate features
        missing_features = set(self.feature_cols) - set(route_df.columns)
        if missing_features:
            raise ValueError(f"Missing required features in route data: {missing_features}")
        
        # Prepare features
        X_route = route_df[self.feature_cols]
        X_route_scaled = self.scaler.transform(X_route)
        
        # Make predictions
        print("\nMaking pace predictions...")
        predictions = self.model.predict(X_route_scaled)
        
        # Calculate summary statistics
        avg_pace = predictions.mean()
        total_time_min = predictions.sum()
        total_hours = int(total_time_min // 60)
        total_minutes = int(total_time_min % 60)
        pace_range = (predictions.min(), predictions.max())
        
        print(f"\nPREDICTION SUMMARY:")
        print(f"Average pace: {avg_pace:.2f} min/km")
        print(f"Total estimated time: {total_time_min:.1f} minutes ({total_hours}:{total_minutes:02d})")
        print(f"Pace range: {pace_range[0]:.2f} to {pace_range[1]:.2f} min/km")
        print(f"Model accuracy (Test MAE): +-{self.metadata['test_mae']:.4f} min/km")
        
        # Validate predictions
        if predictions.min() < 0 or predictions.max() > 15:
            print("WARNING: Some predictions seem unrealistic!")
        else:
            print("Predictions appear reasonable.")
        
        # Create enhanced output dataframe
        output_df = route_df.copy()
        output_df['predicted_pace'] = predictions
        
        # Add confidence intervals based on test MAE
        #mae = self.metadata['test_mae']
        #output_df['pace_lower_bound'] = predictions - mae
        #output_df['pace_upper_bound'] = predictions + mae
        
        # Add cumulative time
        #output_df['cumulative_time_min'] = predictions.cumsum()
        
        # Generate output path if not provided
        if output_csv_path is None:
            raise ValueError("You must provide an output_csv_path for saving results.")
            #input_dir = os.path.dirname(route_csv_path)
            #input_filename = os.path.basename(route_csv_path)
            #input_name, input_ext = os.path.splitext(input_filename)
            #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            #output_csv_path = os.path.join(input_dir, f"{input_name}_pace_predictions_{timestamp}{input_ext}")
        
        # Save results
        output_df.to_csv(output_csv_path, index=False)
        print(f"\nPredictions saved to: {output_csv_path}")
        
        # Show preview
        preview_cols = ['segment_km', 'predicted_pace'] \
        if 'segment_km' in output_df.columns else ['predicted_pace']
        print(f"\nPreview of predictions:")
        print(output_df[preview_cols].head(10).round(3))
        
        # Create plots if requested
        if create_plots:
            self._create_prediction_plots(route_df, predictions)
        
        return predictions, output_csv_path
    
    def _create_prediction_plots(self, route_df, predictions):
        """Create visualization plots for the pace predictions."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Use segment_km if available, otherwise use index
        if 'segment_km' in route_df.columns:
            segments = route_df['segment_km']
            xlabel = 'Distance (km)'
        else:
            segments = range(len(route_df))
            xlabel = 'Segment Index'
        
        # Plot 1: Pace predictions with confidence interval
        mae = self.metadata['test_mae']
        
        ax1.plot(segments, predictions, marker='o', linewidth=2, color='blue', 
                label=f'Predicted Pace', markersize=6)
        
        # Add confidence interval
        ax1.fill_between(segments, 
                        predictions - mae, 
                        predictions + mae, 
                        alpha=0.3, color='blue', 
                        label=f'±{mae:.3f} min/km')
        
        # Add reasonable pace range shading
        ax1.axhspan(3, 8, alpha=0.1, color='green', label='Typical pace range')
        
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Predicted Pace (min/km)')
        ax1.set_title('Pace Prediction Plan')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Elevation profile (if available)
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
            # If no elevation data, show cumulative time instead
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
        
        # Print detailed segment breakdown
        self._print_detailed_breakdown(route_df, predictions)
    
    def _print_detailed_breakdown(self, route_df, predictions):
        """Print detailed segment-by-segment breakdown."""
        print(f"\n{'='*60}")
        print(f"DETAILED SEGMENT BREAKDOWN")
        print(f"{'='*60}")
        
        mae = self.metadata['test_mae']
        cumulative_time = predictions.cumsum()
        
        # Create header
        header = "Seg"
        if 'elevation_gain_m' in route_df.columns and 'elevation_loss_m' in route_df.columns:
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
            
            # Add elevation info if available
            if 'elevation_gain_m' in route_df.columns and 'elevation_loss_m' in route_df.columns:
                net_elev = seg['elevation_gain_m'] - seg['elevation_loss_m']
                line += f"| {net_elev:+5.0f}m  "
            elif 'elevation_gain_m' in route_df.columns:
                line += f"| {seg['elevation_gain_m']:5.0f}m  "
            
            # Add pace and range
            pace = predictions[i]
            pace_min = pace - mae
            pace_max = pace + mae
            line += f"| {pace:4.2f} | {pace_min:4.2f}-{pace_max:4.2f} "
            
            # Add cumulative time
            cum_time = cumulative_time[i]
            hours = int(cum_time // 60)
            minutes = int(cum_time % 60)
            line += f"| {hours:2d}:{minutes:02d}"
            
            print(line)
    
    def get_model_info(self):
        """Return information about the loaded model."""
        return {
            'model_name': self.metadata['model_name'],
            'test_mae': self.metadata['test_mae'],
            'test_rmse': self.metadata['test_rmse'],
            'test_r2': self.metadata['test_r2'],
            'features': self.feature_cols,
            'num_features': len(self.feature_cols)
        }

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Predict running pace for a new route')
    parser.add_argument('route_csv', help='Path to CSV file containing route data')
    parser.add_argument('--model-dir', default='./saved_models', 
                       help='Directory containing saved model files')
    parser.add_argument('--output', help='Output CSV file path (optional)')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Skip creating visualization plots')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = PacePredictor(args.model_dir)
        
        # Make predictions
        predictions, output_path = predictor.predict_route(
            args.route_csv, 
            args.output, 
            create_plots=not args.no_plots
        )
        
        print(f"\nPrediction completed successfully!")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

# Example usage when run directly
if __name__ == "__main__":
    # You can either use command line arguments or run directly
    import sys
    
    if len(sys.argv) > 1:
        # Command line usage
        main()
    else:
        # Direct usage example
        print("=== Standalone Pace Predictor ===")
        
        # Update these paths
        MODEL_DIR = r'D:\Most Recent\Skripsie\FINAL WEB APP PIPELINE\Saved_Models'
        ROUTE_CSV = r'D:\Most Recent\NewRoute.csv'
        
        try:
            # Initialize predictor
            predictor = PacePredictor(MODEL_DIR)
            
            # Show model info
            info = predictor.get_model_info()
            print(f"\nModel Information:")
            for key, value in info.items():
                if key != 'features':
                    print(f"  {key}: {value}")
            
            # Make predictions
            predictions, output_path = predictor.predict_route(ROUTE_CSV, output_csv_path = "D:\\Most Recent\\PredictedPacePlan.csv")
            
        except Exception as e:
            print(f"Error: {e}")
            print("\nMake sure you've:")
            print("1. Run the training script first to create the saved model")
            print("2. Updated the MODEL_DIR and ROUTE_CSV paths above")