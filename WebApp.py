from flask import Flask, request, render_template, flash, redirect, url_for
import os
import pandas as pd
import numpy as np
import time
from CreatesCSVFromNewRouteGPXFile import process_gpx_route_with_enhanced_features
from StandAlonePacePredictor import PacePredictor
from CoachingMethods import get_coached_paces, SimplePaceCoaching, decimal_minutes_to_pace_format, decimal_minutes_to_time_format
from RunTracker import RunTracker
from flask import session

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_DIR = os.environ.get("MODEL_DIR", "Saved_Models")  # adjust to your model folder

# Add session secret key after app initialization
app.secret_key = 'your-secret-key-here'  # Change this to a random string

# Global tracker storage (in production, use database)
active_trackers = {}

def allowed_file(filename):
    """Check if file has .gpx extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'gpx'

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # --- Step 1: Validate and Upload GPX ---
    if 'gpx_file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    gpx_file = request.files["gpx_file"]
    
    if gpx_file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if not allowed_file(gpx_file.filename):
        flash('Invalid file type. Please upload a .gpx file only.', 'error')
        return redirect(url_for('index'))
    
    gpx_path = os.path.join(UPLOAD_FOLDER, gpx_file.filename)
    gpx_file.save(gpx_path)

    try:
        # --- Step 2: GPX → CSV (features) ---
        route_csv_path = os.path.join(UPLOAD_FOLDER, "New_Route.csv")
        route_df = process_gpx_route_with_enhanced_features(gpx_path, output_path=route_csv_path)

        # --- Step 3: Run model prediction ---
        predictor = PacePredictor(MODEL_DIR)
        predictions, pred_csv = predictor.predict_route(route_csv_path, output_csv_path=os.path.join(UPLOAD_FOLDER, "Predicted_Paces.csv"), create_plots=False)
        route_data = pd.read_csv(pred_csv)

        # --- Step 4: Coaching methods ---
        selected_methods = request.form.getlist("methods")
        target_time = request.form.get("target_time")
        time_reduction = request.form.get("time_reduction")

        # Build extra params if needed
        extra_params = {}
        if "Chosen Time" in selected_methods:
            fast_time_params = {}
            if target_time:
                fast_time_params["target_time"] = float(target_time)
            if time_reduction:
                fast_time_params["time_reduction"] = float(time_reduction)
            extra_params["Chosen Time"] = fast_time_params

        results = get_coached_paces(route_data, selected_methods, output_csv_path=os.path.join(UPLOAD_FOLDER, "Coached_Paces.csv"), extra_params=extra_params)

        # --- Step 5: Build a DataFrame with all results ---

        coach = SimplePaceCoaching()
        formatted_results = coach.format_results_for_display(results)

        # Build per-segment DataFrame (paces formatted as mm:ss)
        results_df = pd.DataFrame({"Segment": route_data["segment_km"]})
        for method_name, data in formatted_results.items():
            results_df[method_name] = data["paces_display"]

        # Build totals summary (formatted as h m s)
        totals_summary = {method: data["total_time_display"] for method, data in formatted_results.items()}
        
        return render_template(
            "results.html", 
            tables=[results_df.to_html(classes="data", index=False, escape = False)], 
            totals=totals_summary
        )
    
    except Exception as e:
        flash(f'Error processing GPX file: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route("/start_tracking", methods=["POST"])
def start_tracking():
    """Initialize tracking page with selected pace plan"""
    selected_plan = request.form.get("selected_plan", "Final Plan")
    
    # Load the coached paces CSV
    coached_csv = os.path.join(UPLOAD_FOLDER, "Coached_Paces.csv")
    route_data = pd.read_csv(coached_csv)

    # DEBUG: Print all column names
    print("=== CSV COLUMNS ===")
    print(route_data.columns.tolist())
    print("===================")
    
    # Create tracker
    tracker = RunTracker(route_data, selected_plan)
    session_id = str(time.time())
    active_trackers[session_id] = tracker
    session['tracker_id'] = session_id
    
    # Get predicted paces for display
    predicted_paces = tracker.predicted_paces.tolist()
    
    return render_template("tracking.html", 
                          total_segments=tracker.total_segments,
                          predicted_paces=predicted_paces)

@app.route("/start_run", methods=["POST"])
def start_run():
    tracker_id = session.get('tracker_id')
    if not tracker_id or tracker_id not in active_trackers:
        return {"error": "No active tracker"}
    
    tracker = active_trackers[tracker_id]
    return tracker.start_run()

@app.route("/log_split", methods=["POST"])
def log_split():
    tracker_id = session.get('tracker_id')
    if not tracker_id or tracker_id not in active_trackers:
        return {"error": "No active tracker"}
    
    tracker = active_trackers[tracker_id]
    return tracker.log_km_split()

@app.route("/run_summary")
def run_summary():
    tracker_id = session.get('tracker_id')
    if not tracker_id or tracker_id not in active_trackers:
        return "No completed run found", 404
    
    tracker = active_trackers[tracker_id]
    summary = tracker.get_run_summary()

    # DEBUG: Print to console to verify data
    print(f"Has coached plan: {summary.get('has_coached_plan')}")
    if summary.get('has_coached_plan'):
        print(f"Uncoached paces available: {'uncoached_paces' in summary}")
        print(f"Coached paces available: {'coached_paces' in summary}")
    
    # Format times and paces for display
    from CoachingMethods import decimal_minutes_to_pace_format, decimal_minutes_to_time_format
    
    # Format total times
    summary['total_actual_time_formatted'] = decimal_minutes_to_time_format(summary['total_actual_time_sec'] / 60)
    summary['total_predicted_time_formatted'] = decimal_minutes_to_time_format(summary['total_predicted_time_sec'] / 60)
    
    # Format difference
    diff_sec = summary['total_difference_sec']
    diff_formatted = f"{'-' if diff_sec < 0 else '+'}{decimal_minutes_to_time_format(abs(diff_sec) / 60)}"
    summary['difference_formatted'] = diff_formatted
    
    # Format average pace
    avg_pace = sum(summary['actual_paces_min']) / len(summary['actual_paces_min'])
    summary['avg_actual_pace'] = decimal_minutes_to_pace_format(avg_pace)
    
    # Format all paces for display in table
    summary['actual_paces_display'] = [decimal_minutes_to_pace_format(p) for p in summary['actual_paces_min']]
    
    # Calculate differences per segment
    differences = []
    differences_display = []
    
    if summary['has_coached_plan']:
        summary['uncoached_paces_display'] = [decimal_minutes_to_pace_format(p) for p in summary['uncoached_paces']]
        summary['coached_paces_display'] = [decimal_minutes_to_pace_format(p) for p in summary['coached_paces']]
        
        # Calculate differences against coached plan
        for i, actual in enumerate(summary['actual_paces_min']):
            coached = summary['coached_paces'][i]
            diff = (actual - coached) * 60  # Convert to seconds
            differences.append(diff)
            differences_display.append(f"{'-' if diff < 0 else '+'}{abs(int(diff))}s")
    else:
        summary['predicted_paces_display'] = [decimal_minutes_to_pace_format(p) for p in summary['predicted_paces_min']]
        
        # Calculate differences against prediction
        for i, actual in enumerate(summary['actual_paces_min']):
            predicted = summary['predicted_paces_min'][i]
            diff = (actual - predicted) * 60  # Convert to seconds
            differences.append(diff)
            differences_display.append(f"{'-' if diff < 0 else '+'}{abs(int(diff))}s")
    
    summary['differences'] = differences
    summary['differences_display'] = differences_display
    summary['segments'] = list(range(1, len(summary['actual_paces_min']) + 1))

    # Add elevation data for the elevation chart
    coached_csv = os.path.join(UPLOAD_FOLDER, "Coached_Paces.csv")
    route_data = pd.read_csv(coached_csv)
    summary['elevation_gain'] = route_data['elevation_gain_m'].tolist()
    summary['elevation_loss'] = route_data['elevation_loss_m'].tolist()
    
    return render_template("summary.html", summary=summary)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
