from flask import Flask, request, render_template, flash, redirect, url_for, session
import os
import pandas as pd
import numpy as np
import time
from CreatesCSVFromNewRouteGPXFile import process_gpx_route_with_enhanced_features
from StandAlonePacePredictor import PacePredictor
from CoachingMethods import get_coached_paces, SimplePaceCoaching, decimal_minutes_to_pace_format, decimal_minutes_to_time_format
from RunTracker import RunTracker
from Analytics import PerformanceAnalytics

app = Flask(__name__)
app.jinja_env.globals.update(abs=abs) # allows the use of the abs() function in any templates
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

    # Get selected model
    selected_model = request.form.get("selected_model", "Tristan")

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

        # --- Step 3: Run model prediction with selected model ---
        model_dir = os.path.join(MODEL_DIR, selected_model)
        predictor = PacePredictor(model_dir)
        predictions, pred_csv = predictor.predict_route(route_csv_path, output_csv_path=os.path.join(UPLOAD_FOLDER, "Predicted_Paces.csv"), create_plots=False)
        route_data = pd.read_csv(pred_csv)

        # --- Step 4: Coaching methods ---
        selected_methods = request.form.getlist("methods")
        session['coaching_methods'] = selected_methods # for saving the chosen coaching methods
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

        results = get_coached_paces(route_data, selected_methods, output_csv_path=os.path.join(UPLOAD_FOLDER, "Coached_Paces.csv"), extra_params=extra_params, model_name=selected_model)

        # --- Step 5: Build a DataFrame with all results ---

        coach = SimplePaceCoaching()
        formatted_results = coach.format_results_for_display(results)

        # Build per-segment DataFrame (paces formatted as mm:ss)
        results_df = pd.DataFrame({"Segment": route_data["segment_km"]})
        for method_name, data in formatted_results.items():
            results_df[method_name] = data["paces_display"]

        # Build totals summary (formatted as h m s)
        totals_summary = {method: data["total_time_display"] for method, data in formatted_results.items()}
        
        notes = []

        # Check if there are any uphill or downhill segments
        if "Push Uphills" in selected_methods:
            net_elev = route_data['elevation_gain_m'] - route_data['elevation_loss_m']
            gradient = (net_elev / (route_data['segment_distance_km'] * 1000)) * 100
            if not (gradient > 1).any():
                notes.append("Note: The 'Push Uphills' method was selected, but there are no uphill segments on this route.")

        if "Push Downhills" in selected_methods:
            net_elev = route_data['elevation_gain_m'] - route_data['elevation_loss_m']
            gradient = (net_elev / (route_data['segment_distance_km'] * 1000)) * 100
            if not (gradient < -1).any():
                notes.append("Note: The 'Push Downhills' method was selected, but there are no downhill segments on this route.")

        if "Push Flats" in selected_methods:
            previous_method = None
            for method in selected_methods:
                if method == "Push Flats":
                    break
                previous_method = method
            
            if previous_method:
                push_flats_paces = results["Push Flats"]["paces"]
                prev_method_paces = results[previous_method]["paces"]

                if np.allclose(push_flats_paces, prev_method_paces, atol=1e-4):
                    notes.append("Note: The 'Push Flats' method resulted in no changes — you are already running faster than your baseline pace.")
        
        total_distance = round(route_data["segment_distance_km"].sum(), 2)

        # Filter totals to only show Uncoached and Final plans
        filtered_totals = {k: v for k, v in totals_summary.items() if k in ["Uncoached Pace", "Final Plan"]}
        
        return render_template(
            "results.html", 
            tables=[results_df.to_html(classes="data", index=False, escape = False)], 
            totals=filtered_totals,
            total_distance=total_distance,
            notes=notes
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

    if "Final Plan_pace_min_per_km" not in route_data.columns:
        selected_plan = "Uncoached Pace"

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

    # Get elevation data for display
    elevation_gains = route_data['elevation_gain_m'].tolist()
    elevation_losses = route_data['elevation_loss_m'].tolist()

    segment_distances = route_data['segment_distance_km'].tolist()
    
    return render_template("tracking.html", 
                          total_segments=tracker.total_segments,
                          predicted_paces=predicted_paces,
                          elevation_gains=elevation_gains,
                          elevation_losses=elevation_losses,
                          segment_distances=route_data['segment_distance_km'].tolist(),
                          total_distance=round(route_data['segment_distance_km'].sum(), 2)
    )


@app.route("/pause_run", methods=["POST"])
def pause_run():
    tracker_id = session.get('tracker_id')
    if not tracker_id or tracker_id not in active_trackers:
        return {"error": "No active tracker"}
    return active_trackers[tracker_id].pause_run()

@app.route("/resume_run", methods=["POST"])
def resume_run():
    tracker_id = session.get('tracker_id')
    if not tracker_id or tracker_id not in active_trackers:
        return {"error": "No active tracker"}
    return active_trackers[tracker_id].resume_run()

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

@app.route("/deeper_analytics")
def deeper_analytics():
    """Deep analytics page comparing actual performance to coached plan"""
    tracker_id = session.get('tracker_id')
    if not tracker_id or tracker_id not in active_trackers:
        return "No completed run found", 404
    
    tracker = active_trackers[tracker_id]
    
    # Get basic summary data
    summary = tracker.get_run_summary()
    
    # Load route data for elevation and coaching info
    coached_csv = os.path.join(UPLOAD_FOLDER, "Coached_Paces.csv")
    route_data = pd.read_csv(coached_csv)
    
    # Extract data for analytics
    actual_paces = summary['actual_paces_min']
    
    if summary['has_coached_plan']:
        coached_paces = summary['coached_paces']
        uncoached_paces = summary['uncoached_paces']
    else:
        coached_paces = summary['predicted_paces_min']
        uncoached_paces = summary['predicted_paces_min']
    
    elevation_gains = route_data['elevation_gain_m'].tolist()
    elevation_losses = route_data['elevation_loss_m'].tolist()
    
    # Get coaching methods from session or form data
    # You'll need to store this when the user selects methods
    coaching_methods = session.get('coaching_methods', [])
    
    # Create analytics engine
    analytics = PerformanceAnalytics(
        actual_paces=actual_paces,
        coached_paces=coached_paces,
        uncoached_paces=uncoached_paces,
        elevation_gains=elevation_gains,
        elevation_losses=elevation_losses,
        coaching_methods=coaching_methods,
        segment_distances=route_data['segment_distance_km'].tolist()
    )
    
    # Generate full report
    analytics_report = analytics.get_full_analytics_report()
    
    # Format for display
    from CoachingMethods import decimal_minutes_to_pace_format
    
    # Format terrain analysis
    if analytics_report['terrain_analysis']['uphill']:
        for key in ['avg_actual_pace', 'avg_coached_pace']:
            analytics_report['terrain_analysis']['uphill'][key + '_display'] = \
                decimal_minutes_to_pace_format(analytics_report['terrain_analysis']['uphill'][key])
    
    if analytics_report['terrain_analysis']['downhill']:
        for key in ['avg_actual_pace', 'avg_coached_pace']:
            analytics_report['terrain_analysis']['downhill'][key + '_display'] = \
                decimal_minutes_to_pace_format(analytics_report['terrain_analysis']['downhill'][key])
    
    if analytics_report['terrain_analysis']['flat']:
        for key in ['avg_actual_pace', 'avg_coached_pace']:
            analytics_report['terrain_analysis']['flat'][key + '_display'] = \
                decimal_minutes_to_pace_format(analytics_report['terrain_analysis']['flat'][key])
    
    # Format split analysis
    for key in ['first_half_avg_actual', 'second_half_avg_actual', 
                'first_half_avg_coached', 'second_half_avg_coached']:
        analytics_report['split_analysis'][key + '_display'] = \
            decimal_minutes_to_pace_format(analytics_report['split_analysis'][key])
    
    # Format best/worst segments
    for segment in analytics_report['best_worst_segments']['best_segments']:
        segment['actual_pace_display'] = decimal_minutes_to_pace_format(segment['actual_pace'])
        segment['coached_pace_display'] = decimal_minutes_to_pace_format(segment['coached_pace'])
    
    for segment in analytics_report['best_worst_segments']['worst_segments']:
        segment['actual_pace_display'] = decimal_minutes_to_pace_format(segment['actual_pace'])
        segment['coached_pace_display'] = decimal_minutes_to_pace_format(segment['coached_pace'])
    
    # Add segment-level data for charts
    analytics_report['segments'] = list(range(1, len(actual_paces) + 1))
    analytics_report['actual_paces'] = actual_paces
    analytics_report['coached_paces'] = coached_paces
    analytics_report['net_elevation'] = (np.array(elevation_gains) - np.array(elevation_losses)).tolist()
    
    return render_template("analytics.html", analytics=analytics_report)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
