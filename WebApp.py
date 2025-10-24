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

MODEL_DIR = os.environ.get("MODEL_DIR", "Saved_Models")

app.secret_key = 'secret-key'

active_trackers = {} # global tracker for storage in memory

def allowed_file(filename): # only allow gpx files
    """Check if file has .gpx extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'gpx'

@app.route("/") # home page
def index():
    return render_template("index.html")







@app.route("/predict", methods=["POST"]) # generates the pace plans using the trained model
def predict():

    selected_model = request.form.get("selected_model", "Tristan") # loads selected model

    if 'gpx_file' not in request.files: # validate and upload GPX file and not allow other file types
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
    gpx_file.save(gpx_path) # save file to uploads folder

    try:
        route_csv_path = os.path.join(UPLOAD_FOLDER, "New_Route.csv") # generate the csv file with the route features from the gpx file, ready for prediction
        route_df = process_gpx_route_with_enhanced_features(gpx_path, output_path=route_csv_path)

        model_dir = os.path.join(MODEL_DIR, selected_model) # predict paces using loaded trained model
        predictor = PacePredictor(model_dir)
        predictions, pred_csv = predictor.predict_route(route_csv_path, output_csv_path=os.path.join(UPLOAD_FOLDER, "Predicted_Paces.csv"), create_plots=False)
        route_data = pd.read_csv(pred_csv) # load saved csv with predicted paces

        selected_methods = request.form.getlist("methods") # get coaching methods
        session['coaching_methods'] = selected_methods # for saving the chosen coaching methods
        target_time = request.form.get("target_time") # for chosen time coaching method
        time_reduction = request.form.get("time_reduction")

        extra_params = {} # extra parameters for chosen time coaching method
        if "Chosen Time" in selected_methods:
            fast_time_params = {}
            if target_time:
                fast_time_params["target_time"] = float(target_time)
            if time_reduction:
                fast_time_params["time_reduction"] = float(time_reduction)
            extra_params["Chosen Time"] = fast_time_params

        results = get_coached_paces(route_data, selected_methods, output_csv_path=os.path.join(UPLOAD_FOLDER, "Coached_Paces.csv"), extra_params=extra_params, model_name=selected_model) # get the coached paces after applying coaching methods to predicted paces

        coach = SimplePaceCoaching() # calling coaching class
        formatted_results = coach.format_results_for_display(results) # format results for better viewing

        session['plan_total_times'] = {
            method: data["total_time_seconds"] for method, data in formatted_results.items() # for finishing times
        }

        results_df = pd.DataFrame({"Segment": route_data["segment_km"]}) # dataframe for paces per coaching method per segment
        for method_name, data in formatted_results.items():
            results_df[method_name] = data["paces_display"]

        
        totals_summary = {method: data["total_time_display"] for method, data in formatted_results.items()} # totals summary formatted as h m s

        plan_total_times = {}
        for method, fdata in formatted_results.items():
            total_seconds = 0

            
            if method in results and isinstance(results[method], dict) and 'total_time' in results[method]: # numeric total from the coaching results
                try:
                    total_seconds = float(results[method]['total_time']) * 60.0
                except Exception:
                    total_seconds = 0

            
            elif isinstance(fdata, dict) and 'raw_total_time' in fdata: # if format_results_for_display returned a raw total time in minutes
                try:
                    total_seconds = float(fdata['raw_total_time']) * 60.0
                except Exception:
                    total_seconds = 0

            
            elif isinstance(fdata, dict) and 'total_time_seconds' in fdata: # if it provided total_time_seconds already
                try:
                    total_seconds = float(fdata['total_time_seconds'])
                except Exception:
                    total_seconds = 0

            plan_total_times[method] = total_seconds

        session['plan_total_times'] = plan_total_times


        
        notes = []

        
        if "Push Uphills" in selected_methods: # check for any uphill or downhill segments
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

        
        filtered_totals = {k: v for k, v in totals_summary.items() if k in ["Uncoached Pace", "Final Plan"]} # filter totals to only show Uncoached and Final plans
        
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
    
    
    coached_csv = os.path.join(UPLOAD_FOLDER, "Coached_Paces.csv") # load the coached paces CSV
    route_data = pd.read_csv(coached_csv)

    plan_totals = session.get('plan_total_times', {})
    selected_plan_time_sec = None

    if selected_plan in plan_totals:
        selected_plan_time_sec = plan_totals[selected_plan]

    session['selected_plan'] = selected_plan
    session['selected_plan_time_sec'] = selected_plan_time_sec

    print(f"[DEBUG] Selected plan: {selected_plan}, Total predicted time (sec): {selected_plan_time_sec}")



    
    if selected_plan == "Final Plan" and "Final Plan_pace_min_per_km" in route_data.columns: # store selected plan and its total predicted time
        total_pred_time = sum(route_data["Final Plan_pace_min_per_km"] * route_data["segment_distance_km"]) * 60
    else:
        total_pred_time = sum(route_data["Uncoached Pace_pace_min_per_km"] * route_data["segment_distance_km"]) * 60

    session['selected_plan'] = selected_plan
    session['total_predicted_time_sec'] = total_pred_time

    if "Final Plan_pace_min_per_km" not in route_data.columns:
        selected_plan = "Uncoached Pace"

    
    print("=== CSV COLUMNS ===") # DEBUGGING
    print(route_data.columns.tolist())
    print("===================")
    
    
    tracker = RunTracker(route_data, selected_plan) # creates tracker
    session_id = str(time.time())
    active_trackers[session_id] = tracker
    session['tracker_id'] = session_id
    
    
    predicted_paces = tracker.predicted_paces.tolist() # predicted paces for display

    
    elevation_gains = route_data['elevation_gain_m'].tolist() # elevation data for display
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

    
    stored_pred_time = session.get('selected_plan_time_sec') # override predicted time with the one saved in session

    if stored_pred_time is not None:
        
        try: # value should already be in seconds
            summary['total_predicted_time_sec'] = float(stored_pred_time)
        except Exception: # leave whatever get_run_summary provided
            
            pass

    if 'total_actual_time_sec' in summary and 'total_predicted_time_sec' in summary:
        try:
            summary['total_difference_sec'] = summary['total_actual_time_sec'] - summary['total_predicted_time_sec']
        except Exception:
            # If something unexpected - leave existing value
            pass



    
    print(f"Has coached plan: {summary.get('has_coached_plan')}") # DEBUGGING
    if summary.get('has_coached_plan'):
        print(f"Uncoached paces available: {'uncoached_paces' in summary}")
        print(f"Coached paces available: {'coached_paces' in summary}")
    
    from CoachingMethods import decimal_minutes_to_pace_format, decimal_minutes_to_time_format
    
    
    summary['total_actual_time_formatted'] = decimal_minutes_to_time_format(summary['total_actual_time_sec'] / 60) # format total times
    summary['total_predicted_time_formatted'] = decimal_minutes_to_time_format(summary['total_predicted_time_sec'] / 60)

    
    diff_sec = summary['total_difference_sec']
    diff_formatted = f"{'-' if diff_sec < 0 else '+'}{decimal_minutes_to_time_format(abs(diff_sec) / 60)}" # format difference
    summary['difference_formatted'] = diff_formatted
    
    
    avg_pace = sum(summary['actual_paces_min']) / len(summary['actual_paces_min']) # format average pace
    summary['avg_actual_pace'] = decimal_minutes_to_pace_format(avg_pace)
    
    
    summary['actual_paces_display'] = [decimal_minutes_to_pace_format(p) for p in summary['actual_paces_min']] # format all paces for display in table
    
    
    differences = [] # differences per segment
    differences_display = []
    segment_distances = summary['segment_distances']
    
    if summary['has_coached_plan']:
        summary['uncoached_paces_display'] = [decimal_minutes_to_pace_format(p) for p in summary['uncoached_paces']]
        summary['coached_paces_display'] = [decimal_minutes_to_pace_format(p) for p in summary['coached_paces']]
        
        
        for i, actual_pace in enumerate(summary['actual_paces_min']): # pace differences against coached plan
            coached_pace = summary['coached_paces'][i]
            segment_dist = segment_distances[i]

            #actual_time_sec = actual_pace * segment_dist * 60
            #coached_time_sec = coached_pace * segment_dist * 60

            #if segment_dist < 0.85:
             #   diff = 

            diff = (actual_pace - coached_pace) * 60
            differences.append(diff)
            differences_display.append(f"{'-' if diff < 0 else '+'}{abs(int(diff))}s")
    else:
        summary['predicted_paces_display'] = [decimal_minutes_to_pace_format(p) for p in summary['predicted_paces_min']]
        
        
        for i, actual_pace in enumerate(summary['actual_paces_min']): # differences against prediction
            predicted_pace = summary['predicted_paces_min'][i]
            #segment_dist = segment_distances[i]

            # Convert pace to time for this segment
            #actual_time_sec = actual_pace * segment_dist * 60
            #predicted_time_sec = predicted_pace * segment_dist * 60
            
            
            diff = (actual_pace - coached_pace) * 60 # time difference
            differences.append(diff)
            differences_display.append(f"{'-' if diff < 0 else '+'}{abs(int(diff))}s")
    
    summary['differences'] = differences
    summary['differences_display'] = differences_display
    summary['segments'] = list(range(1, len(summary['actual_paces_min']) + 1))

    
    coached_csv = os.path.join(UPLOAD_FOLDER, "Coached_Paces.csv") # add elevation data for the elevation chart
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
    
    summary = tracker.get_run_summary()
    
    
    coached_csv = os.path.join(UPLOAD_FOLDER, "Coached_Paces.csv") # load route data for elevation and coaching info
    route_data = pd.read_csv(coached_csv)
    
    
    actual_paces = summary['actual_paces_min'] # extract data for analytics
    
    if summary['has_coached_plan']:
        coached_paces = summary['coached_paces']
        uncoached_paces = summary['uncoached_paces']
    else:
        coached_paces = summary['predicted_paces_min']
        uncoached_paces = summary['predicted_paces_min']


    print("\n[DEBUG ANALYTICS]") # right after extracting actual_paces
    print(f"Actual paces (should be min/km): {actual_paces}")
    print(f"Coached paces: {coached_paces}")
    print(f"Segment distances: {route_data['segment_distance_km'].tolist()}")
    print()
    
    elevation_gains = route_data['elevation_gain_m'].tolist()
    elevation_losses = route_data['elevation_loss_m'].tolist()
    
    coaching_methods = session.get('coaching_methods', [])
    
    
    analytics = PerformanceAnalytics( # create analytics 
        actual_paces=actual_paces,
        coached_paces=coached_paces,
        uncoached_paces=uncoached_paces,
        elevation_gains=elevation_gains,
        elevation_losses=elevation_losses,
        coaching_methods=coaching_methods,
        segment_distances=route_data['segment_distance_km'].tolist()
    )
    

    analytics_report = analytics.get_full_analytics_report()
    
   
    from CoachingMethods import decimal_minutes_to_pace_format  # for display
    
    
    if analytics_report['terrain_analysis']['uphill']: # format terrain analysis
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
    
    
    for key in ['first_half_avg_actual', 'second_half_avg_actual', # format split analysis
                'first_half_avg_coached', 'second_half_avg_coached']:
        analytics_report['split_analysis'][key + '_display'] = \
            decimal_minutes_to_pace_format(analytics_report['split_analysis'][key])
    
    
    for segment in analytics_report['best_worst_segments']['best_segments']: # format best/worst segments
        segment['actual_pace_display'] = decimal_minutes_to_pace_format(segment['actual_pace'])
        segment['coached_pace_display'] = decimal_minutes_to_pace_format(segment['coached_pace'])
    
    for segment in analytics_report['best_worst_segments']['worst_segments']:
        segment['actual_pace_display'] = decimal_minutes_to_pace_format(segment['actual_pace'])
        segment['coached_pace_display'] = decimal_minutes_to_pace_format(segment['coached_pace'])
    
    
    analytics_report['segments'] = list(range(1, len(actual_paces) + 1)) # add segment-level data for charts
    analytics_report['actual_paces'] = actual_paces
    analytics_report['coached_paces'] = coached_paces
    analytics_report['net_elevation'] = (np.array(elevation_gains) - np.array(elevation_losses)).tolist()
    
    return render_template("analytics.html", analytics=analytics_report)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
