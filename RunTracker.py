import time
from datetime import datetime
from typing import List, Dict
import pandas as pd
from flask import jsonify

class RunTracker:
    def __init__(self, route_data: pd.DataFrame, selected_plan: str = "Final Plan"):

        self.route_data = route_data
        self.selected_plan_col = f"{selected_plan}_pace_min_per_km"
        
        
        if self.selected_plan_col in route_data.columns: # predicted paces for the selected plan
            self.predicted_paces = route_data[self.selected_plan_col].values
        else:
            raise ValueError(f"Column {self.selected_plan_col} not found in route data")
        
        self.total_segments = len(self.predicted_paces)

        
        self.is_paused = False # pause functionality
        self.pause_start_time = None
        self.total_paused_time = 0.0
        self.segment_paused_time = 0.0
        
        
        self.run_start_time = None # for tracking data
        self.segment_start_time = None
        self.current_segment = 0
        self.actual_splits = []  # actual time for each km (in seconds)
        self.split_timestamps = []  # when each split is logged
        self.is_running = False
        self.is_completed = False
    
    def start_run(self):
        
        if self.is_running: # Start the run, begins both global and segment timer
            return {"error": "Run already started"}
        
        current_time = time.time()
        self.run_start_time = current_time
        self.segment_start_time = current_time
        self.is_running = True
        self.current_segment = 0
        
        return {
            "status": "started",
            "start_time": datetime.fromtimestamp(current_time).isoformat(),
            "total_segments": self.total_segments
        }
    
    def pause_run(self):

        if not self.is_running or self.is_paused or self.is_completed:
            return {"error": "Cannot pause"}
        
        self.is_paused = True
        self.pause_start_time = time.time()
        return {"status": "paused"}

    def resume_run(self):

        if not self.is_paused:
            return {"error": "Not paused"}
        
        pause_duration = time.time() - self.pause_start_time
        self.total_paused_time += pause_duration
        self.segment_paused_time += pause_duration
        self.is_paused = False
        
        return {"status": "resumed", "pause_duration": pause_duration}
    





    def log_km_split(self):
        if self.is_paused:
            return jsonify({"error": "Cannot log split while paused"}) # paused
        
        if not self.is_running:
            return jsonify({"error": "Run not started"})
        
        if self.current_segment >= self.total_segments:
            return jsonify({"error": "All segments already completed"})
        
        current_time = time.time()
        
        segment_time_sec = (current_time - self.segment_start_time) - self.segment_paused_time # taking pause into account
        segment_time_min = segment_time_sec / 60

        # DEBUG LOGGING
        print(f"\n[BACKEND SPLIT] Segment {self.current_segment + 1}")
        print(f"[BACKEND] current_time: {current_time}")
        print(f"[BACKEND] segment_start_time: {self.segment_start_time}")
        print(f"[BACKEND] segment_paused_time: {self.segment_paused_time}")
        print(f"[BACKEND] Calculated segment_time_sec: {segment_time_sec}")
        
        predicted_pace = float(self.predicted_paces[self.current_segment])

        segment_distance = float(self.route_data.iloc[self.current_segment]['segment_distance_km'])

        print(f"[BACKEND] segment_distance: {segment_distance}")
        print(f"[BACKEND] segment_time_min: {segment_time_min}")




        actual_pace_per_km = segment_time_min / segment_distance if segment_distance > 0 else segment_time_min




        print(f"[BACKEND] actual_pace_per_km: {actual_pace_per_km}")
        print(f"[BACKEND] predicted_pace: {predicted_pace}")
        

        pace_diff = actual_pace_per_km - predicted_pace # difference
        
        
        self.actual_splits.append(segment_time_sec) # store the split
        self.split_timestamps.append(current_time)
        
        split_data = {
            "segment": int(self.current_segment + 1),
            "actual_time_sec": float(segment_time_sec),
            "actual_pace_min": round(float(actual_pace_per_km), 2),
            "predicted_pace_min": round(predicted_pace, 2),
            "difference_sec": round(float(pace_diff * 60), 2),
            "faster": bool(pace_diff < 0), 
            "total_elapsed_sec": float(current_time - self.run_start_time),
            "run_completed": False
        }

        print(f"[BACKEND] Returning split_data: {split_data}\n")
        
        
        self.current_segment += 1 # next segment
        self.segment_start_time = current_time
        self.segment_paused_time = 0.0 # reset pause time
        
        
        if self.current_segment >= self.total_segments: # check if run is complete
            self.is_running = False
            self.is_completed = True
            split_data["run_completed"] = True
        
        return jsonify(split_data)
    
    def get_current_status(self):
        if not self.run_start_time:
            return {
                "status": "not_started",
                "current_segment": 0,
                "total_segments": self.total_segments
            }
        
        current_time = time.time()
        total_elapsed = current_time - self.run_start_time
        
        if self.is_running:
            segment_elapsed = current_time - self.segment_start_time
        else:
            segment_elapsed = 0
        
        return {
            "status": "completed" if self.is_completed else "running",
            "current_segment": self.current_segment,
            "total_segments": self.total_segments,
            "total_elapsed_sec": total_elapsed,
            "segment_elapsed_sec": segment_elapsed,
            "predicted_pace_current": float(self.predicted_paces[self.current_segment]) if self.current_segment < self.total_segments else None
        }
    




    def get_run_summary(self):
        if not self.is_completed:
            return {"error": "Run not completed yet"}
        
        total_actual_time = sum(self.actual_splits)
        
        
        actual_paces_per_km = [] # actual pace PER KM for each segment
        for i, split_time_sec in enumerate(self.actual_splits):
            segment_distance = float(self.route_data.iloc[i]['segment_distance_km'])

            if(segment_distance < 0.85):
                pace_per_km = (split_time_sec / 60) / segment_distance if segment_distance > 0 else 0 # was just this line
            if(segment_distance >= 0.85):
                pace_per_km = (split_time_sec / 60)

            actual_paces_per_km.append(pace_per_km)
        
        
        final_plan_col = "Final Plan_pace_min_per_km"
        uncoached_pace_col = "Uncoached Pace_pace_min_per_km"
        has_coached_plan = final_plan_col in self.route_data.columns
        
        summary = {
            "total_actual_time_sec": total_actual_time,
            "actual_paces_min": actual_paces_per_km,
            "segments_faster": 0,
            "segments_slower": 0,
            "has_coached_plan": has_coached_plan,
            "segment_distances": self.route_data['segment_distance_km'].tolist()
        }

        if has_coached_plan:
            uncoached_paces = self.route_data[uncoached_pace_col].values
            coached_paces = self.route_data[final_plan_col].values

            summary["uncoached_paces"] = uncoached_paces.tolist()
            summary["coached_paces"] = coached_paces.tolist()

            
            segment_distances = self.route_data['segment_distance_km'].values # total predicted time from coached paces
            total_predicted_time_min = sum(coached_paces * segment_distances)
            summary["total_predicted_time_sec"] = total_predicted_time_min * 60

            for i, actual_pace in enumerate(actual_paces_per_km):
                if actual_pace < coached_paces[i]:
                    summary["segments_faster"] += 1
                else:
                    summary["segments_slower"] += 1
        else:
            uncoached_paces = self.route_data[uncoached_pace_col].values
            summary["predicted_paces_min"] = uncoached_paces.tolist()

            
            segment_distances = self.route_data['segment_distance_km'].values # total predicted time from uncoached paces
            total_predicted_time_min = sum(uncoached_paces * segment_distances)
            summary["total_predicted_time_sec"] = total_predicted_time_min * 60
            
            for i, actual_pace in enumerate(actual_paces_per_km):
                if actual_pace < uncoached_paces[i]:
                    summary["segments_faster"] += 1
                else:
                    summary["segments_slower"] += 1
        
        summary["total_difference_sec"] = total_actual_time - summary["total_predicted_time_sec"]

        return summary