# Stores km splits, calculates pace differences, manages timer state
# Functions: start_run(), log_km_split(), get_current_stats()

"""
Run Tracker
Handles live run tracking, km splits, and comparison to predicted paces.
"""

import time
from datetime import datetime
from typing import List, Dict
import pandas as pd
from flask import jsonify

class RunTracker:
    def __init__(self, route_data: pd.DataFrame, selected_plan: str = "Final Plan"):
        """
        Initialize run tracker with predicted paces
        
        Args:
            route_data: DataFrame with segment data and predicted paces
            selected_plan: Which pace plan column to use for comparison
        """
        self.route_data = route_data
        self.selected_plan_col = f"{selected_plan}_pace_min_per_km"
        
        # Get the predicted paces for the selected plan
        if self.selected_plan_col in route_data.columns:
            self.predicted_paces = route_data[self.selected_plan_col].values
        else:
            raise ValueError(f"Column {self.selected_plan_col} not found in route data")
        
        self.total_segments = len(self.predicted_paces)
        
        # Tracking data
        self.run_start_time = None
        self.segment_start_time = None
        self.current_segment = 0
        self.actual_splits = []  # Actual time for each km (in seconds)
        self.split_timestamps = []  # When each split was logged
        self.is_running = False
        self.is_completed = False
    
    def start_run(self):
        """Start the run - begins both global and segment timers"""
        if self.is_running:
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
    
    def log_km_split(self):
        """Log completion of current km segment"""
        if not self.is_running:
            return jsonify({"error": "Run not started"})
        
        if self.current_segment >= self.total_segments:
            return jsonify({"error": "All segments already completed"})
        
        current_time = time.time()
        
        # Calculate time for this segment (in seconds)
        segment_time_sec = current_time - self.segment_start_time
        segment_time_min = segment_time_sec / 60
        
        # Get predicted pace for this segment (in minutes)
        predicted_pace = float(self.predicted_paces[self.current_segment])
        
        # Calculate difference
        pace_diff = segment_time_min - predicted_pace
        
        # Store the split
        self.actual_splits.append(segment_time_sec)
        self.split_timestamps.append(current_time)
        
        # Prepare response data
        split_data = {
            "segment": int(self.current_segment + 1),
            "actual_time_sec": float(segment_time_sec),
            "actual_pace_min": round(float(segment_time_min), 2),
            "predicted_pace_min": round(predicted_pace, 2),
            "difference_sec": round(float(pace_diff * 60), 2),
            "faster": bool(pace_diff < 0),  # Explicit conversion
            "total_elapsed_sec": float(current_time - self.run_start_time),
            "run_completed": False
        }
        
        # Move to next segment
        self.current_segment += 1
        self.segment_start_time = current_time
        
        # Check if run is complete
        if self.current_segment >= self.total_segments:
            self.is_running = False
            self.is_completed = True
            split_data["run_completed"] = True
        
        return jsonify(split_data)
    
    def get_current_status(self):
        """Get current run status"""
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
        """Get complete run summary after completion"""
        if not self.is_completed:
            return {"error": "Run not completed yet"}
        
        total_actual_time = sum(self.actual_splits)
        # total_predicted_time = sum(self.predicted_paces) * 60  # Convert to seconds
        
        actual_paces = [split / 60 for split in self.actual_splits]  # Convert to minutes

        # DEBUG: Print all columns to see what exists
        print("=== CHECKING COLUMNS FOR SUMMARY ===")
        print(f"All columns: {self.route_data.columns.tolist()}")

        # CORRECTED: Use the actual column names from the CSV
        final_plan_col = "Final Plan_pace_min_per_km"
        uncoached_pace_col = "Uncoached Pace_pace_min_per_km"  # FIXED: Was "Predicted Pace_pace_min_per_km"

        has_coached_plan = final_plan_col in self.route_data.columns

        print(f"Has Final Plan column: {has_coached_plan}")
        print(f"Has Predicted Pace column: {uncoached_pace_col in self.route_data.columns}")
        print("====================================")
        
        summary = {
            "total_actual_time_sec": total_actual_time,
            #"total_predicted_time_sec": total_predicted_time,
            #"total_difference_sec": total_actual_time - total_predicted_time,
            "actual_paces_min": actual_paces,
            #"predicted_paces_min": self.predicted_paces.tolist(),
            #"segments_faster": sum(1 for i, split in enumerate(self.actual_splits) 
                                  #if split / 60 < self.predicted_paces[i]),
            #"segments_slower": sum(1 for i, split in enumerate(self.actual_splits) 
                                  #if split / 60 > self.predicted_paces[i]),
            "segments_faster": 0,
            "segments_slower": 0,
            "has_coached_plan": has_coached_plan
        }

        if has_coached_plan:

            # Get both uncoached and coached paces
            uncoached_paces = self.route_data[uncoached_pace_col].values # FIXED
            coached_paces = self.route_data[final_plan_col].values

            print(f"Uncoached paces: {uncoached_paces[:3]}...")
            print(f"Coached paces: {coached_paces[:3]}...")

            summary["uncoached_paces"] = uncoached_paces.tolist() #self.route_data["Predicted Pace_pace_min_per_km"].values.tolist()
            summary["coached_paces"] = coached_paces.tolist() #self.predicted_paces.tolist()  # This is the Final Plan
            summary["total_predicted_time_sec"] = sum(coached_paces) * 60

            # Calculate differences against coached plan
            for i, split in enumerate(self.actual_splits):
                if split / 60 < coached_paces[i]:
                    summary["segments_faster"] += 1
                else:
                    summary["segments_slower"] += 1
        else: 
            # Only uncoached prediction exists
            uncoached_paces = self.route_data[uncoached_pace_col].values # FIXED
            summary["predicted_paces_min"] = uncoached_paces.tolist()
            summary["total_predicted_time_sec"] = sum(uncoached_paces) * 60
            
            # Calculate differences against prediction
            for i, split in enumerate(self.actual_splits):
                if split / 60 < uncoached_paces[i]:
                    summary["segments_faster"] += 1
                else:
                    summary["segments_slower"] += 1
        
        summary["total_difference_sec"] = total_actual_time - summary["total_predicted_time_sec"]
    
        return summary