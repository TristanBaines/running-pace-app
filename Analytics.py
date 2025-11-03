import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class PerformanceAnalytics:
    def __init__(self, actual_paces: List[float], coached_paces: List[float], 
                 uncoached_paces: List[float], elevation_gains: List[float], 
                 elevation_losses: List[float], coaching_methods: List[str], segment_distances: List[float] = None):

        self.actual_paces = np.array(actual_paces)
        self.coached_paces = np.array(coached_paces)
        self.uncoached_paces = np.array(uncoached_paces)
        self.elevation_gains = np.array(elevation_gains)
        self.elevation_losses = np.array(elevation_losses)
        self.coaching_methods = coaching_methods
        self.segment_distances = np.array(segment_distances) if segment_distances else np.ones(len(actual_paces))
        
        
        self.net_elevation = self.elevation_gains - self.elevation_losses # calculate derived metrics
        self.pace_differences = self.actual_paces - self.coached_paces

        
    def analyze_terrain_performance(self) -> Dict:
        
        gradients = (self.net_elevation / (self.segment_distances * 1000)) * 100 # gradients
        
        
        uphill_mask = gradients > 1.0 # classify terrains
        downhill_mask = gradients < -1.0
        flat_mask = (gradients >= -1.0) & (gradients <= 1.0)
        
        def terrain_stats(mask):
            if not np.any(mask):
                return None
            return {
                'count': int(np.sum(mask)),
                'avg_actual_pace': float(np.mean(self.actual_paces[mask])),
                'avg_coached_pace': float(np.mean(self.coached_paces[mask])),
                'avg_difference': float(np.mean(self.pace_differences[mask])),
                'avg_gradient': float(np.mean(gradients[mask])),
                'segments_faster': int(np.sum(self.pace_differences[mask] < 0)),
                'segments_slower': int(np.sum(self.pace_differences[mask] > 0))
            }
        
        return {
            'uphill': terrain_stats(uphill_mask),
            'downhill': terrain_stats(downhill_mask),
            'flat': terrain_stats(flat_mask)
        }
    
    def analyze_pacing_consistency(self) -> Dict:
        
        actual_cv = (np.std(self.actual_paces) / np.mean(self.actual_paces)) * 100 # coefficient of variation for actual paces
        coached_cv = (np.std(self.coached_paces) / np.mean(self.coached_paces)) * 100
        
        
        pace_changes = np.abs(np.diff(self.actual_paces)) # pace variability, differences between actual paces
        avg_pace_change = np.mean(pace_changes) # average difference
        
        significant_changes = pace_changes > (np.mean(self.actual_paces) * 0.1) # identify significant pace fluctuations (>10% change)
        
        return {
            'actual_cv': float(actual_cv),
            'coached_cv': float(coached_cv),
            'avg_pace_change': float(avg_pace_change),
            'num_significant_changes': int(np.sum(significant_changes)),
            'most_stable_segment': int(np.argmin(pace_changes)) + 1, # smallest change in actual pace
            'most_variable_segment': int(np.argmax(pace_changes)) + 1 # biggest
        }
    
    def analyze_splits(self) -> Dict:
        n = len(self.actual_paces)
        half = n // 2
        
        first_half_actual = self.actual_paces[:half]
        second_half_actual = self.actual_paces[half:]
        
        first_half_coached = self.coached_paces[:half]
        second_half_coached = self.coached_paces[half:]
        
        return {
            'first_half_avg_actual': float(np.mean(first_half_actual)),
            'second_half_avg_actual': float(np.mean(second_half_actual)),
            'first_half_avg_coached': float(np.mean(first_half_coached)),
            'second_half_avg_coached': float(np.mean(second_half_coached)),
            'split_type': 'negative' if np.mean(second_half_actual) < np.mean(first_half_actual) else 'positive',
            'split_difference': float(np.mean(second_half_actual) - np.mean(first_half_actual))
        }
    
    def analyze_coaching_effectiveness(self) -> Dict:
        mae = np.mean(np.abs(self.pace_differences))
        rmse = np.sqrt(np.mean(self.pace_differences ** 2))
        
        segments_beat = self.pace_differences < 0
        beat_rate = (np.sum(segments_beat) / len(segments_beat)) * 100
        
        
        coached_improvement = self.uncoached_paces - self.coached_paces # compare
        actual_vs_uncoached = self.uncoached_paces - self.actual_paces
        
        return {
            'mae_seconds': float(mae * 60),
            'rmse_seconds': float(rmse * 60),
            'beat_rate': float(beat_rate),
            'segments_beat': int(np.sum(segments_beat)),
            'avg_coached_improvement': float(np.mean(coached_improvement) * 60),
            'avg_actual_improvement': float(np.mean(actual_vs_uncoached) * 60),
            'coaching_methods_used': self.coaching_methods
        }
    
    def identify_best_worst_segments(self) -> Dict:
        best_indices = np.argsort(self.pace_differences)[:3] # shows top 3 best
        
        slower_segments = np.where(self.pace_differences > 0)[0]  # segments where actual slower than coached

        if len(slower_segments) > 0:
            
            slower_sorted = slower_segments[np.argsort(self.pace_differences[slower_segments])[::-1]] # sort slower segments
            worst_indices = slower_sorted[:min(3, len(slower_sorted))]  # up to 3
        else:
            worst_indices = []  # none if runner beat all paces
        
        def segment_info(idx):
            return {
                'segment': int(idx + 1),
                'actual_pace': float(self.actual_paces[idx]),
                'coached_pace': float(self.coached_paces[idx]),
                'difference_seconds': float(self.pace_differences[idx] * 60),
                'net_elevation': float(self.net_elevation[idx])
            }
        
        return {
            'best_segments': [segment_info(i) for i in best_indices],
            'worst_segments': [segment_info(i) for i in worst_indices],
            'has_struggling_segments': len(worst_indices) > 0
        }
    
    def generate_recommendations(self) -> List[str]:
        recommendations = []
        
        
        terrain = self.analyze_terrain_performance() # terrain-based
        
        if terrain['uphill'] and terrain['uphill']['avg_difference'] > 0.5: # more than x min/km slower than coached pace
            recommendations.append(
                f"Focus on uphill training. You averaged {terrain['uphill']['avg_difference']:.2f} min/km "
                f"slower than coached pace on uphills."
            )
        
        if terrain['downhill'] and terrain['downhill']['avg_difference'] > 0.3: # more than x min/km slower than coached pace
            recommendations.append(
                "You were slower on downhills than expected - work on downhill running technique to maximize the benefit of descents."
            )

        if terrain['uphill'] and terrain['uphill']['avg_difference'] < 0.3:
            recommendations.append(
                "You performed great on uphills - good strong running!"
            )
            
        if terrain['downhill'] and terrain['downhill']['avg_difference'] < 0.3:
            recommendations.append(
                "You took advantage of downhills and saved yourself time - excellent!"
            )
        
        
        consistency = self.analyze_pacing_consistency() # pacing consistency
        
        if consistency['actual_cv'] > 15: # more than x% fluctuation in pace
            recommendations.append(
                f"Your pacing was inconsistent (CV: {consistency['actual_cv']:.1f}%). "
                "Practice maintaining steady effort across segments."
            )
        elif consistency['actual_cv'] <= 15:
            recommendations.append(
                f"Your pacing was consistent (CV: {consistency['actual_cv']:.1f}%)."
                "You maintained steady pacing across segments - keep it up!"
            )
        
        
        splits = self.analyze_splits() # splits
        
        if splits['split_type'] == 'positive' and splits['split_difference'] > 0.3:
            recommendations.append(
                "You slowed significantly in the second half. Focus on pacing strategy "
                "and endurance training to maintain speed."
            )
        elif splits['split_type'] == 'negative':
            recommendations.append(
                "Excellent negative split! You maintained or improved pace in the second half."
            )
        
        
        effectiveness = self.analyze_coaching_effectiveness() # beating coach

        if effectiveness['beat_rate'] < 30:
            recommendations.append(
                f"You only beat the coached pace on {effectiveness['beat_rate']:.1f}% of segments. "
                "The coached plan may be too tough - consider running more to improve fitness."
            )
        elif effectiveness['beat_rate'] > 70:
            recommendations.append(
                f"Outstanding! You beat the coached pace on {effectiveness['beat_rate']:.1f}% of segments. "
                "The pacing strategy worked excellently for you."
            )
        elif effectiveness['beat_rate'] >= 50:
            recommendations.append(
                f"Good and well balanced performance! You beat the coached pace on {effectiveness['beat_rate']:.1f}% of segments."
            )
        
        if not recommendations:
            recommendations.append("Overall solid performance! Keep up the good work.")
        
        return recommendations
    
    def get_full_analytics_report(self) -> Dict:
        return {
            'terrain_analysis': self.analyze_terrain_performance(),
            'consistency_analysis': self.analyze_pacing_consistency(),
            'split_analysis': self.analyze_splits(),
            'coaching_effectiveness': self.analyze_coaching_effectiveness(),
            'best_worst_segments': self.identify_best_worst_segments(),
            'recommendations': self.generate_recommendations()
        }