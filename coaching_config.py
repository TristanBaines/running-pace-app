"""
Coaching configuration parameters for different athlete models.
Each model has its own calibrated values based on the athlete's training data.
"""

COACHING_PARAMS = {
    'Tristan': {
        'push_uphills_decrease_factor': 0.95087655222790356,  # 5% faster on uphills
        'push_downhills_decrease_factor': 0.95580715850986121,  # Your calculated value
        'push_flats_baseline_pace': 5.476,  # Your average pace on flats
    },
    'Talia': {
        'push_uphills_decrease_factor': 1 - 0.2005263157894738,  # Talia's calculated value
        'push_downhills_decrease_factor': 1 - 0.002236842105263087,  # Talia's calculated value
        'push_flats_baseline_pace': 7.6,  # Talia's average pace on flats
    }
}

def get_coaching_params(model_name):
    """
    Get coaching parameters for a specific model.
    
    Args:
        model_name: Name of the athlete model (e.g., 'Tristan', 'Talia')
    
    Returns:
        Dictionary of coaching parameters
    
    Raises:
        ValueError: If model_name not found
    """
    if model_name not in COACHING_PARAMS:
        raise ValueError(f"No coaching parameters found for model: {model_name}")
    
    return COACHING_PARAMS[model_name]