COACHING_PARAMS = {
    'Tristan': {
        'push_uphills_decrease_factor': 0.95087655222790356,  # 5% faster on uphills
        'push_downhills_decrease_factor': 0.95580715850986121,
        'push_flats_baseline_pace': 5.476,  # average pace on flats
    },
    'Talia': {
        'push_uphills_decrease_factor': 1 - 0.2005263157894738,  
        'push_downhills_decrease_factor': 1 - 0.002236842105263087, 
        'push_flats_baseline_pace': 7.6,  # average pace on flats
    }
}

def get_coaching_params(model_name):
    if model_name not in COACHING_PARAMS:
        raise ValueError(f"No coaching parameters found for model: {model_name}")
    
    return COACHING_PARAMS[model_name]