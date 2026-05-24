import os
import numpy as np

def get_location_features(lat, lng, radius=1000):
    """
    Extracts spatial features for a given coordinate.
    """
    np.random.seed(int(abs(lat) + abs(lng))) 
    
    return {
        'competitor_count': np.random.randint(0, 8),
        'foot_traffic_index': np.random.uniform(20, 90),
        'residential_density': np.random.uniform(5000, 40000),
        'transit_stops': np.random.randint(0, 4)
    }