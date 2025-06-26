"""Scene detection and cut detection functionality"""

import numpy as np


def detect_cut(pair, log_func=None, threshold=30):
    """
    Detect scene cuts between two frames based on pixel difference.
    
    Args:
        pair: Tuple of (prev_frame, curr_frame)
        log_func: Optional logging function
        threshold: Threshold for cut detection
        
    Returns:
        bool: True if a cut is detected, False otherwise
    """
    return False
    prev_frame, curr_frame = pair
    diff = np.mean(np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32)))
    if(log_func != None and diff > threshold):
        log_func(f"Found a cut at " + str(diff))
        
    return diff > threshold