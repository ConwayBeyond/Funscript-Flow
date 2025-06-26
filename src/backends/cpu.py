"""CPU backend for optical flow computation"""

import cv2
import numpy as np
from typing import Dict, Any
from .base import FlowBackend


class CPUBackend(FlowBackend):
    """CPU-based optical flow backend."""
    
    def compute_flow_info(self, p0: np.ndarray, p1: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        CPU implementation of optical flow computation.
        
        Args:
            p0: Previous frame
            p1: Current frame  
            config: Configuration dictionary
            
        Returns:
            Dictionary containing flow information
        """
        from ..core.motion_analysis import max_divergence
        
        cut_threshold = config.get("cut_threshold", 7)
        
        flow = cv2.calcOpticalFlowFarneback(p0, p1, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        if config.get("pov_mode"):
            # In pov mode, just use the center of the bottom edge of the frame
            max_val = (p0.shape[1] // 2, p0.shape[0] - 1, 0)
        else:
            max_val = max_divergence(flow)
        pos_center = max_val[0:2]
        val_pos = max_val[2]
            
        # Detect cut based on flow map
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mean_mag = np.mean(mag)
        if mean_mag > cut_threshold:
            is_cut = True
        else:
            is_cut = False

        cut_center = pos_center[0]

        return {
            "flow": flow,
            "pos_center": pos_center,
            "neg_center": pos_center,
            "val_pos": val_pos,
            "val_neg": val_pos,
            "cut": is_cut,
            "cut_center": cut_center,
            "mean_mag": mean_mag
        }