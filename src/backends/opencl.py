"""OpenCL backend for optical flow computation"""

import cv2
import numpy as np
from typing import Dict, Any
from .base import FlowBackend


class OpenCLBackend(FlowBackend):
    """OpenCL-based optical flow backend."""
    
    def compute_flow_info(self, p0: np.ndarray, p1: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenCL-accelerated optical flow computation using UMat.
        
        Args:
            p0: Previous frame
            p1: Current frame  
            config: Configuration dictionary
            
        Returns:
            Dictionary containing flow information
        """
        from ..core.motion_analysis import max_divergence
        
        cut_threshold = config.get("cut_threshold", 7)
        
        # Convert to UMat for GPU processing
        gpu_p0 = cv2.UMat(p0)
        gpu_p1 = cv2.UMat(p1)
        
        # Compute optical flow on GPU
        flow_gpu = cv2.calcOpticalFlowFarneback(gpu_p0, gpu_p1, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Get flow back to CPU
        flow = flow_gpu.get()
        
        # Compute divergence on CPU (could be optimized further)
        if config.get("pov_mode"):
            max_val = (p0.shape[1] // 2, p0.shape[0] - 1, 0)
        else:
            max_val = max_divergence(flow)
        pos_center = max_val[0:2]
        val_pos = max_val[2]
        
        # Calculate magnitude
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mean_mag = np.mean(mag)
        is_cut = mean_mag > cut_threshold
        
        return {
            "flow": flow,
            "pos_center": pos_center,
            "neg_center": pos_center,
            "val_pos": val_pos,
            "val_neg": val_pos,
            "cut": is_cut,
            "cut_center": pos_center[0],
            "mean_mag": mean_mag
        }