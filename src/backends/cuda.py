"""CUDA backend for optical flow computation"""

import cv2
import numpy as np
from typing import Dict, Any
from .base import FlowBackend


class CUDABackend(FlowBackend):
    """CUDA-based optical flow backend."""
    
    def compute_flow_info(self, p0: np.ndarray, p1: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        CUDA implementation of optical flow computation.
        
        Args:
            p0: Previous frame
            p1: Current frame  
            config: Configuration dictionary
            
        Returns:
            Dictionary containing flow information
        """
        from ..core.motion_analysis import max_divergence
        
        cut_threshold = config.get("cut_threshold", 7)
        
        # Upload frames to GPU—time to let your GPU do the heavy lifting!
        gpu_p0 = cv2.cuda_GpuMat()
        gpu_p1 = cv2.cuda_GpuMat()
        gpu_p0.upload(p0)
        gpu_p1.upload(p1)

        # Compute optical flow on the GPU
        fb = cv2.cuda_FarnebackOpticalFlow.create(0.5, 3, 15, 3, 5, 1.2, 0)
        flow_gpu = fb.calc(gpu_p0, gpu_p1, None)
        
        # Grab flow back to CPU for any CPU-based ops (like max_divergence)
        flow = flow_gpu.download()
        max_val = max_divergence(flow)
        pos_center = max_val[:2]
        val_pos = max_val[2]

        # Calculate magnitude on GPU (splitting channels)
        channels = cv2.cuda.split(flow_gpu)
        mag_gpu = cv2.cuda.magnitude(channels[0], channels[1])
        mag = mag_gpu.download()
        mean_mag = float(np.mean(mag))
        is_cut = mean_mag > cut_threshold

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