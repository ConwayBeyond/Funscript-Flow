"""Base backend interface and backend detection"""

import cv2
from typing import Dict, Any
from abc import ABC, abstractmethod
import numpy as np


def get_available_backends() -> Dict[str, bool]:
    """Detect available GPU acceleration backends."""
    backends = {
        "CPU": True,  # Always available
        "CUDA": False,
        "OpenCL": False,
        "DNN": False
    }
    
    # Check for CUDA support
    try:
        if hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            backends["CUDA"] = device_count > 0
    except Exception:
        pass
    
    # Check for OpenCL support
    try:
        if hasattr(cv2.ocl, 'haveOpenCL'):
            backends["OpenCL"] = cv2.ocl.haveOpenCL()
    except Exception:
        pass
    
    # Check for DNN backend support
    try:
        if hasattr(cv2, 'dnn'):
            backends["DNN"] = True
    except Exception:
        pass
    
    return backends


def get_gpu_info() -> str:
    """Get information about available GPUs and backends."""
    info = []
    backends = get_available_backends()
    
    # CUDA info with device details
    try:
        if backends["CUDA"] and hasattr(cv2.cuda, 'getCudaEnabledDeviceCount'):
            count = cv2.cuda.getCudaEnabledDeviceCount()
            if count > 0:
                device_names = []
                for i in range(count):
                    try:
                        name = cv2.cuda.getDeviceName(i)
                        device_names.append(name)
                    except Exception:
                        device_names.append(f"Device {i}")
                if len(device_names) == 1:
                    info.append(f"CUDA: {device_names[0]}")
                else:
                    info.append(f"CUDA: {len(device_names)} devices")
    except Exception:
        pass
    
    # OpenCL info (simplified)
    if backends["OpenCL"]:
        info.append("OpenCL supported")
    
    # DNN info
    if backends["DNN"]:
        info.append("DNN module available")
    
    if not info:
        return "CPU only"
    
    return " | ".join(info)


class FlowBackend(ABC):
    """Abstract base class for optical flow backends."""
    
    @abstractmethod
    def compute_flow_info(self, p0: np.ndarray, p1: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
        """Compute optical flow information for two frames."""
        pass