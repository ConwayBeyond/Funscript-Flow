"""GPU and processing backend implementations"""

from .base import get_available_backends, get_gpu_info, FlowBackend
from .cpu import CPUBackend
from .cuda import CUDABackend
from .opencl import OpenCLBackend
from .dnn import DNNBackend
from typing import Dict, Any
import numpy as np


def precompute_flow_info(p0: np.ndarray, p1: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Concurrency-friendly function:
      - compute Farneback flow
      - swarm normal flow => pos_center => val_pos
      - swarm negative flow => neg_center => val_neg
      - detect cut
      - pick a center for cut jumps (could just use pos_center)
    Returns a dict with everything needed for final pass.
    """
    # Try to use the selected backend
    backend = config.get("backend", "CPU")
    
    if backend == "CUDA":
        try:
            cuda_backend = CUDABackend()
            return cuda_backend.compute_flow_info(p0, p1, config)
        except Exception as e:
            # Fall back to CPU if GPU fails
            pass
    elif backend == "OpenCL":
        try:
            opencl_backend = OpenCLBackend()
            return opencl_backend.compute_flow_info(p0, p1, config)
        except Exception as e:
            # Fall back to CPU if OpenCL fails
            pass
    elif backend == "DNN":
        try:
            dnn_backend = DNNBackend()
            return dnn_backend.compute_flow_info(p0, p1, config)
        except Exception as e:
            # Fall back to CPU if DNN fails
            pass
    
    # CPU implementation (default)
    cpu_backend = CPUBackend()
    return cpu_backend.compute_flow_info(p0, p1, config)


def precompute_wrapper(p, params):
    """Wrapper function for multiprocessing compatibility."""
    return precompute_flow_info(p[0], p[1], params)