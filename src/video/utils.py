"""Video processing utilities and frame fetching functions"""

import cv2
import numpy as np
import gc
from typing import List, Dict
from .readers import AsyncVideoReader


def fetch_frames(video_path, chunk, params):
    """Fetch frames from video with parameters for VR mode."""
    frames_gray = []
    try:
        # Use AsyncVideoReader for better performance
        vr = AsyncVideoReader(
            video_path, 
            num_threads=min(params["threads"], 4),  # Limit decoder threads
            width=512 if params.get("vr_mode") else 256, 
            height=512 if params.get("vr_mode") else 256,
            buffer_size=min(len(chunk), 100)  # Adaptive buffer size
        )
        batch_frames = vr.get_batch_parallel(chunk)  # Use parallel decoding
    except Exception as e:
        return frames_gray
    vr = None
    gc.collect()

    for f in batch_frames:
        if params.get("vr_mode"):
            h, w, _ = f.shape
            gray = cv2.cvtColor(f[h // 2:, :w // 2], cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        frames_gray.append(gray)

    return frames_gray


def fetch_frames_optimized(video_path: str, chunk: List[int], params: Dict) -> List[np.ndarray]:
    """Optimized frame fetching with direct grayscale conversion."""
    frames_gray = []
    
    try:
        # Don't resize if we're going to crop anyway in VR mode
        target_size = None if params.get("vr_mode") else (256, 256)
        
        vr = AsyncVideoReader(
            video_path,
            num_threads=min(params["threads"], 4),
            width=target_size[0] if target_size else None,
            height=target_size[1] if target_size else None,
            buffer_size=min(len(chunk), 100)
        )
        
        # Process in smaller sub-batches for better memory efficiency
        sub_batch_size = 50
        for i in range(0, len(chunk), sub_batch_size):
            sub_chunk = chunk[i:i + sub_batch_size]
            batch_frames = vr.get_batch_parallel(sub_chunk)
            
            for f in batch_frames:
                if params.get("vr_mode"):
                    # For VR mode, resize then crop and convert
                    f_resized = cv2.resize(f, (512, 512))
                    h, w = 512, 512
                    # Direct grayscale conversion of the cropped region
                    gray = cv2.cvtColor(f_resized[h // 2:, :w // 2], cv2.COLOR_RGB2GRAY)
                else:
                    # Direct grayscale conversion
                    gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
                frames_gray.append(gray)
                
    except Exception as e:
        return frames_gray
    finally:
        vr = None
        gc.collect()
    
    return frames_gray


def convert_frame_to_photo(frame):
    """Convert a frame to photo format for preview."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb