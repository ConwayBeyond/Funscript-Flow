"""Optical flow computation functions"""

import cv2
import numpy as np


def compute_flow(pair):
    """
    (VR Mode) Process a pair of consecutive 512x512 grayscale frames (cropped from left half).
    Computes optical flow on two regions (middle-center and bottom-center of a 3x3 grid)
    and returns a tuple (avg_flow_middle, avg_flow_bottom).
    """
    prev_frame, curr_frame = pair
    h, w = prev_frame.shape  # expected 512x512
    cell_h = h // 3
    cell_w = w // 3
    prev_middle = prev_frame[cell_h:2*cell_h, cell_w:2*cell_w]
    curr_middle = curr_frame[cell_h:2*cell_h, cell_w:2*cell_w]
    prev_bottom = prev_frame[2*cell_h:3*cell_h, cell_w:2*cell_w]
    curr_bottom = curr_frame[2*cell_h:3*cell_h, cell_w:2*cell_w]
    flow_middle = cv2.calcOpticalFlowFarneback(prev_middle, curr_middle, None,
                                               pyr_scale=0.5, levels=3, winsize=15,
                                               iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    flow_bottom = cv2.calcOpticalFlowFarneback(prev_bottom, curr_bottom, None,
                                               pyr_scale=0.5, levels=3, winsize=15,
                                               iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    avg_middle = np.mean(flow_middle[..., 1])
    avg_bottom = np.mean(flow_bottom[..., 1])
    return avg_middle, avg_bottom


def compute_flow_nonvr_invert(pair):
    """Compute optical flow for non-VR mode with inverted flow."""
    prev_frame, curr_frame = pair
    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    avg_flow = np.mean(-flow[..., 0] + flow[..., 1])
    return avg_flow


def compute_flow_nonvr(pair):
    """Compute optical flow for non-VR mode."""
    prev_frame, curr_frame = pair
    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    avg_flow = np.mean(flow[..., 0] + flow[..., 1])
    return avg_flow