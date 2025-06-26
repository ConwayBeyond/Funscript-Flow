"""Main video processing functionality"""

import os
import time
import math
import threading
import json
import concurrent.futures
import numpy as np
from queue import Queue, Empty
from multiprocessing import Pool

from ..config.localization import STRINGS
from ..video.readers import VideoReaderCV
from ..video.utils import fetch_frames_optimized
from ..backends import precompute_wrapper
from .motion_analysis import radial_motion_weighted


def process_video(video_path, params, log_func, progress_callback=None, cancel_flag=None, preview_callback=None):
    """
    Example usage that:
      1) Reads frames in bracketed intervals
      2) For each consecutive pair, runs precompute_flow_info() in multiple threads
      3) After concurrency finishes, applies inertia in a single-thread pass
      4) In a final concurrency step, computes radial_flow with the final center
         (and sign) for each pair.
    """
    start_time = time.time()
    error_occurred = False
    base, _ = os.path.splitext(video_path)
    output_path = base + ".funscript"
    if os.path.exists(output_path) and not params["overwrite"]:
        log_func(f"Skipping: output file exists ({output_path})")
        return error_occurred

    # Attempt to open video
    try:
        log_func(f"Processing video: {video_path}")
        vr = VideoReaderCV(video_path, width=1024, height=1024, num_threads=params["threads"])
    except Exception as e:
        log_func(f"ERROR: Unable to open video at {video_path}: {e}")
        return True

    # Basic video properties
    try:
        total_frames = len(vr)
        fps = vr.get_avg_fps()
    except Exception as e:
        log_func(f"ERROR: Unable to read video properties: {e}")
        return True

    step = max(1, int(math.ceil(fps / 30.0)))
    effective_fps = fps / step
    indices = list(range(0, total_frames, step))
    log_func(f"FPS: {fps:.2f}; downsampled to ~{effective_fps:.2f} fps; {len(indices)} frames selected.")
    log_func(f"Using backend: {params.get('backend', 'CPU')}")
    bracket_size = int(params.get("batch_size", 3000.0))

    center = None
    velocity = np.zeros(2, dtype=float)
    final_flow_list = []

    # Use a queue for prefetching instead of global variable
    prefetch_queue = Queue(maxsize=1)
    fetch_thread = None
    # We'll collect color frames for preview
    # (and grayscale for computing flow).
    final_con_list = []

    for chunk_start in range(0, len(indices), bracket_size):
        if cancel_flag and cancel_flag():
            log_func("User bailed.")
            return error_occurred

        chunk = indices[chunk_start:chunk_start + bracket_size]
        frame_indices = chunk[:-1]
        if len(chunk) < 2:
            continue

        # Get frames from prefetch queue or fetch directly
        frames_gray = None
        if fetch_thread and fetch_thread.is_alive():
            try:
                frames_gray = prefetch_queue.get(timeout=30.0)
            except Empty:
                log_func(f"WARNING: Prefetch timeout for chunk {chunk_start}")
        
        if frames_gray is None:
            frames_gray = fetch_frames_optimized(video_path, chunk, params)

        if not frames_gray:
            log_func(f"ERROR: Unable to fetch frames for chunk {chunk_start} - skipping.")
            continue
        
        # Start prefetching next batch
        if chunk_start + bracket_size < len(indices):
            next_chunk = indices[chunk_start + bracket_size:chunk_start + 2 * bracket_size]
            
            def prefetch_worker(path, chunk_data, params_data, queue):
                try:
                    frames = fetch_frames_optimized(path, chunk_data, params_data)
                    queue.put(frames)
                except Exception as e:
                    queue.put([])  # Put empty list on error
            
            fetch_thread = threading.Thread(
                target=prefetch_worker,
                args=(video_path, next_chunk, params, prefetch_queue)
            )
            fetch_thread.start()

        # Build consecutive pairs
        pairs = list(zip(frames_gray[:-1], frames_gray[1:]))

        with Pool(processes=params["threads"]) as pool:
            precomputed = pool.starmap(precompute_wrapper, [(p, params) for p in pairs])

        # import matplotlib.pyplot as plt
        # mean_mags = [abs(info["val_pos"]) for info in precomputed]
        # plt.plot(mean_mags)
        # plt.show()
        # Add values of val_pos to final_con_list for preview
        final_con_list.extend([abs(info["val_pos"] * 10) for info in precomputed])

        # 2) Single-thread pass to calculate centers based on the median center of the surrounding second, discarding outliers
        final_centers = []
        chosen_center = None
        for j, info in enumerate(precomputed):
            # Use the mean center of the 6 frames in each direction, discarding outliers
            center_list = [info["pos_center"]]
            for i in range(1, 7):
                if j - i >= 0:
                    center_list.append(precomputed[j - i]["pos_center"])
                if j + i < len(precomputed):
                    center_list.append(precomputed[j + i]["pos_center"])
            center_list = np.array(center_list)
            # Discard outliers from center list
            center = np.mean(center_list, axis=0)
            final_centers.append(center)
            
            # Progress update during center computation
            if progress_callback and j % 24 == 0:  # Update every 24 frames
                frames_processed = chunk_start + j * 0.5  # Center computation is ~50% of processing
                prog = min(100, int(100 * frames_processed / len(indices)))
                progress_callback(prog)
            
            # Show preview with the new center on the "next" color frame
            # e.g. color_pairs[j][1] is the "current" next frame
            # preview_frame = pairs[j][1].copy()
            # cv2.circle(preview_frame, (int(center[0]), int(center[1])), 6, (0,255,0), -1)
            # cv2.imshow("preview", preview_frame)
            # cv2.waitKey(30)

        # 3) Concurrency to compute final weighted dot products with actual final center
        results_in_bracket = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=params["threads"]) as ex:
            dot_futures = []
            for j, info in enumerate(precomputed):
                dot_futures.append(ex.submit(radial_motion_weighted, info["flow"], final_centers[j], info["cut"], params.get("pov_mode", False)))
            dot_vals = [f.result() for f in dot_futures]


        for j, dot_val in enumerate(dot_vals):
            #flow_val = dot_val * signs[j]
            is_cut   = precomputed[j]["cut"]
            final_flow_list.append((dot_val, is_cut, frame_indices[j]))
            
            # More frequent progress updates within chunk processing
            if progress_callback and j % 24 == 0:  # Update every 24 frames
                frames_processed = chunk_start + j + 1
                prog = min(100, int(100 * frames_processed / len(indices)))
                progress_callback(prog)

        # Final progress update for this chunk
        if progress_callback:
            prog = min(100, int(100 * (chunk_start + len(chunk)) / len(indices)))
            progress_callback(prog)

    # Clean up prefetch thread if still running
    if fetch_thread and fetch_thread.is_alive():
        fetch_thread.join(timeout=5.0)
    
    # Clear the prefetch queue
    while not prefetch_queue.empty():
        try:
            prefetch_queue.get_nowait()
        except Empty:
            break

    # --- Piecewise Integration and Timestamping 
    cum_flow = [0]
    time_stamps = [final_flow_list[0][2]]

    for i in range(1, len(final_flow_list)):
        flow_prev, cut_prev, t_prev = final_flow_list[i - 1]
        flow_curr, cut_curr, t_curr = final_flow_list[i]

        if cut_curr:
            cum_flow.append(0)
        else:
            # Midpoint integration to reduce phase lag
            mid_flow = (flow_prev + flow_curr) / 2
            cum_flow.append(cum_flow[-1] + mid_flow)

        time_stamps.append(t_curr)

    # Optional: Shift the result back by half a time step to correct for residual phase offset
    cum_flow = [(cum_flow[i] + cum_flow[i-1]) / 2 if i > 0 else cum_flow[i] for i in range(len(cum_flow))]

    # --- Detrending & Normalization ---
    detrend_win = int(params["detrend_window"] * effective_fps)
    disc_threshold = 1000 #float(params.get("discontinuity_threshold", 0.1))  # tweak as needed

    detrended_data = np.zeros_like(cum_flow)
    weight_sum = np.zeros_like(cum_flow)

    # Find indices where a jump occurs
    disc_indices = np.where(np.abs(np.diff(cum_flow)) > disc_threshold)[0] + 1
    # Break data into continuous segments
    segment_boundaries = [0] + list(disc_indices) + [len(cum_flow)]

    overlap = detrend_win // 2

    for i in range(len(segment_boundaries) - 1):
        seg_start = segment_boundaries[i]
        seg_end = segment_boundaries[i + 1]
        seg_length = seg_end - seg_start

        # Just subtract the average  segments with too few points to converge with polypit
        if seg_length < 5:
            detrended_data[seg_start:seg_end] = cum_flow[seg_start:seg_end] - np.mean(cum_flow[seg_start:seg_end])
            continue
        if seg_length <= detrend_win:
            # If segment is too short, process it in one go
            segment = cum_flow[seg_start:seg_end]
            x = np.arange(len(segment))
            trend = np.polyfit(x, segment, 1)
            detrended_segment = segment - np.polyval(trend, x)
            weights = np.hanning(len(segment))
            detrended_data[seg_start:seg_end] += detrended_segment * weights
            weight_sum[seg_start:seg_end] += weights
        else:
            # Process long segments in overlapping windows
            for start in range(seg_start, seg_end - overlap, overlap):
                end = min(start + detrend_win, seg_end)
                segment = cum_flow[start:end]
                x = np.arange(len(segment))
                trend = np.polyfit(x, segment, 1)
                detrended_segment = segment - np.polyval(trend, x)
                weights = np.hanning(len(segment))
                detrended_data[start:end] += detrended_segment * weights
                weight_sum[start:end] += weights

    # Normalize by weight sum to blend overlapping windows
    detrended_data /= np.maximum(weight_sum, 1e-6)

    smoothed_data = np.convolve(detrended_data, [1/16, 1/4, 3/8, 1/4, 1/16], mode='same')
    # Normalize each window in normalization_window to 0-100
    norm_win = int(params["norm_window"] * effective_fps)
    if norm_win % 2 == 0:
        norm_win += 1
    half_norm = norm_win // 2
    norm_rolling = np.empty_like(smoothed_data)
    for i in range(len(smoothed_data)):
        start_idx = max(0, i - half_norm)
        end_idx = min(len(smoothed_data), i + half_norm + 1)
        local_window = smoothed_data[start_idx:end_idx]
        local_min = local_window.min()
        local_max = local_window.max()
        if local_max - local_min == 0:
            norm_rolling[i] = 50
        else:
            norm_rolling[i] = (smoothed_data[i] - local_min) / (local_max - local_min) * 100
    
    # Sine fit (aborted experiment, left here for reference)
    #sine = sine_fit(norm_rolling)

    #Plot sine against norm_rolling
    # import matplotlib.pyplot as plt
    # plt.plot(norm_rolling)
    # #Smooth final con
    # smoothed_final_con = np.convolve(final_con_list, [1/16, 1/4, 3/8, 1/4, 1/16], mode='same')
    # plt.plot(smoothed_final_con)
    # plt.show()

    # TEST: Raw data
    #norm_rolling = sine

    # 3. Keyframe Reduction. Just use slope inversions for now.
    if(params["keyframe_reduction"]):
        key_indices = [0]
        for i in range(1, len(norm_rolling) - 1):
            d1 = norm_rolling[i] - norm_rolling[i - 1]
            d2 = norm_rolling[i + 1] - norm_rolling[i]
            
            if (d1 < 0) != (d2 < 0):
                key_indices.append(i)
        key_indices.append(len(norm_rolling) - 1)
    else:
        key_indices = range(len(norm_rolling))
    actions = []
    for ki in key_indices:  
        try:
            timestamp_ms = int(((time_stamps[ki]) / fps) * 1000)
            pos = int(round(norm_rolling[ki]))
            actions.append({"at": timestamp_ms, "pos": 100-pos})
        except Exception as e:
            log_func(f"Error computing action at segment index {ki}: {e}")
            error_occurred = True
    final_actions = actions

    log_func(f"Keyframe reduction: {len(final_actions)} actions computed.")
    actions = final_actions

    funscript = {"version": "1.0", "actions": actions}
    try:
        with open(output_path, "w") as f:
            json.dump(funscript, f, indent=2)
        log_func(STRINGS["funscript_saved"].format(output_path=output_path))
    except Exception as e:
        log_func(STRINGS["log_error"].format(error=str(e)))
        error_occurred = True
    
    # Log processing time
    elapsed_time = time.time() - start_time
    log_func(f"Processing time: {elapsed_time:.2f} seconds")
    
    return error_occurred