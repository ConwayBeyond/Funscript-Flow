"""Localization and string loading functionality"""

import json


def load_strings(filename="strings.json"):
    defaults = {
        "app_title": "Funscript Flow",
        "select_videos": "Select Videos",
        "select_folder": "Select Folder",
        "no_files_selected": "No files selected",
        "vr_mode": "VR Mode",
        "vr_mode_tooltip": ("Use this to improve accuracy for VR videos."),
        "overall_progress": "Overall Progress:",
        "current_video_progress": "Current Video Progress:",
        "advanced_settings": "Advanced Settings",
        "threads": "Threads:",
        "detrend_window": "Detrend window (sec):",
        
        "norm_window": "Norm window (sec):",
        "batch_size": "Batch size (frames):",
        "face_inversion": "Enable face‑based inversion",
        "show_preview": "Show Preview",
        "show_advanced": "Show Advanced Settings",
        "overwrite_files": "Overwrite existing files",
        "run": "Run",
        "cancel": "Cancel",
        "readme": "Readme",
        "config_saved": "Config saved to {config_path}",
        "config_load_error": "Error loading config: {error}",
        "no_files_warning": "Please select one or more video files or a folder.",
        "cancelled_by_user": "Processing cancelled by user.",
        "batch_processing_complete": "Batch processing complete.",
        "funscript_saved": "Funscript saved: {output_path}",
        "skipping_file_exists": "Skipping {video_path}: {output_path} exists.",
        "log_error": "ERROR: Could not write output: {error}",
        "found_files": "Found {n} file(s).",
        "processing_file": "--- Processing file {current}/{total}: {video_path} ---",
        "processing_completed_with_errors": "Processing completed with errors. See run.log for details.",
        "face_inversion_tooltip": "Uses face detection to try to determine the angle of motion, and adjust direction accordingly.",
        "pov_mode_tooltip": "Use this to improve stability for POV videos.",
    }
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return defaults


STRINGS = load_strings()