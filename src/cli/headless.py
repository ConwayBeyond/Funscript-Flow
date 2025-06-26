"""Command-line interface for headless processing"""

import os
from ..config.settings import SUPPORTED_VIDEO_EXTENSIONS
from ..config.localization import STRINGS
from ..core.video_processing import process_video


def run_headless(input_path, settings):
    """Run video processing in headless mode."""
    log_filename = "run.log"
    try:
        logf = open(log_filename, "w")
    except Exception as e:
        print(f"Error opening log file: {e}")
        return
    def log_func(msg):
        logf.write(msg + "\n")
        logf.flush()
        print(msg)
    if os.path.isdir(input_path):
        files = []
        for root, dirs, files_in in os.walk(input_path):
            for f in files_in:
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_VIDEO_EXTENSIONS:
                    files.append(os.path.join(root, f))
    else:
        files = [input_path]
    if not files:
        print("No video files found.")
        logf.write("No video files found.\n")
        logf.close()
        return
    total_files = len(files)
    log_func(STRINGS["found_files"].format(n=total_files))
    for idx, video in enumerate(files):
        log_func(STRINGS["processing_file"].format(current=idx+1, total=total_files, video_path=video))
        process_video(video, settings, log_func, progress_callback=lambda prog: print(f"Video progress: {prog}%"))
    log_func(STRINGS["batch_processing_complete"])
    logf.close()
    print("Done. See run.log for details.")