"""Worker threads for GUI processing"""

import os
import time
import threading
from datetime import datetime
from PySide6.QtCore import QThread, Signal

from ..config.localization import STRINGS
from ..core.video_processing import process_video


class WorkerThread(QThread):
    """Thread for processing videos in background."""
    progressChanged = Signal(int)
    videoProgressChanged = Signal(int)
    finished = Signal(bool, str, list, list)  # error_occurred, time_str, log_messages, generated_files
    logMessage = Signal(str)
    
    def __init__(self, files, settings):
        super().__init__()
        self.files = files
        self.settings = settings
        self.cancel_event = threading.Event()
        self.log_messages = []
        self.log_file = None
        self.generated_files = []  # Track (video_path, funscript_path) pairs
        
    def log(self, msg):
        self.log_messages.append(msg)
        if self.log_file:
            self.log_file.write(msg + "\n")
            self.log_file.flush()
        self.logMessage.emit(msg)
    
    def cancel(self):
        self.cancel_event.set()
    
    def run(self):
        error_occurred = False
        
        try:
            # Create logs folder
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../logs")
            os.makedirs(log_path, exist_ok=True)
            
            # Create timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = os.path.join(log_path, f"{timestamp}.log")
            self.log_file = open(log_filename, "w")
        except Exception as e:
            self.finished.emit(True, "0s", [f"Cannot open log file: {e}"], [])
            return
            
        batch_start_time = time.time()
        total_files = len(self.files)
        
        for idx, video in enumerate(self.files):
            if self.cancel_event.is_set():
                self.log(STRINGS["cancelled_by_user"])
                break
                
            self.videoProgressChanged.emit(0)

            # Generate the expected output path
            base = os.path.splitext(video)[0]
            funscript_path = base + ".funscript"

            err = process_video(video, self.settings, self.log,
                              progress_callback=lambda prog: self.videoProgressChanged.emit(prog),
                              cancel_flag=lambda: self.cancel_event.is_set())
            if err:
                error_occurred = True
            else:
                # If processing succeeded, track the generated file
                self.generated_files.append((video, funscript_path))
                
            overall = int(100 * (idx + 1) / total_files)
            self.progressChanged.emit(overall)
        
        # Calculate and format total time
        total_time = time.time() - batch_start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"
        
        self.log(f"{STRINGS['batch_processing_complete']} Total time: {time_str}")
        if self.log_file:
            self.log_file.close()
            
        self.finished.emit(error_occurred, time_str, self.log_messages, self.generated_files)