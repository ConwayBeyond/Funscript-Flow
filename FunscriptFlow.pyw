#!/usr/bin/env python3

import gc
import os, math, threading, concurrent.futures, json, argparse, time, sys
import numpy as np
import cv2
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QCheckBox, QProgressBar, QLineEdit, 
                               QFileDialog, QMessageBox, QTextEdit, QDialog, QComboBox,
                               QFrame, QScrollArea, QTabWidget, QSlider, QGroupBox, QFormLayout)
from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QPixmap, QIcon, QFont
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from multiprocessing import Pool
from datetime import datetime
import asyncio
from queue import Queue, Empty
from typing import List, Tuple, Optional, Dict
import numpy.typing as npt


# ---------- Constants ----------
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".wmv", ".flv", ".mpg", ".mpeg", ".ts"}
SUPPORTED_VIDEO_PATTERNS = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_VIDEO_EXTENSIONS))

# ---------- GPU Backend Detection ----------
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

# ---------- Async Video Reader with Buffering ----------
class AsyncVideoReader:
    """High-performance async video reader with frame buffering and parallel decoding."""
    
    def __init__(self, video_path: str, width: Optional[int] = None, height: Optional[int] = None, 
                 num_threads: int = 4, buffer_size: int = 100):
        self.video_path = video_path
        self.width = width
        self.height = height
        self.num_threads = min(num_threads, 4)  # Limit decoder threads
        self.buffer_size = buffer_size
        
        # Test video can be opened
        test_cap = cv2.VideoCapture(video_path)
        if not test_cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        self.total_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = test_cap.get(cv2.CAP_PROP_FPS)
        self.orig_width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        test_cap.release()
        
        # Frame buffer and memory pool
        self.frame_buffer = Queue(maxsize=buffer_size)
        self.frame_pool = Queue()
        
        # Pre-allocate frame buffers
        for _ in range(buffer_size):
            if self.width and self.height:
                frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
            else:
                frame = np.empty((self.orig_height, self.orig_width, 3), dtype=np.uint8)
            self.frame_pool.put(frame)
        
        # Decoder threads and state
        self.decoders = []
        self.decoder_locks = []
        self.stop_event = threading.Event()
        self.prefetch_thread = None
        
        # Initialize multiple decoders
        for i in range(self.num_threads):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal internal buffering
            self.decoders.append(cap)
            self.decoder_locks.append(threading.Lock())
    
    def __len__(self):
        return self.total_frames
    
    def get_avg_fps(self):
        return self.fps
    
    def _get_frame_buffer(self) -> np.ndarray:
        """Get a frame buffer from the pool or allocate a new one."""
        try:
            return self.frame_pool.get_nowait()
        except Empty:
            if self.width and self.height:
                return np.empty((self.height, self.width, 3), dtype=np.uint8)
            else:
                return np.empty((self.orig_height, self.orig_width, 3), dtype=np.uint8)
    
    def _return_frame_buffer(self, frame: np.ndarray):
        """Return a frame buffer to the pool."""
        try:
            self.frame_pool.put_nowait(frame)
        except:
            pass  # Pool is full, let GC handle it
    
    def _decode_frame(self, idx: int, decoder_id: int) -> Optional[np.ndarray]:
        """Decode a single frame using the specified decoder."""
        with self.decoder_locks[decoder_id]:
            cap = self.decoders[decoder_id]
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB in-place
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, frame)
                
                # Resize if needed
                if self.width and self.height and (frame.shape[1] != self.width or frame.shape[0] != self.height):
                    frame = cv2.resize(frame, (self.width, self.height))
                
                return frame
            return None
    
    def _prefetch_worker(self, indices: List[int]):
        """Worker thread that prefetches frames into the buffer."""
        decoder_id = 0
        
        for idx in indices:
            if self.stop_event.is_set():
                break
            
            frame = self._decode_frame(idx, decoder_id)
            if frame is not None:
                self.frame_buffer.put((idx, frame))
            
            # Round-robin through decoders
            decoder_id = (decoder_id + 1) % self.num_threads
    
    def get_batch(self, indices: List[int]) -> np.ndarray:
        """Get a batch of frames by indices with parallel decoding."""
        frames = [None] * len(indices)
        
        # Start prefetch thread
        self.stop_event.clear()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, args=(indices,))
        self.prefetch_thread.start()
        
        # Collect frames from buffer
        received = 0
        timeout = 10.0  # seconds
        
        while received < len(indices):
            try:
                idx, frame = self.frame_buffer.get(timeout=timeout)
                # Find position in indices
                try:
                    pos = indices.index(idx)
                    frames[pos] = frame
                    received += 1
                except ValueError:
                    # Frame not in requested indices, return to pool
                    self._return_frame_buffer(frame)
            except Empty:
                break
        
        # Stop prefetch and wait
        self.stop_event.set()
        if self.prefetch_thread:
            self.prefetch_thread.join()
        
        # Fill any missing frames with black
        for i, frame in enumerate(frames):
            if frame is None:
                if self.width and self.height:
                    frames[i] = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                else:
                    frames[i] = np.zeros((self.orig_height, self.orig_width, 3), dtype=np.uint8)
        
        return np.array(frames)
    
    def get_batch_parallel(self, indices: List[int]) -> np.ndarray:
        """Alternative parallel batch reading using thread pool."""
        frames = [None] * len(indices)
        
        def decode_with_id(args):
            idx, pos, decoder_id = args
            frame = self._decode_frame(idx, decoder_id)
            return pos, frame
        
        # Distribute indices across decoders
        tasks = []
        for i, idx in enumerate(indices):
            decoder_id = i % self.num_threads
            tasks.append((idx, i, decoder_id))
        
        # Decode in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(decode_with_id, task) for task in tasks]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    pos, frame = future.result()
                    frames[pos] = frame
                except Exception as e:
                    pass
        
        # Fill missing frames
        for i, frame in enumerate(frames):
            if frame is None:
                if self.width and self.height:
                    frames[i] = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                else:
                    frames[i] = np.zeros((self.orig_height, self.orig_width, 3), dtype=np.uint8)
        
        return np.array(frames)
    
    def __del__(self):
        self.stop_event.set()
        if self.prefetch_thread:
            self.prefetch_thread.join()
        
        for cap in self.decoders:
            if cap:
                cap.release()

# ---------- OpenCV VideoReader wrapper to replace decord ----------
class VideoReaderCV:
    def __init__(self, video_path, width=None, height=None, num_threads=None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = width
        self.height = height
        
        # Get original dimensions
        self.orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def __len__(self):
        return self.total_frames
    
    def get_avg_fps(self):
        return self.fps
    
    def get_batch(self, indices):
        frames = []
        for idx in indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize if needed
                if self.width is not None and self.height is not None:
                    frame = cv2.resize(frame, (self.width, self.height))
                frames.append(frame)
            else:
                # If frame reading fails, create a black frame
                if self.width is not None and self.height is not None:
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                else:
                    frame = np.zeros((self.orig_height, self.orig_width, 3), dtype=np.uint8)
                frames.append(frame)
        
        # Return as numpy array to mimic decord's asnumpy() behavior
        return np.array(frames)
    
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()


# ---------- Localization Strings ----------
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

# ---------- Tooltip Implementation ----------
class ToolTip:
    """Simple tooltip for Qt widgets using setToolTip."""
    def __init__(self, widget, text="widget info"):
        widget.setToolTip(text)

class FunscriptVisualizerWidget(QWidget):
    """Widget for visualizing funscript data in real-time."""
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(12, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        # Create the plot
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(0, 100)  # Will be updated with actual data
        self.ax.set_ylim(0, 100)
        self.ax.set_xlabel('Time (seconds)')
        self.ax.set_ylabel('Position (0-100)')
        self.ax.set_title('Funscript Visualization')
        self.ax.grid(True, alpha=0.3)
        
        # Data storage
        self.funscript_data = None
        self.current_time = 0
        self.line = None
        self.current_marker = None
        
    def load_funscript(self, funscript_path):
        """Load funscript data from file."""
        try:
            with open(funscript_path, 'r') as f:
                data = json.load(f)
            
            self.funscript_data = data['actions']
            
            # Extract time and position data
            times = [action['at'] / 1000.0 for action in self.funscript_data]  # Convert ms to seconds
            positions = [action['pos'] for action in self.funscript_data]
            
            # Update plot
            self.ax.clear()
            self.ax.set_xlabel('Time (seconds)')
            self.ax.set_ylabel('Position (0-100)')
            self.ax.set_title('Funscript Visualization')
            self.ax.grid(True, alpha=0.3)
            
            if times:
                self.ax.set_xlim(0, max(times))
                self.line, = self.ax.plot(times, positions, 'b-', linewidth=2, alpha=0.7)
                self.current_marker, = self.ax.plot([], [], 'ro', markersize=8)
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error loading funscript: {e}")
    
    def update_current_time(self, time_seconds):
        """Update the current time marker on the plot."""
        self.current_time = time_seconds
        
        if self.current_marker is not None and self.funscript_data:
            # Find the closest action point for interpolation
            closest_pos = 50  # Default position
            
            # Linear interpolation between adjacent points
            for i in range(len(self.funscript_data) - 1):
                t1 = self.funscript_data[i]['at'] / 1000.0
                t2 = self.funscript_data[i + 1]['at'] / 1000.0
                
                if t1 <= time_seconds <= t2:
                    p1 = self.funscript_data[i]['pos']
                    p2 = self.funscript_data[i + 1]['pos']
                    # Linear interpolation
                    ratio = (time_seconds - t1) / (t2 - t1) if t2 != t1 else 0
                    closest_pos = p1 + ratio * (p2 - p1)
                    break
            
            self.current_marker.set_data([time_seconds], [closest_pos])
            self.canvas.draw_idle()

class VideoPlayerWidget(QWidget):
    """Video player widget with funscript synchronization."""
    timeChanged = Signal(float)  # Emits current time in seconds
    
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # Video widget
        self.video_widget = QVideoWidget()
        self.layout.addWidget(self.video_widget)
        
        # Media player
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        
        # Controls
        self.controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        self.controls_layout.addWidget(self.play_button)
        
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.sliderMoved.connect(self.set_position)
        self.controls_layout.addWidget(self.position_slider)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.controls_layout.addWidget(self.time_label)
        
        self.layout.addLayout(self.controls_layout)
        
        # Connect signals
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.playbackStateChanged.connect(self.state_changed)
        
    def load_video(self, video_path):
        """Load a video file."""
        self.media_player.setSource(QUrl.fromLocalFile(video_path))
        
    def toggle_play(self):
        """Toggle play/pause."""
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()
    
    def set_position(self, position):
        """Set playback position."""
        self.media_player.setPosition(position)
    
    def duration_changed(self, duration):
        """Update slider maximum when duration changes."""
        self.position_slider.setMaximum(duration)
        
    def position_changed(self, position):
        """Update UI when position changes."""
        self.position_slider.setValue(position)
        
        # Update time label
        current_time = self.format_time(position)
        total_time = self.format_time(self.media_player.duration())
        self.time_label.setText(f"{current_time} / {total_time}")
        
        # Emit time changed signal for synchronization
        self.timeChanged.emit(position / 1000.0)  # Convert ms to seconds
        
    def state_changed(self, state):
        """Update play button text when state changes."""
        if state == QMediaPlayer.PlayingState:
            self.play_button.setText("Pause")
        else:
            self.play_button.setText("Play")
    
    def format_time(self, milliseconds):
        """Format time in mm:ss format."""
        seconds = milliseconds // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

class VideoPlayerDialog(QDialog):
    """Dialog for video playback with funscript visualization."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Player with Funscript Visualization")
        self.setModal(False)
        self.resize(1200, 800)
        
        # Application icon is inherited from QApplication
        
        layout = QVBoxLayout(self)
        
        # Create splitter for video and visualization
        splitter_widget = QWidget()
        splitter_layout = QHBoxLayout(splitter_widget)
        
        # Video section
        video_section = QWidget()
        video_layout = QVBoxLayout(video_section)
        video_layout.addWidget(QLabel("Video Player"))
        
        self.video_player = VideoPlayerWidget()
        video_layout.addWidget(self.video_player)
        
        # Visualization section
        viz_section = QWidget()
        viz_layout = QVBoxLayout(viz_section)
        viz_layout.addWidget(QLabel("Funscript Visualization"))
        
        self.funscript_visualizer = FunscriptVisualizerWidget()
        viz_layout.addWidget(self.funscript_visualizer)
        
        # Add sections to splitter
        splitter_layout.addWidget(video_section, 1)  # Video takes more space
        splitter_layout.addWidget(viz_section, 1)
        
        layout.addWidget(splitter_widget)
        
        # Connect video time changes to visualization updates
        self.video_player.timeChanged.connect(self.funscript_visualizer.update_current_time)
        
        # File loading controls
        controls_layout = QHBoxLayout()
        
        load_video_btn = QPushButton("Load Video")
        load_video_btn.clicked.connect(self.load_video)
        controls_layout.addWidget(load_video_btn)
        
        load_funscript_btn = QPushButton("Load Funscript")
        load_funscript_btn.clicked.connect(self.load_funscript)
        controls_layout.addWidget(load_funscript_btn)
        
        layout.addLayout(controls_layout)
        
    def load_video(self):
        """Load a video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.m4v *.webm *.wmv *.flv *.mpg *.mpeg *.ts);;All Files (*)"
        )
        if file_path:
            self.video_player.load_video(file_path)
    
    def load_funscript(self):
        """Load a funscript file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Funscript File", "", 
            "Funscript Files (*.funscript);;JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.funscript_visualizer.load_funscript(file_path)
    
    def load_video_and_funscript(self, video_path, funscript_path):
        """Load both video and funscript files."""
        self.video_player.load_video(video_path)
        self.funscript_visualizer.load_funscript(funscript_path)

def detect_cut(pair, log_func=None, threshold=30):
    return False
    prev_frame, curr_frame = pair
    diff = np.mean(np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32)))
    if(log_func != None and diff > threshold):
        log_func(f"Found a cut at " + str(diff))
        
    return diff > threshold


# ---------- Original flow functions (for VR mode) ------------
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
    prev_frame, curr_frame = pair
    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    avg_flow = np.mean(-flow[..., 0] + flow[..., 1])
    return avg_flow

def compute_flow_nonvr(pair):
    prev_frame, curr_frame = pair
    flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    avg_flow = np.mean(flow[..., 0] + flow[..., 1])
    return avg_flow

# --- PSO mode ---

def center_of_mass_variance(flow, num_cells=32):
    """
    Splits the optical flow into a configurable grid (num_cells x num_cells), 
    computes the variance of the optical flow in each grid cell, 
    and returns the center of mass of the variance.
    """
    h, w, _ = flow.shape
    grid_h, grid_w = h // num_cells, w // num_cells

    variance_grid = np.zeros((num_cells, num_cells))
    y_coords, x_coords = np.meshgrid(np.arange(num_cells), np.arange(num_cells), indexing='ij')

    for i in range(num_cells):
        for j in range(num_cells):
            cell = flow[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
            magnitude = np.sqrt(cell[..., 0]**2 + cell[..., 1]**2)
            variance_grid[i, j] = np.var(magnitude)

    total_variance = np.sum(variance_grid)

    if total_variance == 0:
        return (w // 2, h // 2)  # Default to image center if no variance
    else:
        center_x = np.sum(x_coords * variance_grid) * grid_w / total_variance + grid_w / 2
        center_y = np.sum(y_coords * variance_grid) * grid_h / total_variance + grid_h / 2
        return (center_x, center_y)

def max_divergence(flow):
    """
    Computes the divergence of the optical flow over the whole image and returns
    the pixel (x, y) with the highest absolute divergence along with its value.
    """
    # No grid, just pure per-pixel divergence!
    div = np.gradient(flow[..., 0], axis=0) + np.gradient(flow[..., 1], axis=1)
    
    # Get the index (y, x) of the max abs divergence
    y, x = np.unravel_index(np.argmax(np.abs(div)), div.shape)
    return x, y, div[y, x]


def radial_motion_weighted(flow, center, is_cut, pov_mode=False):
    """
    Computes signed radial motion: positive for outward motion, negative for inward motion.
    Closer pixels have higher weight.
    """
    if(is_cut):
        return 0.0
    h, w, _ = flow.shape
    y, x = np.indices((h, w))
    dx = x - center[0]
    dy = y - center[1]

    dot = flow[..., 0] * dx + flow[..., 1] * dy

    # In POV mode, just return the mean dot product
    if(pov_mode):
        return np.mean(dot)
    
    #Cancel out global motion by balancing the averages
    # multiply products to the right of the center (w-x) / w and to the left by x / w
    weighted_dot = np.where(x > center[0], dot * (w - x) / w, dot * x / w)
    # multiply products below the center (h-y) / h and above by y / h
    weighted_dot = np.where(y > center[1], weighted_dot * (h - y) / h, weighted_dot * y / h)

    return np.mean(weighted_dot)



def largest_cluster_center(positions, threshold=10.0):
    """
    BFS to find the largest cluster of swarm positions, return its centroid.
    """
    num_particles = len(positions)
    adj = [[] for _ in range(num_particles)]
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            if np.linalg.norm(positions[i] - positions[j]) < threshold:
                adj[i].append(j)
                adj[j].append(i)

    visited = set()
    clusters = []
    def bfs(start):
        queue, c = [start], []
        while queue:
            node = queue.pop()
            if node in visited: continue
            visited.add(node)
            c.append(node)
            for nei in adj[node]:
                if nei not in visited:
                    queue.append(nei)
        return c

    for i in range(num_particles):
        if i not in visited:
            group = bfs(i)
            clusters.append(group)

    biggest = max(clusters, key=len)
    return (np.mean(positions[biggest], axis=0), len(biggest))

def swarm_positions(flow, num_particles=30, iterations=50):
    """
    Moves 'num_particles' along 'flow' for 'iterations'. Return final positions for clustering.
    """
    h, w, _ = flow.shape
    positions = np.column_stack([
        np.random.uniform(0, w, num_particles),
        np.random.uniform(0, h, num_particles)
    ])
    for _ in range(iterations):
        for i in range(num_particles):
            x_i = int(np.clip(positions[i, 0], 0, w - 1))
            y_i = int(np.clip(positions[i, 1], 0, h - 1))
            vx = flow[y_i, x_i, 1]
            vy = flow[y_i, x_i, 0]
            positions[i, 0] = np.clip(positions[i, 0] + vx, 0, w - 1)
            positions[i, 1] = np.clip(positions[i, 1] + vy, 0, h - 1)
    return positions


def precompute_flow_info(p0, p1, config):
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
            return precompute_flow_info_gpu(p0, p1, config.get("cut_threshold", 7))
        except Exception as e:
            # Fall back to CPU if GPU fails
            pass
    elif backend == "OpenCL":
        try:
            return precompute_flow_info_opencl(p0, p1, config)
        except Exception as e:
            # Fall back to CPU if OpenCL fails
            pass
    elif backend == "DNN":
        try:
            return precompute_flow_info_dnn(p0, p1, config)
        except Exception as e:
            # Fall back to CPU if DNN fails
            pass
    
    # CPU implementation (default)
    cut_threshold = config.get("cut_threshold", 7)
    
    flow = cv2.calcOpticalFlowFarneback(p0, p1, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    if(config.get("pov_mode")):
        # In pov mode, just use the center of the bottom edge of the frame
        max = (p0.shape[1] // 2, p0.shape[0] - 1, 0)
    else:
        max = max_divergence(flow)
    pos_center = max[0:2]
    val_pos = max[2]
        
    # Detect cut based on flow map
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    mean_mag = np.mean(mag)
    if mean_mag > cut_threshold:
        is_cut = True
    else:
        is_cut = False

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

def precompute_flow_info_opencl(p0, p1, config):
    """OpenCL-accelerated optical flow computation using UMat."""
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

def precompute_flow_info_dnn(p0, p1, config):
    """DNN-accelerated optical flow using DisFlow algorithm."""
    cut_threshold = config.get("cut_threshold", 7)
    
    # Use DIS optical flow (faster than Farneback)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    
    # Compute flow
    flow = dis.calc(p0, p1, None)
    
    # Rest is same as CPU version
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

def precompute_flow_info_gpu(p0, p1, cut_threshold):
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

def precompute_wrapper(p, params):
    return precompute_flow_info(
        p[0], p[1], params)

def fetch_frames(video_path, chunk, params):
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

# ---------- Main Processing Function ----------
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

        # progress
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

# ---------- Sine Fit ------------

def sine_fit(data, error_threshold=5000.0, gain=2.0, min_points=3, max_points=30):
    """
    Fits half-wave sine segments (center=50) to `data` by testing candidate endpoints
    from min_points to max_points ahead. After segmentation, if two consecutive segments
    have the same sign, we split them with an inserted corrective half-wave (with inverted amplitude)
    to help catch missed alternations.

    Returns the fitted array.
    """
    n = len(data)
    segments = []  # each segment is a dict with {'start', 'end', 'A'}
    start = 0

    # --- First pass: Segment the data ---
    while start < n - 1:
        best_err = np.inf
        best_end = None
        best_A = 0.0

        for seg_len in range(min_points, max_points + 1):
            end = start + seg_len
            if end >= n:
                break

            T = seg_len  # segment length (points between endpoints)
            x = np.arange(T + 1)
            model = np.sin(np.pi * x / T)
            segment = data[start:end + 1]
            denom = np.sum(model**2)
            if denom == 0:
                continue
            # Linear LS solution for amplitude A.
            A = np.sum(model * (segment - 50)) / denom
            fit = 50 + A * model
            err = np.sqrt(np.mean((segment - fit) ** 2))

            if err < best_err:
                best_err = err
                best_end = end
                best_A = A

        if best_end is None:
            break

        # Error correction: if error too high, flatten the segment.
        if best_err > error_threshold:
            best_A = 0.0
        # Boost low amplitude segments (because sometimes they're just shy).
        #best_A = np.sign(best_A) * (abs(best_A) ** (1.0 / gain))

        segments.append({'start': start, 'end': best_end, 'A': best_A})
        start = best_end

    # --- Second pass: Correction for consecutive segments with the same sign ---
    corrected_segments = []
    i = 0
    while i < len(segments):
        # If the next segment exists and both segments have nonzero, same-signed amplitude...
        if (i < len(segments) - 1 and segments[i]['A'] != 0 and segments[i+1]['A'] != 0 and
            np.sign(segments[i]['A']) == np.sign(segments[i+1]['A'])):
            combined_start = segments[i]['start']
            combined_end = segments[i+1]['end']
            if (combined_end - combined_start) >= min_points*2:
                L = combined_end - combined_start
                # Split the combined region into three parts.
                mid1 = combined_start + L // 3
                mid2 = combined_start + 2 * L // 3

                # Re-fit first sub-segment.
                T1 = mid1 - combined_start
                if T1 < 2:
                    T1 = 2
                    mid1 = combined_start + T1
                x1 = np.arange(T1 + 1)
                model1 = np.sin(np.pi * x1 / T1)
                seg1 = data[combined_start:mid1 + 1]
                denom1 = np.sum(model1 ** 2)
                A1 = np.sum(model1 * (seg1 - 50)) / denom1 if denom1 != 0 else 0

                # Re-fit third sub-segment.
                T3 = combined_end - mid2
                if T3 < 2:
                    T3 = 2
                    mid2 = combined_end - T3
                x3 = np.arange(T3 + 1)
                model3 = np.sin(np.pi * x3 / T3)
                seg3 = data[mid2:combined_end + 1]
                denom3 = np.sum(model3 ** 2)
                A3 = np.sum(model3 * (seg3 - 50)) / denom3 if denom3 != 0 else 0

                # Corrective (middle) segment: force amplitude opposite in sign.
                A2 = -np.sign(segments[i]['A']) * (0.5 * (abs(A1) + abs(A3)))

                corrected_segments.append({'start': combined_start, 'end': mid1, 'A': A1})
                corrected_segments.append({'start': mid1, 'end': mid2, 'A': A2})
                corrected_segments.append({'start': mid2, 'end': combined_end, 'A': A3})
                i += 2  # skip the next segment; we've merged it
                continue
            else:
                #Comvine them into one segment
                combined_A = segments[i]['A'] + segments[i+1]['A']
                combined_start = segments[i]['start']
                combined_end = segments[i+1]['end']
                corrected_segments.append({'start': combined_start, 'end': combined_end, 'A': combined_A})
                i += 2
                continue

        corrected_segments.append(segments[i])
        i += 1

    # --- Third pass: Detect and fix missed periods ---
    final_segments = []
    for j in range(len(corrected_segments)):
        if j > 0 and j < len(corrected_segments) - 1:
            
            prev_L = corrected_segments[j-1]['end'] - corrected_segments[j-1]['start']
            curr_L = corrected_segments[j]['end'] - corrected_segments[j]['start']
            next_L = corrected_segments[j+1]['end'] - corrected_segments[j+1]['start']
            
            if curr_L > prev_L + next_L:
                # Split into a number of segments depending on the calculated number of missed periods
                missed_periods = round(curr_L / (prev_L + next_L))

                segment_splits = np.linspace(corrected_segments[j]['start'], corrected_segments[j]['end'], 2*missed_periods + 1, dtype=int)
                invert = False
                for split_idx in range(len(segment_splits) - 1):
                    split_segment = {'start': segment_splits[split_idx], 'end': segment_splits[split_idx + 1], 'A': corrected_segments[j]['A'] * (-1 if invert else 1)}
                    invert = not invert
                    final_segments.append(split_segment)
                continue
        final_segments.append(corrected_segments[j])
    #plot the rolling variance of segment lengths, with outliers flagged
    segment_lengths = [seg['end'] - seg['start'] for seg in final_segments]
    # Calculate the rolling variance of segment lengths in a window of 5 segments
    rolling_var = np.full(len(segment_lengths), np.nan)
    for i in range(2, len(segment_lengths) - 2):
        rolling_var[i] = np.var(segment_lengths[i-2:i+3])
    # Flag outliers (variance > 1.5 * median variance)
    var_threshold = 1.5 * np.nanmedian(rolling_var)
    for i in range(len(rolling_var)):
        if rolling_var[i] > var_threshold:
            final_segments[i]['outlier'] = True
    
    # plt.figure(figsize=(10, 5))
    # plt.plot(rolling_var, label='Variance', marker='o')
    # plt.axhline(y=np.mean(rolling_var), color='r', linestyle='--', label='Mean Segment Length')
    # plt.axhline(y=var_threshold, color='g', linestyle=':', label='Variance Threshold', xmin=0, xmax=len(segment_lengths)-1)
    # plt.title('Rolling Variance of Segment Lengths with Outliers Flagged')
    # plt.xlabel('Segment Index')
    # plt.ylabel('Length')
    # plt.legend()
    # plt.show()

    # --- Build the fitted curve from the corrected segments ---
    fitted = np.full(n, 50.0)
    for seg in final_segments:
        s, e = seg['start'], seg['end']
        T = e - s
        if T < 1:
            continue
        x_seg = np.arange(T + 1)
        fitted[s:e + 1] = 50 + seg['A'] * np.sin(np.pi * x_seg / T)

    return fitted

# ---------- Preview Helper ----------
def convert_frame_to_photo(frame):
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        retval, buffer = cv2.imencode('.png', rgb)
        if not retval:
            return None
        img_data = buffer.tobytes()
        return tk.PhotoImage(data=img_data)
    except Exception:
        return None

# ---------- GUI Code ----------
# Old tkinter helper functions removed - no longer needed with PySide6

class WorkerThread(QThread):
    """Thread for processing videos in background."""
    progressChanged = Signal(int)
    videoProgressChanged = Signal(int)
    finished = Signal(bool, str, list)  # error_occurred, time_str, log_messages
    logMessage = Signal(str)
    
    def __init__(self, files, settings):
        super().__init__()
        self.files = files
        self.settings = settings
        self.cancel_event = threading.Event()
        self.log_messages = []
        self.log_file = None
        
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
            log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            os.makedirs(log_path, exist_ok=True)
            
            # Create timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = os.path.join(log_path, f"{timestamp}.log")
            self.log_file = open(log_filename, "w")
        except Exception as e:
            self.finished.emit(True, "0s", [f"Cannot open log file: {e}"])
            return
            
        batch_start_time = time.time()
        total_files = len(self.files)
        
        for idx, video in enumerate(self.files):
            if self.cancel_event.is_set():
                self.log(STRINGS["cancelled_by_user"])
                break
                
            self.videoProgressChanged.emit(0)
            err = process_video(video, self.settings, self.log,
                              progress_callback=lambda prog: self.videoProgressChanged.emit(prog),
                              cancel_flag=lambda: self.cancel_event.is_set())
            if err:
                error_occurred = True
                
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
            
        self.finished.emit(error_occurred, time_str, self.log_messages)

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(STRINGS["app_title"])
        self.setGeometry(100, 100, 800, 600)
        
        # Application icon is set at QApplication level
        
        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Initialize variables first
        self.files = []
        self.worker_thread = None
        self.params = {}
        self.backends = get_available_backends()
        self.available_backends = []
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Funscript Generation Tab
        self.generation_tab = QWidget()
        self.tab_widget.addTab(self.generation_tab, "Funscript Generation")
        self.setup_generation_tab()
        
        # Video Player Tab
        self.player_tab = VideoPlayerDialog()
        self.tab_widget.addTab(self.player_tab, "Video Player & Visualizer")
        
        # Load configuration
        self.load_config()
        
    def setup_generation_tab(self):
        """Setup the funscript generation tab."""
        layout = QVBoxLayout(self.generation_tab)
        
        # File selection section
        file_group = QGroupBox("File Selection")
        file_layout = QHBoxLayout(file_group)
        
        self.btn_select_files = QPushButton(STRINGS["select_videos"])
        self.btn_select_files.clicked.connect(self.select_files)
        file_layout.addWidget(self.btn_select_files)
        
        self.btn_select_folder = QPushButton(STRINGS["select_folder"])
        self.btn_select_folder.clicked.connect(self.select_folder)
        file_layout.addWidget(self.btn_select_folder)
        
        self.lbl_files = QLabel(STRINGS["no_files_selected"])
        file_layout.addWidget(self.lbl_files)
        
        file_layout.addStretch()
        
        self.btn_readme = QPushButton(STRINGS["readme"])
        self.btn_readme.clicked.connect(self.show_readme)
        file_layout.addWidget(self.btn_readme)
        
        layout.addWidget(file_group)
        
        # Mode selection section
        mode_group = QGroupBox("Mode Selection")
        mode_layout = QHBoxLayout(mode_group)
        
        self.chk_vr = QCheckBox(STRINGS["vr_mode"])
        ToolTip(self.chk_vr, STRINGS["vr_mode_tooltip"])
        mode_layout.addWidget(self.chk_vr)
        
        self.chk_pov = QCheckBox("POV Mode")
        ToolTip(self.chk_pov, STRINGS["pov_mode_tooltip"])
        mode_layout.addWidget(self.chk_pov)
        
        mode_layout.addStretch()
        layout.addWidget(mode_group)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        progress_layout.addWidget(QLabel(STRINGS["overall_progress"]))
        self.overall_progress = QProgressBar()
        self.overall_progress.setRange(0, 100)
        progress_layout.addWidget(self.overall_progress)
        
        progress_layout.addWidget(QLabel(STRINGS["current_video_progress"]))
        self.video_progress = QProgressBar()
        self.video_progress.setRange(0, 100)
        progress_layout.addWidget(self.video_progress)
        
        layout.addWidget(progress_group)
        
        # Advanced settings section (collapsible)
        self.adv_group = QGroupBox(STRINGS["advanced_settings"])
        self.adv_group.setCheckable(True)
        self.adv_group.setChecked(False)
        self.setup_advanced_settings()
        layout.addWidget(self.adv_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.btn_run = QPushButton(STRINGS["run"])
        self.btn_run.clicked.connect(self.run_batch)
        button_layout.addWidget(self.btn_run)
        
        self.btn_cancel = QPushButton(STRINGS["cancel"])
        self.btn_cancel.clicked.connect(self.cancel_run)
        self.btn_cancel.setEnabled(False)
        button_layout.addWidget(self.btn_cancel)
        
        # Add Preview Generated Funscript button
        self.btn_preview = QPushButton("Preview Generated Funscript")
        self.btn_preview.clicked.connect(self.preview_funscript)
        button_layout.addWidget(self.btn_preview)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        layout.addStretch()
        
    def setup_advanced_settings(self):
        """Setup advanced settings form."""
        layout = QFormLayout(self.adv_group)
        
        # Get the number of cores available
        num_cores = os.cpu_count()
        
        # Create parameter inputs
        self.params = {}
        
        self.params["threads"] = QLineEdit(str(num_cores))
        ToolTip(self.params["threads"], "Number of threads used for optical flow computation.")
        layout.addRow(STRINGS["threads"], self.params["threads"])
        
        self.params["detrend_window"] = QLineEdit("1.5")
        ToolTip(self.params["detrend_window"], "Controls the aggressiveness of drift removal. See readme for detail. Recommended: 1-10, higher values for more stable cameras.")
        layout.addRow(STRINGS["detrend_window"], self.params["detrend_window"])
        
        self.params["norm_window"] = QLineEdit("4")
        ToolTip(self.params["norm_window"], "Time window to calibrate motion range (seconds). Shorter values amplify motion, but also cause artifacts in long thrusts.")
        layout.addRow(STRINGS["norm_window"], self.params["norm_window"])
        
        self.params["batch_size"] = QLineEdit("3000")
        ToolTip(self.params["batch_size"], "Number of frames to process per batch (Higher values will be faster, but also take more RAM).")
        layout.addRow(STRINGS["batch_size"], self.params["batch_size"])
        
        # GPU Backend selection
        self.backend_combo = QComboBox()
        
        # Format backend options to show availability
        backend_display = []
        self.available_backends = []
        for backend, available in [("CPU", True), ("CUDA", self.backends["CUDA"]), 
                                   ("OpenCL", self.backends["OpenCL"]), ("DNN", self.backends["DNN"])]:
            if available:
                backend_display.append(backend)
                self.available_backends.append(backend)
            else:
                backend_display.append(f"{backend} (unavailable)")
        
        self.backend_combo.addItems(backend_display)
        self.backend_combo.setCurrentText("CPU")
        ToolTip(self.backend_combo, "Select processing backend. GPU acceleration can significantly speed up processing.\nUnavailable options require specific hardware or OpenCV build configuration.")
        layout.addRow("Processing Backend:", self.backend_combo)
        
        # Checkboxes
        self.chk_keyframe = QCheckBox("Enable keyframe reduction")
        self.chk_keyframe.setChecked(True)
        layout.addRow(self.chk_keyframe)
        
        self.chk_overwrite = QCheckBox(STRINGS["overwrite_files"])
        layout.addRow(self.chk_overwrite)
        
    def select_files(self):
        """Select video files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, STRINGS["select_videos"], "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.m4v *.webm *.wmv *.flv *.mpg *.mpeg *.ts);;All Files (*)"
        )
        if files:
            self.files = files
            self.lbl_files.setText(f"{len(self.files)} file(s) selected")
        else:
            self.files = []
            self.lbl_files.setText(STRINGS["no_files_selected"])
    
    def select_folder(self):
        """Select folder containing videos."""
        folder = QFileDialog.getExistingDirectory(self, STRINGS["select_folder"])
        if folder:
            found = []
            for root, dirs, files in os.walk(folder):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in SUPPORTED_VIDEO_EXTENSIONS:
                        found.append(os.path.join(root, f))
            self.files = found
            self.lbl_files.setText(f"{len(self.files)} file(s) found in folder")
    
    def show_readme(self):
        """Show readme in a dialog."""
        try:
            with open("readme.txt", "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            content = f"Error reading readme.txt: {e}"
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Readme")
        dialog.setModal(False)
        dialog.resize(600, 400)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        
        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setPlainText(content)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.show()
    
    def preview_funscript(self):
        """Open the video player tab to preview generated funscripts."""
        self.tab_widget.setCurrentIndex(1)  # Switch to video player tab
    
    def save_config(self):
        """Save configuration to file."""
        config = {key: widget.text() for key, widget in self.params.items()}
        config["overwrite"] = self.chk_overwrite.isChecked()
        config["vr_mode"] = self.chk_vr.isChecked()
        config["pov_mode"] = self.chk_pov.isChecked()
        config["backend"] = self.backend_combo.currentText()
        config["keyframe_reduction"] = self.chk_keyframe.isChecked()
        
        config_path = "config.json"
        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            QMessageBox.information(self, "Config Saved", STRINGS["config_saved"].format(config_path=config_path))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save config: {e}")
    
    def load_config(self):
        """Load configuration from file."""
        config_path = "config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                # Load after UI is set up
                QTimer.singleShot(100, lambda: self._load_config_values(config))
                
            except Exception as e:
                QMessageBox.warning(self, "Config Load", STRINGS["config_load_error"].format(error=str(e)))
    
    def _load_config_values(self, config):
        """Helper to load config values after UI setup."""
        for key, widget in self.params.items():
            if key in config:
                widget.setText(str(config[key]))
        
        if "overwrite" in config:
            self.chk_overwrite.setChecked(config["overwrite"])
        if "vr_mode" in config:
            self.chk_vr.setChecked(config["vr_mode"])
        if "pov_mode" in config:
            self.chk_pov.setChecked(config["pov_mode"])
        if "keyframe_reduction" in config:
            self.chk_keyframe.setChecked(config["keyframe_reduction"])
        if "backend" in config and config["backend"] in self.available_backends:
            self.backend_combo.setCurrentText(config["backend"])
    
    def run_batch(self):
        """Start batch processing."""
        if not self.files:
            QMessageBox.warning(self, "No files", STRINGS["no_files_warning"])
            return
        
        try:
            settings = {
                "threads": int(self.params["threads"].text()),
                "detrend_window": float(self.params["detrend_window"].text()),
                "norm_window": float(self.params["norm_window"].text()),
                "batch_size": int(self.params["batch_size"].text()),
                "overwrite": self.chk_overwrite.isChecked(),
                "keyframe_reduction": self.chk_keyframe.isChecked(),
                "vr_mode": self.chk_vr.isChecked(),
                "pov_mode": self.chk_pov.isChecked(),
                "backend": self.backend_combo.currentText()
            }
        except Exception as e:
            QMessageBox.critical(self, "Parameter Error", f"Invalid parameters: {e}")
            return
        
        # Reset progress
        self.overall_progress.setValue(0)
        self.video_progress.setValue(0)
        
        # Disable controls
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        
        # Start worker thread
        self.worker_thread = WorkerThread(self.files, settings)
        self.worker_thread.progressChanged.connect(self.overall_progress.setValue)
        self.worker_thread.videoProgressChanged.connect(self.video_progress.setValue)
        self.worker_thread.finished.connect(self.on_batch_finished)
        self.worker_thread.start()
    
    def cancel_run(self):
        """Cancel the running batch process."""
        if self.worker_thread:
            self.worker_thread.cancel()
    
    def on_batch_finished(self, error_occurred, time_str, log_messages):
        """Handle batch processing completion."""
        # Re-enable controls
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        
        # Show completion message
        if error_occurred:
            reply = QMessageBox.question(
                self, "Run Finished", 
                f"{STRINGS['processing_completed_with_errors']}\nCompleted in {time_str}\n\nWould you like to view the log?",
                QMessageBox.Yes | QMessageBox.No
            )
        else:
            reply = QMessageBox.question(
                self, "Run Finished", 
                f"Batch processing complete.\nCompleted in {time_str}\n\nWould you like to view the log?",
                QMessageBox.Yes | QMessageBox.No
            )
        
        if reply == QMessageBox.Yes:
            self.show_log_dialog(log_messages)
    
    def show_log_dialog(self, log_messages):
        """Show log messages in a dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Processing Log")
        dialog.setModal(True)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        text_edit = QTextEdit()
        text_edit.setPlainText("\n".join(log_messages))
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.exec()

# ---------- Headless Mode ----------
def run_headless(input_path, settings):
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

# ---------- Main ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optical Flow to Funscript")
    parser.add_argument("input", nargs="?", help="Input video file or folder")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads (default: 8)")
    parser.add_argument("--detrend_window", type=float, default=2.0, help="Detrend window in seconds (default: 2.0)")
    parser.add_argument("--norm_window", type=float, default=3.0, help="Normalization window in seconds (default: 3.0)")
    parser.add_argument("--batch_size", type=int, default=3000, help="Batch size in frames (default: 3000)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--vr_mode", action="store_true", help="Enable VR Mode (if not set, non‑VR mode is used)")
    parser.add_argument("--pov_mode", action="store_true", help="Enable POV Mode (improves stability for POV videos)")
    parser.add_argument("--disable_keyframe_reduction", action="store_false", help="Disable keyframe reduction")
    parser.add_argument("--backend", choices=["CPU", "CUDA", "OpenCL", "DNN"], default="CPU", help="Processing backend (default: CPU)")
    args = parser.parse_args()
    settings = {
        "threads": args.threads,
        "detrend_window": args.detrend_window,
        "norm_window": args.norm_window,
        "batch_size": args.batch_size,
        "overwrite": args.overwrite,
        "vr_mode": args.vr_mode,
        "pov_mode": args.pov_mode,
        "keyframe_reduction": not args.disable_keyframe_reduction,
        "backend": args.backend
    }
    if args.input:
        run_headless(args.input, settings)
    else:
        app = QApplication(sys.argv)
        
        # Set application icon
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "icon.png")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
        
        window = App()
        window.show()
        sys.exit(app.exec())
