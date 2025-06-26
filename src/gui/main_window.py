"""Main application window"""

import os
import json
import subprocess
import platform
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QCheckBox, QProgressBar, QLineEdit, 
                               QFileDialog, QMessageBox, QTextEdit, QDialog, QComboBox,
                               QFrame, QScrollArea, QTabWidget, QSlider, QGroupBox, QFormLayout,
                               QSizePolicy, QStyle)
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QUrl, QMimeData
from PySide6.QtGui import QPixmap, QIcon, QFont, QPainter, QPen, QBrush, QColor, QDragEnterEvent, QDropEvent
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

from ..config.localization import STRINGS
from ..config.settings import SUPPORTED_VIDEO_EXTENSIONS, save_config, load_config, get_default_settings
from ..backends import get_available_backends, get_gpu_info
from ..video.utils import convert_frame_to_photo
from .widgets.tooltip import ToolTip
from .widgets.motion_indicator import MotionIndicatorWindow
from .visualizer import FunScriptVisualizer
from .workers import WorkerThread


class App(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(STRINGS["app_title"])
        self.setGeometry(100, 100, 800, 600)
        self.setAcceptDrops(True)
        
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
        
        # Initialize motion indicator window
        self.motion_indicator = MotionIndicatorWindow(self)
        self.motion_indicator_visible = False
        
        # Video player state
        self.loaded_video_path = None
        self.loaded_funscript_data = None
        self.slider_being_dragged = False
        self.muted = False
        self.previous_volume = 50
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Funscript Generation Tab
        self.generation_tab = QWidget()
        self.generation_tab.setAcceptDrops(True)
        self.tab_widget.addTab(self.generation_tab, "Funscript Generation")
        self.setup_generation_tab()
        
        # Script Preview Tab
        self.preview_tab = QWidget()
        self.preview_tab.setAcceptDrops(True)
        self.tab_widget.addTab(self.preview_tab, "Script Preview")
        self.setup_preview_tab()
        
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
        
        # Add README button (placeholder for now)
        self.btn_readme = QPushButton("README")
        self.btn_readme.clicked.connect(self.show_readme)
        file_layout.addWidget(self.btn_readme)
        
        layout.addWidget(file_group)
        
        # Mode selection section
        mode_group = QGroupBox("Mode Selection")
        mode_layout = QHBoxLayout(mode_group)
        
        self.chk_vr = QCheckBox(STRINGS["vr_mode"])
        ToolTip(self.chk_vr, "Enable VR Mode processing")
        mode_layout.addWidget(self.chk_vr)
        
        self.chk_pov = QCheckBox("POV Mode")
        ToolTip(self.chk_pov, "Enable POV Mode for improved stability in POV videos")
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
        
        # Live Log section (collapsible)
        self.log_group = QGroupBox("Live Log")
        self.log_group.setCheckable(True)
        self.log_group.setChecked(True)  # Expanded by default
        log_layout = QVBoxLayout(self.log_group)
        
        # Log display text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setMinimumHeight(150)
        log_layout.addWidget(self.log_text)
        
        # Clear log button
        log_button_layout = QHBoxLayout()
        self.btn_clear_log = QPushButton("Clear Log")
        self.btn_clear_log.clicked.connect(self.clear_log_display)
        log_button_layout.addWidget(self.btn_clear_log)
        log_button_layout.addStretch()
        log_layout.addLayout(log_button_layout)
        
        layout.addWidget(self.log_group)
        
        # Advanced settings section (collapsible)
        self.adv_group = QGroupBox("Advanced Settings")
        self.adv_group.setCheckable(True)
        self.adv_group.setChecked(False)  # Collapsed by default
        self.setup_advanced_settings()
        layout.addWidget(self.adv_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.btn_run = QPushButton(STRINGS["run"])
        self.btn_run.clicked.connect(self.run_processing)
        button_layout.addWidget(self.btn_run)
        
        self.btn_cancel = QPushButton(STRINGS["cancel"])
        self.btn_cancel.clicked.connect(self.cancel_processing)
        self.btn_cancel.setEnabled(False)
        button_layout.addWidget(self.btn_cancel)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        layout.addStretch()
        
    def setup_advanced_settings(self):
        """Setup advanced settings section."""
        advanced_layout = QFormLayout(self.adv_group)
        
        # Get available backends
        backend_choices = []
        for backend, available in self.backends.items():
            if available:
                backend_choices.append(backend)
        
        # Backend selection
        self.combo_backend = QComboBox()
        self.combo_backend.addItems(backend_choices)
        self.combo_backend.setCurrentText("CPU")
        advanced_layout.addRow(QLabel("Backend:"), self.combo_backend)
        
        # Number of threads
        self.line_threads = QLineEdit("8")
        advanced_layout.addRow(QLabel("Threads:"), self.line_threads)
        
        # Detrend window
        self.line_detrend = QLineEdit("120")
        advanced_layout.addRow(QLabel("Detrend Window:"), self.line_detrend)
        
        # Norm window
        self.line_norm = QLineEdit("120")
        advanced_layout.addRow(QLabel("Norm Window:"), self.line_norm)
        
        # Batch size
        self.line_batch = QLineEdit("50")
        advanced_layout.addRow(QLabel("Batch Size:"), self.line_batch)
        
        # Additional options
        self.chk_keyframe = QCheckBox("Keyframe Reduction")
        self.chk_keyframe.setChecked(True)
        advanced_layout.addRow(self.chk_keyframe)
        
        self.chk_overwrite = QCheckBox(STRINGS["overwrite_files"])
        advanced_layout.addRow(self.chk_overwrite)
        
    def setup_preview_tab(self):
        """Setup the script preview tab with full video player functionality."""
        layout = QVBoxLayout(self.preview_tab)
        
        # Video player
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(300)
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_widget.setAcceptDrops(False)
        layout.addWidget(self.video_widget, 1)
        
        # Media player setup
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        
        # Initialize volume state
        self.audio_output.setVolume(0.5)
        
        # Video controls
        controls_layout = QHBoxLayout()
        
        self.btn_play_pause = QPushButton("Play")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.btn_play_pause.setFixedWidth(60)
        controls_layout.addWidget(self.btn_play_pause)
        
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setMinimum(0)
        self.position_slider.setMaximum(1000)
        self.position_slider.sliderMoved.connect(self.set_position)
        self.position_slider.valueChanged.connect(self.on_slider_value_changed)
        self.position_slider.sliderPressed.connect(self.on_slider_pressed)
        self.position_slider.sliderReleased.connect(self.on_slider_released)
        self.position_slider.setAcceptDrops(False)
        controls_layout.addWidget(self.position_slider)
        
        self.lbl_time = QLabel("00:00 / 00:00")
        controls_layout.addWidget(self.lbl_time)
        
        # Volume controls
        volume_label = QLabel("Volume:")
        controls_layout.addWidget(volume_label)
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(50)
        self.volume_slider.setFixedWidth(80)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        controls_layout.addWidget(self.volume_slider)
        
        self.btn_mute = QPushButton("🔊")
        self.btn_mute.setFixedWidth(30)
        self.btn_mute.clicked.connect(self.toggle_mute)
        controls_layout.addWidget(self.btn_mute)
        
        layout.addLayout(controls_layout)
        
        # FunScript visualizer
        visualizer_group = QGroupBox("FunScript Visualizer")
        visualizer_layout = QVBoxLayout(visualizer_group)
        visualizer_layout.setContentsMargins(6, 6, 6, 6)
        visualizer_layout.setSpacing(3)
        
        self.funscript_visualizer = FunScriptVisualizer()
        self.funscript_visualizer.positionChanged.connect(self.seek_to_position)
        self.funscript_visualizer.set_reference_slider(self.position_slider)
        self.funscript_visualizer.setAcceptDrops(False)
        visualizer_layout.addWidget(self.funscript_visualizer)
        
        # Visualizer controls
        viz_controls_layout = QHBoxLayout()
        viz_controls_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_reset_view = QPushButton("Reset View")
        self.btn_reset_view.clicked.connect(self.funscript_visualizer.reset_view)
        viz_controls_layout.addWidget(self.btn_reset_view)
        
        self.btn_toggle_indicator = QPushButton("Show Indicator")
        self.btn_toggle_indicator.clicked.connect(self.toggle_motion_indicator)
        viz_controls_layout.addWidget(self.btn_toggle_indicator)
        
        self.lbl_zoom = QLabel("Zoom: 1.0x")
        viz_controls_layout.addWidget(self.lbl_zoom)
        
        viz_controls_layout.addStretch()
        visualizer_layout.addLayout(viz_controls_layout)
        
        layout.addWidget(visualizer_group)
        
        # File loading buttons
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        
        self.btn_load_video = QPushButton("Load Video")
        self.btn_load_video.clicked.connect(self.load_video)
        bottom_layout.addWidget(self.btn_load_video)
        
        self.btn_load_funscript = QPushButton("Load FunScript")
        self.btn_load_funscript.clicked.connect(self.load_funscript)
        bottom_layout.addWidget(self.btn_load_funscript)
        
        layout.addLayout(bottom_layout)
        
        # Initialize media player connections
        self.media_player.positionChanged.connect(self.on_position_changed)
        self.media_player.durationChanged.connect(self.on_duration_changed)
        self.media_player.playbackStateChanged.connect(self.on_playback_state_changed)
        
        # Timer for updating visualizer position
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self.update_visualizer_position)
        self.position_timer.start(50)
        
    def load_config(self):
        """Load configuration from file."""
        try:
            config = load_config()
            # Apply basic settings
            if "vr_mode" in config:
                self.chk_vr.setChecked(config["vr_mode"])
            if "overwrite" in config:
                self.chk_overwrite.setChecked(config["overwrite"])
            
            # Apply advanced settings
            if "pov_mode" in config:
                self.chk_pov.setChecked(config["pov_mode"])
            if "keyframe_reduction" in config:
                self.chk_keyframe.setChecked(config["keyframe_reduction"])
            if "backend" in config and config["backend"] in [self.combo_backend.itemText(i) for i in range(self.combo_backend.count())]:
                self.combo_backend.setCurrentText(config["backend"])
            if "threads" in config:
                self.line_threads.setText(str(config["threads"]))
            if "detrend_window" in config:
                self.line_detrend.setText(str(config["detrend_window"]))
            if "norm_window" in config:
                self.line_norm.setText(str(config["norm_window"]))
            if "batch_size" in config:
                self.line_batch.setText(str(config["batch_size"]))
        except Exception as e:
            print(f"Could not load config: {e}")
            
    def save_current_config(self):
        """Save current settings to config file."""
        try:
            config = {
                "vr_mode": self.chk_vr.isChecked(),
                "overwrite": self.chk_overwrite.isChecked(),
                "pov_mode": self.chk_pov.isChecked(),
                "keyframe_reduction": self.chk_keyframe.isChecked(),
                "backend": self.combo_backend.currentText(),
                "threads": int(self.line_threads.text() or "8"),
                "detrend_window": float(self.line_detrend.text() or "120"),
                "norm_window": float(self.line_norm.text() or "120"),
                "batch_size": int(self.line_batch.text() or "50")
            }
            save_config(config)
        except Exception as e:
            print(f"Could not save config: {e}")
            
    def select_files(self):
        """Select video files for processing."""
        files, _ = QFileDialog.getOpenFileNames(
            self, 
            STRINGS["select_videos"],
            "",
            f"Video files ({' '.join(f'*{ext}' for ext in SUPPORTED_VIDEO_EXTENSIONS)})"
        )
        if files:
            self.files = files
            self.update_file_label()
            
    def select_folder(self):
        """Select folder containing video files."""
        folder = QFileDialog.getExistingDirectory(self, STRINGS["select_folder"])
        if folder:
            files = []
            for root, dirs, file_list in os.walk(folder):
                for file in file_list:
                    if any(file.lower().endswith(ext) for ext in SUPPORTED_VIDEO_EXTENSIONS):
                        files.append(os.path.join(root, file))
            self.files = files
            self.update_file_label()
            
    def update_file_label(self):
        """Update the file selection label."""
        if self.files:
            self.lbl_files.setText(f"{len(self.files)} file(s) selected")
        else:
            self.lbl_files.setText(STRINGS["no_files_selected"])
            
    def run_processing(self):
        """Start video processing."""
        if not self.files:
            QMessageBox.warning(self, "Warning", STRINGS["no_files_warning"])
            return
            
        # Get settings
        settings = get_default_settings()
        settings.update({
            "vr_mode": self.chk_vr.isChecked(),
            "overwrite": self.chk_overwrite.isChecked(),
            "pov_mode": self.chk_pov.isChecked(),
            "keyframe_reduction": self.chk_keyframe.isChecked(),
            "backend": self.combo_backend.currentText(),
            "threads": int(self.line_threads.text() or "8"),
            "detrend_window": float(self.line_detrend.text() or "120"),
            "norm_window": float(self.line_norm.text() or "120"),
            "batch_size": int(self.line_batch.text() or "50")
        })
        
        # Start worker thread
        self.worker_thread = WorkerThread(self.files, settings)
        self.worker_thread.progressChanged.connect(self.overall_progress.setValue)
        self.worker_thread.videoProgressChanged.connect(self.video_progress.setValue)
        self.worker_thread.logMessage.connect(self.log_text.append)
        self.worker_thread.finished.connect(self.on_processing_finished)
        
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        
        self.worker_thread.start()
        
    def cancel_processing(self):
        """Cancel processing."""
        if self.worker_thread:
            self.worker_thread.cancel()
            
    def on_processing_finished(self, error_occurred, time_str, log_messages, generated_files):
        """Handle processing completion."""
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        
        # Save current configuration
        self.save_current_config()
        
        if error_occurred:
            QMessageBox.warning(self, "Processing Complete", "Processing completed with errors. See log for details.")
        else:
            if generated_files:
                # Create message with option to open in preview
                msg = QMessageBox()
                msg.setWindowTitle("Processing Complete")
                msg.setText(f"Processing completed successfully in {time_str}")
                msg.setInformativeText(f"Generated {len(generated_files)} funscript(s). Would you like to open the first one in the preview?")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                msg.setDefaultButton(QMessageBox.Yes)
                
                if msg.exec() == QMessageBox.Yes and generated_files:
                    video_path, funscript_path = generated_files[0]
                    self.load_video_and_funscript_in_preview(video_path, funscript_path)
            else:
                QMessageBox.information(self, "Processing Complete", f"Processing completed successfully in {time_str}")
            
        self.worker_thread = None
        
    # Media Player Methods
    def toggle_play_pause(self):
        """Toggle between play and pause states."""
        if self.media_player.playbackState() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()
            
    def on_slider_pressed(self):
        """Handle when user starts interacting with the slider."""
        self.slider_being_dragged = True
        self.set_position(self.position_slider.value())
        
    def on_slider_released(self):
        """Handle when user releases the slider."""
        self.slider_being_dragged = False
        
    def on_slider_value_changed(self, position):
        """Handle slider value changes from user interaction."""
        if self.slider_being_dragged:
            self.set_position(position)
            
    def set_position(self, position):
        """Set the video position from slider."""
        duration = self.media_player.duration()
        if duration > 0:
            new_position = (position * duration) // 1000
            self.media_player.setPosition(new_position)
            
    def seek_to_position(self, position_ms):
        """Seek to a specific position from the visualizer."""
        self.media_player.setPosition(position_ms)
        
    def on_position_changed(self, position):
        """Handle video position changes."""
        duration = self.media_player.duration()
        if duration > 0:
            if not self.slider_being_dragged:
                slider_position = (position * 1000) // duration
                self.position_slider.blockSignals(True)
                self.position_slider.setValue(slider_position)
                self.position_slider.blockSignals(False)
            
            current_time = self.format_time(position)
            total_time = self.format_time(duration)
            self.lbl_time.setText(f"{current_time} / {total_time}")
            
    def on_duration_changed(self, duration):
        """Handle video duration changes."""
        if duration > 0 and hasattr(self, 'loaded_funscript_data') and self.loaded_funscript_data:
            self.funscript_visualizer.set_duration(duration)
            
    def on_playback_state_changed(self, state):
        """Handle playback state changes."""
        if state == QMediaPlayer.PlayingState:
            self.btn_play_pause.setText("Pause")
        else:
            self.btn_play_pause.setText("Play")
            
    def on_volume_changed(self, volume):
        """Handle volume slider changes."""
        volume_float = volume / 100.0
        self.audio_output.setVolume(volume_float)
        
        if volume == 0:
            self.btn_mute.setText("🔇")
        elif volume < 50:
            self.btn_mute.setText("🔉")
        else:
            self.btn_mute.setText("🔊")
            
    def toggle_mute(self):
        """Toggle mute/unmute functionality."""
        if self.muted:
            self.volume_slider.setValue(self.previous_volume)
            self.muted = False
        else:
            self.previous_volume = self.volume_slider.value()
            self.volume_slider.setValue(0)
            self.muted = True
            
    def load_video(self):
        """Load a video file for preview."""
        file_path, _ = QFileDialog.getOpenFileNames(
            self, "Load Video", "",
            f"Video files ({' '.join(f'*{ext}' for ext in SUPPORTED_VIDEO_EXTENSIONS)})"
        )
        if file_path:
            self.loaded_video_path = file_path[0]
            self.media_player.setSource(QUrl.fromLocalFile(file_path[0]))
            
    def load_funscript(self):
        """Load a FunScript file for visualization."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load FunScript", "",
            "FunScript Files (*.funscript);;JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.loaded_funscript_data = json.load(f)
                
                self.funscript_visualizer.load_funscript(self.loaded_funscript_data)
                
                if self.media_player.duration() > 0:
                    self.funscript_visualizer.set_duration(self.media_player.duration())
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load FunScript: {e}")
                
    def update_visualizer_position(self):
        """Update the visualizer with current video position."""
        if hasattr(self, 'media_player') and self.media_player.duration() > 0:
            current_position = self.media_player.position()
            self.funscript_visualizer.set_position(current_position)
            
            zoom_level = self.funscript_visualizer.zoom_level
            self.lbl_zoom.setText(f"Zoom: {zoom_level:.1f}x")
            
            if self.motion_indicator_visible:
                current_value = self.get_current_funscript_value()
                self.motion_indicator.set_position(current_value)
                
    def format_time(self, ms):
        """Format time in milliseconds to MM:SS format."""
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
        
    def get_current_funscript_value(self):
        """Get the current funscript position value at current playback time."""
        if not hasattr(self, 'loaded_funscript_data') or not self.loaded_funscript_data or not hasattr(self.funscript_visualizer, 'actions'):
            return 0
        
        current_time = self.media_player.position()
        actions = self.funscript_visualizer.actions
        
        if not actions:
            return 0
        
        for i in range(len(actions) - 1):
            if actions[i]["at"] <= current_time <= actions[i + 1]["at"]:
                t1, pos1 = actions[i]["at"], actions[i]["pos"]
                t2, pos2 = actions[i + 1]["at"], actions[i + 1]["pos"]
                
                if t2 == t1:
                    return pos1
                
                ratio = (current_time - t1) / (t2 - t1)
                return pos1 + (pos2 - pos1) * ratio
        
        if current_time < actions[0]["at"]:
            return actions[0]["pos"]
        else:
            return actions[-1]["pos"]
            
    def toggle_motion_indicator(self):
        """Toggle the visibility of the motion indicator window."""
        if self.motion_indicator_visible:
            self.motion_indicator.hide()
            self.motion_indicator_visible = False
            self.btn_toggle_indicator.setText("Show Indicator")
        else:
            self.motion_indicator.show()
            self.motion_indicator_visible = True
            self.btn_toggle_indicator.setText("Hide Indicator")
            
    def load_video_and_funscript_in_preview(self, video_path, funscript_path):
        """Load video and funscript in the preview tab and switch to it."""
        try:
            self.tab_widget.setCurrentIndex(1)
            
            self.loaded_video_path = video_path
            self.media_player.setSource(QUrl.fromLocalFile(video_path))
            
            if os.path.exists(funscript_path):
                with open(funscript_path, 'r') as f:
                    self.loaded_funscript_data = json.load(f)
                
                self.funscript_visualizer.load_funscript(self.loaded_funscript_data)
                
                if self.media_player.duration() > 0:
                    self.funscript_visualizer.set_duration(self.media_player.duration())
            
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load video and funscript in preview: {e}")
            return False
    
    def show_readme(self):
        """Show README dialog."""
        QMessageBox.information(self, "README", "README functionality to be implemented.")
        
    def clear_log_display(self):
        """Clear the log display."""
        if hasattr(self, 'log_text'):
            self.log_text.clear()
            
    # Drag & Drop Methods
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            
    def dropEvent(self, event: QDropEvent):
        """Handle drop events."""
        files = []
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isfile(file_path):
                files.append(file_path)
            elif os.path.isdir(file_path):
                # Add all video files from directory
                for root, dirs, file_list in os.walk(file_path):
                    for file in file_list:
                        if any(file.lower().endswith(ext) for ext in SUPPORTED_VIDEO_EXTENSIONS):
                            files.append(os.path.join(root, file))
        
        if files:
            # Determine which tab is active
            current_tab = self.tab_widget.currentIndex()
            
            if current_tab == 0:  # Generation tab
                # Add to file list for processing
                video_files = [f for f in files if any(f.lower().endswith(ext) for ext in SUPPORTED_VIDEO_EXTENSIONS)]
                if video_files:
                    self.files.extend(video_files)
                    self.files = list(set(self.files))  # Remove duplicates
                    self.update_file_label()
                    
            elif current_tab == 1:  # Preview tab
                # Load first video file for preview
                video_files = [f for f in files if any(f.lower().endswith(ext) for ext in SUPPORTED_VIDEO_EXTENSIONS)]
                funscript_files = [f for f in files if f.lower().endswith('.funscript')]
                
                if video_files:
                    video_path = video_files[0]
                    self.loaded_video_path = video_path
                    self.media_player.setSource(QUrl.fromLocalFile(video_path))
                    
                    # Look for matching funscript
                    base_name = os.path.splitext(video_path)[0]
                    funscript_path = base_name + ".funscript"
                    
                    if os.path.exists(funscript_path):
                        try:
                            with open(funscript_path, 'r') as f:
                                self.loaded_funscript_data = json.load(f)
                            self.funscript_visualizer.load_funscript(self.loaded_funscript_data)
                        except Exception as e:
                            print(f"Error loading funscript: {e}")
                    
                    # Load explicit funscript files
                    elif funscript_files:
                        try:
                            with open(funscript_files[0], 'r') as f:
                                self.loaded_funscript_data = json.load(f)
                            self.funscript_visualizer.load_funscript(self.loaded_funscript_data)
                        except Exception as e:
                            print(f"Error loading funscript: {e}")
        
        event.acceptProposedAction()