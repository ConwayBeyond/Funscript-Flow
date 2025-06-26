"""FunScript visualizer widget for the GUI"""

from PySide6.QtWidgets import QWidget, QStyleOptionSlider
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QPen, QBrush, QColor
from PySide6.QtWidgets import QStyle


class FunScriptVisualizer(QWidget):
    """Custom widget for visualizing FunScript data with pan and zoom functionality."""
    
    positionChanged = Signal(int)  # Emitted when user clicks to seek
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setMaximumHeight(150)
        
        # Data
        self.actions = []
        self.duration_ms = 0
        self.current_position_ms = 0
        
        # View state
        self.zoom_level = 1.0
        self.pan_offset = 0.0
        self.min_zoom = 1.0
        self.max_zoom = 50.0
        
        # Interaction state
        self.mouse_pressed = False
        self.last_mouse_x = 0
        self.dragging = False
        
        # Slider synchronization
        self.reference_slider = None
        self.slider_margins = 0
        self.groove_width = 0
        
        # Visual settings
        self.background_color = QColor(40, 40, 40)
        self.grid_color = QColor(60, 60, 60)
        self.line_color = QColor(100, 150, 255)
        self.point_color = QColor(255, 255, 255)
        self.current_pos_color = QColor(255, 100, 100)
        
        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
        
    def load_funscript(self, funscript_data):
        """Load FunScript data."""
        self.actions = funscript_data.get("actions", [])
        if self.actions:
            self.duration_ms = max(action["at"] for action in self.actions)
        else:
            self.duration_ms = 0
        self.reset_view()
        self.update()
        
    def set_duration(self, duration_ms):
        """Set the video duration."""
        self.duration_ms = duration_ms
        self.reset_view()
        self.update()
        
    def set_position(self, position_ms):
        """Set the current playback position."""
        self.current_position_ms = position_ms
        self.update()
        
    def set_reference_slider(self, slider):
        """Set the reference slider for width synchronization."""
        self.reference_slider = slider
        self.update_slider_geometry()
        
    def update_slider_geometry(self):
        """Update geometry to match the slider's groove area."""
        if not self.reference_slider:
            self.slider_margins = 10
            return
            
        # Get the slider's groove rectangle
        slider = self.reference_slider
        style = slider.style()
        
        # Create style option for the slider
        option = QStyleOptionSlider()
        option.initFrom(slider)
        option.minimum = slider.minimum()
        option.maximum = slider.maximum()
        option.sliderPosition = slider.value()
        option.orientation = slider.orientation()
        
        # Get the groove rectangle
        groove_rect = style.subControlRect(QStyle.CC_Slider, option, QStyle.SC_SliderGroove, slider)
        
        # Calculate margins from widget edge to groove
        self.slider_margins = groove_rect.left()
        self.groove_width = groove_rect.width()
        
    def reset_view(self):
        """Reset pan and zoom to show full timeline."""
        self.zoom_level = 1.0
        self.pan_offset = 0.0
        self.update()
        
    def time_to_x(self, time_ms):
        """Convert time to x coordinate, matching slider groove area."""
        if self.duration_ms == 0:
            return 0
        # Apply zoom and pan
        normalized_time = time_ms / self.duration_ms
        zoomed_time = (normalized_time - self.pan_offset) * self.zoom_level
        
        # Use actual groove dimensions
        return self.slider_margins + (zoomed_time * self.groove_width)
        
    def x_to_time(self, x):
        """Convert x coordinate to time, matching slider groove area."""
        if self.duration_ms == 0:
            return 0
            
        # Use actual groove dimensions
        normalized_x = (x - self.slider_margins) / self.groove_width
        time_normalized = (normalized_x / self.zoom_level) + self.pan_offset
        return int(time_normalized * self.duration_ms)
        
    def position_to_y(self, position):
        """Convert position value (0-100) to y coordinate."""
        # Invert Y axis so 0 is at bottom, 100 at top
        return self.height() - (position / 100.0) * self.height()
        
    def paintEvent(self, event):
        """Paint the visualizer."""
        # Update slider geometry on each paint to ensure sync
        self.update_slider_geometry()
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), self.background_color)
        
        if not self.actions or self.duration_ms == 0:
            painter.setPen(QPen(QColor(100, 100, 100)))
            painter.drawText(self.rect(), Qt.AlignCenter, "No FunScript data loaded")
            return
            
        # Draw grid lines
        self.draw_grid(painter)
        
        # Draw FunScript line
        self.draw_funscript_line(painter)
        
        # Draw current position indicator
        self.draw_current_position(painter)
        
    def draw_grid(self, painter):
        """Draw background grid."""
        painter.setPen(QPen(self.grid_color, 1))
        
        # Horizontal lines (position levels)
        for i in range(0, 101, 25):
            y = self.position_to_y(i)
            painter.drawLine(0, y, self.width(), y)
            
        # Vertical lines (time markers)
        visible_duration = self.duration_ms / self.zoom_level
        time_step = max(1000, visible_duration / 10)  # At least 1 second steps
        
        start_time = self.pan_offset * self.duration_ms
        end_time = start_time + visible_duration
        
        current_time = (int(start_time / time_step) * time_step)
        while current_time <= end_time:
            x = self.time_to_x(current_time)
            if 0 <= x <= self.width():
                painter.drawLine(x, 0, x, self.height())
            current_time += time_step
            
    def draw_funscript_line(self, painter):
        """Draw the FunScript data as a connected line."""
        if len(self.actions) < 2:
            return
            
        # Filter actions that are visible in current view
        visible_actions = []
        for action in self.actions:
            x = self.time_to_x(action["at"])
            if -10 <= x <= self.width() + 10:  # Small margin for smooth edges
                visible_actions.append(action)
                
        if len(visible_actions) < 2:
            return
            
        # Draw lines between points
        painter.setPen(QPen(self.line_color, 2))
        for i in range(len(visible_actions) - 1):
            action1 = visible_actions[i]
            action2 = visible_actions[i + 1]
            
            x1 = self.time_to_x(action1["at"])
            y1 = self.position_to_y(action1["pos"])
            x2 = self.time_to_x(action2["at"])
            y2 = self.position_to_y(action2["pos"])
            
            painter.drawLine(x1, y1, x2, y2)
            
        # Draw points
        painter.setPen(QPen(self.point_color, 1))
        painter.setBrush(QBrush(self.point_color))
        for action in visible_actions:
            x = self.time_to_x(action["at"])
            y = self.position_to_y(action["pos"])
            painter.drawEllipse(x - 2, y - 2, 4, 4)
            
    def draw_current_position(self, painter):
        """Draw the current playback position indicator."""
        x = self.time_to_x(self.current_position_ms)
        if 0 <= x <= self.width():
            painter.setPen(QPen(self.current_pos_color, 3))
            painter.drawLine(x, 0, x, self.height())
            
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = True
            self.last_mouse_x = event.position().x()
            self.dragging = False
            
    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        if self.mouse_pressed:
            dx = event.position().x() - self.last_mouse_x
            if abs(dx) > 3:  # Start dragging only after significant movement
                self.dragging = True
                # Pan the view
                pan_delta = -dx / self.width() / self.zoom_level
                self.pan_offset = max(0, min(1 - 1/self.zoom_level, self.pan_offset + pan_delta))
                self.update()
            self.last_mouse_x = event.position().x()
            
    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.LeftButton:
            if self.mouse_pressed and not self.dragging:
                # Single click - seek to position
                time_ms = self.x_to_time(event.position().x())
                time_ms = max(0, min(self.duration_ms, time_ms))
                self.positionChanged.emit(time_ms)
            self.mouse_pressed = False
            self.dragging = False
            
    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        # Get mouse position as a fraction of widget width
        mouse_fraction = event.position().x() / self.width()
        
        # Calculate the time at mouse position before zoom
        time_at_mouse = self.x_to_time(event.position().x())
        
        # Apply zoom
        zoom_factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        new_zoom = self.zoom_level * zoom_factor
        new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
        
        if new_zoom != self.zoom_level:
            self.zoom_level = new_zoom
            
            # Adjust pan to keep the same time under the mouse
            new_mouse_fraction = self.time_to_x(time_at_mouse) / self.width()
            pan_adjustment = (mouse_fraction - new_mouse_fraction) / self.zoom_level
            self.pan_offset = max(0, min(1 - 1/self.zoom_level, self.pan_offset + pan_adjustment))
            
            self.update()