"""Motion indicator window widget"""

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QPen


class MotionIndicatorWindow(QWidget):
    """Motion indicator window that shows current position as a vertical bar."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Motion Indicator")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)
        self.resize(50, 300)
        
        # Make window resizable
        self.setMinimumSize(20, 50)
        
        # Current position value (0-100)
        self.current_position = 0
        
        # Set up the widget
        self.setStyleSheet("background-color: black;")
        
    def set_position(self, position):
        """Set the position of the motion indicator (0-100)"""
        self.current_position = max(0, min(100, position))
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
        # Draw the vertical bar
        width = self.width()
        height = self.height()
        
        # Calculate bar height based on position (0-100)
        bar_height = int((self.current_position / 100.0) * height)
        
        # Draw from bottom up
        bar_rect = self.rect()
        bar_rect.setTop(height - bar_height)
        
        # Use red color for the bar
        painter.fillRect(bar_rect, QColor(255, 0, 0))
        
        # Draw border
        painter.setPen(QPen(QColor(128, 128, 128), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))