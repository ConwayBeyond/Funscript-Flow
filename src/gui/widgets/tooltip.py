"""Tooltip implementation for Qt widgets"""


class ToolTip:
    """Simple tooltip for Qt widgets using setToolTip."""
    def __init__(self, widget, text="widget info"):
        if widget and text:
            widget.setToolTip(str(text))