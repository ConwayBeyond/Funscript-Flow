"""Main entry point for Funscript Flow application"""

import sys
import os
import argparse
import signal
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon

from src.cli.headless import run_headless


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Optical Flow to Funscript")
    parser.add_argument("input", nargs="?", help="Input video file or folder")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads (default: 8)")
    parser.add_argument("--detrend_window", type=float, default=120, help="Detrend window (default: 120)")
    parser.add_argument("--norm_window", type=float, default=120, help="Norm window (default: 120)")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size (default: 50)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--vr_mode", action="store_true", help="Enable VR Mode")
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
        # Import GUI here to avoid circular imports
        from src.gui.main_window import App
        
        # Set application properties before creating QApplication
        QApplication.setApplicationName("Funscript Flow")
        QApplication.setApplicationDisplayName("Funscript Flow")
        QApplication.setOrganizationName("Funscript Flow")
        
        app = QApplication(sys.argv)
        
        # Set application icon
        from pathlib import Path
        icon_path = Path(__file__).parent / "assets" / "icon.png"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
        
        # Install signal handler for graceful shutdown
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        
        # Create timer to allow Python to process signals
        timer = QTimer()
        timer.timeout.connect(lambda: None)
        timer.start(100)  # Process signals every 100ms
        
        window = App()
        window.show()
        sys.exit(app.exec())


if __name__ == '__main__':
    main()