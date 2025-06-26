"""Configuration and settings management"""

import json
import os
from typing import Dict, Any


# ---------- Constants ----------
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".wmv", ".flv", ".mpg", ".mpeg", ".ts"}
SUPPORTED_VIDEO_PATTERNS = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_VIDEO_EXTENSIONS))


def save_config(config: Dict[str, Any], config_path: str = "config.json") -> None:
    """Save configuration to file."""
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        raise Exception(f"Could not save config: {e}")


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from file."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Error loading config: {e}")
    return {}


def get_default_settings() -> Dict[str, Any]:
    """Get default application settings."""
    return {
        "threads": 1,
        "detrend_window": 120,
        "norm_window": 120,
        "batch_size": 50,
        "overwrite": False,
        "vr_mode": False,
        "pov_mode": False,
        "keyframe_reduction": True,
        "backend": "CPU",
        "cut_threshold": 7
    }