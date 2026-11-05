"""
Configuration Module for AI Hand Gesture Gaming Controller
Subway Surfers Motion Controller
"""
import os
import json
from dataclasses import dataclass, asdict

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PACKAGE_DIR, "models")
CALIBRATION_FILE = os.path.join(BASE_DIR, "calibration.json")
CONFIG_FILE = os.path.join(BASE_DIR, "settings.json")

# Default official MediaPipe model URL
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_FILENAME = "hand_landmarker.task"

# Subway Surfers Game URL
GAME_URL = "https://poki.com/en/g/subway-surfers"

# UI Theme Colors (Cyberpunk / Dark Gaming Aesthetic)
THEME = {
    "bg_main": "#080c14",        # Deepest dark navy
    "bg_secondary": "#0d1424",   # Slate midnight
    "card_bg": "#121a2d",        # Glassmorphic card surface
    "card_border": "#1e2c47",    # Card outline
    "card_active": "#182a4d",    # Glow background
    "accent_cyan": "#00f0ff",    # Electric cyan
    "accent_blue": "#38bdf8",    # Electric light blue
    "accent_neon": "#4facfe",    # Neon gradient blue
    "accent_green": "#10b981",   # Active state
    "accent_orange": "#f59e0b",  # Calibration / warning
    "accent_red": "#ef4444",     # Emergency stop / error
    "text_primary": "#ffffff",   # Crisp white
    "text_secondary": "#94a3b8", # Muted slate
    "text_muted": "#475569",     # Dim slate
    "glow_shadow": "#00f0ff33",  # Glow shadow
    "button_primary": "#0284c7", # Action button
    "button_hover": "#0369a1",   # Action hover
}

@dataclass
class ControllerSettings:
    # Performance Mode ("NORMAL", "FAST", "ULTRA FAST")
    performance_mode: str = "ULTRA FAST"

    # Camera settings
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    mirror_camera: bool = True
    fps_limit: int = 60

    # Detection confidence
    min_detection_confidence: float = 0.65
    min_tracking_confidence: float = 0.65

    # Gesture Cooldowns (in seconds)
    cooldown_left_right: float = 0.14   # 140 ms (Ultra-fast lane switching)
    cooldown_jump: float = 0.25         # 250 ms (Ultra-fast jump reaction)
    cooldown_roll: float = 0.25         # 250 ms (Ultra-fast roll reaction)

    # Debouncing / smoothing
    smoothing_window: int = 2           # Frame buffer for consensus
    min_trigger_frames: int = 1         # Frames gesture must be held to register

    # Input Mapping (ARROWS or WASD)
    key_preset: str = "ARROWS"          # "ARROWS" or "WASD"
    key_press_duration: float = 0.025   # 25 ms snappy hardware pulse

    # Key mappings
    keys_arrows: dict = None
    keys_wasd: dict = None

    # Audio feedback
    sound_feedback: bool = True

    def __post_init__(self):
        if self.keys_arrows is None:
            self.keys_arrows = {
                "JUMP": "up",
                "ROLL": "down",
                "MOVE_LEFT": "left",
                "MOVE_RIGHT": "right"
            }
        if self.keys_wasd is None:
            self.keys_wasd = {
                "JUMP": "w",
                "ROLL": "s",
                "MOVE_LEFT": "a",
                "MOVE_RIGHT": "d"
            }

    def apply_performance_mode(self, mode: str):
        """Applies calibrated profiles for Normal, Fast, and Ultra Fast gameplay."""
        mode_upper = mode.upper()
        self.performance_mode = mode_upper
        if mode_upper == "ULTRA FAST":
            self.cooldown_left_right = 0.14
            self.cooldown_jump = 0.25
            self.cooldown_roll = 0.25
            self.smoothing_window = 2
            self.min_trigger_frames = 1
            self.key_press_duration = 0.025
        elif mode_upper == "FAST":
            self.cooldown_left_right = 0.20
            self.cooldown_jump = 0.32
            self.cooldown_roll = 0.32
            self.smoothing_window = 3
            self.min_trigger_frames = 2
            self.key_press_duration = 0.030
        else: # NORMAL
            self.cooldown_left_right = 0.28
            self.cooldown_jump = 0.40
            self.cooldown_roll = 0.40
            self.smoothing_window = 4
            self.min_trigger_frames = 2
            self.key_press_duration = 0.040

    @property
    def current_keys(self) -> dict:
        return self.keys_wasd if self.key_preset.upper() == "WASD" else self.keys_arrows

    def save_to_file(self, filepath: str = CONFIG_FILE):
        try:
            data = asdict(self)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Failed to save settings: {e}")

    @classmethod
    def load_from_file(cls, filepath: str = CONFIG_FILE) -> "ControllerSettings":
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                return cls(**data)
            except Exception as e:
                print(f"Failed to load settings, using defaults: {e}")
        return cls()

# Global default settings instance
settings = ControllerSettings.load_from_file()

