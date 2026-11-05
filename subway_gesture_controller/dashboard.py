"""
Dashboard Module - Modern Dark Cyberpunk GUI for Subway Surfers Motion Controller
Built using CustomTkinter with glassmorphism styling, animated gesture cards,
real-time HUD video display, calibration wizard, and global hotkeys.
"""
import os
import sys
import time
import webbrowser
import threading
from collections import deque
from typing import Optional, Dict
import numpy as np
import cv2
from PIL import Image, ImageTk
import customtkinter as ctk

# Windows global hotkey support
try:
    import ctypes
    HAS_CTYPES = True
    VK_ESCAPE = 0x1B
except ImportError:
    HAS_CTYPES = False

from subway_gesture_controller.config import THEME, GAME_URL, settings, CALIBRATION_FILE
from subway_gesture_controller.hand_detection import HandDetector, HandData
from subway_gesture_controller.gesture_recognition import GestureRecognizer, GestureType, GestureDetails
from subway_gesture_controller.keyboard_controller import KeyboardController, is_game_or_browser_focused
from subway_gesture_controller.calibration import HandCalibrator, CalibrationStep, STEP_DESCRIPTIONS

# Set CustomTkinter theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class GestureCard(ctk.CTkFrame):
    """Visual card displaying a game action that pulses/glows on trigger."""

    def __init__(
        self,
        master,
        action_name: str,
        display_label: str,
        icon: str,
        key_name: str,
        **kwargs
    ):
        super().__init__(
            master,
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=2,
            corner_radius=12,
            **kwargs
        )
        self.action_name = action_name
        self.display_label = display_label
        self.icon = icon
        self.key_name = key_name
        self.is_glowing = False
        self.trigger_count = 0

        self.grid_columnconfigure(0, weight=1)

        # Icon Label
        self.icon_lbl = ctk.CTkLabel(
            self,
            text=icon,
            font=ctk.CTkFont(size=28),
            text_color=THEME["accent_cyan"]
        )
        self.icon_lbl.grid(row=0, column=0, pady=(10, 2))

        # Title Label
        self.title_lbl = ctk.CTkLabel(
            self,
            text=display_label,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=THEME["text_primary"]
        )
        self.title_lbl.grid(row=1, column=0, pady=0)

        # Key Mapping Badge
        self.key_badge = ctk.CTkLabel(
            self,
            text=f"[{key_name.upper()}]",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=THEME["accent_blue"],
            fg_color=THEME["bg_secondary"],
            corner_radius=6,
            padx=8,
            pady=2
        )
        self.key_badge.grid(row=2, column=0, pady=(4, 4))

        # Trigger Count Label
        self.count_lbl = ctk.CTkLabel(
            self,
            text="Triggers: 0",
            font=ctk.CTkFont(size=10),
            text_color=THEME["text_muted"]
        )
        self.count_lbl.grid(row=3, column=0, pady=(0, 8))

    def update_key_name(self, new_key: str):
        self.key_name = new_key
        self.key_badge.configure(text=f"[{new_key.upper()}]")

    def pulse_glow(self):
        """Animates bright neon border glow when gesture triggers."""
        self.trigger_count += 1
        self.count_lbl.configure(text=f"Triggers: {self.trigger_count}")

        # Set glowing electric cyan style
        self.configure(
            fg_color=THEME["card_active"],
            border_color=THEME["accent_cyan"],
            border_width=3
        )
        self.icon_lbl.configure(text_color="#ffffff")
        self.title_lbl.configure(text_color=THEME["accent_cyan"])

        # Reset glow after 280ms
        self.after(280, self._reset_glow)

    def _reset_glow(self):
        self.configure(
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=2
        )
        self.icon_lbl.configure(text_color=THEME["accent_cyan"])
        self.title_lbl.configure(text_color=THEME["text_primary"])


class ControllerDashboard(ctk.CTk):
    """Main Application Window for AI Hand Gesture Gaming Controller."""

    def __init__(self):
        super().__init__()

        # Window properties
        self.title("⚡ AI Hand Gesture Gaming Controller | Subway Surfers")
        self.geometry("1240x840")
        self.minsize(1080, 720)
        self.configure(fg_color=THEME["bg_main"])

        # System Components
        self.detector = HandDetector(
            min_detection_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence
        )
        self.recognizer = GestureRecognizer(
            cooldown_left_right=settings.cooldown_left_right,
            cooldown_jump=settings.cooldown_jump,
            cooldown_roll=settings.cooldown_roll,
            smoothing_frames=settings.smoothing_window,
            min_trigger_consensus=settings.min_trigger_frames
        )
        self.keyboard = KeyboardController(
            key_preset=settings.key_preset,
            key_hold_time=settings.key_press_duration,
            enable_sound=settings.sound_feedback
        )
        self.calibrator = HandCalibrator(save_path=CALIBRATION_FILE)
        self.recognizer.apply_calibration(self.calibrator.parameters)

        # Connect keyboard trigger callback
        self.keyboard.on_action_triggered = self._on_action_executed

        # Video stream state & Threaded Vision Pipeline
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_running = False
        self.fps_tracker = deque(maxlen=20)
        self.vision_fps_tracker = deque(maxlen=20)
        self.last_frame_time = time.time()
        self.last_vision_time = time.time()
        self.current_action_banner = ""
        self.action_banner_expire = 0.0

        # Thread synchronization
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_hand_data: Optional[HandData] = None
        self.latest_details: Optional[GestureDetails] = None
        self.vision_thread: Optional[threading.Thread] = None

        # Build UI layout
        self._build_header()
        self._build_main_layout()

        # Calibration overlay
        self.calibration_active = False

        # Bind window keys
        self.bind("<Escape>", lambda e: self.emergency_stop())

        # Start background camera thread
        self.start_camera()

        # Start global ESC hotkey listener thread for Windows
        if HAS_CTYPES:
            self.hotkey_thread = threading.Thread(target=self._global_hotkey_listener, daemon=True)
            self.hotkey_thread.start()

        # Set window close handler
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_header(self):
        """Top Header Bar with Logo, Title, Emergency Stop Badge, and Game Launcher."""
        header_frame = ctk.CTkFrame(
            self,
            fg_color=THEME["bg_secondary"],
            height=68,
            corner_radius=0,
            border_color=THEME["card_border"],
            border_width=1
        )
        header_frame.pack(fill="x", side="top", pady=(0, 8))
        header_frame.pack_propagate(False)

        # Title Container with Logo
        title_box = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_box.pack(side="left", padx=20, pady=6)

        # Logo Icon
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")
        if not os.path.exists(logo_path):
            logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "logo.png")
        if os.path.exists(logo_path):
            try:
                logo_pil = Image.open(logo_path).resize((44, 44), Image.Resampling.LANCZOS)
                self.logo_ctk = ctk.CTkImage(light_image=logo_pil, dark_image=logo_pil, size=(44, 44))
                logo_lbl = ctk.CTkLabel(title_box, image=self.logo_ctk, text="")
                logo_lbl.pack(side="left", padx=(0, 12))
            except Exception as e:
                print(f"[Dashboard] Logo load error: {e}")

        text_sub_box = ctk.CTkFrame(title_box, fg_color="transparent")
        text_sub_box.pack(side="left")

        app_title = ctk.CTkLabel(
            text_sub_box,
            text="⚡ AI HAND GESTURE GAMING CONTROLLER",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=THEME["accent_cyan"]
        )
        app_title.pack(anchor="w")

        app_subtitle = ctk.CTkLabel(
            text_sub_box,
            text="Subway Surfers Real-Time Motion Control | MediaPipe 3D Landmark Vision",
            font=ctk.CTkFont(size=11),
            text_color=THEME["text_secondary"]
        )
        app_subtitle.pack(anchor="w")

        # Right Action Buttons
        right_box = ctk.CTkFrame(header_frame, fg_color="transparent")
        right_box.pack(side="right", padx=24, pady=8)

        # Emergency Stop pill
        esc_badge = ctk.CTkLabel(
            right_box,
            text="[ESC] EMERGENCY STOP",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=THEME["accent_red"],
            fg_color="#3b1219",
            corner_radius=8,
            padx=12,
            pady=6
        )
        esc_badge.pack(side="left", padx=10)

        # Launch Game Button
        launch_btn = ctk.CTkButton(
            right_box,
            text="🌐 LAUNCH SUBWAY SURFERS",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#0284c7",
            hover_color="#0369a1",
            text_color="#ffffff",
            corner_radius=8,
            height=36,
            command=self.open_game_in_browser
        )
        launch_btn.pack(side="left", padx=5)

    def _build_main_layout(self):
        """Constructs 2-column main responsive dashboard."""
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        self.main_container.grid_columnconfigure(0, weight=6)  # Camera + cards
        self.main_container.grid_columnconfigure(1, weight=4)  # Status + controls
        self.main_container.grid_rowconfigure(0, weight=1)

        # LEFT COLUMN: Camera Feed + Gesture Cards
        left_col = ctk.CTkFrame(self.main_container, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left_col.grid_rowconfigure(0, weight=1)
        left_col.grid_columnconfigure(0, weight=1)

        # Camera Viewport Frame
        self.cam_card = ctk.CTkFrame(
            left_col,
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=2,
            corner_radius=16
        )
        self.cam_card.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
        self.cam_card.grid_rowconfigure(0, weight=1)
        self.cam_card.grid_columnconfigure(0, weight=1)

        # Video Canvas / Label
        self.video_label = ctk.CTkLabel(
            self.cam_card,
            text="INITIALIZING CAMERA STREAM...",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=THEME["text_secondary"]
        )
        self.video_label.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")

        # Camera Bottom Metric Bar (Live Performance Monitoring)
        cam_bar = ctk.CTkFrame(self.cam_card, fg_color=THEME["bg_secondary"], height=38, corner_radius=10)
        cam_bar.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 10))

        self.fps_lbl = ctk.CTkLabel(
            cam_bar,
            text="CAM: 00.0 FPS | AI: 00.0 FPS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=THEME["accent_cyan"]
        )
        self.fps_lbl.pack(side="left", padx=12)

        self.perf_lbl = ctk.CTkLabel(
            cam_bar,
            text="LATENCY: 00ms | INFER: 00ms",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=THEME["accent_blue"]
        )
        self.perf_lbl.pack(side="left", padx=12)

        self.conf_lbl = ctk.CTkLabel(
            cam_bar,
            text="TRACKING: WAITING",
            font=ctk.CTkFont(size=11),
            text_color=THEME["text_secondary"]
        )
        self.conf_lbl.pack(side="left", padx=12)

        self.hand_pos_lbl = ctk.CTkLabel(
            cam_bar,
            text="POS: (---, ---)",
            font=ctk.CTkFont(size=11),
            text_color=THEME["text_muted"]
        )
        self.hand_pos_lbl.pack(side="right", padx=12)

        # Row of 4 Animated Gesture Cards
        cards_row = ctk.CTkFrame(left_col, fg_color="transparent")
        cards_row.grid(row=1, column=0, sticky="ew")
        for i in range(4):
            cards_row.grid_columnconfigure(i, weight=1)

        curr_keys = self.keyboard.get_current_mapping()
        self.cards: Dict[str, GestureCard] = {}

        self.cards["MOVE_LEFT"] = GestureCard(
            cards_row, "MOVE_LEFT", "MOVE LEFT", "👈", curr_keys["MOVE_LEFT"]
        )
        self.cards["MOVE_LEFT"].grid(row=0, column=0, padx=(0, 6), sticky="nsew")

        self.cards["MOVE_RIGHT"] = GestureCard(
            cards_row, "MOVE_RIGHT", "MOVE RIGHT", "👉", curr_keys["MOVE_RIGHT"]
        )
        self.cards["MOVE_RIGHT"].grid(row=0, column=1, padx=6, sticky="nsew")

        self.cards["JUMP"] = GestureCard(
            cards_row, "JUMP", "JUMP", "🖐️", curr_keys["JUMP"]
        )
        self.cards["JUMP"].grid(row=0, column=2, padx=6, sticky="nsew")

        self.cards["ROLL"] = GestureCard(
            cards_row, "ROLL", "ROLL / SLIDE", "✊", curr_keys["ROLL"]
        )
        self.cards["ROLL"].grid(row=0, column=3, padx=(6, 0), sticky="nsew")

        # RIGHT COLUMN: Hero Gesture Display + Status + Controls
        right_col = ctk.CTkFrame(self.main_container, fg_color="transparent")
        right_col.grid(row=0, column=1, sticky="nsew")

        # 1. HERO CURRENT GESTURE DISPLAY CARD
        self.hero_card = ctk.CTkFrame(
            right_col,
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=2,
            corner_radius=16
        )
        self.hero_card.pack(fill="x", pady=(0, 12))

        hero_hdr = ctk.CTkLabel(
            self.hero_card,
            text="CURRENT RECOGNIZED GESTURE",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=THEME["text_muted"]
        )
        hero_hdr.pack(anchor="w", padx=20, pady=(14, 4))

        self.gesture_badge = ctk.CTkLabel(
            self.hero_card,
            text="⏳ NO GESTURE – WAITING",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=THEME["accent_cyan"],
            fg_color=THEME["bg_secondary"],
            corner_radius=10,
            padx=16,
            pady=12
        )
        self.gesture_badge.pack(fill="x", padx=20, pady=(2, 10))

        # Finger State Chips (Thumb, Index, Middle, Ring, Pinky)
        chips_frame = ctk.CTkFrame(self.hero_card, fg_color="transparent")
        chips_frame.pack(fill="x", padx=20, pady=(0, 12))
        for i in range(5):
            chips_frame.grid_columnconfigure(i, weight=1)

        self.finger_labels: Dict[str, ctk.CTkLabel] = {}
        for idx, finger in enumerate(["Thumb", "Index", "Middle", "Ring", "Pinky"]):
            lbl = ctk.CTkLabel(
                chips_frame,
                text=f"{finger[:3]}: -",
                font=ctk.CTkFont(size=11),
                text_color=THEME["text_muted"],
                fg_color=THEME["bg_main"],
                corner_radius=6,
                pady=3
            )
            lbl.grid(row=0, column=idx, padx=2, sticky="ew")
            self.finger_labels[finger] = lbl

        # 2. GAME STATUS CARD
        status_card = ctk.CTkFrame(
            right_col,
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=2,
            corner_radius=16
        )
        status_card.pack(fill="x", pady=(0, 12))

        status_hdr = ctk.CTkLabel(
            status_card,
            text="GAME & SYSTEM STATUS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=THEME["text_muted"]
        )
        status_hdr.pack(anchor="w", padx=20, pady=(12, 6))

        # Status Table Grid
        stat_grid = ctk.CTkFrame(status_card, fg_color="transparent")
        stat_grid.pack(fill="x", padx=20, pady=(0, 14))
        stat_grid.grid_columnconfigure(1, weight=1)

        # 1. Camera Connected Status
        ctk.CTkLabel(stat_grid, text="Camera Connected:", font=ctk.CTkFont(size=12), text_color=THEME["text_secondary"]).grid(row=0, column=0, sticky="w", pady=2)
        self.stat_cam = ctk.CTkLabel(stat_grid, text="INITIALIZING...", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["accent_cyan"])
        self.stat_cam.grid(row=0, column=1, sticky="e", pady=2)

        # 2. Hand Detected Status
        ctk.CTkLabel(stat_grid, text="Hand Detected:", font=ctk.CTkFont(size=12), text_color=THEME["text_secondary"]).grid(row=1, column=0, sticky="w", pady=2)
        self.stat_hand = ctk.CTkLabel(stat_grid, text="Searching...", font=ctk.CTkFont(size=12), text_color=THEME["text_muted"])
        self.stat_hand.grid(row=1, column=1, sticky="e", pady=2)

        # 3. Gesture Recognized Status
        ctk.CTkLabel(stat_grid, text="Gesture Recognized:", font=ctk.CTkFont(size=12), text_color=THEME["text_secondary"]).grid(row=2, column=0, sticky="w", pady=2)
        self.stat_gest = ctk.CTkLabel(stat_grid, text="WAITING", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["text_secondary"])
        self.stat_gest.grid(row=2, column=1, sticky="e", pady=2)

        # 4. Game Control Active Status
        ctk.CTkLabel(stat_grid, text="Game Control Active:", font=ctk.CTkFont(size=12), text_color=THEME["text_secondary"]).grid(row=3, column=0, sticky="w", pady=2)
        self.stat_ctrl = ctk.CTkLabel(stat_grid, text="PAUSED", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["accent_orange"])
        self.stat_ctrl.grid(row=3, column=1, sticky="e", pady=2)

        # 5. Browser / Game Focus Status
        ctk.CTkLabel(stat_grid, text="Game Window Focus:", font=ctk.CTkFont(size=12), text_color=THEME["text_secondary"]).grid(row=4, column=0, sticky="w", pady=2)
        self.stat_focus = ctk.CTkLabel(stat_grid, text="CHECKING...", font=ctk.CTkFont(size=11, weight="bold"), text_color=THEME["accent_orange"])
        self.stat_focus.grid(row=4, column=1, sticky="e", pady=2)

        # 6. Last Action Sent
        ctk.CTkLabel(stat_grid, text="Last Action Sent:", font=ctk.CTkFont(size=12), text_color=THEME["text_secondary"]).grid(row=5, column=0, sticky="w", pady=2)
        self.stat_action = ctk.CTkLabel(stat_grid, text="None", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["text_primary"])
        self.stat_action.grid(row=5, column=1, sticky="e", pady=2)

        # 7. Total Gestures Detected
        ctk.CTkLabel(stat_grid, text="Total Gestures Detected:", font=ctk.CTkFont(size=12), text_color=THEME["text_secondary"]).grid(row=6, column=0, sticky="w", pady=2)
        self.stat_total = ctk.CTkLabel(stat_grid, text="0", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["accent_blue"])
        self.stat_total.grid(row=6, column=1, sticky="e", pady=2)

        # 3. CONTROL BUTTONS CARD
        btn_card = ctk.CTkFrame(
            right_col,
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=2,
            corner_radius=16
        )
        btn_card.pack(fill="x", pady=(0, 12))

        btn_hdr = ctk.CTkLabel(
            btn_card,
            text="CONTROLLER ACTIONS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=THEME["text_muted"]
        )
        btn_hdr.pack(anchor="w", padx=20, pady=(12, 8))

        btn_grid = ctk.CTkFrame(btn_card, fg_color="transparent")
        btn_grid.pack(fill="x", padx=20, pady=(0, 14))
        btn_grid.grid_columnconfigure((0, 1), weight=1)

        # START GAME CONTROL button
        self.start_ctrl_btn = ctk.CTkButton(
            btn_grid,
            text="🎮 START GAME CONTROL",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["accent_green"],
            hover_color="#059669",
            height=36,
            corner_radius=8,
            command=self.start_game_control
        )
        self.start_ctrl_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew", pady=(0, 6))

        # PAUSE CONTROL button
        self.pause_ctrl_btn = ctk.CTkButton(
            btn_grid,
            text="⏸️ PAUSE CONTROL",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["accent_orange"],
            hover_color="#d97706",
            height=36,
            corner_radius=8,
            command=self.pause_game_control
        )
        self.pause_ctrl_btn.grid(row=0, column=1, padx=(4, 0), sticky="ew", pady=(0, 6))

        # START CAMERA button
        self.start_cam_btn = ctk.CTkButton(
            btn_grid,
            text="📹 START CAMERA",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["card_border"],
            hover_color="#334155",
            height=34,
            corner_radius=8,
            command=self.start_camera
        )
        self.start_cam_btn.grid(row=1, column=0, padx=(0, 4), sticky="ew", pady=3)

        # STOP CAMERA button
        self.stop_cam_btn = ctk.CTkButton(
            btn_grid,
            text="⏹️ STOP CAMERA",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["card_border"],
            hover_color="#334155",
            height=34,
            corner_radius=8,
            command=self.stop_camera
        )
        self.stop_cam_btn.grid(row=1, column=1, padx=(4, 0), sticky="ew", pady=3)

        # CALIBRATE HAND button
        self.calib_btn = ctk.CTkButton(
            btn_grid,
            text="🎯 CALIBRATE HAND",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#0284c7",
            hover_color="#0369a1",
            height=34,
            corner_radius=8,
            command=self.open_calibration_dialog
        )
        self.calib_btn.grid(row=2, column=0, padx=(0, 4), sticky="ew", pady=3)

        # OPEN SUBWAY SURFERS button
        self.open_game_btn = ctk.CTkButton(
            btn_grid,
            text="🌐 OPEN SUBWAY SURFERS",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#0891b2",
            hover_color="#0e7490",
            height=34,
            corner_radius=8,
            command=self.open_game_in_browser
        )
        self.open_game_btn.grid(row=2, column=1, padx=(4, 0), sticky="ew", pady=3)

        # EXIT APPLICATION button
        exit_btn = ctk.CTkButton(
            btn_grid,
            text="❌ EXIT APPLICATION",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#3b1219",
            hover_color="#5a1824",
            text_color=THEME["accent_red"],
            height=32,
            corner_radius=8,
            command=self.on_close
        )
        exit_btn.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        # 4. TUNING SETTINGS CARD
        settings_card = ctk.CTkFrame(
            right_col,
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=2,
            corner_radius=16
        )
        settings_card.pack(fill="x")

        set_hdr = ctk.CTkLabel(
            settings_card,
            text="SETTINGS & COOLDOWN TUNING",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=THEME["text_muted"]
        )
        set_hdr.pack(anchor="w", padx=20, pady=(12, 6))

        set_inner = ctk.CTkFrame(settings_card, fg_color="transparent")
        set_inner.pack(fill="x", padx=20, pady=(0, 14))

        # Response Profile (NORMAL, FAST, ULTRA FAST)
        mode_row = ctk.CTkFrame(set_inner, fg_color="transparent")
        mode_row.pack(fill="x", pady=(2, 6))
        ctk.CTkLabel(mode_row, text="Response Mode:", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["accent_cyan"]).pack(side="left")

        self.mode_seg = ctk.CTkSegmentedButton(
            mode_row,
            values=["NORMAL", "FAST", "ULTRA FAST"],
            command=self._on_performance_mode_changed,
            selected_color=THEME["accent_cyan"],
            selected_hover_color="#0284c7"
        )
        self.mode_seg.set(settings.performance_mode)
        self.mode_seg.pack(side="right")

        # Preset Selector (Arrows vs WASD)
        preset_row = ctk.CTkFrame(set_inner, fg_color="transparent")
        preset_row.pack(fill="x", pady=4)
        ctk.CTkLabel(preset_row, text="Control Scheme:", font=ctk.CTkFont(size=12), text_color=THEME["text_secondary"]).pack(side="left")

        self.preset_seg = ctk.CTkSegmentedButton(
            preset_row,
            values=["ARROWS", "WASD"],
            command=self._on_preset_changed,
            selected_color=THEME["accent_blue"],
            selected_hover_color="#0284c7"
        )
        self.preset_seg.set(settings.key_preset.upper())
        self.preset_seg.pack(side="right")

        # Jump Cooldown Slider
        jump_row = ctk.CTkFrame(set_inner, fg_color="transparent")
        jump_row.pack(fill="x", pady=(6, 2))
        self.jump_cd_lbl = ctk.CTkLabel(jump_row, text=f"Jump Cooldown: {int(settings.cooldown_jump*1000)} ms", font=ctk.CTkFont(size=11), text_color=THEME["text_secondary"])
        self.jump_cd_lbl.pack(side="left")

        self.jump_slider = ctk.CTkSlider(
            set_inner,
            from_=100,
            to=600,
            number_of_steps=10,
            command=self._on_jump_cd_slider
        )
        self.jump_slider.set(settings.cooldown_jump * 1000)
        self.jump_slider.pack(fill="x", pady=(0, 6))

        # Lane Change (Left/Right) Cooldown Slider
        lr_row = ctk.CTkFrame(set_inner, fg_color="transparent")
        lr_row.pack(fill="x", pady=(2, 2))
        self.lr_cd_lbl = ctk.CTkLabel(lr_row, text=f"Left/Right Cooldown: {int(settings.cooldown_left_right*1000)} ms", font=ctk.CTkFont(size=11), text_color=THEME["text_secondary"])
        self.lr_cd_lbl.pack(side="left")

        self.lr_slider = ctk.CTkSlider(
            set_inner,
            from_=80,
            to=500,
            number_of_steps=14,
            command=self._on_lr_cd_slider
        )
        self.lr_slider.set(settings.cooldown_left_right * 1000)
        self.lr_slider.pack(fill="x", pady=(0, 6))

        # Sound Feedback Switch
        sound_row = ctk.CTkFrame(set_inner, fg_color="transparent")
        sound_row.pack(fill="x", pady=2)
        ctk.CTkLabel(sound_row, text="Audio Pitch Feedback:", font=ctk.CTkFont(size=12), text_color=THEME["text_secondary"]).pack(side="left")
        self.sound_switch = ctk.CTkSwitch(
            sound_row,
            text="",
            command=self._on_sound_toggle,
            progress_color=THEME["accent_cyan"]
        )
        if settings.sound_feedback:
            self.sound_switch.select()
        else:
            self.sound_switch.deselect()
        self.sound_switch.pack(side="right")

    # ==========================
    # ACTION & CONTROL HANDLERS
    # ==========================

    def start_game_control(self):
        """Enables keyboard controller commands."""
        self.keyboard.set_active(True)
        self.stat_ctrl.configure(text="ACTIVE (READY)", text_color=THEME["accent_green"])
        print("[Dashboard] Game Control ACTIVATED!")

    def pause_game_control(self):
        """Pauses keyboard controller commands."""
        self.keyboard.set_active(False)
        self.stat_ctrl.configure(text="PAUSED", text_color=THEME["accent_orange"])
        print("[Dashboard] Game Control PAUSED!")

    def toggle_game_control(self):
        """Toggles between ACTIVE and PAUSED keyboard controller states."""
        if self.keyboard.is_active:
            self.pause_game_control()
        else:
            self.start_game_control()

    def emergency_stop(self):
        """Immediately pauses the controller and protects game state."""
        self.pause_game_control()
        self.stat_ctrl.configure(text="PAUSED (EMERGENCY)", text_color=THEME["accent_red"])
        print("[Dashboard] EMERGENCY STOP TRIGGERED VIA ESC!")

    def _global_hotkey_listener(self):
        """Polls Windows GetAsyncKeyState for ESC hotkey across background and games."""
        user32 = ctypes.windll.user32
        while True:
            try:
                # Check if ESC key was pressed (bit 15 set)
                if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
                    self.after(0, self.emergency_stop)
                    time.sleep(0.3)  # Debounce hotkey
            except Exception:
                pass
            time.sleep(0.05)

    def open_game_in_browser(self):
        """Opens official Subway Surfers in user's default browser."""
        print(f"[Dashboard] Opening Subway Surfers: {GAME_URL}")
        webbrowser.open(GAME_URL)

    def toggle_camera(self):
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if self.camera_running:
            return
        self.cap = cv2.VideoCapture(settings.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(settings.camera_index)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)
            self.camera_running = True
            if hasattr(self, "stat_cam"):
                self.stat_cam.configure(text="CONNECTED (ACTIVE)", text_color=THEME["accent_green"])

            # Start high-speed dedicated vision worker thread
            self.vision_thread = threading.Thread(target=self._vision_worker_loop, daemon=True)
            self.vision_thread.start()

            # Start smooth decoupled UI refresh loop
            self.after(20, self._ui_update_loop)
        else:
            if hasattr(self, "stat_cam"):
                self.stat_cam.configure(text="OFFLINE (NO WEBCAM)", text_color=THEME["accent_red"])
            self.video_label.configure(text="CAMERA UNAVAILABLE (CHECK WEBCAM)")

    def stop_camera(self):
        self.camera_running = False
        if self.vision_thread and self.vision_thread.is_alive():
            self.vision_thread.join(timeout=0.3)
            self.vision_thread = None
        if self.cap:
            self.cap.release()
            self.cap = None
        if hasattr(self, "stat_cam"):
            self.stat_cam.configure(text="STOPPED", text_color=THEME["accent_orange"])
        self.video_label.configure(image="", text="CAMERA STOPPED")

    def _on_action_executed(self, action: str, key: str):
        """Called asynchronously when a keyboard keystroke is dispatched."""
        self.after(0, lambda: self._update_ui_on_action(action, key))

    def _update_ui_on_action(self, action: str, key: str):
        # Pulse corresponding card glow
        if action in self.cards:
            self.cards[action].pulse_glow()

        # Update last action badge
        self.stat_action.configure(text=f"{action} [{key.upper()}]")
        self.current_action_banner = f"{action} -> [{key.upper()}]"
        self.action_banner_expire = time.time() + 0.4

    # ==========================
    # SETTINGS HANDLERS
    # ==========================

    def _on_performance_mode_changed(self, value: str):
        """Switches active response profile between NORMAL, FAST, and ULTRA FAST."""
        settings.apply_performance_mode(value)
        self.recognizer.update_performance_mode(
            settings.smoothing_window,
            settings.min_trigger_frames
        )
        self.recognizer.update_cooldowns(
            settings.cooldown_left_right,
            settings.cooldown_jump,
            settings.cooldown_roll
        )
        self.keyboard.set_key_hold_time(settings.key_press_duration)

        # Update sliders to match profile
        self.jump_slider.set(settings.cooldown_jump * 1000)
        self.jump_cd_lbl.configure(text=f"Jump Cooldown: {int(settings.cooldown_jump * 1000)} ms")

        self.lr_slider.set(settings.cooldown_left_right * 1000)
        self.lr_cd_lbl.configure(text=f"Left/Right Cooldown: {int(settings.cooldown_left_right * 1000)} ms")

        settings.save_to_file()
        print(f"[Dashboard] Switched to {value} profile (Hold: {int(settings.key_press_duration*1000)}ms, CD: {int(settings.cooldown_left_right*1000)}ms)")

    def _on_preset_changed(self, value: str):
        settings.key_preset = value
        self.keyboard.set_key_preset(value)
        curr_keys = self.keyboard.get_current_mapping()

        for act_name, card in self.cards.items():
            if act_name in curr_keys:
                card.update_key_name(curr_keys[act_name])

        preset_desc = "ARROWS (↑ ↓ ← →)" if value == "ARROWS" else "WASD (W S A D)"
        self.stat_preset.configure(text=preset_desc)
        settings.save_to_file()

    def _on_jump_cd_slider(self, val: float):
        sec = round(val / 1000.0, 2)
        settings.cooldown_jump = sec
        self.jump_cd_lbl.configure(text=f"Jump Cooldown: {int(val)} ms")
        self.recognizer.update_cooldowns(
            settings.cooldown_left_right,
            settings.cooldown_jump,
            settings.cooldown_roll
        )
        settings.save_to_file()

    def _on_lr_cd_slider(self, val: float):
        sec = round(val / 1000.0, 2)
        settings.cooldown_left_right = sec
        self.lr_cd_lbl.configure(text=f"Left/Right Cooldown: {int(val)} ms")
        self.recognizer.update_cooldowns(
            settings.cooldown_left_right,
            settings.cooldown_jump,
            settings.cooldown_roll
        )
        settings.save_to_file()

    def _on_sound_toggle(self):
        settings.sound_feedback = bool(self.sound_switch.get())
        self.keyboard.enable_sound = settings.sound_feedback
        settings.save_to_file()

    # ==========================
    # CALIBRATION DIALOG
    # ==========================

    def open_calibration_dialog(self):
        """Displays interactive calibration wizard."""
        self.calibrator.start_calibration()

        # Create sleek Toplevel popup
        calib_win = ctk.CTkToplevel(self)
        calib_win.title("🎯 Hand Calibration Wizard")
        calib_win.geometry("520x400")
        calib_win.configure(fg_color=THEME["bg_secondary"])
        calib_win.attributes("-topmost", True)

        hdr = ctk.CTkLabel(
            calib_win,
            text="AI HAND CALIBRATION",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=THEME["accent_cyan"]
        )
        hdr.pack(pady=(20, 4))

        step_lbl = ctk.CTkLabel(
            calib_win,
            text="STEP 1 / 4: OPEN PALM",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#ffffff"
        )
        step_lbl.pack(pady=6)

        desc_lbl = ctk.CTkLabel(
            calib_win,
            text="Place hand in camera view and FULLY EXTEND ALL 5 FINGERS.",
            font=ctk.CTkFont(size=13),
            text_color=THEME["text_secondary"],
            wraplength=460
        )
        desc_lbl.pack(pady=10)

        prog_bar = ctk.CTkProgressBar(calib_win, width=420, height=14, progress_color=THEME["accent_cyan"])
        prog_bar.set(0.0)
        prog_bar.pack(pady=16)

        status_msg = ctk.CTkLabel(
            calib_win,
            text="Collecting calibration samples...",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_muted"]
        )
        status_msg.pack(pady=8)

        def step_listener(step_idx, title, desc, progress):
            step_lbl.configure(text=title)
            desc_lbl.configure(text=desc)
            prog_bar.set(progress)
            if step_idx == CalibrationStep.STEP_FINISHED:
                status_msg.configure(text="✅ Calibration parameters saved successfully!", text_color=THEME["accent_green"])
                # Apply new calibration to recognizer
                self.recognizer.apply_calibration(self.calibrator.parameters)
                calib_win.after(2000, calib_win.destroy)

        self.calibrator.on_step_changed = step_listener

        def on_cancel():
            self.calibrator.cancel_calibration()
            calib_win.destroy()

        cancel_btn = ctk.CTkButton(
            calib_win,
            text="Cancel Calibration",
            fg_color="#334155",
            hover_color="#475569",
            command=on_cancel
        )
        cancel_btn.pack(pady=(16, 0))

    # ==========================
    # VIDEO & DETECTION PIPELINE
    # ==========================

    def _vision_worker_loop(self):
        """
        Ultra-fast dedicated camera acquisition and vision inference thread.
        Bypasses Tkinter GUI thread completely to achieve sub-50ms latency.
        Dispatches actions immediately to the keyboard controller on detection.
        """
        print("[Dashboard] Ultra-Fast Vision Worker Thread ACTIVE.")
        while self.camera_running and self.cap is not None:
            ret, frame_bgr = self.cap.read()
            if not ret or frame_bgr is None:
                time.sleep(0.005)
                continue

            now = time.time()
            v_fps = 1.0 / max(1e-4, (now - self.last_vision_time))
            self.vision_fps_tracker.append(v_fps)
            self.last_vision_time = now

            # 1. MediaPipe 3D Landmark Vision
            processed_frame, hand_data = self.detector.process_frame(frame_bgr, mirror=True)

            # 2. Calibration sampling if wizard is open
            if self.calibrator.is_calibrating():
                self.calibrator.process_frame(hand_data)

            # 3. High-Speed Geometric Gesture Classification
            gesture_details: GestureDetails = self.recognizer.process(hand_data)

            # 4. INSTANT ACTION DISPATCH: Fire keyboard event immediately on vision thread!
            if gesture_details.action_to_trigger:
                self.keyboard.trigger_action(gesture_details.action_to_trigger)

            # 5. Check action banner expiration
            banner_text = self.current_action_banner if now < self.action_banner_expire else ""

            # 6. Render futuristic HUD on frame
            hud_frame = self.detector.draw_hud(
                processed_frame,
                hand_data,
                current_gesture=gesture_details.confirmed_gesture,
                action_name=banner_text,
                is_active=self.keyboard.is_active
            )

            # 7. Post to thread-safe shared slot for the decoupled GUI
            with self.frame_lock:
                self.latest_frame = hud_frame
                self.latest_hand_data = hand_data
                self.latest_details = gesture_details

    def _ui_update_loop(self):
        """
        Decoupled 60 FPS UI rendering loop.
        Retrieves the latest available frame and updates metrics without stalling video.
        """
        if not self.camera_running:
            return

        with self.frame_lock:
            frame = self.latest_frame
            hand_data = self.latest_hand_data
            details = self.latest_details

        if frame is not None:
            now = time.time()
            ui_fps = 1.0 / max(1e-4, (now - self.last_frame_time))
            self.fps_tracker.append(ui_fps)
            self.last_frame_time = now

            # Render frame to canvas
            self._render_video_frame(frame)

            # Live telemetry update
            if details is not None:
                cam_fps = np.mean(self.vision_fps_tracker) if len(self.vision_fps_tracker) > 0 else 30.0
                infer_ms = hand_data.inference_ms if hand_data else self.detector.last_inference_ms
                lat_ms = self.keyboard.last_latency_ms
                self._update_dashboard_widgets(hand_data, details, cam_fps, infer_ms, lat_ms)

        # Schedule next GUI refresh at ~60 FPS (16ms)
        self.after(16, self._ui_update_loop)

    def _update_dashboard_widgets(
        self,
        hand_data: Optional[HandData],
        details: GestureDetails,
        cam_fps: float,
        infer_ms: float = 0.0,
        lat_ms: float = 0.0
    ):
        # FPS & Latency Telemetry
        ui_fps = np.mean(self.fps_tracker) if len(self.fps_tracker) > 0 else 30.0
        self.fps_lbl.configure(text=f"CAM: {cam_fps:.1f} FPS | UI: {ui_fps:.1f}")
        self.perf_lbl.configure(text=f"LAT: {lat_ms:.0f}ms | INFER: {infer_ms:.0f}ms")

        # Hand Tracking & Position
        if hand_data is not None:
            conf_pct = int(hand_data.confidence * 100)
            self.conf_lbl.configure(text=f"TRACKING: {conf_pct}% ({hand_data.handedness})", text_color=THEME["accent_green"])
            self.hand_pos_lbl.configure(text=f"POS: ({hand_data.palm_center[0]}, {hand_data.palm_center[1]})")
            self.stat_hand.configure(text="LOCKED", text_color=THEME["accent_green"])
        else:
            self.conf_lbl.configure(text="NO HAND DETECTED", text_color=THEME["text_muted"])
            self.hand_pos_lbl.configure(text="POS: (---, ---)")
            self.stat_hand.configure(text="SEARCHING...", text_color=THEME["accent_orange"])

        # Hero Gesture Badge & Recognized Gesture
        gesture_map_labels = {
            GestureType.NONE: ("⏳ NO GESTURE – WAITING", THEME["text_secondary"]),
            GestureType.NEUTRAL: ("✋ NEUTRAL – READY", THEME["text_secondary"]),
            GestureType.JUMP: ("🖐️ OPEN PALM – JUMP", THEME["accent_cyan"]),
            GestureType.ROLL: ("✊ CLOSED FIST – ROLL", THEME["accent_cyan"]),
            GestureType.MOVE_RIGHT: ("👉 INDEX RIGHT – MOVE RIGHT", THEME["accent_cyan"]),
            GestureType.MOVE_LEFT: ("👈 INDEX LEFT – MOVE LEFT", THEME["accent_cyan"]),
        }
        badge_text, badge_color = gesture_map_labels.get(details.confirmed_gesture, ("WAITING", THEME["text_secondary"]))
        self.gesture_badge.configure(text=badge_text, text_color=badge_color)
        if hasattr(self, "stat_gest"):
            self.stat_gest.configure(text=details.confirmed_gesture)

        # Check Active Window Focus (Subway Surfers Browser)
        if hasattr(self, "stat_focus"):
            if is_game_or_browser_focused():
                self.stat_focus.configure(text="✅ FOCUSED (RECEIVING INPUT)", text_color=THEME["accent_green"])
            else:
                self.stat_focus.configure(text="⚠️ CLICK GAME TO FOCUS", text_color=THEME["accent_orange"])

        # Update Total Gestures Counter
        if hasattr(self, "stat_total"):
            self.stat_total.configure(text=str(self.keyboard.stats["TOTAL"]))

        # Finger State Chips
        for finger, is_ext in details.finger_states.items():
            if finger in self.finger_labels:
                status_char = "✓" if is_ext else "✗"
                color = THEME["accent_cyan"] if is_ext else THEME["text_muted"]
                self.finger_labels[finger].configure(
                    text=f"{finger[:3]}: {status_char}",
                    text_color=color
                )

    def _render_video_frame(self, bgr_frame: np.ndarray):
        """Converts OpenCV BGR image into CTkImage and displays it."""
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # Resize smoothly to fit the view card while maintaining aspect ratio
        target_w = 640
        target_h = 480
        h, w, _ = rgb_frame.shape
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(rgb_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pil_image = Image.fromarray(resized)
        ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(new_w, new_h))

        self.video_label.configure(image=ctk_image, text="")
        self.video_label.image = ctk_image

    def on_close(self):
        """Safe shutdown of camera, keyboard threads, and GUI."""
        print("[Dashboard] Shutting down AI Hand Gesture Controller...")
        self.stop_camera()
        self.keyboard.shutdown()
        self.detector.release()
        try:
            self.destroy()
        except Exception:
            pass
        os._exit(0)
