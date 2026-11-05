"""
Hand Detection Module using MediaPipe HandLandmarker
Provides accurate 21-landmark 3D hand detection, temporal smoothing,
and futuristic gaming HUD visualization.
"""
import os
import time
import urllib.request
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Landmark connectivity mapping for drawing
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckles
    (5, 9), (9, 13), (13, 17)
]

FINGERTIP_INDICES = [4, 8, 12, 16, 20]

@dataclass
class HandData:
    """Encapsulates hand detection output and derived geometry."""
    landmarks: np.ndarray             # 21 x 3 normalized coordinates (x, y, z)
    pixel_landmarks: np.ndarray       # 21 x 2 pixel coordinates (x, y)
    handedness: str                   # 'Right' or 'Left'
    confidence: float                 # Confidence score [0.0 - 1.0]
    palm_center: Tuple[int, int]      # Pixel coordinates (x, y) of palm center
    palm_size: float                  # Palm normalization scale (wrist to middle MCP)
    bbox: Tuple[int, int, int, int]   # (x_min, y_min, x_max, y_max) in pixels
    timestamp_ms: int = 0


class HandDetector:
    """Real-time hand detector using MediaPipe Tasks HandLandmarker."""

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_detection_confidence: float = 0.65,
        min_tracking_confidence: float = 0.65,
        smoothing_alpha: float = 0.70
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smoothing_alpha = smoothing_alpha

        # Resolve model file path
        self.model_path = self._resolve_model(model_path)

        # Initialize detector
        self._init_detector()

        # State tracking
        self.prev_landmarks: Optional[np.ndarray] = None
        self.last_timestamp_ms: int = 0
        self.frame_count: int = 0

    def _resolve_model(self, custom_path: Optional[str]) -> str:
        """Finds model file or downloads it automatically if missing."""
        candidate_paths = []
        if custom_path:
            candidate_paths.append(custom_path)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        current_dir = os.path.dirname(os.path.abspath(__file__))

        candidate_paths.extend([
            os.path.join(current_dir, "models", "hand_landmarker.task"),
            os.path.join(base_dir, "models", "hand_landmarker.task"),
            os.path.join(base_dir, "hand_landmarker.task"),
            os.path.join(current_dir, "hand_landmarker.task"),
            "hand_landmarker.task"
        ])

        for path in candidate_paths:
            if os.path.exists(path) and os.path.getsize(path) > 1000000:
                return os.path.abspath(path)

        # Download if not found
        target_dir = os.path.join(current_dir, "models")
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, "hand_landmarker.task")

        print(f"[HandDetector] Downloading hand landmarker model from {self.MODEL_URL}...")
        try:
            urllib.request.urlretrieve(self.MODEL_URL, target_path)
            print(f"[HandDetector] Model downloaded successfully to {target_path}")
            return target_path
        except Exception as e:
            raise RuntimeError(f"Failed to download MediaPipe model: {e}")

    def _init_detector(self):
        """Creates the HandLandmarker instance."""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray, mirror: bool = True) -> Tuple[np.ndarray, Optional[HandData]]:
        """
        Processes a raw camera frame.
        - Mirrors frame horizontally if mirror=True
        - Performs hand landmark detection
        - Applies temporal smoothing
        - Returns (processed_bgr_frame, HandData or None)
        """
        self.frame_count += 1
        h, w, _ = frame_bgr.shape

        if mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run inference
        try:
            detection_result = self.detector.detect(mp_image)
        except Exception as e:
            # Fallback re-init if an unexpected engine error occurs
            print(f"[HandDetector] Warning: detection error: {e}")
            self.prev_landmarks = None
            return frame_bgr, None

        if not detection_result.hand_landmarks:
            # Hand lost: reset smoothing filter
            self.prev_landmarks = None
            return frame_bgr, None

        # Extract primary hand landmarks (21 points)
        raw_lm = detection_result.hand_landmarks[0]
        points = np.array([[lm.x, lm.y, lm.z] for lm in raw_lm], dtype=np.float32)

        # Apply exponential moving average (EMA) smoothing for stability
        if self.prev_landmarks is not None and self.prev_landmarks.shape == points.shape:
            smoothed_points = self.smoothing_alpha * points + (1.0 - self.smoothing_alpha) * self.prev_landmarks
        else:
            smoothed_points = points

        self.prev_landmarks = smoothed_points.copy()

        # Calculate pixel coordinates
        pixel_x = np.clip(smoothed_points[:, 0] * w, 0, w - 1).astype(np.int32)
        pixel_y = np.clip(smoothed_points[:, 1] * h, 0, h - 1).astype(np.int32)
        pixel_landmarks = np.column_stack((pixel_x, pixel_y))

        # Extract confidence & handedness
        confidence = 0.85
        handedness = "Hand"
        if detection_result.handedness and len(detection_result.handedness) > 0:
            h_info = detection_result.handedness[0][0]
            confidence = float(h_info.score)
            handedness = h_info.category_name

        # Palm center (approximate centroid of wrist and knuckles: 0, 5, 9, 13, 17)
        palm_indices = [0, 5, 9, 13, 17]
        palm_center = (
            int(np.mean(pixel_landmarks[palm_indices, 0])),
            int(np.mean(pixel_landmarks[palm_indices, 1]))
        )

        # Palm size normalization factor (wrist landmark 0 to middle MCP landmark 9)
        wrist_pt = smoothed_points[0, :2]
        mid_mcp_pt = smoothed_points[9, :2]
        palm_size = float(np.linalg.norm(mid_mcp_pt - wrist_pt))
        if palm_size < 1e-4:
            palm_size = 0.1

        # Bounding box with margin
        margin = 25
        min_x = max(0, int(np.min(pixel_landmarks[:, 0])) - margin)
        min_y = max(0, int(np.min(pixel_landmarks[:, 1])) - margin)
        max_x = min(w, int(np.max(pixel_landmarks[:, 0])) + margin)
        max_y = min(h, int(np.max(pixel_landmarks[:, 1])) + margin)
        bbox = (min_x, min_y, max_x, max_y)

        hand_data = HandData(
            landmarks=smoothed_points,
            pixel_landmarks=pixel_landmarks,
            handedness=handedness,
            confidence=confidence,
            palm_center=palm_center,
            palm_size=palm_size,
            bbox=bbox,
            timestamp_ms=int(time.time() * 1000)
        )

        return frame_bgr, hand_data

    def draw_hud(
        self,
        frame: np.ndarray,
        hand_data: Optional[HandData],
        current_gesture: str = "WAITING",
        action_name: str = "",
        is_active: bool = True
    ) -> np.ndarray:
        """
        Renders a futuristic cyberpunk HUD on the video frame:
        - Sleek glowing bones
        - High-tech circular landmark nodes
        - Neon fingertip indicators
        - Hand bounding frame with corner brackets
        - Real-time gesture banner
        """
        hud_frame = frame.copy()
        h, w, _ = hud_frame.shape

        if hand_data is not None:
            pts = hand_data.pixel_landmarks

            # 1. Draw glowing bones
            for start_idx, end_idx in HAND_CONNECTIONS:
                pt1 = tuple(pts[start_idx])
                pt2 = tuple(pts[end_idx])
                # Outer glow line (dark electric blue)
                cv2.line(hud_frame, pt1, pt2, (180, 100, 0), 4, cv2.LINE_AA)
                # Inner crisp line (bright cyan)
                cv2.line(hud_frame, pt1, pt2, (255, 230, 0), 2, cv2.LINE_AA)

            # 2. Draw landmark joints
            for idx, pt in enumerate(pts):
                pt_tuple = tuple(pt)
                if idx in FINGERTIP_INDICES:
                    # Fingertips: larger pulsing glow
                    if idx == 8: # Index finger tip - special highlight
                        cv2.circle(hud_frame, pt_tuple, 9, (255, 0, 128), -1, cv2.LINE_AA)
                        cv2.circle(hud_frame, pt_tuple, 5, (255, 255, 255), -1, cv2.LINE_AA)
                    else:
                        cv2.circle(hud_frame, pt_tuple, 7, (255, 200, 0), -1, cv2.LINE_AA)
                        cv2.circle(hud_frame, pt_tuple, 4, (255, 255, 255), -1, cv2.LINE_AA)
                else:
                    # Intermediate joints
                    cv2.circle(hud_frame, pt_tuple, 4, (255, 120, 0), -1, cv2.LINE_AA)
                    cv2.circle(hud_frame, pt_tuple, 2, (255, 255, 255), -1, cv2.LINE_AA)

            # 3. Draw palm center target
            pc = hand_data.palm_center
            cv2.circle(hud_frame, pc, 6, (0, 240, 255), 1, cv2.LINE_AA)
            cv2.line(hud_frame, (pc[0] - 8, pc[1]), (pc[0] + 8, pc[1]), (0, 240, 255), 1, cv2.LINE_AA)
            cv2.line(hud_frame, (pc[0], pc[1] - 8), (pc[0], pc[1] + 8), (0, 240, 255), 1, cv2.LINE_AA)

            # 4. Sci-Fi corner brackets on hand bounding box
            bx1, by1, bx2, by2 = hand_data.bbox
            corner_len = min(20, (bx2 - bx1) // 4, (by2 - by1) // 4)
            bracket_color = (0, 242, 254) if is_active else (100, 100, 100)

            # Top-left
            cv2.line(hud_frame, (bx1, by1), (bx1 + corner_len, by1), bracket_color, 2, cv2.LINE_AA)
            cv2.line(hud_frame, (bx1, by1), (bx1, by1 + corner_len), bracket_color, 2, cv2.LINE_AA)
            # Top-right
            cv2.line(hud_frame, (bx2, by1), (bx2 - corner_len, by1), bracket_color, 2, cv2.LINE_AA)
            cv2.line(hud_frame, (bx2, by1), (bx2, by1 + corner_len), bracket_color, 2, cv2.LINE_AA)
            # Bottom-left
            cv2.line(hud_frame, (bx1, by2), (bx1 + corner_len, by2), bracket_color, 2, cv2.LINE_AA)
            cv2.line(hud_frame, (bx1, by2), (bx1, by2 - corner_len), bracket_color, 2, cv2.LINE_AA)
            # Bottom-right
            cv2.line(hud_frame, (bx2, by2), (bx2 - corner_len, by2), bracket_color, 2, cv2.LINE_AA)
            cv2.line(hud_frame, (bx2, by2), (bx2, by2 - corner_len), bracket_color, 2, cv2.LINE_AA)

            # Confidence badge near bounding box
            conf_pct = int(hand_data.confidence * 100)
            conf_str = f"TRACKING: {conf_pct}%"
            cv2.putText(hud_frame, conf_str, (bx1, max(18, by1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 242, 254), 1, cv2.LINE_AA)
        else:
            # No hand detected HUD hint
            cv2.putText(hud_frame, "NO HAND DETECTED - PLACE HAND IN VIEW", (w // 2 - 180, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 120, 200), 1, cv2.LINE_AA)

        # 5. Top Status Overlay Strip
        overlay = hud_frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (8, 12, 20), -1)
        cv2.addWeighted(overlay, 0.75, hud_frame, 0.25, 0, hud_frame)
        cv2.line(hud_frame, (0, 42), (w, 42), (0, 200, 255), 1)

        # Status text on the top strip
        status_text = "STATUS: ACTIVE" if is_active else "STATUS: PAUSED"
        status_color = (0, 230, 100) if is_active else (0, 165, 255)
        cv2.putText(hud_frame, status_text, (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

        # Gesture banner
        if action_name:
            banner_text = f"ACTION: {action_name}"
            banner_color = (0, 255, 255)
        else:
            banner_text = f"GESTURE: {current_gesture}"
            banner_color = (200, 200, 200)
        cv2.putText(hud_frame, banner_text, (w // 2 - 70, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, banner_color, 2, cv2.LINE_AA)

        return hud_frame

    def release(self):
        """Releases MediaPipe resources."""
        if hasattr(self, 'detector'):
            self.detector.close()

