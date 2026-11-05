"""
Calibration Module for AI Hand Gesture Controller
Interactive step-by-step calibration measuring user hand geometry,
finger flexibility, and index pointing angles to generate custom thresholds.
"""
import os
import json
import time
import math
from typing import Optional, Dict, List, Tuple, Callable
import numpy as np

from subway_gesture_controller.hand_detection import HandData
from subway_gesture_controller.config import CALIBRATION_FILE


class CalibrationStep:
    STEP_IDLE = 0
    STEP_OPEN_PALM = 1
    STEP_CLOSED_FIST = 2
    STEP_POINT_LEFT = 3
    STEP_POINT_RIGHT = 4
    STEP_FINISHED = 5


STEP_DESCRIPTIONS = {
    CalibrationStep.STEP_IDLE: (
        "READY TO CALIBRATE",
        "Click START CALIBRATION to calibrate hand geometry and angles."
    ),
    CalibrationStep.STEP_OPEN_PALM: (
        "STEP 1 / 4: OPEN PALM",
        "Place hand in camera view and FULLY EXTEND ALL 5 FINGERS."
    ),
    CalibrationStep.STEP_CLOSED_FIST: (
        "STEP 2 / 4: CLOSED FIST",
        "Make a tight CLOSED FIST with all fingers curled."
    ),
    CalibrationStep.STEP_POINT_LEFT: (
        "STEP 3 / 4: POINT LEFT",
        "Point your INDEX FINGER TO THE LEFT (curl other fingers)."
    ),
    CalibrationStep.STEP_POINT_RIGHT: (
        "STEP 4 / 4: POINT RIGHT",
        "Point your INDEX FINGER TO THE RIGHT (curl other fingers)."
    ),
    CalibrationStep.STEP_FINISHED: (
        "CALIBRATION COMPLETE!",
        "Saved custom parameters. You are ready to play Subway Surfers!"
    ),
}


class HandCalibrator:
    """Manages the interactive multi-step calibration routine."""

    SAMPLES_PER_STEP = 30  # Collect 30 stable frames per step

    def __init__(self, save_path: str = CALIBRATION_FILE):
        self.save_path = save_path
        self.current_step = CalibrationStep.STEP_IDLE
        self.samples_collected = 0
        self.step_start_time = 0.0

        # Accumulated sample buffers
        self._palm_samples: List[dict] = []
        self._fist_samples: List[dict] = []
        self._left_samples: List[dict] = []
        self._right_samples: List[dict] = []

        # Current calibration parameters
        self.parameters = self.load_calibration()

        # Step change callback for UI
        self.on_step_changed: Optional[Callable[[int, str, str, float], None]] = None

    def is_calibrating(self) -> bool:
        return self.current_step in [
            CalibrationStep.STEP_OPEN_PALM,
            CalibrationStep.STEP_CLOSED_FIST,
            CalibrationStep.STEP_POINT_LEFT,
            CalibrationStep.STEP_POINT_RIGHT
        ]

    def start_calibration(self):
        """Initiates the 4-step wizard."""
        self._palm_samples.clear()
        self._fist_samples.clear()
        self._left_samples.clear()
        self._right_samples.clear()

        self.current_step = CalibrationStep.STEP_OPEN_PALM
        self.samples_collected = 0
        self.step_start_time = time.time()
        self._notify_step()

    def cancel_calibration(self):
        """Aborts calibration."""
        self.current_step = CalibrationStep.STEP_IDLE
        self.samples_collected = 0
        self._notify_step()

    def _notify_step(self):
        if self.on_step_changed:
            title, desc = STEP_DESCRIPTIONS.get(self.current_step, ("", ""))
            progress = self.samples_collected / max(1, self.SAMPLES_PER_STEP)
            self.on_step_changed(self.current_step, title, desc, progress)

    def process_frame(self, hand_data: Optional[HandData]) -> Tuple[int, float]:
        """
        Gathers geometry samples during active calibration steps.
        Returns (current_step, step_progress [0.0 - 1.0]).
        """
        if not self.is_calibrating():
            return self.current_step, 0.0

        if hand_data is None:
            # Pause progress while hand is out of view
            progress = self.samples_collected / self.SAMPLES_PER_STEP
            return self.current_step, progress

        lm = hand_data.landmarks
        palm_size = hand_data.palm_size
        wrist = lm[0, :2]

        # Extract step measurements
        if self.current_step == CalibrationStep.STEP_OPEN_PALM:
            # Measure extended finger ratios
            idx_tip_dist = np.linalg.norm(lm[8, :2] - wrist)
            mid_tip_dist = np.linalg.norm(lm[12, :2] - wrist)
            thumb_open = np.linalg.norm(lm[4, :2] - lm[5, :2]) / palm_size

            self._palm_samples.append({
                "palm_size": palm_size,
                "idx_ratio": idx_tip_dist / (np.linalg.norm(lm[6, :2] - wrist) + 1e-5),
                "thumb_open": thumb_open
            })
            self.samples_collected += 1

            if self.samples_collected >= self.SAMPLES_PER_STEP:
                self.current_step = CalibrationStep.STEP_CLOSED_FIST
                self.samples_collected = 0
                self._notify_step()

        elif self.current_step == CalibrationStep.STEP_CLOSED_FIST:
            # Measure folded finger ratios
            idx_tip_dist = np.linalg.norm(lm[8, :2] - wrist)
            idx_pip_dist = np.linalg.norm(lm[6, :2] - wrist)
            thumb_fist = np.linalg.norm(lm[4, :2] - lm[5, :2]) / palm_size

            self._fist_samples.append({
                "idx_ratio": idx_tip_dist / (idx_pip_dist + 1e-5),
                "thumb_fist": thumb_fist
            })
            self.samples_collected += 1

            if self.samples_collected >= self.SAMPLES_PER_STEP:
                self.current_step = CalibrationStep.STEP_POINT_LEFT
                self.samples_collected = 0
                self._notify_step()

        elif self.current_step == CalibrationStep.STEP_POINT_LEFT:
            # Measure index pointing left
            idx_mcp = lm[5, :2]
            idx_tip = lm[8, :2]
            dx = float(idx_tip[0] - idx_mcp[0])
            dy = float(idx_tip[1] - idx_mcp[1])
            angle = math.degrees(math.atan2(-dy, dx))

            self._left_samples.append({
                "angle": angle,
                "dx_norm": dx / palm_size
            })
            self.samples_collected += 1

            if self.samples_collected >= self.SAMPLES_PER_STEP:
                self.current_step = CalibrationStep.STEP_POINT_RIGHT
                self.samples_collected = 0
                self._notify_step()

        elif self.current_step == CalibrationStep.STEP_POINT_RIGHT:
            # Measure index pointing right
            idx_mcp = lm[5, :2]
            idx_tip = lm[8, :2]
            dx = float(idx_tip[0] - idx_mcp[0])
            dy = float(idx_tip[1] - idx_mcp[1])
            angle = math.degrees(math.atan2(-dy, dx))

            self._right_samples.append({
                "angle": angle,
                "dx_norm": dx / palm_size
            })
            self.samples_collected += 1

            if self.samples_collected >= self.SAMPLES_PER_STEP:
                self._finalize_calibration()
                self.current_step = CalibrationStep.STEP_FINISHED
                self.samples_collected = self.SAMPLES_PER_STEP
                self._notify_step()

        progress = min(1.0, self.samples_collected / self.SAMPLES_PER_STEP)
        return self.current_step, progress

    def _finalize_calibration(self):
        """Computes optimal thresholds and saves to JSON."""
        calib = self.get_default_calibration()

        if self._palm_samples:
            palm_sizes = [s["palm_size"] for s in self._palm_samples]
            calib["palm_scale"] = float(np.mean(palm_sizes))

            open_ratios = [s["idx_ratio"] for s in self._palm_samples]
            open_thumb = [s["thumb_open"] for s in self._palm_samples]
            mean_open_ratio = float(np.mean(open_ratios))
            mean_thumb_open = float(np.mean(open_thumb))

            calib["curl_pip_ratio"] = max(1.08, min(1.25, mean_open_ratio * 0.88))
            calib["thumb_open_ratio"] = max(0.35, min(0.60, mean_thumb_open * 0.80))

        if self._fist_samples:
            fist_ratios = [s["idx_ratio"] for s in self._fist_samples]
            fist_thumb = [s["thumb_fist"] for s in self._fist_samples]
            # Threshold between fist and open
            mean_fist_ratio = float(np.mean(fist_ratios))
            calib["curl_mcp_ratio"] = max(0.70, min(0.95, mean_fist_ratio * 1.15))

        if self._left_samples:
            left_angles = [s["angle"] for s in self._left_samples]
            left_dx = [s["dx_norm"] for s in self._left_samples]
            mean_left_ang = float(np.mean([abs(a) for a in left_angles]))
            calib["index_left_angle_threshold"] = max(100.0, min(140.0, mean_left_ang - 15.0))

        if self._right_samples:
            right_angles = [s["angle"] for s in self._right_samples]
            mean_right_ang = float(np.mean(right_angles))
            calib["index_right_angle_min"] = max(-80.0, mean_right_ang - 45.0)
            calib["index_right_angle_max"] = min(80.0, mean_right_ang + 45.0)

        self.parameters = calib
        self.save_calibration()

    def get_default_calibration(self) -> dict:
        return {
            "palm_scale": 1.0,
            "curl_pip_ratio": 1.12,
            "curl_mcp_ratio": 0.85,
            "thumb_open_ratio": 0.45,
            "index_point_ratio": 0.35,
            "index_right_angle_min": -65.0,
            "index_right_angle_max": 65.0,
            "index_left_angle_threshold": 115.0
        }

    def save_calibration(self):
        try:
            with open(self.save_path, "w") as f:
                json.dump(self.parameters, f, indent=4)
            print(f"[HandCalibrator] Saved calibration parameters to {self.save_path}")
        except Exception as e:
            print(f"[HandCalibrator] Failed saving calibration: {e}")

    def load_calibration(self) -> dict:
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, "r") as f:
                    data = json.load(f)
                return data
            except Exception as e:
                print(f"[HandCalibrator] Error loading calibration file: {e}")
        return self.get_default_calibration()

