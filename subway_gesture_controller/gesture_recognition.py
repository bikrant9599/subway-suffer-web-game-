"""
Gesture Recognition Module
Accurate geometric landmark calculations for Subway Surfers game actions:
- OPEN PALM -> JUMP
- CLOSED FIST -> ROLL
- INDEX RIGHT -> MOVE RIGHT
- INDEX LEFT -> MOVE LEFT
Includes debouncing, edge-triggered activation, and per-gesture cooldowns.
"""
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import numpy as np

from subway_gesture_controller.hand_detection import HandData


class GestureType:
    NONE = "WAITING"
    NEUTRAL = "NEUTRAL"
    JUMP = "JUMP"
    ROLL = "ROLL"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"


@dataclass
class GestureDetails:
    raw_gesture: str
    confirmed_gesture: str
    action_to_trigger: Optional[str]   # Non-None only on the one-shot transition frame
    finger_states: Dict[str, bool]     # True = extended, False = curled
    index_angle_deg: float
    hand_detected: bool
    confidence: float
    cooldown_remaining: Dict[str, float]


class GestureRecognizer:
    """Classifies hand gestures using anatomical geometry and state transitions."""

    def __init__(
        self,
        cooldown_left_right: float = 0.28,
        cooldown_jump: float = 0.40,
        cooldown_roll: float = 0.40,
        smoothing_frames: int = 4,
        min_trigger_consensus: int = 3
    ):
        self.cooldowns = {
            GestureType.MOVE_LEFT: cooldown_left_right,
            GestureType.MOVE_RIGHT: cooldown_left_right,
            GestureType.JUMP: cooldown_jump,
            GestureType.ROLL: cooldown_roll,
        }

        self.smoothing_frames = max(2, smoothing_frames)
        self.min_trigger_consensus = min(min_trigger_consensus, self.smoothing_frames)

        # History buffer for temporal smoothing
        self.gesture_history = deque(maxlen=self.smoothing_frames)

        # State tracking
        self.confirmed_gesture: str = GestureType.NONE
        self.currently_held_gesture: str = GestureType.NONE
        self.last_trigger_time: Dict[str, float] = {
            GestureType.MOVE_LEFT: 0.0,
            GestureType.MOVE_RIGHT: 0.0,
            GestureType.JUMP: 0.0,
            GestureType.ROLL: 0.0,
        }

        # Customizable calibration parameters
        self.calibration = {
            "palm_scale": 1.0,
            "curl_pip_ratio": 1.08,
            "curl_mcp_ratio": 0.60,
            "thumb_open_ratio": 0.40,
            "index_point_ratio": 0.25,
            "index_right_angle_min": -70.0,
            "index_right_angle_max": 70.0,
            "index_left_angle_threshold": 110.0
        }

    def update_cooldowns(self, lr: float, jump: float, roll: float):
        """Allows live adjustments from settings."""
        self.cooldowns[GestureType.MOVE_LEFT] = lr
        self.cooldowns[GestureType.MOVE_RIGHT] = lr
        self.cooldowns[GestureType.JUMP] = jump
        self.cooldowns[GestureType.ROLL] = roll

    def update_performance_mode(self, smoothing_frames: int, min_trigger_consensus: int):
        """Updates temporal consensus window for instant response."""
        self.smoothing_frames = max(1, smoothing_frames)
        self.min_trigger_consensus = max(1, min_trigger_consensus)
        self.gesture_history = deque(maxlen=self.smoothing_frames)

    def apply_calibration(self, calib_dict: dict):
        """Applies saved or freshly calibrated thresholds."""
        for k, v in calib_dict.items():
            if k in self.calibration:
                self.calibration[k] = v

    def _is_finger_extended(
        self,
        landmarks: np.ndarray,
        mcp_idx: int,
        pip_idx: int,
        dip_idx: int,
        tip_idx: int,
        palm_size: float
    ) -> Tuple[bool, float]:
        """
        Determines if a finger (index, middle, ring, pinky) is extended or curled.
        Uses joint angles at PIP and scale-invariant distances from MCP and wrist.
        """
        wrist = landmarks[0, :2]
        mcp = landmarks[mcp_idx, :2]
        pip = landmarks[pip_idx, :2]
        tip = landmarks[tip_idx, :2]

        d_tip_wrist = float(np.linalg.norm(tip - wrist))
        d_pip_wrist = float(np.linalg.norm(pip - wrist))
        d_tip_mcp = float(np.linalg.norm(tip - mcp))

        # Joint angle at PIP: vector (MCP->PIP) vs (PIP->TIP)
        v_pip_mcp = pip - mcp
        v_tip_pip = tip - pip
        denom = (np.linalg.norm(v_pip_mcp) * np.linalg.norm(v_tip_pip)) + 1e-6
        cos_angle = np.dot(v_pip_mcp, v_tip_pip) / denom
        flexion_angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))

        # Straight alignment
        is_straight = flexion_angle < 50.0
        # Distance checks
        is_long = d_tip_mcp > self.calibration["curl_mcp_ratio"] * palm_size
        is_farther = d_tip_wrist > d_pip_wrist * self.calibration["curl_pip_ratio"]

        # Curled indicators
        is_curled = (flexion_angle > 70.0) or (d_tip_mcp < 0.50 * palm_size) or (d_tip_wrist < d_pip_wrist and d_tip_mcp < 0.60 * palm_size)

        is_extended = bool(is_straight and (is_long or is_farther) and not is_curled)
        return is_extended, flexion_angle

    def _is_thumb_extended(self, landmarks: np.ndarray, palm_size: float) -> bool:
        """Determines if the thumb is abducted/extended or tucked/curled into a fist."""
        thumb_tip = landmarks[4, :2]
        thumb_mcp = landmarks[2, :2]
        wrist = landmarks[0, :2]
        index_mcp = landmarks[5, :2]

        d_tip_index_mcp = float(np.linalg.norm(thumb_tip - index_mcp))
        d_tip_wrist = float(np.linalg.norm(thumb_tip - wrist))

        open_ratio = self.calibration.get("thumb_open_ratio", 0.40)
        is_open = (d_tip_index_mcp > open_ratio * palm_size) and (d_tip_wrist > 0.55 * palm_size)
        return bool(is_open)

    def _classify_raw_gesture(self, hand_data: HandData) -> Tuple[str, Dict[str, bool], float]:
        """
        Classifies instantaneous frame landmarks into a raw gesture.
        Returns (gesture_name, finger_states, index_angle_deg).
        """
        lm = hand_data.landmarks
        palm_size = hand_data.palm_size

        # Analyze individual finger states
        thumb_ext = self._is_thumb_extended(lm, palm_size)
        idx_ext, idx_angle = self._is_finger_extended(lm, 5, 6, 7, 8, palm_size)
        mid_ext, _ = self._is_finger_extended(lm, 9, 10, 11, 12, palm_size)
        rng_ext, _ = self._is_finger_extended(lm, 13, 14, 15, 16, palm_size)
        pnk_ext, _ = self._is_finger_extended(lm, 17, 18, 19, 20, palm_size)

        finger_states = {
            "Thumb": thumb_ext,
            "Index": idx_ext,
            "Middle": mid_ext,
            "Ring": rng_ext,
            "Pinky": pnk_ext
        }

        # Index finger pointing vector and angle in 2D mirrored screen space
        idx_mcp = lm[5, :2]
        idx_pip = lm[6, :2]
        idx_tip = lm[8, :2]

        # Vector from MCP to TIP
        v_idx = idx_tip - idx_mcp
        dx = float(v_idx[0])
        dy = float(v_idx[1]) # Screen coords: Y increases downwards

        # Angle in degrees where 0° = pointing Right (+X), 90° = Up (-Y), 180°/-180° = Left (-X)
        screen_angle = math.degrees(math.atan2(-dy, dx))

        # Palm Vector (Wrist -> Middle MCP) for relative lateral bending
        wrist = lm[0, :2]
        mid_mcp = lm[9, :2]
        v_palm = mid_mcp - wrist
        # 2D cross product: (v_palm.x * v_idx.y - v_palm.y * v_idx.x)
        cross_z = float(v_palm[0] * v_idx[1] - v_palm[1] * v_idx[0])

        other_curled = (not mid_ext) and (not rng_ext) and (not pnk_ext)

        # 1. OPEN PALM (JUMP)
        # All 5 fingers extended (or all 4 long fingers extended and thumb open)
        if idx_ext and mid_ext and rng_ext and pnk_ext:
            if thumb_ext or (np.linalg.norm(lm[4, :2] - lm[5, :2]) > 0.35 * palm_size):
                return GestureType.JUMP, finger_states, screen_angle

        # 2. CLOSED FIST (ROLL)
        # All fingers folded / curled
        if not idx_ext and not mid_ext and not rng_ext and not pnk_ext:
            if not thumb_ext or (np.linalg.norm(lm[4, :2] - lm[5, :2]) < 0.45 * palm_size):
                return GestureType.ROLL, finger_states, screen_angle

        # 3. INDEX POINTING / BENDING RIGHT (MOVE RIGHT)
        # Index extended, middle, ring, pinky curled
        if idx_ext and other_curled:
            right_min = self.calibration["index_right_angle_min"]
            right_max = self.calibration["index_right_angle_max"]
            point_ratio = self.calibration["index_point_ratio"]

            is_angle_right = (right_min <= screen_angle <= right_max) and (dx > 0)
            is_dx_right = (dx > point_ratio * palm_size) and (idx_tip[0] > idx_pip[0])
            is_bend_right = (cross_z < -0.03) and (dx > 0)

            if is_angle_right or is_dx_right or is_bend_right:
                return GestureType.MOVE_RIGHT, finger_states, screen_angle

        # 4. INDEX POINTING / BENDING LEFT (MOVE LEFT)
        if idx_ext and other_curled:
            left_thresh = self.calibration["index_left_angle_threshold"]
            point_ratio = self.calibration["index_point_ratio"]

            is_angle_left = (abs(screen_angle) >= left_thresh) and (dx < 0)
            is_dx_left = (-dx > point_ratio * palm_size) and (idx_tip[0] < idx_pip[0])
            is_bend_left = (cross_z > 0.03) and (dx < 0)

            if is_angle_left or is_dx_left or is_bend_left:
                return GestureType.MOVE_LEFT, finger_states, screen_angle

        # Hand detected in resting / neutral state
        return GestureType.NEUTRAL, finger_states, screen_angle

    def process(self, hand_data: Optional[HandData]) -> GestureDetails:
        """
        Main gesture processing pipeline with temporal smoothing, edge-triggered
        activation, and cooldown management.
        """
        now = time.time()

        # Compute cooldown remaining for all gestures
        cooldown_rem: Dict[str, float] = {}
        for gest, cd_dur in self.cooldowns.items():
            elapsed = now - self.last_trigger_time[gest]
            cooldown_rem[gest] = max(0.0, cd_dur - elapsed)

        if hand_data is None:
            # Hand temporarily lost
            self.gesture_history.clear()
            self.confirmed_gesture = GestureType.NONE
            self.currently_held_gesture = GestureType.NONE

            return GestureDetails(
                raw_gesture=GestureType.NONE,
                confirmed_gesture=GestureType.NONE,
                action_to_trigger=None,
                finger_states={k: False for k in ["Thumb", "Index", "Middle", "Ring", "Pinky"]},
                index_angle_deg=0.0,
                hand_detected=False,
                confidence=0.0,
                cooldown_remaining=cooldown_rem
            )

        # 1. Classify raw instantaneous gesture
        raw_gesture, finger_states, index_angle = self._classify_raw_gesture(hand_data)
        self.gesture_history.append(raw_gesture)

        # 2. Temporal consensus smoothing (filtering transient noise)
        confirmed = self.confirmed_gesture
        active_gestures = [GestureType.JUMP, GestureType.ROLL, GestureType.MOVE_LEFT, GestureType.MOVE_RIGHT]

        if len(self.gesture_history) >= 1:
            counts: Dict[str, int] = {}
            for g in self.gesture_history:
                counts[g] = counts.get(g, 0) + 1

            for g, cnt in counts.items():
                if g in active_gestures and cnt >= self.min_trigger_consensus:
                    confirmed = g
                    break
            else:
                if counts.get(GestureType.NEUTRAL, 0) >= (self.smoothing_frames // 2):
                    confirmed = GestureType.NEUTRAL
                elif counts.get(GestureType.NONE, 0) >= (self.smoothing_frames // 2):
                    confirmed = GestureType.NONE
                else:
                    confirmed = raw_gesture

        self.confirmed_gesture = confirmed

        # 3. Edge-Trigger State Machine:
        # Trigger an action ONLY when newly entering an active gesture or transitioning!
        action_to_trigger: Optional[str] = None

        if confirmed in active_gestures and raw_gesture in active_gestures:
            # Check if this is a newly detected gesture transition
            if confirmed != self.currently_held_gesture:
                # Check cooldown
                if cooldown_rem.get(confirmed, 0.0) <= 0.0:
                    action_to_trigger = confirmed
                    self.last_trigger_time[confirmed] = now
                    # Update cooldown remaining
                    cooldown_rem[confirmed] = self.cooldowns.get(confirmed, 0.3)
                self.currently_held_gesture = confirmed
            else:
                # Still holding the same gesture -> DO NOT TRIGGER AGAIN!
                action_to_trigger = None
        else:
            # User relaxed hand / returned to neutral position
            self.currently_held_gesture = GestureType.NONE
            action_to_trigger = None

        return GestureDetails(
            raw_gesture=raw_gesture,
            confirmed_gesture=self.confirmed_gesture,
            action_to_trigger=action_to_trigger,
            finger_states=finger_states,
            index_angle_deg=index_angle,
            hand_detected=True,
            confidence=hand_data.confidence,
            cooldown_remaining=cooldown_rem
        )
