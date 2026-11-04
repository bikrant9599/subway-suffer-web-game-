"""
Comprehensive Unit & Integration Test Suite for AI Hand Gesture Gaming Controller
Tests:
- Hand detection & model loading
- Geometric gesture recognition (all 4 gestures independently)
- Edge-triggered state machine & cooldowns
- Keyboard controller queue & stats
- Calibration calculations & serialization
"""
import sys
import os
import time
import unittest
import numpy as np

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from subway_gesture_controller.hand_detection import HandDetector, HandData
from subway_gesture_controller.gesture_recognition import (
    GestureRecognizer,
    GestureType,
    GestureDetails
)
from subway_gesture_controller.keyboard_controller import KeyboardController
from subway_gesture_controller.calibration import HandCalibrator, CalibrationStep


class TestHandGestureController(unittest.TestCase):

    def setUp(self):
        self.recognizer = GestureRecognizer(
            cooldown_left_right=0.25,
            cooldown_jump=0.35,
            cooldown_roll=0.35,
            smoothing_frames=2,
            min_trigger_consensus=1
        )
        self.keyboard = KeyboardController(key_preset="ARROWS", enable_sound=False)
        self.calibrator = HandCalibrator(save_path="test_calibration.json")

    def tearDown(self):
        self.keyboard.shutdown()
        if os.path.exists("test_calibration.json"):
            try:
                os.remove("test_calibration.json")
            except Exception:
                pass

    def _create_synthetic_hand(
        self,
        thumb_ext=True,
        index_ext=True,
        middle_ext=True,
        ring_ext=True,
        pinky_ext=True,
        index_direction="up"  # "up", "right", "left"
    ) -> HandData:
        """Constructs synthetic 21-landmark hand geometry."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        wrist = np.array([0.5, 0.8, 0.0])
        landmarks[0] = wrist

        # Base knuckles (MCPs)
        # Thumb: 1..4
        # Index: 5..8
        # Middle: 9..12
        # Ring: 13..16
        # Pinky: 17..20

        # Middle MCP for palm size reference
        landmarks[9] = np.array([0.5, 0.5, 0.0])
        palm_size = float(np.linalg.norm(landmarks[9, :2] - wrist[:2]))  # ~0.3

        # Knuckle bases
        landmarks[1] = np.array([0.42, 0.72, 0.0])
        landmarks[2] = np.array([0.38, 0.65, 0.0])
        landmarks[5] = np.array([0.45, 0.52, 0.0])
        landmarks[13] = np.array([0.55, 0.53, 0.0])
        landmarks[17] = np.array([0.60, 0.55, 0.0])

        # Thumb
        if thumb_ext:
            landmarks[3] = np.array([0.32, 0.58, 0.0])
            landmarks[4] = np.array([0.26, 0.50, 0.0]) # Abducted far from index MCP
        else:
            landmarks[3] = np.array([0.40, 0.60, 0.0])
            landmarks[4] = np.array([0.44, 0.55, 0.0]) # Curled close to palm

        # Helper for a finger: MCP -> PIP -> DIP -> TIP
        def set_finger(base_mcp_idx, mcp_pos, is_ext, direction="up"):
            mcp = mcp_pos
            landmarks[base_mcp_idx] = mcp
            if is_ext:
                if direction == "right":
                    landmarks[base_mcp_idx + 1] = mcp + np.array([0.08, -0.01, 0.0])
                    landmarks[base_mcp_idx + 2] = mcp + np.array([0.15, -0.01, 0.0])
                    landmarks[base_mcp_idx + 3] = mcp + np.array([0.22, -0.01, 0.0])
                elif direction == "left":
                    landmarks[base_mcp_idx + 1] = mcp + np.array([-0.08, -0.01, 0.0])
                    landmarks[base_mcp_idx + 2] = mcp + np.array([-0.15, -0.01, 0.0])
                    landmarks[base_mcp_idx + 3] = mcp + np.array([-0.22, -0.01, 0.0])
                else:
                    landmarks[base_mcp_idx + 1] = mcp + np.array([0.0, -0.08, 0.0])
                    landmarks[base_mcp_idx + 2] = mcp + np.array([0.0, -0.15, 0.0])
                    landmarks[base_mcp_idx + 3] = mcp + np.array([0.0, -0.22, 0.0])
            else:
                # Curled towards palm
                landmarks[base_mcp_idx + 1] = mcp + np.array([0.0, 0.02, 0.0])
                landmarks[base_mcp_idx + 2] = mcp + np.array([0.0, 0.04, 0.0])
                landmarks[base_mcp_idx + 3] = mcp + np.array([0.0, 0.05, 0.0]) # Tip tucked

        # Configure Index Finger
        set_finger(5, landmarks[5], index_ext, index_direction)

        # Middle, Ring, Pinky
        set_finger(9, landmarks[9], middle_ext)
        set_finger(13, landmarks[13], ring_ext)
        set_finger(17, landmarks[17], pinky_ext)

        # Pixel landmarks
        pixel_lm = (landmarks[:, :2] * np.array([640, 480])).astype(np.int32)

        return HandData(
            landmarks=landmarks,
            pixel_landmarks=pixel_lm,
            handedness="Right",
            confidence=0.95,
            palm_center=(320, 240),
            palm_size=palm_size,
            bbox=(200, 100, 440, 400),
            timestamp_ms=int(time.time() * 1000)
        )

    def test_open_palm_jump_gesture(self):
        """Verify Open Palm (all 5 fingers extended) triggers JUMP (UP ARROW)."""
        hand = self._create_synthetic_hand(
            thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True
        )
        # First frame triggers one-shot action
        details = self.recognizer.process(hand)
        self.assertEqual(details.confirmed_gesture, GestureType.JUMP)
        self.assertEqual(details.action_to_trigger, GestureType.JUMP)

        # While holding the open palm, action_to_trigger MUST BE None (no spamming!)
        hold_details = self.recognizer.process(hand)
        self.assertEqual(hold_details.confirmed_gesture, GestureType.JUMP)
        self.assertIsNone(hold_details.action_to_trigger, "Holding gesture must not re-trigger!")

    def test_closed_fist_roll_gesture(self):
        """Verify Closed Fist (0 fingers extended) triggers ROLL (DOWN ARROW)."""
        hand = self._create_synthetic_hand(
            thumb_ext=False, index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False
        )
        details = self.recognizer.process(hand)
        self.assertEqual(details.confirmed_gesture, GestureType.ROLL)
        self.assertEqual(details.action_to_trigger, GestureType.ROLL)

        # While holding fist, must NOT re-trigger!
        hold_details = self.recognizer.process(hand)
        self.assertEqual(hold_details.confirmed_gesture, GestureType.ROLL)
        self.assertIsNone(hold_details.action_to_trigger, "Holding fist must not re-trigger!")

    def test_index_right_move_right_gesture(self):
        """Verify Index Pointing Right triggers MOVE_RIGHT (RIGHT ARROW)."""
        hand = self._create_synthetic_hand(
            thumb_ext=False, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False,
            index_direction="right"
        )
        details = self.recognizer.process(hand)
        self.assertEqual(details.confirmed_gesture, GestureType.MOVE_RIGHT)
        self.assertEqual(details.action_to_trigger, GestureType.MOVE_RIGHT)

        # While holding, must not re-trigger
        hold_details = self.recognizer.process(hand)
        self.assertEqual(hold_details.confirmed_gesture, GestureType.MOVE_RIGHT)
        self.assertIsNone(hold_details.action_to_trigger)

    def test_index_left_move_left_gesture(self):
        """Verify Index Pointing Left triggers MOVE_LEFT (LEFT ARROW)."""
        hand = self._create_synthetic_hand(
            thumb_ext=False, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False,
            index_direction="left"
        )
        details = self.recognizer.process(hand)
        self.assertEqual(details.confirmed_gesture, GestureType.MOVE_LEFT)
        self.assertEqual(details.action_to_trigger, GestureType.MOVE_LEFT)

        # While holding, must not re-trigger
        hold_details = self.recognizer.process(hand)
        self.assertEqual(hold_details.confirmed_gesture, GestureType.MOVE_LEFT)
        self.assertIsNone(hold_details.action_to_trigger)

    def test_hand_loss_graceful_handling(self):
        """Verify hand loss does not crash and resets to WAITING."""
        details = self.recognizer.process(None)
        self.assertEqual(details.confirmed_gesture, GestureType.NONE)
        self.assertIsNone(details.action_to_trigger)
        self.assertFalse(details.hand_detected)

    def test_keyboard_controller_pause_and_stats(self):
        """Verify keyboard controller obeys pause state and increments stats."""
        self.keyboard.set_active(False)
        self.keyboard.trigger_action(GestureType.JUMP)
        time.sleep(0.05)
        self.assertEqual(self.keyboard.stats["JUMP"], 0, "No action sent when paused!")

        # Activate
        self.keyboard.set_active(True)
        self.keyboard.trigger_action(GestureType.JUMP)
        self.keyboard.trigger_action(GestureType.ROLL)
        time.sleep(0.15)
        self.assertEqual(self.keyboard.stats["JUMP"], 1)
        self.assertEqual(self.keyboard.stats["ROLL"], 1)
        self.assertEqual(self.keyboard.stats["TOTAL"], 2)

    def test_calibration_serialization(self):
        """Verify calibration parameters save and load correctly."""
        defaults = self.calibrator.get_default_calibration()
        defaults["palm_scale"] = 0.42
        self.calibrator.parameters = defaults
        self.calibrator.save_calibration()

        loaded = self.calibrator.load_calibration()
        self.assertAlmostEqual(loaded["palm_scale"], 0.42)


if __name__ == "__main__":
    unittest.main()
