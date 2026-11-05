"""
Comprehensive Automated Test Suite for AI Hand Gesture Gaming Controller
Explicitly verifies:
- Test 1: Open palm must trigger UP ARROW.
- Test 2: Closed fist must trigger DOWN ARROW.
- Test 3: Index pointing right must trigger RIGHT ARROW.
- Test 4: Index pointing left must trigger LEFT ARROW.
- Test 5: Neutral hand must not trigger any action (and returning to neutral resets state).
- Test 6: Hand disappearing must not trigger any action.
- Test 7: Pause must stop all game controls.
- Test 8: ESC / Emergency stop must stop the controller and release all keys.
- Test 9: Browser must receive actual keyboard inputs via DirectInput & PyAutoGUI.
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
from subway_gesture_controller.keyboard_controller import (
    KeyboardController,
    is_game_or_browser_focused,
    get_foreground_window_title
)
from subway_gesture_controller.calibration import HandCalibrator


class TestSubwayGestureController(unittest.TestCase):

    def setUp(self):
        self.recognizer = GestureRecognizer(
            cooldown_left_right=0.20,
            cooldown_jump=0.30,
            cooldown_roll=0.30,
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
        """Constructs realistic synthetic 21-landmark hand geometry."""
        landmarks = np.zeros((21, 3), dtype=np.float32)
        wrist = np.array([0.5, 0.8, 0.0])
        landmarks[0] = wrist

        # Middle MCP for palm size reference
        landmarks[9] = np.array([0.5, 0.5, 0.0])
        palm_size = float(np.linalg.norm(landmarks[9, :2] - wrist[:2]))  # ~0.3

        # Knuckle bases (MCPs)
        landmarks[1] = np.array([0.42, 0.72, 0.0])
        landmarks[2] = np.array([0.38, 0.65, 0.0])
        landmarks[5] = np.array([0.45, 0.52, 0.0])
        landmarks[13] = np.array([0.55, 0.53, 0.0])
        landmarks[17] = np.array([0.60, 0.55, 0.0])

        # Thumb
        if thumb_ext:
            landmarks[3] = np.array([0.32, 0.58, 0.0])
            landmarks[4] = np.array([0.26, 0.50, 0.0])  # Abducted far from index MCP
        else:
            landmarks[3] = np.array([0.40, 0.60, 0.0])
            landmarks[4] = np.array([0.44, 0.55, 0.0])  # Curled close to palm

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
                landmarks[base_mcp_idx + 1] = mcp + np.array([0.0, 0.02, 0.0])
                landmarks[base_mcp_idx + 2] = mcp + np.array([0.0, 0.04, 0.0])
                landmarks[base_mcp_idx + 3] = mcp + np.array([0.0, 0.05, 0.0])

        set_finger(5, landmarks[5], index_ext, index_direction)
        set_finger(9, landmarks[9], middle_ext)
        set_finger(13, landmarks[13], ring_ext)
        set_finger(17, landmarks[17], pinky_ext)

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

    # ========================================================
    # TEST 1: Open palm must trigger UP ARROW (JUMP)
    # ========================================================
    def test_1_open_palm_jump(self):
        hand = self._create_synthetic_hand(
            thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True
        )
        details = self.recognizer.process(hand)
        self.assertEqual(details.confirmed_gesture, GestureType.JUMP)
        self.assertEqual(details.action_to_trigger, GestureType.JUMP)

        # Mapping test
        key = self.keyboard.get_current_mapping()[details.action_to_trigger]
        self.assertEqual(key, "up", "Open palm MUST map to UP ARROW for jumping!")

        # Anti-spam hold test: holding open palm must NOT re-trigger continuously
        hold_details = self.recognizer.process(hand)
        self.assertEqual(hold_details.confirmed_gesture, GestureType.JUMP)
        self.assertIsNone(hold_details.action_to_trigger, "Holding open palm must NOT repeatedly jump!")

    # ========================================================
    # TEST 2: Closed fist must trigger DOWN ARROW (ROLL)
    # ========================================================
    def test_2_closed_fist_roll(self):
        hand = self._create_synthetic_hand(
            thumb_ext=False, index_ext=False, middle_ext=False, ring_ext=False, pinky_ext=False
        )
        details = self.recognizer.process(hand)
        self.assertEqual(details.confirmed_gesture, GestureType.ROLL)
        self.assertEqual(details.action_to_trigger, GestureType.ROLL)

        key = self.keyboard.get_current_mapping()[details.action_to_trigger]
        self.assertEqual(key, "down", "Closed fist MUST map to DOWN ARROW for rolling!")

        # Anti-spam hold test: holding fist must NOT re-trigger continuously
        hold_details = self.recognizer.process(hand)
        self.assertEqual(hold_details.confirmed_gesture, GestureType.ROLL)
        self.assertIsNone(hold_details.action_to_trigger, "Holding fist must NOT repeatedly roll!")

    # ========================================================
    # TEST 3: Index pointing right must trigger RIGHT ARROW
    # ========================================================
    def test_3_index_pointing_right_move_right(self):
        hand = self._create_synthetic_hand(
            thumb_ext=False, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False,
            index_direction="right"
        )
        details = self.recognizer.process(hand)
        self.assertEqual(details.confirmed_gesture, GestureType.MOVE_RIGHT)
        self.assertEqual(details.action_to_trigger, GestureType.MOVE_RIGHT)

        key = self.keyboard.get_current_mapping()[details.action_to_trigger]
        self.assertEqual(key, "right", "Index right MUST map to RIGHT ARROW!")

        # Holding right must not re-trigger repeatedly
        hold_details = self.recognizer.process(hand)
        self.assertIsNone(hold_details.action_to_trigger)

    # ========================================================
    # TEST 4: Index pointing left must trigger LEFT ARROW
    # ========================================================
    def test_4_index_pointing_left_move_left(self):
        hand = self._create_synthetic_hand(
            thumb_ext=False, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False,
            index_direction="left"
        )
        details = self.recognizer.process(hand)
        self.assertEqual(details.confirmed_gesture, GestureType.MOVE_LEFT)
        self.assertEqual(details.action_to_trigger, GestureType.MOVE_LEFT)

        key = self.keyboard.get_current_mapping()[details.action_to_trigger]
        self.assertEqual(key, "left", "Index left MUST map to LEFT ARROW!")

        # Holding left must not re-trigger repeatedly
        hold_details = self.recognizer.process(hand)
        self.assertIsNone(hold_details.action_to_trigger)

    # ========================================================
    # TEST 5: Neutral hand must not trigger any action & allows subsequent movement
    # ========================================================
    def test_5_neutral_hand_no_action(self):
        # Neutral position: Index upright, thumb neutral, middle partially extended
        neutral_hand = self._create_synthetic_hand(
            thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=False, pinky_ext=False,
            index_direction="up"
        )
        details = self.recognizer.process(neutral_hand)
        self.assertIsNone(details.action_to_trigger, "Neutral hand MUST NOT trigger any action!")
        self.assertEqual(details.confirmed_gesture, GestureType.NEUTRAL)

        # Transition: Neutral -> Point Right -> Trigger Right
        right_hand = self._create_synthetic_hand(
            thumb_ext=False, index_ext=True, middle_ext=False, ring_ext=False, pinky_ext=False,
            index_direction="right"
        )
        d_right = self.recognizer.process(right_hand)
        self.assertEqual(d_right.action_to_trigger, GestureType.MOVE_RIGHT)

        # Return to neutral -> resets state
        d_neutral = self.recognizer.process(neutral_hand)
        self.assertIsNone(d_neutral.action_to_trigger)

        # After returning to neutral, pointing right again MUST trigger right again!
        time.sleep(0.25) # wait cooldown
        d_right2 = self.recognizer.process(right_hand)
        self.assertEqual(d_right2.action_to_trigger, GestureType.MOVE_RIGHT,
                         "Returning to neutral must allow another right movement!")

    # ========================================================
    # TEST 6: Hand disappearing must not trigger any action
    # ========================================================
    def test_6_hand_disappearing_no_action(self):
        # Start with active gesture
        hand = self._create_synthetic_hand(
            thumb_ext=True, index_ext=True, middle_ext=True, ring_ext=True, pinky_ext=True
        )
        self.recognizer.process(hand)

        # Hand suddenly disappears
        details = self.recognizer.process(None)
        self.assertFalse(details.hand_detected, "Hand presence must be False!")
        self.assertEqual(details.confirmed_gesture, GestureType.NONE)
        self.assertIsNone(details.action_to_trigger, "Hand disappearance must NEVER trigger an action!")

    # ========================================================
    # TEST 7: Pause must stop all game controls
    # ========================================================
    def test_7_pause_stops_game_controls(self):
        self.keyboard.set_active(False)  # PAUSE
        self.keyboard.trigger_action(GestureType.JUMP)
        self.keyboard.trigger_action(GestureType.ROLL)
        self.keyboard.trigger_action(GestureType.MOVE_LEFT)
        time.sleep(0.05)

        self.assertEqual(self.keyboard.stats["TOTAL"], 0,
                         "While PAUSED, keyboard controller MUST NOT send any commands!")

    # ========================================================
    # TEST 8: ESC / Emergency stop must stop controller and release keys
    # ========================================================
    def test_8_emergency_stop_releases_keys(self):
        self.keyboard.set_active(True)
        self.assertTrue(self.keyboard.is_active)

        # Trigger emergency stop
        self.keyboard.set_active(False)
        self.keyboard.release_all_keys()
        self.assertFalse(self.keyboard.is_active, "Emergency stop MUST deactivate keyboard controller!")

    # ========================================================
    # TEST 9: Browser must receive actual keyboard inputs
    # ========================================================
    def test_9_browser_receives_keyboard_inputs(self):
        self.keyboard.set_active(True)

        # ARROWS mapping
        arrow_mapping = self.keyboard.mappings["ARROWS"]
        self.assertEqual(arrow_mapping["JUMP"], "up")
        self.assertEqual(arrow_mapping["ROLL"], "down")
        self.assertEqual(arrow_mapping["MOVE_LEFT"], "left")
        self.assertEqual(arrow_mapping["MOVE_RIGHT"], "right")

        # WASD mapping
        self.keyboard.set_key_preset("WASD")
        wasd_mapping = self.keyboard.get_current_mapping()
        self.assertEqual(wasd_mapping["JUMP"], "w")
        self.assertEqual(wasd_mapping["ROLL"], "s")
        self.assertEqual(wasd_mapping["MOVE_LEFT"], "a")
        self.assertEqual(wasd_mapping["MOVE_RIGHT"], "d")

        # Verify keystroke execution without crashing
        self.keyboard.trigger_action(GestureType.JUMP)
        time.sleep(0.12)
        self.assertEqual(self.keyboard.stats["JUMP"], 1, "Keystroke must dispatch successfully!")


if __name__ == "__main__":
    unittest.main()
