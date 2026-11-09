"""
Keyboard Controller Module
Sends reliable hardware-level DirectInput scan-codes and PyAutoGUI keystrokes
to the Subway Surfers browser game. Handles asynchronous execution,
preset mapping, statistics, and pause states.
"""
import time
import threading
import queue
from typing import Dict, Optional, Callable
import pyautogui
import pydirectinput

# Try importing winsound for Windows audio feedback
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Disable PyAutoGUI pause/failsafe lag for instant gaming response
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = False
pydirectinput.PAUSE = 0.0
pydirectinput.FAILSAFE = False

# Try importing ctypes for Windows active window detection
try:
    import ctypes
    HAS_CTYPES = True
    user32 = ctypes.windll.user32
except Exception:
    HAS_CTYPES = False


def get_foreground_window_title() -> str:
    """Retrieves title of currently focused Windows application."""
    if not HAS_CTYPES:
        return ""
    try:
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return ""
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        return buff.value
    except Exception:
        return ""


def is_game_or_browser_focused() -> bool:
    """
    Checks if Subway Surfers or a web browser is currently the active foreground window.
    Detects Chrome, Edge, Firefox, Brave, Opera, and Poki Subway Surfers.
    """
    title = get_foreground_window_title().lower()
    keywords = ["subway surfers", "poki", "chrome", "edge", "firefox", "brave", "opera", "browser"]
    return any(k in title for k in keywords)


class KeyboardController:
    """Dispatches asynchronous keyboard inputs for game actions."""

    # Sound frequencies (Hz) for distinct action feedback
    SOUND_PITCHES = {
        "JUMP": (880, 50),       # High tone
        "ROLL": (350, 50),       # Low thud
        "MOVE_LEFT": (550, 40),  # Mid tone
        "MOVE_RIGHT": (700, 40), # Mid-high tone
    }

    def __init__(
        self,
        key_preset: str = "ARROWS",
        key_hold_time: float = 0.04,
        enable_sound: bool = True
    ):
        self.key_preset = key_preset.upper()
        self.key_hold_time = key_hold_time
        self.enable_sound = enable_sound

        self.is_active = False   # Master control toggle
        self._action_queue = queue.Queue()
        self._running = True

        # Key mapping definitions
        self.mappings = {
            "ARROWS": {
                "JUMP": "up",
                "ROLL": "down",
                "MOVE_LEFT": "left",
                "MOVE_RIGHT": "right"
            },
            "WASD": {
                "JUMP": "w",
                "ROLL": "s",
                "MOVE_LEFT": "a",
                "MOVE_RIGHT": "d"
            }
        }

        # Action statistics
        self.stats = {
            "JUMP": 0,
            "ROLL": 0,
            "MOVE_LEFT": 0,
            "MOVE_RIGHT": 0,
            "TOTAL": 0
        }
        self.last_latency_ms: float = 0.0

        # Listener callback for UI animation triggers
        self.on_action_triggered: Optional[Callable[[str, str], None]] = None

        # Dedicated sound worker queue and thread to eliminate per-keypress thread spawning latency
        self._sound_queue: queue.Queue = queue.Queue(maxsize=10)
        self._sound_thread = threading.Thread(target=self._sound_worker, daemon=True)
        self._sound_thread.start()

        # Start asynchronous worker thread
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()

    def release_all_keys(self):
        """Emergency key-up for all game control keys to prevent stuck inputs."""
        all_keys = ["up", "down", "left", "right", "w", "s", "a", "d"]
        for k in all_keys:
            try:
                pydirectinput.keyUp(k)
            except Exception:
                pass
            try:
                pyautogui.keyUp(k)
            except Exception:
                pass

    def set_active(self, active: bool):
        """Enables or disables sending keystrokes."""
        self.is_active = active
        if not active:
            self.release_all_keys()

    def set_key_preset(self, preset: str):
        """Switch between ARROWS and WASD."""
        if preset.upper() in self.mappings:
            self.key_preset = preset.upper()

    def get_current_mapping(self) -> Dict[str, str]:
        """Returns the active key mapping dictionary."""
        return self.mappings.get(self.key_preset, self.mappings["ARROWS"])

    def set_key_hold_time(self, hold_time: float):
        """Sets minimum keypress duration (e.g. 0.025s for Ultra Fast)."""
        self.key_hold_time = max(0.015, hold_time)

    def trigger_action(self, action: str):
        """Queues an action to be dispatched via keyboard with timing watermark."""
        if not self.is_active:
            return

        mapping = self.get_current_mapping()
        if action in mapping:
            key = mapping[action]
            enqueue_time = time.perf_counter()
            self._action_queue.put((action, key, enqueue_time))

    def _sound_worker(self):
        """Dedicated background thread handling audio beeps with zero main thread overhead."""
        while self._running:
            try:
                action = self._sound_queue.get(timeout=0.1)
                if action is None:
                    break
                self._play_sound(action)
            except queue.Empty:
                continue
            except Exception:
                pass

    def _play_sound(self, action: str):
        """Plays subtle sound effect asynchronously."""
        if not self.enable_sound or not HAS_WINSOUND:
            return
        pitch_info = self.SOUND_PITCHES.get(action)
        if pitch_info:
            freq, duration = pitch_info
            try:
                winsound.Beep(freq, duration)
            except Exception:
                pass

    def _process_queue(self):
        """Worker thread executing keyboard commands without blocking main thread."""
        while self._running:
            try:
                action, key, enqueue_time = self._action_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            if not self.is_active:
                continue

            # 1. SEND KEYSTROKE FIRST: Instantaneous OS hardware-level dispatch!
            self._send_keypress(key)
            self.last_latency_ms = (time.perf_counter() - enqueue_time) * 1000

            # 2. Update stats
            self.stats[action] = self.stats.get(action, 0) + 1
            self.stats["TOTAL"] += 1

            # 3. Audio feedback via non-blocking sound queue
            if self.enable_sound and self._sound_queue is not None:
                try:
                    self._sound_queue.put_nowait(action)
                except Exception:
                    pass

            # 4. Dispatch UI callback asynchronously
            if self.on_action_triggered:
                try:
                    self.on_action_triggered(action, key)
                except Exception as e:
                    print(f"[KeyboardController] UI Callback error: {e}")

    def _send_keypress(self, key: str):
        """
        Sends hardware-level keypress using DirectInput with PyAutoGUI fallback.
        Ensures cross-origin browser iframe canvas receives the keystroke.
        """
        try:
            # First try pydirectinput (DirectX / DirectInput scancodes)
            pydirectinput.keyDown(key)
            time.sleep(self.key_hold_time)
            pydirectinput.keyUp(key)
        except Exception:
            try:
                # Fallback to PyAutoGUI
                pyautogui.keyDown(key)
                time.sleep(self.key_hold_time)
                pyautogui.keyUp(key)
            except Exception as e:
                print(f"[KeyboardController] Keystroke error for '{key}': {e}")

    def reset_stats(self):
        """Resets action counters."""
        for k in self.stats:
            self.stats[k] = 0

    def shutdown(self):
        """Stops the worker threads and releases all keys."""
        self._running = False
        self.is_active = False
        if hasattr(self, "_sound_queue") and self._sound_queue is not None:
            try:
                self._sound_queue.put_nowait(None)
            except Exception:
                pass
        self.release_all_keys()

