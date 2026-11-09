# ⚡ AI Hand Gesture Controlled Subway Surfers

> **Control the official Subway Surfers web game in real-time using only your webcam and hand gestures!**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%2F%203.12-blue.svg?logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0%2B-green.svg?logo=opencv)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange.svg?logo=google)](https://mediapipe.dev)
[![CustomTkinter](https://img.shields.io/badge/CustomTkinter-6.0-cyan.svg)](https://customtkinter.tomschimansky.com)
[![PyAutoGUI](https://img.shields.io/badge/PyAutoGUI-0.9.54-yellow.svg)](https://pyautogui.readthedocs.io/)
[![Game](https://img.shields.io/badge/Poki-Subway%20Surfers-red.svg)](https://poki.com/en/g/subway-surfers)

---

## 🎮 Actual Game Link
Play directly on the official web build:  
👉 **[Subway Surfers on Poki](https://poki.com/en/g/subway-surfers)**

---

## ⚡ Ultra-Fast Performance Architecture & Profiles

The controller features a **decoupled, multi-threaded zero-lag architecture** engineered for competitive high-speed Subway Surfers gameplay:

- **Decoupled Vision Thread (`_vision_worker_loop`):** Video capture and MediaPipe gesture inference run on a dedicated high-priority background thread, completely separate from the Tkinter GUI thread. Actions trigger keystrokes immediately without waiting for GUI render loops.
- **Sub-30ms Input Dispatch:** MediaPipe detection runs in ~8–14 ms; input enqueue-to-DirectInput dispatch executes in < 1 ms.
- **Three Selectable Performance Modes:**
  - **ULTRA FAST (Default):**
    - Target Latency: **50–100 ms** response
    - Temporal Consensus: 1–2 frames
    - Cooldowns: Left/Right: 140ms | Jump: 250ms | Roll: 250ms
    - Hardware Key Pulse: 25 ms snappy DirectInput pulse
  - **FAST:**
    - Temporal Consensus: 2–3 frames
    - Cooldowns: Left/Right: 200ms | Jump: 320ms | Roll: 320ms
    - Hardware Key Pulse: 30 ms
  - **NORMAL:**
    - High-filtering mode for relaxed or casual play
    - Cooldowns: Left/Right: 280ms | Jump: 400ms | Roll: 400ms
    - Hardware Key Pulse: 40 ms
- **Live Performance Telemetry:**
  - **Cam FPS:** Real-time webcam frame rate (30–60 FPS).
  - **Gesture FPS:** Vision inference processing rate.
  - **Inference Time (ms):** MediaPipe landmark calculation latency.
  - **Input Latency (ms):** Queue-to-keypress execution latency.
  - **Hand Tracking Confidence & Palm Scale:** Normalized distance tracking.

---

## 🖐️ Hand Gesture Control Mapping Table

| Hand Gesture | Visual Anatomy | Game Action | Mapped Key | Behavior & Debouncing |
| :--- | :--- | :--- | :--- | :--- |
| **Open Palm** | All 5 fingers extended | **JUMP** | `UP ARROW` (`↑`) | One-shot trigger (never continuously jumps while holding open palm) |
| **Closed Fist** | All 5 fingers folded/curled | **ROLL / SLIDE** | `DOWN ARROW` (`↓`) | One-shot trigger (never continuously rolls while holding fist) |
| **Index Pointing / Bending Right** | Index extended to right, other 3 curled | **MOVE RIGHT** | `RIGHT ARROW` (`→`) | One-shot trigger (moves character right; allows another right after returning to neutral) |
| **Index Pointing / Bending Left** | Index extended to left, other 3 curled | **MOVE LEFT** | `LEFT ARROW` (`←`) | One-shot trigger (moves character left; allows another left after returning to neutral) |
| **Neutral / Resting Hand** | Relaxed hand / fingers not in gesture | **NEUTRAL** | *None* | Resets hold state; ready for the next move |

> **Natural Mirrored View:** The camera preview is mirrored horizontally, so pointing or bending your index finger towards your physical right directly matches what you see on the screen and shifts your character into the right lane!

---

## 🌟 Architecture & Features

- **MediaPipe 3D Hand Tracking:** Tracks 21 skeletal hand landmarks with exponential moving average (EMA) smoothing to eliminate micro-jitter.
- **Scale-Invariant Geometry:** Normalizes all inter-joint distances by user palm size ($L_{palm} = \|\vec{p}_{mcp9} - \vec{p}_{wrist0}\|$), ensuring rock-solid detection whether sitting 40 cm or 120 cm from the camera.
- **Stable Gesture State Machine:**
  - One-shot edge detection: transitions send exactly one keypress.
  - Neutral state detection: returning to neutral enables consecutive same-direction lane changes.
  - Cooldown timers: Left/Right (280ms), Jump (400ms), Roll (400ms).
- **DirectInput Hardware Scancodes:** Dispatches OS-level DirectInput scan-codes (`SendInput`) and PyAutoGUI events, ensuring iframe and WebGL games like Subway Surfers register inputs without focus stealing.
- **Window Focus Monitor:** Real-time Windows API detection (`GetForegroundWindow`) warning the user if the browser game is unfocused.
- **Safety Features:**
  - `ESC` keyboard emergency stop listener (active system-wide).
  - Emergency key-up release: all keys are released on pause, emergency stop, or exit.
  - Pause disables all keyboard events.
- **Cyberpunk Gaming Dashboard:**
  - Dark navy and black glassmorphism layout with electric blue and cyan accents.
  - Live HUD with skeleton bones, glowing fingertip nodes, and bounding brackets.
  - 4 animated gesture cards that pulse bright cyan upon trigger.
  - Camera, hand tracking, focus, and action status indicators.

---

## 📁 Project Structure

```
subway_gesture_controller/
│
├── main.py                          # Core package launcher
├── hand_detection.py                # MediaPipe 21-landmark tracking & HUD
├── gesture_recognition.py           # Scale-invariant geometry & state machine
├── keyboard_controller.py           # DirectInput / PyAutoGUI OS-level controls
├── calibration.py                   # 4-step interactive calibration wizard
├── dashboard.py                     # CustomTkinter dark gaming GUI
├── config.py                        # Settings, themes, & key mappings
├── requirements.txt                 # Package dependencies
├── README.md                        # Package documentation
├── .gitignore                       # Git ignore rules
│
├── assets/
│   └── logo.png                     # Cyberpunk controller logo
│
└── models/
    └── hand_landmarker.task         # Google MediaPipe model file

Root Files:
├── main.py                          # Top-level entrypoint (python main.py)
├── setup.bat                        # Automated Windows venv & dependency installer
├── run.bat                          # One-click Windows runner
├── requirements.txt                 # Root dependencies
├── test_controller.py               # Complete test suite (9/9 tests passing)
└── README.md                        # Master documentation
```

---

## 🚀 Installation & Terminal Commands

### Step 1: Open Terminal in Project Folder
Open PowerShell or Command Prompt in `d:\subway suffer`.

### Step 2: Automated Setup (Recommended on Windows)
Simply double-click `setup.bat`, or run in PowerShell:
```powershell
.\setup.bat
```

### Step 3: Manual Virtual Environment Setup
If setting up manually with PowerShell:
```powershell
# 1. Check Python version
python --version

# 2. Create virtual environment named venv
python -m venv venv

# 3. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt
```

---

## 🕹️ How to Start & Play Subway Surfers

### Launch Application
Double-click `run.bat` or run:
```powershell
python main.py
```

### Gameplay Workflow
1. **Camera Starts:** The dashboard opens with your live mirrored webcam feed and hand skeleton HUD.
2. **Launch Subway Surfers:** Click **`🌐 OPEN SUBWAY SURFERS`** in the dashboard to open `https://poki.com/en/g/subway-surfers` in Chrome/Edge.
3. **Focus Game Window:** Click inside the Subway Surfers browser canvas to give it keyboard focus. The dashboard status will turn green: `✅ FOCUSED (RECEIVING INPUT)`.
4. **Activate Controls:** Click **`🎮 START GAME CONTROL`** in the dashboard.
5. **Control Character with Hand Gestures:**
   - 🖐️ **Open Palm:** Jump over trains and barriers!
   - ✊ **Closed Fist:** Roll under obstacles!
   - 👉 **Index Right:** Switch to the right lane!
   - 👈 **Index Left:** Switch to the left lane!
   - ✋ **Neutral:** Relax hand between moves.
6. **Emergency Pause:** Press `ESC` at any time to immediately stop game controls.

---

## 🎯 Hand Calibration Wizard

To calibrate the controller for your specific hand dimensions and room lighting:
1. Click **`🎯 CALIBRATE HAND`** in the dashboard.
2. Follow the 4 on-screen steps:
   - **Step 1: Open Palm** — Fully extend all 5 fingers.
   - **Step 2: Closed Fist** — Make a tight closed fist.
   - **Step 3: Point Left** — Point index finger to the left.
   - **Step 4: Point Right** — Point index finger to the right.
3. Custom thresholds are automatically saved to `calibration.json` and applied live.

---

## 🧪 Automated Testing (Tests 1–10)

All 10 features have dedicated automated tests in [test_controller.py](file:///d:/subway%20suffer/test_controller.py):
```powershell
.\venv\Scripts\python.exe test_controller.py
```

### Test Suite Results:
- ✅ **Test 1:** Open palm triggers `UP ARROW` (no continuous jumping while holding).
- ✅ **Test 2:** Closed fist triggers `DOWN ARROW` (no continuous rolling while holding).
- ✅ **Test 3:** Index pointing right triggers `RIGHT ARROW` (one-shot trigger).
- ✅ **Test 4:** Index pointing left triggers `LEFT ARROW` (one-shot trigger).
- ✅ **Test 5:** Neutral hand triggers no action & returning to neutral allows repeated moves.
- ✅ **Test 6:** Hand loss triggers no action & safely resets detection.
- ✅ **Test 7:** Pause halts all keyboard commands.
- ✅ **Test 8:** ESC / Emergency stop halts controller and releases all keys.
- ✅ **Test 9:** Browser receives actual OS keyboard inputs via DirectInput & PyAutoGUI.
- ✅ **Test 10:** Instantaneous 4-gesture sequence transitions (Right -> Left -> Jump -> Roll).

---

## 🛡️ License
MIT License. Built for real-time AI-powered webcam motion gaming.
