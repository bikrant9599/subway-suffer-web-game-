# ⚡ AI Hand Gesture Gaming Controller for Subway Surfers

> **Control the official Subway Surfers web game in real-time using only your webcam and hand gestures!**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.0%2B-green.svg?logo=opencv)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange.svg?logo=google)](https://mediapipe.dev)
[![CustomTkinter](https://img.shields.io/badge/CustomTkinter-6.0-cyan.svg)](https://customtkinter.tomschimansky.com)
[![Game](https://img.shields.io/badge/Poki-Subway%20Surfers-red.svg)](https://poki.com/en/g/subway-surfers)

---

## 🎮 Game Link
Play on the official web build:  
👉 **[Subway Surfers on Poki](https://poki.com/en/g/subway-surfers)**

---

## 🖐️ Gesture Control Mapping Table

| Hand Gesture | Visual Anatomy | Game Action | Mapped Key | Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Open Palm** | All 5 fingers extended | **JUMP** | `UP ARROW` (`↑`) | Edge-triggered once (no continuous jumping) |
| **Closed Fist** | All 5 fingers folded/curled | **ROLL / SLIDE** | `DOWN ARROW` (`↓`) | Edge-triggered once (no continuous rolling) |
| **Index Pointing/Bending Right** | Index extended to right, other 3 curled | **MOVE RIGHT** | `RIGHT ARROW` (`→`) | Edge-triggered once (moves to right lane) |
| **Index Pointing/Bending Left** | Index extended to left, other 3 curled | **MOVE LEFT** | `LEFT ARROW` (`←`) | Edge-triggered once (moves to left lane) |

> **Natural Mirrored View:** The webcam feed is mirrored horizontally, so pointing to your physical right directly matches what you see on screen and moves the character right!

---

## 🌟 Key Architecture & Highlights

- **3D Landmark Hand Detection:** Powered by Google MediaPipe Tasks with 21 tracked skeletal landmarks.
- **Scale-Invariant Geometry:** Distance calculations are normalized by palm length ($L_{palm} = \|\vec{p}_{mid\_mcp} - \vec{p}_{wrist}\|$), ensuring consistent gesture accuracy whether sitting close or far from the webcam.
- **Edge-Triggered State Machine:**
  - One-shot command dispatch: Entering a gesture sends exactly one keypress.
  - Holding an open palm or fist will **not** trigger unwanted repeated jumps or rolls.
  - Requires transitioning to neutral or another gesture before refiring.
- **DirectInput Hardware Scancodes:** Sends OS-level DirectInput scancodes (`SendInput`) via `pydirectinput` and `pyautogui`, guaranteeing keystrokes register inside browser WebGL/canvas iframes.
- **Premium Cyberpunk Dashboard:**
  - Glassmorphic dark navy and black interface with electric blue and cyan accents.
  - Live mirrored camera feed with neon skeletal bones and glowing joint nodes.
  - Four animated gesture cards that light up and pulse on keystroke dispatch.
  - Real-time tracking status, FPS counter, confidence gauge, and finger flexion indicators.
- **Interactive Calibration Wizard:** 4-step guided calibration measuring user palm dimensions, finger curl thresholds, and lateral angles, stored in `calibration.json`.
- **Global Emergency Stop:** Press `ESC` at any time (even when the browser game is focused) to immediately pause input.

---

## 📁 Project Structure

```
subway-suffer-web-game/
├── main.py                          # Root launcher
├── run.bat                          # Double-click Windows launcher
├── requirements.txt                 # Python dependencies
├── test_controller.py               # Unit and integration test suite
├── README.md                        # Documentation
├── subway_gesture_controller/
│   ├── __init__.py                  # Package init
│   ├── main.py                      # Main entrypoint
│   ├── hand_detection.py            # MediaPipe HandLandmarker & HUD
│   ├── gesture_recognition.py       # Geometric classification & state machine
│   ├── keyboard_controller.py       # Asynchronous DirectInput & PyAutoGUI
│   ├── calibration.py               # Step-by-step interactive calibrator
│   ├── dashboard.py                 # CustomTkinter dark gaming GUI
│   ├── config.py                    # Settings, themes, & key mappings
│   ├── requirements.txt             # Package requirements
│   ├── README.md                    # Subpackage docs
│   └── models/
│       └── hand_landmarker.task     # MediaPipe HandLandmarker model file
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.
- Laptop or USB webcam.

### 2. Clone the Repository
```bash
git clone https://github.com/bikrant9599/subway-suffer-web-game-.git
cd subway-suffer-web-game-
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🕹️ How to Run & Play

### Option A: Double-Click Launcher (Windows)
Double-click `run.bat` in the project root.

### Option B: Command Line
```bash
python main.py
```

### Step-by-Step Gameplay Workflow:
1. **Launch Dashboard:** Run `python main.py`. The webcam initializes and displays your hand skeleton.
2. **Launch Game:** Click the blue **`🌐 LAUNCH SUBWAY SURFERS`** button in the dashboard, or open [Poki Subway Surfers](https://poki.com/en/g/subway-surfers) in Chrome/Edge/Firefox.
3. **Give Game Focus:** Click on the Subway Surfers browser canvas so it receives keyboard focus.
4. **Start Motion Control:** In the dashboard, click **`🎮 START GAME CONTROL`**.
5. **Play the Game:**
   - 🖐️ **Open Palm:** Jump over trains and barriers!
   - ✊ **Closed Fist:** Roll under high obstacles!
   - 👉 **Index Right:** Switch to the right lane!
   - 👈 **Index Left:** Switch to the left lane!
6. **Emergency Pause:** Press `ESC` at any moment to pause the controller immediately.

---

## 🎯 Hand Calibration

To optimize detection for your hand shape and lighting:
1. Click **`🎯 CALIBRATE HAND`** in the dashboard.
2. Follow the 4 on-screen steps:
   - **Step 1:** Open palm fully with all 5 fingers spread.
   - **Step 2:** Form a tight closed fist.
   - **Step 3:** Point your index finger to the left.
   - **Step 4:** Point your index finger to the right.
3. Calibration is saved to `calibration.json` and applied automatically.

---

## 🧪 Running Automated Tests

Run the test suite to verify all gestures, state transitions, and keyboard queues:
```bash
python test_controller.py
```

---

## 🛡️ License
Released under the MIT License. Developed for seamless AI-powered motion gaming.
