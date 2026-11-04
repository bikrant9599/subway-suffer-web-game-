# Subway Gesture Controller Package

Modular core package for the AI Hand Gesture Gaming Controller.

## Modules:
- `main.py`: Entrypoint
- `hand_detection.py`: MediaPipe 21-landmark tracking and HUD rendering
- `gesture_recognition.py`: Scale-invariant geometric angle classification and edge-trigger debouncing
- `keyboard_controller.py`: Asynchronous DirectInput hardware scancode dispatcher
- `calibration.py`: Interactive 4-step hand calibration wizard
- `dashboard.py`: CustomTkinter dark cyberpunk GUI
- `config.py`: Persistent configuration and theme settings
