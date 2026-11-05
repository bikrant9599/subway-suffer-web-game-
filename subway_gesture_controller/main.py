"""
AI Hand Gesture Gaming Controller - Main Entry Point
Play Subway Surfers with real-time hand gestures.
"""
import sys
import os

# Ensure package root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from subway_gesture_controller.dashboard import ControllerDashboard


def main():
    print("=" * 60)
    print(" ⚡ AI HAND GESTURE GAMING CONTROLLER FOR SUBWAY SURFERS ⚡")
    print("=" * 60)
    print(" [GESTURE MAPPINGS]")
    print("  • OPEN PALM (5 fingers)       -> JUMP (UP ARROW)")
    print("  • CLOSED FIST (0 fingers)     -> ROLL / SLIDE (DOWN ARROW)")
    print("  • INDEX POINTING RIGHT        -> MOVE RIGHT (RIGHT ARROW)")
    print("  • INDEX POINTING LEFT         -> MOVE LEFT (LEFT ARROW)")
    print(" [EMERGENCY STOP]")
    print("  • Press 'ESC' at any time to pause game control.")
    print("=" * 60)

    try:
        app = ControllerDashboard()
        app.mainloop()
    except KeyboardInterrupt:
        print("\n[Application] Stopped by user.")
    except Exception as e:
        print(f"\n[Application Error] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

