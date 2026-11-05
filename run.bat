@echo off
title AI Hand Gesture Gaming Controller - Subway Surfers
echo Starting AI Hand Gesture Gaming Controller for Subway Surfers...
if exist .\venv\Scripts\python.exe (
    .\venv\Scripts\python.exe main.py
) else (
    python main.py
)
pause
