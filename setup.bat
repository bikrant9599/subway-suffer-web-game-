@echo off
title AI Hand Gesture Gaming Controller - Setup
echo ========================================================
echo Setting up AI Hand Gesture Gaming Controller Environment
echo ========================================================
if not exist venv (
    echo Creating virtual environment 'venv'...
    python -m venv venv
)
call .\venv\Scripts\activate.bat
echo Upgrading pip...
python -m pip install --upgrade pip
echo Installing required dependencies from requirements.txt...
pip install -r requirements.txt
echo ========================================================
echo Setup complete! Launch the controller with run.bat or:
echo   .\venv\Scripts\python.exe main.py
echo ========================================================
pause

