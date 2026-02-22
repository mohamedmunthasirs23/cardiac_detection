@echo off
echo ========================================
echo Cardiac Abnormality Detection System
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created!
    echo.
)

REM Install dependencies if needed
echo Checking dependencies...
venv\Scripts\python.exe -m pip list | findstr tensorflow >nul
if errorlevel 1 (
    echo Installing dependencies (this may take 5-10 minutes)...
    venv\Scripts\python.exe -m pip install -r requirements.txt
    echo Dependencies installed!
) else (
    echo Dependencies already installed!
)

echo.
echo Starting web application...
echo Navigate to: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the application
venv\Scripts\python.exe app/main.py
