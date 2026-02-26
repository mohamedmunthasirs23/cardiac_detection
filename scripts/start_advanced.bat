@echo off
echo ========================================
echo   ADVANCED CARDIAC MONITORING SYSTEM
echo ========================================
echo.
echo Starting advanced version with:
echo   - Real-time WebSocket streaming
echo   - Explainable AI (Grad-CAM, SHAP)
echo   - PDF report generation
echo   - Multi-patient monitoring
echo.
echo ========================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Starting advanced server...
echo Navigate to: http://localhost:5000
echo.

python app\main_advanced.py

pause
