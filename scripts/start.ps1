# Cardiac Abnormality Detection - Startup Script
# Run this script to start the web application

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Cardiac Abnormality Detection System" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-Not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
    Write-Host ""
}

# Check if packages are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$packages = .\venv\Scripts\python.exe -m pip list
if ($packages -notmatch "tensorflow") {
    Write-Host "Installing dependencies (this may take 5-10 minutes)..." -ForegroundColor Yellow
    .\venv\Scripts\python.exe -m pip install -r requirements.txt
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "✓ Dependencies already installed" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting web application..." -ForegroundColor Yellow
Write-Host "Navigate to: http://localhost:5000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

# Run the application
.\venv\Scripts\python.exe app/main.py
