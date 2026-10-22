@echo off
cd /d "%~dp0"
echo Starting 2D Optics CAD & Simulation Studio...
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
)
