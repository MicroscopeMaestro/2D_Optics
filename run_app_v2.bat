@echo off
cd /d "%~dp0"
echo Launching 2D Optics - V2 Spectrometer Tuner...
python launch_v2.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with an error.
    pause
)
