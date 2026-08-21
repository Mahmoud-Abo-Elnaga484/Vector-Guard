@echo off
cd /d "%~dp0src"
echo ================================================
echo   3-case evaluation
echo   Close the app window first (Ctrl+C) or this
echo   will fail: Qdrant allows one process at a time.
echo ================================================
echo.
python main.py demo
echo.
echo Report written to: src\evaluation\results\report.md
pause
