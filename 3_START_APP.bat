@echo off
cd /d "%~dp0src"
echo ================================================
echo   VectorGuard AI
echo   Keep this window OPEN while using the app.
echo   Press Ctrl+C here to stop it.
echo ================================================
echo.
python -m streamlit run ui.py
pause
