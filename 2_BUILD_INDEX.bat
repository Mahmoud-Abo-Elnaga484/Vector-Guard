@echo off
cd /d "%~dp0src"
echo ================================================
echo   STEP 2 of 3  -  BUILD THE SEARCH INDEX
echo ================================================
echo.
echo This downloads the embedding model the first time
echo and indexes the WHO guideline. Takes a few minutes.
echo Do NOT close this window until it says DONE.
echo.
python main.py clean
if errorlevel 1 goto failed
echo.
python main.py ingest
if errorlevel 1 goto failed
echo.
echo ================================================
echo   DONE.  Now run:  3_START_APP.bat
echo ================================================
goto end
:failed
echo.
echo FAILED. Read the error above.
:end
echo.
pause
