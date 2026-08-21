@echo off
setlocal
cd /d "%~dp0"
echo ================================================
echo   STEP 1 of 3  -  SETUP
echo ================================================
echo.
echo Installing required packages. This takes a few minutes.
echo.
python -m pip install --quiet --upgrade pip
python -m pip install python-dotenv "openai>=1.40" langchain-core langchain-text-splitters langchain-huggingface sentence-transformers qdrant-client "pydantic>=2" streamlit
if errorlevel 1 goto failed
echo.
echo Packages installed.
echo.
if exist ".env" (
  echo A .env file already exists. Leaving it alone.
  goto done
)
echo ------------------------------------------------
echo   Paste your Google Gemini API key below.
echo   Get one free at: https://aistudio.google.com/apikey
echo ------------------------------------------------
set /p GKEY=API key: 
if "%GKEY%"=="" goto nokey
> .env echo LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
>> .env echo LLM_API_KEY=%GKEY%
>> .env echo GENERATION_MODEL=gemini-3.7-flash
>> .env echo JUDGE_MODEL=gemini-3.7-flash
>> .env echo LLM_JSON_MODE=true
>> .env echo LLM_MIN_INTERVAL=4
>> .env echo LLM_MAX_RETRIES=5
echo.
echo .env created.
goto done

:nokey
echo.
echo No key entered. Run this file again when you have one.
goto end

:failed
echo.
echo Package installation FAILED. Read the error above.
goto end

:done
echo.
echo ================================================
echo   SETUP COMPLETE.  Now run:  2_BUILD_INDEX.bat
echo ================================================
:end
echo.
pause
