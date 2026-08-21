@echo off
cd /d "%~dp0src"
echo ######## CASE A - dengue-leaning presentation ########
python main.py query "Patient with abrupt high fever for 3 days, severe headache, retro-orbital pain and myalgia."
echo.
echo ######## CASE B - multi-section synthesis ########
python main.py query "Day 5 of fever, now afebrile but with abdominal pain, persistent vomiting and rising haematocrit."
echo.
echo ######## CASE C - must be refused ########
python main.py query "What is the first-line treatment for adult hypertension?"
echo.
pause
