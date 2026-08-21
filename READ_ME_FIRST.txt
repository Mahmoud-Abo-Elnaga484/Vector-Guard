MansouraHack - Evidence-Grounded Differential Diagnosis Assistant
=================================================================

This folder is complete and self-contained.
Do NOT copy it over any old folder. Put it anywhere and run it.

HOW TO RUN - double click these three, in order:

  1_SETUP.bat        installs packages, asks for your Gemini API key
  2_BUILD_INDEX.bat  indexes the WHO guideline (a few minutes, once only)
  3_START_APP.bat    opens the web interface

Extras (optional):

  4_RUN_EVALUATION.bat   3-case evaluation, writes report.md
  5_TEST_IN_CONSOLE.bat  the same 3 cases printed in the console

IMPORTANT
---------
Never run two of these at the same time.
Qdrant locks its files to a single process.
Close the running window with Ctrl+C before starting another.

Get a free Gemini API key at: https://aistudio.google.com/apikey
