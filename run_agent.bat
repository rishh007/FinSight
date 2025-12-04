@echo off
echo Starting Finsight - Financial Analyst Agent...

:: Open website first in default browser
start "" "http://127.0.0.1:5500"

:: Start backend without blocking the batch script
start "" .\venv\Scripts\python.exe main_prv_backend.py


