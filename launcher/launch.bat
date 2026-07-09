@echo off
rem Double-click me: starts the launcher server and opens the dashboard.
start "" http://localhost:8090/
python "%~dp0server.py"
pause
