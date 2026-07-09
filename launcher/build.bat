@echo off
rem Thin wrapper: build.py is the single source of truth (cross-platform Python).
python "%~dp0build.py" %*
if errorlevel 1 pause
