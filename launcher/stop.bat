@echo off
rem Stop a launcher server that got orphaned (e.g. still serving :8090 after the
rem window was closed).  Kills whatever is listening on the launcher port.
setlocal
set PORT=8090
set FOUND=
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT%" ^| findstr LISTENING') do (
  echo Stopping launcher (PID %%p) on port %PORT% ...
  taskkill /F /PID %%p >nul 2>nul
  set FOUND=1
)
if not defined FOUND echo Nothing was listening on port %PORT%.
