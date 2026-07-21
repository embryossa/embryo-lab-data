@echo off
REM ================================================================
REM  Run the batch analysis, then push results to Teable.
REM  Point the two paths below at your project, then use this as
REM  your one-click "/run-analysis + upload" launcher.
REM ================================================================
setlocal
cd /d "%~dp0"

set "PY=python"
set "PROJECT=C:\Users\User\Desktop\IVF\AI\embryo-lab-data"
set "INPUT=%PROJECT%\data\raw\report.xlsx"
set "OUTDIR=%PROJECT%\data\processed"

echo [1/2] Running Digital Twin batch analysis...
"%PY%" "%PROJECT%\dt_batch.py" "%INPUT%" "%OUTDIR%"
if %ERRORLEVEL% NEQ 0 ( echo [ERROR] analysis failed & exit /b 1 )

echo [2/2] Syncing results to Teable (self-hosted, http://127.0.0.1:3000)...
"%PY%" "%~dp0teable_sync.py" --config "%~dp0teable_config.local.json"
if %ERRORLEVEL% NEQ 0 ( echo [ERROR] sync failed & exit /b 1 )

echo Done.
endlocal
