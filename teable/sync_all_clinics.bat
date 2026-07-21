@echo off
chcp 65001 >nul
REM ================================================================
REM  One-click sync of existing processed results to Teable.
REM  Output goes to sync_log.txt; sync_done.flag marks completion.
REM  Writes to self-hosted Teable (Docker, http://127.0.0.1:3000) — the
REM  cloud free-plan space hit its 1000-row cap. Start the container first:
REM  cd self-hosted && docker compose up -d
REM  Two-Line  -> base bse4b0RSHeKI7ZGEdvC  (processed\)
REM  Fertimed  -> base bsef6WTgVoVeJeJD1N7  (processed\fertimed\)
REM ================================================================
setlocal
cd /d "%~dp0"
set "PY=python"
del /q sync_done.flag 2>nul

(
  echo ==== DEPENDENCIES ====
  "%PY%" -m pip install -r requirements.txt --disable-pip-version-check
  echo.
  echo ==== TWO-LINE ====
  "%PY%" teable_sync.py --config teable_config.local.json
  echo.
  echo ==== FERTIMED ====
  "%PY%" teable_sync.py --config teable_config.local.json --base bsef6WTgVoVeJeJD1N7 --clinic "Fertimed" --processed "C:\Users\User\Desktop\IVF\AI\embryo-lab-data\data\processed\fertimed"
  echo.
  echo ==== FINISHED ====
) > sync_log.txt 2>&1

echo done> sync_done.flag
endlocal
