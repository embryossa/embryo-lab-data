@echo off
chcp 65001 >nul
REM ================================================================
REM  IVF Digital Twin -> Teable  (обе клиники, без нового анализа)
REM  Пишет в self-hosted Teable (Docker, http://127.0.0.1:3000) — облачный
REM  free-план исчерпан лимитом строк спейса, см. teable_config.json.
REM  Two-Line -> base bse4b0RSHeKI7ZGEdvC   (папка ...\processed)
REM  Fertimed -> base bsef6WTgVoVeJeJD1N7   (папка ...\processed\fertimed)
REM
REM  Требуется рядом: teable_sync.py, teable_config.local.json (с локальным
REM  токеном), requirements.txt, и запущенный контейнер self-hosted Teable
REM  (cd self-hosted && docker compose up -d). Просто дважды кликните по
REM  этому файлу.
REM ================================================================
setlocal
cd /d "%~dp0"
set "PY=python"

echo Проверка зависимостей Python...
"%PY%" -m pip install -r requirements.txt --quiet --disable-pip-version-check
if %ERRORLEVEL% NEQ 0 (
    echo [!] Не удалось установить зависимости. Проверьте, что Python в PATH.
    echo     Если у вас Anaconda, откройте Anaconda Prompt и запустите отсюда,
    echo     либо задайте полный путь к python.exe в переменной PY выше.
    pause
    exit /b 1
)

echo.
echo ===================== TWO-LINE (local Teable, base bse4b0RSHeKI7ZGEdvC) =====================
"%PY%" teable_sync.py --config teable_config.local.json
if %ERRORLEVEL% NEQ 0 echo [!] Two-Line: синхронизация завершилась с ошибкой (код %ERRORLEVEL%).

echo.
echo ===================== FERTIMED (local Teable, base bsef6WTgVoVeJeJD1N7) =====================
"%PY%" teable_sync.py --config teable_config.local.json --base bsef6WTgVoVeJeJD1N7 --clinic "Fertimed" --processed "C:\Users\User\Desktop\IVF\AI\embryo-lab-data\data\processed\fertimed"
if %ERRORLEVEL% NEQ 0 echo [!] Fertimed: синхронизация завершилась с ошибкой (код %ERRORLEVEL%).

echo.
echo ===================== ГОТОВО =====================
echo Проверьте базы в Teable. Окно можно закрыть.
pause
endlocal
