@echo off
chcp 65001 >nul
REM ================================================================
REM  IVF Digital Twin -> Teable: все клиники, без нового анализа.
REM  Интерактивный вариант (с паузой в конце). Для расписания —
REM  sync_all_clinics.bat.
REM
REM  Клиники, их base_id и папки processed берутся ТОЛЬКО из
REM  teable_config.local.json (секция "clinics") — в этом файле
REM  ничего не зашито. Пишет в self-hosted Teable http://127.0.0.1:3000,
REM  карточки пациенток — http://127.0.0.1:8788.
REM
REM  Требуется рядом: teable_sync.py, teable_common.py, teable_schema.py,
REM  cards_build.py, teable_config.local.json (с локальным токеном),
REM  requirements.txt и поднятые контейнеры:
REM      cd self-hosted ^&^& docker compose up -d
REM  Просто дважды кликните по этому файлу.
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
echo ===================== СХЕМА БАЗ =====================
"%PY%" teable_bootstrap.py --all
if %ERRORLEVEL% NEQ 0 echo [!] Схема: ошибка (код %ERRORLEVEL%).

echo.
echo ===================== СИНХРОНИЗАЦИЯ =====================
"%PY%" teable_sync.py --all
if %ERRORLEVEL% NEQ 0 echo [!] Синхронизация завершилась с ошибкой (код %ERRORLEVEL%).

echo.
echo ===================== ГОТОВО =====================
echo Базы: http://127.0.0.1:3000    Карточки: http://127.0.0.1:8788
pause
endlocal
