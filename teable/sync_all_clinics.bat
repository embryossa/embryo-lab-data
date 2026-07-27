@echo off
chcp 65001 >nul
REM ================================================================
REM  Синхронизация processed-файлов ВСЕХ клиник в self-hosted Teable
REM  + пересборка локального вьюера карточек пациенток.
REM
REM  Ничего не пересчитывает: берёт уже готовые data/processed/<клиника>.
REM  Список клиник и их base_id — в teable_config.local.json ("clinics"),
REM  здесь никаких id не зашито. Новая клиника: добавить её в конфиг и
REM  выполнить `python teable_bootstrap.py --all`.
REM
REM  Неинтерактивный вариант (для Планировщика заданий): вывод в
REM  sync_log.txt, sync_done.flag отмечает завершение.
REM  Контейнеры должны быть подняты: cd self-hosted ^&^& docker compose up -d
REM ================================================================
setlocal
cd /d "%~dp0"
set "PY=python"
del /q sync_done.flag 2>nul

(
  echo ==== DEPENDENCIES ====
  "%PY%" -m pip install -r requirements.txt --disable-pip-version-check
  echo.
  echo ==== SCHEMA ^(создать/дополнить базы всех клиник^) ====
  "%PY%" teable_bootstrap.py --all
  echo.
  echo ==== SYNC ^(все клиники + карточки^) ====
  "%PY%" teable_sync.py --all
  echo.
  echo ==== FINISHED ====
) > sync_log.txt 2>&1

echo done> sync_done.flag
endlocal
