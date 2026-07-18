@echo off
setlocal
set "APP_DIR=%~dp0"
set "PYTHON_EXE="

if exist "%APP_DIR%.venv_tabpfn\Scripts\python.exe" set "PYTHON_EXE=%APP_DIR%.venv_tabpfn\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%APP_DIR%.venv\Scripts\python.exe" set "PYTHON_EXE=%APP_DIR%.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "%APP_DIR%venv\Scripts\python.exe" set "PYTHON_EXE=%APP_DIR%venv\Scripts\python.exe"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

"%PYTHON_EXE%" "%APP_DIR%run_tabpfn_batch.py" %*
exit /b %ERRORLEVEL%
