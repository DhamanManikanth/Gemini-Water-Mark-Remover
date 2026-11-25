@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    py --version >nul 2>&1
    if %errorlevel%==0 (
        py app.py
    ) else (
        python app.py
    )
)
endlocal
