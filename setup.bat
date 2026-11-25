@echo off
setlocal enabledelayedexpansion

echo === AutoMarkRemover Windows Setup ===
cd /d "%~dp0"

REM Find Python 3.11 or fallback to default py / python
set "PYTHON_EXE="
py -3.11 --version >nul 2>&1
if %errorlevel%==0 (
    set "PYTHON_EXE=py -3.11"
) else (
    py --version >nul 2>&1
    if %errorlevel%==0 (
        set "PYTHON_EXE=py"
    ) else (
        python --version >nul 2>&1
        if %errorlevel%==0 (
            set "PYTHON_EXE=python"
        )
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python 3.11+ not found. Please install Python 3.11 and re-run setup.bat.
    exit /b 1
)

echo Using Python: %PYTHON_EXE%
%PYTHON_EXE% --version

if not exist ".venv" (
    echo Creating virtual environment in .venv ...
    %PYTHON_EXE% -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create .venv
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip

REM Check for NVIDIA GPU to install CUDA PyTorch
nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo NVIDIA GPU detected. Installing CUDA-enabled PyTorch...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
) else (
    echo No NVIDIA GPU detected. Installing CPU PyTorch...
    pip install torch torchvision
)

echo Installing project requirements...
pip install -r requirements.txt

REM Check ffmpeg / ffprobe
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] ffmpeg not found in PATH. Checking WinGet packages...
    echo Please ensure ffmpeg and ffprobe are on PATH for video processing.
) else (
    echo ffmpeg found:
    where ffmpeg
)

REM Clone ProPainter if not present
if not exist "third_party\ProPainter" (
    where git >nul 2>&1
    if %errorlevel%==0 (
        echo Cloning ProPainter into third_party\ProPainter ...
        if not exist "third_party" mkdir "third_party"
        git clone --depth 1 https://github.com/sczhou/ProPainter.git "third_party\ProPainter"
    ) else (
        echo [WARNING] git not found. ProPainter will need to be placed in third_party\ProPainter manually.
    )
)

echo.
echo === Setup complete! Run run.bat to launch AutoMarkRemover. ===
endlocal
