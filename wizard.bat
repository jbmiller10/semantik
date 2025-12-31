@echo off
REM Semantik Setup Wizard Launcher for Windows
REM This script ensures dependencies are installed before running the wizard

echo Semantik Setup Wizard
echo ========================
echo.

REM Check if we're in the right directory
if not exist "pyproject.toml" (
    echo Error: Please run this script from the Semantik root directory
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python 3 is required but not found
    echo Please install Python 3.11 or higher from https://www.python.org/downloads/
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python %PYTHON_VERSION% detected

REM Check if uv is installed
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo uv not found. Installing uv...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex"
    
    REM Add uv to PATH for this session
    set PATH=%USERPROFILE%\.local\bin;%PATH%
    set PATH=%USERPROFILE%\AppData\Local\uv;%PATH%
    
    REM Verify installation
    where uv >nul 2>&1
    if %errorlevel% neq 0 (
        echo Error: Failed to install uv
        echo Please install uv manually: https://github.com/astral-sh/uv#installation
        exit /b 1
    )
    echo uv installed successfully
)

REM Check if dependencies are installed
echo Checking dependencies...
uv run python -c "import questionary, rich" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    uv sync --frozen
    echo Dependencies installed
) else (
    echo Dependencies already installed
)

REM Run the wizard
echo.
echo Starting interactive setup wizard...
echo.
uv run python docker_setup_tui.py
