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
    echo Please install Python 3.12 or higher from https://www.python.org/downloads/
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python %PYTHON_VERSION% detected

REM Check if Poetry is installed
where poetry >nul 2>&1
if %errorlevel% neq 0 (
    echo Poetry not found. Installing Poetry...
    curl -sSL https://install.python-poetry.org | python -
    
    REM Add Poetry to PATH for this session
    set PATH=%APPDATA%\Python\Scripts;%PATH%
    
    REM Verify installation
    where poetry >nul 2>&1
    if %errorlevel% neq 0 (
        echo Error: Failed to install Poetry
        echo Please install Poetry manually: https://python-poetry.org/docs/#installation
        exit /b 1
    )
    echo Poetry installed successfully
)

REM Check if dependencies are installed
echo Checking dependencies...
poetry run python -c "import questionary, rich" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing dependencies...
    poetry install --no-interaction --no-ansi
    echo Dependencies installed
) else (
    echo Dependencies already installed
)

REM Run the wizard
echo.
echo Starting interactive setup wizard...
echo.
poetry run python docker_setup_tui.py