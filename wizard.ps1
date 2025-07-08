# Semantik Setup Wizard Launcher for Windows PowerShell
# This script ensures dependencies are installed before running the wizard

Write-Host "🚀 Semantik Setup Wizard" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "pyproject.toml")) {
    Write-Host "❌ Error: Please run this script from the Semantik root directory" -ForegroundColor Red
    exit 1
}

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        Write-Host "✅ $pythonVersion detected" -ForegroundColor Green
        
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 12)) {
            Write-Host "❌ Error: Python 3.12 or higher is required" -ForegroundColor Red
            exit 1
        }
    }
} catch {
    Write-Host "❌ Error: Python 3 is required but not found" -ForegroundColor Red
    Write-Host "Please install Python 3.12 or higher from https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Check if Poetry is installed
$poetryPath = Get-Command poetry -ErrorAction SilentlyContinue
if (-not $poetryPath) {
    Write-Host "📦 Poetry not found. Installing Poetry..." -ForegroundColor Yellow
    
    try {
        (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
        
        # Add Poetry to PATH for this session
        $env:Path = "$env:APPDATA\Python\Scripts;$env:Path"
        
        # Verify installation
        $poetryPath = Get-Command poetry -ErrorAction SilentlyContinue
        if ($poetryPath) {
            Write-Host "✅ Poetry installed successfully" -ForegroundColor Green
        } else {
            throw "Poetry installation verification failed"
        }
    } catch {
        Write-Host "❌ Error: Failed to install Poetry" -ForegroundColor Red
        Write-Host "Please install Poetry manually: https://python-poetry.org/docs/#installation" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "✅ Poetry is installed" -ForegroundColor Green
}

# Check if dependencies are installed
Write-Host "📋 Checking dependencies..." -ForegroundColor Cyan
$depsCheck = poetry run python -c "import questionary, rich" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
    poetry install --no-interaction --no-ansi
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "✅ Dependencies already installed" -ForegroundColor Green
}

# Run the wizard
Write-Host ""
Write-Host "🧙 Starting interactive setup wizard..." -ForegroundColor Magenta
Write-Host ""
poetry run python docker_setup_tui.py