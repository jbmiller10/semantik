#!/bin/bash
# Semantik Setup Wizard Launcher
# This script ensures dependencies are installed before running the wizard

set -e

echo "🚀 Semantik Setup Wizard"
echo "========================"
echo

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the Semantik root directory"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not found"
    echo "Please install Python 3.12 or higher"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.12"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "📦 Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add Poetry to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Verify installation
    if ! command -v poetry &> /dev/null; then
        echo "❌ Error: Failed to install Poetry"
        echo "Please install Poetry manually: https://python-poetry.org/docs/#installation"
        exit 1
    fi
    echo "✅ Poetry installed successfully"
fi

# Check if dependencies are installed
echo "📋 Checking dependencies..."
if ! poetry run python -c "import questionary, rich" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    poetry install --no-interaction --no-ansi
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Run the wizard
echo
echo "🧙 Starting interactive setup wizard..."
echo
poetry run python docker_setup_tui.py