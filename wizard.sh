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
    echo "Please install Python 3.11 or higher"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"

    # Verify installation
    if ! command -v uv &> /dev/null; then
        echo "❌ Error: Failed to install uv"
        echo "Please install uv manually: https://github.com/astral-sh/uv#installation"
        exit 1
    fi
    echo "✅ uv installed successfully"
fi

# Check if dependencies are installed
echo "📋 Checking dependencies..."
if ! uv run python -c "import questionary, rich" 2>/dev/null; then
    echo "📦 Installing dependencies with uv..."
    uv sync --frozen
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Run the wizard
echo
echo "🧙 Starting interactive setup wizard..."
echo
uv run python docker_setup_tui.py
