#!/usr/bin/env python3
"""
Semantik Setup Wizard Launcher
This script can run without any dependencies and will bootstrap the setup process
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.12 or higher"""
    if sys.version_info < (3, 12):
        print(f"❌ Error: Python 3.12 or higher is required (found {sys.version})")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")


def check_poetry():
    """Check if Poetry is installed, install if not"""
    # Check common Poetry locations based on platform
    poetry_cmd = None

    # First check if poetry is in PATH
    if shutil.which("poetry"):
        poetry_cmd = "poetry"
    else:
        # Check platform-specific locations
        if sys.platform == "win32":
            # Windows: Check AppData locations
            appdata = os.environ.get("APPDATA", "")
            possible_paths = [
                Path(appdata) / "Python" / "Scripts" / "poetry.exe",
                Path(appdata) / "pypoetry" / "venv" / "Scripts" / "poetry.exe",
                Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Scripts" / "poetry.exe",
            ]
        else:
            # macOS/Linux: Check home directory locations
            home = Path.home()
            possible_paths = [
                home / ".local" / "bin" / "poetry",
                home / ".poetry" / "bin" / "poetry",
                "/usr/local/bin/poetry",
            ]

        for path in possible_paths:
            if Path(path).exists():
                poetry_cmd = str(path)
                break

    if poetry_cmd:
        print(f"✅ Poetry is installed at: {poetry_cmd}")
        # Store for later use
        os.environ["POETRY_CMD"] = poetry_cmd
        return True

    print("📦 Poetry not found. Would you like to install it? (recommended)")
    response = input("Install Poetry? [Y/n]: ").strip().lower()

    if response in ["", "y", "yes"]:
        print("Installing Poetry...")
        try:
            # Download and run Poetry installer
            import urllib.request

            installer_url = "https://install.python-poetry.org"
            with urllib.request.urlopen(installer_url) as response:
                installer_script = response.read().decode("utf-8")

            # Run the installer
            result = subprocess.run([sys.executable, "-c", installer_script], capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Poetry installed successfully")

                # Platform-specific instructions
                if sys.platform == "win32":
                    print("ℹ️  You may need to restart your terminal or add Poetry to PATH:")
                    print(f"   {os.environ.get('APPDATA')}\\Python\\Scripts")
                else:
                    print("ℹ️  You may need to restart your terminal or run:")
                    print('   export PATH="$HOME/.local/bin:$PATH"')

                # Try to find Poetry again after installation
                return check_poetry()
            print(f"❌ Failed to install Poetry: {result.stderr}")
            return False

        except Exception as e:
            print(f"❌ Error installing Poetry: {e}")
            return False
    else:
        print("❌ Poetry is required to run the setup wizard")
        print("Please install it manually: https://python-poetry.org/docs/#installation")
        return False


def get_poetry_cmd():
    """Get the Poetry command to use"""
    return os.environ.get("POETRY_CMD", "poetry")


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        # Try to import the required modules
        poetry_cmd = get_poetry_cmd()
        result = subprocess.run(
            [poetry_cmd, "run", "python", "-c", "import questionary, rich"], capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def install_dependencies():
    """Install required dependencies using Poetry"""
    print("📦 Installing dependencies...")
    try:
        poetry_cmd = get_poetry_cmd()
        result = subprocess.run([poetry_cmd, "install", "--no-interaction"], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        print(f"❌ Failed to install dependencies: {result.stderr}")
        return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def run_wizard():
    """Run the actual setup wizard"""
    print("\n🧙 Starting interactive setup wizard...\n")
    try:
        poetry_cmd = get_poetry_cmd()
        subprocess.run([poetry_cmd, "run", "python", "docker_setup_tui.py"])
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error running wizard: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    print("🚀 Semantik Setup Wizard")
    print("========================\n")

    # Check we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: Please run this script from the Semantik root directory")
        sys.exit(1)

    # Check Python version
    check_python_version()

    # Check/install Poetry
    if not check_poetry():
        sys.exit(1)

    # Check/install dependencies
    if not check_dependencies():
        if not install_dependencies():
            sys.exit(1)
    else:
        print("✅ Dependencies already installed")

    # Run the wizard
    run_wizard()


if __name__ == "__main__":
    main()
