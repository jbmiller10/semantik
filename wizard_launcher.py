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


MIN_PYTHON_VERSION = (3, 11)


def check_python_version() -> None:
    """Ensure the current interpreter meets the minimum supported version."""
    if sys.version_info < MIN_PYTHON_VERSION:
        required = ".".join(str(part) for part in MIN_PYTHON_VERSION)
        detected = ".".join(str(part) for part in sys.version_info[:3])
        print(f"❌ Python {detected} detected. Semantik requires Python >= {required}.")
        sys.exit(1)

    required = ".".join(str(part) for part in MIN_PYTHON_VERSION)
    print(f"✅ Python {sys.version.split()[0]} detected (>= {required})")


def check_uv() -> bool:
    """Check if uv is installed, install if not"""
    uv_cmd = None

    # First check if uv is in PATH
    if shutil.which("uv"):
        uv_cmd = "uv"
    else:
        # Check platform-specific locations
        if sys.platform == "win32":
            # Windows: Check common installation locations
            local_app = Path(os.environ.get("LOCALAPPDATA", ""))
            possible_paths = [
                local_app / "uv" / "uv.exe",
                Path(os.environ.get("APPDATA", "")) / "Python" / "Scripts" / "uv.exe",
                Path.home() / ".local" / "bin" / "uv.exe",
            ]
        else:
            # macOS/Linux: Check home directory locations
            home = Path.home()
            possible_paths = [
                home / ".local" / "bin" / "uv",
                Path("/usr/local/bin/uv"),
            ]

        for path in possible_paths:
            if path and path.exists():
                uv_cmd = str(path)
                break

    if uv_cmd:
        print(f"✅ uv is installed at: {uv_cmd}")
        # Store for later use
        os.environ["UV_CMD"] = uv_cmd
        return True

    print("📦 uv not found. Would you like to install it? (recommended)")
    response = input("Install uv? [Y/n]: ").strip().lower()

    if response in ["", "y", "yes"]:
        print("Installing uv...")
        try:
            if sys.platform == "win32":
                cmd = [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    "iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex",
                ]
            else:
                cmd = ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ uv installed successfully")

                # Platform-specific instructions
                if sys.platform == "win32":
                    print("ℹ️  You may need to restart your terminal or add uv to PATH:")
                    print(f"   {Path(os.environ.get('LOCALAPPDATA', '')) / 'uv'}")
                    print(f"   {Path(os.environ.get('APPDATA', '')) / 'Python' / 'Scripts'}")
                else:
                    print("ℹ️  You may need to restart your terminal or run:")
                    print('   export PATH="$HOME/.local/bin:$PATH"')

                # Try to find Poetry again after installation
                return check_uv()
            print(f"❌ Failed to install uv: {result.stderr}")
            return False

        except Exception as e:
            print(f"❌ Error installing uv: {e}")
            return False
    else:
        print("❌ uv is required to run the setup wizard")
        print("Please install it manually: https://github.com/astral-sh/uv#installation")
        return False


def get_uv_cmd() -> str:
    """Get the uv command to use"""
    return os.environ.get("UV_CMD", "uv")


def check_dependencies() -> bool:
    """Check if required dependencies are installed"""
    try:
        # Try to import the required modules
        uv_cmd = get_uv_cmd()
        result = subprocess.run(
            [uv_cmd, "run", "python", "-c", "import questionary, rich"], capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def install_dependencies() -> bool:
    """Install required dependencies using uv"""
    print("📦 Installing dependencies...")
    try:
        uv_cmd = get_uv_cmd()
        result = subprocess.run([uv_cmd, "sync", "--frozen"], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        print(f"❌ Failed to install dependencies: {result.stderr}")
        return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def run_wizard() -> None:
    """Run the actual setup wizard"""
    print("\n🧙 Starting interactive setup wizard...\n")
    try:
        uv_cmd = get_uv_cmd()
        subprocess.run([uv_cmd, "run", "python", "docker_setup_tui.py"])
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error running wizard: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point"""
    print("🚀 Semantik Setup Wizard")
    print("========================\n")

    # Check we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: Please run this script from the Semantik root directory")
        sys.exit(1)

    # Check Python version
    check_python_version()

    # Check/install uv
    if not check_uv():
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
