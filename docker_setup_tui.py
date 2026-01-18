#!/usr/bin/env python3
"""Interactive Docker Setup TUI for Semantik"""

import json
import os
import platform
import re
import secrets
import shutil
import string
import subprocess
import sys
import threading
import time
from pathlib import Path
from textwrap import dedent
from typing import Any
from urllib.parse import quote

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


FLOWER_PASSWORD_SYMBOLS = "!@#$%^*-_=+"
MIN_FLOWER_PASSWORD_LENGTH = 16


def generate_flower_credentials() -> tuple[str, str]:
    """Return a random Flower username/password pair meeting strength requirements."""

    username = f"flower_{secrets.token_hex(4)}"

    categories = (
        string.ascii_lowercase,
        string.ascii_uppercase,
        string.digits,
        FLOWER_PASSWORD_SYMBOLS,
    )

    password_chars = [secrets.choice(category) for category in categories]
    all_chars = "".join(categories)
    remaining = max(MIN_FLOWER_PASSWORD_LENGTH - len(password_chars), 0)
    password_chars.extend(secrets.choice(all_chars) for _ in range(remaining))
    secrets.SystemRandom().shuffle(password_chars)
    password = "".join(password_chars)

    return username, password


def mask_secret(value: str, visible: int = 4) -> str:
    """Return a masked representation of a secret, exposing only the last characters."""

    if not value:
        return "(unset)"
    if len(value) <= visible:
        return "*" * len(value)
    return "*" * (len(value) - visible) + value[-visible:]


class DockerSetupTUI:
    def __init__(self) -> None:
        self.config: dict[str, str] = {}
        self.gpu_available = False
        self.docker_available = False
        self.compose_available = False
        self.docker_gpu_available = False
        self.buildx_available = False
        self.driver_version: str | None = None
        self.is_wsl2 = False

    def run(self) -> None:
        """Main entry point for the TUI"""
        try:
            self.show_welcome()
            if not self.check_system():
                return

            # Check for existing configuration
            if self._check_existing_config():
                action = questionary.select(
                    "Existing configuration found. What would you like to do?",
                    choices=[
                        "Use existing configuration and manage services",
                        "Create new configuration (will backup existing)",
                        "Exit",
                    ],
                ).ask()

                if action is None or "Exit" in action:
                    return

                if "Use existing" in action:
                    self._load_existing_config()
                    self._service_monitor()
                    return

            # Run through setup steps
            if not self.select_deployment_type():
                return
            if not self.configure_database():
                return
            if not self.configure_directories():
                return
            if not self.configure_security():
                return
            if not self.review_configuration():
                return

            # Execute Docker setup
            self.execute_setup()

        except KeyboardInterrupt:
            console.print("\n[yellow]Setup cancelled by user[/yellow]")
            sys.exit(0)
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            sys.exit(1)

    def show_welcome(self) -> None:
        """Display welcome screen with ASCII art"""
        welcome_text = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███████╗███████╗███╗   ███╗ █████╗ ███╗   ██╗████████╗██╗██╗  ██╗
║   ██╔════╝██╔════╝████╗ ████║██╔══██╗████╗  ██║╚══██╔══╝██║██║ ██╔╝
║   ███████╗█████╗  ██╔████╔██║███████║██╔██╗ ██║   ██║   ██║█████╔╝
║   ╚════██║██╔══╝  ██║╚██╔╝██║██╔══██║██║╚██╗██║   ██║   ██║██╔═██╗
║   ███████║███████╗██║ ╚═╝ ██║██║  ██║██║ ╚████║   ██║   ██║██║  ██╗
║   ╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═╝
║                                                               ║
║              Docker Setup Wizard                              ║
╚═══════════════════════════════════════════════════════════════╝
        """
        console.print(Panel(welcome_text, style="bright_blue"))
        console.print("\n[bold]Welcome to the Semantik Docker Setup![/bold]")
        console.print("This wizard will help you configure and launch Semantik with Docker.\n")

    def check_system(self) -> bool:
        """Check system requirements"""
        console.print("[bold]Checking system requirements...[/bold]\n")

        # Check Docker
        self.docker_available = shutil.which("docker") is not None
        if self.docker_available:
            console.print("[green]✓[/green] Docker found")
        else:
            console.print("[red]✗[/red] Docker not found")
            console.print("\n[yellow]Docker is required to run Semantik.[/yellow]")
            console.print("\nTo install Docker:")
            if platform.system() == "Darwin":
                console.print("  → Download Docker Desktop: https://docker.com/get-docker")
                console.print("  → Or use Homebrew: brew install --cask docker")
            elif platform.system() == "Linux":
                console.print("  → Ubuntu/Debian: sudo apt-get install docker.io docker-compose")
                console.print("  → Fedora/RHEL: sudo dnf install docker docker-compose")
                console.print("  → Arch/Manjaro: sudo pacman -S docker docker-compose")
                console.print("\n  → Don't forget to add your user to the docker group:")
                console.print("    sudo usermod -aG docker $USER")
                console.print("    (Log out and back in for this to take effect)")
            else:
                console.print("  → Download Docker Desktop: https://docker.com/get-docker")
            return False

        # Check Docker Compose
        self.compose_available = self._check_docker_compose()
        if self.compose_available:
            console.print("[green]✓[/green] Docker Compose found")
        else:
            console.print("[red]✗[/red] Docker Compose not found")
            console.print("\n[yellow]Docker Compose is required.[/yellow]")
            console.print("Docker Desktop includes Docker Compose by default.")
            console.print("For manual installation: https://docs.docker.com/compose/install/")
            return False

        # Check Docker Buildx plugin (required for Bake-based builds)
        self.buildx_available = self._check_docker_buildx()
        if self.buildx_available:
            console.print("[green]✓[/green] Docker Buildx plugin found")
        else:
            console.print("[red]✗[/red] Docker Buildx plugin not found")
            console.print(
                "\n[yellow]Docker Buildx is required because Semantik's Docker Compose configuration uses Bake for builds.[/yellow]"
            )
            system_name = platform.system()
            if system_name == "Linux":
                console.print("Install the Buildx plugin using your package manager, for example:")
                console.print("  → Debian/Ubuntu: sudo apt-get install docker-buildx-plugin")
                console.print("  → Fedora/RHEL: sudo dnf install docker-buildx-plugin")
                console.print("  → Arch/Manjaro: sudo pacman -S docker-buildx")
            elif system_name == "Darwin":
                console.print(
                    "Update Docker Desktop from https://www.docker.com/products/docker-desktop/ (Buildx is included)."
                )
            else:
                console.print("Ensure Docker Desktop is up to date; Buildx ships with current releases.")
            console.print("\nAfter installing Buildx, re-run the wizard.")
            return False

        # Check GPU availability
        self.gpu_available = self._check_gpu()
        if self.gpu_available:
            gpu_msg = "[green]✓[/green] NVIDIA GPU detected on host system"
            if self.driver_version:
                gpu_msg += f" (Driver: {self.driver_version})"
                # Check driver compatibility
                try:
                    driver_major = int(self.driver_version.split(".")[0])
                    if driver_major >= 525:
                        gpu_msg += " [green]✓ CUDA 12.x compatible[/green]"
                    elif driver_major >= 470:
                        gpu_msg += " [yellow]⚠ CUDA 11.x compatible (older)[/yellow]"
                    else:
                        gpu_msg += " [red]⚠ Driver may be too old[/red]"
                except Exception:
                    pass
            console.print(gpu_msg)

            # Check Docker GPU runtime
            console.print("  [dim]Checking if Docker can access GPU...[/dim]")
            self.docker_gpu_available = self._check_docker_gpu_runtime()
            if self.docker_gpu_available:
                console.print("  [green]✓[/green] Docker GPU runtime ready (NVIDIA Container Toolkit installed)")
            else:
                console.print("  [yellow]![/yellow] Docker cannot access GPU (NVIDIA Container Toolkit not installed)")
                if self.is_wsl2:
                    console.print("  [dim]→ WSL2 detected - GPU support requires Windows NVIDIA drivers[/dim]")
                console.print("  [dim]→ The wizard will offer to install it if you choose GPU mode[/dim]")
                console.print("  [dim]→ This won't affect your host CUDA installation[/dim]")
        else:
            console.print("[yellow]![/yellow] No NVIDIA GPU detected (CPU mode will be used)")
            self.docker_gpu_available = False

        console.print()
        return True

    def _check_docker_compose(self) -> bool:
        """Check if docker compose is available"""
        try:
            # Try new syntax first
            result = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
            if result.returncode == 0:
                return True

            # Try old syntax
            return shutil.which("docker-compose") is not None
        except Exception:
            return False

    def _check_docker_buildx(self) -> bool:
        """Check if docker buildx plugin is available"""
        try:
            result = subprocess.run(["docker", "buildx", "version"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def _check_gpu(self) -> bool:
        """Check if NVIDIA GPU is available"""
        # Check if running in WSL2
        self.is_wsl2 = False
        try:
            with Path("/proc/version").open() as f:
                if "microsoft" in f.read().lower():
                    self.is_wsl2 = True
        except Exception:
            pass

        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                # Try to get driver version
                driver_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                )
                if driver_result.returncode == 0:
                    self.driver_version = driver_result.stdout.strip()
                else:
                    self.driver_version = None
                return True
            if self.is_wsl2 and Path("/dev/dxg").exists():
                # In WSL2, nvidia-smi might not work but GPU could still be available
                return True
            return False
        except Exception:
            return False

    def _check_docker_gpu_runtime(self) -> bool:
        """Check if Docker can use GPU runtime (NVIDIA Container Toolkit)"""
        if not self.docker_available:
            return False

        try:
            # Test if Docker can run with GPU support
            # First try a simple test to see if --gpus flag works
            result = subprocess.run(
                ["docker", "run", "--rm", "--gpus", "all", "ubuntu:22.04", "echo", "GPU test"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # First check if --gpus flag is recognized at all
            if "unknown flag: --gpus" in result.stderr.lower():
                return False

            # If basic test succeeded, Docker recognizes GPU flag
            if result.returncode == 0:
                # Now test with actual CUDA image to verify GPU access
                cuda_result = subprocess.run(
                    ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.8.0-base-ubuntu22.04", "nvidia-smi"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                # Check if nvidia-smi worked in CUDA container
                if cuda_result.returncode == 0 and "NVIDIA-SMI" in cuda_result.stdout:
                    return True

                # Check for nvidia-container-toolkit specific errors
                error_indicators = [
                    "could not select device driver",
                    "nvidia-container-cli",
                    "libnvidia-ml.so.1: cannot open shared object file",
                ]
                error_text = cuda_result.stderr.lower() if cuda_result.stderr else ""
                for indicator in error_indicators:
                    if indicator.lower() in error_text:
                        return False

            return False
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

    def _install_nvidia_container_toolkit(self) -> bool:
        """Install NVIDIA Container Toolkit based on the platform"""
        console.print("\n[bold yellow]NVIDIA GPU detected but Docker GPU support is not configured.[/bold yellow]")
        console.print("The NVIDIA Container Toolkit is required to use GPUs with Docker.\n")

        install = questionary.confirm("Would you like to install the NVIDIA Container Toolkit now?", default=True).ask()

        if not install:
            console.print("[yellow]Skipping NVIDIA Container Toolkit installation.[/yellow]")
            console.print("[yellow]GPU mode will not be available. Falling back to CPU mode.[/yellow]")
            return False

        console.print("\n[bold]Installing NVIDIA Container Toolkit...[/bold]\n")

        system = platform.system()

        if system == "Linux":
            # Detect Linux distribution
            distro_info = self._get_linux_distro()

            if distro_info and ("ubuntu" in distro_info.lower() or "debian" in distro_info.lower()):
                return self._install_nvidia_toolkit_debian()
            if distro_info and (
                "fedora" in distro_info.lower() or "rhel" in distro_info.lower() or "centos" in distro_info.lower()
            ):
                return self._install_nvidia_toolkit_rhel()
            if distro_info and ("arch" in distro_info.lower() or "manjaro" in distro_info.lower()):
                return self._install_nvidia_toolkit_arch()
                # Generic Linux instructions
                console.print("[yellow]Automatic installation not available for your Linux distribution.[/yellow]")
                console.print("\nPlease install manually by following:")
                console.print("https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html")
                return False

        elif system == "Darwin":
            console.print("[red]GPU support is not available on macOS.[/red]")
            console.print("Docker Desktop for Mac does not support GPU passthrough.")
            return False

        elif system == "Windows":
            console.print("[yellow]On Windows, GPU support requires WSL2.[/yellow]")
            console.print("\nPlease ensure:")
            console.print("1. You are using Docker Desktop with WSL2 backend")
            console.print("2. Install NVIDIA drivers in Windows (not WSL2)")
            console.print("3. Follow: https://docs.nvidia.com/cuda/wsl-user-guide/index.html")
            return False

        return False

    def _get_linux_distro(self) -> str | None:
        """Get Linux distribution information"""
        try:
            # Try lsb_release first
            result = subprocess.run(["lsb_release", "-si"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()

            # Try /etc/os-release
            if Path("/etc/os-release").exists():
                with Path("/etc/os-release").open() as f:
                    for line in f:
                        if line.startswith("ID="):
                            return line.split("=")[1].strip().strip('"')
        except Exception:
            pass

        return None

    def _install_nvidia_toolkit_debian(self) -> bool:
        """Install NVIDIA Container Toolkit on Debian/Ubuntu"""
        console.print("Running installation commands for Debian/Ubuntu...\n")

        try:
            # Step 1: Download and add GPG key
            console.print("[dim]$ Downloading NVIDIA GPG key...[/dim]")
            curl_gpg = subprocess.Popen(
                ["curl", "-fsSL", "https://nvidia.github.io/libnvidia-container/gpgkey"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            gpg_result = subprocess.run(
                ["sudo", "gpg", "--dearmor", "-o", "/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"],
                stdin=curl_gpg.stdout,
                capture_output=True,
                text=True,
            )
            curl_gpg.wait()

            if gpg_result.returncode != 0:
                console.print(f"[red]Error adding GPG key: {gpg_result.stderr}[/red]")
                return False
            console.print("[green]✓ GPG key added[/green]\n")

            # Step 2: Download and process repository list
            console.print("[dim]$ Setting up NVIDIA repository...[/dim]")
            curl_list = subprocess.run(
                [
                    "curl",
                    "-s",
                    "-L",
                    "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list",
                ],
                capture_output=True,
                text=True,
            )

            if curl_list.returncode != 0:
                console.print(f"[red]Error downloading repository list: {curl_list.stderr}[/red]")
                return False

            # Process the list with sed replacement
            processed_list = curl_list.stdout.replace(
                "deb https://", "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://"
            )

            # Write to the repository list file
            tee_result = subprocess.run(
                ["sudo", "tee", "/etc/apt/sources.list.d/nvidia-container-toolkit.list"],
                input=processed_list,
                capture_output=True,
                text=True,
            )

            if tee_result.returncode != 0:
                console.print(f"[red]Error writing repository list: {tee_result.stderr}[/red]")
                return False
            console.print("[green]✓ Repository configured[/green]\n")

        except Exception as e:
            console.print(f"[red]Error during repository setup: {e}[/red]")
            return False

        # These commands can be run normally
        regular_commands: list[list[str]] = [
            ["sudo", "apt-get", "update"],
            ["sudo", "apt-get", "install", "-y", "nvidia-container-toolkit"],
            ["sudo", "nvidia-ctk", "runtime", "configure", "--runtime=docker"],
            # Ensure Docker daemon reloads config
            ["sudo", "systemctl", "daemon-reload"],
            ["sudo", "systemctl", "restart", "docker"],
        ]

        # Execute regular commands
        for cmd_list in regular_commands:
            console.print(f"[dim]$ {' '.join(cmd_list)}[/dim]")
            result = subprocess.run(cmd_list, capture_output=True, text=True)

            if result.returncode != 0:
                console.print(f"[red]Error: {result.stderr}[/red]")
                return False
            console.print("[green]✓ Success[/green]\n")

        # Test if it works now, with retries
        console.print("[bold]Testing GPU support...[/bold]")
        console.print("[dim]Waiting for Docker daemon to fully restart...[/dim]")

        # Give Docker daemon time to restart
        time.sleep(5)

        # Check Docker daemon status first
        daemon_check = subprocess.run(["sudo", "systemctl", "is-active", "docker"], capture_output=True, text=True)
        if daemon_check.stdout.strip() != "active":
            console.print("[yellow]Docker service is not active. Attempting to start it...[/yellow]")
            subprocess.run(["sudo", "systemctl", "start", "docker"], capture_output=True)
            time.sleep(3)

        # Try up to 3 times with delays
        for attempt in range(3):
            if attempt > 0:
                console.print(f"[dim]Retry attempt {attempt + 1}/3...[/dim]")
                time.sleep(5)  # Increase delay between retries

            if self._check_docker_gpu_runtime():
                console.print("[green]✓ NVIDIA Container Toolkit installed successfully![/green]")
                return True

        # If still failing, provide detailed help
        console.print("[yellow]GPU support test failed after installation.[/yellow]")
        console.print("\nPossible solutions:")

        if self.is_wsl2:
            console.print("1. [bold]WSL2 detected![/bold] Run: [cyan]sudo bash fix_wsl2_gpu.sh[/cyan]")
            console.print("2. Ensure Windows has NVIDIA drivers installed (not WSL)")
            console.print("3. Restart WSL from Windows: [cyan]wsl --shutdown[/cyan]")
            console.print("4. Check WSL2 GPU support: [cyan]ls -la /dev/dxg[/cyan]")
        else:
            console.print("1. Run the fix script: [cyan]sudo bash fix_nvidia_toolkit.sh[/cyan]")
            console.print("2. Or manually run:")
            console.print("   [cyan]sudo nvidia-ctk runtime configure --runtime=docker[/cyan]")
            console.print("   [cyan]sudo systemctl daemon-reload[/cyan]")
            console.print("   [cyan]sudo systemctl restart docker[/cyan]")

        console.print("5. Check Docker logs: [cyan]sudo journalctl -u docker -n 50[/cyan]")
        console.print("6. As a last resort, reboot the system")
        console.print("\nYou can continue with CPU mode for now and enable GPU later.")
        console.print("To retry GPU setup later, run: [cyan]make wizard[/cyan]")
        return False

    def _install_nvidia_toolkit_rhel(self) -> bool:
        """Install NVIDIA Container Toolkit on RHEL/Fedora/CentOS"""
        console.print("Running installation commands for RHEL/Fedora/CentOS...\n")

        try:
            # Download repository configuration
            console.print("[dim]$ Setting up NVIDIA repository...[/dim]")
            curl_result = subprocess.run(
                [
                    "curl",
                    "-s",
                    "-L",
                    "https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo",
                ],
                capture_output=True,
                text=True,
            )

            if curl_result.returncode != 0:
                console.print(f"[red]Error downloading repository config: {curl_result.stderr}[/red]")
                return False

            # Write to repository file
            tee_result = subprocess.run(
                ["sudo", "tee", "/etc/yum.repos.d/nvidia-container-toolkit.repo"],
                input=curl_result.stdout,
                capture_output=True,
                text=True,
            )

            if tee_result.returncode != 0:
                console.print(f"[red]Error writing repository config: {tee_result.stderr}[/red]")
                return False

            console.print("[green]✓ Repository configured[/green]\n")

        except Exception as e:
            console.print(f"[red]Error during repository setup: {e}[/red]")
            return False

        # Regular commands
        regular_commands: list[list[str]] = [
            ["sudo", "yum", "install", "-y", "nvidia-container-toolkit"],
            ["sudo", "nvidia-ctk", "runtime", "configure", "--runtime=docker"],
            ["sudo", "systemctl", "daemon-reload"],
            ["sudo", "systemctl", "restart", "docker"],
        ]

        # Execute regular commands
        for cmd_list in regular_commands:
            console.print(f"[dim]$ {' '.join(cmd_list)}[/dim]")
            result = subprocess.run(cmd_list, capture_output=True, text=True)

            if result.returncode != 0:
                console.print(f"[red]Error: {result.stderr}[/red]")
                return False
            console.print("[green]✓ Success[/green]\n")

        # Test if it works now, with retries
        console.print("[bold]Testing GPU support...[/bold]")
        console.print("[dim]Waiting for Docker daemon to fully restart...[/dim]")

        # Give Docker daemon time to restart
        time.sleep(5)

        # Check Docker daemon status first
        daemon_check = subprocess.run(["sudo", "systemctl", "is-active", "docker"], capture_output=True, text=True)
        if daemon_check.stdout.strip() != "active":
            console.print("[yellow]Docker service is not active. Attempting to start it...[/yellow]")
            subprocess.run(["sudo", "systemctl", "start", "docker"], capture_output=True)
            time.sleep(3)

        # Try up to 3 times with delays
        for attempt in range(3):
            if attempt > 0:
                console.print(f"[dim]Retry attempt {attempt + 1}/3...[/dim]")
                time.sleep(5)  # Increase delay between retries

            if self._check_docker_gpu_runtime():
                console.print("[green]✓ NVIDIA Container Toolkit installed successfully![/green]")
                return True

        # If still failing, provide detailed help
        console.print("[yellow]GPU support test failed after installation.[/yellow]")
        console.print("\nPossible solutions:")

        if self.is_wsl2:
            console.print("1. [bold]WSL2 detected![/bold] Run: [cyan]sudo bash fix_wsl2_gpu.sh[/cyan]")
            console.print("2. Ensure Windows has NVIDIA drivers installed (not WSL)")
            console.print("3. Restart WSL from Windows: [cyan]wsl --shutdown[/cyan]")
            console.print("4. Check WSL2 GPU support: [cyan]ls -la /dev/dxg[/cyan]")
        else:
            console.print("1. Run the fix script: [cyan]sudo bash fix_nvidia_toolkit.sh[/cyan]")
            console.print("2. Or manually run:")
            console.print("   [cyan]sudo nvidia-ctk runtime configure --runtime=docker[/cyan]")
            console.print("   [cyan]sudo systemctl daemon-reload[/cyan]")
            console.print("   [cyan]sudo systemctl restart docker[/cyan]")

        console.print("5. Check Docker logs: [cyan]sudo journalctl -u docker -n 50[/cyan]")
        console.print("6. As a last resort, reboot the system")
        console.print("\nYou can continue with CPU mode for now and enable GPU later.")
        console.print("To retry GPU setup later, run: [cyan]make wizard[/cyan]")
        return False

    def _install_nvidia_toolkit_arch(self) -> bool:
        """Install NVIDIA Container Toolkit on Arch Linux"""
        console.print("Detected Arch Linux or derivative.\n")

        # Check for AUR helpers
        aur_helper = None
        for helper in ["yay", "paru", "trizen"]:
            if shutil.which(helper):
                aur_helper = helper
                break

        if aur_helper:
            # Use AUR helper
            console.print(f"Found AUR helper: {aur_helper}")
            console.print("Installing from AUR...\n")

            commands = [
                [aur_helper, "-S", "--noconfirm", "nvidia-container-toolkit"],
                ["sudo", "nvidia-ctk", "runtime", "configure", "--runtime=docker"],
                ["sudo", "systemctl", "restart", "docker"],
            ]

            for cmd in commands:
                console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    console.print(f"[red]Error: {result.stderr}[/red]")
                    # Try alternative method
                    console.print("\n[yellow]AUR installation failed. Trying official NVIDIA method...[/yellow]")
                    return self._install_nvidia_toolkit_arch_official()
                console.print("[green]✓ Success[/green]\n")
        else:
            # No AUR helper, use official method
            console.print("No AUR helper found. Using official NVIDIA repositories...\n")
            return self._install_nvidia_toolkit_arch_official()

        # Test if it works now, with retries
        console.print("[bold]Testing GPU support...[/bold]")
        console.print("[dim]Waiting for Docker daemon to fully restart...[/dim]")

        # Give Docker daemon time to restart
        time.sleep(5)

        # Check Docker daemon status first
        daemon_check = subprocess.run(["sudo", "systemctl", "is-active", "docker"], capture_output=True, text=True)
        if daemon_check.stdout.strip() != "active":
            console.print("[yellow]Docker service is not active. Attempting to start it...[/yellow]")
            subprocess.run(["sudo", "systemctl", "start", "docker"], capture_output=True)
            time.sleep(3)

        # Try up to 3 times with delays
        for attempt in range(3):
            if attempt > 0:
                console.print(f"[dim]Retry attempt {attempt + 1}/3...[/dim]")
                time.sleep(5)  # Increase delay between retries

            if self._check_docker_gpu_runtime():
                console.print("[green]✓ NVIDIA Container Toolkit installed successfully![/green]")
                return True

        # If still failing, provide detailed help
        console.print("[yellow]GPU support test failed after installation.[/yellow]")
        console.print("\nPossible solutions:")

        if self.is_wsl2:
            console.print("1. [bold]WSL2 detected![/bold] Run: [cyan]sudo bash fix_wsl2_gpu.sh[/cyan]")
            console.print("2. Ensure Windows has NVIDIA drivers installed (not WSL)")
            console.print("3. Restart WSL from Windows: [cyan]wsl --shutdown[/cyan]")
            console.print("4. Check WSL2 GPU support: [cyan]ls -la /dev/dxg[/cyan]")
        else:
            console.print("1. Run the fix script: [cyan]sudo bash fix_nvidia_toolkit.sh[/cyan]")
            console.print("2. Or manually run:")
            console.print("   [cyan]sudo nvidia-ctk runtime configure --runtime=docker[/cyan]")
            console.print("   [cyan]sudo systemctl daemon-reload[/cyan]")
            console.print("   [cyan]sudo systemctl restart docker[/cyan]")

        console.print("5. Check Docker logs: [cyan]sudo journalctl -u docker -n 50[/cyan]")
        console.print("6. As a last resort, reboot the system")
        console.print("\nYou can continue with CPU mode for now and enable GPU later.")
        console.print("To retry GPU setup later, run: [cyan]make wizard[/cyan]")
        return False

    def _install_nvidia_toolkit_arch_official(self) -> bool:
        """Install NVIDIA Container Toolkit on Arch using manual build"""
        console.print("[yellow]The NVIDIA Container Toolkit is not in the official Arch repositories.[/yellow]")
        console.print("\nTo install it, you have the following options:\n")

        console.print("[bold]Option 1: Install an AUR helper and use AUR (Recommended)[/bold]")
        console.print("$ sudo pacman -S --needed base-devel git")
        console.print("$ git clone https://aur.archlinux.org/yay.git")
        console.print("$ cd yay && makepkg -si")
        console.print("$ yay -S nvidia-container-toolkit")
        console.print()

        console.print("[bold]Option 2: Build from AUR manually[/bold]")
        console.print("$ git clone https://aur.archlinux.org/nvidia-container-toolkit.git")
        console.print("$ cd nvidia-container-toolkit")
        console.print("$ makepkg -si")
        console.print()

        console.print("[bold]Option 3: Use libnvidia-container from official repos[/bold]")
        console.print("$ sudo pacman -S libnvidia-container")
        console.print("(Note: This may not include all features)")
        console.print()

        console.print("After installation, run:")
        console.print("$ sudo nvidia-ctk runtime configure --runtime=docker")
        console.print("$ sudo systemctl restart docker")
        console.print()

        console.print("For more information, see:")
        console.print("https://wiki.archlinux.org/title/Docker#GPU_support")

        return False

    def _get_all_gpu_info(self) -> list[dict[str, Any]]:
        """Get detailed information about all GPUs using nvidia-smi"""
        try:
            # Query all relevant GPU information
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            gpus = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append(
                            {
                                "index": int(parts[0]),
                                "name": parts[1],
                                "memory_total_mb": int(parts[2]),
                            }
                        )

            return gpus
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

    def _show_progress(self, step: int, total: int, title: str) -> None:
        """Show progress indicator"""
        progress_bar = "━" * step + "○" * (total - step)
        console.print(f"\n[bold cyan]Step {step} of {total}: {title}[/bold cyan]")
        console.print(f"[dim]{progress_bar}[/dim]\n")

    def select_deployment_type(self) -> bool:
        """Select GPU or CPU deployment"""
        self._show_progress(1, 5, "Deployment Type")
        console.print("[bold]Select Deployment Type[/bold]\n")

        choices = []
        if self.gpu_available:
            choices.append("GPU - Faster processing with NVIDIA GPU")
            choices.append("CPU - Run without GPU (slower but works everywhere)")
        else:
            choices.append("CPU - Run without GPU")

        deployment = questionary.select("Choose deployment type:", choices=choices).ask()

        if deployment is None:
            return False

        if "GPU" in deployment:
            # Check if Docker GPU runtime is available
            if not self.docker_gpu_available:
                # Offer to install NVIDIA Container Toolkit
                if self._install_nvidia_container_toolkit():
                    self.docker_gpu_available = True
                    self.config["USE_GPU"] = "true"
                else:
                    # Fall back to CPU mode
                    console.print("\n[yellow]Falling back to CPU mode.[/yellow]")
                    self.config["USE_GPU"] = "false"
            else:
                self.config["USE_GPU"] = "true"
        else:
            self.config["USE_GPU"] = "false"

        console.print()
        return True

    def configure_database(self) -> bool:
        """Configure database settings"""
        self._show_progress(2, 5, "Database Configuration")
        console.print("[bold]Database Configuration[/bold]\n")
        console.print("Semantik uses PostgreSQL for storing user data, collections, and metadata.\n")

        # PostgreSQL password configuration
        postgres_choice = questionary.select(
            "PostgreSQL Password:",
            choices=["Generate secure password automatically (Recommended)", "Enter custom password"],
        ).ask()

        if postgres_choice is None:
            return False

        if "Generate" in postgres_choice:
            self.config["POSTGRES_PASSWORD"] = secrets.token_hex(32)
            console.print("[green]Generated secure PostgreSQL password[/green]")
        else:
            custom_password = questionary.password("Enter PostgreSQL password (min 16 chars):").ask()
            if custom_password is None or len(custom_password) < 16:
                console.print("[red]PostgreSQL password must be at least 16 characters[/red]")
                return False
            self.config["POSTGRES_PASSWORD"] = custom_password

        # Set default PostgreSQL values
        self.config["POSTGRES_DB"] = "semantik"
        self.config["POSTGRES_USER"] = "semantik"
        self.config["POSTGRES_HOST"] = "postgres"  # Docker service name
        self.config["POSTGRES_PORT"] = "5432"

        console.print()
        return True

    def configure_directories(self) -> bool:
        """Configure directory mappings"""
        self._show_progress(3, 5, "Directory Configuration")
        console.print("[bold]Configure Document Directories[/bold]\n")
        console.print("You can add multiple directories containing documents to process.")
        console.print("These directories will be mounted read-only in the Docker container.\n")

        document_dirs: list[Path] = []

        # First, ask if they want to use a file browser or type paths
        browse_mode = questionary.select(
            "How would you like to select directories?",
            choices=["Browse filesystem (recommended)", "Type paths manually"],
        ).ask()

        if browse_mode is None:
            return False

        use_browser = "Browse" in browse_mode

        # Add directories loop
        while True:
            if use_browser:
                # Interactive directory browser
                selected_dir = self._browse_for_directory()
                if selected_dir is None:
                    if not document_dirs:
                        # Must have at least one directory
                        console.print("[yellow]You must select at least one directory[/yellow]")
                        continue
                    break
            else:
                # Manual path entry
                if document_dirs:
                    console.print("\n[green]Selected directories:[/green]")
                    for d in document_dirs:
                        console.print(f"  • {d}")

                console.print("\n[dim]Examples: /home/user/documents, ./my-docs, ~/Downloads[/dim]")
                path_input = questionary.text(
                    "Enter directory path (or press Enter to finish):",
                    default="./documents" if not document_dirs else "",
                ).ask()

                if path_input is None:
                    return False

                if not path_input and document_dirs:
                    break
                if not path_input:
                    console.print("[yellow]You must add at least one directory[/yellow]")
                    continue

                selected_dir = Path(path_input).resolve()

            # Validate and add directory
            if selected_dir and selected_dir not in document_dirs:
                if not selected_dir.exists():
                    create = questionary.confirm(f"Directory {selected_dir} doesn't exist. Create it?").ask()
                    if create:
                        selected_dir.mkdir(parents=True, exist_ok=True)
                        console.print(f"[green]Created {selected_dir}[/green]")
                    else:
                        continue

                document_dirs.append(selected_dir)
                console.print(f"[green]Added: {selected_dir}[/green]")

                # Show directory contents preview
                self._preview_directory_contents(selected_dir)

            # Ask if they want to add more
            if not use_browser:
                add_more = questionary.confirm("Add another directory?", default=False).ask()
                if not add_more:
                    break

        # Store document paths
        self.config["DOCUMENT_PATHS"] = ":".join(str(d) for d in document_dirs)
        # For now, Docker compose only supports one path, so use the first one
        self.config["DOCUMENT_PATH"] = str(document_dirs[0])

        # Show final selection
        console.print("\n[bold green]Selected document directories:[/bold green]")
        for i, d in enumerate(document_dirs):
            if i == 0:
                console.print(f"  • {d} [cyan](primary)[/cyan]")
            else:
                console.print(f"  • {d}")

        if len(document_dirs) > 1:
            console.print("\n[yellow]Note: Docker compose currently only mounts the primary directory.[/yellow]")
            console.print("[yellow]Support for multiple directories is planned for a future update.[/yellow]")

        # Data directory (always use default)
        console.print("\n[bold]System Directories[/bold] (automatically configured)")
        data_path = Path("./data").resolve()
        data_path.mkdir(exist_ok=True)
        console.print(f"Data directory: {data_path}")

        # Logs directory (always use default)
        logs_path = Path("./logs").resolve()
        logs_path.mkdir(exist_ok=True)
        console.print(f"Logs directory: {logs_path}")

        console.print()
        return True

    def _browse_for_directory(self) -> Path | None:
        """Interactive directory browser"""
        current_path = Path.cwd()

        while True:
            # Show current location
            console.print(f"\n[bold cyan]Current location:[/bold cyan] {current_path}")

            # List directories
            try:
                dirs = sorted([d for d in current_path.iterdir() if d.is_dir() and not d.name.startswith(".")])

                choices = ["📁 [Select this directory]", "⬆️  [Go up one level]"]

                # Add subdirectories
                for d in dirs[:20]:  # Limit to 20 for readability
                    choices.append(f"📂 {d.name}")

                if len(dirs) > 20:
                    choices.append("... (more directories not shown)")

                # Add option to type path
                choices.append("✏️  [Type a path manually]")
                choices.append("❌ [Cancel]")

                selection = questionary.select("Navigate to directory:", choices=choices).ask()

                if selection is None or "Cancel" in selection:
                    return None

                if "Select this directory" in selection:
                    return current_path

                if "Go up" in selection:
                    current_path = current_path.parent
                    continue

                if "Type a path" in selection:
                    manual_path = questionary.text("Enter directory path:", default=str(current_path)).ask()
                    if manual_path:
                        typed_path = Path(manual_path).resolve()
                        if typed_path.exists() and typed_path.is_dir():
                            return typed_path
                        console.print("[red]Invalid directory path[/red]")
                    continue

                # Navigate to subdirectory
                dir_name = selection.replace("📂 ", "")
                new_path = current_path / dir_name
                if new_path.exists() and new_path.is_dir():
                    current_path = new_path

            except PermissionError:
                console.print("[red]Permission denied accessing this directory[/red]")
                current_path = current_path.parent

    def _preview_directory_contents(self, directory: Path) -> None:
        """Show a preview of files in the directory"""
        try:
            files = list(directory.iterdir())
            doc_files = [
                f
                for f in files
                if f.is_file()
                and f.suffix.lower() in [".pdf", ".docx", ".txt", ".md", ".html", ".pptx", ".csv", ".json"]
            ]

            if doc_files:
                console.print(f"  [dim]Found {len(doc_files)} document(s):[/dim]")
                for f in doc_files[:5]:  # Show first 5
                    console.print(f"    • {f.name}")
                if len(doc_files) > 5:
                    console.print(f"    ... and {len(doc_files) - 5} more")
            else:
                console.print("  [dim]No documents found in this directory[/dim]")
        except Exception:
            pass

    def configure_security(self) -> bool:
        """Configure security and advanced settings"""
        self._show_progress(4, 5, "Security & Advanced Settings")
        console.print("[bold]Security & Advanced Settings[/bold]\n")

        # GPU-specific settings first if GPU is selected
        if self.config["USE_GPU"] == "true":
            console.print("[bold]GPU Configuration[/bold]\n")

            # Get all available GPUs
            gpu_info = self._get_all_gpu_info()

            if gpu_info:
                # Build choices list
                choices = ["Auto (GPU 0)"]
                for gpu in gpu_info:
                    memory_gb = gpu["memory_total_mb"] / 1024
                    choices.append(f"GPU {gpu['index']}: {gpu['name']} ({memory_gb:.1f}GB)")

                gpu_choice = questionary.select("Select GPU device:", choices=choices).ask()

                if gpu_choice is None:
                    return False

                # Parse the selection
                if "Auto" in gpu_choice:
                    selected_gpu_idx = 0
                else:
                    # Extract GPU index from choice string
                    selected_gpu_idx = int(gpu_choice.split(":")[0].replace("GPU", "").strip())

                self.config["CUDA_VISIBLE_DEVICES"] = str(selected_gpu_idx)

                # Get the selected GPU's info for memory limit
                selected_gpu = next((g for g in gpu_info if g["index"] == selected_gpu_idx), gpu_info[0])
                max_memory_gb = selected_gpu["memory_total_mb"] / 1024

                # Memory limit - default to max available
                console.print(f"\n[dim]GPU has {max_memory_gb:.1f}GB total memory[/dim]")
                mem_limit = questionary.text(
                    "GPU memory limit in GB:",
                    default=f"{max_memory_gb:.0f}",
                    validate=lambda x: x.replace(".", "").isdigit() or "Please enter a number",
                ).ask()
                if mem_limit is None:
                    return False
                self.config["MODEL_MAX_MEMORY_GB"] = mem_limit
            else:
                # Fallback if GPU info cannot be retrieved
                console.print("[yellow]Could not detect GPU details. Using defaults.[/yellow]")
                self.config["CUDA_VISIBLE_DEVICES"] = "0"
                self.config["MODEL_MAX_MEMORY_GB"] = "8"

            console.print()

        # JWT Secret
        jwt_choice = questionary.select(
            "JWT Secret Key:", choices=["Generate secure key automatically (Recommended)", "Enter custom key"]
        ).ask()

        if jwt_choice is None:
            return False

        if "Generate" in jwt_choice:
            self.config["JWT_SECRET_KEY"] = secrets.token_hex(32)
            console.print("[green]Generated secure JWT key[/green]")
        else:
            custom_jwt = questionary.password("Enter JWT secret key (min 32 chars):").ask()
            if custom_jwt is None or len(custom_jwt) < 32:
                console.print("[red]JWT key must be at least 32 characters[/red]")
                return False
            self.config["JWT_SECRET_KEY"] = custom_jwt

        # Redis Password
        redis_choice = questionary.select(
            "Redis Password:", choices=["Generate secure password automatically (Recommended)", "Enter custom password"]
        ).ask()

        if redis_choice is None:
            return False

        if "Generate" in redis_choice:
            self.config["REDIS_PASSWORD"] = secrets.token_hex(32)
            console.print("[green]Generated secure Redis password[/green]")
        else:
            custom_redis = questionary.password("Enter Redis password (min 16 chars):").ask()
            if custom_redis is None or len(custom_redis) < 16:
                console.print("[red]Redis password must be at least 16 characters[/red]")
                return False
            self.config["REDIS_PASSWORD"] = custom_redis

        # Qdrant API Key
        qdrant_choice = questionary.select(
            "Qdrant API Key:", choices=["Generate secure key automatically (Recommended)", "Enter custom key"]
        ).ask()

        if qdrant_choice is None:
            return False

        if "Generate" in qdrant_choice:
            self.config["QDRANT_API_KEY"] = secrets.token_hex(32)
            console.print("[green]Generated secure Qdrant API key[/green]")
        else:
            custom_qdrant = questionary.password("Enter Qdrant API key (min 32 chars):").ask()
            if custom_qdrant is None or len(custom_qdrant) < 32:
                console.print("[red]Qdrant API key must be at least 32 characters[/red]")
                return False
            self.config["QDRANT_API_KEY"] = custom_qdrant

        # Connector Secrets Key (Fernet)
        connector_key_choice = questionary.select(
            "Connector Secrets Key (for encrypting passwords/tokens):",
            choices=[
                "Generate secure key automatically (Recommended)",
                "Enter custom Fernet key",
                "Skip (disable encryption)",
            ],
        ).ask()

        if connector_key_choice is None:
            return False

        if "Generate" in connector_key_choice:
            # Generate Fernet key (base64-encoded 32 bytes)
            import base64

            self.config["CONNECTOR_SECRETS_KEY"] = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
            console.print("[green]Generated secure connector secrets key[/green]")
        elif "Skip" in connector_key_choice:
            self.config["CONNECTOR_SECRETS_KEY"] = ""
            console.print(
                "[yellow]Connector secrets encryption disabled - credentials will not be stored securely[/yellow]"
            )
        else:
            custom_key = questionary.password("Enter Fernet key (44 chars, base64):").ask()
            if custom_key is None or len(custom_key) != 44:
                console.print("[red]Fernet key must be exactly 44 characters[/red]")
                return False
            self.config["CONNECTOR_SECRETS_KEY"] = custom_key

        # Access token expiration
        expiry = questionary.text("Access token expiration (minutes):", default="1440").ask()
        if expiry is None:
            return False
        self.config["ACCESS_TOKEN_EXPIRE_MINUTES"] = expiry

        # Log level
        log_level = questionary.select("Log level:", choices=["INFO", "DEBUG", "WARNING", "ERROR"]).ask()
        if log_level is None:
            return False
        self.config["LOG_LEVEL"] = log_level

        # Set hardcoded defaults
        self.config["DEFAULT_EMBEDDING_MODEL"] = "Qwen/Qwen3-Embedding-0.6B"
        self.config["DEFAULT_QUANTIZATION"] = "float16"
        self.config["ENABLE_LOCAL_LLM"] = "true"
        self.config["DEFAULT_LLM_QUANTIZATION"] = "int8"
        self.config["LLM_UNLOAD_AFTER_SECONDS"] = "300"
        self.config["LLM_KV_CACHE_BUFFER_MB"] = "1024"
        self.config["LLM_TRUST_REMOTE_CODE"] = "false"
        self.config["WEBUI_WORKERS"] = "auto"
        self.config["HF_CACHE_DIR"] = "./models"
        self.config["HF_HUB_OFFLINE"] = "false"
        self.config["ENVIRONMENT"] = "production"
        self.config["DEFAULT_COLLECTION"] = "work_docs"

        flower_username, flower_password = generate_flower_credentials()
        self.config["FLOWER_USERNAME"] = flower_username
        self.config["FLOWER_PASSWORD"] = flower_password

        console.print("[green]Generated Flower monitoring credentials[/green]")
        console.print("[dim]Credentials will be written to .env and masked in the summary below.[/dim]")

        console.print()
        return True

    def review_configuration(self) -> bool:
        """Review and confirm configuration"""
        self._show_progress(5, 5, "Review & Confirm")
        console.print("[bold]Review Configuration[/bold]\n")

        # Create review table
        table = Table(title="Semantik Docker Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        # Add configuration items
        table.add_row("Deployment Type", "GPU" if self.config["USE_GPU"] == "true" else "CPU")

        # Show document directories
        doc_dirs = self.config["DOCUMENT_PATHS"].split(":")
        if len(doc_dirs) == 1:
            table.add_row("Documents Directory", doc_dirs[0])
        else:
            table.add_row("Documents Directories", f"{len(doc_dirs)} directories selected")
            for i, d in enumerate(doc_dirs):
                table.add_row(f"  Directory {i+1}", d)

        if self.config["USE_GPU"] == "true":
            table.add_row("GPU Device", self.config["CUDA_VISIBLE_DEVICES"])
            table.add_row("GPU Memory Limit", f"{self.config['MODEL_MAX_MEMORY_GB']} GB")
        table.add_row(
            "Local LLM",
            "enabled" if self.config.get("ENABLE_LOCAL_LLM") == "true" else "disabled",
        )

        # Database settings
        table.add_row("Database", "PostgreSQL")
        table.add_row("PostgreSQL Password", mask_secret(self.config["POSTGRES_PASSWORD"]))

        # Infrastructure security
        table.add_row("Redis Password", mask_secret(self.config["REDIS_PASSWORD"]))
        table.add_row("Qdrant API Key", mask_secret(self.config["QDRANT_API_KEY"]))

        # Security settings
        table.add_row("JWT Secret", mask_secret(self.config["JWT_SECRET_KEY"]))
        connector_key = self.config.get("CONNECTOR_SECRETS_KEY", "")
        if connector_key:
            table.add_row("Connector Secrets Key", mask_secret(connector_key))
        else:
            table.add_row("Connector Secrets Key", "[dim]disabled[/dim]")
        table.add_row("Token Expiration", f"{self.config['ACCESS_TOKEN_EXPIRE_MINUTES']} minutes")
        table.add_row("Log Level", self.config["LOG_LEVEL"])
        table.add_row("WebUI Workers", "auto")
        table.add_row("Flower Username", mask_secret(self.config["FLOWER_USERNAME"]))
        table.add_row("Flower Password", mask_secret(self.config["FLOWER_PASSWORD"]))

        console.print(table)
        console.print()

        if self.config.get("ENABLE_LOCAL_LLM") == "true" and self.config["USE_GPU"] != "true":
            console.print("[bold yellow]Local LLM is enabled but GPU mode is off.[/bold yellow]")
            console.print("Local LLMs are GPU-intensive and may fail or be very slow without CUDA support.")
            console.print("Set ENABLE_LOCAL_LLM=false in .env if you don't have a compatible NVIDIA GPU.\n")

        # Confirm
        confirm = questionary.confirm("Proceed with this configuration?", default=True).ask()

        if not confirm:
            return False

        generate_env_test = questionary.confirm(
            "Generate .env.test for host-side integration tests?", default=True
        ).ask()

        if generate_env_test is None:
            return False

        env_test_db_name: str | None = None
        if generate_env_test:
            default_test_db = f"{self.config['POSTGRES_DB']}_test"
            env_test_db_name = questionary.text(
                "Enter PostgreSQL database name for host-side tests:",
                default=default_test_db,
            ).ask()

            if env_test_db_name is None:
                return False

            env_test_db_name = env_test_db_name.strip() or default_test_db

        # Save configuration
        env_test_written = self._save_env_file(
            generate_env_test=bool(generate_env_test),
            env_test_db_name=env_test_db_name,
        )
        self._save_config()  # Save to JSON for future use

        if env_test_written:
            console.print("[green]Configuration saved to .env, .env.test, and .semantik-config.json[/green]\n")
        else:
            console.print("[green]Configuration saved to .env and .semantik-config.json[/green]\n")
        return True

    def _save_env_file(self, *, generate_env_test: bool, env_test_db_name: str | None) -> bool:
        """Save configuration to .env file"""
        # Backup existing .env if present
        env_path = Path(".env")
        if env_path.exists():
            backup_path = env_path.with_suffix(".env.backup")
            shutil.copy(env_path, backup_path)
            console.print(f"[yellow]Backed up existing .env to {backup_path}[/yellow]")

        # Read template
        template_path = Path(".env.docker.example")
        if not template_path.exists():
            raise FileNotFoundError(".env.docker.example not found")

        with template_path.open() as f:
            content = f.read()

        # Replace values
        replacements = {
            "CHANGE_THIS_TO_A_STRONG_SECRET_KEY": self.config["JWT_SECRET_KEY"],
            "CHANGE_THIS_TO_A_FERNET_KEY": self.config.get("CONNECTOR_SECRETS_KEY", ""),
            "ACCESS_TOKEN_EXPIRE_MINUTES=1440": f"ACCESS_TOKEN_EXPIRE_MINUTES={self.config['ACCESS_TOKEN_EXPIRE_MINUTES']}",
            "DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B": f"DEFAULT_EMBEDDING_MODEL={self.config['DEFAULT_EMBEDDING_MODEL']}",
            "DEFAULT_QUANTIZATION=float16": f"DEFAULT_QUANTIZATION={self.config['DEFAULT_QUANTIZATION']}",
            "ENABLE_LOCAL_LLM=true": f"ENABLE_LOCAL_LLM={self.config['ENABLE_LOCAL_LLM']}",
            "DEFAULT_LLM_QUANTIZATION=int8": f"DEFAULT_LLM_QUANTIZATION={self.config['DEFAULT_LLM_QUANTIZATION']}",
            "LLM_UNLOAD_AFTER_SECONDS=300": f"LLM_UNLOAD_AFTER_SECONDS={self.config['LLM_UNLOAD_AFTER_SECONDS']}",
            "LLM_KV_CACHE_BUFFER_MB=1024": f"LLM_KV_CACHE_BUFFER_MB={self.config['LLM_KV_CACHE_BUFFER_MB']}",
            "LLM_TRUST_REMOTE_CODE=false": f"LLM_TRUST_REMOTE_CODE={self.config['LLM_TRUST_REMOTE_CODE']}",
            "DOCUMENT_PATH=./documents": f"DOCUMENT_PATH={self.config['DOCUMENT_PATH']}",
            "WEBUI_WORKERS=1": "WEBUI_WORKERS=auto",
            "POSTGRES_PASSWORD=CHANGE_THIS_TO_A_STRONG_PASSWORD": f"POSTGRES_PASSWORD={self.config['POSTGRES_PASSWORD']}",
            "REDIS_PASSWORD=CHANGE_THIS_TO_A_STRONG_PASSWORD": f"REDIS_PASSWORD={self.config['REDIS_PASSWORD']}",
            "QDRANT_API_KEY=CHANGE_THIS_TO_A_STRONG_API_KEY": f"QDRANT_API_KEY={self.config['QDRANT_API_KEY']}",
            "LOG_LEVEL=INFO": f"LOG_LEVEL={self.config['LOG_LEVEL']}",
            "HF_CACHE_DIR=./models": f"HF_CACHE_DIR={self.config['HF_CACHE_DIR']}",
            "HF_HUB_OFFLINE=false": f"HF_HUB_OFFLINE={self.config['HF_HUB_OFFLINE']}",
            "DEFAULT_COLLECTION=work_docs": f"DEFAULT_COLLECTION={self.config['DEFAULT_COLLECTION']}",
            "FLOWER_USERNAME=replace-me-with-flower-user": f"FLOWER_USERNAME={self.config['FLOWER_USERNAME']}",
            "FLOWER_PASSWORD=replace-me-with-strong-flower-password": f"FLOWER_PASSWORD={self.config['FLOWER_PASSWORD']}",
        }

        if self.config["USE_GPU"] == "true":
            replacements.update(
                {
                    "CUDA_VISIBLE_DEVICES=0": f"CUDA_VISIBLE_DEVICES={self.config['CUDA_VISIBLE_DEVICES']}",
                    "MODEL_MAX_MEMORY_GB=8": f"MODEL_MAX_MEMORY_GB={self.config['MODEL_MAX_MEMORY_GB']}",
                }
            )

        for old, new in replacements.items():
            content = content.replace(old, new)

        # Write .env
        with Path(".env").open("w") as f:
            f.write(content)

        env_test_written = False

        if generate_env_test and env_test_db_name:
            env_test_path = Path(".env.test")
            if env_test_path.exists():
                backup_path = Path(f"{env_test_path}.backup")
                shutil.copy(env_test_path, backup_path)
                console.print(f"[yellow]Backed up existing .env.test to {backup_path}[/yellow]")

            postgres_user = self.config["POSTGRES_USER"]
            postgres_password = self.config["POSTGRES_PASSWORD"]

            encoded_user = quote(postgres_user, safe="")
            encoded_password = quote(postgres_password, safe="")
            encoded_db = quote(env_test_db_name, safe="")

            env_test_content = dedent(
                f"""
                POSTGRES_HOST=localhost
                POSTGRES_PORT=5432
                POSTGRES_DB={env_test_db_name}
                POSTGRES_USER={postgres_user}
                POSTGRES_PASSWORD={postgres_password}
                DATABASE_URL=postgresql://{encoded_user}:{encoded_password}@localhost:5432/{encoded_db}
                """
            ).strip()

            with env_test_path.open("w") as f:
                f.write(env_test_content + "\n")

            env_test_written = True

        return env_test_written

    def execute_setup(self) -> None:
        """Execute Docker setup with selected options"""
        console.print("[bold]Docker Setup[/bold]\n")

        # Check ports availability
        if not self._check_ports():
            return

        # Choose action
        action = questionary.select(
            "What would you like to do?",
            choices=[
                "Build and start services (Recommended for first run)",
                "Start services only",
                "Build images only",
            ],
        ).ask()

        if action is None:
            return

        console.print()

        # Get compose files
        compose_files = self._get_compose_files()

        # Execute based on choice
        if "Build and start" in action:
            if self.config.get("USE_GPU") == "true":
                console.print("[bold yellow]GPU Build Notice:[/bold yellow]")
                console.print("• First-time GPU builds download NVIDIA CUDA base images (~3-5GB)")
                console.print("• This can take 10-30 minutes depending on your internet speed")
                console.print("• Subsequent builds will be much faster due to Docker layer caching")
                console.print()

            self._run_docker_command(["docker", "compose"] + compose_files + ["build"], "Building images")
            self._run_docker_command(["docker", "compose"] + compose_files + ["up", "-d"], "Starting services")
        elif "Start services" in action:
            self._run_docker_command(["docker", "compose"] + compose_files + ["up", "-d"], "Starting services")
        else:
            if self.config.get("USE_GPU") == "true":
                console.print("[bold yellow]GPU Build Notice:[/bold yellow]")
                console.print("• First-time GPU builds download NVIDIA CUDA base images (~3-5GB)")
                console.print("• This can take 10-30 minutes depending on your internet speed")
                console.print("• Subsequent builds will be much faster due to Docker layer caching")
                console.print()

            self._run_docker_command(["docker", "compose"] + compose_files + ["build"], "Building images")
            return

        # Check services
        console.print("\n[bold]Checking service status...[/bold]")
        self._run_docker_command(["docker", "compose"] + compose_files + ["ps"], "Service status")

        # Success message
        console.print("\n[bold green]Setup Complete![/bold green]")
        console.print("\nSemantiK is now running!")
        console.print("Access the web interface at: [link]http://localhost:8080[/link]")
        console.print("\nUseful commands:")
        console.print("  View logs:       make docker-logs")
        console.print("  Stop services:   make docker-down")
        console.print("  Restart:         make docker-restart")

        # Ask about logs
        view_logs = questionary.confirm("\nWould you like to view the logs now?", default=False).ask()

        if view_logs:
            console.print("\n[yellow]Press Ctrl+C to exit logs[/yellow]\n")
            subprocess.run(["docker", "compose"] + compose_files + ["logs", "-f"])

    def _run_docker_command(self, cmd: list[str], description: str) -> bool:
        """Run a Docker command with progress display"""
        # For build commands, show real-time output
        if "build" in cmd or "--build" in cmd:
            console.print(f"\n[bold cyan]{description}[/bold cyan]")
            console.print("[dim]This may take several minutes, especially for GPU images...[/dim]\n")

            try:
                # Run without capturing output to show progress in real-time
                result = subprocess.run(cmd)

                if result.returncode == 0:
                    console.print(f"\n[green]✓ {description} completed successfully[/green]")
                    return True
                console.print(f"\n[red]✗ {description} failed with exit code {result.returncode}[/red]")
                return False
            except Exception as e:
                console.print(f"\n[red]Error running command: {e}[/red]")
                return False
        else:
            # For non-build commands, use the spinner
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
            ) as progress:
                task = progress.add_task(description, total=None)

                try:
                    cmd_result: subprocess.CompletedProcess[str] = subprocess.run(cmd, capture_output=True, text=True)

                    if cmd_result.returncode == 0:
                        progress.update(task, completed=True)
                        return True
                    error_msg = cmd_result.stderr if cmd_result.stderr else "Unknown error"
                    console.print(f"\n[red]Error: {error_msg}[/red]")
                    return False
                except Exception as e:
                    console.print(f"\n[red]Error running command: {e}[/red]")
                    return False

    def _get_compose_files(self) -> list[str]:
        """Get the appropriate docker-compose file arguments based on configuration"""
        files = ["-f", "docker-compose.yml"]

        # Add CUDA override if GPU mode is selected
        # This ensures proper CUDA libraries and environment variables for bitsandbytes
        if self.config.get("USE_GPU") == "true":
            files.extend(["-f", "docker-compose.cuda.yml"])

        return files

    def _check_existing_config(self) -> bool:
        """Check if an existing configuration exists"""
        env_path = Path(".env")
        config_path = Path(".semantik-config.json")
        return env_path.exists() or config_path.exists()

    def _load_existing_config(self) -> None:
        """Load existing configuration from files"""
        # Load from .env file
        env_path = Path(".env")
        if env_path.exists():
            with env_path.open() as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        self.config[key] = value

        # Load from config file if exists
        config_path = Path(".semantik-config.json")
        if config_path.exists():
            with config_path.open() as f:
                saved_config = json.load(f)
                self.config.update(saved_config)

        # Determine if using GPU based on config
        if "USE_GPU" not in self.config:
            # Check if CUDA devices are configured in .env
            if "CUDA_VISIBLE_DEVICES" in self.config:
                self.config["USE_GPU"] = "true"
            else:
                # Default to GPU if hardware is available
                self.config["USE_GPU"] = "true" if self._check_gpu() else "false"

    def _save_config(self) -> None:
        """Save configuration to JSON file for future use"""
        config_path = Path(".semantik-config.json")
        with config_path.open("w") as f:
            json.dump(self.config, f, indent=2)
        console.print(f"[green]Configuration saved to {config_path}[/green]")

    def _select_with_timeout(self, prompt: str, choices: list[str], timeout: int = 5) -> str | None:
        """Select menu with timeout for auto-refresh"""
        result: str | None = None
        event = threading.Event()

        def get_input() -> None:
            nonlocal result
            result = questionary.select(prompt, choices=choices).ask()
            event.set()

        # Start input thread
        input_thread = threading.Thread(target=get_input)
        input_thread.daemon = True
        input_thread.start()

        # Wait for input or timeout
        if event.wait(timeout):
            return result
        # Timeout occurred, return None to trigger refresh
        return None

    def _service_monitor(self) -> None:
        """Interactive service monitoring and management interface"""
        while True:
            console.clear()
            console.print(Panel("[bold]Semantik Service Monitor[/bold]", style="bright_blue"))
            console.print()

            # Get service status
            self._show_service_status()

            console.print("\n[bold]Service Management[/bold]")
            console.print("[dim]Auto-refresh in 10 seconds...[/dim]")

            action = self._select_with_timeout(
                "What would you like to do?",
                choices=[
                    "Start all services",
                    "Stop all services",
                    "Restart all services",
                    "Rebuild and start services",
                    "Reset database (permanent delete)",
                    "View logs",
                    "View specific service logs",
                    "Check service health",
                    "Exit monitor",
                ],
                timeout=10,
            )

            # If timeout occurred, refresh
            if action is None:
                continue

            # If user selected Exit
            if "Exit" in action:
                break

            compose_files = self._get_compose_files()

            if "Reset database" in action:
                self._handle_database_reset()
                continue

            if "Start all" in action:
                # Check if services are already running
                if self._are_services_running():
                    console.print("[yellow]Services are already running. Restarting...[/yellow]")
                    self._run_docker_command(["docker", "compose"] + compose_files + ["down"], "Stopping services")
                    self._run_docker_command(["docker", "compose"] + compose_files + ["up", "-d"], "Starting services")
                else:
                    self._run_docker_command(["docker", "compose"] + compose_files + ["up", "-d"], "Starting services")
            elif "Stop all" in action:
                self._run_docker_command(["docker", "compose"] + compose_files + ["down"], "Stopping services")
            elif "Restart all" in action:
                self._run_docker_command(["docker", "compose"] + compose_files + ["restart"], "Restarting services")
            elif "Rebuild" in action:
                if self.config.get("USE_GPU") == "true":
                    console.print("\n[bold yellow]GPU Rebuild Notice:[/bold yellow]")
                    console.print("• Rebuilding will re-download any updated base images")
                    console.print("• This may take several minutes")
                    console.print()

                self._run_docker_command(["docker", "compose"] + compose_files + ["build"], "Building images")
                self._run_docker_command(["docker", "compose"] + compose_files + ["up", "-d"], "Starting services")
            elif "View logs" in action and "specific" not in action:
                console.print("\n[yellow]Press Ctrl+C to exit logs[/yellow]\n")
                subprocess.run(["docker", "compose"] + compose_files + ["logs", "-f", "--tail", "50"])
            elif "specific service" in action:
                services = self._get_services()
                if services:
                    service = questionary.select("Select service:", choices=services).ask()
                    if service:
                        console.print(f"\n[yellow]Viewing logs for {service}. Press Ctrl+C to exit[/yellow]\n")
                        subprocess.run(["docker", "compose"] + compose_files + ["logs", "-f", "--tail", "50", service])
            elif "health" in action:
                self._check_service_health()

            if "Exit" not in action:
                questionary.press_any_key_to_continue("Press any key to continue...").ask()

    def _handle_database_reset(self) -> None:
        """Interactively confirm and execute a destructive database reset."""
        console.print("\n[bold red]Database Reset[/bold red]")
        console.print(
            "[red]This will stop all Docker services and permanently delete the PostgreSQL data volume.[/red]"
        )
        console.print("[red]All collections, documents, and metadata will be lost. This cannot be undone.[/red]\n")

        proceed = questionary.confirm("Do you want to continue?", default=False).ask()
        if not proceed:
            console.print("[yellow]Database reset cancelled.[/yellow]")
            return

        confirmation = questionary.text("Type DELETE DATABASE to confirm:").ask()
        if confirmation is None or confirmation.strip().upper() != "DELETE DATABASE":
            console.print("[yellow]Database reset aborted. Confirmation phrase did not match.[/yellow]")
            return

        self._perform_database_reset()

    def _perform_database_reset(self) -> None:
        """Stop services, delete database volume, and optionally restart services."""
        compose_files = self._get_compose_files()

        if not self._run_docker_command(["docker", "compose"] + compose_files + ["down"], "Stopping services"):
            console.print("[red]Unable to stop services. Database reset halted.[/red]")
            return

        project_name = self._get_compose_project_name()
        postgres_volume = f"{project_name}_postgres_data"

        volume_removed = self._remove_docker_volume(postgres_volume)
        if not volume_removed:
            console.print("[red]Database volume could not be removed. See errors above.[/red]")
            return

        # Offer to reset the vector index as well since it often mirrors database content
        reset_qdrant = questionary.confirm("Also delete the Qdrant vector index data?", default=False).ask()
        if reset_qdrant:
            qdrant_volume = f"{project_name}_qdrant_storage"
            self._remove_docker_volume(qdrant_volume)

        console.print("\n[green]PostgreSQL volume deleted. Database reset complete.[/green]")
        console.print(
            "[yellow]You will need to re-run migrations or the initial setup the next time services start.[/yellow]"
        )

        restart = questionary.confirm("Start Semantik services now?", default=True).ask()
        if restart:
            self._run_docker_command(["docker", "compose"] + compose_files + ["up", "-d"], "Starting services")
        else:
            console.print("[yellow]Services remain stopped. Run `docker compose up -d` when ready.[/yellow]")

    def _get_compose_project_name(self) -> str:
        """Determine the Docker Compose project name used for volume prefixes."""
        for key in ("COMPOSE_PROJECT_NAME", "project_name"):
            value = os.environ.get(key)
            if value:
                return value

        for key in ("COMPOSE_PROJECT_NAME", "project_name"):
            value = self.config.get(key)
            if value:
                return value

        base = Path.cwd().name.lower()
        safe_name = re.sub(r"[^a-z0-9_-]", "", base)
        return safe_name or "semantik"

    def _remove_docker_volume(self, volume_name: str) -> bool:
        """Remove a Docker volume, handling missing-volume errors gracefully."""
        console.print(f"[cyan]Removing Docker volume: {volume_name}[/cyan]")

        try:
            result = subprocess.run(
                ["docker", "volume", "rm", volume_name],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            console.print("[red]Docker is not installed or not accessible on this system.[/red]")
            return False
        except Exception as exc:  # pragma: no cover - defensive logging
            console.print(f"[red]Unexpected error removing volume: {exc}[/red]")
            return False

        if result.returncode == 0:
            console.print(f"[green]✓ Removed volume {volume_name}[/green]")
            return True

        stderr = (result.stderr or result.stdout or "").strip()
        if "No such volume" in stderr:
            console.print(f"[yellow]Volume {volume_name} not found (already removed).[/yellow]")
            return True

        console.print(f"[red]Failed to remove volume {volume_name}: {stderr}[/red]")
        return False

    def _show_service_status(self) -> None:
        """Display current status of all services"""
        compose_files = self._get_compose_files()

        result = subprocess.run(
            ["docker", "compose"] + compose_files + ["ps", "--format", "json"], capture_output=True, text=True
        )

        if result.returncode == 0 and result.stdout:
            # Create status table
            table = Table(title="Service Status")
            table.add_column("Service", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Ports", style="yellow")
            table.add_column("Health", style="magenta")

            try:
                # Parse JSON output line by line
                for line in result.stdout.strip().split("\n"):
                    if line:
                        service = json.loads(line)
                        name = service.get("Service", "Unknown")
                        state = service.get("State", "Unknown")

                        # Color code status
                        if state == "running":
                            status = "[green]● Running[/green]"
                        elif state == "exited":
                            status = "[red]● Stopped[/red]"
                        else:
                            status = f"[yellow]● {state}[/yellow]"

                        # Get ports
                        ports = service.get("Publishers", [])
                        port_str = ", ".join(
                            [
                                f"{p.get('PublishedPort', '')}:{p.get('TargetPort', '')}"
                                for p in ports
                                if p.get("PublishedPort")
                            ]
                        )

                        # Get health status
                        health = service.get("Health", "")
                        if not health:
                            health = "N/A"
                        elif "healthy" in health.lower():
                            health = "[green]Healthy[/green]"
                        elif "unhealthy" in health.lower():
                            health = "[red]Unhealthy[/red]"
                        else:
                            health = "[yellow]" + health + "[/yellow]"

                        table.add_row(name, status, port_str or "None", health)
            except Exception:
                # Fallback to simple ps output
                result = subprocess.run(["docker", "compose"] + compose_files + ["ps"], capture_output=True, text=True)
                console.print("[yellow]Service Status:[/yellow]")
                console.print(result.stdout)
                return

            console.print(table)
        else:
            console.print("[red]No services found or Docker Compose error[/red]")

    def _get_services(self) -> list[str]:
        """Get list of service names"""
        compose_files = self._get_compose_files()

        result = subprocess.run(
            ["docker", "compose"] + compose_files + ["config", "--services"], capture_output=True, text=True
        )

        if result.returncode == 0:
            return result.stdout.strip().split("\n")
        return []

    def _check_service_health(self) -> None:
        """Check detailed health status of services"""
        console.print("\n[bold]Checking service health...[/bold]\n")

        services = self._get_services()
        for service in services:
            console.print(f"[cyan]{service}:[/cyan]")

            # Try to access health endpoint
            port_map = {"webui": 8080, "vecpipe": 8000, "qdrant": 6333, "postgres": 5432, "redis": 6379, "flower": 5555}

            if service in port_map:
                port = port_map[service]
                # Special handling for services without HTTP health endpoints
                if service == "worker":
                    # Worker health is checked via docker healthcheck
                    result = subprocess.run(
                        ["docker", "inspect", "--format", "{{.State.Health.Status}}", "semantik-worker"],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        health_status = result.stdout.strip()
                        if health_status == "healthy":
                            console.print("  [green]✓ Health check passed (Docker healthcheck)[/green]")
                        else:
                            console.print(f"  [yellow]! Health status: {health_status}[/yellow]")
                    else:
                        console.print("  [red]✗ Could not check health status[/red]")
                else:
                    try:
                        import requests

                        response = requests.get(f"http://localhost:{port}/health", timeout=5)
                        if response.status_code == 200:
                            console.print("  [green]✓ Health check passed[/green]")
                        else:
                            console.print(f"  [yellow]! Health check returned {response.status_code}[/yellow]")
                    except Exception as e:
                        console.print(f"  [red]✗ Health check failed: {str(e)}[/red]")
            else:
                console.print("  [dim]No health endpoint[/dim]")

            console.print()

    def _are_services_running(self) -> bool:
        """Check if any services are currently running"""
        compose_files = self._get_compose_files()

        result = subprocess.run(
            ["docker", "compose"] + compose_files + ["ps", "--format", "json"], capture_output=True, text=True
        )

        if result.returncode == 0 and result.stdout:
            try:
                # Check if any service is running
                for line in result.stdout.strip().split("\n"):
                    if line:
                        service = json.loads(line)
                        if service.get("State") == "running":
                            return True
            except Exception:
                # Fallback - check with simple ps
                result = subprocess.run(
                    ["docker", "compose"] + compose_files + ["ps", "-q"], capture_output=True, text=True
                )
                return bool(result.stdout.strip())

        return False

    def _detect_common_directories(self) -> list[Path]:
        """Detect common document directories"""
        home = Path.home()
        common_paths = [
            home / "Documents",
            home / "Downloads",
            home / "Desktop",
            Path.cwd() / "documents",
            Path.cwd() / "data",
        ]

        existing_dirs = []
        for path in common_paths:
            if path.exists() and path.is_dir() and self._count_documents(path) > 0:
                existing_dirs.append(path)

        return existing_dirs

    def _count_documents(self, directory: Path) -> int:
        """Count document files in a directory"""
        try:
            doc_extensions = {".pdf", ".docx", ".txt", ".md", ".html", ".pptx", ".csv", ".json"}
            count = 0
            for file in directory.iterdir():
                if file.is_file() and file.suffix.lower() in doc_extensions:
                    count += 1
                    if count >= 10:  # Stop counting after 10 for performance
                        return count
            return count
        except Exception:
            return 0

    def _check_ports(self) -> bool:
        """Check if required ports are available"""
        import socket

        console.print("[bold]Checking port availability...[/bold]\n")

        required_ports = [
            (8080, "WebUI"),
            (8000, "VecPipe API"),
            (6333, "Qdrant"),
            (6334, "Qdrant gRPC"),
            (5432, "PostgreSQL"),
        ]

        blocked_ports = []

        for port, service in required_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            try:
                result = sock.connect_ex(("localhost", port))
                if result == 0:
                    # Port is in use
                    blocked_ports.append((port, service))
                    console.print(f"[red]✗[/red] Port {port} ({service}) is already in use")
                else:
                    console.print(f"[green]✓[/green] Port {port} ({service}) is available")
            except Exception:
                console.print(f"[green]✓[/green] Port {port} ({service}) is available")
            finally:
                sock.close()

        if blocked_ports:
            console.print(f"\n[red]Error: {len(blocked_ports)} port(s) are already in use.[/red]")
            console.print("\n[yellow]To fix this:[/yellow]")
            console.print("1. Stop the services using these ports:")
            for port, _ in blocked_ports:
                console.print(f"   → Port {port}: sudo lsof -i :{port} or sudo netstat -tlnp | grep {port}")
            console.print("2. Or change the ports in docker-compose.yml")

            # Offer to try stopping existing Semantik containers
            if questionary.confirm("\nWould you like to try stopping existing Semantik containers?").ask():
                compose_file = (
                    "docker-compose.yml" if self.config.get("USE_GPU") == "true" else "docker-compose-cpu-only.yml"
                )
                self._run_docker_command(
                    ["docker", "compose", "-f", compose_file, "down"], "Stopping existing containers"
                )
                console.print("\nPlease run the setup again.")

            return False

        console.print("\n[green]All required ports are available![/green]\n")
        return True


def main() -> None:
    """Main entry point"""
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Run TUI
    tui = DockerSetupTUI()
    tui.run()


if __name__ == "__main__":
    main()
