#!/usr/bin/env python3
"""Interactive Docker Setup TUI for Semantik"""

import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import questionary
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


class DockerSetupTUI:
    def __init__(self) -> None:
        self.config: Dict[str, str] = {}
        self.gpu_available = False
        self.docker_available = False
        self.compose_available = False
        self.docker_gpu_available = False

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

        # Check GPU availability
        self.gpu_available = self._check_gpu()
        if self.gpu_available:
            console.print("[green]✓[/green] GPU detected (NVIDIA)")

            # Check Docker GPU runtime
            console.print("  [dim]Checking Docker GPU support...[/dim]")
            self.docker_gpu_available = self._check_docker_gpu_runtime()
            if self.docker_gpu_available:
                console.print("  [green]✓[/green] Docker GPU runtime available")
            else:
                console.print("  [yellow]![/yellow] Docker GPU runtime not configured")
                console.print("  [dim]Installation will be offered if you select GPU mode[/dim]")
        else:
            console.print("[yellow]![/yellow] No GPU detected (CPU mode available)")
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
        except:
            return False

    def _check_gpu(self) -> bool:
        """Check if NVIDIA GPU is available"""
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

    def _check_docker_gpu_runtime(self) -> bool:
        """Check if Docker can use GPU runtime (NVIDIA Container Toolkit)"""
        if not self.docker_available:
            return False

        try:
            # Test if Docker can run with GPU support
            result = subprocess.run(
                ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:11.0-base", "nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Check if the command succeeded and nvidia-smi output is present
            if result.returncode == 0 and "NVIDIA-SMI" in result.stdout:
                return True

            # Check for common error indicators
            error_indicators = [
                "nvidia-container-cli",
                "libnvidia-ml.so",
                "could not select device driver",
                "unknown flag: --gpus",
            ]
            error_text = result.stderr.lower() if result.stderr else ""
            for indicator in error_indicators:
                if indicator in error_text:
                    return False

            return False
        except subprocess.TimeoutExpired:
            return False
        except:
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
            elif distro_info and (
                "fedora" in distro_info.lower() or "rhel" in distro_info.lower() or "centos" in distro_info.lower()
            ):
                return self._install_nvidia_toolkit_rhel()
            elif distro_info and ("arch" in distro_info.lower() or "manjaro" in distro_info.lower()):
                return self._install_nvidia_toolkit_arch()
            else:
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

    def _get_linux_distro(self) -> Optional[str]:
        """Get Linux distribution information"""
        try:
            # Try lsb_release first
            result = subprocess.run(["lsb_release", "-si"], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()

            # Try /etc/os-release
            if Path("/etc/os-release").exists():
                with open("/etc/os-release", "r") as f:
                    for line in f:
                        if line.startswith("ID="):
                            return line.split("=")[1].strip().strip('"')
        except:
            pass

        return None

    def _install_nvidia_toolkit_debian(self) -> bool:
        """Install NVIDIA Container Toolkit on Debian/Ubuntu"""
        # These commands need to be run with shell=True due to pipes
        shell_commands: List[str] = [
            # Add the package repository
            "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
            "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list",
        ]

        # These commands can be run normally
        regular_commands: List[List[str]] = [
            ["sudo", "apt-get", "update"],
            ["sudo", "apt-get", "install", "-y", "nvidia-container-toolkit"],
            ["sudo", "nvidia-ctk", "runtime", "configure", "--runtime=docker"],
            ["sudo", "systemctl", "restart", "docker"],
        ]

        console.print("Running installation commands for Debian/Ubuntu...\n")

        # Execute shell commands first
        for shell_cmd in shell_commands:
            console.print(f"[dim]$ {shell_cmd}[/dim]")
            result = subprocess.run(shell_cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                console.print(f"[red]Error: {result.stderr}[/red]")
                return False
            else:
                console.print("[green]✓ Success[/green]\n")

        # Execute regular commands
        for cmd_list in regular_commands:
            console.print(f"[dim]$ {' '.join(cmd_list)}[/dim]")
            result = subprocess.run(cmd_list, capture_output=True, text=True)

            if result.returncode != 0:
                console.print(f"[red]Error: {result.stderr}[/red]")
                return False
            else:
                console.print("[green]✓ Success[/green]\n")

        # Test if it works now
        console.print("[bold]Testing GPU support...[/bold]")
        if self._check_docker_gpu_runtime():
            console.print("[green]✓ NVIDIA Container Toolkit installed successfully![/green]")
            return True
        else:
            console.print("[red]Installation completed but GPU support test failed.[/red]")
            console.print("You may need to log out and back in, or reboot.")
            return False

    def _install_nvidia_toolkit_rhel(self) -> bool:
        """Install NVIDIA Container Toolkit on RHEL/Fedora/CentOS"""
        # Shell command for repository setup
        shell_command = "curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo"

        # Regular commands
        regular_commands: List[List[str]] = [
            ["sudo", "yum", "install", "-y", "nvidia-container-toolkit"],
            ["sudo", "nvidia-ctk", "runtime", "configure", "--runtime=docker"],
            ["sudo", "systemctl", "restart", "docker"],
        ]

        console.print("Running installation commands for RHEL/Fedora/CentOS...\n")

        # Execute shell command
        console.print(f"[dim]$ {shell_command}[/dim]")
        result = subprocess.run(shell_command, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            console.print(f"[red]Error: {result.stderr}[/red]")
            return False
        else:
            console.print("[green]✓ Success[/green]\n")

        # Execute regular commands
        for cmd_list in regular_commands:
            console.print(f"[dim]$ {' '.join(cmd_list)}[/dim]")
            result = subprocess.run(cmd_list, capture_output=True, text=True)

            if result.returncode != 0:
                console.print(f"[red]Error: {result.stderr}[/red]")
                return False
            else:
                console.print("[green]✓ Success[/green]\n")

        # Test if it works now
        console.print("[bold]Testing GPU support...[/bold]")
        if self._check_docker_gpu_runtime():
            console.print("[green]✓ NVIDIA Container Toolkit installed successfully![/green]")
            return True
        else:
            console.print("[red]Installation completed but GPU support test failed.[/red]")
            console.print("You may need to log out and back in, or reboot.")
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
                else:
                    console.print("[green]✓ Success[/green]\n")
        else:
            # No AUR helper, use official method
            console.print("No AUR helper found. Using official NVIDIA repositories...\n")
            return self._install_nvidia_toolkit_arch_official()

        # Test if it works now
        console.print("[bold]Testing GPU support...[/bold]")
        if self._check_docker_gpu_runtime():
            console.print("[green]✓ NVIDIA Container Toolkit installed successfully![/green]")
            return True
        else:
            console.print("[red]Installation completed but GPU support test failed.[/red]")
            console.print("You may need to log out and back in, or reboot.")
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

    def _get_all_gpu_info(self) -> List[Dict[str, Any]]:
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
        self._show_progress(1, 4, "Deployment Type")
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

    def configure_directories(self) -> bool:
        """Configure directory mappings"""
        self._show_progress(2, 4, "Directory Configuration")
        console.print("[bold]Configure Document Directories[/bold]\n")
        console.print("You can add multiple directories containing documents to process.")
        console.print("These directories will be mounted read-only in the Docker container.\n")

        document_dirs: List[Path] = []

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
                    console.print(f"\n[green]Selected directories:[/green]")
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
                elif not path_input:
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

    def _browse_for_directory(self) -> Optional[Path]:
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
                        else:
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
        except:
            pass

    def configure_security(self) -> bool:
        """Configure security and advanced settings"""
        self._show_progress(3, 4, "Security & Advanced Settings")
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
        self.config["WEBUI_WORKERS"] = "auto"

        console.print()
        return True

    def review_configuration(self) -> bool:
        """Review and confirm configuration"""
        self._show_progress(4, 4, "Review & Confirm")
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

        table.add_row("JWT Secret", "***" + self.config["JWT_SECRET_KEY"][-8:])
        table.add_row("Token Expiration", f"{self.config['ACCESS_TOKEN_EXPIRE_MINUTES']} minutes")
        table.add_row("Log Level", self.config["LOG_LEVEL"])
        table.add_row("WebUI Workers", "auto")

        console.print(table)
        console.print()

        # Confirm
        confirm = questionary.confirm("Proceed with this configuration?", default=True).ask()

        if not confirm:
            return False

        # Save configuration
        self._save_env_file()
        self._save_config()  # Save to JSON for future use
        console.print("[green]Configuration saved to .env and .semantik-config.json[/green]\n")
        return True

    def _save_env_file(self) -> None:
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

        with open(template_path, "r") as f:
            content = f.read()

        # Replace values
        replacements = {
            "CHANGE_THIS_TO_A_STRONG_SECRET_KEY": self.config["JWT_SECRET_KEY"],
            "ACCESS_TOKEN_EXPIRE_MINUTES=1440": f"ACCESS_TOKEN_EXPIRE_MINUTES={self.config['ACCESS_TOKEN_EXPIRE_MINUTES']}",
            "DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B": f"DEFAULT_EMBEDDING_MODEL={self.config['DEFAULT_EMBEDDING_MODEL']}",
            "DEFAULT_QUANTIZATION=float16": f"DEFAULT_QUANTIZATION={self.config['DEFAULT_QUANTIZATION']}",
            "DOCUMENT_PATH=./documents": f"DOCUMENT_PATH={self.config['DOCUMENT_PATH']}",
            "WEBUI_WORKERS=1": "WEBUI_WORKERS=auto",
            "LOG_LEVEL=INFO": f"LOG_LEVEL={self.config['LOG_LEVEL']}",
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
        with open(".env", "w") as f:
            f.write(content)

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

        # Determine compose file
        compose_file = "docker-compose.yml" if self.config["USE_GPU"] == "true" else "docker-compose-cpu-only.yml"

        # Execute based on choice
        if "Build and start" in action:
            self._run_docker_command(["docker", "compose", "-f", compose_file, "build"], "Building images")
            self._run_docker_command(["docker", "compose", "-f", compose_file, "up", "-d"], "Starting services")
        elif "Start services" in action:
            self._run_docker_command(["docker", "compose", "-f", compose_file, "up", "-d"], "Starting services")
        else:
            self._run_docker_command(["docker", "compose", "-f", compose_file, "build"], "Building images")
            return

        # Check services
        console.print("\n[bold]Checking service status...[/bold]")
        self._run_docker_command(["docker", "compose", "-f", compose_file, "ps"], "Service status")

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
            subprocess.run(["docker", "compose", "-f", compose_file, "logs", "-f"])

    def _run_docker_command(self, cmd: List[str], description: str) -> bool:
        """Run a Docker command with progress display"""
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console
        ) as progress:
            task = progress.add_task(description, total=None)

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    progress.update(task, completed=True)
                    return True
                else:
                    console.print(f"\n[red]Error: {result.stderr}[/red]")
                    return False
            except Exception as e:
                console.print(f"\n[red]Error running command: {e}[/red]")
                return False

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
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        self.config[key] = value

        # Load from config file if exists
        config_path = Path(".semantik-config.json")
        if config_path.exists():
            with open(config_path, "r") as f:
                saved_config = json.load(f)
                self.config.update(saved_config)

        # Determine if using GPU based on config or docker-compose file
        if "USE_GPU" not in self.config:
            # Check which docker-compose is being used
            if Path("docker-compose-cpu-only.yml").exists():
                compose_files = subprocess.run(
                    ["docker", "compose", "config", "--services"], capture_output=True, text=True
                )
                self.config["USE_GPU"] = "true"  # Default to GPU unless we detect CPU-only
            else:
                self.config["USE_GPU"] = "true"

    def _save_config(self) -> None:
        """Save configuration to JSON file for future use"""
        config_path = Path(".semantik-config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        console.print(f"[green]Configuration saved to {config_path}[/green]")

    def _service_monitor(self) -> None:
        """Interactive service monitoring and management interface"""
        while True:
            console.clear()
            console.print(Panel("[bold]Semantik Service Monitor[/bold]", style="bright_blue"))
            console.print()

            # Get service status
            self._show_service_status()

            console.print("\n[bold]Service Management[/bold]")
            action = questionary.select(
                "What would you like to do?",
                choices=[
                    "Start all services",
                    "Stop all services",
                    "Restart all services",
                    "Rebuild and start services",
                    "View logs",
                    "View specific service logs",
                    "Check service health",
                    "Refresh status",
                    "Exit monitor",
                ],
            ).ask()

            if action is None or "Exit" in action:
                break

            compose_file = (
                "docker-compose.yml" if self.config.get("USE_GPU") == "true" else "docker-compose-cpu-only.yml"
            )

            if "Start all" in action:
                # Check if services are already running
                if self._are_services_running():
                    console.print("[yellow]Services are already running. Restarting...[/yellow]")
                    self._run_docker_command(["docker", "compose", "-f", compose_file, "down"], "Stopping services")
                    self._run_docker_command(["docker", "compose", "-f", compose_file, "up", "-d"], "Starting services")
                else:
                    self._run_docker_command(["docker", "compose", "-f", compose_file, "up", "-d"], "Starting services")
            elif "Stop all" in action:
                self._run_docker_command(["docker", "compose", "-f", compose_file, "down"], "Stopping services")
            elif "Restart all" in action:
                self._run_docker_command(["docker", "compose", "-f", compose_file, "restart"], "Restarting services")
            elif "Rebuild" in action:
                self._run_docker_command(["docker", "compose", "-f", compose_file, "build"], "Building images")
                self._run_docker_command(["docker", "compose", "-f", compose_file, "up", "-d"], "Starting services")
            elif "View logs" in action and "specific" not in action:
                console.print("\n[yellow]Press Ctrl+C to exit logs[/yellow]\n")
                subprocess.run(["docker", "compose", "-f", compose_file, "logs", "-f", "--tail", "50"])
            elif "specific service" in action:
                services = self._get_services()
                if services:
                    service = questionary.select("Select service:", choices=services).ask()
                    if service:
                        console.print(f"\n[yellow]Viewing logs for {service}. Press Ctrl+C to exit[/yellow]\n")
                        subprocess.run(["docker", "compose", "-f", compose_file, "logs", "-f", "--tail", "50", service])
            elif "health" in action:
                self._check_service_health()

            if "Exit" not in action:
                questionary.press_any_key_to_continue("Press any key to continue...").ask()

    def _show_service_status(self) -> None:
        """Display current status of all services"""
        compose_file = "docker-compose.yml" if self.config.get("USE_GPU") == "true" else "docker-compose-cpu-only.yml"

        result = subprocess.run(
            ["docker", "compose", "-f", compose_file, "ps", "--format", "json"], capture_output=True, text=True
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
            except:
                # Fallback to simple ps output
                result = subprocess.run(["docker", "compose", "-f", compose_file, "ps"], capture_output=True, text=True)
                console.print("[yellow]Service Status:[/yellow]")
                console.print(result.stdout)
                return

            console.print(table)
        else:
            console.print("[red]No services found or Docker Compose error[/red]")

    def _get_services(self) -> List[str]:
        """Get list of service names"""
        compose_file = "docker-compose.yml" if self.config.get("USE_GPU") == "true" else "docker-compose-cpu-only.yml"

        result = subprocess.run(
            ["docker", "compose", "-f", compose_file, "config", "--services"], capture_output=True, text=True
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
            port_map = {"webui": 8080, "vecpipe": 8000, "qdrant": 6333}

            if service in port_map:
                port = port_map[service]
                try:
                    import requests

                    response = requests.get(f"http://localhost:{port}/health", timeout=5)
                    if response.status_code == 200:
                        console.print(f"  [green]✓ Health check passed[/green]")
                    else:
                        console.print(f"  [yellow]! Health check returned {response.status_code}[/yellow]")
                except Exception as e:
                    console.print(f"  [red]✗ Health check failed: {str(e)}[/red]")
            else:
                console.print("  [dim]No health endpoint[/dim]")

            console.print()

    def _are_services_running(self) -> bool:
        """Check if any services are currently running"""
        compose_file = "docker-compose.yml" if self.config.get("USE_GPU") == "true" else "docker-compose-cpu-only.yml"

        result = subprocess.run(
            ["docker", "compose", "-f", compose_file, "ps", "--format", "json"], capture_output=True, text=True
        )

        if result.returncode == 0 and result.stdout:
            try:
                # Check if any service is running
                for line in result.stdout.strip().split("\n"):
                    if line:
                        service = json.loads(line)
                        if service.get("State") == "running":
                            return True
            except:
                # Fallback - check with simple ps
                result = subprocess.run(
                    ["docker", "compose", "-f", compose_file, "ps", "-q"], capture_output=True, text=True
                )
                return bool(result.stdout.strip())

        return False

    def _detect_common_directories(self) -> List[Path]:
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
            if path.exists() and path.is_dir():
                # Check if it contains any documents
                if self._count_documents(path) > 0:
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
        except:
            return 0

    def _check_ports(self) -> bool:
        """Check if required ports are available"""
        import socket

        console.print("[bold]Checking port availability...[/bold]\n")

        required_ports = [(8080, "WebUI"), (8000, "VecPipe API"), (6333, "Qdrant"), (6334, "Qdrant gRPC")]

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
            except:
                console.print(f"[green]✓[/green] Port {port} ({service}) is available")
            finally:
                sock.close()

        if blocked_ports:
            console.print(f"\n[red]Error: {len(blocked_ports)} port(s) are already in use.[/red]")
            console.print("\n[yellow]To fix this:[/yellow]")
            console.print("1. Stop the services using these ports:")
            for port, service in blocked_ports:
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
