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
from typing import Dict, List, Optional, Tuple

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

            # Ask for setup type
            setup_type = questionary.select(
                "Choose setup type:",
                choices=["Quick Setup (Recommended) - Use sensible defaults", "Custom Setup - Configure all options"],
            ).ask()

            if setup_type is None:
                return

            if "Quick Setup" in setup_type:
                self._quick_setup()
            else:
                # Run through full setup steps
                if not self.select_deployment_type():
                    return
                if not self.configure_directories():
                    return
                if not self.configure_model():
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
                console.print("  → Arch: sudo pacman -S docker docker-compose")
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
        else:
            console.print("[yellow]![/yellow] No GPU detected (CPU mode available)")

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

        self.config["USE_GPU"] = "true" if "GPU" in deployment else "false"
        console.print()
        return True

    def configure_directories(self) -> bool:
        """Configure directory mappings"""
        self._show_progress(2, 5, "Directory Configuration")
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

    def configure_model(self) -> bool:
        """Configure embedding model and GPU settings"""
        self._show_progress(3, 5, "Model Configuration")
        console.print("[bold]Configure Embedding Model[/bold]\n")

        # Model selection
        models = [
            "Qwen/Qwen3-Embedding-0.6B (Recommended - 600M params)",
            "sentence-transformers/all-MiniLM-L6-v2 (Lightweight - 22M params)",
            "Custom model",
        ]

        model_choice = questionary.select("Select embedding model:", choices=models).ask()

        if model_choice is None:
            return False

        if "Qwen" in model_choice:
            self.config["DEFAULT_EMBEDDING_MODEL"] = "Qwen/Qwen3-Embedding-0.6B"
        elif "MiniLM" in model_choice:
            self.config["DEFAULT_EMBEDDING_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
        else:
            custom_model = questionary.text("Enter model name:", default="Qwen/Qwen3-Embedding-0.6B").ask()
            if custom_model is None:
                return False
            self.config["DEFAULT_EMBEDDING_MODEL"] = custom_model

        # Quantization
        quant_choice = questionary.select(
            "Select quantization (reduces memory usage):",
            choices=[
                "float16 (Recommended - good balance)",
                "int8 (More compression, slight quality loss)",
                "none (Full precision)",
            ],
        ).ask()

        if quant_choice is None:
            return False

        if "float16" in quant_choice:
            self.config["DEFAULT_QUANTIZATION"] = "float16"
        elif "int8" in quant_choice:
            self.config["DEFAULT_QUANTIZATION"] = "int8"
        else:
            self.config["DEFAULT_QUANTIZATION"] = "none"

        # GPU-specific settings
        if self.config["USE_GPU"] == "true":
            console.print("\n[bold]GPU Configuration[/bold]")
            console.print("[dim]Leave defaults unless you have multiple GPUs or limited memory[/dim]\n")

            # GPU selection
            gpu_id = questionary.text(
                "GPU device ID (0 for first GPU):",
                default="0",
                validate=lambda x: x.isdigit() or "Please enter a number",
            ).ask()
            if gpu_id is None:
                return False
            self.config["CUDA_VISIBLE_DEVICES"] = gpu_id

            # Memory limit
            console.print("\n[dim]Recommended: 8GB for most models, 4GB if you have limited VRAM[/dim]")
            mem_limit = questionary.text(
                "GPU memory limit in GB:",
                default="8",
                validate=lambda x: x.replace(".", "").isdigit() or "Please enter a number",
            ).ask()
            if mem_limit is None:
                return False
            self.config["MODEL_MAX_MEMORY_GB"] = mem_limit

        console.print()
        return True

    def configure_security(self) -> bool:
        """Configure security and advanced settings"""
        self._show_progress(4, 5, "Security & Advanced Settings")
        console.print("[bold]Security & Advanced Settings[/bold]\n")

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

        # Workers
        workers = questionary.text("Number of WebUI workers:", default="1").ask()
        if workers is None:
            return False
        self.config["WEBUI_WORKERS"] = workers

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

        table.add_row("Embedding Model", self.config["DEFAULT_EMBEDDING_MODEL"])
        table.add_row("Quantization", self.config["DEFAULT_QUANTIZATION"])

        if self.config["USE_GPU"] == "true":
            table.add_row("GPU Device", self.config["CUDA_VISIBLE_DEVICES"])
            table.add_row("GPU Memory Limit", f"{self.config['MODEL_MAX_MEMORY_GB']} GB")

        table.add_row("JWT Secret", "***" + self.config["JWT_SECRET_KEY"][-8:])
        table.add_row("Token Expiration", f"{self.config['ACCESS_TOKEN_EXPIRE_MINUTES']} minutes")
        table.add_row("Log Level", self.config["LOG_LEVEL"])
        table.add_row("WebUI Workers", self.config["WEBUI_WORKERS"])

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
            "WEBUI_WORKERS=1": f"WEBUI_WORKERS={self.config['WEBUI_WORKERS']}",
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

    def _quick_setup(self) -> None:
        """Quick setup with sensible defaults"""
        console.print("\n[bold]Quick Setup Mode[/bold]")
        console.print("Using recommended defaults for most settings.\n")

        # Auto-detect GPU
        self.config["USE_GPU"] = "true" if self.gpu_available else "false"
        console.print(f"Deployment type: {'GPU' if self.gpu_available else 'CPU'}")

        # Ask only for document directory
        console.print("\n[bold]Select Document Directory[/bold]")
        console.print("Choose the directory containing documents to process:")

        # Try to auto-detect common directories
        common_dirs = self._detect_common_directories()

        if common_dirs:
            console.print("\n[green]Found these potential document directories:[/green]")
            choices = []
            for d in common_dirs:
                choices.append(f"{d} ({self._count_documents(d)} documents)")
            choices.append("Browse for a different directory")
            choices.append("Create new directory")

            selection = questionary.select("Select directory:", choices=choices).ask()

            if selection is None:
                return

            if "Browse" in selection:
                selected_dir = self._browse_for_directory()
                if selected_dir is None:
                    return
            elif "Create new" in selection:
                dir_path = questionary.text("Enter path for new directory:", default="./documents").ask()
                if dir_path:
                    selected_dir = Path(dir_path).resolve()
                    selected_dir.mkdir(parents=True, exist_ok=True)
                    console.print(f"[green]Created {selected_dir}[/green]")
                else:
                    return
            else:
                # Extract path from selection
                selected_dir = Path(selection.split(" (")[0])
        else:
            # No common directories found, use browser
            selected_dir = self._browse_for_directory()
            if selected_dir is None:
                return

        self.config["DOCUMENT_PATHS"] = str(selected_dir)
        self.config["DOCUMENT_PATH"] = str(selected_dir)

        # Set all other defaults
        console.print("\n[bold]Applying default settings:[/bold]")
        self.config["DEFAULT_EMBEDDING_MODEL"] = "Qwen/Qwen3-Embedding-0.6B"
        console.print("  • Embedding model: Qwen/Qwen3-Embedding-0.6B")

        self.config["DEFAULT_QUANTIZATION"] = "float16"
        console.print("  • Quantization: float16")

        if self.config["USE_GPU"] == "true":
            self.config["CUDA_VISIBLE_DEVICES"] = "0"
            self.config["MODEL_MAX_MEMORY_GB"] = "8"
            console.print("  • GPU device: 0 (8GB memory limit)")

        self.config["JWT_SECRET_KEY"] = secrets.token_hex(32)
        console.print("  • Security: Generated secure JWT key")

        self.config["ACCESS_TOKEN_EXPIRE_MINUTES"] = "1440"
        self.config["LOG_LEVEL"] = "INFO"
        self.config["WEBUI_WORKERS"] = "1"
        console.print("  • Token expiry: 24 hours")
        console.print("  • Log level: INFO")
        console.print("  • Workers: 1")

        # Create data and logs directories
        Path("./data").mkdir(exist_ok=True)
        Path("./logs").mkdir(exist_ok=True)

        console.print("\n")

        # Review and confirm
        if not self.review_configuration():
            return

        # Execute setup
        self.execute_setup()

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
