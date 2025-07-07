#!/usr/bin/env python3
"""Interactive Docker Setup TUI for Semantik"""

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

            # Run through setup steps
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
            console.print("\nPlease install Docker: https://docs.docker.com/get-docker/")
            return False

        # Check Docker Compose
        self.compose_available = self._check_docker_compose()
        if self.compose_available:
            console.print("[green]✓[/green] Docker Compose found")
        else:
            console.print("[red]✗[/red] Docker Compose not found")
            console.print("\nPlease install Docker Compose")
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

    def select_deployment_type(self) -> bool:
        """Select GPU or CPU deployment"""
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
            # GPU selection
            gpu_id = questionary.text("GPU device ID (0 for first GPU):", default="0").ask()
            if gpu_id is None:
                return False
            self.config["CUDA_VISIBLE_DEVICES"] = gpu_id

            # Memory limit
            mem_limit = questionary.text("GPU memory limit in GB:", default="8").ask()
            if mem_limit is None:
                return False
            self.config["MODEL_MAX_MEMORY_GB"] = mem_limit

        console.print()
        return True

    def configure_security(self) -> bool:
        """Configure security and advanced settings"""
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
        console.print("[green]Configuration saved to .env[/green]\n")
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
