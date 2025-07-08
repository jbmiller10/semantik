# Docker Setup TUI for Semantik - Implementation Plan

## Overview
Create an interactive text-based user interface (TUI) to help users easily configure and launch Semantik with Docker, replacing the manual setup process with a user-friendly wizard.

## Implementation Plan

### 1. **Create TUI Script** (`docker_setup_tui.py`)
- Add to root directory
- Use `questionary` library for interactive prompts (simple, clean UI)
- Structure:
  ```
  - Welcome screen with ASCII art/branding
  - System checks (Docker, docker-compose, GPU availability)
  - Configuration wizard
  - Review & confirmation
  - Execution with progress display
  ```

### 2. **Add Dependencies to `pyproject.toml`**
- Add `questionary = "^2.0.1"` for interactive prompts
- Add `rich = "^13.0.0"` for colored output and progress bars

### 3. **TUI Flow Design**

**Step 1: Welcome & System Check**
- Display Semantik logo/banner
- Check Docker & docker-compose availability
- Auto-detect GPU availability (nvidia-smi check)
- Show system status

**Step 2: Deployment Type**
- Radio selection: GPU (recommended if available) / CPU Only
- Explain performance implications

**Step 3: Directory Configuration**
- Documents directory (default: ./documents)
  - Validate path exists or offer to create
  - Show current path contents preview
- Data directory (default: ./data)
- Logs directory (default: ./logs)

**Step 4: Model Configuration**
- Embedding model selection:
  - Default: Qwen/Qwen3-Embedding-0.6B
  - Option to enter custom model
- Quantization: float16 / int8 / none
- If GPU selected:
  - GPU device selection (if multiple)
  - Memory limit (default: 8GB, show available)

**Step 5: Security & Advanced**
- JWT Secret: Auto-generate (default) / Custom
- Access token expiration (default: 24h)
- Log level: INFO (default) / DEBUG / WARNING
- Number of workers (default: 1)

**Step 6: Review Configuration**
- Display all settings in a table
- Option to go back and modify
- Save configuration option

**Step 7: Execution Options**
- Build fresh images
- Start services
- Both (recommended for first run)
- Show real-time progress with spinner/progress bar

**Step 8: Post-Setup**
- Display service status
- Show access URL: http://localhost:8080
- Quick health check
- Option to view logs
- Save configuration for future use

### 4. **Key Features**
- **Validation**: Path existence, GPU availability, port conflicts
- **Smart Defaults**: Pre-fill with sensible values
- **Error Recovery**: Graceful handling of Docker errors
- **Configuration Persistence**: Save/load previous configurations
- **Help Text**: Context-sensitive help for each option
- **Progress Feedback**: Real-time output from Docker commands

### 5. **Integration**
- Add `make docker-setup` target to Makefile
- Update README with TUI instructions
- Maintain compatibility with existing `make docker-up`

### 6. **Error Handling**
- Docker not installed → Installation instructions
- Ports in use → Offer to stop conflicting services
- GPU not available → Automatic CPU fallback
- Build failures → Show logs and troubleshooting steps

### 7. **Configuration Management**
- Create `.env` from template
- Backup existing `.env` if present
- Option to save configuration profiles (e.g., `.semantik-config.json`)
- Load previous configurations for quick setup

### 8. **Post-Setup Options**
- View service logs
- Stop services
- Restart services
- Open web UI in browser (if possible)

## Files to Create/Modify
1. `docker_setup_tui.py` - Main TUI script
2. `pyproject.toml` - Add dependencies
3. `Makefile` - Add docker-setup target
4. `README.md` - Add TUI setup instructions

## Technical Considerations
- Use subprocess for Docker commands with real-time output
- Implement proper signal handling (Ctrl+C)
- Support both Linux and macOS differences
- Validate all user inputs
- Provide rollback on failure
- Keep existing manual setup as fallback

## Todo List
1. ✅ Add TUI dependencies (questionary and rich) to pyproject.toml
2. ✅ Create docker_setup_tui.py with system checks and welcome screen
3. ⏳ Implement deployment type selection (GPU/CPU)
4. ⏳ Implement directory configuration step
5. ⏳ Implement model and GPU configuration
6. ⏳ Implement security and advanced settings
7. ⏳ Implement configuration review and save functionality
8. ⏳ Implement Docker execution with progress display
9. ⏳ Add docker-setup target to Makefile
10. ⏳ Test the complete TUI flow
11. ⏳ Run make format and make type-check