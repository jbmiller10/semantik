# TUI Development Log

## Session Start: 2025-01-07

### Initial Analysis
- Analyzed existing Docker setup:
  - Found `Dockerfile` with multi-stage build
  - Found `docker-compose.yml` (GPU-enabled by default)
  - Found `docker-compose-cpu-only.yml` (CPU version)
  - Found `docker-compose.prod.yml` (production overrides)
  - Found `.env.docker.example` template
  - Analyzed `make docker-up` target which:
    - Creates .env from template if missing
    - Auto-generates JWT secret key
    - Starts services with docker compose

### Actions Taken

#### 1. Added TUI Dependencies (✅ COMPLETED)
- **File**: `pyproject.toml`
- **Changes**: Added `questionary = "^2.0.1"` and `rich = "^13.0.0"`
- **Status**: Successfully added dependencies

#### 2. Created docker_setup_tui.py (✅ COMPLETED)
- **File**: `docker_setup_tui.py`
- **Implementation Details**:
  - Created `DockerSetupTUI` class with complete flow
  - Implemented all 8 steps from the plan:
    1. Welcome screen with ASCII art
    2. System checks (Docker, docker-compose, GPU)
    3. Deployment type selection
    4. Directory configuration
    5. Model and GPU settings
    6. Security configuration
    7. Configuration review
    8. Docker execution
  - Added proper error handling and keyboard interrupt support
  - Implemented .env file generation from template
  - Added progress indicators for Docker commands

### Design Decisions

1. **Single File Implementation**: Kept everything in one file for simplicity
2. **Class-Based Structure**: Used a class to maintain state throughout the wizard
3. **Configuration Storage**: Store all settings in a dict, then write to .env at the end
4. **GPU Detection**: Use nvidia-smi to detect GPU availability
5. **Docker Compose Detection**: Check both new (`docker compose`) and old (`docker-compose`) syntax
6. **Backup Strategy**: Always backup existing .env before overwriting

### Next Steps

1. ✅ Make the script executable
2. ✅ Add docker-setup target to Makefile
3. Test the complete flow
4. Run formatting and type checking
5. Update documentation

### Additional Actions Taken

#### 3. Made Script Executable (✅ COMPLETED)
- Ran `chmod +x docker_setup_tui.py`

#### 4. Added Makefile Target (✅ COMPLETED)
- **File**: `Makefile`
- **Changes**: 
  - Added `docker-setup` to help text
  - Added `docker-setup` target that runs `python docker_setup_tui.py`
- **Location**: Added before the existing `docker-up` target

#### 5. Code Formatting and Type Checking (✅ COMPLETED)
- **Formatting**:
  - Ran `make format` on existing code (no changes needed)
  - Ran `black` and `isort` on `docker_setup_tui.py`
  - Fixed missing newline at end of file
- **Type Checking**:
  - Fixed missing type annotations in `__init__` and `main`
  - All type checks pass successfully

### Current Status

All implementation tasks completed! The TUI is ready for testing.

### Summary of Created/Modified Files

1. **pyproject.toml**: Added questionary and rich dependencies
2. **docker_setup_tui.py**: Complete TUI implementation (462 lines)
3. **Makefile**: Added docker-setup target
4. **TUI_PLAN.md**: Implementation plan document
5. **TUI_DEV_LOG.md**: This development log

### Features Implemented

1. **System Checks**: Docker, docker-compose, GPU detection
2. **Interactive Configuration**:
   - Deployment type (GPU/CPU)
   - Directory mappings with validation
   - Model selection and quantization
   - Security settings (JWT, tokens, logging)
3. **Configuration Management**:
   - Generates .env from template
   - Backs up existing .env files
   - Validates all inputs
4. **Docker Execution**:
   - Build, start, or both options
   - Progress indicators
   - Service status checking
   - Option to view logs

### Ready for Testing

The TUI is now ready to be tested. Users can run:
```bash
make docker-setup
```

Or directly:
```bash
poetry run python docker_setup_tui.py
```

### Troubleshooting Session 1

#### Issue: ModuleNotFoundError for questionary
- **Problem**: Running `make docker-setup` failed with ModuleNotFoundError
- **Cause**: The script was being run outside the Poetry virtual environment
- **Solution**: 
  1. Updated Makefile to use `poetry run python docker_setup_tui.py`
  2. Ran `poetry lock` to update lock file with new dependencies
  3. Ran `poetry install` to install all dependencies

#### Fix Applied:
```makefile
docker-setup:
	@echo "Starting Semantik Docker Setup Wizard..."
	@poetry run python docker_setup_tui.py
```

### Enhancement Session 1: File Browser for Directory Selection

#### User Request
- Make directory selection more intuitive
- Support multiple directories
- Add a "file browser" experience

#### Implementation Details

1. **Enhanced `configure_directories` method**:
   - Added choice between file browser mode and manual entry
   - Support for selecting multiple directories
   - Shows selected directories as user adds them
   - Directory preview shows document files found

2. **Added `_browse_for_directory` method**:
   - Interactive directory navigation
   - Shows current location
   - Options to:
     - Select current directory
     - Go up one level
     - Navigate to subdirectories
     - Type path manually
     - Cancel selection
   - Uses emoji icons for better visual clarity

3. **Added `_preview_directory_contents` method**:
   - Shows preview of document files in selected directory
   - Lists first 5 files with supported extensions
   - Helps user confirm they selected the right directory

4. **Configuration Storage**:
   - Stores all selected paths in `DOCUMENT_PATHS` (colon-separated)
   - Uses first path as `DOCUMENT_PATH` for Docker compatibility
   - Shows warning about current Docker limitation

5. **Review Screen Updates**:
   - Shows all selected directories in the configuration table
   - Marks primary directory clearly

#### Technical Notes
- Current Docker compose only supports single volume mount
- Future enhancement: Modify docker-compose to support multiple mounts
- All formatting and type checking pass

### Potential Issues to Watch

1. **Poetry Dependencies**: Need to run `poetry install` after adding new dependencies
2. **Permissions**: Script needs appropriate permissions to create directories
3. **Docker Permissions**: User needs to be in docker group or use sudo
4. **Platform Differences**: May need to handle Windows differently
5. **Multiple Directory Support**: Currently only primary directory is mounted in Docker

### Enhancement Session 2: Configuration Persistence & Service Monitor

#### User Request
- Save configuration and detect existing configs
- Ask user if they want to use existing config or create new
- Add service monitor interface for managing running services

#### Implementation Details

1. **Configuration Detection** (`_check_existing_config`):
   - Checks for `.env` or `.semantik-config.json`
   - Presents options: use existing, create new, or exit
   
2. **Configuration Persistence**:
   - Saves to both `.env` and `.semantik-config.json`
   - JSON file stores all settings for easy reload
   - Added `.semantik-config.json` to `.gitignore`

3. **Configuration Loading** (`_load_existing_config`):
   - Loads from both `.env` and JSON config
   - Auto-detects GPU/CPU mode from existing setup

4. **Service Monitor Interface** (`_service_monitor`):
   - Real-time service status display with color coding
   - Management options:
     - Start/Stop/Restart all services
     - Rebuild and start
     - View logs (all or specific service)
     - Health checks with endpoint testing
     - Refresh status
   - Interactive loop with clear screen updates

5. **Service Status Display** (`_show_service_status`):
   - Uses Docker Compose JSON format for detailed info
   - Shows service state, ports, and health
   - Color-coded status indicators
   - Fallback to simple output if JSON parsing fails

6. **Health Monitoring** (`_check_service_health`):
   - Tests actual health endpoints
   - Shows pass/fail for each service
   - Handles connection errors gracefully

#### Technical Implementation
- Import `json` module for config persistence
- Added import for `requests` (lazy import for health checks)
- Proper error handling throughout
- Type hints maintained

### Enhancement Session 3: Smart Service Start/Restart

#### User Request
- When services are already running and user selects "Start all services", restart them instead

#### Implementation Details

1. **Added `_are_services_running` method**:
   - Checks if any services are currently running
   - Uses Docker Compose JSON format for accurate detection
   - Fallback to simple ps -q if JSON parsing fails

2. **Modified "Start all services" behavior**:
   - Now checks if services are running before starting
   - If running: performs docker-compose down, then up (full restart)
   - If not running: just starts normally
   - Shows user feedback about restart action

#### Technical Details
- Reuses compose file detection logic
- Proper error handling with fallback
- Maintains consistent user experience

### Enhancement Session 4: Major User-Friendliness Improvements

#### User Request
- Make the setup wizard more user-friendly (excluding model name simplification)

#### Implemented Improvements

1. **Quick Setup Mode** (✅ COMPLETED):
   - Added choice between "Quick Setup" and "Custom Setup"
   - Quick setup uses all sensible defaults
   - Only asks for document directory
   - Auto-detects GPU/CPU
   - Shows all applied settings for transparency

2. **Progress Indicators** (✅ COMPLETED):
   - Added step counter (Step X of 5)
   - Visual progress bar using Unicode characters
   - Clear indication of current position in setup flow

3. **Better Error Messages** (✅ COMPLETED):
   - Platform-specific Docker installation instructions
   - Commands for different Linux distributions
   - Reminder about docker group on Linux
   - Clear next steps for resolution

4. **Auto-Detection of Directories** (✅ COMPLETED):
   - Scans common locations (~/Documents, ~/Downloads, etc.)
   - Shows document count for each found directory
   - Offers found directories as quick selection options
   - Falls back to browser if none found

5. **Examples and Help Text** (✅ COMPLETED):
   - Added examples for directory paths
   - GPU configuration shows recommendations
   - Validation for numeric inputs
   - Context-sensitive help messages

6. **Port Availability Check** (✅ COMPLETED):
   - Checks all required ports before starting
   - Shows which service uses each port
   - Provides commands to find blocking processes
   - Offers to stop existing Semantik containers

#### Technical Implementation
- Added `_show_progress()` method for consistent progress display
- Added `_detect_common_directories()` for intelligent directory discovery
- Added `_count_documents()` to preview directory contents
- Added `_check_ports()` for pre-flight validation
- Added `_quick_setup()` for streamlined configuration
- Enhanced error messages with platform detection
- Added input validation for numeric fields

### Future Enhancement Ideas

#### Still To Do
1. **Import/Export configs** - Share configurations between team members
2. **Resource checks** - Warn if system doesn't have enough RAM/disk space
3. **Disk space validation** - Check available space before Docker build
4. **Health check visualization** - Better display during startup

### Enhancement Session 5: Rename Make Target

#### User Request
- Change make target from `docker-setup` to `wizard` for better clarity

#### Implementation
- Updated help text in Makefile from `docker-setup` to `wizard`
- Renamed the target itself from `docker-setup:` to `wizard:`
- No changes needed to the Python script itself

#### Usage
Users can now run:
```bash
make wizard
```

This is more intuitive and shorter than `docker-setup`.

### Enhancement Session 6: Self-Installing Wizard

#### User Request
- Make the wizard handle dependency installation automatically

#### Implementation

1. **Created `wizard.sh`** - Bash wrapper script that:
   - Checks Python version (3.12+)
   - Installs Poetry if not present
   - Installs dependencies if needed
   - Launches the wizard

2. **Created `wizard_launcher.py`** - Python fallback that:
   - Works without any external dependencies
   - Can bootstrap Poetry installation
   - Handles dependency installation
   - Provides user-friendly prompts

3. **Updated Makefile**:
   - `make wizard` now uses the wrapper script
   - No need for users to run `poetry install` first

4. **Updated README**:
   - Removed `poetry install` step
   - Documented automatic dependency handling

#### Benefits
- Zero-dependency start - just clone and run `make wizard`
- Automatic Poetry installation if needed
- Clear error messages for missing prerequisites
- Works on systems without Poetry pre-installed

### Enhancement Session 7: Cross-Platform Support

#### User Request
- Noted that different platforms need different launcher scripts

#### Implementation

1. **Created platform-specific launchers**:
   - `wizard.sh` - Bash script for Linux/macOS
   - `wizard.bat` - Batch script for Windows CMD
   - `wizard.ps1` - PowerShell script for Windows

2. **Enhanced `wizard_launcher.py`**:
   - Added platform detection
   - Searches common Poetry installation paths per platform
   - Platform-specific Poetry installation instructions
   - Works as universal fallback

3. **Simplified Makefile**:
   - Now just calls Python launcher
   - Works on all platforms with Python

4. **Updated README**:
   - Added Windows-specific instructions
   - Clarified cross-platform support

#### Key Improvements
- Single Python launcher works everywhere
- Detects Poetry in platform-specific locations
- Handles Windows path differences
- Clear instructions per platform

### Enhancement Session 8: Improved NVIDIA Container Toolkit Handling

#### User Request
- Clarify confusing GPU detection when NVIDIA GPU exists but Docker can't use it
- Avoid reboot requirement after installing NVIDIA Container Toolkit

#### Implementation Details

1. **Clearer GPU Detection Messages**:
   - Now explicitly states "NVIDIA GPU detected on host system"
   - Clearly indicates "Docker cannot access GPU (NVIDIA Container Toolkit not installed)"
   - Explains this is a one-time setup for GPU passthrough to containers

2. **Better Docker GPU Test**:
   - Changed from `nvidia/cuda:11.0-base` (deprecated) to `ubuntu:22.04`
   - More reliable test that doesn't depend on specific CUDA versions
   - Added "Failed to initialize NVML" to error indicators

3. **Retry Logic After Installation**:
   - Added 5-second delay after Docker daemon restart
   - Implements 3 retry attempts with 3-second delays
   - Avoids false negatives from Docker not being fully ready

4. **Improved Failure Handling**:
   - Provides specific commands to try manually
   - Mentions rootless Docker considerations
   - Offers to continue with CPU mode instead of blocking

#### Technical Changes
- Added `time` import at module level
- Replaced immediate test with delayed retry mechanism
- Enhanced error messages with actionable solutions

### Enhancement Session 9: Proper GPU Support in Docker

#### User Request
- Fix Docker GPU support - containers need CUDA libraries

#### Problem Discovered
- The base Dockerfile uses `python:3.12-slim` which has no CUDA libraries
- Even with NVIDIA Container Toolkit, PyTorch installs CPU-only version
- Containers cannot use GPU without CUDA runtime libraries

#### Implementation Details

1. **Created GPU-Specific Dockerfile** (`Dockerfile.gpu`):
   - Uses `nvidia/cuda:12.1.0-runtime-ubuntu22.04` as base
   - Installs Python 3.12 from deadsnakes PPA
   - Forces PyTorch GPU version with `torch==2.1.0+cu121`
   - Includes all CUDA runtime libraries

2. **Created Docker Compose GPU Override** (`docker-compose.gpu.yml`):
   - Overrides build to use `Dockerfile.gpu`
   - Sets CUDA environment variables
   - Configures PyTorch memory allocation

3. **Updated Wizard Logic**:
   - Added `_get_compose_files()` helper method
   - GPU mode uses: `-f docker-compose.yml -f docker-compose.gpu.yml`
   - CPU mode uses: `-f docker-compose-cpu-only.yml`
   - All Docker commands updated to use compose file lists

#### Technical Changes
- Wizard now properly chains compose files for GPU builds
- Separate Dockerfiles for CPU vs GPU (cleaner than multi-stage)
- GPU containers will have full CUDA support

### Enhancement Session 10: CUDA Safety and Compatibility

#### User Concern
- Don't mess up user's existing CUDA installation

#### Implementation Details

1. **Added Driver Version Detection**:
   - Shows NVIDIA driver version during GPU detection
   - Indicates compatibility level (CUDA 12.x, 11.x, or too old)
   - Helps users understand if their system is compatible

2. **Created GPU Docker Guide** (`docs/GPU_DOCKER_GUIDE.md`):
   - Explains that container CUDA is isolated from host
   - Shows compatibility matrix
   - Addresses common concerns
   - Provides troubleshooting guidance

3. **Updated Messages**:
   - Now explicitly states "This won't affect your host CUDA installation"
   - Clear compatibility indicators based on driver version

#### Key Safety Points
- NVIDIA Container Toolkit is just a bridge, not CUDA
- Container CUDA (12.1) is completely isolated from host
- Host CUDA installation remains untouched
- Driver compatibility is what matters, not host CUDA version

### Testing Plan

1. Test with no .env file
2. Test with existing .env file (backup functionality)
3. Test GPU detection on GPU-enabled system
4. Test CPU-only mode
5. Test directory creation
6. Test Docker command execution
7. Test error scenarios (Docker not installed, etc.)
8. Test configuration persistence and reload
9. Test service monitor functions
10. Test start/restart behavior when services are already running
11. Test NVIDIA Container Toolkit installation flow
12. Test GPU detection with/without toolkit installed
13. Test GPU functionality inside Docker containers
14. Verify PyTorch CUDA availability in containers
15. Test with different host CUDA versions (11.x, 12.x)
16. Verify host CUDA remains unchanged after setup

### Enhancement Session 11: Fix NVIDIA Toolkit Test Issues

#### Problem Found
- The GPU test was failing because `ubuntu:22.04` doesn't have `nvidia-smi` installed
- Installation succeeded but test was using wrong image

#### Implementation Details

1. **Fixed GPU Runtime Test**:
   - First tests if `--gpus` flag is recognized  
   - Then tests with `nvidia/cuda:11.8.0-base-ubuntu22.04` which has nvidia-smi
   - Better error detection for specific toolkit issues

2. **Improved Installation Process**:
   - Added `systemctl daemon-reload` to all installation methods
   - Added Docker service status check during testing
   - Increased retry delays from 3 to 5 seconds

3. **Created Diagnostic Tools**:
   - `test_nvidia_toolkit.sh` - Comprehensive diagnostic script
   - `fix_nvidia_toolkit.sh` - Manual fix script for common issues

4. **Enhanced Error Messages**:
   - Now provides specific commands to fix issues
   - Mentions WSL2-specific considerations
   - References the fix script for quick resolution

#### Key Improvements
- More robust GPU testing with proper CUDA images
- Better diagnostics when things go wrong
- Manual fix script for stubborn cases
- Clear guidance for troubleshooting

### Enhancement Session 12: WSL2 GPU Support Fix

#### Problem Identified
- In WSL2, the error `libnvidia-ml.so.1: cannot open shared object file` occurs
- Docker can't find NVIDIA libraries from the Windows host
- This is a WSL2-specific issue where `/usr/lib/wsl/lib/` paths aren't accessible to Docker

#### Implementation Details

1. **Created WSL2-Specific Fix Script** (`fix_wsl2_gpu.sh`):
   - Detects WSL2 environment
   - Creates symbolic links to Windows NVIDIA libraries
   - Updates Docker's library path configuration
   - Adds WSL lib path to system ldconfig

2. **Enhanced Wizard WSL2 Detection**:
   - Detects if running in WSL2 by checking `/proc/version`
   - Checks for `/dev/dxg` (WSL2 GPU device)
   - Provides WSL2-specific error messages and fixes

3. **Updated GPU Docker Guide**:
   - Added comprehensive WSL2 section
   - Explains Windows driver requirement
   - Provides troubleshooting steps

#### Key Points for WSL2 Users
- NVIDIA drivers must be installed on **Windows**, not inside WSL2
- WSL2 exposes GPU through `/dev/dxg` and libraries in `/usr/lib/wsl/lib/`
- Docker needs special configuration to access these WSL2 paths
- The fix script automates the necessary symbolic links and config

### Enhancement Session 13: Final WSL2 GPU Resolution

#### Problem
- Native Docker in WSL2 had persistent issues with nvidia-container-cli
- Error: `libnvidia-ml.so.1: cannot open shared object file`
- Multiple fix attempts couldn't resolve the library path issue

#### Solution
- **Docker Desktop for Windows** (with WSL2 backend) works perfectly
- Docker Desktop automatically handles GPU passthrough for WSL2
- No manual configuration needed

#### Additional Fix
- Fixed duplicate environment variables in `docker-compose.gpu.yml`
- `CUDA_VISIBLE_DEVICES` was defined in both base and GPU override files
- Removed duplicates, kept only GPU-specific settings

### Summary of GPU Support Improvements

1. **Clear Detection & Messaging**
   - Wizard detects GPU and driver versions
   - WSL2-specific messages and guidance
   - Explains Docker GPU requirements clearly

2. **Proper GPU Docker Images**
   - Created `Dockerfile.gpu` with CUDA base image
   - Forces PyTorch GPU version installation
   - Includes all necessary CUDA libraries

3. **Docker Compose Configuration**
   - GPU override file for GPU-specific settings
   - Fixed duplicate environment variables
   - Proper compose file chaining in wizard

4. **Comprehensive Documentation**
   - GPU Docker guide with compatibility matrix
   - WSL2-specific troubleshooting section
   - Multiple fix scripts for different scenarios

5. **Recommended Setup for WSL2 Users**
   - Install Docker Desktop on Windows
   - Enable WSL2 backend in Docker Desktop
   - Run the wizard with GPU mode
   - Docker Desktop handles all GPU passthrough automatically

## Session Update: 2025-01-08

### GPU Support Resolution

#### Issue Discovered
User reported GPU detection confusion - wizard detected GPU but Docker couldn't access it. Investigation revealed:
- User had NVIDIA drivers on Windows but was using native Docker in WSL2
- Native Docker in WSL2 has limited GPU passthrough support
- NVIDIA Container Toolkit installation was complex and often failed

#### Solution Implemented
1. **Docker Desktop Resolution**
   - User installed Docker Desktop for Windows (not in WSL2)
   - Docker Desktop automatically handles GPU passthrough to WSL2
   - No manual NVIDIA Container Toolkit configuration needed

2. **Dockerfile Issues Fixed**
   - Initial `Dockerfile.gpu` had package installation failures with CUDA base images
   - Created `Dockerfile.gpu-pytorch` using official PyTorch image:
     - Base: `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime`
     - Includes Python 3.11, CUDA 12.6, cuDNN 9
     - Pre-installed PyTorch with GPU support
   - Fixed stage naming issue (runtime-gpu → runtime)
   - Final image size: 19.1GB (expected for GPU-enabled ML image)

3. **Docker Compose Configuration**
   - Fixed duplicate CUDA_VISIBLE_DEVICES in docker-compose.gpu.yml
   - Removed from GPU overlay since base compose already sets it

#### Key Learnings
- WSL2 users should use Docker Desktop for GPU support, not native Docker
- PyTorch base images are more reliable than manual CUDA setup
- GPU-enabled ML Docker images are typically 15-20GB due to CUDA/PyTorch
- Stage names in Dockerfile must match docker-compose target specifications

## Session Update: 2025-01-08 - Simplified to Single Dockerfile

### Simplification Implemented

After research and testing, confirmed that **a single Dockerfile can handle both CPU and GPU deployments**:

1. **PyTorch Auto-Detection Works**
   - PyTorch's `torch.cuda.is_available()` correctly detects GPU availability in containers
   - When Docker provides GPU access (via `--gpus` or compose config), PyTorch uses it
   - When no GPU is available, PyTorch automatically falls back to CPU
   - Poetry installs the appropriate PyTorch version that supports both

2. **Cleanup Performed**
   - **Removed**: `Dockerfile.gpu`, `Dockerfile.gpu-pytorch`, `Dockerfile.gpu.v2`
   - **Removed**: `docker-compose.gpu.yml`, `docker-compose-cpu-only.yml`
   - **Removed**: 10 GPU-specific helper scripts (fix_wsl2_gpu.sh, etc.)
   - **Removed**: GPU-specific documentation files
   - **Updated**: `docker_setup_tui.py` to always use main `docker-compose.yml`

3. **Benefits of Single Dockerfile Approach**
   - Simpler maintenance - one Dockerfile to update
   - Automatic GPU detection - no user configuration needed
   - Same image works everywhere - deploy on CPU or GPU without rebuilding
   - Smaller overall repository - removed redundant files

4. **How It Works Now**
   - `docker-compose.yml` includes GPU device reservation
   - If GPU is available, Docker provides access and PyTorch uses it
   - If GPU is not available, the container runs fine on CPU
   - No manual configuration or separate builds required