# Fix for Docker Build Hanging in Setup Wizard

## Issue
When using `make wizard` with GPU setup and selecting "rebuild images", the build process appears to hang indefinitely with no visible progress.

## Root Cause
The `_run_docker_command` method in `docker_setup_tui.py` was using `capture_output=True` for all commands, including Docker builds. This prevented users from seeing:
- Docker layer download progress
- Build step execution
- Any build output or errors

For GPU builds, this is especially problematic because:
- NVIDIA CUDA base images are 3-5GB in size
- The build process involves multiple stages
- Total build time can be 10-30 minutes on first run

## Solution
Modified `_run_docker_command` to detect build commands and run them without capturing output, allowing real-time progress display.

### Changes Made
1. **docker_setup_tui.py** (line 1185-1222):
   - Added detection for "build" commands
   - Build commands now show real-time output
   - Non-build commands still use the progress spinner
   - Added informative messages about GPU build times

2. **Added GPU build warnings** in multiple places:
   - Initial build selection
   - Build-only option
   - Service monitor rebuild option

## Testing the Fix
Run the test script to see the difference:
```bash
python test_docker_build_progress.py
```

## User Impact
- Users will now see Docker build progress in real-time
- Clear expectations set for GPU build download sizes and times
- No more "hanging" builds - users can see what's happening
- Build failures are immediately visible with error messages

## Additional Recommendations
1. For faster subsequent builds, Docker layer caching is preserved
2. Users with slow internet should be patient during first GPU build
3. Consider using `--no-cache` flag only when necessary to avoid re-downloads