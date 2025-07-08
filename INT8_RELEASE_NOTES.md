# INT8 Quantization Support in Standard Build

## Changes Made

### Dockerfile Updates
1. Added `gcc` and `g++` packages to the runtime stage of the standard Dockerfile
2. Added `CC=gcc` and `CXX=g++` environment variables
3. These changes enable bitsandbytes JIT compilation for INT8 quantization

### Code Improvements
1. Enhanced `check_int8_compatibility()` function to:
   - Auto-set CC/CXX if not present
   - Provide clearer error messages
   - Better handle C compiler detection
2. Added `ALLOW_QUANTIZATION_FALLBACK` environment variable (default: true)
3. Improved error propagation for INT8 failures
4. Better logging throughout the INT8 loading process

### Documentation Updates
1. Updated `.env.docker.example` to reflect INT8 works in both builds
2. Updated `docs/docker-cuda-fixes.md` with note about standard build support
3. Updated `QUICKFIX.md` to reflect gcc is now included
4. Clarified that CUDA build is still recommended for GPU users

## Benefits
- Users can now use INT8 quantization without needing the CUDA build
- Simpler deployment - one standard build works for all quantization modes
- CUDA build remains available for users wanting optimal GPU performance
- Clear fallback behavior with option to disable

## Breaking Changes
None - this is backwards compatible. Existing deployments will continue to work.

## Migration
Users only need to rebuild their containers to get INT8 support:
```bash
docker compose down
docker compose up -d --build
```