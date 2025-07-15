# Troubleshooting Guide

This guide helps you resolve common issues when running Semantik.

## CUDA Library Issues on Linux

If you encounter errors like `libcudart.so: cannot open shared object file` when running Semantik, you need to configure your CUDA library path.

### Symptoms
- Error messages containing `libcudart.so` not found
- CUDA initialization failures
- GPU not being detected despite having CUDA installed

### Solution

The exact solution depends on how you installed CUDA and your Python environment:

#### 1. System-wide CUDA Installation
If you installed CUDA using your system's package manager or NVIDIA's installer:

```bash
# Common CUDA installation paths
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# or
export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
```

#### 2. Python Virtual Environment with pip-installed CUDA
If you're using Poetry/pip and installed CUDA packages via pip:

```bash
# Find your Python site-packages directory
python -c "import site; print(site.getsitepackages()[0])"

# Then export the CUDA library path (adjust the path based on the output above)
export LD_LIBRARY_PATH=/path/to/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
```

#### 3. Conda/Mamba Environment
If you're using Conda or Mamba:

```bash
# The CUDA libraries are usually in your conda environment
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### Making the Change Permanent

To avoid setting this every time, add the export command to your shell configuration:

```bash
# For bash
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# For zsh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.zshrc
source ~/.zshrc
```

### Verifying Your Setup

After setting the library path, verify CUDA is working:

```bash
# Check if CUDA libraries are found
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Alternative: Creating Symlinks

Some CUDA installations have versioned libraries (e.g., `libcudart.so.12`) but applications look for the unversioned name. You can create symlinks:

```bash
# Example for CUDA 12
cd /path/to/cuda/lib
sudo ln -sf libcudart.so.12 libcudart.so
```

**Note:** Use symlinks with caution and only if setting `LD_LIBRARY_PATH` doesn't work.

## Other Common Issues

### Port Already in Use

If you see errors about ports 8000 or 8080 being in use:

```bash
# Find and kill processes using these ports
lsof -ti:8000 | xargs kill -9
lsof -ti:8080 | xargs kill -9
```

### Poetry Installation Issues

If Poetry commands fail:

```bash
# Ensure you're in the project root
cd /path/to/semantik

# Install dependencies
poetry install

# If that fails, try updating Poetry
poetry self update
```

### Model Download Failures

If models fail to download:

1. Check your internet connection
2. Ensure you have sufficient disk space in your home directory
3. Try manually downloading models:
   ```bash
   # Models are stored in ~/.cache/huggingface
   ls -la ~/.cache/huggingface
   ```

### Out of Memory Errors

If you encounter GPU out of memory errors:

1. Use a smaller model or enable quantization in the WebUI settings
2. Close other GPU-intensive applications
3. Monitor GPU memory usage:
   ```bash
   nvidia-smi
   ```
4. Set memory limits in configuration:
   ```bash
   export MODEL_MAX_MEMORY_GB=8  # Limit GPU memory usage
   ```

### JWT Secret Key Missing in Production

If you see "JWT_SECRET_KEY not set in production environment":

1. Generate a secure key:
   ```bash
   openssl rand -hex 32
   ```
2. Set it in your `.env` file:
   ```bash
   JWT_SECRET_KEY=your-generated-key
   ```
3. Never use the default key in production

### Frontend Testing Commands Not Working

If `make frontend-test` fails:

The frontend testing infrastructure is not currently configured. To run frontend tests:
1. Tests need to be implemented in the React application
2. Testing libraries (Jest, React Testing Library) need to be installed

### Job Cancellation Not Working

If job cancellation fails:

This is a known limitation - Celery task IDs are not currently persisted. Jobs can be monitored but not cancelled once started.

### Model Loading Timeouts

If models timeout during first load:

1. Use the preload endpoint before searching:
   ```bash
   curl -X POST http://localhost:8080/api/preload_model \
     -H "Content-Type: application/json" \
     -d '{"model_name": "Qwen/Qwen3-Embedding-0.6B"}'
   ```
2. Increase timeout settings if needed
3. Models are cached after first download

### WebSocket Connection Failed

If real-time updates aren't working:

1. Check browser console for errors
2. Ensure your reverse proxy supports WebSocket:
   ```nginx
   # nginx.conf
   location /ws/ {
       proxy_pass http://localhost:8080;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection "upgrade";
   }
   ```
3. Check firewall rules for WebSocket connections

### Document Preview Not Working

If documents fail to preview:

1. Check file permissions on the document
2. Ensure the document path is accessible to the container
3. For PPTX files, temporary image extraction may take time
4. Check logs for extraction errors

### Rate Limiting Errors

If you encounter 429 Too Many Requests:

1. Default rate limit is 60 requests per minute
2. Implement client-side rate limiting
3. Use batch endpoints when available
4. Configure `RATE_LIMIT_PER_MINUTE` if needed

For additional help, please open an issue on our GitHub repository with:
- Your system specifications (OS, GPU model, CUDA version)
- The complete error message
- Steps to reproduce the issue