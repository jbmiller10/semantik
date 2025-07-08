#!/bin/bash
# Script to pre-download models for Semantik

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
MODEL_DIR="${HF_CACHE_DIR:-./models}"
MODEL_NAME="${DEFAULT_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"

echo -e "${GREEN}Semantik Model Downloader${NC}"
echo "========================="
echo

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --model NAME    Model to download (default: Qwen/Qwen3-Embedding-0.6B)"
            echo "  --dir PATH      Directory to store models (default: ./models)"
            echo "  --help          Show this help message"
            echo
            echo "Available models:"
            echo "  - Qwen/Qwen3-Embedding-0.6B (1.2GB) - Recommended for most use cases"
            echo "  - Qwen/Qwen3-Embedding-4B (8GB) - Better quality, more VRAM"
            echo "  - Qwen/Qwen3-Embedding-8B (16GB) - Best quality, requires 8GB+ VRAM"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Create model directory
echo -e "${YELLOW}Creating model directory: $MODEL_DIR${NC}"
mkdir -p "$MODEL_DIR"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Build the image if it doesn't exist
if ! docker images | grep -q "semantik-webui"; then
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker compose build webui
fi

# Download the model
echo -e "${GREEN}Downloading model: $MODEL_NAME${NC}"
echo "This may take several minutes depending on model size and internet speed..."
echo

docker run --rm -it \
    -v "$(realpath "$MODEL_DIR")":/app/.cache/huggingface \
    -e HF_HOME=/app/.cache/huggingface \
    semantik-webui \
    python -c "
import sys
from transformers import AutoModel, AutoTokenizer
print(f'Downloading model: $MODEL_NAME')
try:
    # Download tokenizer
    print('Downloading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME')
    print('✓ Tokenizer downloaded')
    
    # Download model
    print('Downloading model weights...')
    model = AutoModel.from_pretrained('$MODEL_NAME')
    print('✓ Model downloaded')
    
    # Get model info
    print(f'\\nModel info:')
    print(f'- Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B')
    print(f'- Model type: {model.config.model_type}')
    
except Exception as e:
    print(f'\\nError downloading model: {e}')
    sys.exit(1)

print('\\n✅ Model downloaded successfully!')
"

if [ $? -eq 0 ]; then
    echo
    echo -e "${GREEN}✅ Model downloaded successfully!${NC}"
    echo
    echo "Next steps:"
    echo "1. Add to your .env file:"
    echo "   HF_CACHE_DIR=$MODEL_DIR"
    echo "   HF_HUB_OFFLINE=true"
    echo
    echo "2. Start Semantik:"
    echo "   docker compose up -d"
else
    echo
    echo -e "${RED}❌ Model download failed${NC}"
    exit 1
fi