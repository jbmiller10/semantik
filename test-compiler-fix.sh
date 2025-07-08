#!/bin/bash
# Quick test script to verify INT8 fix works

echo "Testing INT8 quantization with C compiler fix..."
echo "============================================="

# Copy updated files to container
echo "1. Copying updated files to container..."
docker cp packages/webui/embedding_service.py semantik-webui:/app/packages/webui/embedding_service.py
docker cp docker-entrypoint.sh semantik-webui:/app/docker-entrypoint.sh

# Test with CC environment variable
echo -e "\n2. Testing with CC=gcc environment variable..."
docker exec -e CC=gcc -e CXX=g++ semantik-webui python -c "
import os
print(f'CC={os.environ.get(\"CC\")}')
print(f'CXX={os.environ.get(\"CXX\")}')

import sys
sys.path.insert(0, '/app/packages')
from webui.embedding_service import check_int8_compatibility

is_compatible, msg = check_int8_compatibility()
print(f'INT8 Compatible: {is_compatible}')
print(f'Message: {msg}')

if is_compatible:
    from webui.embedding_service import EmbeddingService
    service = EmbeddingService()
    model = 'Qwen/Qwen3-Embedding-0.6B'
    loaded = service.load_model(model, quantization='int8')
    print(f'Model loaded with INT8: {loaded}')
    if loaded:
        embeddings = service.generate_embeddings(['test'], model, 'int8', batch_size=1, show_progress=False)
        print(f'Embeddings generated: {embeddings is not None}')
"

echo -e "\n============================================="
echo "To apply the fix permanently:"
echo "1. Rebuild with: docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build"
echo "2. Or set environment variable: docker exec -e CC=gcc -e CXX=g++ semantik-webui ..."