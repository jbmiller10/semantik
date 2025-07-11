#!/bin/bash
# Test script to verify INT8 works with standard build after adding gcc

echo "Testing INT8 with updated standard Dockerfile..."
echo "================================================"

echo "1. Rebuilding containers with updated Dockerfile..."
docker compose down
docker compose build --no-cache webui

echo -e "\n2. Starting containers..."
docker compose up -d

echo -e "\n3. Waiting for services to be ready..."
sleep 10

echo -e "\n4. Checking if gcc is installed..."
docker exec semantik-webui which gcc && echo "✓ gcc found" || echo "✗ gcc not found"

echo -e "\n5. Testing INT8 quantization..."
docker exec semantik-webui python -c "
import sys
sys.path.insert(0, '/app/packages')
from shared.embedding_service import check_int8_compatibility, EmbeddingService

# Check compatibility
is_compatible, msg = check_int8_compatibility()
print(f'INT8 Compatible: {is_compatible}')
print(f'Message: {msg}')

if is_compatible:
    # Test with Qwen3 model
    service = EmbeddingService()
    model = 'Qwen/Qwen3-Embedding-0.6B'
    print(f'\\nLoading {model} with INT8...')
    loaded = service.load_model(model, quantization='int8')
    print(f'Model loaded: {loaded}')
    
    if loaded:
        embeddings = service.generate_embeddings(['test text'], model, 'int8', batch_size=1, show_progress=False)
        print(f'Embeddings generated: {embeddings is not None}')
        if embeddings is not None:
            print(f'Shape: {embeddings.shape}')
            print('\\n✓ INT8 quantization works with standard build!')
else:
    print('\\n✗ INT8 still not working')
"

echo -e "\n================================================"
echo "Test complete!"