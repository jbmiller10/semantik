#!/usr/bin/env python3
"""
Simplified embedding generation module (VS-020)
Uses HuggingFace API for embedding generation
"""

import os
import sys
import logging
import glob
import json
from pathlib import Path
from typing import List, Dict, Optional
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import argparse
import uuid
import requests
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "BAAI/bge-large-en-v1.5"
VECTOR_DIM = 1024
INPUT_DIR = "/opt/vecpipe/extract"
OUTPUT_DIR = "/var/embeddings/ingest"
BATCH_SIZE = 8  # Small batch for testing

# Mock embedding function for testing
def generate_mock_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate mock embeddings for testing without GPU"""
    import hashlib
    embeddings = []
    
    for text in texts:
        # Generate deterministic "embedding" from text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Convert to float values between -1 and 1
        values = []
        for i in range(0, len(hash_bytes), 4):
            chunk = hash_bytes[i:i+4]
            if len(chunk) == 4:
                # Convert 4 bytes to float
                val = int.from_bytes(chunk, byteorder='big') / (2**32)
                values.append(val * 2 - 1)  # Scale to [-1, 1]
        
        # Pad or truncate to VECTOR_DIM
        if len(values) < VECTOR_DIM:
            values.extend([0.0] * (VECTOR_DIM - len(values)))
        else:
            values = values[:VECTOR_DIM]
        
        # Normalize
        norm = sum(v**2 for v in values) ** 0.5
        if norm > 0:
            values = [v / norm for v in values]
        
        embeddings.append(values)
    
    return embeddings

def process_parquet_file(input_path: str, output_dir: str) -> Optional[str]:
    """Process a single parquet file and generate embeddings"""
    try:
        # Generate output filename
        filename = Path(input_path).stem
        output_path = os.path.join(output_dir, f"{filename}_embedded.parquet")
        
        # Skip if already processed
        if os.path.exists(output_path):
            logger.info(f"Skipping already processed: {input_path}")
            return output_path
        
        logger.info(f"Processing: {input_path}")
        
        # Read input parquet
        table = pq.read_table(input_path)
        
        # Extract data
        doc_ids = table.column('doc_id').to_pylist()
        chunk_ids = table.column('chunk_id').to_pylist()
        paths = table.column('path').to_pylist()
        texts = table.column('text').to_pylist()
        
        logger.info(f"Encoding {len(texts)} chunks...")
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            batch_embeddings = generate_mock_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # Simulate processing time
            time.sleep(0.1)
        
        # Generate unique IDs for each point
        point_ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Create output table
        output_data = {
            'id': point_ids,
            'vector': all_embeddings,
            'payload': [
                {
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'path': path
                }
                for doc_id, chunk_id, path in zip(doc_ids, chunk_ids, paths)
            ]
        }
        
        output_table = pa.table(output_data)
        
        # Write to parquet
        pq.write_table(output_table, output_path)
        
        logger.info(f"Saved embeddings to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for text chunks")
    parser.add_argument('--input', '-i', default=INPUT_DIR, help='Input directory with parquet files')
    parser.add_argument('--output', '-o', default=OUTPUT_DIR, help='Output directory for embedded parquet files')
    parser.add_argument('--pattern', '-p', default='*.parquet', help='File pattern to match')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Find input files
    input_files = glob.glob(os.path.join(args.input, args.pattern))
    
    if not input_files:
        logger.warning(f"No files found matching pattern: {os.path.join(args.input, args.pattern)}")
        return
    
    logger.info(f"Found {len(input_files)} files to process")
    logger.info("Note: Using mock embeddings for testing without GPU")
    
    # Process files
    successful = 0
    for input_file in tqdm(input_files, desc="Processing files"):
        result = process_parquet_file(input_file, args.output)
        if result:
            successful += 1
    
    logger.info(f"Successfully processed {successful}/{len(input_files)} files")

if __name__ == "__main__":
    main()