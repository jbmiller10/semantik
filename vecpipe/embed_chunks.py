#!/usr/bin/env python3
"""
Embedding generation module (VS-020)
Converts text chunks to vectors using BGE-large-en-v1.5 model
"""

import os
import sys
import logging
import glob
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import argparse
import uuid
import torch
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 96  # Initial batch size
MIN_BATCH_SIZE = 4  # Minimum batch size on OOM
VECTOR_DIM = 1024
INPUT_DIR = "/opt/vecpipe/extract"
OUTPUT_DIR = "/var/embeddings/ingest"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class EmbeddingGenerator:
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        """Initialize the embedding model"""
        logger.info(f"Loading model {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.batch_size = BATCH_SIZE
        
        # Verify model dimensions
        test_embedding = self.model.encode("test", normalize_embeddings=True)
        if len(test_embedding) != VECTOR_DIM:
            raise ValueError(f"Model produces {len(test_embedding)}-dim vectors, expected {VECTOR_DIM}")
        
        logger.info(f"Model loaded successfully, vector dimension: {VECTOR_DIM}")
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts with OOM handling"""
        current_batch_size = self.batch_size
        
        while current_batch_size >= MIN_BATCH_SIZE:
            try:
                # Process in sub-batches if needed
                all_embeddings = []
                
                for i in range(0, len(texts), current_batch_size):
                    batch = texts[i:i + current_batch_size]
                    
                    # Encode with normalization
                    embeddings = self.model.encode(
                        batch,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        batch_size=current_batch_size
                    )
                    
                    all_embeddings.append(embeddings)
                
                # Concatenate all embeddings
                return np.vstack(all_embeddings)
                
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM with batch size {current_batch_size}, reducing to {current_batch_size // 2}")
                current_batch_size = current_batch_size // 2
                torch.cuda.empty_cache()
                
                # Update for future batches
                if current_batch_size < self.batch_size:
                    self.batch_size = current_batch_size
        
        raise RuntimeError("Unable to process batch even with minimum batch size")

def process_parquet_file(input_path: str, output_dir: str, encoder: EmbeddingGenerator) -> Optional[str]:
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
        
        # Generate embeddings
        embeddings = encoder.encode_batch(texts)
        
        # Generate unique IDs for each point
        point_ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Create output table
        output_data = {
            'id': point_ids,
            'vector': embeddings.tolist(),  # Convert to list of lists
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
        return None

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for text chunks")
    parser.add_argument('--input', '-i', default=INPUT_DIR, help='Input directory with parquet files')
    parser.add_argument('--output', '-o', default=OUTPUT_DIR, help='Output directory for embedded parquet files')
    parser.add_argument('--pattern', '-p', default='*.parquet', help='File pattern to match')
    parser.add_argument('--device', '-d', default=DEVICE, help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize encoder
    encoder = EmbeddingGenerator(device=args.device)
    
    # Find input files
    input_files = glob.glob(os.path.join(args.input, args.pattern))
    
    if not input_files:
        logger.warning(f"No files found matching pattern: {os.path.join(args.input, args.pattern)}")
        return
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Process files
    successful = 0
    for input_file in tqdm(input_files, desc="Processing files"):
        result = process_parquet_file(input_file, args.output, encoder)
        if result:
            successful += 1
    
    logger.info(f"Successfully processed {successful}/{len(input_files)} files")
    
    # Log GPU usage if available
    if args.device == "cuda":
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"Peak GPU memory usage: {memory_used:.2f} GB")

if __name__ == "__main__":
    main()