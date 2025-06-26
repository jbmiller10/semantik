#!/usr/bin/env python3
"""
Parallel embedding generation module
Uses asyncio for better GPU utilization
Implements batch processing with concurrent I/O
"""

import os
import sys
import logging
import glob
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.asyncio import tqdm
import argparse
import uuid
import torch
from sentence_transformers import SentenceTransformer
import concurrent.futures
from dataclasses import dataclass
import queue
import threading
import time
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 96
MIN_BATCH_SIZE = 4
INPUT_DIR = "/opt/vecpipe/extract"
OUTPUT_DIR = "/var/embeddings/ingest"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CONCURRENT_IO = 4
EMBEDDING_QUEUE_SIZE = 10

@dataclass
class EmbeddingTask:
    """Task for embedding generation"""
    file_path: str
    texts: List[str]
    metadata: Dict[str, Any]
    future: asyncio.Future

class ParallelEmbeddingService:
    """Service for parallel embedding generation with optimized GPU usage"""
    
    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE, batch_size: int = BATCH_SIZE, mock_mode: bool = False):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.original_batch_size = batch_size  # Store original for restoration
        self.mock_mode = mock_mode
        self.model = None
        self.vector_dim = None  # Will be set after model loads
        self.embedding_queue = queue.Queue(maxsize=EMBEDDING_QUEUE_SIZE)
        self.gpu_thread = None
        self.stop_event = threading.Event()
        self.successful_batches = 0  # Track successful batches for restoration
        self._initialize()
    
    def _initialize(self):
        """Initialize model and start GPU thread"""
        if self.mock_mode:
            logger.info("Initializing embedding service in MOCK mode")
            self.vector_dim = 1024  # Default dimension for mock embeddings
        else:
            logger.info(f"Initializing embedding service with {self.model_name} on {self.device}")
        
        # Start GPU processing thread
        self.gpu_thread = threading.Thread(target=self._gpu_worker, daemon=True)
        self.gpu_thread.start()
        
        # Wait for model to load
        time.sleep(2 if not self.mock_mode else 0.1)
    
    def _gpu_worker(self):
        """Worker thread for GPU processing"""
        logger.info("Starting GPU worker thread")
        
        if not self.mock_mode:
            # Load model in GPU thread
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get model dimensions dynamically
            self.vector_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully on {self.device}, embedding dimension: {self.vector_dim}")
        
        # Process tasks from queue
        while not self.stop_event.is_set():
            try:
                # Get task with timeout
                task = self.embedding_queue.get(timeout=1.0)
                
                if task is None:  # Shutdown signal
                    break
                
                # Generate embeddings
                try:
                    embeddings = self._generate_embeddings_batch(task.texts)
                    task.future.set_result(embeddings)
                except Exception as e:
                    task.future.set_exception(e)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"GPU worker error: {e}")
    
    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings with OOM handling and batch size restoration"""
        if self.mock_mode:
            # Generate mock embeddings based on text hash
            embeddings = []
            for text in texts:
                # Create deterministic embedding from text hash
                text_hash = hashlib.sha256(text.encode()).digest()
                # Convert hash to float array (using first bytes)
                embedding = np.frombuffer(text_hash, dtype=np.float32)[:self.vector_dim // 4]
                # Repeat and normalize to get correct dimension
                embedding = np.tile(embedding, (self.vector_dim // len(embedding) + 1))[:self.vector_dim]
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            return np.array(embeddings)
        
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
                
                # Success! Track it for potential restoration
                if current_batch_size == self.batch_size:
                    self.successful_batches += 1
                    # Try to restore batch size after 10 successful batches
                    if self.successful_batches > 10 and self.batch_size < self.original_batch_size:
                        new_size = min(self.batch_size * 2, self.original_batch_size)
                        logger.info(f"Restoring batch size from {self.batch_size} to {new_size}")
                        self.batch_size = new_size
                        self.successful_batches = 0
                
                # Concatenate all embeddings
                return np.vstack(all_embeddings)
                
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM with batch size {current_batch_size}, reducing to {current_batch_size // 2}")
                current_batch_size = current_batch_size // 2
                torch.cuda.empty_cache()
                
                # Update for future batches and reset success counter
                if current_batch_size < self.batch_size:
                    self.batch_size = current_batch_size
                    self.successful_batches = 0
        
        raise RuntimeError("Unable to process batch even with minimum batch size")
    
    async def generate_embeddings_async(self, texts: List[str]) -> np.ndarray:
        """Async wrapper for embedding generation"""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        # Create task and add to queue
        task = EmbeddingTask(
            file_path="",
            texts=texts,
            metadata={},
            future=future
        )
        
        # Add to queue (this might block if queue is full)
        await loop.run_in_executor(None, self.embedding_queue.put, task)
        
        # Wait for result
        return await future
    
    def shutdown(self):
        """Shutdown the service"""
        logger.info("Shutting down embedding service")
        self.stop_event.set()
        self.embedding_queue.put(None)  # Shutdown signal
        if self.gpu_thread:
            self.gpu_thread.join(timeout=5)

async def read_parquet_async(file_path: str) -> Dict[str, Any]:
    """Async read parquet file"""
    loop = asyncio.get_event_loop()
    
    def _read():
        table = pq.read_table(file_path)
        return {
            'doc_ids': table.column('doc_id').to_pylist(),
            'chunk_ids': table.column('chunk_id').to_pylist(),
            'paths': table.column('path').to_pylist(),
            'texts': table.column('text').to_pylist(),
            'file_path': file_path
        }
    
    return await loop.run_in_executor(None, _read)

async def write_parquet_async(output_path: str, data: Dict[str, Any]):
    """Async write parquet file"""
    loop = asyncio.get_event_loop()
    
    def _write():
        output_table = pa.table(data)
        pq.write_table(output_table, output_path)
    
    await loop.run_in_executor(None, _write)

async def process_file_async(file_path: str, output_dir: str, embedding_service: ParallelEmbeddingService) -> Optional[str]:
    """Process a single file asynchronously"""
    try:
        # Generate output filename
        filename = Path(file_path).stem
        output_path = os.path.join(output_dir, f"{filename}_embedded.parquet")
        
        # Skip if already processed
        if os.path.exists(output_path):
            logger.info(f"Skipping already processed: {file_path}")
            return output_path
        
        logger.info(f"Processing: {file_path}")
        
        # Read input asynchronously
        data = await read_parquet_async(file_path)
        
        texts = data['texts']
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings asynchronously
        embeddings = await embedding_service.generate_embeddings_async(texts)
        
        # Generate unique IDs for each point
        point_ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Prepare output data
        output_data = {
            'id': point_ids,
            'vector': embeddings.tolist(),
            'payload': [
                {
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'path': path
                }
                for doc_id, chunk_id, path in zip(data['doc_ids'], data['chunk_ids'], data['paths'])
            ]
        }
        
        # Write output asynchronously
        await write_parquet_async(output_path, output_data)
        
        logger.info(f"Saved embeddings to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return None

async def process_files_parallel(file_paths: List[str], output_dir: str, embedding_service: ParallelEmbeddingService):
    """Process multiple files in parallel"""
    # Create semaphore to limit concurrent I/O operations
    io_semaphore = asyncio.Semaphore(MAX_CONCURRENT_IO)
    
    async def process_with_limit(file_path):
        async with io_semaphore:
            return await process_file_async(file_path, output_dir, embedding_service)
    
    # Process all files concurrently
    tasks = [process_with_limit(fp) for fp in file_paths]
    
    # Use tqdm for progress tracking
    results = []
    for coro in tqdm.as_completed(tasks, desc="Processing files"):
        result = await coro
        results.append(result)
    
    return results

async def main_async(args):
    """Main async function"""
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize embedding service
    embedding_service = ParallelEmbeddingService(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        mock_mode=args.mock
    )
    
    try:
        # Find input files
        input_files = glob.glob(os.path.join(args.input, args.pattern))
        
        if not input_files:
            logger.warning(f"No files found matching pattern: {os.path.join(args.input, args.pattern)}")
            return
        
        logger.info(f"Found {len(input_files)} files to process")
        
        # Process files in parallel
        results = await process_files_parallel(input_files, args.output, embedding_service)
        
        # Count successes
        successful = sum(1 for r in results if r is not None)
        logger.info(f"Successfully processed {successful}/{len(input_files)} files")
        
        # Log GPU usage if available
        if args.device == "cuda":
            memory_used = torch.cuda.max_memory_allocated() / 1e9
            logger.info(f"Peak GPU memory usage: {memory_used:.2f} GB")
    
    finally:
        # Shutdown embedding service
        embedding_service.shutdown()

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings in parallel")
    parser.add_argument('--input', '-i', default=INPUT_DIR, help='Input directory with parquet files')
    parser.add_argument('--output', '-o', default=OUTPUT_DIR, help='Output directory for embedded parquet files')
    parser.add_argument('--pattern', '-p', default='*.parquet', help='File pattern to match')
    parser.add_argument('--device', '-d', default=DEVICE, help='Device to use (cuda/cpu)')
    parser.add_argument('--model', '-m', default=MODEL_NAME, help='Model name')
    parser.add_argument('--batch-size', '-b', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--mock', action='store_true', help='Use mock embeddings for testing (no GPU required)')
    
    args = parser.parse_args()
    
    # Run async main
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()