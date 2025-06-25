#!/usr/bin/env python3
"""
Qdrant ingestion script (VS-030)
Bulk loads vectors from parquet files into Qdrant collection
"""

import os
import sys
import logging
import glob
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import pyarrow.parquet as pq
from tqdm import tqdm
import argparse
import httpx
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
QDRANT_HOST = "192.168.1.173"
QDRANT_PORT = 6333
COLLECTION_NAME = "work_docs"
BATCH_SIZE = 4000
PARALLEL_WORKERS = 4
INPUT_DIR = "/var/embeddings/ingest"
LOADED_DIR = "/var/embeddings/loaded"
REJECT_DIR = "/var/embeddings/rejects"
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds

class QdrantClient:
    """Simple HTTP client for Qdrant"""
    
    def __init__(self, host: str, port: int, timeout: int = 600):
        self.base_url = f"http://{host}:{port}"
        self.timeout = httpx.Timeout(timeout=timeout)
        self.client = httpx.Client(timeout=self.timeout)
    
    def upload_points(self, collection_name: str, points: List[Dict[str, Any]]) -> bool:
        """Upload points to collection"""
        try:
            response = self.client.put(
                f"{self.base_url}/collections/{collection_name}/points",
                json={"points": points}
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to upload points: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Get collection information"""
        try:
            response = self.client.get(f"{self.base_url}/collections/{collection_name}")
            response.raise_for_status()
            return response.json()['result']
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
    
    def close(self):
        """Close the client"""
        self.client.close()

def process_parquet_file(file_path: str, client: QdrantClient, 
                        batch_size: int = BATCH_SIZE) -> bool:
    """Process a single parquet file and upload to Qdrant"""
    try:
        logger.info(f"Processing: {file_path}")
        
        # Read parquet file
        table = pq.read_table(file_path)
        
        # Extract data
        ids = table.column('id').to_pylist()
        vectors = table.column('vector').to_pylist()
        payloads = table.column('payload').to_pylist()
        
        total_points = len(ids)
        logger.info(f"Uploading {total_points} points from {Path(file_path).name}")
        
        # Upload in batches
        successful_batches = 0
        for i in range(0, total_points, batch_size):
            batch_end = min(i + batch_size, total_points)
            
            # Prepare batch
            batch_points = []
            for j in range(i, batch_end):
                point = {
                    "id": ids[j],
                    "vector": vectors[j],
                    "payload": payloads[j]
                }
                batch_points.append(point)
            
            # Upload with retry
            success = False
            for retry in range(MAX_RETRIES):
                if client.upload_points(COLLECTION_NAME, batch_points):
                    successful_batches += 1
                    success = True
                    break
                else:
                    if retry < MAX_RETRIES - 1:
                        logger.warning(f"Retry {retry + 1}/{MAX_RETRIES} for batch {i//batch_size + 1}")
                        time.sleep(RETRY_DELAY * (2 ** retry))  # Exponential backoff
            
            if not success:
                logger.error(f"Failed to upload batch {i//batch_size + 1} after {MAX_RETRIES} retries")
                return False
        
        logger.info(f"Successfully uploaded {successful_batches} batches ({total_points} points)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def move_file(src: str, dst_dir: str):
    """Move file to destination directory"""
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, Path(src).name)
    os.rename(src, dst)
    logger.info(f"Moved {Path(src).name} to {dst_dir}")

def main():
    parser = argparse.ArgumentParser(description="Ingest embeddings into Qdrant")
    parser.add_argument('--input', '-i', default=INPUT_DIR, 
                       help='Input directory with embedded parquet files')
    parser.add_argument('--loaded', '-l', default=LOADED_DIR,
                       help='Directory for successfully loaded files')
    parser.add_argument('--rejects', '-r', default=REJECT_DIR,
                       help='Directory for rejected files')
    parser.add_argument('--pattern', '-p', default='*_embedded.parquet',
                       help='File pattern to match')
    parser.add_argument('--batch-size', '-b', type=int, default=BATCH_SIZE,
                       help='Batch size for uploads')
    parser.add_argument('--host', default=QDRANT_HOST,
                       help='Qdrant host')
    parser.add_argument('--port', type=int, default=QDRANT_PORT,
                       help='Qdrant port')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.loaded, exist_ok=True)
    os.makedirs(args.rejects, exist_ok=True)
    
    # Initialize client
    client = QdrantClient(args.host, args.port)
    
    # Get collection info
    info = client.get_collection_info(COLLECTION_NAME)
    if info:
        logger.info(f"Collection '{COLLECTION_NAME}' has {info['points_count']} points")
    else:
        logger.error(f"Collection '{COLLECTION_NAME}' not found!")
        sys.exit(1)
    
    # Find input files
    input_files = glob.glob(os.path.join(args.input, args.pattern))
    
    if not input_files:
        logger.info(f"No files found matching pattern: {os.path.join(args.input, args.pattern)}")
        return
    
    logger.info(f"Found {len(input_files)} files to ingest")
    
    # Process files
    successful = 0
    failed = 0
    
    for file_path in tqdm(input_files, desc="Ingesting files"):
        if process_parquet_file(file_path, client, args.batch_size):
            move_file(file_path, args.loaded)
            successful += 1
        else:
            move_file(file_path, args.rejects)
            failed += 1
    
    # Get final collection info
    final_info = client.get_collection_info(COLLECTION_NAME)
    if final_info:
        logger.info(f"Collection now has {final_info['points_count']} points")
    
    logger.info(f"Ingestion complete: {successful} successful, {failed} failed")
    
    # Cleanup
    client.close()

if __name__ == "__main__":
    main()