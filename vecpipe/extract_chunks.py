#!/usr/bin/env python3
"""
Document extraction and chunking module (VS-011)
Parses PDF/DOCX/TXT files and chunks them into embeddable segments
"""

import os
import sys
import hashlib
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional, Generator
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import argparse

# Document parsing libraries
try:
    from pypdf import PdfReader
    from docx import Document as DocxDocument
except ImportError:
    print("Installing required packages...")
    os.system("pip install pypdf python-docx")
    from pypdf import PdfReader
    from docx import Document as DocxDocument

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 600  # tokens (approx 4 chars per token)
CHUNK_OVERLAP = 200  # tokens
CHARS_PER_TOKEN = 4  # rough estimate
OUTPUT_DIR = "/opt/vecpipe/extract"
ERROR_LOG = "/tmp/error_extract.log"

def get_file_hash(filepath: str) -> str:
    """Calculate SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def extract_text_pdf(filepath: str) -> str:
    """Extract text from PDF file"""
    try:
        reader = PdfReader(filepath)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num} from {filepath}: {e}")
        
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"Failed to extract PDF {filepath}: {e}")
        raise

def extract_text_docx(filepath: str) -> str:
    """Extract text from DOCX file"""
    try:
        doc = DocxDocument(filepath)
        paragraphs = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        
        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    paragraphs.append(" | ".join(row_text))
        
        return "\n\n".join(paragraphs)
    except Exception as e:
        logger.error(f"Failed to extract DOCX {filepath}: {e}")
        raise

def extract_text_txt(filepath: str) -> str:
    """Extract text from TXT file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to extract TXT {filepath}: {e}")
        raise

def extract_text(filepath: str) -> str:
    """Extract text from file based on extension"""
    ext = Path(filepath).suffix.lower()
    
    if ext == '.pdf':
        return extract_text_pdf(filepath)
    elif ext in ['.docx', '.doc']:
        return extract_text_docx(filepath)
    elif ext in ['.txt', '.text']:
        return extract_text_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def chunk_text(text: str, doc_id: str) -> List[Dict]:
    """Split text into overlapping chunks"""
    if not text.strip():
        return []
    
    # Convert sizes to character counts
    chunk_chars = CHUNK_SIZE * CHARS_PER_TOKEN
    overlap_chars = CHUNK_OVERLAP * CHARS_PER_TOKEN
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        # Find chunk boundary
        end = start + chunk_chars
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end near chunk boundary
            search_start = max(start, end - 100)
            search_end = min(len(text), end + 100)
            search_text = text[search_start:search_end]
            
            # Find last sentence boundary
            for delimiter in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                pos = search_text.rfind(delimiter)
                if pos != -1:
                    end = search_start + pos + len(delimiter)
                    break
        
        chunk_text = text[start:end].strip()
        
        if chunk_text:
            chunks.append({
                'doc_id': doc_id,
                'chunk_id': f"{doc_id}_{chunk_id:04d}",
                'text': chunk_text,
                'start_char': start,
                'end_char': end
            })
            chunk_id += 1
        
        # Move start position with overlap
        start = end - overlap_chars
        
        # Ensure progress
        if start <= end - chunk_chars:
            start = end
    
    return chunks

def process_file(filepath: str, output_dir: str) -> Optional[str]:
    """Process a single file and save chunks to parquet"""
    try:
        # Generate document ID from file path
        doc_id = hashlib.md5(filepath.encode()).hexdigest()[:16]
        
        # Skip if already processed
        output_path = os.path.join(output_dir, f"{doc_id}.parquet")
        if os.path.exists(output_path):
            logger.info(f"Skipping already processed: {filepath}")
            return output_path
        
        # Extract text
        logger.info(f"Extracting: {filepath}")
        text = extract_text(filepath)
        
        if not text.strip():
            logger.warning(f"No text extracted from: {filepath}")
            return None
        
        # Chunk text
        chunks = chunk_text(text, doc_id)
        logger.info(f"Created {len(chunks)} chunks from: {filepath}")
        
        # Convert to parquet
        if chunks:
            df_data = {
                'doc_id': [c['doc_id'] for c in chunks],
                'chunk_id': [c['chunk_id'] for c in chunks],
                'path': [filepath] * len(chunks),
                'text': [c['text'] for c in chunks]
            }
            
            table = pa.table(df_data)
            pq.write_table(table, output_path)
            
            return output_path
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to process {filepath}: {e}")
        # Log error to file
        with open(ERROR_LOG, 'a') as f:
            f.write(f"{filepath}\t{str(e)}\n")
        return None

def process_files_parallel(file_list: List[str], output_dir: str, num_workers: int = None):
    """Process multiple files in parallel"""
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create pool and process files
    with mp.Pool(num_workers) as pool:
        # Create tasks
        tasks = [(f, output_dir) for f in file_list]
        
        # Process with progress bar
        results = []
        with tqdm(total=len(file_list), desc="Processing files") as pbar:
            for result in pool.starmap(process_file, tasks):
                results.append(result)
                pbar.update(1)
    
    # Summary
    successful = sum(1 for r in results if r is not None)
    logger.info(f"Processed {successful}/{len(file_list)} files successfully")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Extract and chunk documents")
    parser.add_argument('--input', '-i', required=True, help='Input file list (null-delimited)')
    parser.add_argument('--output', '-o', default=OUTPUT_DIR, help='Output directory for parquet files')
    parser.add_argument('--workers', '-w', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--batch-size', '-b', type=int, default=100, help='Files to process per batch')
    
    args = parser.parse_args()
    
    # Read file list
    with open(args.input, 'rb') as f:
        file_list = f.read().decode('utf-8').split('\0')
        file_list = [f for f in file_list if f]  # Remove empty strings
    
    logger.info(f"Found {len(file_list)} files to process")
    
    # Process in batches
    for i in range(0, len(file_list), args.batch_size):
        batch = file_list[i:i + args.batch_size]
        logger.info(f"Processing batch {i//args.batch_size + 1} ({len(batch)} files)")
        process_files_parallel(batch, args.output, args.workers)

if __name__ == "__main__":
    main()