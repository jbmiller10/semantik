# Semantik Embedding Generation Diagnostics

This document describes the diagnostic tools created to help troubleshoot why embeddings aren't being generated in Semantik despite the TokenChunker fix.

## Understanding the Embedding Generation Flow

1. **Collection Creation**: Creates a Qdrant vector collection (INDEX operation)
2. **Document Registration**: Scans files and registers them in the database (APPEND operation)
3. **Text Extraction**: Extracts text content from documents
4. **Chunking**: Splits text into chunks using TokenChunker
5. **Embedding Generation**: Calls vecpipe service to generate embeddings
6. **Vector Storage**: Uploads embeddings to Qdrant

## Diagnostic Scripts

### 1. Check Collection Status (`check_collection_status.py`)

Shows the current state of all collections, operations, and documents.

```bash
python check_collection_status.py
```

This will display:
- All collections and their status (PENDING, READY, PROCESSING, ERROR, DEGRADED)
- Recent operations for each collection
- Failed operations with error messages
- Document processing status summary
- Recent document failures

### 2. Check Vecpipe Status (`check_vecpipe_status.py`)

Tests if the vecpipe service is running and can generate embeddings.

```bash
python check_vecpipe_status.py
```

This will:
- Check if vecpipe service is accessible
- Test embedding generation with sample text
- Verify all required services are running

### 3. Test Document Processing (`test_document_processing.py`)

Tests the document processing pipeline for a specific file.

```bash
# Test with a specific file
python test_document_processing.py /path/to/your/file.txt

# Or run without arguments to test with sample files
python test_document_processing.py
```

This will:
- Test text extraction from the file
- Test chunking with TokenChunker
- Diagnose any issues in the pipeline

### 4. Manual Embedding Generation (`manual_embed_collection.py`)

Manually process documents and generate embeddings for a collection, bypassing Celery.

```bash
# Process by collection name
python manual_embed_collection.py "My Collection"

# Or by collection ID
python manual_embed_collection.py "550e8400-e29b-41d4-a716-446655440000"
```

This will:
- Process all unprocessed documents in the collection
- Generate embeddings directly without using Celery
- Update document and collection status

## Common Issues and Solutions

### Issue: Vecpipe Service Not Running

**Symptoms**: 
- `check_vecpipe_status.py` shows vecpipe is not accessible
- Operations fail with connection errors

**Solution**:
```bash
# Check if vecpipe container is running
docker ps | grep vecpipe

# If not running, start it
docker compose --profile backend up -d vecpipe

# Check logs
docker logs semantik-vecpipe-1
```

### Issue: Operations Stuck in PENDING/PROCESSING

**Symptoms**:
- `check_collection_status.py` shows operations in PENDING or PROCESSING state
- No progress on document processing

**Solution**:
```bash
# Check if Celery workers are running
docker ps | grep worker

# Check worker logs
docker logs semantik-worker-1

# Restart workers if needed
docker compose --profile backend restart worker
```

### Issue: Text Extraction Failures

**Symptoms**:
- Documents show status FAILED with "No text content extracted"
- `test_document_processing.py` fails to extract text

**Possible Causes**:
- Unsupported file format
- Missing dependencies for specific file types
- Corrupted files

### Issue: No Chunks Created

**Symptoms**:
- Documents show status FAILED with "No chunks created"
- Text is extracted but chunking fails

**Possible Causes**:
- Text too short for chunk size
- TokenChunker configuration issues
- Empty or whitespace-only text

## Quick Troubleshooting Steps

1. **Check overall status**:
   ```bash
   python check_collection_status.py
   ```

2. **Verify services are running**:
   ```bash
   python check_vecpipe_status.py
   ```

3. **Test with a sample document**:
   ```bash
   python test_document_processing.py /path/to/test/file.txt
   ```

4. **If everything seems OK but embeddings aren't generated, try manual processing**:
   ```bash
   python manual_embed_collection.py "Your Collection Name"
   ```

## Important Notes

- The manual embedding script (`manual_embed_collection.py`) bypasses the Celery task queue, which can help identify if the issue is with task processing or the embedding pipeline itself.
- Always check the vecpipe service first, as it's required for embedding generation.
- Document processing is done asynchronously through Celery workers, so there may be a delay between creating a collection and embeddings being generated.
- Check Docker logs for detailed error messages if scripts report failures.