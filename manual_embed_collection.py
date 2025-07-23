#!/usr/bin/env python3
"""
Manually process documents and generate embeddings for a collection.
This bypasses the Celery task queue to help diagnose issues.
"""

import asyncio
import sys
import uuid
from datetime import datetime

# Add the packages directory to the Python path
sys.path.insert(0, '/home/dockertest/semantik/packages')

from shared.database import pg_connection_manager
from shared.database.database import AsyncSessionLocal
from shared.database.models import Collection, Document, DocumentStatus
from shared.text_extraction.text_extractor import extract_text_and_serialize
from shared.chunking.token_chunker import TokenChunker
from shared.database.qdrant_manager import qdrant_manager
from qdrant_client.models import PointStruct
from sqlalchemy import select
import httpx


async def process_document(document, collection, session):
    """Process a single document and generate embeddings."""
    print(f"\nProcessing: {document.file_path}")
    print("-" * 50)
    
    try:
        # Extract text
        print("  Extracting text...")
        text_blocks = extract_text_and_serialize(document.file_path)
        
        if not text_blocks:
            print("  ❌ No text extracted")
            await session.execute(
                document.update().values(
                    status=DocumentStatus.FAILED,
                    error_message="No text content extracted",
                    updated_at=datetime.utcnow()
                ).where(Document.id == document.id)
            )
            await session.commit()
            return False
        
        print(f"  ✓ Extracted {len(text_blocks)} text block(s)")
        
        # Create chunks
        chunker = TokenChunker(
            chunk_size=collection.chunk_size or 1000,
            chunk_overlap=collection.chunk_overlap or 200
        )
        
        all_chunks = []
        for text, metadata in text_blocks:
            if not text.strip():
                continue
            chunks = chunker.chunk_text(text, document.id, metadata)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            print("  ❌ No chunks created")
            await session.execute(
                document.update().values(
                    status=DocumentStatus.FAILED,
                    error_message="No chunks created from text",
                    updated_at=datetime.utcnow()
                ).where(Document.id == document.id)
            )
            await session.commit()
            return False
        
        print(f"  ✓ Created {len(all_chunks)} chunks")
        
        # Generate embeddings via vecpipe
        texts = [chunk["text"] for chunk in all_chunks]
        
        embed_request = {
            "texts": texts,
            "model_name": collection.embedding_model or "Qwen/Qwen3-Embedding-0.6B",
            "quantization": collection.quantization or "float16",
            "instruction": None,
            "batch_size": 32
        }
        
        print(f"  Generating embeddings for {len(texts)} chunks...")
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.post("http://vecpipe:8000/embed", json=embed_request)
                
                if response.status_code != 200:
                    print(f"  ❌ Embedding API error: {response.status_code} - {response.text}")
                    raise Exception(f"Embedding API returned {response.status_code}")
                
                embed_response = response.json()
                embeddings = embed_response.get("embeddings")
                
                if not embeddings:
                    print("  ❌ No embeddings returned")
                    raise Exception("No embeddings in API response")
                
                print(f"  ✓ Generated {len(embeddings)} embeddings")
                
            except httpx.RequestError as e:
                print(f"  ❌ Cannot connect to vecpipe: {e}")
                raise
        
        # Prepare points for Qdrant
        points = []
        for i, chunk in enumerate(all_chunks):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload={
                    "collection_id": collection.id,
                    "doc_id": document.id,
                    "chunk_id": chunk["chunk_id"],
                    "path": document.file_path,
                    "content": chunk["text"],
                    "metadata": chunk.get("metadata", {}),
                }
            )
            points.append(point)
        
        # Upload to Qdrant
        print(f"  Uploading {len(points)} vectors to Qdrant...")
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            
            # Convert PointStruct objects to dict format for API
            points_data = []
            for point in batch:
                points_data.append({
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload
                })
            
            upsert_request = {
                "collection_name": collection.vector_store_name,
                "points": points_data,
                "wait": True
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post("http://vecpipe:8000/upsert", json=upsert_request)
                
                if response.status_code != 200:
                    print(f"  ❌ Upsert error: {response.status_code} - {response.text}")
                    raise Exception(f"Upsert API returned {response.status_code}")
        
        print(f"  ✓ Uploaded all vectors")
        
        # Update document status
        await session.execute(
            document.update().values(
                status=DocumentStatus.COMPLETED,
                chunk_count=len(all_chunks),
                updated_at=datetime.utcnow()
            ).where(Document.id == document.id)
        )
        await session.commit()
        
        print(f"  ✓ Document processed successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        await session.execute(
            document.update().values(
                status=DocumentStatus.FAILED,
                error_message=str(e),
                updated_at=datetime.utcnow()
            ).where(Document.id == document.id)
        )
        await session.commit()
        return False


async def process_collection(collection_name_or_id):
    """Process all documents in a collection."""
    await pg_connection_manager.initialize()
    
    async with AsyncSessionLocal() as session:
        # Find collection
        stmt = select(Collection).where(
            (Collection.name == collection_name_or_id) | 
            (Collection.id == collection_name_or_id)
        )
        result = await session.execute(stmt)
        collection = result.scalar_one_or_none()
        
        if not collection:
            print(f"❌ Collection '{collection_name_or_id}' not found")
            return
        
        print(f"\nProcessing collection: {collection.name}")
        print(f"ID: {collection.id}")
        print(f"Vector store: {collection.vector_store_name}")
        print(f"Status: {collection.status.value if collection.status else 'None'}")
        print("=" * 80)
        
        # Check if Qdrant collection exists
        if not collection.vector_store_name:
            print("❌ Collection has no vector_store_name. Cannot process.")
            return
        
        # Get all documents in collection
        stmt = select(Document).where(Document.collection_id == collection.id)
        result = await session.execute(stmt)
        documents = result.scalars().all()
        
        if not documents:
            print("No documents found in collection")
            return
        
        print(f"\nFound {len(documents)} documents")
        
        # Group by status
        status_counts = {}
        for doc in documents:
            status = doc.status.value if doc.status else 'None'
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("\nDocument status:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
        
        # Process unprocessed documents
        unprocessed = [d for d in documents if d.status != DocumentStatus.COMPLETED]
        
        if not unprocessed:
            print("\nAll documents already processed")
            return
        
        print(f"\nProcessing {len(unprocessed)} unprocessed documents...")
        
        success_count = 0
        failed_count = 0
        
        for doc in unprocessed:
            success = await process_document(doc, collection, session)
            if success:
                success_count += 1
            else:
                failed_count += 1
        
        print("\n" + "=" * 80)
        print(f"Processing complete:")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {failed_count}")
        
        # Update collection vector count
        if success_count > 0:
            # Get actual vector count from Qdrant
            try:
                qdrant_client = qdrant_manager.get_client()
                collection_info = qdrant_client.get_collection(collection.vector_store_name)
                vector_count = collection_info.vectors_count
                
                await session.execute(
                    Collection.update().values(
                        vector_count=vector_count,
                        updated_at=datetime.utcnow()
                    ).where(Collection.id == collection.id)
                )
                await session.commit()
                
                print(f"\nUpdated collection vector count: {vector_count}")
            except Exception as e:
                print(f"\n⚠️  Could not update vector count: {e}")


async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python manual_embed_collection.py <collection_name_or_id>")
        print("\nThis script manually processes documents and generates embeddings")
        print("for a collection, bypassing the Celery task queue.")
        return
    
    collection_id = sys.argv[1]
    
    try:
        await process_collection(collection_id)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await pg_connection_manager.close()


if __name__ == "__main__":
    asyncio.run(main())