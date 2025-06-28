#!/usr/bin/env python3
"""
Embedding service with HuggingFace model support
Handles embedding generation for the web UI
"""

import os
import logging
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings with various HuggingFace models"""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.current_model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"EmbeddingService initialized with device: {self.device}")
    
    def load_model(self, model_name: str) -> bool:
        """Load a model from HuggingFace"""
        try:
            # If model already loaded, return
            if model_name == self.current_model_name:
                return True
            
            # Clear previous model to save memory
            if self.current_model is not None:
                del self.current_model
                torch.cuda.empty_cache() if self.device == "cuda" else None
                gc.collect()
            
            logger.info(f"Loading model: {model_name}")
            
            # Load new model
            self.current_model = SentenceTransformer(model_name, device=self.device)
            self.current_model_name = model_name
            
            # Test model
            test_embedding = self.current_model.encode("test", convert_to_numpy=True)
            logger.info(f"Model loaded successfully. Embedding dimension: {len(test_embedding)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a model"""
        try:
            if model_name != self.current_model_name:
                self.load_model(model_name)
            
            if self.current_model is None:
                return {"error": "Model not loaded"}
            
            # Get embedding dimension
            test_embedding = self.current_model.encode("test", convert_to_numpy=True)
            
            return {
                "model_name": model_name,
                "embedding_dim": len(test_embedding),
                "device": self.device,
                "max_seq_length": getattr(self.current_model, 'max_seq_length', 512)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def generate_embeddings(self, texts: List[str], model_name: str, 
                          batch_size: int = 32, show_progress: bool = True) -> Optional[np.ndarray]:
        """Generate embeddings for a list of texts"""
        try:
            # Load model if needed
            if model_name != self.current_model_name:
                if not self.load_model(model_name):
                    return None
            
            if not texts:
                return np.array([])
            
            logger.info(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}")
            
            # Handle memory constraints
            if self.device == "cuda":
                # Adjust batch size based on available memory
                try:
                    embeddings = self.current_model.encode(
                        texts,
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=show_progress
                    )
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM with batch size {batch_size}, reducing to {batch_size // 2}")
                    torch.cuda.empty_cache()
                    batch_size = max(1, batch_size // 2)
                    
                    embeddings = self.current_model.encode(
                        texts,
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=show_progress
                    )
            else:
                # CPU processing
                embeddings = self.current_model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=show_progress
                )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return None
    
    def generate_single_embedding(self, text: str, model_name: str) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        embeddings = self.generate_embeddings([text], model_name, show_progress=False)
        if embeddings is not None and len(embeddings) > 0:
            return embeddings[0].tolist()
        return None

# Global instance
embedding_service = EmbeddingService()

# Popular models with their dimensions
POPULAR_MODELS = {
    # Qwen3 Embedding Models
    "Qwen/Qwen3-Embedding-0.6B": {
        "dim": 1024,
        "description": "Qwen3 small model, instruction-aware (1024d)"
    },
    "Qwen/Qwen3-Embedding-4B": {
        "dim": 2560,
        "description": "Qwen3 medium model, MTEB top performer (2560d)"
    },
    "Qwen/Qwen3-Embedding-8B": {
        "dim": 4096,
        "description": "Qwen3 large model, MTEB #1 (4096d)"
    },
    # BGE Models
    "BAAI/bge-large-en-v1.5": {
        "dim": 1024,
        "description": "High-quality general purpose embeddings (1024d)"
    },
    "BAAI/bge-base-en-v1.5": {
        "dim": 768,
        "description": "Balanced quality and speed (768d)"
    },
    "BAAI/bge-small-en-v1.5": {
        "dim": 384,
        "description": "Fast and efficient (384d)"
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "dim": 384,
        "description": "Very fast, good quality (384d)"
    },
    "sentence-transformers/all-mpnet-base-v2": {
        "dim": 768,
        "description": "High quality general purpose (768d)"
    },
    "thenlper/gte-large": {
        "dim": 1024,
        "description": "State-of-the-art quality (1024d)"
    },
    "thenlper/gte-base": {
        "dim": 768,
        "description": "Good balance of quality and speed (768d)"
    },
    "intfloat/e5-large-v2": {
        "dim": 1024,
        "description": "Excellent for semantic search (1024d)"
    },
    "intfloat/e5-base-v2": {
        "dim": 768,
        "description": "Good for semantic search (768d)"
    }
}

def test_embedding_service():
    """Test the embedding service"""
    service = EmbeddingService()
    
    # Test loading a model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Testing model: {model_name}")
    
    if service.load_model(model_name):
        # Test single embedding
        embedding = service.generate_single_embedding("Hello world", model_name)
        print(f"Single embedding shape: {len(embedding) if embedding else 'Failed'}")
        
        # Test batch embeddings
        texts = ["First document", "Second document", "Third document"]
        embeddings = service.generate_embeddings(texts, model_name)
        if embeddings is not None:
            print(f"Batch embeddings shape: {embeddings.shape}")
            print(f"Embeddings are normalized: {np.allclose(np.linalg.norm(embeddings[0]), 1.0)}")
    else:
        print("Failed to load model")

if __name__ == "__main__":
    test_embedding_service()