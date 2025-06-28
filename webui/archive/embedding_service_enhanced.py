#!/usr/bin/env python3
"""
Enhanced Embedding service with quantization support
Handles embedding generation with various HuggingFace models and quantization
"""

import os
import sys
import logging
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import gc
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# Qwen3 specific pooling function
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with instruction for Qwen3 models"""
    return f'Instruct: {task_description}\nQuery:{query}'

class EnhancedEmbeddingService:
    """Service for generating embeddings with quantization support"""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.current_quantization = None
        self.is_qwen3_model = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Enhanced EmbeddingService initialized with device: {self.device}")
    
    def load_model(self, model_name: str, quantization: str = "float32") -> bool:
        """Load a model from HuggingFace with specified quantization"""
        try:
            # Create unique key for model+quantization combo
            model_key = f"{model_name}_{quantization}"
            
            # If same model+quantization already loaded, return
            if model_key == f"{self.current_model_name}_{self.current_quantization}":
                return True
            
            # Clear previous model to save memory
            if self.current_model is not None:
                del self.current_model
                if self.current_tokenizer is not None:
                    del self.current_tokenizer
                torch.cuda.empty_cache() if self.device == "cuda" else None
                gc.collect()
            
            logger.info(f"Loading model: {model_name} with quantization: {quantization}")
            
            # Check if this is a Qwen3 model
            self.is_qwen3_model = "Qwen3-Embedding" in model_name
            
            # Load model based on type and quantization
            if self.is_qwen3_model:
                # Load Qwen3 model using transformers directly
                logger.info("Loading Qwen3 embedding model")
                self.current_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
                
                if quantization == "int8" and self.device == "cuda":
                    try:
                        import bitsandbytes as bnb
                        from transformers import BitsAndBytesConfig
                        
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            bnb_8bit_compute_dtype=torch.float16,
                            bnb_8bit_use_double_quant=True,
                            bnb_8bit_quant_type="nf4"
                        )
                        
                        self.current_model = AutoModel.from_pretrained(
                            model_name,
                            quantization_config=quantization_config,
                            device_map="auto"
                        )
                        logger.info("Loaded Qwen3 model with INT8 quantization")
                    except ImportError:
                        logger.warning("bitsandbytes not available, loading standard model")
                        self.current_model = AutoModel.from_pretrained(model_name).to(self.device)
                        
                elif quantization == "float16" and self.device == "cuda":
                    self.current_model = AutoModel.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="auto" if self.device == "cuda" else None
                    )
                    if self.device == "cpu":
                        self.current_model = self.current_model.to(self.device)
                    logger.info("Loaded Qwen3 model in float16 precision")
                else:
                    self.current_model = AutoModel.from_pretrained(model_name).to(self.device)
                    logger.info("Loaded Qwen3 model in float32 precision")
                    
            elif quantization == "int8" and self.device == "cuda":
                # Use bitsandbytes for INT8 quantization
                try:
                    import bitsandbytes as bnb
                    from transformers import BitsAndBytesConfig
                    
                    # Configure 8-bit quantization
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float16,
                        bnb_8bit_use_double_quant=True,
                        bnb_8bit_quant_type="nf4"
                    )
                    
                    # Load model with quantization
                    self.current_model = SentenceTransformer(
                        model_name, 
                        device=self.device,
                        model_kwargs={
                            "quantization_config": quantization_config,
                            "device_map": "auto"
                        }
                    )
                    logger.info("Loaded model with INT8 quantization")
                    
                except ImportError:
                    logger.warning("bitsandbytes not available, falling back to post-quantization")
                    self.current_model = SentenceTransformer(model_name, device=self.device)
                    
            elif quantization == "float16" and self.device == "cuda":
                # Load model in float16
                self.current_model = SentenceTransformer(model_name, device=self.device)
                # Convert model to float16
                self.current_model = self.current_model.half()
                logger.info("Loaded model in float16 precision")
                
            else:
                # Load standard float32 model
                self.current_model = SentenceTransformer(model_name, device=self.device)
                logger.info("Loaded model in float32 precision")
            
            self.current_model_name = model_name
            self.current_quantization = quantization
            
            # Test model
            if self.is_qwen3_model:
                test_inputs = self.current_tokenizer(["test"], padding=True, truncation=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.current_model(**test_inputs)
                    embeddings = last_token_pool(outputs.last_hidden_state, test_inputs['attention_mask'])
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                test_embedding = embeddings[0].cpu().numpy()
            else:
                test_embedding = self.current_model.encode("test", convert_to_numpy=True)
            logger.info(f"Model loaded successfully. Embedding dimension: {len(test_embedding)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: str, quantization: str = "float32") -> Dict:
        """Get information about a model"""
        try:
            model_key = f"{model_name}_{quantization}"
            if model_key != f"{self.current_model_name}_{self.current_quantization}":
                self.load_model(model_name, quantization)
            
            if self.current_model is None:
                return {"error": "Model not loaded"}
            
            # Get embedding dimension
            if self.is_qwen3_model:
                test_inputs = self.current_tokenizer(["test"], padding=True, truncation=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.current_model(**test_inputs)
                    embeddings = last_token_pool(outputs.last_hidden_state, test_inputs['attention_mask'])
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                test_embedding = embeddings[0].cpu().numpy()
            else:
                test_embedding = self.current_model.encode("test", convert_to_numpy=True)
            
            # Calculate model size estimate
            model_size = 0
            if hasattr(self.current_model, 'parameters'):
                for param in self.current_model.parameters():
                    model_size += param.numel() * param.element_size()
            
            return {
                "model_name": model_name,
                "embedding_dim": len(test_embedding),
                "device": self.device,
                "quantization": quantization,
                "model_size_mb": model_size / 1024 / 1024,
                "max_seq_length": getattr(self.current_model, 'max_seq_length', 32768 if self.is_qwen3_model else 512),
                "is_qwen3": self.is_qwen3_model
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def generate_embeddings(self, texts: List[str], model_name: str, 
                          quantization: str = "float32",
                          batch_size: int = 32, 
                          show_progress: bool = True,
                          instruction: Optional[str] = None) -> Optional[np.ndarray]:
        """Generate embeddings for a list of texts"""
        try:
            # Load model with specified quantization if needed
            model_key = f"{model_name}_{quantization}"
            if model_key != f"{self.current_model_name}_{self.current_quantization}":
                if not self.load_model(model_name, quantization):
                    return None
            
            if not texts:
                return np.array([])
            
            logger.info(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}")
            
            # Generate embeddings based on model type
            if self.is_qwen3_model:
                # Use instruction if provided for Qwen3 models
                if instruction:
                    texts = [get_detailed_instruct(instruction, text) for text in texts]
                    logger.info(f"Using instruction: {instruction}")
                
                # Process in batches for Qwen3
                all_embeddings = []
                from tqdm import tqdm
                
                for i in tqdm(range(0, len(texts), batch_size), disable=not show_progress, desc="Encoding"):
                    batch_texts = texts[i:i + batch_size]
                    
                    # Tokenize batch
                    batch_dict = self.current_tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=32768,
                        return_tensors="pt",
                    ).to(self.device)
                    
                    # Generate embeddings
                    with torch.no_grad():
                        if quantization == "float16" and self.device == "cuda":
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                outputs = self.current_model(**batch_dict)
                        else:
                            outputs = self.current_model(**batch_dict)
                        
                        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                        embeddings = F.normalize(embeddings, p=2, dim=1)
                        all_embeddings.append(embeddings.cpu().numpy())
                
                embeddings = np.vstack(all_embeddings)
                
            elif quantization == "float16" and self.device == "cuda":
                # Ensure inputs are float16 compatible
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    embeddings = self.current_model.encode(
                        texts,
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=show_progress
                    )
            else:
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
    
    def generate_single_embedding(self, text: str, model_name: str, 
                                quantization: str = "float32",
                                instruction: Optional[str] = None) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        embeddings = self.generate_embeddings([text], model_name, quantization, 
                                            show_progress=False, instruction=instruction)
        if embeddings is not None and len(embeddings) > 0:
            return embeddings[0].tolist()
        return None

# Global instance
enhanced_embedding_service = EnhancedEmbeddingService()

# Helper function to install bitsandbytes if needed
def ensure_bitsandbytes():
    """Ensure bitsandbytes is installed for INT8 quantization"""
    try:
        import bitsandbytes
        return True
    except ImportError:
        logger.info("Installing bitsandbytes for INT8 quantization support...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "bitsandbytes"])
            return True
        except:
            logger.warning("Failed to install bitsandbytes. INT8 quantization will not be available.")
            return False

# Popular models with quantization recommendations
QUANTIZED_MODEL_INFO = {
    # Qwen3 Embedding Models
    "Qwen/Qwen3-Embedding-0.6B": {
        "dim": 1024,
        "description": "Qwen3 small model, instruction-aware (1024d)",
        "recommended_quantization": {
            "float32": "Best quality, 2.5GB VRAM",
            "float16": "Good quality, 1.3GB VRAM",
            "int8": "Efficient, 0.7GB VRAM"
        }
    },
    "Qwen/Qwen3-Embedding-4B": {
        "dim": 2560,
        "description": "Qwen3 medium model, MTEB top performer (2560d)",
        "recommended_quantization": {
            "float32": "Best quality, 16GB VRAM",
            "float16": "Recommended, 8GB VRAM",
            "int8": "Memory efficient, 4GB VRAM"
        }
    },
    "Qwen/Qwen3-Embedding-8B": {
        "dim": 4096,
        "description": "Qwen3 large model, MTEB #1 (4096d)",
        "recommended_quantization": {
            "float32": "Best quality, 32GB VRAM",
            "float16": "Recommended, 16GB VRAM",
            "int8": "Memory efficient, 8GB VRAM"
        }
    },
    # BGE Models
    "BAAI/bge-large-en-v1.5": {
        "dim": 1024,
        "description": "High-quality general purpose embeddings (1024d)",
        "recommended_quantization": {
            "float32": "Best quality, 4GB VRAM",
            "float16": "Good quality, 2GB VRAM", 
            "int8": "Slight quality loss, 1GB VRAM"
        }
    },
    "BAAI/bge-base-en-v1.5": {
        "dim": 768,
        "description": "Balanced quality and speed (768d)",
        "recommended_quantization": {
            "float32": "Best quality, 3GB VRAM",
            "float16": "Good quality, 1.5GB VRAM",
            "int8": "Slight quality loss, 0.75GB VRAM"
        }
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "dim": 384,
        "description": "Very fast, good quality (384d)",
        "recommended_quantization": {
            "float32": "Best quality, 1GB VRAM",
            "float16": "Good quality, 0.5GB VRAM",
            "int8": "Minimal benefit due to small size"
        }
    }
}

if __name__ == "__main__":
    # Test the service
    service = EnhancedEmbeddingService()
    
    # Test different quantization modes
    test_text = "This is a test sentence."
    
    for quant in ["float32", "float16", "int8"]:
        print(f"\nTesting {quant} quantization...")
        info = service.get_model_info("sentence-transformers/all-MiniLM-L6-v2", quant)
        print(f"Model info: {info}")
        
        if not info.get("error"):
            embedding = service.generate_single_embedding(test_text, 
                                                        "sentence-transformers/all-MiniLM-L6-v2", 
                                                        quant)
            if embedding:
                print(f"Embedding shape: {len(embedding)}")
                print(f"First 5 values: {embedding[:5]}")