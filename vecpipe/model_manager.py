"""
Model lifecycle manager with lazy loading and automatic unloading
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from webui.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages embedding model lifecycle with lazy loading and automatic unloading"""
    
    def __init__(self, unload_after_seconds: int = 300):  # 5 minutes default
        """
        Initialize the model manager
        
        Args:
            unload_after_seconds: Unload model after this many seconds of inactivity
        """
        self.embedding_service: Optional[EmbeddingService] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self.unload_after_seconds = unload_after_seconds
        self.last_used = 0
        self.current_model_key: Optional[str] = None
        self.lock = Lock()
        self.unload_task: Optional[asyncio.Task] = None
        self.is_mock_mode = False
        
    def _get_model_key(self, model_name: str, quantization: str) -> str:
        """Generate a unique key for model/quantization combination"""
        return f"{model_name}_{quantization}"
        
    def _ensure_service_initialized(self):
        """Ensure the embedding service is initialized"""
        with self.lock:
            if self.embedding_service is None:
                logger.info("Initializing embedding service")
                self.embedding_service = EmbeddingService()
                self.executor = ThreadPoolExecutor(max_workers=4)
                self.is_mock_mode = self.embedding_service.mock_mode
                
    def _update_last_used(self):
        """Update the last used timestamp"""
        self.last_used = time.time()
        
    async def _schedule_unload(self):
        """Schedule model unloading after inactivity"""
        if self.unload_task:
            self.unload_task.cancel()
            
        async def unload_after_delay():
            await asyncio.sleep(self.unload_after_seconds)
            with self.lock:
                if time.time() - self.last_used >= self.unload_after_seconds:
                    logger.info(f"Unloading model after {self.unload_after_seconds}s of inactivity")
                    self.unload_model()
                    
        self.unload_task = asyncio.create_task(unload_after_delay())
        
    def ensure_model_loaded(self, model_name: str, quantization: str) -> bool:
        """
        Ensure the specified model is loaded
        
        Returns:
            True if model is loaded successfully, False otherwise
        """
        self._ensure_service_initialized()
        
        if self.is_mock_mode:
            return True
            
        model_key = self._get_model_key(model_name, quantization)
        
        # Check if correct model is already loaded
        if self.current_model_key == model_key:
            self._update_last_used()
            return True
            
        # Need to load the model
        logger.info(f"Loading model: {model_name} with {quantization}")
        with self.lock:
            if self.embedding_service.load_model(model_name, quantization):
                self.current_model_key = model_key
                self._update_last_used()
                return True
            else:
                logger.error(f"Failed to load model: {model_name}")
                return False
                
    def unload_model(self):
        """Unload the current model to free memory"""
        with self.lock:
            if self.embedding_service and hasattr(self.embedding_service, 'current_model'):
                logger.info("Unloading current model")
                self.embedding_service.current_model = None
                self.embedding_service.current_tokenizer = None
                self.embedding_service.current_model_name = None
                self.embedding_service.current_quantization = None
                self.current_model_key = None
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear GPU cache if using CUDA
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
                    
    async def generate_embedding_async(self, text: str, model_name: str, 
                                     quantization: str, instruction: Optional[str] = None) -> Optional[list]:
        """
        Generate embedding with lazy model loading
        
        Args:
            text: Text to embed
            model_name: Model to use
            quantization: Quantization type
            instruction: Optional instruction for the model
            
        Returns:
            Embedding vector or None if failed
        """
        # Ensure model is loaded
        if not self.ensure_model_loaded(model_name, quantization):
            raise RuntimeError(f"Failed to load model {model_name}")
            
        # Schedule unloading
        await self._schedule_unload()
        
        # Generate embedding
        if self.is_mock_mode:
            # Use mock embedding
            import hashlib
            hash_bytes = hashlib.sha256(text.encode()).digest()
            values = []
            for i in range(0, len(hash_bytes), 4):
                chunk = hash_bytes[i:i+4]
                if len(chunk) == 4:
                    val = int.from_bytes(chunk, byteorder='big') / (2**32)
                    values.append(val * 2 - 1)
            # Pad to standard size
            while len(values) < 256:
                values.append(0.0)
            return values[:1024]  # Standard mock size
            
        # Use real embedding service
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            self.executor,
            self.embedding_service.generate_single_embedding,
            text,
            model_name,
            quantization,
            instruction
        )
        
        return embedding
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the model manager"""
        return {
            "model_loaded": self.current_model_key is not None,
            "current_model": self.current_model_key,
            "last_used": self.last_used,
            "seconds_since_last_use": int(time.time() - self.last_used) if self.last_used > 0 else None,
            "unload_after_seconds": self.unload_after_seconds,
            "is_mock_mode": self.is_mock_mode
        }
        
    def shutdown(self):
        """Shutdown the model manager"""
        if self.unload_task:
            self.unload_task.cancel()
        self.unload_model()
        if self.executor:
            self.executor.shutdown(wait=True)