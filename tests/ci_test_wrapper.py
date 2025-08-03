#!/usr/bin/env python3
"""
CI test wrapper that ensures proper environment setup before any imports.

This script must be imported before any other test imports to ensure
NLTK and other dependencies are properly mocked.
"""

import os
import sys
import importlib.util
from unittest.mock import MagicMock

# Set test environment
os.environ["TESTING"] = "true"
os.environ["USE_MOCK_EMBEDDINGS"] = "true"

# Check if in CI
IS_CI = os.getenv("CI", "false").lower() == "true"

if IS_CI:
    # Disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    def create_nltk_module_mock(module_name):
        """Create a proper module mock with __spec__ attribute."""
        mock = MagicMock()
        mock.__name__ = module_name
        mock.__package__ = module_name.rsplit('.', 1)[0] if '.' in module_name else None
        mock.__spec__ = importlib.util.spec_from_loader(module_name, loader=None)
        mock.__version__ = "3.8"
        return mock
    
    # Pre-mock all NLTK modules with proper specs
    nltk_modules = [
        "nltk",
        "nltk.data",
        "nltk.tokenize",
        "nltk.tokenize.punkt",
        "nltk.corpus",
        "nltk.download",
        "nltk.stem",
        "nltk.tag",
        "nltk.chunk",
        "nltk.parse",
        "nltk.tree",
        "nltk.grammar",
        "nltk.sem",
        "nltk.metrics",
        "nltk.classify",
        "nltk.cluster",
    ]
    
    for module_name in nltk_modules:
        sys.modules[module_name] = create_nltk_module_mock(module_name)
    
    # Setup basic tokenizer
    def mock_sent_tokenize(text):
        if not text:
            return []
        sentences = []
        for s in text.split("."):
            s = s.strip()
            if s:
                sentences.append(s + ".")
        return sentences if sentences else [text]
    
    sys.modules["nltk.tokenize"].sent_tokenize = mock_sent_tokenize
    sys.modules["nltk"].tokenize = sys.modules["nltk.tokenize"]
    sys.modules["nltk"].sent_tokenize = mock_sent_tokenize
    
    # Mock torch CUDA
    torch_mock = MagicMock()
    torch_mock.cuda = MagicMock()
    torch_mock.cuda.is_available = MagicMock(return_value=False)
    torch_mock.cuda.device_count = MagicMock(return_value=0)
    sys.modules["torch.cuda"] = torch_mock.cuda