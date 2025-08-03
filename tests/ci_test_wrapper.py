#!/usr/bin/env python3
"""
CI test wrapper that ensures proper environment setup before any imports.

This script must be imported before any other test imports to ensure
NLTK and other dependencies are properly mocked.
"""

import os
import sys
from unittest.mock import MagicMock

# Set test environment
os.environ["TESTING"] = "true"
os.environ["USE_MOCK_EMBEDDINGS"] = "true"

# Check if in CI
IS_CI = os.getenv("CI", "false").lower() == "true"

if IS_CI:
    # Disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Mock NLTK before any imports
    nltk_mock = MagicMock()
    nltk_mock.__version__ = "3.8"
    
    # Pre-mock all NLTK modules
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
    ]
    
    for module in nltk_modules:
        sys.modules[module] = MagicMock()
    
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
    
    # Mock torch CUDA
    torch_mock = MagicMock()
    torch_mock.cuda = MagicMock()
    torch_mock.cuda.is_available = MagicMock(return_value=False)
    torch_mock.cuda.device_count = MagicMock(return_value=0)
    sys.modules["torch.cuda"] = torch_mock.cuda