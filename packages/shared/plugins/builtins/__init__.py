"""Built-in plugins for Semantik.

This package contains the built-in plugins that ship with Semantik:

Extractor Plugins:
    - KeywordExtractorPlugin: Keyword extraction from text

Reranker Plugins:
    - Qwen3RerankerPlugin: Search result reranking

Sparse Indexer Plugins:
    - BM25SparseIndexerPlugin: BM25 sparse indexing for hybrid search
    - SPLADESparseIndexerPlugin: SPLADE learned sparse indexing
"""

from shared.plugins.builtins.bm25_sparse_indexer import BM25SparseIndexerPlugin
from shared.plugins.builtins.keyword_extractor import KeywordExtractorPlugin
from shared.plugins.builtins.qwen3_reranker import Qwen3RerankerPlugin
from shared.plugins.builtins.splade_indexer import SPLADESparseIndexerPlugin

__all__ = [
    "BM25SparseIndexerPlugin",
    "KeywordExtractorPlugin",
    "Qwen3RerankerPlugin",
    "SPLADESparseIndexerPlugin",
]
