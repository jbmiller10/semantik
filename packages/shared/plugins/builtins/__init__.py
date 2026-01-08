"""Built-in plugins for Semantik.

This package contains the built-in plugins that ship with Semantik:

Agent Plugins:
    - ClaudeAgentPlugin: LLM agent powered by Claude

Extractor Plugins:
    - KeywordExtractorPlugin: Keyword extraction from text

Reranker Plugins:
    - Qwen3RerankerPlugin: Search result reranking
"""

from shared.plugins.builtins.claude_agent import ClaudeAgentPlugin
from shared.plugins.builtins.keyword_extractor import KeywordExtractorPlugin
from shared.plugins.builtins.qwen3_reranker import Qwen3RerankerPlugin

__all__ = [
    "ClaudeAgentPlugin",
    "KeywordExtractorPlugin",
    "Qwen3RerankerPlugin",
]
