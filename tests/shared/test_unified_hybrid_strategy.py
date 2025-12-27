import pytest

from shared.chunking.domain.value_objects.chunk_config import ChunkConfig
from shared.chunking.unified.hybrid_strategy import HybridChunkingStrategy


def test_analyze_content_recommends_hybrid_for_mixed_content():
    strategy = HybridChunkingStrategy()
    content = """# Heading

Some narrative text.

```python
print('hello')
```

- item one
- item two
"""

    analysis = strategy._analyze_content(content)

    assert analysis["is_mixed"] is True
    assert analysis["recommended_strategy"] == "hybrid"


@pytest.mark.asyncio()
async def test_chunk_async_returns_chunks_for_content():
    strategy = HybridChunkingStrategy()
    config = ChunkConfig(strategy_name="hybrid", min_tokens=10, max_tokens=30, overlap_tokens=5)
    content = "This is a short test document. " * 20

    chunks = await strategy.chunk_async(content, config)

    assert chunks
