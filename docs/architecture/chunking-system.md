# Chunking System Architecture

> **Location:** `packages/shared/chunking/`

## Overview

Domain-driven chunking system supporting 6 strategies for splitting documents into searchable chunks. Follows clean architecture with domain, application, and infrastructure layers.

## Directory Structure

```
packages/shared/chunking/
├── domain/
│   ├── entities/
│   │   ├── chunk.py         # Chunk entity
│   │   └── document.py      # Document entity
│   ├── strategies/
│   │   ├── base.py          # BaseChunkingStrategy
│   │   ├── character.py     # Fixed character splits
│   │   ├── recursive.py     # Recursive text splitting
│   │   ├── markdown.py      # Markdown-aware splitting
│   │   ├── semantic.py      # AI-powered splitting
│   │   ├── hierarchical.py  # Parent-child relationships
│   │   └── hybrid.py        # Auto-selection
│   └── value_objects/
│       ├── chunk_config.py  # Configuration values
│       └── chunk_metadata.py
├── application/
│   ├── services/
│   │   └── chunking_service.py
│   └── dtos/
│       ├── chunk_request.py
│       └── chunk_response.py
└── infrastructure/
    ├── factories/
    │   └── strategy_factory.py
    └── repositories/
        └── chunk_repository.py
```

## Chunking Strategies

### CHARACTER
Fixed character-size splits with overlap.

```python
class CharacterChunkingStrategy(BaseChunkingStrategy):
    def chunk(self, content: str, config: ChunkingConfig) -> list[Chunk]:
        chunks = []
        start = 0

        while start < len(content):
            end = min(start + config.chunk_size, len(content))
            chunk_content = content[start:end]

            chunks.append(Chunk(
                content=chunk_content,
                index=len(chunks),
                start_offset=start,
                end_offset=end
            ))

            start += config.chunk_size - config.chunk_overlap

        return chunks
```

**Use Case:** Simple text, logs, raw data

### RECURSIVE
Intelligent splitting at natural boundaries.

```python
class RecursiveChunkingStrategy(BaseChunkingStrategy):
    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, content: str, config: ChunkingConfig) -> list[Chunk]:
        return self._split_text(content, config, self.SEPARATORS)

    def _split_text(
        self,
        text: str,
        config: ChunkingConfig,
        separators: list[str]
    ) -> list[Chunk]:
        if len(text) <= config.chunk_size:
            return [Chunk(content=text, index=0)]

        separator = separators[0]
        remaining_separators = separators[1:]

        splits = text.split(separator)
        chunks = []
        current_chunk = ""

        for split in splits:
            if len(current_chunk) + len(split) + len(separator) <= config.chunk_size:
                current_chunk += (separator if current_chunk else "") + split
            else:
                if current_chunk:
                    chunks.append(Chunk(content=current_chunk, index=len(chunks)))
                current_chunk = split

        if current_chunk:
            chunks.append(Chunk(content=current_chunk, index=len(chunks)))

        # Recursively split chunks that are still too large
        if remaining_separators:
            final_chunks = []
            for chunk in chunks:
                if len(chunk.content) > config.chunk_size:
                    sub_chunks = self._split_text(
                        chunk.content, config, remaining_separators
                    )
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)
            return final_chunks

        return chunks
```

**Use Case:** General documents, articles, prose

### MARKDOWN
Structure-aware splitting for Markdown documents.

```python
class MarkdownChunkingStrategy(BaseChunkingStrategy):
    def chunk(self, content: str, config: ChunkingConfig) -> list[Chunk]:
        # Parse Markdown structure
        sections = self._parse_sections(content)

        chunks = []
        for section in sections:
            # Preserve heading hierarchy in metadata
            metadata = {
                "heading": section.heading,
                "level": section.level,
                "path": section.heading_path
            }

            # Split section if too large
            if len(section.content) > config.chunk_size:
                sub_chunks = self._split_section(section, config)
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata.update(metadata)
                chunks.extend(sub_chunks)
            else:
                chunks.append(Chunk(
                    content=section.content,
                    index=len(chunks),
                    metadata=metadata
                ))

        return chunks

    def _parse_sections(self, content: str) -> list[Section]:
        """Parse Markdown into hierarchical sections."""
        sections = []
        lines = content.split("\n")
        current_section = None

        for line in lines:
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading_match:
                if current_section:
                    sections.append(current_section)
                level = len(heading_match.group(1))
                heading = heading_match.group(2)
                current_section = Section(
                    heading=heading,
                    level=level,
                    content=""
                )
            elif current_section:
                current_section.content += line + "\n"

        if current_section:
            sections.append(current_section)

        return sections
```

**Use Case:** Documentation, READMEs, wikis

### SEMANTIC
AI-powered splitting using sentence embeddings.

```python
class SemanticChunkingStrategy(BaseChunkingStrategy):
    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider

    async def chunk(self, content: str, config: ChunkingConfig) -> list[Chunk]:
        # Split into sentences
        sentences = self._split_sentences(content)

        # Generate embeddings for each sentence
        embeddings = await self.embedding_provider.embed_texts(
            sentences, mode=EmbeddingMode.DOCUMENT
        )

        # Group semantically similar sentences
        groups = self._group_by_similarity(
            sentences,
            embeddings,
            threshold=config.semantic_threshold
        )

        # Create chunks from groups
        chunks = []
        for group in groups:
            chunk_content = " ".join(group)
            chunks.append(Chunk(
                content=chunk_content,
                index=len(chunks),
                metadata={"semantic_group": True}
            ))

        return chunks

    def _group_by_similarity(
        self,
        sentences: list[str],
        embeddings: list[list[float]],
        threshold: float
    ) -> list[list[str]]:
        """Group sentences by semantic similarity."""
        groups = []
        current_group = [sentences[0]]
        current_embedding = embeddings[0]

        for i in range(1, len(sentences)):
            similarity = cosine_similarity(current_embedding, embeddings[i])

            if similarity >= threshold:
                current_group.append(sentences[i])
                # Update group embedding (average)
                current_embedding = np.mean(
                    [current_embedding, embeddings[i]], axis=0
                )
            else:
                groups.append(current_group)
                current_group = [sentences[i]]
                current_embedding = embeddings[i]

        if current_group:
            groups.append(current_group)

        return groups
```

**Use Case:** Complex documents, research papers, varied content

### HIERARCHICAL
Parent-child chunk relationships for context preservation.

```python
class HierarchicalChunkingStrategy(BaseChunkingStrategy):
    def chunk(self, content: str, config: ChunkingConfig) -> list[Chunk]:
        # Create parent chunks (larger)
        parent_config = ChunkingConfig(
            chunk_size=config.chunk_size * 3,
            chunk_overlap=config.chunk_overlap
        )
        parents = RecursiveChunkingStrategy().chunk(content, parent_config)

        # Create child chunks within each parent
        all_chunks = []
        for parent in parents:
            parent_chunk = Chunk(
                content=parent.content,
                index=len(all_chunks),
                metadata={"is_parent": True, "children": []}
            )
            all_chunks.append(parent_chunk)

            children = RecursiveChunkingStrategy().chunk(parent.content, config)
            for child in children:
                child_chunk = Chunk(
                    content=child.content,
                    index=len(all_chunks),
                    metadata={
                        "is_parent": False,
                        "parent_index": parent_chunk.index
                    }
                )
                parent_chunk.metadata["children"].append(child_chunk.index)
                all_chunks.append(child_chunk)

        return all_chunks
```

**Use Case:** Long documents where context matters, books, reports

### HYBRID
Auto-selects strategy based on content analysis.

```python
class HybridChunkingStrategy(BaseChunkingStrategy):
    def chunk(self, content: str, config: ChunkingConfig) -> list[Chunk]:
        detected_type = self._detect_content_type(content)

        strategy_map = {
            "markdown": MarkdownChunkingStrategy(),
            "code": RecursiveChunkingStrategy(),  # With code separators
            "prose": RecursiveChunkingStrategy(),
            "structured": CharacterChunkingStrategy(),
        }

        strategy = strategy_map.get(detected_type, RecursiveChunkingStrategy())
        return strategy.chunk(content, config)

    def _detect_content_type(self, content: str) -> str:
        """Analyze content to determine best strategy."""
        # Check for Markdown indicators
        if re.search(r"^#{1,6}\s", content, re.MULTILINE):
            return "markdown"

        # Check for code patterns
        code_patterns = [
            r"def \w+\(",
            r"function \w+\(",
            r"class \w+",
            r"import \w+"
        ]
        if any(re.search(p, content) for p in code_patterns):
            return "code"

        # Check for structured data
        if content.strip().startswith(("{", "[")):
            return "structured"

        return "prose"
```

**Use Case:** Mixed content, unknown formats

## Configuration

### ChunkingConfig
```python
class ChunkingConfig(BaseModel):
    chunk_size: int = Field(512, ge=100, le=4000)
    chunk_overlap: int = Field(64, ge=0, le=500)
    strategy: ChunkingStrategyType = ChunkingStrategyType.RECURSIVE

    # Strategy-specific options
    semantic_threshold: float = Field(0.5, ge=0.0, le=1.0)  # For semantic
    preserve_headings: bool = True  # For markdown
    min_chunk_size: int = 50  # Minimum chunk size

    @validator("chunk_overlap")
    def overlap_less_than_size(cls, v, values):
        if v >= values.get("chunk_size", 512):
            raise ValueError("overlap must be less than chunk_size")
        return v
```

### Strategy Factory
```python
class ChunkingStrategyFactory:
    @staticmethod
    def create(
        strategy_type: ChunkingStrategyType,
        embedding_provider: EmbeddingProvider | None = None
    ) -> BaseChunkingStrategy:
        strategies = {
            ChunkingStrategyType.CHARACTER: CharacterChunkingStrategy,
            ChunkingStrategyType.RECURSIVE: RecursiveChunkingStrategy,
            ChunkingStrategyType.MARKDOWN: MarkdownChunkingStrategy,
            ChunkingStrategyType.SEMANTIC: lambda: SemanticChunkingStrategy(
                embedding_provider
            ),
            ChunkingStrategyType.HIERARCHICAL: HierarchicalChunkingStrategy,
            ChunkingStrategyType.HYBRID: HybridChunkingStrategy,
        }

        factory = strategies.get(strategy_type)
        if not factory:
            raise ValueError(f"Unknown strategy: {strategy_type}")

        return factory()
```

## Chunking Service

```python
class ChunkingService:
    def __init__(
        self,
        strategy_factory: ChunkingStrategyFactory,
        chunk_repository: ChunkRepository
    ):
        self.strategy_factory = strategy_factory
        self.chunk_repository = chunk_repository

    async def chunk_document(
        self,
        document_id: str,
        content: str,
        config: ChunkingConfig
    ) -> list[Chunk]:
        """Chunk a document and store the chunks."""
        strategy = self.strategy_factory.create(config.strategy)

        if asyncio.iscoroutinefunction(strategy.chunk):
            chunks = await strategy.chunk(content, config)
        else:
            chunks = strategy.chunk(content, config)

        # Store chunks
        await self.chunk_repository.bulk_create(
            document_id=document_id,
            chunks=chunks
        )

        return chunks

    async def preview_chunks(
        self,
        content: str,
        config: ChunkingConfig
    ) -> ChunkingPreviewResponse:
        """Preview chunking without storing."""
        strategy = self.strategy_factory.create(config.strategy)
        chunks = strategy.chunk(content, config)

        return ChunkingPreviewResponse(
            chunks=[ChunkPreview(
                content=c.content[:200],
                size=len(c.content),
                index=c.index
            ) for c in chunks],
            statistics=ChunkingStatistics(
                total_chunks=len(chunks),
                avg_chunk_size=sum(len(c.content) for c in chunks) / len(chunks),
                min_chunk_size=min(len(c.content) for c in chunks),
                max_chunk_size=max(len(c.content) for c in chunks)
            )
        )

    async def compare_strategies(
        self,
        content: str,
        strategies: list[ChunkingStrategyType],
        base_config: ChunkingConfig
    ) -> list[ChunkingComparisonResult]:
        """Compare multiple strategies on same content."""
        results = []

        for strategy_type in strategies:
            config = base_config.copy(update={"strategy": strategy_type})
            preview = await self.preview_chunks(content, config)

            results.append(ChunkingComparisonResult(
                strategy=strategy_type,
                statistics=preview.statistics,
                sample_chunks=preview.chunks[:3]
            ))

        return results
```

## Extension Points

### Adding a New Strategy
1. Create strategy class in `domain/strategies/`
2. Extend `BaseChunkingStrategy`
3. Implement `chunk()` method
4. Add to `ChunkingStrategyType` enum
5. Register in `ChunkingStrategyFactory`
6. Add frontend UI option
7. Write tests

### Adding Strategy Parameters
1. Add to `ChunkingConfig` model
2. Add validation if needed
3. Pass to strategy in factory
4. Update frontend parameter tuner
