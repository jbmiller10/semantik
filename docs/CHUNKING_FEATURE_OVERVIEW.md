# Chunking System

Splits documents into chunks for semantic search. 6 strategies, 40+ file types.

## Strategies

**Character** - Fixed-size splitting

**Recursive** - Smart general-purpose splitting (default)

**Markdown** - Preserves headers and structure

**Semantic** - Finds natural topic boundaries using embeddings

**Hierarchical** - Creates parent-child relationships

**Hybrid** - Auto-switches based on content

### Code Files
Handled via recursive chunking with optimized parameters (400 chars, reduced overlap). Dedicated code-aware chunking coming ~2 weeks post-launch.

## Architecture

**ChunkingOrchestrator** - Handles security, caching, errors, analytics

**Database** - Normalized schema, partitioned chunks (16), materialized views

**Performance** (4-core, 8GB):
- Character: 1000 chunks/sec
- Recursive: 800/sec
- Markdown: 600/sec
- Semantic: 150/sec (embeddings overhead)
- Hierarchical: 400/sec (multiple passes)

## Implementation Timeline

**Week 1**: Foundation (BaseChunker, orchestrator, 3 core strategies)

**Week 2**: Advanced strategies, database schema, API, async processing

**Week 3**: Tests (>90% coverage), optimization, monitoring, docs

**Week 4**: Scale validation, security audit, deployment prep

## Stack

UI → REST API + WebSocket → ChunkingOrchestrator → Strategies → Celery/Redis/PostgreSQL/Qdrant

## Technical Decisions

**LlamaIndex** - Industry-standard, well-tested, saves time

**No backwards compatibility** - 40% less code, cleaner API, faster shipping

**Ship docs first** - Launch with 6 strategies, add code-specific later

**Performance-first** - Benchmarks defined, streaming, caching, async everywhere

## Code Files

**Now**: Recursive chunking with optimized params (works well)
**Later** (~2 weeks): Language-aware CodeChunker with syntax preservation

Why defer? Ship faster, learn from usage, build it right.

## For Developers

Validate inputs. Benchmark often. Plan for failure. Test edge cases.

**Resources**:
- LlamaIndex: https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/
- Sentence Transformers: https://www.sbert.net/

## Success Criteria

- 6 strategies working, >90% test coverage
- <100MB memory for 10MB doc
- Performance benchmarks hit
- Ship in 4 weeks

## FAQ

**Why defer code support?** Ship faster, build it right later. Current recursive chunking works reasonably well (~80% quality).

**Re-index needed later?** Optional for improvements, existing chunks still work.

**Timeline?** 4 weeks: foundation (1), strategies+integration (2), testing (3), buffer (4)

## Roadmap

**+2 weeks**: Code support (syntax-aware, language-specific)
**+4 weeks**: A/B testing, custom strategies, advanced analytics