# Chunking System

Semantik's document chunking system splits documents into meaningful chunks for semantic search. 6 strategies, 40+ file types, configurable per collection.

## Chunking Strategies

### Level 1: Basic
1. **Character** (TokenTextSplitter) - Simple fixed-size splitting for basic needs

### Level 2: Smart Text Splitting  
2. **Recursive** (SentenceSplitter) - Intelligent general-purpose splitting (our recommended default)
   - Also handles code files with optimized parameters until dedicated support arrives

### Level 3: Document-Aware
3. **Markdown** (MarkdownNodeParser) - Respects headers and formatting in technical docs

### Level 4: Advanced
4. **Semantic** (SemanticSplitterNodeParser) - Uses AI embeddings to find natural topic boundaries
5. **Hierarchical** (HierarchicalNodeParser) - Creates parent-child chunks for better context
6. **Hybrid** - Automatically switches strategies based on content

### Note on Code Files
Code files (.py, .js, .java, etc.) are fully supported using optimized recursive chunking:
- Smaller chunk sizes (400 vs 600 characters)
- Reduced overlap for efficiency
- Line-break aware splitting

Dedicated code-aware chunking with syntax understanding is planned for ~2 weeks post-launch.

---

## Architecture Enhancements

Based on review feedback, our architecture includes:

### 1. **Service Layer Architecture**
```python
ChunkingOrchestrator       # Business logic separation
├── Security validation
├── Caching layer
├── Error handling
└── Analytics tracking
```

### 2. **Normalized Database Schema**
- Separate tables for strategies, configs, and metrics
- Partitioned chunks table for scale (16 partitions)
- Materialized views for performance
- Proper indexes throughout

### 3. **Security Throughout**
- Input validation at every layer
- Rate limiting on API endpoints
- Parameter bounds checking
- Document size limits

### 4. **Performance Targets**
```
Hardware: 4-core CPU, 8GB RAM
Character: 1000 chunks/sec single-threaded
Recursive: 800 chunks/sec single-threaded
Markdown: 600 chunks/sec single-threaded
Semantic: 150 chunks/sec (due to embeddings)
Hierarchical: 400 chunks/sec (multiple passes)
```

---

## Phase Breakdown

### Week 1: Core Foundation & Architecture 🏗️
**Goal**: Build solid foundation with 3 core strategies

**Key Tasks**:
- Implement BaseChunker interface with async support
- Create ChunkingOrchestrator layer for business logic
- Build 3 core strategies using LlamaIndex:
  - Character (TokenTextSplitter)
  - Recursive (SentenceSplitter)
  - Markdown (MarkdownNodeParser)
- Establish error handling framework
- Create performance testing framework

**Success Metrics**:
- Core strategies meet performance benchmarks
- Security validation prevents malicious inputs
- Error handling covers all failure modes
- Architecture supports future strategies

### Week 2: Complete Strategies & Integration 🔌
**Goal**: Implement remaining strategies and integrate with Semantik

**Key Tasks**:
- Implement 3 advanced strategies using LlamaIndex:
  - Semantic (SemanticSplitterNodeParser)
  - Hierarchical (HierarchicalNodeParser)
  - Hybrid (custom combination)
- Create normalized database schema with partitioning
- Build comprehensive API with security and rate limiting
- Implement async processing with priority queues

**Success Metrics**:
- All 6 strategies working correctly
- Database operations performant at scale
- API secure with proper validation
- Async processing handles failures gracefully

### Week 3: Testing, Performance & Polish ✨
**Goal**: Ensure production readiness

**Key Tasks**:
- Build comprehensive test suite with edge cases
- Performance optimization and streaming support
- Add monitoring and analytics
- Create documentation and guides

**Success Metrics**:
- >90% test coverage achieved
- Performance benchmarks met or exceeded
- Monitoring provides actionable insights
- Documentation clear and complete

### Week 4: Final Polish & Buffer 🚀
**Goal**: Final validation and deployment preparation

**Key Tasks**:
- Performance validation at scale
- Security audit and penetration testing
- Documentation review
- Deployment preparation

**Success Metrics**:
- All systems validated at production scale
- Security audit passed
- Ready for deployment
- Team prepared for support

---

## Enhanced Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   User Interface                     │
│  (Collection Creation, Strategy Selection, Progress) │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│               REST API + WebSocket                   │
│  - Rate limiting            - Input validation       │
│  - Security checks          - Progress streaming     │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│              ChunkingOrchestrator Layer                   │
│  - Business logic           - Error handling         │
│  - Cache management         - Analytics tracking     │
│  - Security validation      - Progress monitoring    │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│              Chunking System                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐│
│  │BaseChunker  │  │ChunkingFactory│  │FileDetector││
│  │(Async+Sync) │  │(w/Validation) │  │(40+ types) ││
│  └──────┬──────┘  └───────┬──────┘  └────────────┘│
│         │                  │                        │
│  ┌──────▼──────────────────▼────────────────────┐  │
│  │   6 Strategy Implementations                  │  │
│  │ - Resource management   - Streaming support   │  │
│  │ - Error recovery        - Performance opts   │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────┘
                      │
        ┌─────────────▼──────────┐
        │   Distributed Systems  │
        │  - Celery (Priority Q) │
        │  - Redis (Caching)     │
        │  - PostgreSQL (Normal) │
        │  - Qdrant (Vectors)    │
        └────────────────────────┘
```

---

## Technical Decisions

### Why These Choices?

1. **LlamaIndex Integration**
   - Industry-standard text splitting
   - Well-tested and optimized
   - Saves development time
   - Regular updates and improvements

2. **Enhanced Architecture**
   - ChunkingOrchestrator layer for clean separation
   - Normalized schema for query performance
   - Security validation throughout
   - Error handling at every level

3. **No Backwards Compatibility**
   - Simpler code (40% less)
   - Better performance
   - Cleaner API
   - Faster development

4. **6 Strategies Now, Code Later**
   - Ship faster with solid foundation
   - Focus on document excellence
   - Gather code usage insights
   - Build better code support

5. **Performance-First Design**
   - Concrete benchmarks defined
   - Streaming for large documents
   - Caching for repeated operations
   - Async processing throughout

---

## Key Improvements from Review

### 1. **Reduced Week 1 Scope**
- Start with 3 core strategies instead of all 6
- Focus on architecture and foundation
- Ensure performance testing from day 1

### 2. **Enhanced Architecture**
- Added ChunkingOrchestrator layer
- Normalized database schema
- Comprehensive security validation
- Robust error handling framework

### 3. **Concrete Performance Targets**
- Defined hardware baseline
- Strategy-specific benchmarks
- Memory usage limits
- Parallel processing targets

### 4. **Operational Excellence**
- Monitoring and metrics from start
- Idempotent operations
- Graceful degradation
- Recovery strategies

---

## Code File Handling Strategy

### Current Approach (Launch)
- **Detection**: Code files are detected and categorized
- **Processing**: Use recursive chunker with optimized parameters
- **Tracking**: Analytics track code file usage
- **Communication**: Clear messaging about roadmap

### Future Approach (~2 weeks post-launch)
- **Dedicated Strategy**: Language-aware CodeChunker
- **Syntax Understanding**: Preserve functions, classes, imports
- **Language-Specific**: Optimize for Python, JavaScript, etc.
- **Seamless Migration**: Re-chunk collections automatically

### Why Defer Code Support?
1. **Complexity**: Code chunking requires language-specific parsing
2. **Quality**: Better to do it right than rush it
3. **Insights**: Learn from real usage patterns first
4. **Speed**: Ship core features faster

---

## For Developers

### Key Principles
1. **Security First**: Validate everything, trust nothing
2. **Performance Matters**: Benchmark early and often
3. **Error Handling**: Plan for failure at every step
4. **Clean Architecture**: Maintain separation of concerns
5. **Test Everything**: Edge cases matter
6. **Document as You Go**: Future you will thank present you
7. **Communicate Roadmap**: Be clear about what's coming

### Getting Started
1. Read the detailed implementation plan
2. Review architecture decisions  
3. Understand performance targets
4. Check security requirements
5. Review your assigned tasks
6. Ask questions early and often
7. Test thoroughly before marking complete

### Available Tools & Subagents
Each implementation task has access to:
- **Tools**: context7 (library docs), playwright (UI testing), Read/Edit/MultiEdit, Bash, Grep/Glob, WebSearch, TodoWrite
- **Implementation Subagents**: backend-api-architect, database-migrations-engineer, performance-profiler, vector-search-architect, devops-sentinel
- **Review Subagents**: backend-code-reviewer, tech-debt-hunter, qa-bug-hunter, test-maestro, docs-scribe
- **Frontend Subagents**: frontend-state-architect, ui-component-craftsman (if UI work needed)

See the implementation plan for specific subagent recommendations per task.

### Resources
- Implementation Plan: `CHUNKING_IMPLEMENTATION_PLAN_NO_CODE.md`
- LlamaIndex Docs: https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/
- Sentence Transformers: https://www.sbert.net/

---

## Success Criteria

### Technical Success
- ✅ All 6 strategies implemented and tested
- ✅ 40+ file types supported (including code files)
- ✅ Performance benchmarks achieved
- ✅ <100MB memory for 10MB document
- ✅ >90% test coverage
- ✅ Security validation comprehensive
- ✅ Error handling robust
- ✅ Monitoring operational

### User Success
- ✅ Users can easily select appropriate strategy
- ✅ Search quality improves measurably
- ✅ Code files processed without errors
- ✅ Clear communication about code support timeline
- ✅ Analytics provide actionable insights
- ✅ Documentation is clear and helpful

### Business Success
- ✅ Ship with confidence (4-week timeline)
- ✅ Differentiates Semantik from competitors
- ✅ Reduces support burden through smart defaults
- ✅ Enables new use cases
- ✅ Foundation for excellent code support

---

## Risk Mitigation

### Technical Risks
1. **Semantic chunker complexity** → Start in Week 2, not Week 1
2. **Memory issues with large docs** → Streaming support built-in
3. **Performance bottlenecks** → Early benchmarking, optimization time
4. **Security vulnerabilities** → Validation at every layer

### Process Risks
1. **Timeline slippage** → Buffer week included
2. **Integration issues** → Phased approach
3. **Quality concerns** → Multiple review checkpoints
4. **Team coordination** → Clear task separation

---

## Common Questions

### Q: Why not include code support at launch?
**A**: Code chunking is complex and deserves dedicated focus. By deferring it, we can ship faster and build better code support based on real usage data.

### Q: How well does recursive chunking work for code?
**A**: Quite well! With optimized parameters (smaller chunks, less overlap), it maintains readability and searchability. It's ~80% as good as dedicated code chunking.

### Q: What changed from the original plan?
**A**: Based on review feedback: added buffer time (3.5→4 weeks), enhanced architecture, concrete performance targets, comprehensive error handling, and security throughout.

### Q: When will code support arrive?
**A**: Approximately 2 weeks after launch. We'll use insights from initial usage to build excellent language-aware chunking.

### Q: Will I need to re-index when code support arrives?
**A**: You'll have the option to re-index for improved results, but existing chunks will continue to work.

### Q: Why 40+ file types?
**A**: Covers all common document, code, and data formats users need.

---

## Timeline & Milestones

```
Week 1: Foundation & Architecture
├── Day 1-2: BaseChunker + ChunkingOrchestrator
├── Day 3: Core strategies (character, recursive, markdown)
├── Day 4: Error handling + Security
├── Day 5: Performance testing framework
└── Review: Foundation validation

Week 2: Strategies & Integration  
├── Day 1-2: Advanced strategies (semantic, hierarchical, hybrid)
├── Day 3: Normalized database + partitioning
├── Day 4: API with security + rate limiting
├── Day 5: Async processing + priority queues
└── Review: Integration validation

Week 3: Testing & Polish
├── Day 1-2: Comprehensive test suite
├── Day 3: Performance optimization
├── Day 4: Monitoring + analytics
├── Day 5: Documentation
└── Review: Pre-launch validation

Week 4: Buffer & Launch Prep
├── Day 1: Performance validation at scale
├── Day 2: Security audit
├── Day 3: Documentation review
├── Day 4: Deployment preparation
└── Final Review: Launch readiness
```

---

## Future Roadmap

### 2 Weeks Post-Launch: Code Support
- Research best practices
- Implement CodeChunker with syntax awareness
- Add language detection
- Support function/class preservation
- Enable seamless migration

### 4 Weeks Post-Launch: Enhanced Features
- A/B testing for strategies
- Custom strategy creation
- Advanced analytics
- Performance optimizations

---

## Conclusion

We're building a robust, scalable chunking system with a solid foundation for future enhancements. The architectural improvements from review feedback ensure we'll deliver a system that can scale to millions of chunks while maintaining security and performance.

By deferring code support, we can ship a polished document chunking system faster while gathering insights to build best-in-class code support.

Every task contributes to this vision. Your work matters and will directly impact how well users can find information in their documents.

Let's build something amazing! 🚀

---

*Questions? Reach out to the team lead or post in #semantik-chunking*