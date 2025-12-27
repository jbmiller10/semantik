# Extension Cookbook: Search Enhancements

> **Audience:** Software architects implementing new search functionality in Semantik
> **Prerequisites:** Understanding of VecPipe architecture, FastAPI patterns, React Query

---

## Table of Contents

1. [Add a New Search Mode](#1-add-a-new-search-mode)
2. [Add a New Reranker](#2-add-a-new-reranker)
3. [Extend GraphRAG Capabilities](#3-extend-graphrag-capabilities)
4. [Add Search Filters](#4-add-search-filters)
5. [Implement Search Analytics](#5-implement-search-analytics)

---

## 1. Add a New Search Mode

### Overview

Search modes in Semantik define how queries are processed and matched against vectors. Current modes include `semantic`, `keyword`, and `hybrid`. This guide shows how to add a new mode (example: `mmr` - Maximal Marginal Relevance).

### Step 1: Define the Search Mode Enum

**File:** `packages/shared/contracts/search.py`

```python
from enum import Enum

class SearchMode(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    MMR = "mmr"  # Add new mode
```

### Step 2: Update VecPipe Search Service

**File:** `packages/vecpipe/services/search_service.py`

```python
class SearchService:
    async def search(
        self,
        query: str,
        collections: list[str],
        mode: SearchMode,
        top_k: int = 10,
        **kwargs
    ) -> SearchResponse:
        """Route to appropriate search implementation."""

        if mode == SearchMode.MMR:
            return await self._mmr_search(
                query=query,
                collections=collections,
                top_k=top_k,
                lambda_mult=kwargs.get("lambda_mult", 0.5),
                fetch_k=kwargs.get("fetch_k", top_k * 4)
            )
        # ... existing modes

    async def _mmr_search(
        self,
        query: str,
        collections: list[str],
        top_k: int,
        lambda_mult: float,
        fetch_k: int
    ) -> SearchResponse:
        """Maximal Marginal Relevance search for diversity."""

        # 1. Generate query embedding
        query_embedding = await self.embedding_service.embed_query(query)

        # 2. Fetch more candidates than needed
        candidates = await self.qdrant_client.search(
            collection_name=collections[0],
            query_vector=query_embedding,
            limit=fetch_k
        )

        # 3. Apply MMR algorithm
        selected = []
        candidate_embeddings = [c.vector for c in candidates]

        while len(selected) < top_k and candidates:
            mmr_scores = []

            for i, candidate in enumerate(candidates):
                if candidate in selected:
                    continue

                # Relevance to query
                relevance = self._cosine_similarity(
                    query_embedding,
                    candidate_embeddings[i]
                )

                # Maximum similarity to already selected
                if selected:
                    max_sim = max(
                        self._cosine_similarity(
                            candidate_embeddings[i],
                            candidate_embeddings[candidates.index(s)]
                        )
                        for s in selected
                    )
                else:
                    max_sim = 0

                # MMR score
                mmr = lambda_mult * relevance - (1 - lambda_mult) * max_sim
                mmr_scores.append((candidate, mmr))

            # Select highest MMR score
            best = max(mmr_scores, key=lambda x: x[1])
            selected.append(best[0])
            candidates.remove(best[0])

        return SearchResponse(
            results=[self._to_search_result(r) for r in selected],
            mode=SearchMode.MMR
        )
```

### Step 3: Add API Parameters

**File:** `packages/vecpipe/api/search.py`

```python
class SearchRequest(BaseModel):
    query: str
    collections: list[str]
    mode: SearchMode = SearchMode.SEMANTIC
    top_k: int = Field(10, ge=1, le=100)

    # Mode-specific parameters
    lambda_mult: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="MMR diversity factor (0=max diversity, 1=max relevance)"
    )
    fetch_k: int | None = Field(
        None,
        ge=1,
        description="Number of candidates to fetch for MMR"
    )

    @validator("lambda_mult", "fetch_k", always=True)
    def validate_mmr_params(cls, v, values):
        if values.get("mode") == SearchMode.MMR and v is None:
            # Set defaults for MMR mode
            if "lambda_mult" not in values:
                return 0.5
            return values.get("top_k", 10) * 4
        return v
```

### Step 4: Update WebUI Service

**File:** `packages/webui/services/search_service.py`

```python
class SearchService:
    async def search(
        self,
        request: SearchRequest,
        user_id: int
    ) -> SearchResponse:
        # Validate collection access
        await self._validate_access(request.collections, user_id)

        # Build VecPipe request with mode-specific params
        vecpipe_params = {
            "query": request.query,
            "collections": request.collections,
            "mode": request.mode,
            "top_k": request.top_k,
        }

        if request.mode == SearchMode.MMR:
            vecpipe_params["lambda_mult"] = request.lambda_mult
            vecpipe_params["fetch_k"] = request.fetch_k

        return await self.vecpipe_client.search(**vecpipe_params)
```

### Step 5: Add Frontend Support

**File:** `apps/webui-react/src/components/search/SearchModeSelector.tsx`

```typescript
const SEARCH_MODES = [
  { value: 'semantic', label: 'Semantic', description: 'Vector similarity' },
  { value: 'keyword', label: 'Keyword', description: 'BM25 text matching' },
  { value: 'hybrid', label: 'Hybrid', description: 'Combined scoring' },
  { value: 'mmr', label: 'Diverse', description: 'Varied results (MMR)' },
] as const;

interface MMROptions {
  lambdaMult: number;
  fetchK: number;
}

export function SearchModeSelector({
  mode,
  onModeChange,
  mmrOptions,
  onMMROptionsChange
}: Props) {
  return (
    <div className="space-y-4">
      <Select value={mode} onValueChange={onModeChange}>
        {SEARCH_MODES.map(m => (
          <SelectItem key={m.value} value={m.value}>
            {m.label} - {m.description}
          </SelectItem>
        ))}
      </Select>

      {mode === 'mmr' && (
        <div className="pl-4 border-l-2 space-y-2">
          <Slider
            label="Diversity Factor"
            value={mmrOptions.lambdaMult}
            onChange={(v) => onMMROptionsChange({ ...mmrOptions, lambdaMult: v })}
            min={0}
            max={1}
            step={0.1}
          />
          <p className="text-xs text-muted-foreground">
            Lower = more diverse results, Higher = more relevant
          </p>
        </div>
      )}
    </div>
  );
}
```

**File:** `apps/webui-react/src/stores/searchStore.ts`

```typescript
interface SearchState {
  mode: SearchMode;
  mmrOptions: { lambdaMult: number; fetchK: number };
  setMode: (mode: SearchMode) => void;
  setMMROptions: (options: Partial<MMROptions>) => void;
}

export const useSearchStore = create<SearchState>((set) => ({
  mode: 'semantic',
  mmrOptions: { lambdaMult: 0.5, fetchK: 40 },

  setMode: (mode) => set({ mode }),
  setMMROptions: (options) => set((state) => ({
    mmrOptions: { ...state.mmrOptions, ...options }
  })),
}));
```

### Step 6: Add Tests

**File:** `packages/vecpipe/tests/test_mmr_search.py`

```python
import pytest
from packages.vecpipe.services.search_service import SearchService

@pytest.fixture
def search_service(mock_qdrant, mock_embedder):
    return SearchService(
        qdrant_client=mock_qdrant,
        embedding_service=mock_embedder
    )

class TestMMRSearch:
    async def test_mmr_returns_diverse_results(self, search_service):
        """MMR should return more diverse results than pure semantic."""
        # Arrange: Create clustered test data
        # (documents that are semantically similar to each other)

        # Act
        mmr_results = await search_service.search(
            query="test query",
            collections=["test"],
            mode=SearchMode.MMR,
            top_k=5,
            lambda_mult=0.3  # Favor diversity
        )

        semantic_results = await search_service.search(
            query="test query",
            collections=["test"],
            mode=SearchMode.SEMANTIC,
            top_k=5
        )

        # Assert: MMR results should be more spread out
        mmr_diversity = calculate_pairwise_diversity(mmr_results)
        semantic_diversity = calculate_pairwise_diversity(semantic_results)

        assert mmr_diversity > semantic_diversity

    async def test_mmr_with_high_lambda_approximates_semantic(
        self,
        search_service
    ):
        """With lambda=1.0, MMR should match semantic search."""
        mmr_results = await search_service.search(
            query="test",
            collections=["test"],
            mode=SearchMode.MMR,
            lambda_mult=1.0
        )

        semantic_results = await search_service.search(
            query="test",
            collections=["test"],
            mode=SearchMode.SEMANTIC
        )

        # Results should be identical
        assert [r.id for r in mmr_results] == [r.id for r in semantic_results]
```

### Checklist

- [ ] Add enum value to `SearchMode`
- [ ] Implement search logic in `SearchService`
- [ ] Add API parameters with validation
- [ ] Update WebUI service to pass parameters
- [ ] Add frontend selector component
- [ ] Update Zustand store
- [ ] Write unit tests for new mode
- [ ] Write integration tests
- [ ] Update API documentation
- [ ] Add mode to search analytics tracking

---

## 2. Add a New Reranker

### Overview

Rerankers improve search quality by re-scoring initial results using a more sophisticated model. This guide shows how to add a new reranker (example: Cohere Rerank).

### Step 1: Create Reranker Interface

**File:** `packages/vecpipe/rerankers/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class RerankResult:
    index: int
    score: float
    original_score: float

class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this reranker."""
        pass

    @property
    @abstractmethod
    def max_documents(self) -> int:
        """Maximum documents this reranker can process."""
        pass

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[RerankResult]:
        """Rerank documents by relevance to query."""
        pass
```

### Step 2: Implement the Reranker

**File:** `packages/vecpipe/rerankers/cohere_reranker.py`

```python
import cohere
from .base import BaseReranker, RerankResult

class CohereReranker(BaseReranker):
    """Cohere Rerank API integration."""

    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        self.client = cohere.AsyncClient(api_key)
        self.model = model

    @property
    def name(self) -> str:
        return "cohere"

    @property
    def max_documents(self) -> int:
        return 1000  # Cohere limit

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None
    ) -> list[RerankResult]:
        response = await self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_k or len(documents),
            return_documents=False
        )

        return [
            RerankResult(
                index=r.index,
                score=r.relevance_score,
                original_score=0.0  # Will be filled by caller
            )
            for r in response.results
        ]
```

### Step 3: Register in Factory

**File:** `packages/vecpipe/rerankers/factory.py`

```python
from .base import BaseReranker
from .cross_encoder import CrossEncoderReranker
from .cohere_reranker import CohereReranker

class RerankerFactory:
    _rerankers: dict[str, type[BaseReranker]] = {}

    @classmethod
    def register(cls, reranker_class: type[BaseReranker]) -> None:
        instance = reranker_class.__new__(reranker_class)
        cls._rerankers[instance.name] = reranker_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseReranker:
        if name not in cls._rerankers:
            raise ValueError(f"Unknown reranker: {name}")
        return cls._rerankers[name](**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        return list(cls._rerankers.keys())

# Register built-in rerankers
RerankerFactory.register(CrossEncoderReranker)
RerankerFactory.register(CohereReranker)
```

### Step 4: Use in Search Pipeline

**File:** `packages/vecpipe/services/search_service.py`

```python
class SearchService:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_service: EmbeddingService,
        reranker_factory: RerankerFactory
    ):
        self.reranker_factory = reranker_factory

    async def search(
        self,
        query: str,
        collections: list[str],
        top_k: int = 10,
        use_reranker: bool = False,
        reranker_name: str = "cross-encoder",
        **kwargs
    ) -> SearchResponse:
        # Initial search (fetch more for reranking)
        fetch_k = top_k * 4 if use_reranker else top_k

        initial_results = await self._vector_search(
            query=query,
            collections=collections,
            limit=fetch_k
        )

        if use_reranker and initial_results:
            reranker = self.reranker_factory.create(
                reranker_name,
                **self._get_reranker_config(reranker_name)
            )

            documents = [r.content for r in initial_results]
            reranked = await reranker.rerank(
                query=query,
                documents=documents,
                top_k=top_k
            )

            # Reorder results
            final_results = [
                initial_results[r.index]._replace(
                    score=r.score,
                    rerank_score=r.score,
                    original_score=initial_results[r.index].score
                )
                for r in reranked
            ]
        else:
            final_results = initial_results[:top_k]

        return SearchResponse(
            results=final_results,
            reranking_used=use_reranker,
            reranker_name=reranker_name if use_reranker else None
        )
```

### Step 5: Configuration

**File:** `packages/vecpipe/config.py`

```python
class RerankerConfig(BaseSettings):
    default_reranker: str = "cross-encoder"

    # Cross-encoder settings
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_device: str = "cuda"

    # Cohere settings
    cohere_api_key: str | None = None
    cohere_model: str = "rerank-english-v3.0"

    class Config:
        env_prefix = "RERANKER_"
```

### Checklist

- [ ] Create reranker class implementing `BaseReranker`
- [ ] Register in `RerankerFactory`
- [ ] Add configuration for API keys/models
- [ ] Update search service to accept reranker selection
- [ ] Add frontend reranker selector
- [ ] Write unit tests with mocked API
- [ ] Add rate limiting if using external API
- [ ] Document reranker characteristics

---

## 3. Extend GraphRAG Capabilities

### Overview

GraphRAG enhances retrieval by building a knowledge graph from documents and using graph traversal during search. This guide shows how to extend it with new capabilities.

### Current GraphRAG Architecture

```
packages/vecpipe/graphrag/
├── entities/           # Entity types (Person, Concept, etc.)
├── extractors/         # NLP extractors for entities/relations
├── graph/              # Neo4j/in-memory graph operations
├── services/           # GraphRAG search service
└── types.py           # Core type definitions
```

### Example: Add Custom Entity Type

**File:** `packages/vecpipe/graphrag/entities/organization.py`

```python
from dataclasses import dataclass
from .base import BaseEntity

@dataclass
class OrganizationEntity(BaseEntity):
    """Represents an organization in the knowledge graph."""

    entity_type: str = "ORGANIZATION"

    # Organization-specific attributes
    org_type: str | None = None  # company, nonprofit, government
    industry: str | None = None
    headquarters: str | None = None

    @classmethod
    def from_ner_span(cls, span, doc_id: str) -> "OrganizationEntity":
        """Create from spaCy NER span."""
        return cls(
            id=f"org_{hash(span.text)}",
            name=span.text,
            source_doc_id=doc_id,
            start_char=span.start_char,
            end_char=span.end_char,
            confidence=span._.confidence if hasattr(span._, 'confidence') else 1.0
        )
```

### Example: Add Relation Extractor

**File:** `packages/vecpipe/graphrag/extractors/employment_extractor.py`

```python
import spacy
from .base import BaseRelationExtractor
from ..types import Relation

class EmploymentRelationExtractor(BaseRelationExtractor):
    """Extract employment relationships (works_for, founded_by, etc.)."""

    EMPLOYMENT_PATTERNS = [
        {"pattern": "CEO of", "relation": "CEO_OF"},
        {"pattern": "works at", "relation": "WORKS_FOR"},
        {"pattern": "founded", "relation": "FOUNDED"},
        {"pattern": "employed by", "relation": "WORKS_FOR"},
    ]

    def __init__(self, nlp: spacy.Language):
        self.nlp = nlp
        self._add_patterns()

    def _add_patterns(self):
        """Add dependency patterns to spaCy pipeline."""
        from spacy.matcher import DependencyMatcher

        self.matcher = DependencyMatcher(self.nlp.vocab)

        # Pattern: PERSON [verb] ORGANIZATION
        pattern = [
            {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": {"IN": ["work", "found", "lead"]}}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "subject", "RIGHT_ATTRS": {"DEP": "nsubj"}},
            {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "pobj"]}}},
        ]
        self.matcher.add("EMPLOYMENT", [pattern])

    def extract(self, doc: spacy.tokens.Doc) -> list[Relation]:
        """Extract employment relations from document."""
        relations = []

        matches = self.matcher(doc)
        for match_id, token_ids in matches:
            verb = doc[token_ids[0]]
            subject = doc[token_ids[1]]
            obj = doc[token_ids[2]]

            # Verify entity types
            if subject.ent_type_ == "PERSON" and obj.ent_type_ == "ORG":
                relation_type = self._determine_relation_type(verb.lemma_)

                relations.append(Relation(
                    source_id=f"person_{hash(subject.text)}",
                    target_id=f"org_{hash(obj.text)}",
                    relation_type=relation_type,
                    confidence=0.8,
                    source_text=doc.text[subject.start_char:obj.end_char]
                ))

        return relations
```

### Example: Add Graph Query Pattern

**File:** `packages/vecpipe/graphrag/graph/queries.py`

```python
class GraphQueryBuilder:
    """Build graph traversal queries for search enhancement."""

    @staticmethod
    def find_related_entities(
        entity_ids: list[str],
        max_hops: int = 2,
        relation_types: list[str] | None = None
    ) -> str:
        """Cypher query to find entities within N hops."""
        relation_filter = ""
        if relation_types:
            types = "|".join(relation_types)
            relation_filter = f":{types}"

        return f"""
        MATCH (start)
        WHERE start.id IN $entity_ids
        MATCH path = (start)-[r{relation_filter}*1..{max_hops}]-(related)
        WHERE related <> start
        RETURN DISTINCT related,
               length(path) as distance,
               [r in relationships(path) | type(r)] as relation_path
        ORDER BY distance
        LIMIT 50
        """

    @staticmethod
    def find_connecting_paths(
        entity_a_id: str,
        entity_b_id: str,
        max_length: int = 4
    ) -> str:
        """Find shortest paths between two entities."""
        return f"""
        MATCH (a {{id: $entity_a_id}}), (b {{id: $entity_b_id}})
        MATCH path = shortestPath((a)-[*..{max_length}]-(b))
        RETURN path, length(path) as path_length
        """
```

### Example: Enhanced GraphRAG Search

**File:** `packages/vecpipe/graphrag/services/graph_search_service.py`

```python
class GraphSearchService:
    """Combines vector search with graph traversal."""

    async def search_with_graph_context(
        self,
        query: str,
        collections: list[str],
        top_k: int = 10,
        expand_entities: bool = True,
        max_hops: int = 2
    ) -> GraphSearchResponse:
        """Search with automatic entity expansion."""

        # 1. Extract entities from query
        query_entities = await self.entity_extractor.extract(query)

        # 2. Standard vector search
        vector_results = await self.vector_search(
            query=query,
            collections=collections,
            limit=top_k * 2  # Fetch extra for graph filtering
        )

        # 3. If entities found, expand via graph
        if expand_entities and query_entities:
            # Find related entities in graph
            related = await self.graph_client.query(
                GraphQueryBuilder.find_related_entities(
                    entity_ids=[e.id for e in query_entities],
                    max_hops=max_hops
                )
            )

            # Get documents containing related entities
            related_doc_ids = {r.source_doc_id for r in related}

            # Boost results from related documents
            for result in vector_results:
                if result.doc_id in related_doc_ids:
                    result.score *= 1.2  # 20% boost

        # 4. Extract entities from results for context
        result_entities = []
        for result in vector_results[:top_k]:
            entities = await self.entity_extractor.extract(result.content)
            result_entities.extend(entities)

        return GraphSearchResponse(
            results=vector_results[:top_k],
            query_entities=query_entities,
            related_entities=result_entities[:20],
            graph_context=self._build_context_summary(
                query_entities,
                result_entities
            )
        )
```

### Frontend: Visualize Graph Context

**File:** `apps/webui-react/src/components/search/GraphContext.tsx`

```typescript
import { useCallback } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls
} from 'reactflow';

interface GraphContextProps {
  queryEntities: Entity[];
  relatedEntities: Entity[];
  relations: Relation[];
}

export function GraphContext({
  queryEntities,
  relatedEntities,
  relations
}: GraphContextProps) {
  const { nodes, edges } = useMemo(() => {
    const nodes: Node[] = [
      // Query entities (highlighted)
      ...queryEntities.map((e, i) => ({
        id: e.id,
        type: 'entity',
        position: { x: 100, y: i * 80 },
        data: { ...e, isQuery: true }
      })),
      // Related entities
      ...relatedEntities.map((e, i) => ({
        id: e.id,
        type: 'entity',
        position: { x: 400, y: i * 60 },
        data: { ...e, isQuery: false }
      }))
    ];

    const edges: Edge[] = relations.map(r => ({
      id: `${r.sourceId}-${r.targetId}`,
      source: r.sourceId,
      target: r.targetId,
      label: r.relationType,
      animated: true
    }));

    return { nodes, edges };
  }, [queryEntities, relatedEntities, relations]);

  return (
    <div className="h-64 border rounded-lg">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={{ entity: EntityNode }}
        fitView
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
}
```

### Checklist for GraphRAG Extensions

- [ ] Define new entity type with attributes
- [ ] Implement entity extractor (spaCy patterns or ML model)
- [ ] Create relation extractor if needed
- [ ] Add graph query patterns
- [ ] Update search service to use new entities
- [ ] Add frontend visualization
- [ ] Write tests for extraction accuracy
- [ ] Document entity schema

---

## 4. Add Search Filters

### Overview

Search filters allow users to narrow results by metadata. This guide shows how to add filtering support.

### Step 1: Define Filter Schema

**File:** `packages/shared/contracts/search.py`

```python
from pydantic import BaseModel
from typing import Any
from enum import Enum

class FilterOperator(str, Enum):
    EQ = "eq"          # Equals
    NE = "ne"          # Not equals
    GT = "gt"          # Greater than
    GTE = "gte"        # Greater than or equal
    LT = "lt"          # Less than
    LTE = "lte"        # Less than or equal
    IN = "in"          # In list
    CONTAINS = "contains"  # String contains
    RANGE = "range"    # Between two values

class SearchFilter(BaseModel):
    field: str
    operator: FilterOperator
    value: Any

    def to_qdrant_filter(self) -> dict:
        """Convert to Qdrant filter format."""
        if self.operator == FilterOperator.EQ:
            return {"key": self.field, "match": {"value": self.value}}
        elif self.operator == FilterOperator.IN:
            return {"key": self.field, "match": {"any": self.value}}
        elif self.operator == FilterOperator.RANGE:
            return {
                "key": self.field,
                "range": {"gte": self.value[0], "lte": self.value[1]}
            }
        # ... other operators

class SearchFilters(BaseModel):
    filters: list[SearchFilter] = []
    must_match_all: bool = True  # AND vs OR logic

    def to_qdrant_filter(self) -> dict | None:
        if not self.filters:
            return None

        qdrant_filters = [f.to_qdrant_filter() for f in self.filters]

        if self.must_match_all:
            return {"must": qdrant_filters}
        else:
            return {"should": qdrant_filters}
```

### Step 2: Apply in VecPipe

**File:** `packages/vecpipe/services/search_service.py`

```python
async def _vector_search(
    self,
    query: str,
    collections: list[str],
    limit: int,
    filters: SearchFilters | None = None
) -> list[SearchResult]:
    query_embedding = await self.embedding_service.embed_query(query)

    qdrant_filter = filters.to_qdrant_filter() if filters else None

    results = await self.qdrant_client.search(
        collection_name=collections[0],
        query_vector=query_embedding,
        limit=limit,
        query_filter=qdrant_filter
    )

    return [self._to_search_result(r) for r in results]
```

### Step 3: Frontend Filter Builder

**File:** `apps/webui-react/src/components/search/FilterBuilder.tsx`

```typescript
interface FilterBuilderProps {
  availableFields: FieldDefinition[];
  filters: SearchFilter[];
  onChange: (filters: SearchFilter[]) => void;
}

export function FilterBuilder({
  availableFields,
  filters,
  onChange
}: FilterBuilderProps) {
  const addFilter = () => {
    onChange([...filters, { field: '', operator: 'eq', value: '' }]);
  };

  const removeFilter = (index: number) => {
    onChange(filters.filter((_, i) => i !== index));
  };

  const updateFilter = (index: number, update: Partial<SearchFilter>) => {
    onChange(filters.map((f, i) => i === index ? { ...f, ...update } : f));
  };

  return (
    <div className="space-y-2">
      {filters.map((filter, index) => (
        <div key={index} className="flex gap-2 items-center">
          <Select
            value={filter.field}
            onValueChange={(v) => updateFilter(index, { field: v })}
          >
            {availableFields.map(f => (
              <SelectItem key={f.name} value={f.name}>
                {f.label}
              </SelectItem>
            ))}
          </Select>

          <Select
            value={filter.operator}
            onValueChange={(v) => updateFilter(index, { operator: v })}
          >
            <SelectItem value="eq">equals</SelectItem>
            <SelectItem value="contains">contains</SelectItem>
            <SelectItem value="gt">greater than</SelectItem>
            <SelectItem value="range">between</SelectItem>
          </Select>

          <FilterValueInput
            field={availableFields.find(f => f.name === filter.field)}
            operator={filter.operator}
            value={filter.value}
            onChange={(v) => updateFilter(index, { value: v })}
          />

          <Button
            variant="ghost"
            size="icon"
            onClick={() => removeFilter(index)}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      ))}

      <Button variant="outline" onClick={addFilter}>
        <Plus className="h-4 w-4 mr-2" /> Add Filter
      </Button>
    </div>
  );
}
```

---

## 5. Implement Search Analytics

### Overview

Track search behavior to improve relevance and understand user needs.

### Step 1: Define Analytics Events

**File:** `packages/shared/contracts/analytics.py`

```python
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

class SearchEventType(str, Enum):
    SEARCH = "search"
    CLICK = "click"
    DWELL = "dwell"
    FEEDBACK = "feedback"

class SearchEvent(BaseModel):
    event_type: SearchEventType
    session_id: str
    user_id: int | None
    timestamp: datetime

    # Search context
    query: str
    collections: list[str]
    search_mode: str
    filters_used: dict | None

    # Results context
    result_count: int
    result_ids: list[str] | None

    # Interaction context (for click/dwell)
    clicked_result_id: str | None
    clicked_position: int | None
    dwell_time_seconds: float | None

    # Feedback
    relevance_rating: int | None  # 1-5
```

### Step 2: Backend Analytics Service

**File:** `packages/webui/services/analytics_service.py`

```python
class SearchAnalyticsService:
    def __init__(self, db: AsyncSession, redis: Redis):
        self.db = db
        self.redis = redis

    async def log_search(
        self,
        session_id: str,
        user_id: int | None,
        query: str,
        results: list[SearchResult],
        **context
    ) -> None:
        """Log a search event."""
        event = SearchEvent(
            event_type=SearchEventType.SEARCH,
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            query=query,
            result_count=len(results),
            result_ids=[r.id for r in results[:20]],
            **context
        )

        # Store in Redis for real-time analytics
        await self.redis.xadd(
            "search:events",
            event.dict(),
            maxlen=100000  # Keep last 100k events
        )

        # Batch write to database
        await self._queue_for_batch_insert(event)

    async def log_click(
        self,
        session_id: str,
        query: str,
        result_id: str,
        position: int
    ) -> None:
        """Log a result click."""
        event = SearchEvent(
            event_type=SearchEventType.CLICK,
            session_id=session_id,
            timestamp=datetime.utcnow(),
            query=query,
            clicked_result_id=result_id,
            clicked_position=position,
            result_count=0,
            collections=[]
        )

        await self.redis.xadd("search:events", event.dict())

    async def get_popular_queries(
        self,
        days: int = 7,
        limit: int = 20
    ) -> list[dict]:
        """Get most popular queries."""
        return await self.db.execute("""
            SELECT query, COUNT(*) as count
            FROM search_events
            WHERE timestamp > NOW() - INTERVAL :days DAY
              AND event_type = 'search'
            GROUP BY query
            ORDER BY count DESC
            LIMIT :limit
        """, {"days": days, "limit": limit})
```

### Step 3: Frontend Analytics Hook

**File:** `apps/webui-react/src/hooks/useSearchAnalytics.ts`

```typescript
export function useSearchAnalytics() {
  const sessionId = useSessionId();
  const { mutate: logEvent } = useMutation({
    mutationFn: (event: SearchEvent) =>
      apiClient.post('/analytics/search', event)
  });

  const logSearch = useCallback((
    query: string,
    results: SearchResult[],
    context: SearchContext
  ) => {
    logEvent({
      eventType: 'search',
      sessionId,
      query,
      resultCount: results.length,
      resultIds: results.slice(0, 20).map(r => r.id),
      ...context
    });
  }, [sessionId, logEvent]);

  const logClick = useCallback((
    query: string,
    resultId: string,
    position: number
  ) => {
    logEvent({
      eventType: 'click',
      sessionId,
      query,
      clickedResultId: resultId,
      clickedPosition: position
    });
  }, [sessionId, logEvent]);

  return { logSearch, logClick };
}
```

### Checklist for Search Analytics

- [ ] Define event schema
- [ ] Create backend analytics service
- [ ] Set up Redis streams for real-time events
- [ ] Create database table for historical data
- [ ] Add frontend tracking hooks
- [ ] Build analytics dashboard
- [ ] Add A/B testing support
- [ ] Create relevance improvement pipeline
