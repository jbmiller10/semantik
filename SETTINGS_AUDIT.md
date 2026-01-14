# Settings Audit

Comprehensive catalog of settings, constants, and configurable values in the Semantik codebase.

**Legend:**
- ✅ Exposed in Settings UI
- ⚠️ Configurable via environment variable only
- ❌ Hardcoded constant

---

## 1. User Preferences (Search)

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| search_top_k | 10 | `shared/database/models.py:1134` | ✅ |
| search_mode | "dense" | `shared/database/models.py:1134` | ✅ |
| search_use_reranker | false | `shared/database/models.py:1134` | ✅ |
| search_rrf_k | 60 | `shared/database/models.py:1134` | ✅ |
| search_similarity_threshold | null | `shared/database/models.py:1134` | ✅ |

---

## 2. User Preferences (Collection Defaults)

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| default_embedding_model | null (system default) | `shared/database/models.py:1134` | ✅ |
| default_quantization | "float16" | `shared/database/models.py:1134` | ✅ |
| default_chunking_strategy | "recursive" | `shared/database/models.py:1134` | ✅ |
| default_chunk_size | 1024 | `shared/database/models.py:1134` | ✅ |
| default_chunk_overlap | 200 | `shared/database/models.py:1134` | ✅ |
| default_enable_sparse | false | `shared/database/models.py:1134` | ✅ |
| default_sparse_type | "bm25" | `shared/database/models.py:1134` | ✅ |
| default_enable_hybrid | false | `shared/database/models.py:1134` | ✅ |

---

## 3. LLM Configuration

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| API keys (Anthropic/OpenAI) | - | `llm_provider_api_keys` table | ✅ |
| high_quality_provider | "anthropic" | `llm_provider_configs` table | ✅ |
| high_quality_model | - | `llm_provider_configs` table | ✅ |
| low_quality_provider | "anthropic" | `llm_provider_configs` table | ✅ |
| low_quality_model | - | `llm_provider_configs` table | ✅ |
| default_temperature | null | `llm_provider_configs` table | ✅ |
| default_max_tokens | null | `llm_provider_configs` table | ✅ |

---

## 4. User Resource Limits

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| MAX_COLLECTIONS_PER_USER | 10 | `shared/config/webui.py:37` | ⚠️ |
| MAX_STORAGE_GB_PER_USER | 50.0 | `shared/config/webui.py:38` | ⚠️ |
| MAX_ARTIFACT_BYTES | 50MB | `shared/config/webui.py:68` | ⚠️ |

---

## 5. Authentication

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| ACCESS_TOKEN_EXPIRE_MINUTES | 1440 (24h) | `shared/config/webui.py:18` | ⚠️ |
| JWT_SECRET_KEY | (required) | `shared/config/webui.py` | ⚠️ |
| ALGORITHM | "HS256" | `shared/config/webui.py` | ⚠️ |
| DISABLE_AUTH | false | `shared/config/webui.py` | ⚠️ |

---

## 6. Embedding & Model Configuration

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| DEFAULT_EMBEDDING_MODEL | "Qwen/Qwen3-Embedding-0.6B" | `shared/config/vecpipe.py:16` | ⚠️ |
| DEFAULT_QUANTIZATION | "float16" | `shared/config/vecpipe.py:17` | ⚠️ |
| USE_MOCK_EMBEDDINGS | false | `shared/config/vecpipe.py:15` | ⚠️ |
| MODEL_UNLOAD_AFTER_SECONDS | 300 (5 min) | `shared/config/vecpipe.py:28` | ⚠️ |

---

## 7. GPU Memory Management

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| ENABLE_MEMORY_GOVERNOR | true | `shared/config/vecpipe.py:31` | ⚠️ |
| GPU_MEMORY_MAX_PERCENT | 0.90 | `shared/config/vecpipe.py:32` | ⚠️ |
| CPU_MEMORY_MAX_PERCENT | 0.50 | `shared/config/vecpipe.py:33` | ⚠️ |
| ENABLE_CPU_OFFLOAD | true | `shared/config/vecpipe.py:34` | ⚠️ |
| EVICTION_IDLE_THRESHOLD_SECONDS | 120 | `shared/config/vecpipe.py:35` | ⚠️ |
| PRESSURE_CHECK_INTERVAL_SECONDS | 15 | `shared/config/vecpipe.py:36` | ⚠️ |

---

## 8. Adaptive Batch Sizing

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| ENABLE_ADAPTIVE_BATCH_SIZE | true | `shared/config/vecpipe.py:41` | ⚠️ |
| MIN_BATCH_SIZE | 1 | `shared/config/vecpipe.py:42` | ⚠️ |
| MAX_BATCH_SIZE | 256 | `shared/config/vecpipe.py:43` | ⚠️ |
| BATCH_SIZE_SAFETY_MARGIN | 0.2 | `shared/config/vecpipe.py:44` | ⚠️ |
| BATCH_SIZE_INCREASE_THRESHOLD | 10 | `shared/config/vecpipe.py:45` | ⚠️ |

---

## 9. Search Tuning

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| SEARCH_CANDIDATE_MULTIPLIER | 3 | `shared/config/webui.py:34` | ⚠️ |
| rerank_candidate_multiplier | 5 | `vecpipe/qwen3_search_config.py:93` | ❌ |
| rerank_min_candidates | 20 | `vecpipe/qwen3_search_config.py:94` | ❌ |
| rerank_max_candidates | 200 | `vecpipe/qwen3_search_config.py:95` | ❌ |
| rerank_hybrid_weight | 0.3 | `vecpipe/qwen3_search_config.py:98` | ❌ |
| slow_query_threshold_ms | 1000 | `vecpipe/qwen3_search_config.py:131` | ❌ |

---

## 10. Cache Configuration

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| CACHE_DEFAULT_TTL_SECONDS | 300 (5 min) | `shared/config/webui.py:41` | ⚠️ |
| CACHE_TTL_SECONDS (vecpipe) | 300 (5 min) | `vecpipe/search/cache.py:17` | ❌ |
| MAX_CACHE_ENTRIES | 100 | `vecpipe/search/cache.py:20` | ❌ |
| PREVIEW_CACHE_TTL_SECONDS | 900 (15 min) | `webui/services/chunking_config.py:60` | ❌ |
| OPERATION_STATE_TTL_SECONDS | 86400 (24h) | `webui/services/chunking_config.py:61` | ❌ |
| ERROR_HISTORY_TTL_SECONDS | 604800 (7d) | `webui/services/chunking_config.py:62` | ❌ |
| STRATEGY_CACHE_TTL | 3600 (1h) | `webui/services/chunking_constants.py:83` | ❌ |
| METRICS_CACHE_TTL | 300 (5 min) | `webui/services/chunking_constants.py:84` | ❌ |

---

## 11. Document Size Limits

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| MAX_DOCUMENT_SIZE | 100MB | `webui/services/chunking_constants.py:12` | ❌ |
| MAX_PREVIEW_CONTENT_SIZE | 10MB | `webui/services/chunking_constants.py:11` | ❌ |
| MIN_CHUNK_SIZE | 100 | `webui/services/chunking_constants.py:13` | ❌ |
| MAX_CHUNK_SIZE | 4096 | `webui/services/chunking_constants.py:14` | ❌ |
| DEFAULT_CHUNK_SIZE | 512 | `webui/services/chunking_constants.py:15` | ❌ |
| DEFAULT_CHUNK_OVERLAP | 50 | `webui/services/chunking_constants.py:16` | ❌ |
| MAX_CHUNK_OVERLAP | 500 | `webui/services/chunking_constants.py:17` | ❌ |

---

## 12. Segmentation (Large Documents)

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| SEGMENT_SIZE_THRESHOLD | 5MB | `webui/services/chunking_constants.py:20` | ❌ |
| DEFAULT_SEGMENT_SIZE | 1MB | `webui/services/chunking_constants.py:21` | ❌ |
| DEFAULT_SEGMENT_OVERLAP | 10KB | `webui/services/chunking_constants.py:22` | ❌ |
| MAX_SEGMENTS_PER_DOCUMENT | 100 | `webui/services/chunking_constants.py:23` | ❌ |
| semantic strategy threshold | 2MB | `webui/services/chunking_constants.py:27` | ❌ |
| markdown strategy threshold | 10MB | `webui/services/chunking_constants.py:28` | ❌ |
| recursive strategy threshold | 8MB | `webui/services/chunking_constants.py:29` | ❌ |

---

## 13. Preview Configuration

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| MAX_PREVIEW_CHUNKS | 50 | `webui/services/chunking_constants.py:41` | ❌ |
| DEFAULT_PREVIEW_CHUNKS | 10 | `webui/services/chunking_constants.py:42` | ❌ |
| MAX_PREVIEW_LENGTH | 1000 chars | `webui/services/chunking_constants.py:44` | ❌ |
| PREVIEW_TIMEOUT_SECONDS | 30.0 | `webui/services/chunking_config.py:41` | ❌ |

---

## 14. Operation Limits

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| MAX_CONCURRENT_OPERATIONS | 10 | `webui/services/chunking_constants.py:58` | ❌ |
| MAX_CONCURRENT_OPERATIONS_PER_USER | 3 | `webui/services/chunking_config.py:32` | ❌ |
| MAX_QUEUED_OPERATIONS | 50 | `webui/services/chunking_config.py:33` | ❌ |
| MAX_CHUNKS_PER_DOCUMENT | 10000 | `webui/services/chunking_config.py:26` | ❌ |
| MAX_CHUNKS_PER_OPERATION | 100000 | `webui/services/chunking_config.py:27` | ❌ |
| OPERATION_TIMEOUT | 3600 (1h) | `webui/services/chunking_constants.py:62` | ❌ |

---

## 15. Celery Task Configuration

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| CHUNKING_SOFT_TIME_LIMIT | 3600 (1h) | `webui/chunking_tasks.py:153` | ❌ |
| CHUNKING_HARD_TIME_LIMIT | 7200 (2h) | `webui/chunking_tasks.py:154` | ❌ |
| CHUNKING_MAX_RETRIES | 3 | `webui/chunking_tasks.py:155` | ❌ |
| CHUNKING_MEMORY_LIMIT_GB | 4 | `webui/chunking_tasks.py:158` | ❌ |
| CHUNKING_CPU_TIME_LIMIT | 1800 (30 min) | `webui/chunking_tasks.py:159` | ❌ |

---

## 16. Retry & Recovery

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| DEFAULT_MAX_RETRIES | 3 | `webui/services/chunking_config.py:74` | ❌ |
| MEMORY_ERROR_MAX_RETRIES | 2 | `webui/services/chunking_config.py:75` | ❌ |
| TIMEOUT_ERROR_MAX_RETRIES | 3 | `webui/services/chunking_config.py:76` | ❌ |
| NETWORK_ERROR_MAX_RETRIES | 5 | `webui/services/chunking_config.py:77` | ❌ |
| RETRY_BACKOFF_MAX_SECONDS | 600 (10 min) | `webui/services/chunking_config.py:81` | ❌ |
| MIN_RETRY_DELAY_SECONDS | 10 | `webui/services/chunking_config.py:50` | ❌ |
| MAX_RETRY_DELAY_SECONDS | 300 (5 min) | `webui/services/chunking_config.py:51` | ❌ |

---

## 17. Batch Processing

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| DEFAULT_BATCH_SIZE | 32 | `webui/services/chunking_config.py:90` | ❌ |
| REDUCED_BATCH_SIZE | 16 | `webui/services/chunking_config.py:91` | ❌ |
| MIN_BATCH_SIZE (chunking) | 4 | `webui/services/chunking_config.py:92` | ❌ |
| EMBEDDING_BATCH_SIZE | 100 | `webui/services/chunking_config.py:95` | ❌ |
| VECTOR_UPLOAD_BATCH_SIZE | 100 | `webui/services/chunking_config.py:96` | ❌ |
| DOCUMENT_REMOVAL_BATCH_SIZE | 100 | `webui/services/chunking_config.py:97` | ❌ |
| BATCH_SIZE (qdrant ingest) | 4000 | `vecpipe/ingest_qdrant.py:25` | ❌ |

---

## 18. Circuit Breaker

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| CIRCUIT_BREAKER_FAILURES | 5 | `webui/config/rate_limits.py:31` | ⚠️ |
| CIRCUIT_BREAKER_TIMEOUT | 60s | `webui/config/rate_limits.py:32` | ⚠️ |
| CB_FAILURE_THRESHOLD | 5 | `webui/services/chunking_config.py:118` | ❌ |
| CB_SUCCESS_THRESHOLD | 2 | `webui/services/chunking_config.py:119` | ❌ |
| CB_TIMEOUT_SECONDS | 60 | `webui/services/chunking_config.py:120` | ❌ |
| CB_HALF_OPEN_REQUESTS | 3 | `webui/services/chunking_config.py:121` | ❌ |

---

## 19. Rate Limiting

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| CHUNKING_PREVIEW_RATE_LIMIT | 10/minute | `webui/config/rate_limits.py:18` | ⚠️ |
| CHUNKING_COMPARE_RATE_LIMIT | 5/minute | `webui/config/rate_limits.py:19` | ⚠️ |
| CHUNKING_PROCESS_RATE_LIMIT | 20/hour | `webui/config/rate_limits.py:20` | ⚠️ |
| CHUNKING_READ_RATE_LIMIT | 60/minute | `webui/config/rate_limits.py:21` | ⚠️ |
| CHUNKING_ANALYTICS_RATE_LIMIT | 30/minute | `webui/config/rate_limits.py:22` | ⚠️ |
| PLUGIN_INSTALL_RATE_LIMIT | 2/minute | `webui/config/rate_limits.py:42` | ⚠️ |
| PLUGIN_UNINSTALL_RATE_LIMIT | 5/minute | `webui/config/rate_limits.py:43` | ⚠️ |
| PLUGIN_HEALTH_RATE_LIMIT | 30/minute | `webui/config/rate_limits.py:44` | ⚠️ |
| PLUGIN_LIST_RATE_LIMIT | 60/minute | `webui/config/rate_limits.py:45` | ⚠️ |
| LLM_TEST_RATE_LIMIT | 5/minute | `webui/config/rate_limits.py:54` | ⚠️ |

---

## 20. WebSocket Configuration

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| MAX_WEBSOCKET_CONNECTIONS_PER_USER | 10 | `webui/services/chunking_constants.py:65` | ❌ |
| MAX_TOTAL_WEBSOCKET_CONNECTIONS | 1000 | `webui/services/chunking_constants.py:66` | ❌ |
| WEBSOCKET_PROGRESS_THROTTLE_MS | 500 | `webui/services/chunking_constants.py:67` | ❌ |
| WEBSOCKET_PING_INTERVAL | 30s | `webui/services/chunking_constants.py:68` | ❌ |
| WEBSOCKET_TIMEOUT | 60s | `webui/services/chunking_constants.py:69` | ❌ |
| Connection timeout (frontend) | 5000ms | `webui-react/hooks/useWebSocket.ts:77` | ❌ |
| Reconnect interval (frontend) | 3000ms | `webui-react/hooks/useWebSocket.ts:25` | ❌ |
| Reconnect attempts (frontend) | 5 | `webui-react/hooks/useWebSocket.ts:26` | ❌ |

---

## 21. Database Connection Pool

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| DB_POOL_SIZE | 20 | `shared/config/postgres.py:30` | ⚠️ |
| DB_MAX_OVERFLOW | 40 | `shared/config/postgres.py:31` | ⚠️ |
| DB_POOL_TIMEOUT | 30s | `shared/config/postgres.py:32` | ⚠️ |
| DB_POOL_RECYCLE | 3600s | `shared/config/postgres.py:33` | ⚠️ |
| DB_POOL_PRE_PING | true | `shared/config/postgres.py:34` | ⚠️ |
| DB_IDLE_IN_TX_TIMEOUT_MS | 60000 | `shared/config/postgres.py:38` | ⚠️ |
| DB_QUERY_TIMEOUT | 30s | `shared/config/postgres.py:45` | ⚠️ |
| DB_RETRY_LIMIT | 3 | `shared/config/postgres.py:48` | ⚠️ |
| DB_RETRY_INTERVAL | 0.5s | `shared/config/postgres.py:49` | ⚠️ |
| CHUNK_PARTITION_COUNT | 100 | `shared/config/postgres.py:53` | ⚠️ |

---

## 22. Redis Configuration

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| REDIS_URL | redis://localhost:6379/0 | `shared/config/webui.py` | ⚠️ |
| REDIS_CLEANUP_INTERVAL_SECONDS | 60 | `shared/config/webui.py:47` | ⚠️ |
| REDIS_CLEANUP_MAX_CONSECUTIVE_FAILURES | 5 | `shared/config/webui.py:48` | ⚠️ |
| REDIS_CLEANUP_BACKOFF_MULTIPLIER | 2.0 | `shared/config/webui.py:49` | ⚠️ |
| REDIS_CLEANUP_MAX_BACKOFF_SECONDS | 300 | `shared/config/webui.py:50` | ⚠️ |
| REDIS_MAX_CONNECTIONS | 50 | `webui/services/chunking_constants.py:108` | ❌ |
| REDIS_HEALTH_CHECK_INTERVAL | 30s | `webui/services/chunking_constants.py:109` | ❌ |

---

## 23. Monitoring Thresholds

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| HIGH_MEMORY_THRESHOLD | 0.8 (80%) | `webui/services/chunking_constants.py:77` | ❌ |
| HIGH_CPU_THRESHOLD | 0.9 (90%) | `webui/services/chunking_constants.py:78` | ❌ |
| ERROR_RATE_THRESHOLD | 0.1 (10%) | `webui/services/chunking_config.py:108` | ❌ |
| MEMORY_USAGE_THRESHOLD | 0.8 (80%) | `webui/services/chunking_config.py:109` | ❌ |
| QUEUE_SIZE_THRESHOLD | 40 | `webui/services/chunking_config.py:110` | ❌ |
| HEALTH_CHECK_TIMEOUT | 5.0s | `webui/api/health.py:20` | ❌ |

---

## 24. Partition Monitoring

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| SKEW_WARNING_THRESHOLD | 0.3 (30%) | `webui/services/partition_monitoring_service.py:78` | ❌ |
| SKEW_CRITICAL_THRESHOLD | 0.5 (50%) | `webui/services/partition_monitoring_service.py:79` | ❌ |
| REBALANCE_THRESHOLD | 0.4 (40%) | `webui/services/partition_monitoring_service.py:80` | ❌ |

---

## 25. Reindexing Validation

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| REINDEX_VECTOR_COUNT_VARIANCE | 0.1 (10%) | `webui/services/chunking_config.py:135` | ❌ |
| REINDEX_SEARCH_MISMATCH_THRESHOLD | 0.3 (30%) | `webui/services/chunking_config.py:136` | ❌ |
| REINDEX_SCORE_DIFF_THRESHOLD | 0.1 | `webui/services/chunking_config.py:137` | ❌ |

---

## 26. Cleanup Timings

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| CLEANUP_DELAY_SECONDS | 300 (5 min) | `webui/services/chunking_config.py:141` | ❌ |
| CLEANUP_DELAY_MIN_SECONDS | 300 (5 min) | `webui/services/chunking_config.py:142` | ❌ |
| CLEANUP_DELAY_MAX_SECONDS | 1800 (30 min) | `webui/services/chunking_config.py:143` | ❌ |
| CLEANUP_DELAY_PER_10K_VECTORS | 60s | `webui/services/chunking_config.py:144` | ❌ |

---

## 27. Memory Limits

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| PREVIEW_MEMORY_LIMIT_BYTES | 512MB | `webui/services/chunking_config.py:17` | ❌ |
| OPERATION_MEMORY_LIMIT_BYTES | 2GB | `webui/services/chunking_config.py:18` | ❌ |
| MIN_MEMORY_LIMIT_BYTES | 100MB | `webui/services/chunking_config.py:19` | ❌ |
| MAX_CHUNK_SIZE_BYTES | 50KB | `webui/services/chunking_config.py:28` | ❌ |

---

## 28. Streaming Configuration

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| STREAMING_ENABLED | true | `webui/services/chunking_constants.py:35` | ❌ |
| STREAMING_BUFFER_SIZE | 64KB | `webui/services/chunking_constants.py:37` | ❌ |
| STREAMING_WINDOW_SIZE | 256KB | `webui/services/chunking_constants.py:38` | ❌ |
| CHECKPOINT_INTERVAL | 100MB | `shared/chunking/infrastructure/streaming/processor.py:49` | ❌ |
| PROGRESS_INTERVAL | 1.0s | `shared/chunking/infrastructure/streaming/processor.py:50` | ❌ |

---

## 29. Quality Scoring

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| GOOD_QUALITY_THRESHOLD | 0.7 | `webui/services/chunking_constants.py:89` | ❌ |
| EXCELLENT_QUALITY_THRESHOLD | 0.9 | `webui/services/chunking_constants.py:90` | ❌ |

---

## 30. Pagination

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| DEFAULT_PAGE_SIZE | 20 | `webui/services/chunking_constants.py:53` | ❌ |
| MAX_PAGE_SIZE | 100 | `webui/services/chunking_constants.py:54` | ❌ |
| Documents per page (frontend) | 50 | `webui-react/hooks/useCollectionDocuments.ts:26` | ❌ |

---

## 31. Frontend Cache Timings (React Query)

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| Collections staleTime | 5000ms | `webui-react/hooks/useCollections.ts:41` | ❌ |
| Collections refetchInterval (active) | 5000ms | `webui-react/hooks/useCollections.ts:39` | ❌ |
| Collections refetchInterval (inactive) | 30000ms | `webui-react/hooks/useCollections.ts:39` | ❌ |
| Models staleTime | 300000ms (5 min) | `webui-react/hooks/useModels.ts:29` | ❌ |
| Models gcTime | 600000ms (10 min) | `webui-react/hooks/useModels.ts:30` | ❌ |
| Preferences staleTime | 300000ms (5 min) | `webui-react/hooks/usePreferences.ts:31` | ❌ |
| SystemInfo staleTime | 300000ms (5 min) | `webui-react/hooks/useSystemInfo.ts:30` | ❌ |
| SystemInfo gcTime | 1800000ms (30 min) | `webui-react/hooks/useSystemInfo.ts:31` | ❌ |
| SystemHealth staleTime | 15000ms (15s) | `webui-react/hooks/useSystemInfo.ts:46` | ❌ |
| SystemHealth refetchInterval | 30000ms (30s) | `webui-react/hooks/useSystemInfo.ts:47` | ❌ |
| SystemStatus staleTime | 60000ms (1 min) | `webui-react/hooks/useSystemInfo.ts:61` | ❌ |
| MCPProfiles staleTime | 30000ms (30s) | `webui-react/hooks/useMCPProfiles.ts:41` | ❌ |
| LLMModels staleTime | Infinity | `webui-react/hooks/useLLMSettings.ts:91` | ❌ |
| LLMModels gcTime | 1800000ms (30 min) | `webui-react/hooks/useLLMSettings.ts:92` | ❌ |
| LLMUsage staleTime | 60000ms (1 min) | `webui-react/hooks/useLLMSettings.ts:155` | ❌ |
| CollectionDocuments staleTime | 30000ms (30s) | `webui-react/hooks/useCollectionDocuments.ts:40` | ❌ |
| Projections refetchInterval | 5000ms | `webui-react/hooks/useProjections.ts:11` | ❌ |
| Connectors staleTime | 300000ms (5 min) | `webui-react/hooks/useConnectors.ts:31` | ❌ |
| Plugins staleTime | 30000ms (30s) | `webui-react/hooks/usePlugins.ts:41` | ❌ |

---

## 32. Frontend Visualization

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| SAMPLE_LIMIT_CAP | 200000 | `webui-react/components/EmbeddingVisualizationTab.tsx:169` | ❌ |
| DENSITY_THRESHOLD | 20000 | `webui-react/components/EmbeddingVisualizationTab.tsx:174` | ❌ |
| Visualization size | 960x540 px | `webui-react/components/EmbeddingVisualizationTab.tsx:292` | ❌ |

---

## 33. Frontend Tooltip/Projection

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| TOOLTIP_CACHE_SIZE | 512 | `webui-react/hooks/useProjectionTooltip.ts:26` | ❌ |
| TOOLTIP_CACHE_TTL_MS | 60000 (1 min) | `webui-react/hooks/useProjectionTooltip.ts:27` | ❌ |
| TOOLTIP_DEBOUNCE_MS | 50 | `webui-react/hooks/useProjectionTooltip.ts:28` | ❌ |
| TOOLTIP_MAX_INFLIGHT | 5 | `webui-react/hooks/useProjectionTooltip.ts:29` | ❌ |

---

## 34. Frontend Search Validation

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| Query minLength | 1 | `webui-react/utils/searchValidation.ts` | ❌ |
| Query maxLength | 1000 | `webui-react/utils/searchValidation.ts` | ❌ |
| TopK min | 1 | `webui-react/utils/searchValidation.ts` | ❌ |
| TopK max | 100 | `webui-react/utils/searchValidation.ts` | ❌ |
| Max collections per search | 10 | `webui-react/utils/searchValidation.ts` | ❌ |

---

## 35. BM25/RRF Defaults

| Setting | Default | Location | Status |
|---------|---------|----------|--------|
| BM25 k1 | 1.5 (range: 0.5-3.0) | `webui-react/types/sparse-index.ts` | ❌ |
| BM25 b | 0.75 (range: 0-1.0) | `webui-react/types/sparse-index.ts` | ❌ |
| RRF k default | 60 | `webui-react/types/sparse-index.ts` | ❌ |
| RRF k range | 1-1000 | `webui-react/types/sparse-index.ts` | ❌ |

---

## Recommendations

### High Priority - Expose in Settings UI

These affect user experience directly and users may want to customize:

1. **MAX_COLLECTIONS_PER_USER** - Power users may need more collections
2. **MAX_STORAGE_GB_PER_USER** - Storage limits vary by deployment
3. **ACCESS_TOKEN_EXPIRE_MINUTES** - Security/convenience tradeoff
4. **MAX_DOCUMENT_SIZE** - Some users have larger documents
5. **MODEL_UNLOAD_AFTER_SECONDS** - Memory vs latency tradeoff

### Medium Priority - Admin Settings Page

Operators may want to tune without code changes:

1. **GPU/Memory thresholds** - Hardware-dependent
2. **Rate limits** - Capacity planning
3. **WebSocket limits** - Concurrency control
4. **Cache TTLs** - Freshness vs performance

### Low Priority - Keep as Environment Variables

Infrastructure settings better managed at deployment:

1. **Database connection pool** - DBA concern
2. **Redis configuration** - Ops concern
3. **Celery timeouts** - Workload dependent
4. **Circuit breaker settings** - Stability tuning

### Keep Hardcoded

Some values should remain hardcoded:

1. **Algorithm/protocol constants** (HS256, partition counts)
2. **Internal retry logic** (backoff factors)
3. **Validation ranges** (chunk size bounds that preserve correctness)
4. **Frontend polling intervals** (React Query internals)
