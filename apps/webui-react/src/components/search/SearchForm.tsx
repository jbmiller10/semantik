import React, { useEffect, useState } from 'react';
import { Search, X, Loader2 } from 'lucide-react';
import { useSearchStore } from '../../stores/searchStore';
import { CollectionMultiSelect } from '../CollectionMultiSelect';
import SearchOptions from './SearchOptions';
import { SearchModeSelector } from './SearchModeSelector';
import { searchV2Api } from '../../services/api/v2/collections';
import { useUIStore } from '../../stores/uiStore';
import { sanitizeQuery } from '../../utils/searchValidation';
import { usePreferences } from '../../hooks/usePreferences';

import type { Collection } from '../../types/collection';
import type { SearchResult as ApiSearchResult } from '../../services/api/v2/types';
import type { SearchMode } from '../../types/sparse-index';

interface SearchFormProps {
    collections: Collection[];
}

interface SearchError {
    name?: string;
    message?: string;
    code?: string;
    response?: {
        status?: number;
        data?: {
            detail?: string | {
                error?: string;
                message?: string;
                suggestion?: string;
            };
        };
    };
}

export default function SearchForm({ collections }: SearchFormProps) {
    const {
        searchParams,
        loading,
        validateAndUpdateSearchParams,
        setResults,
        setLoading,
        setError,
        setFailedCollections,
        setPartialFailure,
        setGpuMemoryError,
        hasValidationErrors,
        setFieldTouched,
        getValidationError,
        abortController,
        setAbortController,
        setRerankingMetrics,
        setHydeUsed,
        setHydeInfo
    } = useSearchStore();

    const addToast = useUIStore((state) => state.addToast);

    // User preferences for search defaults
    const { data: prefs } = usePreferences();
    const [prefsInitialized, setPrefsInitialized] = useState(false);

    // Initialize search params from user preferences on first load
    useEffect(() => {
        if (prefs && !prefsInitialized) {
            validateAndUpdateSearchParams({
                topK: prefs.search.top_k,
                searchMode: prefs.search.mode,
                useReranker: prefs.search.use_reranker,
                rrfK: prefs.search.rrf_k,
                scoreThreshold: prefs.search.similarity_threshold ?? 0.0,
                useHyde: prefs.search.use_hyde,
            });
            setPrefsInitialized(true);
        }
    }, [prefs, prefsInitialized, validateAndUpdateSearchParams]);

    // Cleanup abort controller on unmount
    useEffect(() => {
        return () => {
            if (abortController) {
                abortController.abort();
            }
        };
    }, [abortController]);

    const handleSearch = async (e: React.FormEvent) => {
        e.preventDefault();

        // Mark all fields as touched to show validation errors
        setFieldTouched('query', true);
        setFieldTouched('collections', true);
        setFieldTouched('topK', true);
        setFieldTouched('scoreThreshold', true);
        setFieldTouched('hybridAlpha', true);

        // Validate before search
        if (hasValidationErrors()) {
            addToast({ type: 'error', message: 'Please fix validation errors before searching' });
            return;
        }

        if (!searchParams.query.trim()) {
            addToast({ type: 'error', message: 'Please enter a search query' });
            return;
        }

        if (searchParams.selectedCollections.length === 0) {
            addToast({ type: 'error', message: 'Please select at least one collection' });
            return;
        }

        // Cancel previous request if exists
        if (abortController) {
            abortController.abort();
        }

        const newController = new AbortController();
        setAbortController(newController);

        setLoading(true);
        setError(null);
        setFailedCollections([]);
        setPartialFailure(false);
        setGpuMemoryError(null);
        setResults([]); // Clear previous results
        setRerankingMetrics(null);

        try {
            const response = await searchV2Api.search(
                {
                    query: sanitizeQuery(searchParams.query),
                    collection_uuids: searchParams.selectedCollections,
                    k: searchParams.topK,
                    score_threshold: searchParams.scoreThreshold,
                    search_type: searchParams.searchType,
                    use_reranker: searchParams.useReranker,
                    rerank_model: searchParams.useReranker ? searchParams.rerankModel : null,
                    // New sparse/hybrid search parameters
                    search_mode: searchParams.searchMode,
                    rrf_k: searchParams.searchMode === 'hybrid' ? searchParams.rrfK : undefined,
                    // HyDE query expansion
                    use_hyde: searchParams.useHyde,
                    // Legacy hybrid parameters (deprecated - kept for backward compatibility)
                    hybrid_alpha: searchParams.searchType === 'hybrid' ? searchParams.hybridAlpha : undefined,
                    hybrid_mode: searchParams.searchType === 'hybrid' ? searchParams.hybridMode : undefined,
                    keyword_mode: searchParams.searchType === 'hybrid' ? searchParams.keywordMode : undefined,
                },
                { signal: newController.signal }
            );

            if (response.data) {
                // Map results preserving all fields
                const mappedResults = response.data.results.map((result: ApiSearchResult) => ({
                    doc_id: result.document_id,
                    chunk_id: result.chunk_id,
                    score: result.score,
                    content: result.text,
                    file_path: result.file_path,
                    file_name: result.file_name,
                    chunk_index: result.chunk_index ?? 0,
                    total_chunks: result.total_chunks ?? 1,
                    collection_id: result.collection_id,
                    collection_name: result.collection_name,
                    original_score: result.original_score,
                    reranked_score: result.reranked_score,
                    embedding_model: result.embedding_model,
                    metadata: result.metadata
                }));

                setResults(mappedResults);

                if (response.data.partial_failure) {
                    setPartialFailure(true);
                    setFailedCollections(response.data.failed_collections || []);
                    addToast({
                        type: 'warning',
                        message: 'Some collections failed to search. Showing partial results.'
                    });
                }

                if (response.data.reranking_time_ms) {
                    setRerankingMetrics({
                        rerankingUsed: true,
                        rerankingTimeMs: response.data.reranking_time_ms,
                        original_count: response.data.total_results, // Approximation
                        reranked_count: mappedResults.length
                    });
                }

                // Handle HyDE metadata
                setHydeUsed(response.data.hyde_used);
                setHydeInfo(response.data.hyde_info ?? null);

                // Show warnings from the backend (e.g., sparse fallback to dense)
                if (response.data.warnings && response.data.warnings.length > 0) {
                    response.data.warnings.forEach((warning: string) => {
                        addToast({ type: 'warning', message: warning, duration: 5000 });
                    });
                }
            }
        } catch (error: unknown) {
            const err = error as SearchError;
            if (err.name === 'CanceledError' || err.code === 'ERR_CANCELED' || err.message === 'canceled') {
                return; // Ignore cancellations
            }

            console.error('Search error:', error);

            // Handle specific error types
            if (err.response?.status === 500 && typeof err.response?.data?.detail === 'string' && err.response.data.detail.includes('GPU')) {
                setGpuMemoryError({
                    message: 'GPU memory limit exceeded',
                    suggestion: 'Try reducing the batch size or using a smaller model.',
                    currentModel: searchParams.rerankModel || 'unknown'
                });
                setError('GPU_MEMORY_ERROR');
            } else if (err.response?.status === 507) {
                // Handle insufficient storage/memory
                const detail = err.response?.data?.detail;
                if (typeof detail === 'object' && detail.error === 'insufficient_memory') {
                    setGpuMemoryError({
                        message: detail.message || 'Insufficient GPU memory',
                        suggestion: detail.suggestion || '',
                        currentModel: searchParams.rerankModel || 'Unknown'
                    });
                    setError('GPU_MEMORY_ERROR');
                    addToast({
                        type: 'error',
                        message: detail.message || 'Insufficient GPU memory',
                        duration: 5000
                    });
                } else {
                    setError(typeof detail === 'string' ? detail : 'Insufficient resources');
                }
            } else if (err.code === 'ECONNABORTED') {
                setError('Search timed out. Please try again with fewer collections or a simpler query.');
            } else {
                const detail = err.response?.data?.detail;
                const errorMessage = typeof detail === 'string' ? detail : 'An unexpected error occurred during search.';
                setError(errorMessage);
            }
        } finally {
            setLoading(false);
            setAbortController(null);
        }
    };

    const handleCancel = () => {
        if (abortController) {
            abortController.abort();
            setLoading(false);
            setAbortController(null);
            addToast({ type: 'info', message: 'Search cancelled' });
        }
    };

    return (
        <form onSubmit={handleSearch} className="space-y-8">
            {/* Search Input Group */}
            <div className="space-y-4">
                <div className="relative group">
                    <input
                        type="text"
                        value={searchParams.query}
                        onChange={(e) => {
                            setFieldTouched('query', true);
                            validateAndUpdateSearchParams({ query: e.target.value });
                        }}
                        placeholder="Enter your search query..."
                        className={`w-full pl-5 pr-12 py-4 text-lg border rounded-xl shadow-sm transition-all duration-200 outline-none
                            ${getValidationError('query')
                                ? 'border-red-500/50 bg-red-500/10 focus:ring-2 focus:ring-red-500/20 text-red-600 dark:text-red-200'
                                : 'input-field text-[var(--text-primary)] focus:ring-2 focus:ring-[var(--accent-primary)]/20 focus:border-[var(--accent-primary)]'
                            }`}
                        disabled={loading}
                    />
                    <div className="absolute right-4 top-1/2 -translate-y-1/2">
                        {loading ? (
                            <button
                                type="button"
                                onClick={handleCancel}
                                className="p-1.5 hover:bg-[var(--bg-tertiary)] rounded-full text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
                                title="Cancel search"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        ) : (
                            <Search className="w-6 h-6 text-[var(--text-muted)] group-focus-within:text-[var(--accent-primary)] transition-colors" />
                        )}
                    </div>
                </div>
                {getValidationError('query') && (
                    <p className="text-sm text-red-600 dark:text-red-400 font-medium ml-1">{getValidationError('query')}</p>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Collection Selector */}
                    <div className="lg:col-span-2 space-y-1.5">
                        <label className="block text-sm font-bold text-[var(--text-secondary)] uppercase tracking-wider ml-1">
                            Collections
                        </label>
                        <CollectionMultiSelect
                            collections={collections}
                            selectedCollections={searchParams.selectedCollections}
                            onChange={(selected) => {
                                setFieldTouched('collections', true);
                                validateAndUpdateSearchParams({ selectedCollections: selected });
                            }}
                        />
                        {getValidationError('collections') && (
                            <p className="text-sm text-red-600 font-medium ml-1">{getValidationError('collections')}</p>
                        )}
                    </div>

                    {/* Embedding Mode Selector */}
                    <div className="space-y-1.5">
                        <label htmlFor="search-type" className="block text-sm font-bold text-[var(--text-secondary)] uppercase tracking-wider ml-1">
                            Embedding Mode
                        </label>
                        <select
                            id="search-type"
                            value={searchParams.searchType}
                            onChange={(e) => {
                                setFieldTouched('searchType', true);
                                validateAndUpdateSearchParams({ searchType: e.target.value as 'semantic' | 'hybrid' | 'question' | 'code' });
                            }}
                            className="w-full px-4 py-2.5 input-field rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-[var(--accent-primary)]/20 focus:border-[var(--accent-primary)] transition-all duration-200"
                            disabled={loading}
                        >
                            <option value="semantic">General</option>
                            <option value="question">Question Answering</option>
                            <option value="code">Code Search</option>
                        </select>
                        <p className="text-xs text-[var(--text-muted)] ml-1">
                            Controls embedding instruction/prefix
                        </p>
                    </div>
                </div>
            </div>

            <div className="border-t border-[var(--border)] pt-6">
                {/* Search Mode Selector (Dense/Sparse/Hybrid) */}
                <SearchModeSelector
                    searchMode={searchParams.searchMode}
                    rrfK={searchParams.rrfK}
                    onSearchModeChange={(mode: SearchMode) => validateAndUpdateSearchParams({ searchMode: mode })}
                    onRrfKChange={(k: number) => validateAndUpdateSearchParams({ rrfK: k })}
                    disabled={loading}
                    sparseAvailable={true}
                />

                {/* HyDE Query Expansion Toggle */}
                <div className="mt-4 flex items-center space-x-2">
                    <input
                        type="checkbox"
                        id="use-hyde"
                        checked={searchParams.useHyde}
                        onChange={(e) => validateAndUpdateSearchParams({ useHyde: e.target.checked })}
                        disabled={loading}
                        className="h-4 w-4 bg-[var(--bg-secondary)] border-[var(--border)] text-[var(--accent-primary)] rounded focus:ring-[var(--accent-primary)]"
                    />
                    <label htmlFor="use-hyde" className="text-sm text-[var(--text-primary)] font-medium">
                        Use HyDE query expansion
                    </label>
                    <span className="text-xs text-[var(--text-muted)]" title="Generates a hypothetical document to improve search quality">
                        (?)
                    </span>
                </div>
            </div>

            {/* Advanced Options */}
            <div className="card bg-[var(--bg-secondary)] rounded-xl p-4 border border-[var(--border)]">
                <SearchOptions />
            </div>

            {/* Search Button */}
            <div className="flex justify-end pt-2">
                <button
                    type="submit"
                    disabled={loading}
                    className={`btn-primary px-8 py-3.5 rounded-xl font-bold shadow-lg transition-all duration-200
            flex items-center space-x-2.5 transform hover:-translate-y-0.5
            ${loading ? 'cursor-not-allowed shadow-none translate-y-0 opacity-50' : 'hover:shadow-lg active:translate-y-0'}`}
                >
                    {loading ? (
                        <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            <span>Searching...</span>
                        </>
                    ) : (
                        <>
                            <Search className="w-5 h-5" />
                            <span>Search</span>
                        </>
                    )}
                </button>
            </div>
        </form>
    );
}
