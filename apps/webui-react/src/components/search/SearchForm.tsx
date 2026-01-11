import React, { useEffect } from 'react';
import { Search, X, Loader2 } from 'lucide-react';
import { useSearchStore } from '../../stores/searchStore';
import { CollectionMultiSelect } from '../CollectionMultiSelect';
import SearchOptions from './SearchOptions';
import { SearchModeSelector } from './SearchModeSelector';
import { searchV2Api } from '../../services/api/v2/collections';
import { useUIStore } from '../../stores/uiStore';
import { sanitizeQuery } from '../../utils/searchValidation';

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
        setRerankingMetrics
    } = useSearchStore();

    const addToast = useUIStore((state) => state.addToast);

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
        <form onSubmit={handleSearch} className="space-y-6">
            {/* Search Input Group */}
            <div className="space-y-4">
                <div className="relative">
                    <input
                        type="text"
                        value={searchParams.query}
                        onChange={(e) => {
                            setFieldTouched('query', true);
                            validateAndUpdateSearchParams({ query: e.target.value });
                        }}
                        placeholder="Enter your search query..."
                        className={`w-full pl-4 pr-12 py-3 text-lg border rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent ${getValidationError('query') ? 'border-red-300 bg-red-50' : 'border-gray-300'
                            }`}
                        disabled={loading}
                    />
                    <div className="absolute right-3 top-1/2 -translate-y-1/2">
                        {loading ? (
                            <button
                                type="button"
                                onClick={handleCancel}
                                className="p-1 hover:bg-gray-100 rounded-full text-gray-500"
                                title="Cancel search"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        ) : (
                            <Search className="w-5 h-5 text-gray-400" />
                        )}
                    </div>
                </div>
                {getValidationError('query') && (
                    <p className="text-sm text-red-600">{getValidationError('query')}</p>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Collection Selector */}
                    <div className="lg:col-span-2">
                        <label className="block text-sm font-medium text-gray-700 mb-1">
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
                            <p className="mt-1 text-sm text-red-600">{getValidationError('collections')}</p>
                        )}
                    </div>

                    {/* Embedding Mode Selector (for instruction prefix) */}
                    <div>
                        <label htmlFor="search-type" className="block text-sm font-medium text-gray-700 mb-1">
                            Embedding Mode
                        </label>
                        <select
                            id="search-type"
                            value={searchParams.searchType}
                            onChange={(e) => {
                                setFieldTouched('searchType', true);
                                validateAndUpdateSearchParams({ searchType: e.target.value as 'semantic' | 'hybrid' | 'question' | 'code' });
                            }}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                            disabled={loading}
                        >
                            <option value="semantic">General</option>
                            <option value="question">Question Answering</option>
                            <option value="code">Code Search</option>
                        </select>
                        <p className="mt-1 text-xs text-gray-500">
                            Controls embedding instruction/prefix
                        </p>
                    </div>
                </div>
            </div>

            {/* Search Mode Selector (Dense/Sparse/Hybrid) */}
            <SearchModeSelector
                searchMode={searchParams.searchMode}
                rrfK={searchParams.rrfK}
                onSearchModeChange={(mode: SearchMode) => validateAndUpdateSearchParams({ searchMode: mode })}
                onRrfKChange={(k: number) => validateAndUpdateSearchParams({ rrfK: k })}
                disabled={loading}
                sparseAvailable={true}
            />

            {/* Advanced Options */}
            <SearchOptions />

            {/* Search Button */}
            <div className="flex justify-end pt-4">
                <button
                    type="submit"
                    disabled={loading}
                    className={`
            px-8 py-3 rounded-lg font-medium text-white shadow-sm transition-all
            flex items-center space-x-2
            ${loading
                            ? 'bg-blue-400 cursor-not-allowed'
                            : 'bg-blue-600 hover:bg-blue-700 hover:shadow-md active:transform active:scale-95'
                        }
          `}
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
