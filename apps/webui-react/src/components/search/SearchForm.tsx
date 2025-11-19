import React, { useEffect } from 'react';
import { Search, X, Loader2 } from 'lucide-react';
import { useSearchStore } from '../../stores/searchStore';
import { CollectionMultiSelect } from '../CollectionMultiSelect';
import SearchOptions from './SearchOptions';
import { searchV2Api } from '../../services/api/v2/collections';
import { useUIStore } from '../../stores/uiStore';
import { sanitizeQuery } from '../../utils/searchValidation';

interface SearchFormProps {
    collections: any[]; // Using any[] for now to avoid circular deps, or import Collection type
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
            const response = await searchV2Api.search({
                query: sanitizeQuery(searchParams.query),
                collection_uuids: searchParams.selectedCollections,
                k: searchParams.topK,
                score_threshold: searchParams.scoreThreshold,
                search_type: searchParams.searchType,
                use_reranker: searchParams.useReranker,
                rerank_model: searchParams.useReranker ? searchParams.rerankModel : null,
                hybrid_alpha: searchParams.searchType === 'hybrid' ? searchParams.hybridAlpha : undefined,
                hybrid_mode: searchParams.searchType === 'hybrid' ? searchParams.hybridMode : undefined,
                keyword_mode: searchParams.searchType === 'hybrid' ? searchParams.keywordMode : undefined,
            }); // Note: We can't pass signal to axios here easily without modifying the API client generator or manually handling it, 
            // but for now we manage the state. Ideally, the API client should accept a signal.

            if (response.data) {
                // Map results preserving all fields
                const mappedResults = response.data.results.map((result: any) => ({
                    doc_id: result.document_id,
                    chunk_id: result.chunk_id,
                    score: result.score,
                    content: result.text,
                    file_path: result.file_path,
                    file_name: result.file_name,
                    chunk_index: result.chunk_index || 0,
                    total_chunks: result.metadata?.total_chunks || 1,
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
            }
        } catch (error: any) {
            if (error.name === 'CanceledError' || error.message === 'canceled') {
                return; // Ignore cancellations
            }

            console.error('Search error:', error);

            // Handle specific error types
            if (error.response?.status === 500 && error.response?.data?.detail?.includes('GPU')) {
                setGpuMemoryError({
                    message: 'GPU memory limit exceeded',
                    suggestion: 'Try reducing the batch size or using a smaller model.',
                    currentModel: searchParams.rerankModel || 'unknown'
                });
                setError('GPU_MEMORY_ERROR');
            } else if (error.code === 'ECONNABORTED') {
                setError('Search timed out. Please try again with fewer collections or a simpler query.');
            } else {
                setError(error.response?.data?.detail || 'An unexpected error occurred during search.');
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

                    {/* Search Type Selector */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            Search Type
                        </label>
                        <select
                            value={searchParams.searchType}
                            onChange={(e) => {
                                setFieldTouched('searchType', true);
                                validateAndUpdateSearchParams({ searchType: e.target.value as any });
                            }}
                            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                            disabled={loading}
                        >
                            <option value="semantic">Semantic Search</option>
                            <option value="hybrid">Hybrid Search</option>
                            {/* Backend supports these, exposing them now */}
                            <option value="question">Question Answering</option>
                            <option value="code">Code Search</option>
                        </select>
                        <p className="mt-1 text-xs text-gray-500">
                            {searchParams.searchType === 'semantic' && 'Finds meaning-based matches'}
                            {searchParams.searchType === 'hybrid' && 'Combines keyword and semantic search'}
                            {searchParams.searchType === 'question' && 'Optimized for natural language questions'}
                            {searchParams.searchType === 'code' && 'Optimized for code snippets'}
                        </p>
                    </div>
                </div>
            </div>

            {/* Hybrid Search Configuration */}
            {searchParams.searchType === 'hybrid' && (
                <div className="p-4 bg-blue-50 rounded-lg border border-blue-100 space-y-4">
                    <h4 className="font-medium text-blue-900">Hybrid Search Configuration</h4>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <label className="block text-sm font-medium text-blue-800 mb-1">
                                Alpha (Semantic Weight): {searchParams.hybridAlpha}
                            </label>
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.1"
                                value={searchParams.hybridAlpha}
                                onChange={(e) => {
                                    setFieldTouched('hybridAlpha', true);
                                    validateAndUpdateSearchParams({ hybridAlpha: parseFloat(e.target.value) });
                                }}
                                className="w-full"
                            />
                            <div className="flex justify-between text-xs text-blue-600">
                                <span>Keyword (0.0)</span>
                                <span>Semantic (1.0)</span>
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-blue-800 mb-1">
                                Fusion Mode
                            </label>
                            <select
                                value={searchParams.hybridMode}
                                onChange={(e) => validateAndUpdateSearchParams({ hybridMode: e.target.value as any })}
                                className="w-full px-3 py-2 border border-blue-200 rounded-md text-sm"
                            >
                                <option value="weighted">Weighted Sum</option>
                                <option value="reciprocal_rank">Reciprocal Rank Fusion (RRF)</option>
                                <option value="relative_score">Relative Score Fusion</option>
                            </select>
                        </div>
                    </div>
                </div>
            )}

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
