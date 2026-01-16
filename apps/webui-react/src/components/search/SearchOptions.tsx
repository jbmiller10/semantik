import React from 'react';
import { useSearchStore } from '../../stores/searchStore';
import { ChevronDown, ChevronUp, Settings } from 'lucide-react';
import { RerankingConfiguration } from '../RerankingConfiguration';
import { useUpdatePreferences } from '../../hooks/usePreferences';
import type { SearchMode } from '../../types/preferences';

export default function SearchOptions() {
    const [isExpanded, setIsExpanded] = React.useState(false);
    const {
        searchParams,
        validateAndUpdateSearchParams,
        getValidationError,
        setFieldTouched
    } = useSearchStore();
    const { mutate: updatePrefs, isPending: savingPrefs } = useUpdatePreferences();

    const handleParamChange = (key: string, value: string | number | boolean) => {
        setFieldTouched(key, true);
        validateAndUpdateSearchParams({ [key]: value });
    };

    const handleSaveAsDefaults = () => {
        updatePrefs({
            search: {
                top_k: searchParams.topK,
                mode: searchParams.searchMode as SearchMode,
                use_reranker: searchParams.useReranker,
                rrf_k: searchParams.rrfK,
                similarity_threshold: searchParams.scoreThreshold || null,
            },
        });
    };

    return (
        <div className="border border-[var(--border)] rounded-2xl bg-[var(--bg-secondary)] shadow-sm overflow-hidden">
            <button
                type="button"
                onClick={() => setIsExpanded(!isExpanded)}
                className="w-full flex items-center justify-between p-4 bg-transparent hover:bg-[var(--bg-tertiary)] transition-colors"
            >
                <div className="flex items-center space-x-2 text-[var(--text-primary)]">
                    <Settings className="w-4 h-4" />
                    <span className="font-bold uppercase tracking-wide text-xs">Advanced Options</span>
                </div>
                {isExpanded ? (
                    <ChevronUp className="w-4 h-4 text-[var(--text-muted)]" />
                ) : (
                    <ChevronDown className="w-4 h-4 text-[var(--text-muted)]" />
                )}
            </button>

            {isExpanded && (
                <div className="p-6 space-y-6 border-t border-[var(--border)] bg-[var(--bg-tertiary)]">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Top K Results */}
                        <div>
                            <label className="block text-xs font-bold text-[var(--text-secondary)] uppercase tracking-wider mb-2">
                                Top K Results
                            </label>
                            <input
                                type="number"
                                value={searchParams.topK}
                                onChange={(e) => handleParamChange('topK', parseInt(e.target.value))}
                                min={1}
                                max={250}
                                className={`w-full px-3 py-2 border rounded-xl input-field focus:ring-[var(--accent-primary)] focus:border-[var(--accent-primary)] ${getValidationError('topK') ? 'border-red-500/50 bg-red-500/10' : 'border-[var(--border)]'
                                    }`}
                            />
                            {getValidationError('topK') && (
                                <p className="mt-1 text-xs text-red-600 dark:text-red-400">{getValidationError('topK')}</p>
                            )}
                            <p className="mt-1 text-xs text-[var(--text-muted)]">
                                Number of results to return (1-250)
                            </p>
                        </div>

                        {/* Score Threshold */}
                        <div>
                            <label className="block text-xs font-bold text-[var(--text-secondary)] uppercase tracking-wider mb-2">
                                Score Threshold
                            </label>
                            <input
                                type="number"
                                value={searchParams.scoreThreshold}
                                onChange={(e) => handleParamChange('scoreThreshold', parseFloat(e.target.value))}
                                min={0}
                                max={1}
                                step={0.05}
                                className={`w-full px-3 py-2 border rounded-xl input-field focus:ring-[var(--accent-primary)] focus:border-[var(--accent-primary)] ${getValidationError('scoreThreshold') ? 'border-red-500/50 bg-red-500/10' : 'border-[var(--border)]'
                                    }`}
                            />
                            {getValidationError('scoreThreshold') && (
                                <p className="mt-1 text-xs text-red-600 dark:text-red-400">{getValidationError('scoreThreshold')}</p>
                            )}
                            <p className="mt-1 text-xs text-[var(--text-muted)]">
                                Minimum similarity score (0.0-1.0)
                            </p>
                        </div>
                    </div>

                    {/* Reranking Configuration */}
                    <div className="pt-4 border-t border-[var(--border)]">
                        <RerankingConfiguration
                            enabled={searchParams.useReranker}
                            model={searchParams.rerankModel}
                            quantization={searchParams.rerankQuantization}
                            onChange={validateAndUpdateSearchParams}
                            hideToggle={true}
                        />
                    </div>

                    {/* Save as Defaults */}
                    <div className="pt-4 border-t border-[var(--border)] flex justify-end">
                        <button
                            type="button"
                            onClick={handleSaveAsDefaults}
                            disabled={savingPrefs}
                            className="text-xs font-bold uppercase tracking-wider text-[var(--accent-primary)] hover:text-[var(--accent-primary-hover)] disabled:opacity-50 transition-colors"
                        >
                            {savingPrefs ? 'Saving...' : 'Save as defaults'}
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
