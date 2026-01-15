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
        <div className="border border-white/5 rounded-2xl bg-void-900/50 shadow-sm overflow-hidden backdrop-blur-md">
            <button
                type="button"
                onClick={() => setIsExpanded(!isExpanded)}
                className="w-full flex items-center justify-between p-4 bg-transparent hover:bg-white/5 transition-colors"
            >
                <div className="flex items-center space-x-2 text-gray-200">
                    <Settings className="w-4 h-4" />
                    <span className="font-bold uppercase tracking-wide text-xs">Advanced Options</span>
                </div>
                {isExpanded ? (
                    <ChevronUp className="w-4 h-4 text-gray-400" />
                ) : (
                    <ChevronDown className="w-4 h-4 text-gray-400" />
                )}
            </button>

            {isExpanded && (
                <div className="p-6 space-y-6 border-t border-white/5 bg-void-950/30">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Top K Results */}
                        <div>
                            <label className="block text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">
                                Top K Results
                            </label>
                            <input
                                type="number"
                                value={searchParams.topK}
                                onChange={(e) => handleParamChange('topK', parseInt(e.target.value))}
                                min={1}
                                max={250}
                                className={`w-full px-3 py-2 border rounded-xl input-glass focus:ring-signal-500 focus:border-signal-500 ${getValidationError('topK') ? 'border-red-500/50 bg-red-500/10' : 'border-white/10'
                                    }`}
                            />
                            {getValidationError('topK') && (
                                <p className="mt-1 text-xs text-red-600">{getValidationError('topK')}</p>
                            )}
                            <p className="mt-1 text-xs text-gray-500">
                                Number of results to return (1-250)
                            </p>
                        </div>

                        {/* Score Threshold */}
                        <div>
                            <label className="block text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">
                                Score Threshold
                            </label>
                            <input
                                type="number"
                                value={searchParams.scoreThreshold}
                                onChange={(e) => handleParamChange('scoreThreshold', parseFloat(e.target.value))}
                                min={0}
                                max={1}
                                step={0.05}
                                className={`w-full px-3 py-2 border rounded-xl input-glass focus:ring-signal-500 focus:border-signal-500 ${getValidationError('scoreThreshold') ? 'border-red-500/50 bg-red-500/10' : 'border-white/10'
                                    }`}
                            />
                            {getValidationError('scoreThreshold') && (
                                <p className="mt-1 text-xs text-red-600">{getValidationError('scoreThreshold')}</p>
                            )}
                            <p className="mt-1 text-xs text-gray-500">
                                Minimum similarity score (0.0-1.0)
                            </p>
                        </div>
                    </div>

                    {/* Reranking Configuration */}
                    <div className="pt-4 border-t border-white/5">
                        <RerankingConfiguration
                            enabled={searchParams.useReranker}
                            model={searchParams.rerankModel}
                            quantization={searchParams.rerankQuantization}
                            onChange={validateAndUpdateSearchParams}
                        />
                    </div>

                    {/* Save as Defaults */}
                    <div className="pt-4 border-t border-white/5 flex justify-end">
                        <button
                            type="button"
                            onClick={handleSaveAsDefaults}
                            disabled={savingPrefs}
                            className="text-xs font-bold uppercase tracking-wider text-signal-400 hover:text-signal-300 disabled:opacity-50 transition-colors"
                        >
                            {savingPrefs ? 'Saving...' : 'Save as defaults'}
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
