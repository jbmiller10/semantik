import React from 'react';
import { useSearchStore } from '../../stores/searchStore';
import { ChevronDown, ChevronUp, Settings, Combine, HelpCircle } from 'lucide-react';
import { RerankingConfiguration } from '../RerankingConfiguration';
import { useUpdatePreferences } from '../../hooks/usePreferences';
import type { SearchMode } from '../../types/preferences';
import { RRF_DEFAULTS } from '../../types/sparse-index';

const RRF_SLIDER = {
    min: 0,
    max: 100,
    step: 1,
    // Log-scale "feel": lower values provide finer control near default k.
    // Calibrated so ~50% maps near k≈60 when max=1000.
    gamma: 0.75,
} as const;

function clampNumber(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value));
}

function rrfKFromSlider(sliderValue: number): number {
    const minK = RRF_DEFAULTS.min;
    const maxK = RRF_DEFAULTS.max;

    if (minK <= 0 || maxK <= minK) return RRF_DEFAULTS.k;

    const slider = clampNumber(sliderValue, RRF_SLIDER.min, RRF_SLIDER.max);
    const t = slider / RRF_SLIDER.max;

    const logMin = Math.log(minK);
    const logMax = Math.log(maxK);
    const scaled = logMin + Math.pow(t, RRF_SLIDER.gamma) * (logMax - logMin);

    return Math.round(Math.exp(scaled));
}

function sliderFromRrfK(k: number): number {
    const minK = RRF_DEFAULTS.min;
    const maxK = RRF_DEFAULTS.max;

    if (minK <= 0 || maxK <= minK) return Math.round(RRF_SLIDER.max / 2);
    if (!Number.isFinite(k)) return Math.round(RRF_SLIDER.max / 2);

    const clampedK = clampNumber(k, minK, maxK);
    const logMin = Math.log(minK);
    const logMax = Math.log(maxK);

    const tPowGamma = clampNumber((Math.log(clampedK) - logMin) / (logMax - logMin), 0, 1);
    const t = Math.pow(tPowGamma, 1 / RRF_SLIDER.gamma);

    return Math.round(t * RRF_SLIDER.max);
}

export default function SearchOptions() {
    const [isExpanded, setIsExpanded] = React.useState(false);
    const [showAdvancedK, setShowAdvancedK] = React.useState(false);
    const {
        searchParams,
        validateAndUpdateSearchParams,
        getValidationError,
        setFieldTouched
    } = useSearchStore();
    const { mutate: updatePrefs, isPending: savingPrefs } = useUpdatePreferences();
    const [rrfKInput, setRrfKInput] = React.useState(() => String(searchParams.rrfK));

    const isHybridMode = searchParams.searchMode === 'hybrid';

    React.useEffect(() => {
        setRrfKInput(String(searchParams.rrfK));
    }, [searchParams.rrfK]);

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

                    {/* RRF Configuration */}
                    <div className={`pt-4 border-t border-[var(--border)] ${!isHybridMode ? 'opacity-50' : ''}`}>
                        <div className="flex items-center justify-between mb-3">
                            <h4 className="text-sm font-medium text-[var(--text-primary)] flex items-center gap-1.5">
                                <Combine className="h-4 w-4 text-[var(--text-muted)]" />
                                RRF Weighting
                                {!isHybridMode && (
                                    <span className="text-xs text-[var(--text-muted)] font-normal">(requires Hybrid mode)</span>
                                )}
                            </h4>
                            <div className="group relative">
                                <HelpCircle className="h-4 w-4 text-[var(--text-muted)] cursor-help" />
                                <div className="absolute right-0 w-72 p-3 bg-[var(--bg-primary)] text-[var(--text-primary)] text-xs rounded-lg shadow-lg border border-[var(--border)] opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
                                    <p className="font-medium mb-2">RRF (Reciprocal Rank Fusion)</p>
                                    <p className="text-[var(--text-secondary)] mb-2">
                                        Combines dense and sparse search results.
                                    </p>
                                    <ul className="text-[var(--text-secondary)] space-y-1">
                                        <li><span className="font-medium">Top-heavy (low k):</span> Use when top results from each method are clearly the best matches</li>
                                        <li><span className="font-medium">Uniform (high k):</span> Use when good results might be ranked lower in one method</li>
                                        <li><span className="font-medium">Balanced (default):</span> Good starting point for most use cases</li>
                                    </ul>
                                </div>
                            </div>
                        </div>

                        <div className={!isHybridMode ? 'pointer-events-none' : ''}>
                            <div className="flex items-center justify-between mb-1">
                                <label className="text-sm text-[var(--text-secondary)]">
                                    Rank Fusion Constant
                                </label>
                                <span className="text-sm font-mono text-[var(--accent-primary)] bg-[var(--accent-primary)]/10 px-2 py-0.5 rounded">
                                    k={searchParams.rrfK}
                                </span>
                            </div>
                            <input
                                type="range"
                                min={RRF_SLIDER.min}
                                max={RRF_SLIDER.max}
                                step={RRF_SLIDER.step}
                                value={sliderFromRrfK(searchParams.rrfK)}
                                onChange={(e) => validateAndUpdateSearchParams({ rrfK: rrfKFromSlider(parseInt(e.target.value, 10)) })}
                                disabled={!isHybridMode}
                                className="w-full h-2 bg-[var(--bg-secondary)] rounded-lg appearance-none cursor-pointer accent-[var(--accent-primary)] disabled:cursor-not-allowed"
                            />
                            <div className="flex justify-between text-xs text-[var(--text-muted)] mt-1">
                                <span>Top-heavy (k={RRF_DEFAULTS.min})</span>
                                <span>Balanced (k={RRF_DEFAULTS.k})</span>
                                <span>Uniform (k={RRF_DEFAULTS.max})</span>
                            </div>

                            <div className="mt-3">
                                <button
                                    type="button"
                                    onClick={() => setShowAdvancedK(!showAdvancedK)}
                                    className="flex items-center gap-2 text-xs text-[var(--accent-primary)] hover:text-[var(--accent-primary-hover)] disabled:opacity-50"
                                    disabled={!isHybridMode}
                                >
                                    <span className={`transform transition-transform ${showAdvancedK ? 'rotate-90' : ''}`}>
                                        ▶
                                    </span>
                                    Advanced
                                </button>

                                {showAdvancedK && (
                                    <div className="mt-3 p-3 bg-[var(--bg-secondary)] rounded-lg border border-[var(--border)]">
                                        <label className="block text-xs font-medium text-[var(--text-secondary)] mb-1" htmlFor="rrf-k-input">
                                            Exact RRF constant (k)
                                        </label>
                                        <input
                                            id="rrf-k-input"
                                            type="number"
                                            inputMode="numeric"
                                            min={RRF_DEFAULTS.min}
                                            max={RRF_DEFAULTS.max}
                                            step={1}
                                            value={rrfKInput}
                                            onChange={(e) => {
                                                const next = e.target.value;
                                                setRrfKInput(next);
                                                const parsed = Number.parseInt(next, 10);
                                                if (Number.isFinite(parsed)) validateAndUpdateSearchParams({ rrfK: parsed });
                                            }}
                                            onBlur={() => {
                                                const parsed = Number.parseInt(rrfKInput, 10);
                                                if (!Number.isFinite(parsed)) setRrfKInput(String(searchParams.rrfK));
                                            }}
                                            disabled={!isHybridMode}
                                            className="w-full px-3 py-2 border border-[var(--border)] rounded-md focus:ring-[var(--accent-primary)] focus:border-[var(--accent-primary)] bg-[var(--bg-primary)] text-[var(--text-primary)] disabled:opacity-50"
                                        />
                                        <p className="mt-1 text-xs text-[var(--text-muted)]">
                                            Valid range: {RRF_DEFAULTS.min}–{RRF_DEFAULTS.max}. Default: {RRF_DEFAULTS.k}
                                        </p>
                                    </div>
                                )}
                            </div>
                        </div>
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
