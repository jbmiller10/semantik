/**
 * SearchModeSelector Component
 *
 * Allows users to select the search mode:
 * - Dense: Traditional semantic vector search
 * - Sparse: BM25/SPLADE keyword search
 * - Hybrid: Combined search with RRF fusion
 *
 * When hybrid mode is selected, shows the RRF k parameter configuration.
 */

import React from 'react';
import { Zap, Search, Combine, AlertCircle, HelpCircle } from 'lucide-react';
import type { SearchMode } from '../../types/sparse-index';
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

interface SearchModeSelectorProps {
  /** Currently selected search mode */
  searchMode: SearchMode;
  /** RRF constant k for hybrid search */
  rrfK: number;
  /** Callback when search mode changes */
  onSearchModeChange: (mode: SearchMode) => void;
  /** Callback when RRF k changes */
  onRrfKChange: (k: number) => void;
  /** Whether the selector is disabled */
  disabled?: boolean;
  /** Whether any selected collections support sparse indexing */
  sparseAvailable?: boolean;
}

interface ModeOption {
  value: SearchMode;
  label: string;
  icon: React.ReactNode;
  description: string;
  requiresSparse: boolean;
}

const MODE_OPTIONS: ModeOption[] = [
  {
    value: 'dense',
    label: 'Dense',
    icon: <Search className="h-4 w-4" />,
    description: 'Semantic vector search using embeddings',
    requiresSparse: false,
  },
  {
    value: 'sparse',
    label: 'Sparse',
    icon: <Zap className="h-4 w-4" />,
    description: 'BM25/SPLADE keyword matching',
    requiresSparse: true,
  },
  {
    value: 'hybrid',
    label: 'Hybrid',
    icon: <Combine className="h-4 w-4" />,
    description: 'Combined dense + sparse with RRF fusion',
    requiresSparse: true,
  },
];

export function SearchModeSelector({
  searchMode,
  rrfK,
  onSearchModeChange,
  onRrfKChange,
  disabled = false,
  sparseAvailable = true,
}: SearchModeSelectorProps) {
  const [showAdvanced, setShowAdvanced] = React.useState(false);
  const [rrfKInput, setRrfKInput] = React.useState(() => String(rrfK));

  React.useEffect(() => {
    setRrfKInput(String(rrfK));
  }, [rrfK]);

  return (
    <div className="space-y-4">
      {/* Search Mode Selector */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Search Mode
        </label>
        <div className="grid grid-cols-3 gap-2">
          {MODE_OPTIONS.map((option) => {
            const isDisabled =
              disabled || (option.requiresSparse && !sparseAvailable);
            const isSelected = searchMode === option.value;

            return (
              <button
                key={option.value}
                type="button"
                onClick={() => !isDisabled && onSearchModeChange(option.value)}
                disabled={isDisabled}
                className={`
                  relative flex flex-col items-center justify-center p-3 rounded-lg border-2 transition-all
                  ${
                    isSelected
                      ? 'border-blue-500 bg-blue-50 text-blue-700'
                      : 'border-gray-200 bg-white hover:border-gray-300 text-gray-700'
                  }
                  ${isDisabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                `}
                title={option.description}
              >
                <span
                  className={`mb-1 ${isSelected ? 'text-blue-600' : 'text-gray-500'}`}
                >
                  {option.icon}
                </span>
                <span className="text-sm font-medium">{option.label}</span>
                {option.requiresSparse && !sparseAvailable && (
                  <span className="absolute -top-1 -right-1">
                    <AlertCircle className="h-4 w-4 text-yellow-500" />
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Mode description */}
        <p className="mt-2 text-xs text-gray-500">
          {MODE_OPTIONS.find((o) => o.value === searchMode)?.description}
        </p>

        {/* Sparse not available warning */}
        {(searchMode === 'sparse' || searchMode === 'hybrid') &&
          !sparseAvailable && (
            <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded-md">
              <p className="text-xs text-yellow-800 flex items-start gap-1.5">
                <AlertCircle className="h-3.5 w-3.5 mt-0.5 flex-shrink-0" />
                <span>
                  Sparse search requires collections with sparse indexing enabled.
                  Results may fall back to dense search.
                </span>
              </p>
            </div>
          )}
      </div>

      {/* Hybrid RRF Configuration */}
      {searchMode === 'hybrid' && (
        <div className="p-4 bg-purple-50 rounded-lg border border-purple-100">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium text-purple-900 flex items-center gap-1.5">
              <Combine className="h-4 w-4" />
              Hybrid Search Configuration
            </h4>
            <div className="group relative">
              <HelpCircle className="h-4 w-4 text-purple-400 cursor-help" />
              <div className="absolute right-0 w-64 p-3 bg-gray-900 text-white text-xs rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
                <p className="font-medium mb-1">RRF (Reciprocal Rank Fusion)</p>
                <p className="text-gray-300">
                  Combines dense and sparse search results. Lower k values give
                  more weight to top-ranked results. Higher values produce more
                  uniform weighting.
                </p>
              </div>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-sm text-purple-800">
                RRF Weighting
              </label>
              <span className="text-sm font-mono text-purple-700 bg-purple-100 px-2 py-0.5 rounded">
                k={rrfK}
              </span>
            </div>
            <input
              type="range"
              min={RRF_SLIDER.min}
              max={RRF_SLIDER.max}
              step={RRF_SLIDER.step}
              value={sliderFromRrfK(rrfK)}
              onChange={(e) => onRrfKChange(rrfKFromSlider(parseInt(e.target.value, 10)))}
              disabled={disabled}
              className="w-full h-2 bg-purple-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
            />
            <div className="flex justify-between text-xs text-purple-600 mt-1">
              <span>Top-heavy (k={RRF_DEFAULTS.min})</span>
              <span>Balanced (k={RRF_DEFAULTS.k})</span>
              <span>Uniform (k={RRF_DEFAULTS.max})</span>
            </div>
          </div>

          <p className="mt-3 text-xs text-purple-600">
            Default value: {RRF_DEFAULTS.k}. Slider is logarithmic; use Advanced for an exact k.
          </p>

          <div className="mt-3">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-xs text-purple-700 hover:text-purple-900"
              disabled={disabled}
            >
              <span
                className={`transform transition-transform ${
                  showAdvanced ? 'rotate-90' : ''
                }`}
              >
                ▶
              </span>
              Advanced k
              <span className="text-xs text-purple-500">(optional)</span>
            </button>

            {showAdvanced && (
              <div className="mt-3 p-3 bg-white/60 rounded-lg border border-purple-100">
                <label className="block text-xs font-medium text-purple-800 mb-1" htmlFor="rrf-k-input">
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
                    if (Number.isFinite(parsed)) onRrfKChange(parsed);
                  }}
                  onBlur={() => {
                    const parsed = Number.parseInt(rrfKInput, 10);
                    if (!Number.isFinite(parsed)) setRrfKInput(String(rrfK));
                  }}
                  disabled={disabled}
                  className="w-full px-3 py-2 border border-purple-200 rounded-md focus:ring-purple-500 focus:border-purple-500 bg-white"
                />
                <p className="mt-1 text-xs text-purple-600">
                  Valid range: {RRF_DEFAULTS.min}–{RRF_DEFAULTS.max}.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default SearchModeSelector;
