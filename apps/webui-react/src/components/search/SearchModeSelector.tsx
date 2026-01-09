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
                RRF Constant (k)
              </label>
              <span className="text-sm font-mono text-purple-700 bg-purple-100 px-2 py-0.5 rounded">
                {rrfK}
              </span>
            </div>
            <input
              type="range"
              min={RRF_DEFAULTS.min}
              max={200}
              step={5}
              value={rrfK}
              onChange={(e) => onRrfKChange(parseInt(e.target.value, 10))}
              disabled={disabled}
              className="w-full h-2 bg-purple-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
            />
            <div className="flex justify-between text-xs text-purple-600 mt-1">
              <span>Top-heavy ({RRF_DEFAULTS.min})</span>
              <span>Balanced ({RRF_DEFAULTS.k})</span>
              <span>Uniform (200)</span>
            </div>
          </div>

          <p className="mt-3 text-xs text-purple-600">
            Default value: {RRF_DEFAULTS.k}. Lower values weight top results more heavily.
          </p>
        </div>
      )}
    </div>
  );
}

export default SearchModeSelector;
