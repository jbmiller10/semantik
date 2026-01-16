/**
 * SearchModeSelector Component
 *
 * Allows users to select the search mode:
 * - Dense: Traditional semantic vector search
 * - Sparse: BM25/SPLADE keyword search
 * - Hybrid: Combined search with RRF fusion
 */

import React from 'react';
import { Zap, Search, Combine, AlertCircle } from 'lucide-react';
import type { SearchMode } from '../../types/sparse-index';

interface SearchModeSelectorProps {
  /** Currently selected search mode */
  searchMode: SearchMode;
  /** Callback when search mode changes */
  onSearchModeChange: (mode: SearchMode) => void;
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
  onSearchModeChange,
  disabled = false,
  sparseAvailable = true,
}: SearchModeSelectorProps) {
  return (
    <div className="space-y-4">
      {/* Search Mode Selector */}
      <div>
        <label className="block text-xs font-bold text-[var(--text-secondary)] uppercase tracking-wider mb-2">
          Search Mode
        </label>
        <div className="grid grid-cols-3 gap-3">
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
                  relative flex flex-col items-center justify-center p-3 rounded-xl border transition-all duration-200
                  ${isSelected
                    ? 'border-[var(--accent-primary)] bg-[var(--accent-primary)]/10 text-[var(--accent-primary)] shadow-lg shadow-[var(--accent-primary)]/10'
                    : 'border-[var(--border)] bg-[var(--bg-tertiary)] hover:bg-[var(--bg-secondary)] hover:border-[var(--border-strong)] text-[var(--text-secondary)]'
                  }
                  ${isDisabled ? 'opacity-50 cursor-not-allowed grayscale' : 'cursor-pointer'}
                `}
                title={option.description}
              >
                <span
                  className={`mb-2 ${isSelected ? 'text-[var(--accent-primary)]' : 'text-[var(--text-muted)]'}`}
                >
                  {option.icon}
                </span>
                <span className="text-xs font-bold uppercase tracking-wide">{option.label}</span>
                {option.requiresSparse && !sparseAvailable && (
                  <span className="absolute -top-1.5 -right-1.5 bg-[var(--bg-primary)] rounded-full p-0.5 border border-[var(--border)]">
                    <AlertCircle className="h-4 w-4 text-yellow-500" />
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Mode description */}
        <p className="mt-2 text-xs text-[var(--text-muted)]">
          {MODE_OPTIONS.find((o) => o.value === searchMode)?.description}
        </p>

        {/* Sparse not available warning */}
        {(searchMode === 'sparse' || searchMode === 'hybrid') &&
          !sparseAvailable && (
            <div className="mt-2 p-2 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 rounded-md">
              <p className="text-xs text-yellow-800 dark:text-yellow-400 flex items-start gap-1.5">
                <AlertCircle className="h-3.5 w-3.5 mt-0.5 flex-shrink-0" />
                <span>
                  Sparse search requires collections with sparse indexing enabled.
                  Results may fall back to dense search.
                </span>
              </p>
            </div>
          )}
      </div>
    </div>
  );
}

export default SearchModeSelector;
