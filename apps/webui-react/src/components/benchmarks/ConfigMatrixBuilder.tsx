/**
 * ConfigMatrixBuilder - Form component for building search configuration matrix
 */

import { useState, useMemo } from 'react';
import { Info, AlertTriangle, Sparkles } from 'lucide-react';
import type { ConfigMatrixItem, SearchMode } from '../../types/benchmark';

interface ConfigMatrixBuilderProps {
  value: ConfigMatrixItem;
  onChange: (config: ConfigMatrixItem) => void;
  hasReranker?: boolean;
  hasSparseIndex?: boolean;
  disabled?: boolean;
}

const TOP_K_OPTIONS = [5, 10, 20, 50, 100];
const RRF_K_OPTIONS = [20, 40, 60, 80, 100];

interface Preset {
  name: string;
  description: string;
  config: ConfigMatrixItem;
}

const PRESETS: Preset[] = [
  {
    name: 'Dense vs Hybrid',
    description: 'Compare dense-only to hybrid search',
    config: {
      search_modes: ['dense', 'hybrid'],
      use_reranker: [false],
      top_k_values: [10],
      rrf_k_values: [60],
      score_thresholds: [null],
    },
  },
  {
    name: 'Reranker A/B',
    description: 'Test with and without reranking',
    config: {
      search_modes: ['dense'],
      use_reranker: [false, true],
      top_k_values: [10, 20],
      rrf_k_values: [60],
      score_thresholds: [null],
    },
  },
  {
    name: 'Top-K Sensitivity',
    description: 'Test different result set sizes',
    config: {
      search_modes: ['dense'],
      use_reranker: [false],
      top_k_values: [5, 10, 20, 50],
      rrf_k_values: [60],
      score_thresholds: [null],
    },
  },
];

export function ConfigMatrixBuilder({
  value,
  onChange,
  hasReranker = true,
  hasSparseIndex = false,
  disabled = false,
}: ConfigMatrixBuilderProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Calculate total configuration count
  const configCount = useMemo(() => {
    const modes = value.search_modes.length || 1;
    const rerankerOptions = value.use_reranker.length || 1;
    const topKCount = value.top_k_values.length || 1;
    const hybridSelected = value.search_modes.includes('hybrid');
    const rrfVariations = hybridSelected ? Math.max(0, (value.rrf_k_values.length || 1) - 1) : 0;

    return modes * rerankerOptions * topKCount + rrfVariations;
  }, [value]);

  // Helper to toggle an item in a set while preventing removal of the last item
  const toggleInSet = <T,>(set: Set<T>, item: T): Set<T> => {
    const updated = new Set(set);
    if (updated.has(item)) {
      if (updated.size > 1) {
        updated.delete(item);
      }
    } else {
      updated.add(item);
    }
    return updated;
  };

  const toggleSearchMode = (mode: SearchMode) => {
    const updated = toggleInSet(new Set(value.search_modes), mode);
    onChange({ ...value, search_modes: Array.from(updated) as SearchMode[] });
  };

  const toggleReranker = (useReranker: boolean) => {
    const updated = toggleInSet(new Set(value.use_reranker), useReranker);
    onChange({ ...value, use_reranker: Array.from(updated) });
  };

  const toggleTopK = (k: number) => {
    const updated = toggleInSet(new Set(value.top_k_values), k);
    onChange({ ...value, top_k_values: Array.from(updated).sort((a, b) => a - b) });
  };

  const toggleRrfK = (k: number) => {
    const updated = toggleInSet(new Set(value.rrf_k_values), k);
    onChange({ ...value, rrf_k_values: Array.from(updated).sort((a, b) => a - b) });
  };

  const applyPreset = (preset: Preset) => {
    // Adjust preset based on available features
    const config = { ...preset.config };
    if (!hasSparseIndex) {
      const filtered = config.search_modes.filter((m) => m === 'dense');
      // Ensure at least 'dense' remains if preset had only sparse/hybrid modes
      config.search_modes = filtered.length > 0 ? filtered : ['dense'];
    }
    if (!hasReranker) {
      config.use_reranker = [false];
    }
    onChange(config);
  };

  const countWarning =
    configCount > 50 ? 'error' : configCount > 25 ? 'warning' : 'ok';

  return (
    <div className="space-y-5">
      {/* Mode Banner */}
      <div className="flex items-start gap-3 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
        <Info className="h-5 w-5 text-blue-400 flex-shrink-0 mt-0.5" />
        <div className="text-sm">
          <p className="font-medium text-blue-300">Search-time Benchmarking (MVP)</p>
          <p className="text-blue-400/80 mt-1">
            Evaluates search configurations at query time. Index-time variations (chunking,
            embedding models) will be available in a future release.
          </p>
        </div>
      </div>

      {/* Quick Presets */}
      <div>
        <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
          Quick Presets
        </label>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((preset) => (
            <button
              key={preset.name}
              type="button"
              onClick={() => applyPreset(preset)}
              disabled={disabled}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm text-[var(--text-secondary)] border border-[var(--border)] rounded-lg hover:bg-[var(--bg-tertiary)] hover:text-[var(--text-primary)] transition-colors disabled:opacity-50"
            >
              <Sparkles className="h-3.5 w-3.5" />
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      {/* Search Modes */}
      <div>
        <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
          Search Modes
        </label>
        <div className="flex flex-wrap gap-2">
          {(['dense', 'sparse', 'hybrid'] as SearchMode[]).map((mode) => {
            const isSelected = value.search_modes.includes(mode);
            const isDisabled =
              disabled ||
              (mode !== 'dense' && !hasSparseIndex);
            const unavailable = mode !== 'dense' && !hasSparseIndex;

            return (
              <button
                key={mode}
                type="button"
                onClick={() => toggleSearchMode(mode)}
                disabled={isDisabled}
                className={`
                  px-4 py-2 text-sm font-medium rounded-xl border transition-all
                  ${isSelected
                    ? 'border-gray-400 dark:border-white bg-gray-100 dark:bg-white/10 text-gray-800 dark:text-white'
                    : 'border-[var(--border)] bg-[var(--bg-secondary)] text-[var(--text-muted)] hover:bg-[var(--bg-tertiary)]'
                  }
                  ${isDisabled ? 'opacity-50 cursor-not-allowed' : ''}
                `}
              >
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
                {unavailable && (
                  <span className="ml-1 text-xs text-amber-400">(no sparse index)</span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Reranker Toggle */}
      <div>
        <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
          Reranking
        </label>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => toggleReranker(false)}
            disabled={disabled}
            className={`
              px-4 py-2 text-sm font-medium rounded-xl border transition-all
              ${value.use_reranker.includes(false)
                ? 'border-gray-400 dark:border-white bg-gray-100 dark:bg-white/10 text-gray-800 dark:text-white'
                : 'border-[var(--border)] bg-[var(--bg-secondary)] text-[var(--text-muted)] hover:bg-[var(--bg-tertiary)]'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
            `}
          >
            Without Reranker
          </button>
          <button
            type="button"
            onClick={() => toggleReranker(true)}
            disabled={disabled || !hasReranker}
            className={`
              px-4 py-2 text-sm font-medium rounded-xl border transition-all
              ${value.use_reranker.includes(true)
                ? 'border-gray-400 dark:border-white bg-gray-100 dark:bg-white/10 text-gray-800 dark:text-white'
                : 'border-[var(--border)] bg-[var(--bg-secondary)] text-[var(--text-muted)] hover:bg-[var(--bg-tertiary)]'
              }
              ${disabled || !hasReranker ? 'opacity-50 cursor-not-allowed' : ''}
            `}
          >
            With Reranker
            {!hasReranker && (
              <span className="ml-1 text-xs text-amber-400">(unavailable)</span>
            )}
          </button>
        </div>
      </div>

      {/* Top-K Values */}
      <div>
        <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
          Top-K Values
        </label>
        <div className="flex flex-wrap gap-2">
          {TOP_K_OPTIONS.map((k) => {
            const isSelected = value.top_k_values.includes(k);
            return (
              <button
                key={k}
                type="button"
                onClick={() => toggleTopK(k)}
                disabled={disabled}
                className={`
                  px-4 py-2 text-sm font-medium rounded-xl border transition-all
                  ${isSelected
                    ? 'border-gray-400 dark:border-white bg-gray-100 dark:bg-white/10 text-gray-800 dark:text-white'
                    : 'border-[var(--border)] bg-[var(--bg-secondary)] text-[var(--text-muted)] hover:bg-[var(--bg-tertiary)]'
                  }
                  ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
                `}
              >
                {k}
              </button>
            );
          })}
        </div>
      </div>

      {/* RRF-K Values (only when hybrid is selected) */}
      {value.search_modes.includes('hybrid') && (
        <div>
          <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
            RRF-K Values (Hybrid Fusion)
          </label>
          <div className="flex flex-wrap gap-2">
            {RRF_K_OPTIONS.map((k) => {
              const isSelected = value.rrf_k_values.includes(k);
              return (
                <button
                  key={k}
                  type="button"
                  onClick={() => toggleRrfK(k)}
                  disabled={disabled}
                  className={`
                    px-4 py-2 text-sm font-medium rounded-xl border transition-all
                    ${isSelected
                      ? 'border-gray-400 dark:border-white bg-gray-100 dark:bg-white/10 text-gray-800 dark:text-white'
                      : 'border-[var(--border)] bg-[var(--bg-secondary)] text-[var(--text-muted)] hover:bg-[var(--bg-tertiary)]'
                    }
                    ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
                  `}
                >
                  {k}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Advanced Settings */}
      <div>
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
        >
          {showAdvanced ? '- Hide' : '+ Show'} Advanced Settings
        </button>
        {showAdvanced && (
          <div className="mt-3 p-4 bg-[var(--bg-tertiary)] rounded-xl space-y-4">
            <div>
              <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                Score Threshold (Optional)
              </label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={value.score_thresholds[0] ?? ''}
                onChange={(e) =>
                  onChange({
                    ...value,
                    score_thresholds: e.target.value
                      ? [parseFloat(e.target.value)]
                      : [null],
                  })
                }
                disabled={disabled}
                placeholder="No threshold"
                className="w-32 px-4 py-2 input-field rounded-xl"
              />
              <p className="mt-1 text-xs text-[var(--text-muted)]">
                Filter results below this similarity score (0-1)
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Configuration Count */}
      <div
        className={`
          flex items-start gap-3 p-3 rounded-lg border
          ${countWarning === 'error'
            ? 'bg-red-500/10 border-red-500/30'
            : countWarning === 'warning'
            ? 'bg-amber-500/10 border-amber-500/30'
            : 'bg-[var(--bg-tertiary)] border-[var(--border)]'
          }
        `}
      >
        {countWarning !== 'ok' && (
          <AlertTriangle
            className={`h-5 w-5 flex-shrink-0 mt-0.5 ${
              countWarning === 'error' ? 'text-red-400' : 'text-amber-400'
            }`}
          />
        )}
        <div className="text-sm">
          <p
            className={`font-medium ${
              countWarning === 'error'
                ? 'text-red-300'
                : countWarning === 'warning'
                ? 'text-amber-300'
                : 'text-[var(--text-primary)]'
            }`}
          >
            {configCount} configuration{configCount !== 1 ? 's' : ''} selected
          </p>
          <p
            className={`mt-0.5 ${
              countWarning === 'error'
                ? 'text-red-400/80'
                : countWarning === 'warning'
                ? 'text-amber-400/80'
                : 'text-[var(--text-muted)]'
            }`}
          >
            {countWarning === 'error'
              ? 'Too many configurations. Please reduce selections to 50 or fewer.'
              : countWarning === 'warning'
              ? 'Large number of configurations may take a while to run.'
              : 'Each configuration will be evaluated against all queries.'}
          </p>
        </div>
      </div>
    </div>
  );
}
