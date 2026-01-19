/**
 * CreateBenchmarkModal - Modal for creating a new benchmark
 */

import { useState, useEffect, useMemo } from 'react';
import { X, Loader2, Database, Link2, AlertCircle } from 'lucide-react';
import {
  useBenchmarkDatasets,
  useDatasetMappings,
  useCreateBenchmark,
} from '../../hooks/useBenchmarks';
import { useCollections } from '../../hooks/useCollections';
import { ConfigMatrixBuilder } from './ConfigMatrixBuilder';
import type { ConfigMatrixItem } from '../../types/benchmark';

interface CreateBenchmarkModalProps {
  onClose: () => void;
  onSuccess: () => void;
}

interface FormState {
  name: string;
  description: string;
  selectedDatasetId: string;
  selectedMappingId: number | null;
  configMatrix: ConfigMatrixItem;
  selectedMetrics: Set<string>;
  topK: number;
}

const DEFAULT_CONFIG: ConfigMatrixItem = {
  search_modes: ['dense'],
  use_reranker: [false],
  top_k_values: [10],
  rrf_k_values: [60],
  score_thresholds: [null],
};

const AVAILABLE_METRICS = [
  { id: 'precision', label: 'Precision@K', description: 'Fraction of retrieved docs that are relevant' },
  { id: 'recall', label: 'Recall@K', description: 'Fraction of relevant docs that were retrieved' },
  { id: 'mrr', label: 'MRR', description: 'Mean Reciprocal Rank of first relevant result' },
  { id: 'ndcg', label: 'nDCG@K', description: 'Normalized Discounted Cumulative Gain' },
];

const DEFAULT_FORM_STATE: FormState = {
  name: '',
  description: '',
  selectedDatasetId: '',
  selectedMappingId: null,
  configMatrix: DEFAULT_CONFIG,
  selectedMetrics: new Set(['precision', 'recall', 'mrr', 'ndcg']),
  topK: 10,
};

export function CreateBenchmarkModal({ onClose, onSuccess }: CreateBenchmarkModalProps): React.ReactElement {
  const [form, setForm] = useState<FormState>(DEFAULT_FORM_STATE);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const { data: datasetsResponse } = useBenchmarkDatasets();
  const { data: mappings } = useDatasetMappings(form.selectedDatasetId);
  const { data: collections } = useCollections();
  const createMutation = useCreateBenchmark();

  // Build collection name lookup from already-loaded collections list (avoids N+1 queries)
  const collectionNameById = useMemo(() => {
    const map = new Map<string, string>();
    for (const c of collections ?? []) {
      map.set(c.id, c.name);
    }
    return map;
  }, [collections]);

  // Filter to only resolved mappings
  const resolvedMappings = useMemo(
    () => mappings?.filter((m) => m.mapping_status === 'resolved') ?? [],
    [mappings]
  );

  // Get collection info for the selected mapping to determine feature availability
  const selectedMapping = mappings?.find((m) => m.id === form.selectedMappingId);
  const selectedCollection = collections?.find((c) => c.id === selectedMapping?.collection_id);

  // Determine feature availability from collection
  const hasReranker = true; // Assume reranker is available
  const hasSparseIndex = Boolean(
    (selectedCollection?.metadata as Record<string, unknown> | undefined)?.sparse_index_enabled
  );

  // Compute the first resolved mapping ID (stable reference for useEffect)
  const firstResolvedMappingId = resolvedMappings[0]?.id ?? null;

  // Reset mapping when dataset changes, auto-select first resolved mapping
  useEffect(() => {
    setForm((prev) => ({ ...prev, selectedMappingId: firstResolvedMappingId }));
  }, [form.selectedDatasetId, firstResolvedMappingId]);

  // Handle escape key
  useEffect(() => {
    function handleEscape(e: KeyboardEvent): void {
      if (e.key === 'Escape' && !createMutation.isPending) {
        onClose();
      }
    }
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [onClose, createMutation.isPending]);

  function toggleMetric(metricId: string): void {
    setForm((prev) => {
      const updated = new Set(prev.selectedMetrics);
      if (updated.has(metricId)) {
        updated.delete(metricId);
      } else {
        updated.add(metricId);
      }
      return { ...prev, selectedMetrics: updated };
    });
  }

  function validateForm(): boolean {
    const newErrors: Record<string, string> = {};

    if (!form.name.trim()) {
      newErrors.name = 'Benchmark name is required';
    }

    if (!form.selectedMappingId) {
      newErrors.mapping = 'Please select a dataset and mapping';
    }

    if (form.configMatrix.search_modes.length === 0) {
      newErrors.config = 'Select at least one search mode';
    } else if (form.configMatrix.use_reranker.length === 0) {
      newErrors.config = 'Select at least one reranker option';
    } else if (form.configMatrix.top_k_values.length === 0) {
      newErrors.config = 'Select at least one Top-K value';
    }

    if (form.selectedMetrics.size === 0) {
      newErrors.metrics = 'Select at least one metric';
    }

    // Check config count
    const modes = form.configMatrix.search_modes.length || 1;
    const rerankerOptions = form.configMatrix.use_reranker.length || 1;
    const topKCount = form.configMatrix.top_k_values.length || 1;
    const configCount = modes * rerankerOptions * topKCount;

    if (configCount > 50) {
      newErrors.config = 'Too many configurations (max 50). Please reduce selections.';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }

  async function handleSubmit(e: React.FormEvent): Promise<void> {
    e.preventDefault();

    if (!validateForm() || !form.selectedMappingId) {
      return;
    }

    try {
      await createMutation.mutateAsync({
        name: form.name.trim(),
        description: form.description.trim() || undefined,
        mapping_id: form.selectedMappingId,
        config_matrix: form.configMatrix,
        top_k: form.topK,
        metrics_to_compute: Array.from(form.selectedMetrics),
      });
      onSuccess();
    } catch {
      // Error handled by mutation
    }
  }

  const datasets = datasetsResponse?.datasets ?? [];
  const isSubmitting = createMutation.isPending;

  function updateField<K extends keyof FormState>(field: K, value: FormState[K]): void {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  return (
    <div className="fixed inset-0 bg-black/50 dark:bg-black/80 flex items-center justify-center p-4 z-50">
      <div className="panel w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl shadow-2xl">
        {/* Header */}
        <div className="px-6 py-5 border-b border-[var(--border)] flex items-center justify-between sticky top-0 bg-[var(--bg-secondary)] z-10">
          <div>
            <h3 className="text-xl font-bold text-[var(--text-primary)]">Create Benchmark</h3>
            <p className="mt-1 text-sm text-[var(--text-muted)]">
              Configure and create a new retrieval quality benchmark
            </p>
          </div>
          <button
            onClick={onClose}
            disabled={isSubmitting}
            className="p-2 text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded-lg transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="px-6 py-4 space-y-6">
            {/* Name */}
            <div>
              <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                Benchmark Name <span className="text-red-400">*</span>
              </label>
              <input
                type="text"
                value={form.name}
                onChange={(e) => updateField('name', e.target.value)}
                disabled={isSubmitting}
                className={`w-full px-4 py-2.5 input-field rounded-xl ${errors.name ? 'border-red-500/50' : ''}`}
                placeholder="Q1 2024 Search Quality Test"
              />
              {errors.name && (
                <p className="mt-1 text-sm text-red-400">{errors.name}</p>
              )}
            </div>

            {/* Description */}
            <div>
              <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                Description
              </label>
              <textarea
                value={form.description}
                onChange={(e) => updateField('description', e.target.value)}
                disabled={isSubmitting}
                rows={2}
                className="w-full px-4 py-2.5 input-field rounded-xl"
                placeholder="Optional description..."
              />
            </div>

            {/* Dataset & Mapping Selection */}
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                  Dataset <span className="text-red-400">*</span>
                </label>
                <div className="relative">
                  <Database className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--text-muted)]" />
                  <select
                    value={form.selectedDatasetId}
                    onChange={(e) => updateField('selectedDatasetId', e.target.value)}
                    disabled={isSubmitting}
                    className="w-full pl-10 pr-4 py-2.5 input-field rounded-xl"
                  >
                    <option value="">Select a dataset...</option>
                    {datasets.map((dataset) => (
                      <option key={dataset.id} value={dataset.id}>
                        {dataset.name} ({dataset.query_count} queries)
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {form.selectedDatasetId && (
                <div>
                  <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                    Collection Mapping <span className="text-red-400">*</span>
                  </label>
                  {resolvedMappings.length > 0 ? (
                    <div className="relative">
                      <Link2 className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--text-muted)]" />
                      <select
                        value={form.selectedMappingId ?? ''}
                        onChange={(e) => updateField('selectedMappingId', Number(e.target.value))}
                        disabled={isSubmitting}
                        className="w-full pl-10 pr-4 py-2.5 input-field rounded-xl"
                      >
                        {resolvedMappings.map((mapping) => (
                          <option key={mapping.id} value={mapping.id}>
                            {collectionNameById.get(mapping.collection_id) ?? mapping.collection_id} ({mapping.mapped_count}/{mapping.total_count} docs)
                          </option>
                        ))}
                      </select>
                    </div>
                  ) : (
                    <div className="flex items-start gap-2 p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
                      <AlertCircle className="h-5 w-5 text-amber-400 flex-shrink-0 mt-0.5" />
                      <div className="text-sm">
                        <p className="font-medium text-amber-300">No resolved mappings</p>
                        <p className="text-amber-400/80 mt-1">
                          This dataset has no fully resolved collection mappings. Please resolve a
                          mapping first in the Datasets tab.
                        </p>
                      </div>
                    </div>
                  )}
                  {errors.mapping && (
                    <p className="mt-1 text-sm text-red-400">{errors.mapping}</p>
                  )}
                </div>
              )}
            </div>

            {/* Configuration Matrix */}
            {form.selectedMappingId && (
              <div>
                <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-3">
                  Search Configuration
                </label>
                <ConfigMatrixBuilder
                  value={form.configMatrix}
                  onChange={(value) => updateField('configMatrix', value)}
                  hasReranker={hasReranker}
                  hasSparseIndex={hasSparseIndex}
                  disabled={isSubmitting}
                />
                {errors.config && (
                  <p className="mt-2 text-sm text-red-400">{errors.config}</p>
                )}
              </div>
            )}

            {/* Evaluation Top-K */}
            <div>
              <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                Evaluation Top-K
              </label>
              <input
                type="number"
                min={1}
                max={100}
                value={form.topK}
                onChange={(e) => updateField('topK', parseInt(e.target.value, 10) || 10)}
                disabled={isSubmitting}
                className="w-24 px-4 py-2 input-field rounded-xl"
              />
              <p className="mt-1 text-xs text-[var(--text-muted)]">
                Number of top results to consider for metric calculation
              </p>
            </div>

            {/* Metrics Selection */}
            <div>
              <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                Metrics to Compute
              </label>
              <div className="space-y-2">
                {AVAILABLE_METRICS.map((metric) => (
                  <label
                    key={metric.id}
                    className="flex items-start gap-3 p-3 bg-[var(--bg-tertiary)] rounded-lg cursor-pointer hover:bg-[var(--bg-secondary)] transition-colors"
                  >
                    <input
                      type="checkbox"
                      checked={form.selectedMetrics.has(metric.id)}
                      onChange={() => toggleMetric(metric.id)}
                      disabled={isSubmitting}
                      className="h-4 w-4 mt-0.5 bg-[var(--input-bg)] border-[var(--input-border)] text-gray-600 dark:text-white focus:ring-gray-400 dark:focus:ring-white rounded"
                    />
                    <div>
                      <span className="font-medium text-[var(--text-primary)]">{metric.label}</span>
                      <p className="text-xs text-[var(--text-muted)] mt-0.5">{metric.description}</p>
                    </div>
                  </label>
                ))}
              </div>
              {errors.metrics && (
                <p className="mt-1 text-sm text-red-400">{errors.metrics}</p>
              )}
            </div>
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-[var(--border)] flex justify-end gap-3 sticky bottom-0 bg-[var(--bg-secondary)]">
            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting}
              className="px-4 py-2 text-sm font-medium text-[var(--text-secondary)] border border-[var(--border)] rounded-xl hover:bg-[var(--bg-tertiary)] transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting || !form.selectedMappingId}
              className="px-6 py-2 text-sm font-bold text-gray-900 bg-gray-200 dark:bg-white rounded-xl hover:bg-gray-300 dark:hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Creating...
                </span>
              ) : (
                'Create Benchmark'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
