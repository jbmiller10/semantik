/**
 * DatasetCard - Displays a single benchmark dataset
 */

import { Database, Link2, Trash2, Eye } from 'lucide-react';
import type { BenchmarkDataset } from '../../types/benchmark';

interface DatasetCardProps {
  dataset: BenchmarkDataset;
  mappingCount: number;
  onViewMappings: () => void;
  onDelete: () => void;
  isDeleting?: boolean;
}

export function DatasetCard({
  dataset,
  mappingCount,
  onViewMappings,
  onDelete,
  isDeleting,
}: DatasetCardProps) {
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-4 hover:border-[var(--border-strong)] transition-colors">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-[var(--bg-tertiary)] rounded-lg">
            <Database className="h-5 w-5 text-[var(--text-muted)]" />
          </div>
          <div>
            <h3 className="font-medium text-[var(--text-primary)]">{dataset.name}</h3>
            {dataset.description && (
              <p className="text-sm text-[var(--text-muted)] line-clamp-1">
                {dataset.description}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="flex items-center gap-4 text-sm text-[var(--text-secondary)] mb-4">
        <div className="flex items-center gap-1.5">
          <span className="font-medium text-[var(--text-primary)]">{dataset.query_count}</span>
          <span>queries</span>
        </div>
        <div className="flex items-center gap-1.5">
          <Link2 className="h-4 w-4" />
          <span className="font-medium text-[var(--text-primary)]">{mappingCount}</span>
          <span>mappings</span>
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between pt-3 border-t border-[var(--border)]">
        <span className="text-xs text-[var(--text-muted)]">
          Created {formatDate(dataset.created_at)}
        </span>
        <div className="flex items-center gap-2">
          <button
            onClick={onViewMappings}
            className="p-2 text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded-lg transition-colors"
            title="View mappings"
          >
            <Eye className="h-4 w-4" />
          </button>
          <button
            onClick={onDelete}
            disabled={isDeleting}
            className="p-2 text-[var(--text-secondary)] hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors disabled:opacity-50"
            title="Delete dataset"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
