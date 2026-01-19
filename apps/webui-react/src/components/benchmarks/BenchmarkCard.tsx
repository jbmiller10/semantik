/**
 * BenchmarkCard - Displays a single benchmark
 */

import {
  BarChart3,
  Play,
  Square,
  Trash2,
  Eye,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  AlertCircle,
} from 'lucide-react';
import type { Benchmark, BenchmarkStatus } from '../../types/benchmark';

interface BenchmarkCardProps {
  benchmark: Benchmark;
  onStart: () => void;
  onCancel: () => void;
  onViewResults: () => void;
  onDelete: () => void;
  isStarting?: boolean;
  isCancelling?: boolean;
  isDeleting?: boolean;
}

function getStatusConfig(status: BenchmarkStatus) {
  switch (status) {
    case 'completed':
      return {
        icon: CheckCircle,
        label: 'Completed',
        className: 'bg-green-500/20 text-green-400',
      };
    case 'running':
      return {
        icon: Loader2,
        label: 'Running',
        className: 'bg-blue-500/20 text-blue-400',
        animate: true,
      };
    case 'failed':
      return {
        icon: XCircle,
        label: 'Failed',
        className: 'bg-red-500/20 text-red-400',
      };
    case 'cancelled':
      return {
        icon: AlertCircle,
        label: 'Cancelled',
        className: 'bg-amber-500/20 text-amber-400',
      };
    case 'pending':
    default:
      return {
        icon: Clock,
        label: 'Pending',
        className: 'bg-gray-500/20 text-gray-400',
      };
  }
}

export function BenchmarkCard({
  benchmark,
  onStart,
  onCancel,
  onViewResults,
  onDelete,
  isStarting,
  isCancelling,
  isDeleting,
}: BenchmarkCardProps) {
  const statusConfig = getStatusConfig(benchmark.status);
  const StatusIcon = statusConfig.icon;

  const formatDate = (dateString: string | null) => {
    if (!dateString) return null;
    return new Date(dateString).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const progress =
    benchmark.total_runs > 0
      ? Math.round((benchmark.completed_runs / benchmark.total_runs) * 100)
      : 0;

  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-4 hover:border-[var(--border-strong)] transition-colors">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-[var(--bg-tertiary)] rounded-lg">
            <BarChart3 className="h-5 w-5 text-[var(--text-muted)]" />
          </div>
          <div>
            <h3 className="font-medium text-[var(--text-primary)]">{benchmark.name}</h3>
            {benchmark.description && (
              <p className="text-sm text-[var(--text-muted)] line-clamp-1">
                {benchmark.description}
              </p>
            )}
          </div>
        </div>
        <span
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${statusConfig.className}`}
        >
          <StatusIcon
            className={`h-3.5 w-3.5 ${statusConfig.animate ? 'animate-spin' : ''}`}
          />
          {statusConfig.label}
        </span>
      </div>

      {/* Progress (for running benchmarks) */}
      {benchmark.status === 'running' && (
        <div className="mb-4">
          <div className="flex items-center justify-between text-xs text-[var(--text-muted)] mb-1">
            <span>
              {benchmark.completed_runs} / {benchmark.total_runs} runs
            </span>
            <span>{progress}%</span>
          </div>
          <div className="h-1.5 bg-[var(--bg-tertiary)] rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="flex items-center gap-4 text-sm text-[var(--text-secondary)] mb-4">
        <div>
          <span className="font-medium text-[var(--text-primary)]">{benchmark.total_runs}</span>{' '}
          <span>total runs</span>
        </div>
        {benchmark.completed_runs > 0 && (
          <div>
            <span className="font-medium text-green-400">{benchmark.completed_runs}</span>{' '}
            <span>completed</span>
          </div>
        )}
        {benchmark.failed_runs > 0 && (
          <div>
            <span className="font-medium text-red-400">{benchmark.failed_runs}</span>{' '}
            <span>failed</span>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between pt-3 border-t border-[var(--border)]">
        <span className="text-xs text-[var(--text-muted)]">
          {benchmark.completed_at
            ? `Completed ${formatDate(benchmark.completed_at)}`
            : benchmark.started_at
            ? `Started ${formatDate(benchmark.started_at)}`
            : `Created ${formatDate(benchmark.created_at)}`}
        </span>
        <div className="flex items-center gap-2">
          {benchmark.status === 'pending' && (
            <button
              onClick={onStart}
              disabled={isStarting}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-gray-900 bg-gray-200 dark:bg-white rounded-lg hover:bg-gray-300 dark:hover:bg-gray-100 transition-colors disabled:opacity-50"
              title="Start benchmark"
            >
              {isStarting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Start
                </>
              )}
            </button>
          )}
          {benchmark.status === 'running' && (
            <button
              onClick={onCancel}
              disabled={isCancelling}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-[var(--text-secondary)] border border-[var(--border)] rounded-lg hover:bg-[var(--bg-tertiary)] transition-colors disabled:opacity-50"
              title="Cancel benchmark"
            >
              {isCancelling ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <>
                  <Square className="h-4 w-4" />
                  Cancel
                </>
              )}
            </button>
          )}
          {benchmark.status === 'completed' && (
            <button
              onClick={onViewResults}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-[var(--text-secondary)] border border-[var(--border)] rounded-lg hover:bg-[var(--bg-tertiary)] transition-colors"
              title="View results"
            >
              <Eye className="h-4 w-4" />
              Results
            </button>
          )}
          <button
            onClick={onDelete}
            disabled={isDeleting || benchmark.status === 'running'}
            className="p-2 text-[var(--text-secondary)] hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title={benchmark.status === 'running' ? 'Cannot delete running benchmark' : 'Delete benchmark'}
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
