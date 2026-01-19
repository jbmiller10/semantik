/**
 * BenchmarkProgress - Real-time progress display for running benchmarks
 */

import { Square, CheckCircle, Loader2, Wifi, WifiOff } from 'lucide-react';
import { useBenchmarkProgress } from '../../hooks/useBenchmarkProgress';
import { useCancelBenchmark } from '../../hooks/useBenchmarks';
import type { Benchmark } from '../../types/benchmark';

interface BenchmarkProgressProps {
  benchmark: Benchmark;
  onComplete?: () => void;
}

export function BenchmarkProgress({ benchmark, onComplete }: BenchmarkProgressProps) {
  const cancelMutation = useCancelBenchmark();

  const { progress, isConnected } = useBenchmarkProgress(
    benchmark.id,
    benchmark.operation_uuid,
    {
      onComplete,
    }
  );

  // Use WebSocket progress if available, otherwise fall back to benchmark data
  const totalRuns = progress.totalRuns || benchmark.total_runs;
  const completedRuns = progress.completedRuns || benchmark.completed_runs;
  const failedRuns = progress.failedRuns || benchmark.failed_runs;
  const doneRuns = completedRuns + failedRuns;
  const overallProgress = totalRuns > 0 ? Math.round((doneRuns / totalRuns) * 100) : 0;

  const currentQueryProgress =
    progress.currentQueries.total > 0
      ? Math.round((progress.currentQueries.processed / progress.currentQueries.total) * 100)
      : 0;

  const handleCancel = () => {
    if (confirm('Are you sure you want to cancel this benchmark?')) {
      cancelMutation.mutate(benchmark.id);
    }
  };

  // Format config for display
  const formatConfig = (config: Record<string, unknown>) => {
    const parts: string[] = [];
    if (config.search_mode) parts.push(String(config.search_mode));
    if (config.use_reranker) parts.push('rerank');
    if (config.top_k) parts.push(`k=${config.top_k}`);
    return parts.join(', ') || 'default';
  };

  const metricAtK = (values: Record<string, number> | undefined, k: number) => {
    if (!values) return null;
    const val = values[String(k)];
    return typeof val === 'number' ? val : null;
  };

  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Loader2 className="h-6 w-6 text-blue-400 animate-spin" />
          </div>
          <div>
            <h3 className="font-medium text-[var(--text-primary)]">{benchmark.name}</h3>
            <p className="text-sm text-[var(--text-muted)]">
              Running benchmark evaluation
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {/* Connection status */}
          <div
            className={`flex items-center gap-1.5 text-xs ${
              isConnected ? 'text-green-400' : 'text-amber-400'
            }`}
          >
            {isConnected ? (
              <Wifi className="h-4 w-4" />
            ) : (
              <WifiOff className="h-4 w-4" />
            )}
            {isConnected ? 'Live' : 'Reconnecting...'}
          </div>
          <button
            onClick={handleCancel}
            disabled={cancelMutation.isPending}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-[var(--text-secondary)] border border-[var(--border)] rounded-lg hover:bg-[var(--bg-tertiary)] transition-colors disabled:opacity-50"
          >
            {cancelMutation.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <>
                <Square className="h-4 w-4" />
                Cancel
              </>
            )}
          </button>
        </div>
      </div>

      {/* Overall Progress */}
      <div>
        <div className="flex items-center justify-between text-sm mb-2">
          <span className="text-[var(--text-secondary)]">Overall Progress</span>
          <span className="font-medium text-[var(--text-primary)]">
            {doneRuns} / {totalRuns} runs ({overallProgress}%)
            {failedRuns > 0 && (
              <span className="ml-2 text-xs text-red-400">
                ({failedRuns} failed)
              </span>
            )}
          </span>
        </div>
        <div className="h-3 bg-[var(--bg-tertiary)] rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 transition-all duration-300"
            style={{ width: `${overallProgress}%` }}
          />
        </div>
      </div>

      {/* Current Run Progress */}
      {progress.currentRunOrder > 0 && (
        <div className="p-4 bg-[var(--bg-tertiary)] rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-[var(--text-primary)]">
              Run {progress.currentRunOrder} of {totalRuns}
            </span>
            {progress.currentRunConfig && (
              <span className="text-xs text-[var(--text-muted)] font-mono">
                {formatConfig(progress.currentRunConfig)}
              </span>
            )}
          </div>
          <div className="flex items-center justify-between text-xs mb-1.5">
            <span className="text-[var(--text-muted)]">
              {progress.stage === 'evaluating' ? 'Evaluating queries' : progress.stage}
            </span>
            <span className="text-[var(--text-secondary)]">
              {progress.currentQueries.processed} / {progress.currentQueries.total} queries
            </span>
          </div>
          <div className="h-1.5 bg-[var(--bg-secondary)] rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500 transition-all duration-200"
              style={{ width: `${currentQueryProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Recent Completed Runs */}
      {progress.recentMetrics.length > 0 && (
        <div>
          <h4 className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-3">
            Recent Results
          </h4>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left py-2 px-3 text-xs font-medium text-[var(--text-muted)]">
                    Run
                  </th>
                  <th className="text-left py-2 px-3 text-xs font-medium text-[var(--text-muted)]">
                    Config
                  </th>
                  <th className="text-right py-2 px-3 text-xs font-medium text-[var(--text-muted)]">
                    MRR
                  </th>
                  <th className="text-right py-2 px-3 text-xs font-medium text-[var(--text-muted)]">
                    nDCG
                  </th>
                  <th className="text-right py-2 px-3 text-xs font-medium text-[var(--text-muted)]">
                    P@K
                  </th>
                  <th className="text-right py-2 px-3 text-xs font-medium text-[var(--text-muted)]">
                    Latency
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border)]">
                {progress.recentMetrics.map((run) => (
                  <tr key={run.runOrder}>
                    <td className="py-2 px-3 text-[var(--text-secondary)]">
                      <span className="flex items-center gap-1.5">
                        <CheckCircle className="h-3.5 w-3.5 text-green-400" />
                        #{run.runOrder}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-[var(--text-muted)] font-mono text-xs">
                      {formatConfig(run.config)}
                    </td>
                    <td className="py-2 px-3 text-right text-[var(--text-secondary)]">
                      {run.metrics.mrr != null ? run.metrics.mrr.toFixed(3) : '-'}
                    </td>
                    <td className="py-2 px-3 text-right text-[var(--text-secondary)]">
                      {metricAtK(run.metrics.ndcg, progress.primaryK)?.toFixed(3) ?? '-'}
                    </td>
                    <td className="py-2 px-3 text-right text-[var(--text-secondary)]">
                      {metricAtK(run.metrics.precision, progress.primaryK)?.toFixed(3) ?? '-'}
                    </td>
                    <td className="py-2 px-3 text-right text-[var(--text-muted)]">
                      {run.timing.total_ms != null
                        ? `${run.timing.total_ms.toFixed(0)}ms`
                        : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Completion Banner */}
      {progress.stage === 'completed' && (
        <div className="flex items-center gap-3 p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
          <CheckCircle className="h-5 w-5 text-green-400" />
          <div>
            <p className="font-medium text-green-300">Benchmark Complete</p>
            <p className="text-sm text-green-400/80">
              All {doneRuns} runs have finished. View results in the Results tab.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
