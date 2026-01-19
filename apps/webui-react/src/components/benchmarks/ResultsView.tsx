/**
 * ResultsView - List of completed benchmarks with results preview
 */

import { useState } from 'react';
import { Search, BarChart3, Loader2, ChevronRight, Calendar } from 'lucide-react';
import { useBenchmarks, useBenchmarkResults } from '../../hooks/useBenchmarks';
import { ResultsComparison } from './ResultsComparison';
import type { Benchmark } from '../../types/benchmark';

export function ResultsView() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedBenchmarkId, setSelectedBenchmarkId] = useState<string | null>(null);

  const { data: benchmarksResponse, isLoading, error } = useBenchmarks();

  // Filter to only completed benchmarks
  const completedBenchmarks = (benchmarksResponse?.benchmarks ?? []).filter(
    (b) => b.status === 'completed'
  );

  // Filter by search
  const filteredBenchmarks = completedBenchmarks.filter(
    (b) =>
      b.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      b.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-[var(--text-muted)]" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
        <p className="text-red-400">Failed to load results: {error.message}</p>
      </div>
    );
  }

  // Show comparison view if a benchmark is selected
  if (selectedBenchmarkId) {
    return (
      <ResultsComparison
        benchmarkId={selectedBenchmarkId}
        onBack={() => setSelectedBenchmarkId(null)}
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Search */}
      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--text-muted)]" />
        <input
          type="text"
          placeholder="Search completed benchmarks..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full pl-10 pr-4 py-2 input-field rounded-xl"
        />
      </div>

      {/* Results List */}
      {filteredBenchmarks.length === 0 ? (
        <div className="text-center py-12 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl">
          <BarChart3 className="mx-auto h-12 w-12 text-[var(--text-muted)]" />
          <h3 className="mt-4 text-sm font-medium text-[var(--text-primary)]">
            {searchQuery ? 'No matching results' : 'No completed benchmarks'}
          </h3>
          <p className="mt-2 text-sm text-[var(--text-muted)]">
            {searchQuery
              ? 'Try a different search term'
              : 'Run a benchmark to see results here'}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredBenchmarks.map((benchmark) => (
            <BenchmarkResultCard
              key={benchmark.id}
              benchmark={benchmark}
              onClick={() => setSelectedBenchmarkId(benchmark.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

interface BenchmarkResultCardProps {
  benchmark: Benchmark;
  onClick: () => void;
}

function BenchmarkResultCard({ benchmark, onClick }: BenchmarkResultCardProps) {
  // Fetch results summary for preview
  const { data: results } = useBenchmarkResults(benchmark.id);

  const formatDate = (dateString: string | null) => {
    if (!dateString) return 'Unknown';
    return new Date(dateString).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  // Find best metrics from completed runs
  const bestRun = results?.runs.reduce(
    (best, run) => {
      if (run.status !== 'completed') return best;
      const mrr = run.metrics?.mrr ?? 0;
      if (mrr > (best?.metrics?.mrr ?? 0)) return run;
      return best;
    },
    results?.runs[0]
  );

  return (
    <button
      onClick={onClick}
      className="w-full text-left bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-4 hover:border-[var(--border-strong)] hover:bg-[var(--bg-tertiary)] transition-all group"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4 flex-1 min-w-0">
          <div className="p-2 bg-[var(--bg-tertiary)] rounded-lg group-hover:bg-[var(--bg-secondary)]">
            <BarChart3 className="h-5 w-5 text-[var(--text-muted)]" />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="font-medium text-[var(--text-primary)] truncate">
              {benchmark.name}
            </h3>
            <div className="flex items-center gap-4 mt-1">
              <span className="text-sm text-[var(--text-muted)] flex items-center gap-1">
                <Calendar className="h-3.5 w-3.5" />
                {formatDate(benchmark.completed_at)}
              </span>
              <span className="text-sm text-[var(--text-muted)]">
                {benchmark.completed_runs} runs
              </span>
            </div>
          </div>
        </div>

        {/* Metrics Preview */}
        {bestRun && (
          <div className="flex items-center gap-6 mr-4">
            {bestRun.metrics?.mrr !== undefined && (
              <div className="text-right">
                <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider">MRR</p>
                <p className="text-lg font-medium text-[var(--text-primary)]">
                  {bestRun.metrics.mrr.toFixed(3)}
                </p>
              </div>
            )}
            {bestRun.metrics?.ndcg !== undefined && (
              <div className="text-right">
                <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider">nDCG</p>
                <p className="text-lg font-medium text-[var(--text-primary)]">
                  {bestRun.metrics.ndcg.toFixed(3)}
                </p>
              </div>
            )}
            {bestRun.metrics?.precision_at_k !== undefined && (
              <div className="text-right">
                <p className="text-xs text-[var(--text-muted)] uppercase tracking-wider">P@K</p>
                <p className="text-lg font-medium text-[var(--text-primary)]">
                  {bestRun.metrics.precision_at_k.toFixed(3)}
                </p>
              </div>
            )}
          </div>
        )}

        <ChevronRight className="h-5 w-5 text-[var(--text-muted)] group-hover:text-[var(--text-primary)] transition-colors" />
      </div>
    </button>
  );
}
