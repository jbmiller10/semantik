/**
 * QueryResultsPanel - Per-query results drill-down
 */

import { useState } from 'react';
import { ArrowLeft, ChevronLeft, ChevronRight, Loader2, Search } from 'lucide-react';
import { useBenchmarkQueryResults } from '../../hooks/useBenchmarks';

interface QueryResultsPanelProps {
  benchmarkId: string;
  runId: string;
  onBack: () => void;
}

const PAGE_SIZE = 20;

export function QueryResultsPanel({ benchmarkId, runId, onBack }: QueryResultsPanelProps) {
  const [page, setPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');

  const { data, isLoading, error } = useBenchmarkQueryResults(benchmarkId, runId, {
    page,
    per_page: PAGE_SIZE,
  });

  const results = data?.results ?? [];
  const totalResults = data?.total ?? 0;
  const totalPages = Math.ceil(totalResults / PAGE_SIZE);

  // Filter by search
  const filteredResults = searchQuery
    ? results.filter(
        (r) =>
          r.query_text.toLowerCase().includes(searchQuery.toLowerCase()) ||
          r.query_key.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : results;

  if (isLoading && results.length === 0) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-[var(--text-muted)]" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
        <p className="text-red-400">Failed to load query results: {error.message}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={onBack}
            aria-label="Back"
            className="p-2 text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded-lg transition-colors"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <div>
            <h3 className="text-lg font-bold text-[var(--text-primary)]">Query Results</h3>
            <p className="text-sm text-[var(--text-muted)]">
              {totalResults} queries evaluated
            </p>
          </div>
        </div>

        {/* Search */}
        <div className="relative w-64">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--text-muted)]" />
          <input
            type="text"
            placeholder="Filter queries..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 input-field rounded-xl text-sm"
          />
        </div>
      </div>

      {/* Results Table */}
      <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-[var(--border)] bg-[var(--bg-tertiary)]">
                <th className="px-4 py-3 text-left text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">
                  Query
                </th>
                <th className="px-4 py-3 text-right text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">
                  P@K
                </th>
                <th className="px-4 py-3 text-right text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">
                  R@K
                </th>
                <th className="px-4 py-3 text-right text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">
                  RR
                </th>
                <th className="px-4 py-3 text-right text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">
                  nDCG
                </th>
                <th className="px-4 py-3 text-right text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">
                  Search ms
                </th>
                <th className="px-4 py-3 text-right text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">
                  Rerank ms
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--border)]">
              {filteredResults.map((result) => (
                <tr key={result.query_id} className="hover:bg-[var(--bg-tertiary)] transition-colors">
                  <td className="px-4 py-3">
                    <div className="max-w-md">
                      <p className="text-sm text-[var(--text-primary)] truncate">
                        {result.query_text}
                      </p>
                      <p className="text-xs text-[var(--text-muted)] font-mono mt-0.5">
                        {result.query_key}
                      </p>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <MetricValue value={result.precision_at_k} />
                  </td>
                  <td className="px-4 py-3 text-right">
                    <MetricValue value={result.recall_at_k} />
                  </td>
                  <td className="px-4 py-3 text-right">
                    <MetricValue value={result.reciprocal_rank} />
                  </td>
                  <td className="px-4 py-3 text-right">
                    <MetricValue value={result.ndcg_at_k} />
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className="text-sm text-[var(--text-secondary)]">
                      {result.search_time_ms != null
                        ? `${result.search_time_ms.toFixed(0)}`
                        : '-'}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <span className="text-sm text-[var(--text-secondary)]">
                      {result.rerank_time_ms != null
                        ? `${result.rerank_time_ms.toFixed(0)}`
                        : '-'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="px-4 py-3 border-t border-[var(--border)] flex items-center justify-between">
            <p className="text-sm text-[var(--text-muted)]">
              Showing {(page - 1) * PAGE_SIZE + 1} to{' '}
              {Math.min(page * PAGE_SIZE, totalResults)} of {totalResults} results
            </p>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                aria-label="Previous page"
                className="p-2 text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <span className="text-sm text-[var(--text-secondary)]">
                Page {page} of {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                aria-label="Next page"
                className="p-2 text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Empty State */}
      {filteredResults.length === 0 && !isLoading && (
        <div className="text-center py-8 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl">
          <p className="text-[var(--text-muted)]">
            {searchQuery ? 'No matching queries found' : 'No query results available'}
          </p>
        </div>
      )}
    </div>
  );
}

function MetricValue({ value }: { value: number | null }) {
  if (value === null) {
    return <span className="text-[var(--text-muted)]">-</span>;
  }

  const colorClass = getMetricColorClass(value);
  return <span className={`text-sm ${colorClass}`}>{value.toFixed(3)}</span>;
}

function getMetricColorClass(value: number): string {
  if (value >= 0.8) return 'text-green-400';
  if (value >= 0.5) return 'text-amber-400';
  if (value > 0) return 'text-red-400';
  return 'text-[var(--text-secondary)]';
}
