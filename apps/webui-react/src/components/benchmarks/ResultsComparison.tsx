/**
 * ResultsComparison - Full results visualization with table and charts
 */

import { useState, useMemo } from 'react';
import {
  ArrowLeft,
  Download,
  ArrowUpDown,
  Trophy,
  Loader2,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useBenchmark, useBenchmarkResults } from '../../hooks/useBenchmarks';
import { QueryResultsPanel } from './QueryResultsPanel';

interface ResultsComparisonProps {
  benchmarkId: string;
  onBack: () => void;
}

type SortField = 'config' | 'precision_at_k' | 'recall_at_k' | 'mrr' | 'ndcg' | 'latency';
type SortDirection = 'asc' | 'desc';

export function ResultsComparison({ benchmarkId, onBack }: ResultsComparisonProps) {
  const [sortField, setSortField] = useState<SortField>('mrr');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [showChart, setShowChart] = useState(true);
  const [selectedK, setSelectedK] = useState<number | null>(null);

  const { data: benchmark } = useBenchmark(benchmarkId);
  const { data: results, isLoading, error } = useBenchmarkResults(benchmarkId);

  const primaryK = results?.primary_k ?? 10;
  const activeK = selectedK ?? primaryK;
  const availableKValues = results?.k_values_for_metrics ?? [primaryK];

  const metricAtK = (values: Record<string, number> | undefined, k: number) => {
    if (!values) return undefined;
    const val = values[String(k)];
    return typeof val === 'number' ? val : undefined;
  };

  // Sort and process runs
  const sortedRuns = useMemo(() => {
    if (!results?.runs) return [];

    const completedRuns = results.runs.filter((r) => r.status === 'completed');

    return [...completedRuns].sort((a, b) => {
      let aVal: number | string = 0;
      let bVal: number | string = 0;

      switch (sortField) {
        case 'config':
          aVal = formatConfig(a.config);
          bVal = formatConfig(b.config);
          return sortDirection === 'asc'
            ? aVal.localeCompare(bVal)
            : bVal.localeCompare(aVal);
        case 'precision_at_k':
          aVal = metricAtK(a.metrics?.precision, activeK) ?? 0;
          bVal = metricAtK(b.metrics?.precision, activeK) ?? 0;
          break;
        case 'recall_at_k':
          aVal = metricAtK(a.metrics?.recall, activeK) ?? 0;
          bVal = metricAtK(b.metrics?.recall, activeK) ?? 0;
          break;
        case 'mrr':
          aVal = a.metrics?.mrr ?? 0;
          bVal = b.metrics?.mrr ?? 0;
          break;
        case 'ndcg':
          aVal = metricAtK(a.metrics?.ndcg, activeK) ?? 0;
          bVal = metricAtK(b.metrics?.ndcg, activeK) ?? 0;
          break;
        case 'latency':
          aVal = a.timing?.total_ms ?? 0;
          bVal = b.timing?.total_ms ?? 0;
          break;
      }

      return sortDirection === 'asc'
        ? (aVal as number) - (bVal as number)
        : (bVal as number) - (aVal as number);
    });
  }, [results?.runs, sortField, sortDirection, activeK]);

  // Find best values for highlighting
  const bestValues = useMemo(() => {
    if (sortedRuns.length === 0) return {};

    return {
      precision_at_k: Math.max(...sortedRuns.map((r) => metricAtK(r.metrics?.precision, activeK) ?? 0)),
      recall_at_k: Math.max(...sortedRuns.map((r) => metricAtK(r.metrics?.recall, activeK) ?? 0)),
      mrr: Math.max(...sortedRuns.map((r) => r.metrics?.mrr ?? 0)),
      ndcg: Math.max(...sortedRuns.map((r) => metricAtK(r.metrics?.ndcg, activeK) ?? 0)),
      latency: Math.min(...sortedRuns.map((r) => r.timing?.total_ms ?? Infinity)),
    };
  }, [sortedRuns, activeK]);

  // Chart data
  const chartData = useMemo(() => {
    return sortedRuns.slice(0, 10).map((run) => ({
      name: formatConfigShort(run.config),
      [`P@${activeK}`]: metricAtK(run.metrics?.precision, activeK) ?? 0,
      [`R@${activeK}`]: metricAtK(run.metrics?.recall, activeK) ?? 0,
      MRR: run.metrics?.mrr ?? 0,
      [`nDCG@${activeK}`]: metricAtK(run.metrics?.ndcg, activeK) ?? 0,
    }));
  }, [sortedRuns, activeK]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const exportResults = (format: 'json' | 'csv') => {
    if (!results) return;

    let content: string;
    let filename: string;
    let mimeType: string;

    if (format === 'json') {
      content = JSON.stringify(results, null, 2);
      filename = `benchmark-results-${benchmarkId}.json`;
      mimeType = 'application/json';
    } else {
      // CSV format
      const headers = [
        'Run',
        'Config',
        `P@${activeK}`,
        `R@${activeK}`,
        'MRR',
        `nDCG@${activeK}`,
        'Latency (ms)',
      ];
      const rows = sortedRuns.map((run) => [
        run.run_order,
        formatConfig(run.config),
        metricAtK(run.metrics?.precision, activeK)?.toFixed(4) ?? '',
        metricAtK(run.metrics?.recall, activeK)?.toFixed(4) ?? '',
        run.metrics?.mrr?.toFixed(4) ?? '',
        metricAtK(run.metrics?.ndcg, activeK)?.toFixed(4) ?? '',
        run.timing?.total_ms?.toFixed(0) ?? '',
      ]);

      content = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
      filename = `benchmark-results-${benchmarkId}.csv`;
      mimeType = 'text/csv';
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Show query results panel if a run is selected
  if (selectedRunId) {
    return (
      <QueryResultsPanel
        benchmarkId={benchmarkId}
        runId={selectedRunId}
        onBack={() => setSelectedRunId(null)}
      />
    );
  }

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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={onBack}
            className="p-2 text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded-lg transition-colors"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <div>
            <h3 className="text-lg font-bold text-[var(--text-primary)]">
              {benchmark?.name ?? 'Benchmark Results'}
            </h3>
            <p className="text-sm text-[var(--text-muted)]">
              {sortedRuns.length} configuration{sortedRuns.length !== 1 ? 's' : ''} evaluated
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {availableKValues.length > 1 && (
            <select
              value={activeK}
              onChange={(e) => setSelectedK(Number(e.target.value))}
              className="px-3 py-1.5 text-sm bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg text-[var(--text-secondary)]"
            >
              {availableKValues.map((k) => (
                <option key={k} value={k}>
                  K={k}
                </option>
              ))}
            </select>
          )}
          <button
            onClick={() => exportResults('csv')}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-[var(--text-secondary)] border border-[var(--border)] rounded-lg hover:bg-[var(--bg-tertiary)] transition-colors"
          >
            <Download className="h-4 w-4" />
            CSV
          </button>
          <button
            onClick={() => exportResults('json')}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-[var(--text-secondary)] border border-[var(--border)] rounded-lg hover:bg-[var(--bg-tertiary)] transition-colors"
          >
            <Download className="h-4 w-4" />
            JSON
          </button>
        </div>
      </div>

      {/* Chart Toggle */}
      <button
        onClick={() => setShowChart(!showChart)}
        className="flex items-center gap-2 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
      >
        {showChart ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        {showChart ? 'Hide Chart' : 'Show Chart'}
      </button>

      {/* Chart */}
      {showChart && chartData.length > 0 && (
        <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-4">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <XAxis
                dataKey="name"
                angle={-45}
                textAnchor="end"
                height={80}
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--bg-secondary)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                }}
                labelStyle={{ color: 'var(--text-primary)' }}
              />
              <Legend />
              <Bar dataKey={`P@${activeK}`} fill="#3b82f6" />
              <Bar dataKey={`R@${activeK}`} fill="#10b981" />
              <Bar dataKey="MRR" fill="#f59e0b" />
              <Bar dataKey={`nDCG@${activeK}`} fill="#8b5cf6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Results Table */}
      <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-[var(--border)] bg-[var(--bg-tertiary)]">
                <SortableHeader
                  field="config"
                  currentSort={sortField}
                  direction={sortDirection}
                  onSort={handleSort}
                >
                  Configuration
                </SortableHeader>
                <SortableHeader
                  field="precision_at_k"
                  currentSort={sortField}
                  direction={sortDirection}
                  onSort={handleSort}
                  align="right"
                >
                  P@{activeK}
                </SortableHeader>
                <SortableHeader
                  field="recall_at_k"
                  currentSort={sortField}
                  direction={sortDirection}
                  onSort={handleSort}
                  align="right"
                >
                  R@{activeK}
                </SortableHeader>
                <SortableHeader
                  field="mrr"
                  currentSort={sortField}
                  direction={sortDirection}
                  onSort={handleSort}
                  align="right"
                >
                  MRR
                </SortableHeader>
                <SortableHeader
                  field="ndcg"
                  currentSort={sortField}
                  direction={sortDirection}
                  onSort={handleSort}
                  align="right"
                >
                  nDCG@{activeK}
                </SortableHeader>
                <SortableHeader
                  field="latency"
                  currentSort={sortField}
                  direction={sortDirection}
                  onSort={handleSort}
                  align="right"
                >
                  Latency
                </SortableHeader>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--border)]">
              {sortedRuns.map((run) => {
                const isBestOverall =
                  run.metrics?.mrr === bestValues.mrr &&
                  (metricAtK(run.metrics?.ndcg, activeK) ?? 0) === bestValues.ndcg;

                return (
                  <tr
                    key={run.id}
                    onClick={() => setSelectedRunId(run.id)}
                    className={`
                      cursor-pointer hover:bg-[var(--bg-tertiary)] transition-colors
                      ${isBestOverall ? 'bg-green-500/5' : ''}
                    `}
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        {isBestOverall && (
                          <Trophy className="h-4 w-4 text-amber-400" />
                        )}
                        <span className="font-mono text-sm text-[var(--text-primary)]">
                          {formatConfig(run.config)}
                        </span>
                      </div>
                    </td>
                    <MetricCell
                      value={metricAtK(run.metrics?.precision, activeK)}
                      best={bestValues.precision_at_k}
                    />
                    <MetricCell
                      value={metricAtK(run.metrics?.recall, activeK)}
                      best={bestValues.recall_at_k}
                    />
                    <MetricCell value={run.metrics?.mrr ?? undefined} best={bestValues.mrr} />
                    <MetricCell value={metricAtK(run.metrics?.ndcg, activeK)} best={bestValues.ndcg} />
                    <td className="px-4 py-3 text-right">
                      <span
                        className={`text-sm ${
                          run.timing?.total_ms === bestValues.latency
                            ? 'text-green-400 font-medium'
                            : 'text-[var(--text-secondary)]'
                        }`}
                      >
                        {run.timing?.total_ms != null
                          ? `${run.timing.total_ms.toFixed(0)}ms`
                          : '-'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// Helper components and functions
function SortableHeader({
  field,
  currentSort,
  direction,
  onSort,
  align = 'left',
  children,
}: {
  field: SortField;
  currentSort: SortField;
  direction: SortDirection;
  onSort: (field: SortField) => void;
  align?: 'left' | 'right';
  children: React.ReactNode;
}) {
  const isActive = currentSort === field;

  return (
    <th
      className={`px-4 py-3 text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider cursor-pointer hover:text-[var(--text-primary)] transition-colors ${
        align === 'right' ? 'text-right' : 'text-left'
      }`}
      onClick={() => onSort(field)}
      aria-sort={isActive ? (direction === 'asc' ? 'ascending' : 'descending') : undefined}
    >
      <span className="inline-flex items-center gap-1">
        {children}
        <ArrowUpDown
          className={`h-3 w-3 ${isActive ? 'text-[var(--text-primary)]' : ''}`}
        />
      </span>
    </th>
  );
}

function MetricCell({ value, best }: { value?: number; best?: number }) {
  const isBest = value !== undefined && value === best;

  return (
    <td className="px-4 py-3 text-right">
      {value !== undefined ? (
        <span
          className={`text-sm ${
            isBest
              ? 'text-green-400 font-medium'
              : 'text-[var(--text-secondary)]'
          }`}
        >
          {value.toFixed(3)}
          {isBest && (
            <span className="ml-1 px-1.5 py-0.5 text-xs bg-green-500/20 rounded">
              best
            </span>
          )}
        </span>
      ) : (
        <span className="text-[var(--text-muted)]">-</span>
      )}
    </td>
  );
}

function formatConfig(config: Record<string, unknown>): string {
  const parts: string[] = [];
  if (config.search_mode) parts.push(String(config.search_mode));
  if (config.use_reranker === true) parts.push('rerank');
  if (config.top_k) parts.push(`k=${config.top_k}`);
  if (config.rrf_k && config.search_mode === 'hybrid') parts.push(`rrf=${config.rrf_k}`);
  return parts.join(' + ') || 'default';
}

function formatConfigShort(config: Record<string, unknown>): string {
  const parts: string[] = [];
  if (config.search_mode) parts.push(String(config.search_mode).charAt(0).toUpperCase());
  if (config.use_reranker === true) parts.push('R');
  if (config.top_k) parts.push(`${config.top_k}`);
  return parts.join('-') || 'def';
}
