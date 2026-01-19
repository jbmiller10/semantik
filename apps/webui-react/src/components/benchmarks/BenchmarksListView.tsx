/**
 * BenchmarksListView - Benchmark list view with status filtering
 */

import { useState } from 'react';
import { Plus, Search, BarChart3, Loader2, Filter } from 'lucide-react';
import {
  useBenchmarks,
  useStartBenchmark,
  useCancelBenchmark,
  useDeleteBenchmark,
} from '../../hooks/useBenchmarks';
import { BenchmarkCard } from './BenchmarkCard';
import { CreateBenchmarkModal } from './CreateBenchmarkModal';
import type { BenchmarkStatus } from '../../types/benchmark';

interface BenchmarksListViewProps {
  onViewResults: (benchmarkId: string, status: BenchmarkStatus) => void;
}

const STATUS_FILTERS: Array<{ value: BenchmarkStatus | 'all'; label: string }> = [
  { value: 'all', label: 'All' },
  { value: 'pending', label: 'Pending' },
  { value: 'running', label: 'Running' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
];

export function BenchmarksListView({ onViewResults }: BenchmarksListViewProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<BenchmarkStatus | 'all'>('all');
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);

  const { data: benchmarksResponse, isLoading, error } = useBenchmarks();
  const startMutation = useStartBenchmark();
  const cancelMutation = useCancelBenchmark();
  const deleteMutation = useDeleteBenchmark();

  const benchmarks = benchmarksResponse?.benchmarks ?? [];

  // Filter benchmarks by search and status
  const filteredBenchmarks = benchmarks.filter((benchmark) => {
    const matchesSearch =
      benchmark.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      benchmark.description?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'all' || benchmark.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const handleStart = (benchmarkId: string) => {
    startMutation.mutate(benchmarkId);
  };

  const handleCancel = (benchmarkId: string) => {
    cancelMutation.mutate(benchmarkId);
  };

  const handleDelete = (benchmarkId: string) => {
    if (confirm('Are you sure you want to delete this benchmark? This action cannot be undone.')) {
      deleteMutation.mutate(benchmarkId);
    }
  };

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
        <p className="text-red-400">Failed to load benchmarks: {error.message}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row gap-4 sm:items-center sm:justify-between">
        <div className="flex items-center gap-3 flex-1">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--text-muted)]" />
            <input
              type="text"
              placeholder="Search benchmarks..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 input-field rounded-xl"
            />
          </div>
          <div className="relative">
            <Filter className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--text-muted)]" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as BenchmarkStatus | 'all')}
              className="pl-10 pr-8 py-2 input-field rounded-xl appearance-none cursor-pointer"
            >
              {STATUS_FILTERS.map((filter) => (
                <option key={filter.value} value={filter.value}>
                  {filter.label}
                </option>
              ))}
            </select>
          </div>
        </div>
        <button
          onClick={() => setIsCreateModalOpen(true)}
          className="inline-flex items-center gap-2 px-4 py-2 bg-gray-200 dark:bg-white text-gray-900 font-medium rounded-xl hover:bg-gray-300 dark:hover:bg-gray-100 transition-colors"
        >
          <Plus className="h-4 w-4" />
          New Benchmark
        </button>
      </div>

      {/* Benchmarks Grid */}
      {filteredBenchmarks.length === 0 ? (
        <div className="text-center py-12 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl">
          <BarChart3 className="mx-auto h-12 w-12 text-[var(--text-muted)]" />
          <h3 className="mt-4 text-sm font-medium text-[var(--text-primary)]">
            {searchQuery || statusFilter !== 'all' ? 'No benchmarks found' : 'No benchmarks yet'}
          </h3>
          <p className="mt-2 text-sm text-[var(--text-muted)]">
            {searchQuery || statusFilter !== 'all'
              ? 'Try different filters'
              : 'Create a benchmark to evaluate search quality'}
          </p>
          {!searchQuery && statusFilter === 'all' && (
            <button
              onClick={() => setIsCreateModalOpen(true)}
              className="mt-4 inline-flex items-center gap-2 px-4 py-2 bg-[var(--bg-tertiary)] border border-[var(--border)] text-[var(--text-primary)] font-medium rounded-xl hover:bg-[var(--bg-secondary)] transition-colors"
            >
              <Plus className="h-4 w-4" />
              Create Your First Benchmark
            </button>
          )}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filteredBenchmarks.map((benchmark) => (
            <BenchmarkCard
              key={benchmark.id}
              benchmark={benchmark}
              onStart={() => handleStart(benchmark.id)}
              onCancel={() => handleCancel(benchmark.id)}
              onViewResults={() => onViewResults(benchmark.id, benchmark.status)}
              onDelete={() => handleDelete(benchmark.id)}
              isStarting={startMutation.isPending && startMutation.variables === benchmark.id}
              isCancelling={cancelMutation.isPending && cancelMutation.variables === benchmark.id}
              isDeleting={deleteMutation.isPending && deleteMutation.variables === benchmark.id}
            />
          ))}
        </div>
      )}

      {/* Create Modal */}
      {isCreateModalOpen && (
        <CreateBenchmarkModal
          onClose={() => setIsCreateModalOpen(false)}
          onSuccess={() => setIsCreateModalOpen(false)}
        />
      )}
    </div>
  );
}
