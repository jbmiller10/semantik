/**
 * BenchmarksTab - Placeholder component for Phase 5
 * Full implementation will be added in Phase 6
 */

import { BarChart3 } from 'lucide-react';

function BenchmarksTab() {
  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg">
      <div className="text-center py-12">
        <BarChart3 className="mx-auto h-12 w-12 text-[var(--text-muted)]" />
        <h3 className="mt-4 text-sm font-medium text-[var(--text-primary)]">
          Benchmarks
        </h3>
        <p className="mt-2 text-sm text-[var(--text-muted)]">
          Retrieval quality benchmarking coming soon
        </p>
      </div>
    </div>
  );
}

export default BenchmarksTab;
