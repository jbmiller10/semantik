/**
 * BenchmarksTab - Main container for benchmark management UI
 * Sub-navigation: Datasets | Benchmarks | Results
 */

import { useState } from 'react';
import { Database, BarChart3, Trophy } from 'lucide-react';
import { DatasetsView } from './benchmarks/DatasetsView';
import { BenchmarksListView } from './benchmarks/BenchmarksListView';
import { ResultsView } from './benchmarks/ResultsView';
import { useBenchmark } from '../hooks/useBenchmarks';
import { BenchmarkProgress } from './benchmarks/BenchmarkProgress';
import type { BenchmarkStatus } from '../types/benchmark';

type BenchmarkSubTab = 'datasets' | 'benchmarks' | 'results';

interface TabConfig {
  id: BenchmarkSubTab;
  label: string;
  icon: typeof Database;
}

const tabs: TabConfig[] = [
  { id: 'datasets', label: 'Datasets', icon: Database },
  { id: 'benchmarks', label: 'Benchmarks', icon: BarChart3 },
  { id: 'results', label: 'Results', icon: Trophy },
];

function BenchmarksTab() {
  const [activeSubTab, setActiveSubTab] = useState<BenchmarkSubTab>('datasets');
  const [selectedBenchmarkId, setSelectedBenchmarkId] = useState<string | null>(null);

  // If viewing a running benchmark, show progress
  const { data: selectedBenchmark } = useBenchmark(selectedBenchmarkId ?? '');

  // Show progress view for running benchmark
  if (selectedBenchmark?.status === 'running') {
    return (
      <div className="space-y-6">
        <BenchmarkProgress
          benchmark={selectedBenchmark}
          onComplete={() => {
            setActiveSubTab('results');
          }}
        />
        <button
          onClick={() => setSelectedBenchmarkId(null)}
          className="text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
        >
          &larr; Back to benchmarks list
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Sub-Tab Navigation */}
      <div className="border-b border-[var(--border)]">
        <nav className="-mb-px flex space-x-8" aria-label="Benchmark tabs">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeSubTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveSubTab(tab.id);
                  setSelectedBenchmarkId(null);
                }}
                className={`
                  whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm
                  ${
                    isActive
                      ? 'border-[var(--accent-primary)] text-[var(--accent-primary)]'
                      : 'border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--border-strong)]'
                  }
                `}
              >
                <Icon
                  className={`inline-block w-5 h-5 mr-2 -mt-0.5 ${
                    isActive ? 'text-[var(--accent-primary)]' : 'text-[var(--text-muted)]'
                  }`}
                />
                {tab.label}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Tab Content */}
      <div>
        {activeSubTab === 'datasets' && <DatasetsView />}
        {activeSubTab === 'benchmarks' && (
          <BenchmarksListView
            onViewResults={(benchmarkId: string, status: BenchmarkStatus) => {
              setSelectedBenchmarkId(benchmarkId);
              // Check if benchmark is running - if so, stay on benchmarks tab to show progress
              // Otherwise switch to results tab
              if (status !== 'running') {
                setActiveSubTab('results');
              }
            }}
          />
        )}
        {activeSubTab === 'results' && <ResultsView />}
      </div>
    </div>
  );
}

export default BenchmarksTab;
