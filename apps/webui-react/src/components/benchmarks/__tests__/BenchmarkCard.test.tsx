import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import type { Benchmark } from '../../../types/benchmark';
import { BenchmarkCard } from '../BenchmarkCard';

const baseBenchmark: Benchmark = {
  id: 'bench-1',
  name: 'Bench 1',
  description: null,
  owner_id: 1,
  mapping_id: 1,
  status: 'pending',
  total_runs: 10,
  completed_runs: 0,
  failed_runs: 0,
  created_at: '2025-01-01T00:00:00Z',
  started_at: null,
  completed_at: null,
  operation_uuid: null,
};

describe('BenchmarkCard', () => {
  it('renders pending benchmarks with a start button', () => {
    const onStart = vi.fn();
    render(
      <BenchmarkCard
        benchmark={{ ...baseBenchmark, status: 'pending' }}
        onStart={onStart}
        onCancel={vi.fn()}
        onViewResults={vi.fn()}
        onDelete={vi.fn()}
      />
    );

    expect(screen.getByText('Pending')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Start/i }));
    expect(onStart).toHaveBeenCalledTimes(1);
  });

  it('renders running benchmarks with progress and disables delete', () => {
    render(
      <BenchmarkCard
        benchmark={{ ...baseBenchmark, status: 'running', total_runs: 4, completed_runs: 1 }}
        onStart={vi.fn()}
        onCancel={vi.fn()}
        onViewResults={vi.fn()}
        onDelete={vi.fn()}
      />
    );

    expect(screen.getByText('Running')).toBeInTheDocument();
    expect(screen.getByText('1 / 4 runs')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Cancel/i })).toBeInTheDocument();

    const deleteButton = screen.getByTitle('Cannot delete running benchmark');
    expect(deleteButton).toBeDisabled();
  });

  it('renders completed benchmarks with a results button', () => {
    const onViewResults = vi.fn();
    render(
      <BenchmarkCard
        benchmark={{ ...baseBenchmark, status: 'completed', completed_runs: 10, total_runs: 10 }}
        onStart={vi.fn()}
        onCancel={vi.fn()}
        onViewResults={onViewResults}
        onDelete={vi.fn()}
      />
    );

    expect(screen.getByText('Completed')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Results/i }));
    expect(onViewResults).toHaveBeenCalledTimes(1);
  });
});

