import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import BenchmarksTab from '../BenchmarksTab'

import { useBenchmark } from '../../hooks/useBenchmarks'

vi.mock('../benchmarks/DatasetsView', () => ({
  DatasetsView: () => <div>DatasetsView</div>,
}))

vi.mock('../benchmarks/ResultsView', () => ({
  ResultsView: () => <div>ResultsView</div>,
}))

vi.mock('../benchmarks/BenchmarkProgress', () => ({
  BenchmarkProgress: ({ onComplete }: { onComplete: () => void }) => (
    <div>
      BenchmarkProgress <button onClick={onComplete}>complete</button>
    </div>
  ),
}))

vi.mock('../benchmarks/BenchmarksListView', () => ({
  BenchmarksListView: ({
    onViewResults,
  }: {
    onViewResults: (benchmarkId: string, status: 'running' | 'completed') => void
  }) => (
    <div>
      BenchmarksListView{' '}
      <button onClick={() => onViewResults('bench-1', 'completed')}>view results</button>{' '}
      <button onClick={() => onViewResults('bench-1', 'running')}>view running</button>
    </div>
  ),
}))

vi.mock('../../hooks/useBenchmarks', () => ({
  useBenchmark: vi.fn(),
}))

describe('BenchmarksTab', () => {
  it('defaults to Datasets sub-tab and can switch to Results', async () => {
    const user = userEvent.setup()
    vi.mocked(useBenchmark).mockReturnValue({ data: null } as unknown as ReturnType<typeof useBenchmark>)

    render(<BenchmarksTab />)
    expect(screen.getByText('DatasetsView')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /Benchmarks/i }))
    expect(screen.getByText('BenchmarksListView')).toBeInTheDocument()

    await user.click(screen.getByText('view results'))
    expect(screen.getByText('ResultsView')).toBeInTheDocument()
  })

  it('shows progress view when selected benchmark is running and allows returning', async () => {
    const user = userEvent.setup()

    vi.mocked(useBenchmark).mockImplementation((benchmarkId: string) => {
      return { data: benchmarkId ? { id: benchmarkId, status: 'running' } : null } as unknown as ReturnType<
        typeof useBenchmark
      >
    })

    render(<BenchmarksTab />)

    await user.click(screen.getByRole('button', { name: /Benchmarks/i }))
    await user.click(screen.getByText('view running'))
    expect(screen.getByText(/BenchmarkProgress/i)).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /Back to benchmarks list/i }))
    // After clearing selection, show list view again
    expect(screen.getByText('BenchmarksListView')).toBeInTheDocument()
  })
})
