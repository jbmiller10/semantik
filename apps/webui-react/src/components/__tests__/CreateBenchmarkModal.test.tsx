import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import { CreateBenchmarkModal } from '../benchmarks/CreateBenchmarkModal'
import { useCollection } from '../../hooks/useCollections'
import { useBenchmarkDatasets, useCreateBenchmark, useDatasetMappings } from '../../hooks/useBenchmarks'
import type { BenchmarkDataset, DatasetMapping } from '@/types/benchmark'

const COLLECTION_ID = '123e4567-e89b-12d3-a456-426614174000'

vi.mock('../../hooks/useBenchmarks', () => ({
  useBenchmarkDatasets: vi.fn(),
  useDatasetMappings: vi.fn(),
  useCreateBenchmark: vi.fn(),
}))

vi.mock('../../hooks/useCollections', () => ({
  useCollection: vi.fn(),
}))

const dataset: BenchmarkDataset = {
  id: 'ds-1',
  name: 'Seed Dataset',
  description: null,
  owner_id: 1,
  query_count: 1,
  schema_version: '1.0',
  created_at: '2025-01-01T00:00:00Z',
  updated_at: null,
}

const resolvedMapping: DatasetMapping = {
  id: 1,
  dataset_id: dataset.id,
  collection_id: COLLECTION_ID,
  mapping_status: 'resolved',
  mapped_count: 1,
  total_count: 1,
  created_at: '2025-01-01T00:00:00Z',
  resolved_at: '2025-01-02T00:00:00Z',
}

describe('CreateBenchmarkModal', () => {
  beforeEach(() => {
    vi.clearAllMocks()

    vi.mocked(useBenchmarkDatasets).mockReturnValue({
      data: { datasets: [dataset], total: 1, page: 1, per_page: 50 },
    } as never)

    vi.mocked(useDatasetMappings).mockImplementation((datasetId: string) => {
      return {
        data: datasetId ? [resolvedMapping] : [],
        isLoading: false,
      } as never
    })

    vi.mocked(useCollection).mockImplementation((collectionId: string) => {
      if (collectionId !== COLLECTION_ID) {
        return { data: undefined, isLoading: false } as never
      }
      return {
        data: { id: COLLECTION_ID, name: 'Test Collection 1', metadata: {} },
        isLoading: false,
      } as never
    })

    vi.mocked(useCreateBenchmark).mockReturnValue({ mutateAsync: vi.fn().mockResolvedValue({}), isPending: false } as never)
  })

  it('creates a benchmark using a resolved mapping', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    const onSuccess = vi.fn()

    const mutateAsync = vi.fn().mockResolvedValue({})
    vi.mocked(useCreateBenchmark).mockReturnValue({ mutateAsync, isPending: false } as never)

    render(<CreateBenchmarkModal onClose={onClose} onSuccess={onSuccess} />)

    await screen.findByRole('option', { name: /seed dataset/i })

    // Select dataset
    const datasetSelect = screen.getByRole('combobox')
    await user.selectOptions(datasetSelect, dataset.id)

    // Mapping select appears once mappings load
    await waitFor(() => expect(screen.getAllByRole('combobox')).toHaveLength(2))

    await user.type(screen.getByPlaceholderText('Q1 2024 Search Quality Test'), 'Bench 1')
    await user.click(screen.getByRole('button', { name: /create benchmark/i }))

    await waitFor(() => expect(onSuccess).toHaveBeenCalledTimes(1))
    expect(mutateAsync).toHaveBeenCalledWith(expect.objectContaining({ mapping_id: resolvedMapping.id }))
    expect(onClose).not.toHaveBeenCalled()
  })

  it('shows a warning when a dataset has no resolved mappings', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    const onSuccess = vi.fn()

    const pendingMapping: DatasetMapping = {
      ...resolvedMapping,
      mapping_status: 'pending',
      mapped_count: 0,
      resolved_at: null,
    }
    vi.mocked(useDatasetMappings).mockImplementation((datasetId: string) => {
      return {
        data: datasetId ? [pendingMapping] : [],
        isLoading: false,
      } as never
    })

    render(<CreateBenchmarkModal onClose={onClose} onSuccess={onSuccess} />)

    await screen.findByRole('option', { name: /seed dataset/i })
    const datasetSelect = screen.getByRole('combobox')
    await user.selectOptions(datasetSelect, dataset.id)

    await screen.findByText(/No resolved mappings/i)
    expect(screen.getByRole('button', { name: /create benchmark/i })).toBeDisabled()
  })

  it('requires at least one metric', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    const onSuccess = vi.fn()

    render(<CreateBenchmarkModal onClose={onClose} onSuccess={onSuccess} />)

    await screen.findByRole('option', { name: /seed dataset/i })
    const datasetSelect = screen.getByRole('combobox')
    await user.selectOptions(datasetSelect, dataset.id)
    await waitFor(() => expect(screen.getAllByRole('combobox')).toHaveLength(2))

    await user.type(screen.getByPlaceholderText('Q1 2024 Search Quality Test'), 'Bench 2')

    // Deselect all metrics
    await user.click(screen.getByRole('checkbox', { name: /precision@k/i }))
    await user.click(screen.getByRole('checkbox', { name: /recall@k/i }))
    await user.click(screen.getByRole('checkbox', { name: /\bmrr\b/i }))
    await user.click(screen.getByRole('checkbox', { name: /ndcg@k/i }))

    await user.click(screen.getByRole('button', { name: /create benchmark/i }))

    await screen.findByText(/select at least one metric/i)
    expect(onSuccess).not.toHaveBeenCalled()
  })
})
