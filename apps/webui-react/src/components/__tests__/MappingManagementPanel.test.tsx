import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import { MappingManagementPanel } from '../benchmarks/MappingManagementPanel'
import { useCollections } from '../../hooks/useCollections'
import { useDatasetMappings, useCreateMapping, useResolveMapping } from '../../hooks/useBenchmarks'
import { useMappingResolutionProgress } from '../../hooks/useMappingResolutionProgress'
import type { BenchmarkDataset, DatasetMapping } from '@/types/benchmark'

vi.mock('../../hooks/useCollections', () => ({
  useCollections: vi.fn(),
}))

vi.mock('../../hooks/useBenchmarks', () => ({
  useDatasetMappings: vi.fn(),
  useCreateMapping: vi.fn(),
  useResolveMapping: vi.fn(),
}))

vi.mock('../../hooks/useMappingResolutionProgress', () => ({
  useMappingResolutionProgress: vi.fn(),
}))

const dataset: BenchmarkDataset = {
  id: 'ds-1',
  name: 'Dataset 1',
  description: null,
  owner_id: 1,
  query_count: 3,
  schema_version: '1.0',
  created_at: '2025-01-01T00:00:00Z',
  updated_at: null,
}

describe('MappingManagementPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('shows empty state when no mappings exist', () => {
    vi.mocked(useDatasetMappings).mockReturnValue({ data: [], isLoading: false } as never)
    vi.mocked(useCollections).mockReturnValue({
      data: [{ id: 'c1', name: 'Collection 1', document_count: 10 }],
    } as never)
    vi.mocked(useCreateMapping).mockReturnValue({ mutateAsync: vi.fn(), isPending: false } as never)
    vi.mocked(useResolveMapping).mockReturnValue({ mutateAsync: vi.fn(), isPending: false } as never)
    vi.mocked(useMappingResolutionProgress).mockReturnValue({
      progress: {
        stage: 'pending',
        totalRefs: 0,
        processedRefs: 0,
        resolvedRefs: 0,
        ambiguousRefs: 0,
        unresolvedRefs: 0,
      },
      isConnected: false,
    } as never)

    render(<MappingManagementPanel dataset={dataset} onBack={vi.fn()} />)

    expect(screen.getByText(/no mappings yet/i)).toBeInTheDocument()
  })

  it('creates a mapping and then shows live resolution progress', async () => {
    const user = userEvent.setup()

    const mappings: DatasetMapping[] = [
      {
        id: 1,
        dataset_id: dataset.id,
        collection_id: 'c1',
        mapping_status: 'pending',
        mapped_count: 0,
        total_count: 10,
        created_at: '2025-01-01T00:00:00Z',
        resolved_at: null,
      },
      {
        id: 2,
        dataset_id: dataset.id,
        collection_id: 'c2',
        mapping_status: 'partial',
        mapped_count: 3,
        total_count: 10,
        created_at: '2025-01-01T00:00:00Z',
        resolved_at: null,
      },
      {
        id: 3,
        dataset_id: dataset.id,
        collection_id: 'c3',
        mapping_status: 'resolved',
        mapped_count: 10,
        total_count: 10,
        created_at: '2025-01-01T00:00:00Z',
        resolved_at: '2025-01-02T00:00:00Z',
      },
    ]

    vi.mocked(useDatasetMappings).mockReturnValue({ data: mappings, isLoading: false } as never)
    vi.mocked(useCollections).mockReturnValue({
      data: [
        { id: 'c1', name: 'Collection 1', document_count: 10 },
        { id: 'c2', name: 'Collection 2', document_count: 20 },
        { id: 'c3', name: 'Collection 3', document_count: 30 },
        { id: 'c4', name: 'Collection 4', document_count: 40 },
      ],
    } as never)

    const createMapping = vi.fn().mockResolvedValue({})
    vi.mocked(useCreateMapping).mockReturnValue({ mutateAsync: createMapping, isPending: false } as never)

    const resolveMapping = vi.fn().mockResolvedValue({ operation_uuid: 'op-1' })
    vi.mocked(useResolveMapping).mockReturnValue({ mutateAsync: resolveMapping, isPending: false } as never)

    vi.mocked(useMappingResolutionProgress).mockReturnValue({
      progress: {
        stage: 'resolving',
        totalRefs: 100,
        processedRefs: 25,
        resolvedRefs: 20,
        ambiguousRefs: 3,
        unresolvedRefs: 2,
      },
      isConnected: true,
    } as never)

    render(<MappingManagementPanel dataset={dataset} onBack={vi.fn()} />)

    // Add mapping flow (c4 is the only available collection)
    await user.click(screen.getByRole('button', { name: /add collection mapping/i }))
    const collectionSelect = screen.getByRole('combobox')
    await user.selectOptions(collectionSelect, 'c4')
    await user.click(screen.getByRole('button', { name: /create/i }))
    expect(createMapping).toHaveBeenCalledWith({ datasetId: dataset.id, data: { collection_id: 'c4' } })

    // Resolve a pending mapping
    await user.click(screen.getAllByRole('button', { name: /resolve/i })[0])
    expect(resolveMapping).toHaveBeenCalledWith({ datasetId: dataset.id, mappingId: 1 })

    // Live progress section appears
    expect(await screen.findByText(/^resolving$/i)).toBeInTheDocument()
    expect(screen.getByText(/live/i)).toBeInTheDocument()
    expect(screen.getByText(/25\s*\/\s*100 processed/i)).toBeInTheDocument()

    // Status badges
    expect(screen.getAllByText(/^resolved$/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/^partial$/i).length).toBeGreaterThan(0)
  })
})
