import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { DatasetsView } from '../DatasetsView'

import { useBenchmarkDatasets, useDatasetMappings, useDeleteDataset } from '../../../hooks/useBenchmarks'

vi.mock('../../../hooks/useBenchmarks', () => ({
  useBenchmarkDatasets: vi.fn(),
  useDeleteDataset: vi.fn(),
  useDatasetMappings: vi.fn(),
}))

vi.mock('../DatasetUploadModal', () => ({
  DatasetUploadModal: ({ onClose }: { onClose: () => void }) => (
    <div>
      Upload modal <button onClick={onClose}>close</button>
    </div>
  ),
}))

vi.mock('../MappingManagementPanel', () => ({
  MappingManagementPanel: ({
    dataset,
    onBack,
  }: {
    dataset: { name: string }
    onBack: () => void
  }) => (
    <div>
      Mapping panel for {dataset.name} <button onClick={onBack}>back</button>
    </div>
  ),
}))

describe('DatasetsView', () => {
  it('shows loading state', () => {
    vi.mocked(useBenchmarkDatasets).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
    } as any)
    vi.mocked(useDeleteDataset).mockReturnValue({ mutate: vi.fn() } as any)

    const { container } = render(<DatasetsView />)
    expect(container.querySelector('.animate-spin')).toBeTruthy()
  })

  it('shows error state', () => {
    vi.mocked(useBenchmarkDatasets).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: { message: 'nope' },
    } as any)
    vi.mocked(useDeleteDataset).mockReturnValue({ mutate: vi.fn() } as any)

    render(<DatasetsView />)
    expect(screen.getByText(/Failed to load datasets: nope/i)).toBeInTheDocument()
  })

  it('renders datasets, filters by search, and can open mapping panel', async () => {
    const user = userEvent.setup()
    const deleteMutate = vi.fn()

    ;(window as any).confirm = vi.fn(() => true)

    vi.mocked(useBenchmarkDatasets).mockReturnValue({
      data: {
        datasets: [
          {
            id: 'ds-1',
            name: 'Alpha',
            description: 'First',
            owner_id: 1,
            query_count: 1,
            schema_version: '1.0',
            created_at: '2024-01-01T00:00:00Z',
            updated_at: null,
          },
          {
            id: 'ds-2',
            name: 'Beta',
            description: null,
            owner_id: 1,
            query_count: 2,
            schema_version: '1.0',
            created_at: '2024-01-01T00:00:00Z',
            updated_at: null,
          },
        ],
      },
      isLoading: false,
      error: null,
    } as any)

    vi.mocked(useDeleteDataset).mockReturnValue({
      mutate: deleteMutate,
      isPending: false,
      variables: undefined,
    } as any)

    vi.mocked(useDatasetMappings).mockImplementation((datasetId: string) => {
      return {
        data: datasetId === 'ds-1' ? [{ id: 1 }] : [],
      } as any
    })

    render(<DatasetsView />)

    expect(screen.getByText('Alpha')).toBeInTheDocument()
    expect(screen.getByText('Beta')).toBeInTheDocument()

    await user.type(screen.getByPlaceholderText('Search datasets...'), 'alp')
    expect(screen.getByText('Alpha')).toBeInTheDocument()
    expect(screen.queryByText('Beta')).not.toBeInTheDocument()

    await user.clear(screen.getByPlaceholderText('Search datasets...'))
    await user.click(screen.getAllByTitle('View mappings')[0])
    expect(screen.getByText(/Mapping panel for Alpha/i)).toBeInTheDocument()

    await user.click(screen.getByText('back'))
    expect(screen.getByText('Alpha')).toBeInTheDocument()

    await user.click(screen.getAllByTitle('Delete dataset')[0])
    expect(deleteMutate).toHaveBeenCalledWith('ds-1')
  })

  it('can open and close upload modal', async () => {
    const user = userEvent.setup()

    vi.mocked(useBenchmarkDatasets).mockReturnValue({
      data: { datasets: [] },
      isLoading: false,
      error: null,
    } as any)
    vi.mocked(useDeleteDataset).mockReturnValue({ mutate: vi.fn(), isPending: false } as any)

    render(<DatasetsView />)

    await user.click(screen.getByRole('button', { name: /Upload Dataset/i }))
    expect(screen.getByText(/Upload modal/i)).toBeInTheDocument()

    await user.click(screen.getByText('close'))
    expect(screen.queryByText(/Upload modal/i)).not.toBeInTheDocument()
  })
})
