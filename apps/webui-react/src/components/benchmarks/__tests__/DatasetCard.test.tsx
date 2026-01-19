import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { DatasetCard } from '../DatasetCard'

describe('DatasetCard', () => {
  it('renders dataset details and triggers actions', async () => {
    const user = userEvent.setup()
    const onViewMappings = vi.fn()
    const onDelete = vi.fn()

    render(
      <DatasetCard
        dataset={{
          id: 'ds-1',
          name: 'My Dataset',
          description: 'A description',
          owner_id: 1,
          query_count: 12,
          schema_version: '1.0',
          created_at: '2024-01-01T00:00:00Z',
          updated_at: null,
        }}
        mappingCount={3}
        onViewMappings={onViewMappings}
        onDelete={onDelete}
      />
    )

    expect(screen.getByText('My Dataset')).toBeInTheDocument()
    expect(screen.getByText('A description')).toBeInTheDocument()
    expect(screen.getByText('12')).toBeInTheDocument()
    expect(screen.getByText('3')).toBeInTheDocument()

    await user.click(screen.getByTitle('View mappings'))
    expect(onViewMappings).toHaveBeenCalledTimes(1)

    await user.click(screen.getByTitle('Delete dataset'))
    expect(onDelete).toHaveBeenCalledTimes(1)
  })

  it('disables delete button while deleting', () => {
    render(
      <DatasetCard
        dataset={{
          id: 'ds-1',
          name: 'My Dataset',
          description: null,
          owner_id: 1,
          query_count: 0,
          schema_version: '1.0',
          created_at: '2024-01-01T00:00:00Z',
          updated_at: null,
        }}
        mappingCount={0}
        onViewMappings={vi.fn()}
        onDelete={vi.fn()}
        isDeleting
      />
    )

    expect(screen.getByTitle('Delete dataset')).toBeDisabled()
  })
})
