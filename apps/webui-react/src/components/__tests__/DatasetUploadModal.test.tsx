import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import { fireEvent } from '@testing-library/react'
import { DatasetUploadModal } from '../benchmarks/DatasetUploadModal'
import { resetBenchmarkMocks } from '@/tests/mocks/handlers'
import { useCreateMapping, useUploadDataset } from '../../hooks/useBenchmarks'

vi.mock('../../hooks/useBenchmarks', () => ({
  useUploadDataset: vi.fn(),
  useCreateMapping: vi.fn(),
}))

describe('DatasetUploadModal', () => {
  beforeEach(() => {
    resetBenchmarkMocks()
    vi.mocked(useUploadDataset).mockReturnValue({ mutateAsync: vi.fn(), isPending: false } as never)
    vi.mocked(useCreateMapping).mockReturnValue({ mutateAsync: vi.fn(), isPending: false } as never)
  })

  it('uploads a dataset and creates an optional mapping', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    const onSuccess = vi.fn()

    const uploadMutateAsync = vi.fn().mockResolvedValue({ id: 'ds-1', name: 'my-dataset' })
    const createMappingMutateAsync = vi.fn().mockResolvedValue({})
    vi.mocked(useUploadDataset).mockReturnValue({ mutateAsync: uploadMutateAsync, isPending: false } as never)
    vi.mocked(useCreateMapping).mockReturnValue({ mutateAsync: createMappingMutateAsync, isPending: false } as never)

    const { container } = render(<DatasetUploadModal onClose={onClose} onSuccess={onSuccess} />)

    await screen.findByText('Test Collection 1')

    const datasetFile = new File(
      [
        JSON.stringify({
          schema_version: '1.0',
          queries: [
            {
              query_key: 'q1',
              query_text: 'hello world',
              relevant_docs: ['doc-a', { doc_ref: 'doc-b', relevance_grade: 3 }],
            },
          ],
        }),
      ],
      'my-dataset.json',
      { type: 'application/json' }
    )

    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement | null
    expect(fileInput).toBeTruthy()
    await user.upload(fileInput!, datasetFile)

    await screen.findByText('q1')

    const nameInput = screen.getByPlaceholderText('My Benchmark Dataset') as HTMLInputElement
    await waitFor(() => expect(nameInput.value).toBe('my-dataset'))

    const mappingSelect = screen.getByRole('combobox') as HTMLSelectElement
    await user.selectOptions(mappingSelect, '123e4567-e89b-12d3-a456-426614174000')

    await user.click(screen.getByRole('button', { name: /upload dataset/i }))

    await waitFor(() => expect(onSuccess).toHaveBeenCalledTimes(1))
    expect(uploadMutateAsync).toHaveBeenCalledTimes(1)
    expect(createMappingMutateAsync).toHaveBeenCalledWith({
      datasetId: 'ds-1',
      data: { collection_id: '123e4567-e89b-12d3-a456-426614174000' },
    })
  })

  it('parses CSV and supports closing with escape', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    const onSuccess = vi.fn()

    const { container } = render(<DatasetUploadModal onClose={onClose} onSuccess={onSuccess} />)

    const csvText = [
      'query_key,query_text,doc1,doc2',
      'q1,"hello, world",doc-a,doc-b',
      'q2,bye,doc-c',
      '',
    ].join('\n')
    const csvFile = new File([csvText], 'example.csv', { type: 'text/csv' })

    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement | null
    expect(fileInput).toBeTruthy()
    await user.upload(fileInput!, csvFile)

    await screen.findByText('q1')
    await screen.findByText('q2')

    const submit = screen.getByRole('button', { name: /upload dataset/i })
    expect(submit).toBeEnabled()

    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(onSuccess).not.toHaveBeenCalled()
  })

  it('shows validation errors when submitting without required fields', async () => {
    const onClose = vi.fn()
    const onSuccess = vi.fn()

    const { container } = render(<DatasetUploadModal onClose={onClose} onSuccess={onSuccess} />)

    const form = container.querySelector('form')
    expect(form).toBeTruthy()

    fireEvent.submit(form!)

    expect(await screen.findByText('Dataset name is required')).toBeInTheDocument()
    expect(await screen.findByText('Please select a file to upload')).toBeInTheDocument()
  })

  it('shows a parse error when dropping an unsupported file type', async () => {
    const onClose = vi.fn()
    const onSuccess = vi.fn()

    render(<DatasetUploadModal onClose={onClose} onSuccess={onSuccess} />)

    const dropZone = screen
      .getByText('Drag and drop a file here, or click to browse')
      .closest('div')
    expect(dropZone).toBeTruthy()

    const badFile = new File(['hello'], 'oops.txt', { type: 'text/plain' })

    fireEvent.drop(dropZone!, { dataTransfer: { files: [badFile] } })

    expect(await screen.findByText('Please drop a JSON or CSV file')).toBeInTheDocument()
  })

  it('shows a parse error for invalid dataset schema', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    const onSuccess = vi.fn()

    const { container } = render(<DatasetUploadModal onClose={onClose} onSuccess={onSuccess} />)

    const badJson = new File([JSON.stringify({ schema_version: '1.0' })], 'bad.json', {
      type: 'application/json',
    })

    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement | null
    expect(fileInput).toBeTruthy()
    await user.upload(fileInput!, badJson)

    expect(
      await screen.findByText('Invalid dataset format: missing "queries" array')
    ).toBeInTheDocument()
  })
})
