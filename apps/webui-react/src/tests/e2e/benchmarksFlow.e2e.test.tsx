import React from 'react'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { App } from '../../App'
import { resetBenchmarkMocks, setBenchmarkAutoCompleteOnStart } from '../mocks/handlers'

describe('E2E Benchmarks Flow (mocked)', () => {
  beforeEach(() => {
    localStorage.clear()
    sessionStorage.clear()
    resetBenchmarkMocks()
    setBenchmarkAutoCompleteOnStart(true)
  })

  it('uploads dataset, maps to a collection, creates benchmark, starts, and views results', async () => {
    const user = userEvent.setup()

    render(<App />)

    // Navigate to Benchmarks main tab
    const benchmarksTab = await screen.findByRole('tab', { name: /benchmarks/i })
    await user.click(benchmarksTab)

    // Upload dataset
    await user.click(screen.getByRole('button', { name: /upload dataset/i }))

    const datasetFile = new File(
      [
        JSON.stringify({
          schema_version: '1.0',
          queries: [
            {
              query_key: 'q1',
              query_text: 'hello',
              relevant_docs: [{ doc_ref: { uri: 'file:///tmp/doc.txt' }, relevance_grade: 2 }],
            },
          ],
        }),
      ],
      'my-dataset.json',
      { type: 'application/json' }
    )

    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement | null
    expect(fileInput).not.toBeNull()
    await user.upload(fileInput as HTMLInputElement, datasetFile)

    // Wait for parse preview and then submit
    await screen.findByText('my-dataset.json')
    const uploadButtons = screen.getAllByRole('button', { name: /upload dataset/i })
    await user.click(uploadButtons[uploadButtons.length - 1])

    // Dataset appears in list
    await screen.findByText(/my-dataset/i)

    // Open mapping management
    await user.click(screen.getByTitle('View mappings'))
    await screen.findByText(/Manage collection mappings/i)

    // Create mapping
    await user.click(screen.getByRole('button', { name: /add collection mapping/i }))
    const mappingPanel = screen.getByText(/Manage collection mappings/i).closest('div') ?? document.body
    const collectionSelect = within(mappingPanel).getByRole('combobox')
    await user.selectOptions(collectionSelect, ['123e4567-e89b-12d3-a456-426614174000'])
    await user.click(within(mappingPanel).getByRole('button', { name: /^Create$/i }))

    // Resolve mapping
    await screen.findByText(/Existing Mappings/i)
    await user.click(screen.getByRole('button', { name: /Resolve/i }))
    await screen.findByText(/Resolved/i)

    // Switch to Benchmarks sub-tab and create a new benchmark
    await user.click(screen.getByRole('button', { name: /^Benchmarks$/i }))
    await user.click(screen.getByRole('button', { name: /new benchmark/i }))

    const modalTitle = await screen.findByText('Create Benchmark')
    const modal = modalTitle.closest('div') ?? document.body

    await user.type(within(modal).getByPlaceholderText(/Q1 2024/i), 'E2E Benchmark')

    const selects = within(modal).getAllByRole('combobox')
    // Dataset select is first
    const datasetOption = Array.from(selects[0].querySelectorAll('option')).find((o) =>
      (o.textContent ?? '').toLowerCase().includes('my-dataset')
    )
    expect(datasetOption).toBeTruthy()
    await user.selectOptions(selects[0], (datasetOption as HTMLOptionElement).value)
    // Mapping select should appear after dataset selection
    await waitFor(() => {
      expect(within(modal).getAllByRole('combobox').length).toBeGreaterThanOrEqual(2)
    })
    const updatedSelects = within(modal).getAllByRole('combobox')
    const mappingOption = Array.from(updatedSelects[1].querySelectorAll('option')).find((o) =>
      (o.textContent ?? '').includes('Test Collection 1')
    )
    expect(mappingOption).toBeTruthy()
    await user.selectOptions(updatedSelects[1], (mappingOption as HTMLOptionElement).value)

    await user.click(within(modal).getByRole('button', { name: /create benchmark/i }))

    // Start benchmark (auto-completes via mock)
    await screen.findByText('E2E Benchmark')
    await user.click(screen.getByRole('button', { name: /Start/i }))

    // Benchmark should transition to completed
    await screen.findByText(/Completed/i)

    // Navigate to results and open comparison
    await user.click(screen.getByRole('button', { name: /^Results$/i }))
    await screen.findByPlaceholderText(/Search completed benchmarks/i)
    await user.click(screen.getByRole('button', { name: /E2E Benchmark/i }))

    await screen.findByText(/configuration.*evaluated/i)
    await screen.findByText(/P@10/i)
    await screen.findByText(/MRR/i)
  })
})
