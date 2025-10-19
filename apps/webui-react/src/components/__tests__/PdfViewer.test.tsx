import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import PdfViewer from '../PdfViewer'

const { mockGetDocument, mockDestroy } = vi.hoisted(() => ({
  mockGetDocument: vi.fn(),
  mockDestroy: vi.fn(),
}))

vi.mock('pdfjs-dist/build/pdf.worker.min.mjs?url', () => ({
  default: 'worker-src',
}))

vi.mock('pdfjs-dist', () => ({
  getDocument: mockGetDocument,
  GlobalWorkerOptions: { workerSrc: '' },
}))

describe('PdfViewer', () => {
  beforeEach(() => {
    mockGetDocument.mockReset()
    mockDestroy.mockReset()
  })

  it('renders all PDF pages into canvas elements', async () => {
    const renderPromise = Promise.resolve()
    const renderPage = vi.fn(() => ({ promise: renderPromise }))
    const getPage = vi.fn(async () => ({
      getViewport: () => ({ width: 100, height: 200 }),
      render: renderPage,
    }))

    mockGetDocument.mockReturnValue({
      promise: Promise.resolve({
        numPages: 2,
        getPage,
        destroy: mockDestroy,
      }),
    })

    render(<PdfViewer src="blob:test" />)

    await waitFor(() => {
      expect(getPage).toHaveBeenCalledTimes(2)
      expect(renderPage).toHaveBeenCalledTimes(2)
      expect(screen.queryByRole('status')).not.toBeInTheDocument()
      const canvases = document.querySelectorAll('canvas')
      expect(canvases.length).toBe(2)
    })
  })

  it('calls onError when rendering fails', async () => {
    const onError = vi.fn()
    mockGetDocument.mockReturnValue({
      promise: Promise.reject(new Error('render failed')),
    })

    render(<PdfViewer src="blob:error" onError={onError} />)

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith('Unable to render this PDF in the browser. You can download the file instead.')
      expect(screen.getByRole('alert')).toHaveTextContent('render failed')
    })
  })

  it('cleans up the PDF document on unmount', async () => {
    const renderPromise = Promise.resolve()
    const renderPage = vi.fn(() => ({ promise: renderPromise }))
    const getPage = vi.fn(async () => ({
      getViewport: () => ({ width: 100, height: 200 }),
      render: renderPage,
    }))

    mockGetDocument.mockReturnValue({
      promise: Promise.resolve({
        numPages: 1,
        getPage,
        destroy: mockDestroy,
      }),
    })

    const { unmount } = render(<PdfViewer src="blob:cleanup" />)

    await waitFor(() => {
      expect(getPage).toHaveBeenCalled()
    })

    unmount()
    expect(mockDestroy).toHaveBeenCalled()
  })
})
