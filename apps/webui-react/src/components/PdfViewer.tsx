import { useEffect, useRef, useState } from 'react'
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist'
import type { PDFDocumentProxy } from 'pdfjs-dist/types/src/display/api'
import workerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url'

interface PdfViewerProps {
  src: string
  className?: string
  onError?: (message: string) => void
}

if (typeof window !== 'undefined') {
  GlobalWorkerOptions.workerSrc = workerSrc
}

interface RenderState {
  totalPages: number
  renderedPages: number
  isRendering: boolean
  error: string | null
}

const INITIAL_RENDER_STATE: RenderState = {
  totalPages: 0,
  renderedPages: 0,
  isRendering: true,
  error: null,
}

function PdfViewer({ src, className, onError }: PdfViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [renderState, setRenderState] = useState<RenderState>(INITIAL_RENDER_STATE)

  useEffect(() => {
    let cancelled = false
    let pdfDoc: PDFDocumentProxy | null = null

    const renderDocument = async () => {
      setRenderState({ ...INITIAL_RENDER_STATE })

      try {
        const loadingTask = getDocument({ url: src })
        pdfDoc = await loadingTask.promise
        if (cancelled) {
          pdfDoc.destroy()
          return
        }

        setRenderState((state) => ({
          ...state,
          totalPages: pdfDoc!.numPages,
          error: null,
        }))

        const container = containerRef.current
        if (!container) {
          return
        }
        container.innerHTML = ''

        for (let pageNumber = 1; pageNumber <= pdfDoc.numPages; pageNumber += 1) {
          if (cancelled) break

          const page = await pdfDoc.getPage(pageNumber)
          if (cancelled) break

          const viewport = page.getViewport({ scale: 1.2 })
          const canvas = document.createElement('canvas')
          const context = canvas.getContext('2d')

          if (!context) {
            throw new Error('Unable to initialise canvas context for PDF rendering')
          }

          canvas.width = viewport.width
          canvas.height = viewport.height
          canvas.className = 'w-full rounded border border-gray-200 shadow-sm bg-white'

          container.appendChild(canvas)

          await page.render({ canvasContext: context, viewport }).promise
          setRenderState((state) => ({
            ...state,
            renderedPages: pageNumber,
          }))
        }

        setRenderState((state) => ({
          ...state,
          isRendering: false,
        }))
      } catch (error) {
        if (cancelled) return

        const message = error instanceof Error
          ? error.message
          : 'Failed to render PDF document'

        setRenderState({
          totalPages: 0,
          renderedPages: 0,
          isRendering: false,
          error: message,
        })

        onError?.('Unable to render this PDF in the browser. You can download the file instead.')
      }
    }

    renderDocument()

    return () => {
      cancelled = true
      if (pdfDoc) {
        pdfDoc.destroy()
      }
    }
  }, [src, onError])

  return (
    <div className={className}>
      {renderState.isRendering && !renderState.error && (
        <div className="flex items-center justify-center py-6 text-sm text-gray-600" role="status">
          <svg
            className="h-5 w-5 animate-spin text-blue-600"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
            />
          </svg>
          <span className="ml-3">
            Rendering PDF {renderState.renderedPages}/{renderState.totalPages || '?'}…
          </span>
        </div>
      )}

      {renderState.error ? (
        <div
          className="bg-red-50 border border-red-200 rounded-md p-4 text-sm text-red-700"
          role="alert"
        >
          {renderState.error}
        </div>
      ) : (
        <div
          ref={containerRef}
          className="space-y-6"
          data-testid="pdf-container"
          aria-label="PDF document"
        />
      )}
    </div>
  )
}

export default PdfViewer

