import { describe, it, expect, vi, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import { act } from '@testing-library/react';
import EmbeddingVisualizationTab from '../EmbeddingVisualizationTab';
import type { ProjectionMetadata } from '../../types/projection';

let lastEmbeddingViewProps: any | null = null;
const setShowDocumentViewerMock = vi.fn();
const addToastMock = vi.fn();

vi.mock('../../hooks/useProjections', () => ({
  useCollectionProjections: vi.fn(),
  useStartProjection: vi.fn(),
  useDeleteProjection: vi.fn(),
}));

vi.mock('../../hooks/useProjectionTooltip', () => ({
  useProjectionTooltip: vi.fn(() => ({
    tooltipState: { status: 'idle', position: null, metadata: null },
    handleTooltip: vi.fn(),
    handleTooltipLeave: vi.fn(),
    clearTooltipCache: vi.fn(),
  })),
}));

vi.mock('../../hooks/useOperationProgress', () => ({
  useOperationProgress: vi.fn(() => ({ isConnected: false })),
}));

vi.mock('../../stores/uiStore', () => ({
  useUIStore: vi.fn(() => ({
    setShowDocumentViewer: setShowDocumentViewerMock,
    addToast: addToastMock,
  })),
}));

vi.mock('embedding-atlas/react', () => ({
  EmbeddingView: (props: any) => {
    lastEmbeddingViewProps = props;
    const labelCount = Array.isArray(props.labels) ? props.labels.length : 0;
    const labelsEnabled = props.labels ? 'true' : 'false';
    return (
      <div
        data-testid="embedding-view"
        data-label-count={labelCount}
        data-labels-enabled={labelsEnabled}
      />
    );
  },
}));

vi.mock('../../services/api/v2/projections', () => ({
  projectionsV2Api: {
    list: vi.fn(),
    getMetadata: vi.fn(),
    getArtifact: vi.fn(),
    start: vi.fn(),
    delete: vi.fn(),
    select: vi.fn(),
  },
}));

vi.mock('../../services/api/v2/collections', () => ({
  searchV2Api: {
    search: vi.fn(),
  },
}));

import { useCollectionProjections, useStartProjection, useDeleteProjection } from '../../hooks/useProjections';
import { useUIStore } from '../../stores/uiStore';
import { projectionsV2Api } from '../../services/api/v2/projections';
import { searchV2Api } from '../../services/api/v2/collections';

describe('EmbeddingVisualizationTab', () => {
  const collectionId = 'test-collection-id';

  beforeEach(() => {
    vi.clearAllMocks();
    setShowDocumentViewerMock.mockReset();
    addToastMock.mockReset();
    lastEmbeddingViewProps = null;

    const mockProjections: ProjectionMetadata[] = [
      {
        id: 'projection-1',
        collection_id: collectionId,
        status: 'completed',
        reducer: 'umap',
        dimensionality: 2,
        created_at: new Date().toISOString(),
        meta: {},
      },
    ];

    vi.mocked(useCollectionProjections).mockReturnValue({
      data: mockProjections,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    vi.mocked(useStartProjection).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false,
    } as any);

    vi.mocked(useDeleteProjection).mockReturnValue({
      mutateAsync: vi.fn(),
      isPending: false,
    } as any);

    vi.mocked(useUIStore).mockReturnValue({
      setShowDocumentViewer: setShowDocumentViewerMock,
      addToast: addToastMock,
    } as any);
  });

  it('passes labels to EmbeddingView when legend is present and toggle is used', async () => {
    const x = new Float32Array(20);
    const y = new Float32Array(20);
    const category = new Uint8Array(20);
    const ids = new Int32Array(20);

    for (let i = 0; i < 20; i += 1) {
      x[i] = i;
      y[i] = 0;
      category[i] = 0;
      ids[i] = i;
    }

    vi.mocked(projectionsV2Api.getMetadata).mockResolvedValue({
      data: {
        id: 'projection-1',
        collection_id: collectionId,
        status: 'completed',
        reducer: 'umap',
        dimensionality: 2,
        created_at: new Date().toISOString(),
        meta: {
          legend: [{ index: 0, label: 'Cluster A', count: 20 }],
          color_by: 'document_id',
          sampled: false,
          shown_count: 20,
          total_count: 20,
          degraded: false,
        },
      },
    } as any);

    vi.mocked(projectionsV2Api.getArtifact).mockImplementation(
      async (_collection, _projection, artifactName) => {
        if (artifactName === 'x') {
          return { data: x.buffer } as any;
        }
        if (artifactName === 'y') {
          return { data: y.buffer } as any;
        }
        if (artifactName === 'cat') {
          return { data: category.buffer } as any;
        }
        if (artifactName === 'ids') {
          return { data: ids.buffer } as any;
        }
        throw new Error(`Unexpected artifact request: ${artifactName}`);
      }
    );

    const user = userEvent.setup();

    render(<EmbeddingVisualizationTab collectionId={collectionId} />);

    const viewButton = await screen.findByRole('button', { name: /view/i });
    await user.click(viewButton);

    const embeddingView = await screen.findByTestId('embedding-view');

    // Labels should be enabled by default when legend is present and
    // createCategoryLabels returns at least one label.
    await waitFor(() => {
      expect(embeddingView.getAttribute('data-labels-enabled')).toBe('true');
      const labelCount = Number(embeddingView.getAttribute('data-label-count') ?? '0');
      expect(labelCount).toBeGreaterThan(0);
    });

    const labelsToggle = await screen.findByRole('checkbox', { name: /show labels/i });
    expect(labelsToggle).toBeChecked();

    await user.click(labelsToggle);

    await waitFor(() => {
      const updatedView = screen.getByTestId('embedding-view');
      expect(updatedView.getAttribute('data-labels-enabled')).toBe('false');
      const labelCount = Number(updatedView.getAttribute('data-label-count') ?? '0');
      expect(labelCount).toBe(0);
    });
  });

  it('shows degraded messaging in the selection panel when the projection is marked degraded', async () => {
    const x = new Float32Array(1);
    const y = new Float32Array(1);
    const category = new Uint8Array(1);
    const ids = new Int32Array(1);

    x[0] = 0;
    y[0] = 0;
    category[0] = 0;
    ids[0] = 100;

    vi.mocked(projectionsV2Api.getMetadata).mockResolvedValue({
      data: {
        id: 'projection-1',
        collection_id: collectionId,
        status: 'completed',
        reducer: 'umap',
        dimensionality: 2,
        created_at: new Date().toISOString(),
        meta: {
          color_by: 'document_id',
          degraded: true,
        },
      },
    } as any);

    vi.mocked(projectionsV2Api.getArtifact).mockImplementation(
      async (_collection, _projection, artifactName) => {
        if (artifactName === 'x') {
          return { data: x.buffer } as any;
        }
        if (artifactName === 'y') {
          return { data: y.buffer } as any;
        }
        if (artifactName === 'cat') {
          return { data: category.buffer } as any;
        }
        if (artifactName === 'ids') {
          return { data: ids.buffer } as any;
        }
        throw new Error(`Unexpected artifact request: ${artifactName}`);
      }
    );

    vi.mocked(projectionsV2Api.select).mockResolvedValue({
      data: {
        projection_id: 'projection-1',
        items: [
          {
            selected_id: 100,
            index: 0,
            document_id: 'doc-1',
            chunk_id: 11,
            chunk_index: 5,
            content_preview: 'Chunk content preview',
          },
        ],
        missing_ids: [],
        degraded: true,
      },
    } as any);

    const user = userEvent.setup();

    render(<EmbeddingVisualizationTab collectionId={collectionId} />);

    const viewButton = await screen.findByRole('button', { name: /view/i });
    await user.click(viewButton);

    await waitFor(() => {
      expect(lastEmbeddingViewProps).not.toBeNull();
    });

    await act(async () => {
      lastEmbeddingViewProps?.onSelection?.([0]);
    });

    await waitFor(() => {
      expect(
        screen.getByText(
          /This projection is marked as degraded; selection results may be incomplete\. Consider recomputing\./
        )
      ).toBeInTheDocument();
    });
  });

  it('maps selection indices to ids and renders selection items with actions', async () => {
    const x = new Float32Array(3);
    const y = new Float32Array(3);
    const category = new Uint8Array(3);
    const ids = new Int32Array(3);

    for (let i = 0; i < 3; i += 1) {
      x[i] = i;
      y[i] = 0;
      category[i] = 0;
      ids[i] = 100 + i;
    }

    vi.mocked(projectionsV2Api.getMetadata).mockResolvedValue({
      data: {
        id: 'projection-1',
        collection_id: collectionId,
        status: 'completed',
        reducer: 'umap',
        dimensionality: 2,
        created_at: new Date().toISOString(),
        meta: {
          color_by: 'document_id',
          degraded: false,
        },
      },
    } as any);

    vi.mocked(projectionsV2Api.getArtifact).mockImplementation(
      async (_collection, _projection, artifactName) => {
        if (artifactName === 'x') {
          return { data: x.buffer } as any;
        }
        if (artifactName === 'y') {
          return { data: y.buffer } as any;
        }
        if (artifactName === 'cat') {
          return { data: category.buffer } as any;
        }
        if (artifactName === 'ids') {
          return { data: ids.buffer } as any;
        }
        throw new Error(`Unexpected artifact request: ${artifactName}`);
      }
    );

    vi.mocked(projectionsV2Api.select).mockResolvedValue({
      data: {
        projection_id: 'projection-1',
        items: [
          {
            selected_id: 100,
            index: 0,
            document_id: 'doc-1',
            chunk_id: 11,
            chunk_index: 5,
            content_preview: 'Chunk content preview',
          },
        ],
        missing_ids: [],
        degraded: false,
      },
    } as any);

    vi.mocked(searchV2Api.search).mockResolvedValue({
      data: {
        query: 'Chunk content preview',
        results: [],
        total_results: 0,
        collections_searched: [],
        search_type: 'semantic',
        reranking_used: false,
        search_time_ms: 0,
        total_time_ms: 0,
        partial_failure: false,
        api_version: 'v2',
      },
    } as any);

    const user = userEvent.setup();

    render(<EmbeddingVisualizationTab collectionId={collectionId} />);

    const viewButton = await screen.findByRole('button', { name: /view/i });
    await user.click(viewButton);

    await waitFor(() => {
      expect(lastEmbeddingViewProps).not.toBeNull();
    });

    await act(async () => {
      lastEmbeddingViewProps?.onSelection?.([0]);
    });

    await waitFor(() => {
      expect(projectionsV2Api.select).toHaveBeenCalledWith(collectionId, 'projection-1', [100]);
      expect(screen.getByText('Selection')).toBeInTheDocument();
    });

    expect(screen.getByText(/Indices:/)).toHaveTextContent('Indices: 1');
    expect(screen.getByText(/Point #1/)).toBeInTheDocument();
    expect(screen.getByText(/ID 100/)).toBeInTheDocument();
    expect(screen.getByText(/Document doc-1/)).toBeInTheDocument();
    expect(screen.getByText(/Chunk #5/)).toBeInTheDocument();
    expect(screen.getByText('Chunk content preview')).toBeInTheDocument();

    const openButtons = screen.getAllByRole('button', { name: /^open$/i });
    await user.click(openButtons[0]);

    expect(setShowDocumentViewerMock).toHaveBeenCalledWith({
      collectionId,
      docId: 'doc-1',
      chunkId: '11',
    });

    const findSimilarButton = screen.getByRole('button', { name: /find similar/i });
    await user.click(findSimilarButton);

    await waitFor(() => {
      expect(searchV2Api.search).toHaveBeenCalledWith({
        query: 'Chunk content preview',
        collection_uuids: [collectionId],
        k: 10,
        search_type: 'semantic',
      });
    });
  });

  it('shows a helpful message when selected points cannot be mapped to documents', async () => {
    const x = new Float32Array(2);
    const y = new Float32Array(2);
    const category = new Uint8Array(2);
    const ids = new Int32Array(2);

    for (let i = 0; i < 2; i += 1) {
      x[i] = i;
      y[i] = 0;
      category[i] = 0;
      ids[i] = 200 + i;
    }

    vi.mocked(projectionsV2Api.getMetadata).mockResolvedValue({
      data: {
        id: 'projection-1',
        collection_id: collectionId,
        status: 'completed',
        reducer: 'umap',
        dimensionality: 2,
        created_at: new Date().toISOString(),
        meta: {
          color_by: 'document_id',
          degraded: false,
        },
      },
    } as any);

    vi.mocked(projectionsV2Api.getArtifact).mockImplementation(
      async (_collection, _projection, artifactName) => {
        if (artifactName === 'x') {
          return { data: x.buffer } as any;
        }
        if (artifactName === 'y') {
          return { data: y.buffer } as any;
        }
        if (artifactName === 'cat') {
          return { data: category.buffer } as any;
        }
        if (artifactName === 'ids') {
          return { data: ids.buffer } as any;
        }
        throw new Error(`Unexpected artifact request: ${artifactName}`);
      }
    );

    vi.mocked(projectionsV2Api.select).mockResolvedValue({
      data: {
        projection_id: 'projection-1',
        items: [],
        missing_ids: [200],
        degraded: false,
      },
    } as any);

    const user = userEvent.setup();

    render(<EmbeddingVisualizationTab collectionId={collectionId} />);

    const viewButton = await screen.findByRole('button', { name: /view/i });
    await user.click(viewButton);

    await waitFor(() => {
      expect(lastEmbeddingViewProps).not.toBeNull();
    });

    await act(async () => {
      lastEmbeddingViewProps?.onSelection?.([0]);
    });

    await waitFor(() => {
      expect(projectionsV2Api.select).toHaveBeenCalledWith(collectionId, 'projection-1', [200]);
      expect(screen.getByText('Selection')).toBeInTheDocument();
    });

    expect(screen.getByText(/Indices:/)).toHaveTextContent('Indices: 1');
    expect(
      screen.getByText(
        /Selected points could not be mapped to documents\. Try recomputing the projection or refining your selection\./
      )
    ).toBeInTheDocument();
    expect(screen.getByText(/1 point\(s\) could not be resolved\./)).toBeInTheDocument();
  });
});
