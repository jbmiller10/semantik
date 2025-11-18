import { describe, it, expect, vi, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import { act } from '@testing-library/react';
import EmbeddingVisualizationTab from '../EmbeddingVisualizationTab';
import type { ProjectionMetadata } from '../../types/projection';
import type { EmbeddingViewProps } from 'embedding-atlas/react';
import type { UseMutationResult } from '@tanstack/react-query';
import type { StartProjectionRequest, StartProjectionResponse } from '../../types/projection';

type UseOperationProgressOptions = {
  onComplete?: () => void;
  onError?: (error: string) => void;
  showToasts?: boolean;
};

let lastEmbeddingViewProps: EmbeddingViewProps | null = null;
const setShowDocumentViewerMock = vi.fn();
const addToastMock = vi.fn();

let lastOperationId: string | null = null;
let lastOperationOptions: UseOperationProgressOptions | null = null;

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
  useOperationProgress: vi.fn((operationId: string | null, options: UseOperationProgressOptions = {}) => {
    lastOperationId = operationId;
    lastOperationOptions = options;
    return { isConnected: false };
  }),
}));

vi.mock('../../stores/uiStore', () => ({
  useUIStore: vi.fn(() => ({
    setShowDocumentViewer: setShowDocumentViewerMock,
    addToast: addToastMock,
  })),
}));

vi.mock('embedding-atlas/react', () => ({
  EmbeddingView: (props: EmbeddingViewProps) => {
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

type ProjectionMetadataResponse = Awaited<ReturnType<typeof projectionsV2Api.getMetadata>>;
type ProjectionArtifactResponse = Awaited<ReturnType<typeof projectionsV2Api.getArtifact>>;
type ProjectionSelectResponse = Awaited<ReturnType<typeof projectionsV2Api.select>>;
type SearchResponse = Awaited<ReturnType<typeof searchV2Api.search>>;

describe('EmbeddingVisualizationTab', () => {
  const collectionId = 'test-collection-id';

  beforeEach(() => {
    vi.clearAllMocks();
    setShowDocumentViewerMock.mockReset();
    addToastMock.mockReset();
    lastEmbeddingViewProps = null;
    lastOperationId = null;
    lastOperationOptions = null;

    const mockProjections: ProjectionMetadata[] = [
      {
        id: 'projection-1',
        collection_id: collectionId,
        status: 'completed',
        operation_id: 'op-existing',
        operation_status: 'completed',
        reducer: 'umap',
        dimensionality: 2,
        created_at: new Date().toISOString(),
        meta: {},
      },
    ];

    const projectionsResult: ReturnType<typeof useCollectionProjections> = {
      data: mockProjections,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useCollectionProjections>;

    const startMutation: UseMutationResult<StartProjectionResponse, unknown, StartProjectionRequest> = {
      mutateAsync: vi.fn(),
      isPending: false,
    } as unknown as UseMutationResult<StartProjectionResponse, unknown, StartProjectionRequest>;

    const deleteMutation: UseMutationResult<string, unknown, string> = {
      mutateAsync: vi.fn(),
      isPending: false,
    } as unknown as UseMutationResult<string, unknown, string>;

    vi.mocked(useCollectionProjections).mockReturnValue(projectionsResult);
    vi.mocked(useStartProjection).mockReturnValue(startMutation);
    vi.mocked(useDeleteProjection).mockReturnValue(deleteMutation);

    vi.mocked(useUIStore).mockReturnValue({
      setShowDocumentViewer: setShowDocumentViewerMock,
      addToast: addToastMock,
    });
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
    } as ProjectionMetadataResponse);

    vi.mocked(projectionsV2Api.getArtifact).mockImplementation(
      async (_collection, _projection, artifactName) => {
        if (artifactName === 'x') {
          return { data: x.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'y') {
          return { data: y.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'cat') {
          return { data: category.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'ids') {
          return { data: ids.buffer } as ProjectionArtifactResponse;
        }
        throw new Error(`Unexpected artifact request: ${artifactName}`);
      }
    );

    const user = userEvent.setup();

    render(
      <EmbeddingVisualizationTab
        collectionId={collectionId}
        collectionEmbeddingModel="test-model"
        collectionVectorCount={100}
        collectionUpdatedAt={new Date().toISOString()}
      />
    );

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
    } as ProjectionMetadataResponse);

    vi.mocked(projectionsV2Api.getArtifact).mockImplementation(
      async (_collection, _projection, artifactName) => {
        if (artifactName === 'x') {
          return { data: x.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'y') {
          return { data: y.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'cat') {
          return { data: category.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'ids') {
          return { data: ids.buffer } as ProjectionArtifactResponse;
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
    } as ProjectionMetadataResponse);

    const user = userEvent.setup();

    render(
      <EmbeddingVisualizationTab
        collectionId={collectionId}
        collectionEmbeddingModel="test-model"
        collectionVectorCount={100}
        collectionUpdatedAt={new Date().toISOString()}
      />
    );

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
    } as ProjectionMetadataResponse);

    vi.mocked(projectionsV2Api.getArtifact).mockImplementation(
      async (_collection, _projection, artifactName) => {
        if (artifactName === 'x') {
          return { data: x.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'y') {
          return { data: y.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'cat') {
          return { data: category.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'ids') {
          return { data: ids.buffer } as ProjectionArtifactResponse;
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
    } as ProjectionMetadataResponse);

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
    } as ProjectionMetadataResponse);

    const user = userEvent.setup();

    render(
      <EmbeddingVisualizationTab
        collectionId={collectionId}
        collectionEmbeddingModel="test-model"
        collectionVectorCount={100}
        collectionUpdatedAt={new Date().toISOString()}
      />
    );

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
    } as ProjectionMetadataResponse);

    vi.mocked(projectionsV2Api.getArtifact).mockImplementation(
      async (_collection, _projection, artifactName) => {
        if (artifactName === 'x') {
          return { data: x.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'y') {
          return { data: y.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'cat') {
          return { data: category.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'ids') {
          return { data: ids.buffer } as ProjectionArtifactResponse;
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
    } as ProjectionMetadataResponse);

    const user = userEvent.setup();

    render(
      <EmbeddingVisualizationTab
        collectionId={collectionId}
        collectionEmbeddingModel="test-model"
        collectionVectorCount={100}
        collectionUpdatedAt={new Date().toISOString()}
      />
    );

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

  it('shows a loading indicator while projection arrays are being fetched', async () => {
    const neverResolving: Promise<ProjectionMetadataResponse> = new Promise(() => {
      // Intentionally never resolve to keep the component in loading state
    });
    vi.mocked(projectionsV2Api.getMetadata).mockReturnValue(neverResolving);
    vi.mocked(projectionsV2Api.getArtifact).mockImplementation(
      async () => {
        throw new Error('getArtifact should not be called while metadata is unresolved');
      }
    );

    const user = userEvent.setup();

    render(
      <EmbeddingVisualizationTab
        collectionId={collectionId}
        collectionEmbeddingModel="test-model"
        collectionVectorCount={100}
        collectionUpdatedAt={new Date().toISOString()}
      />
    );

    const viewButton = await screen.findByRole('button', { name: /view/i });
    await user.click(viewButton);

    expect(
      await screen.findByText(/Loading projection data…/)
    ).toBeInTheDocument();
  });

  it('shows an error message when projection arrays are inconsistent', async () => {
    const x = new Float32Array(10);
    const y = new Float32Array(5);
    const category = new Uint8Array(10);
    const ids = new Int32Array(10);

    for (let i = 0; i < 10; i += 1) {
      x[i] = i;
      category[i] = 0;
      ids[i] = i + 1;
    }
    for (let i = 0; i < 5; i += 1) {
      y[i] = i;
    }

    vi.mocked(projectionsV2Api.getMetadata).mockResolvedValue({
      data: {
        id: 'projection-1',
        collection_id: collectionId,
        status: 'completed',
        reducer: 'umap',
        dimensionality: 2,
        created_at: new Date().toISOString(),
        meta: {},
      },
    } as ProjectionMetadataResponse);

    vi.mocked(projectionsV2Api.getArtifact).mockImplementation(
      async (_collection, _projection, artifactName) => {
        if (artifactName === 'x') {
          return { data: x.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'y') {
          return { data: y.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'cat') {
          return { data: category.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'ids') {
          return { data: ids.buffer } as ProjectionArtifactResponse;
        }
        throw new Error(`Unexpected artifact request: ${artifactName}`);
      }
    );

    const user = userEvent.setup();

    render(
      <EmbeddingVisualizationTab
        collectionId={collectionId}
        collectionEmbeddingModel="test-model"
        collectionVectorCount={100}
        collectionUpdatedAt={new Date().toISOString()}
      />
    );

    const viewButton = await screen.findByRole('button', { name: /view/i });
    await user.click(viewButton);

    expect(
      await screen.findByText(/Projection arrays have inconsistent lengths/)
    ).toBeInTheDocument();
  });

  it('shows sampling badge with N of M points when projection is sampled', async () => {
    const x = new Float32Array(50);
    const y = new Float32Array(50);
    const category = new Uint8Array(50);
    const ids = new Int32Array(50);

    for (let i = 0; i < 50; i += 1) {
      x[i] = i;
      y[i] = 0;
      category[i] = 0;
      ids[i] = i + 1;
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
          sampled: true,
          shown_count: 10,
          total_count: 50,
        },
      },
    } as ProjectionMetadataResponse);

    vi.mocked(projectionsV2Api.getArtifact).mockImplementation(
      async (_collection, _projection, artifactName) => {
        if (artifactName === 'x') {
          return { data: x.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'y') {
          return { data: y.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'cat') {
          return { data: category.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'ids') {
          return { data: ids.buffer } as ProjectionArtifactResponse;
        }
        throw new Error(`Unexpected artifact request: ${artifactName}`);
      }
    );

    const user = userEvent.setup();

    render(
      <EmbeddingVisualizationTab
        collectionId={collectionId}
        collectionEmbeddingModel="test-model"
        collectionVectorCount={100}
        collectionUpdatedAt={new Date().toISOString()}
      />
    );

    const viewButton = await screen.findByRole('button', { name: /view/i });
    await user.click(viewButton);

    const sampledBadge = await screen.findByText('Sampled');
    expect(sampledBadge).toBeInTheDocument();
    expect(sampledBadge).toHaveAttribute(
      'title',
      'Showing 10 of 50 points'
    );
  });

  it('hides sampling badge when projection is not sampled', async () => {
    const x = new Float32Array(30);
    const y = new Float32Array(30);
    const category = new Uint8Array(30);
    const ids = new Int32Array(30);

    for (let i = 0; i < 30; i += 1) {
      x[i] = i;
      y[i] = 0;
      category[i] = 0;
      ids[i] = i + 1;
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
          sampled: false,
        },
      },
    } as ProjectionMetadataResponse);

    vi.mocked(projectionsV2Api.getArtifact).mockImplementation(
      async (_collection, _projection, artifactName) => {
        if (artifactName === 'x') {
          return { data: x.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'y') {
          return { data: y.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'cat') {
          return { data: category.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'ids') {
          return { data: ids.buffer } as ProjectionArtifactResponse;
        }
        throw new Error(`Unexpected artifact request: ${artifactName}`);
      }
    );

    const user = userEvent.setup();

    render(
      <EmbeddingVisualizationTab
        collectionId={collectionId}
        collectionEmbeddingModel="test-model"
        collectionVectorCount={100}
        collectionUpdatedAt={new Date().toISOString()}
      />
    );

    const viewButton = await screen.findByRole('button', { name: /view/i });
    await user.click(viewButton);

    await waitFor(() => {
      expect(
        screen.queryByText('Sampled')
      ).not.toBeInTheDocument();
    });
  });

  it('validates recompute dialog inputs and shows error messages', async () => {
    const x = new Float32Array(10);
    const y = new Float32Array(10);
    const category = new Uint8Array(10);
    const ids = new Int32Array(10);

    for (let i = 0; i < 10; i += 1) {
      x[i] = i;
      y[i] = 0;
      category[i] = 0;
      ids[i] = i + 1;
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
        },
      },
    } as ProjectionMetadataResponse);

    vi.mocked(projectionsV2Api.getArtifact).mockImplementation(
      async (_collection, _projection, artifactName) => {
        if (artifactName === 'x') {
          return { data: x.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'y') {
          return { data: y.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'cat') {
          return { data: category.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'ids') {
          return { data: ids.buffer } as ProjectionArtifactResponse;
        }
        throw new Error(`Unexpected artifact request: ${artifactName}`);
      }
    );

    const user = userEvent.setup();

    render(
      <EmbeddingVisualizationTab
        collectionId={collectionId}
        collectionEmbeddingModel="test-model"
        collectionVectorCount={100}
        collectionUpdatedAt={new Date().toISOString()}
      />
    );

    const colorBySelect = screen.getAllByRole('combobox')[0];
    await user.selectOptions(colorBySelect, 'filetype');

    const viewButton = await screen.findByRole('button', { name: /view/i });
    await user.click(viewButton);

    const recomputeButton = await screen.findByRole('button', {
      name: /Recompute with File Type/i,
    });
    await user.click(recomputeButton);

    const sampleSizeInput = await screen.findByPlaceholderText(/Optional/i);
    await user.clear(sampleSizeInput);
    await user.type(sampleSizeInput, '0');

    const startButton = await screen.findByRole('button', { name: /^Start$/i });
    await user.click(startButton);

    expect(
      await screen.findByText(/Sample size must be a positive number\./)
    ).toBeInTheDocument();
  });

  it('starts recompute, shows progress banner, and closes dialog on success', async () => {
    const x = new Float32Array(5);
    const y = new Float32Array(5);
    const category = new Uint8Array(5);
    const ids = new Int32Array(5);

    for (let i = 0; i < 5; i += 1) {
      x[i] = i;
      y[i] = 0;
      category[i] = 0;
      ids[i] = i + 1;
    }

    const refetchMock = vi.fn();

    const projectionsOverride: ReturnType<typeof useCollectionProjections> = {
      data: [
        {
          id: 'projection-1',
          collection_id: collectionId,
          status: 'completed',
          operation_id: 'op-existing',
          operation_status: 'completed',
          reducer: 'umap',
          dimensionality: 2,
          created_at: new Date().toISOString(),
          meta: {},
        },
      ],
      isLoading: false,
      error: null,
      refetch: refetchMock,
    } as unknown as ReturnType<typeof useCollectionProjections>;
    vi.mocked(useCollectionProjections).mockReturnValue(projectionsOverride);

    const mutateAsyncMock = vi.fn(async () => ({
      id: 'projection-2',
      collection_id: collectionId,
      status: 'pending',
      operation_id: 'op-123',
      operation_status: 'processing',
      reducer: 'umap',
      dimensionality: 2,
      created_at: new Date().toISOString(),
      meta: {},
    }));

    const startOverride: UseMutationResult<StartProjectionResponse, unknown, StartProjectionRequest> = {
      mutateAsync: mutateAsyncMock,
      isPending: false,
    } as unknown as UseMutationResult<StartProjectionResponse, unknown, StartProjectionRequest>;
    vi.mocked(useStartProjection).mockReturnValue(startOverride);

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
        },
      },
    } as ProjectionMetadataResponse);

    vi.mocked(projectionsV2Api.getArtifact).mockImplementation(
      async (_collection, _projection, artifactName) => {
        if (artifactName === 'x') {
          return { data: x.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'y') {
          return { data: y.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'cat') {
          return { data: category.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'ids') {
          return { data: ids.buffer } as ProjectionArtifactResponse;
        }
        throw new Error(`Unexpected artifact request: ${artifactName}`);
      }
    );

    const user = userEvent.setup();

    render(
      <EmbeddingVisualizationTab
        collectionId={collectionId}
        collectionEmbeddingModel="test-model"
        collectionVectorCount={100}
        collectionUpdatedAt={new Date().toISOString()}
      />
    );

    const colorBySelect = screen.getAllByRole('combobox')[0];
    await user.selectOptions(colorBySelect, 'filetype');

    const viewButton = await screen.findByRole('button', { name: /view/i });
    await user.click(viewButton);

    const recomputeButton = await screen.findByRole('button', {
      name: /Recompute with File Type/i,
    });
    await user.click(recomputeButton);

    const sampleSizeInput = await screen.findByPlaceholderText(/Optional/i);
    await user.clear(sampleSizeInput);
    await user.type(sampleSizeInput, '1000');

    const startButton = await screen.findByRole('button', { name: /^Start$/i });
    await user.click(startButton);

    await waitFor(() => {
      expect(mutateAsyncMock).toHaveBeenCalledTimes(1);
    });

    expect(refetchMock).toHaveBeenCalled();

    expect(
      screen.getByText(/Projection recompute in progress…/)
    ).toBeInTheDocument();
    expect(screen.getByText(/Operation ID: op-123/)).toBeInTheDocument();
    expect(screen.getByText(/Last known status: processing/)).toBeInTheDocument();

    expect(lastOperationId).toBe('op-123');
    expect(lastOperationOptions).toMatchObject({
      showToasts: false,
    });

    // Simulate websocket completion callback
    await act(async () => {
      lastOperationOptions?.onComplete?.();
    });

    await waitFor(() => {
      expect(
        screen.queryByText(/Projection recompute in progress…/)
      ).not.toBeInTheDocument();
    });
  });

  it('resets recompute dialog fields when reopened', async () => {
    const x = new Float32Array(5);
    const y = new Float32Array(5);
    const category = new Uint8Array(5);
    const ids = new Int32Array(5);

    for (let i = 0; i < 5; i += 1) {
      x[i] = i;
      y[i] = 0;
      category[i] = 0;
      ids[i] = i + 1;
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
        },
      },
    } as ProjectionMetadataResponse);

    vi.mocked(projectionsV2Api.getArtifact).mockImplementation(
      async (_collection, _projection, artifactName) => {
        if (artifactName === 'x') {
          return { data: x.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'y') {
          return { data: y.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'cat') {
          return { data: category.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'ids') {
          return { data: ids.buffer } as ProjectionArtifactResponse;
        }
        throw new Error(`Unexpected artifact request: ${artifactName}`);
      }
    );

    const user = userEvent.setup();

    render(<EmbeddingVisualizationTab collectionId={collectionId} />);

    const colorBySelect = screen.getAllByRole('combobox')[0];
    await user.selectOptions(colorBySelect, 'filetype');

    const viewButton = await screen.findByRole('button', { name: /view/i });
    await user.click(viewButton);

    const openRecompute = async () => {
      const recomputeButton = await screen.findByRole('button', {
        name: /Recompute with File Type/i,
      });
      await user.click(recomputeButton);
    };

    await openRecompute();

    const sampleSizeInput = await screen.findByPlaceholderText(/Optional/i);
    await user.clear(sampleSizeInput);
    await user.type(sampleSizeInput, '500');

    const cancelButton = await screen.findByRole('button', { name: /Cancel/i });
    await user.click(cancelButton);

    await openRecompute();

    const reopenedSampleInput = await screen.findByPlaceholderText(/Optional/i);
    expect((reopenedSampleInput as HTMLInputElement).value).toBe('');
  });

  it('renders multi-selection state, applies stale-response guard, and toggles action button enabled states', async () => {
    const x = new Float32Array(3);
    const y = new Float32Array(3);
    const category = new Uint8Array(3);
    const ids = new Int32Array(3);

    for (let i = 0; i < 3; i += 1) {
      x[i] = i;
      y[i] = 0;
      category[i] = 0;
      ids[i] = 300 + i;
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
        },
      },
    } as ProjectionMetadataResponse);

    vi.mocked(projectionsV2Api.getArtifact).mockImplementation(
      async (_collection, _projection, artifactName) => {
        if (artifactName === 'x') {
          return { data: x.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'y') {
          return { data: y.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'cat') {
          return { data: category.buffer } as ProjectionArtifactResponse;
        }
        if (artifactName === 'ids') {
          return { data: ids.buffer } as ProjectionArtifactResponse;
        }
        throw new Error(`Unexpected artifact request: ${artifactName}`);
      }
    );

    const firstSelectionResponse = {
      data: {
        projection_id: 'projection-1',
        items: [
          {
            selected_id: 300,
            index: 0,
            document_id: 'doc-first',
            chunk_id: 1,
            chunk_index: 0,
            content_preview: 'First preview',
          },
          {
            selected_id: 301,
            index: 1,
            document_id: null,
            chunk_id: 2,
            chunk_index: 1,
            content_preview: null,
          },
        ],
        missing_ids: [],
        degraded: false,
      },
    };

    const secondSelectionResponse = {
      data: {
        projection_id: 'projection-1',
        items: [
          {
            selected_id: 302,
            index: 2,
            document_id: 'doc-second',
            chunk_id: 3,
            chunk_index: 2,
            content_preview: 'Second preview',
          },
        ],
        missing_ids: [],
        degraded: false,
      },
    };

    let resolveFirstSelection: ((value: ProjectionSelectResponse) => void) | null = null;
    const firstPromise: Promise<ProjectionSelectResponse> = new Promise((resolve) => {
      resolveFirstSelection = resolve;
    });

    vi.mocked(projectionsV2Api.select)
      .mockImplementationOnce(async () => firstPromise)
      .mockResolvedValueOnce(secondSelectionResponse as ProjectionSelectResponse);

    const user = userEvent.setup();

    render(<EmbeddingVisualizationTab collectionId={collectionId} />);

    const viewButton = await screen.findByRole('button', { name: /view/i });
    await user.click(viewButton);

    await waitFor(() => {
      expect(lastEmbeddingViewProps).not.toBeNull();
    });

    await act(async () => {
      lastEmbeddingViewProps?.onSelection?.([0, 1]);
    });

    await act(async () => {
      lastEmbeddingViewProps?.onSelection?.([2]);
    });

    await waitFor(() => {
      expect(screen.getByText('Selection')).toBeInTheDocument();
      expect(screen.getByText(/Indices: 1/)).toBeInTheDocument();
      expect(screen.getByText(/Point #3/)).toBeInTheDocument();
      expect(screen.getByText(/ID 302/)).toBeInTheDocument();
    });

    await act(async () => {
      resolveFirstSelection?.(firstSelectionResponse as ProjectionSelectResponse);
    });

    await waitFor(() => {
      expect(screen.getByText(/Point #3/)).toBeInTheDocument();
      expect(screen.getByText(/ID 302/)).toBeInTheDocument();
    });

    vi.mocked(projectionsV2Api.select).mockResolvedValue({
      data: {
        projection_id: 'projection-1',
        items: firstSelectionResponse.data.items,
        missing_ids: [],
        degraded: false,
      },
    } as ProjectionSelectResponse);

    await act(async () => {
      lastEmbeddingViewProps?.onSelection?.([0, 1]);
    });

    await waitFor(() => {
      expect(screen.getByText(/Selection/)).toBeInTheDocument();
      expect(screen.getByText(/Indices: 2/)).toBeInTheDocument();
      expect(
        screen.getByText(/Actions apply to the first selected point/)
      ).toBeInTheDocument();
    });

    const openButtons = screen.getAllByRole('button', { name: /^open$/i });
    const similarButtons = screen.getAllByRole('button', { name: /find similar/i });

    expect(openButtons[0]).toBeEnabled();
    expect(openButtons[1]).toBeDisabled();
    expect(similarButtons[0]).toBeEnabled();
    expect(similarButtons[1]).toBeDisabled();
  });
});
