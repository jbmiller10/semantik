import { describe, it, expect, vi, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import { act, within } from '@testing-library/react';
import type { ComponentProps } from 'react';
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
const collectionId = 'test-collection-id';
type EmbeddingVisualizationTabTestProps = Partial<ComponentProps<typeof EmbeddingVisualizationTab>>;

type ProjectionMockOptions = {
  pointCount: number;
  idsStart?: number;
  categoryGenerator?: (index: number) => number;
  meta?: Record<string, unknown>;
};

function createProjectionArrays({ pointCount, idsStart = 0, categoryGenerator }: ProjectionMockOptions) {
  const x = new Float32Array(pointCount);
  const y = new Float32Array(pointCount);
  const category = new Uint8Array(pointCount);
  const ids = new Int32Array(pointCount);
  for (let index = 0; index < pointCount; index += 1) {
    x[index] = index;
    y[index] = 0;
    category[index] = categoryGenerator ? categoryGenerator(index) : 0;
    ids[index] = idsStart + index;
  }
  return { x, y, category, ids };
}

function setupProjectionApiMocks(options: ProjectionMockOptions) {
  const arrays = createProjectionArrays(options);
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
        ...options.meta,
      },
    },
  } as ProjectionMetadataResponse);

  vi.mocked(projectionsV2Api.getArtifact).mockImplementation(
    async (_collection, _projection, artifactName) => {
      if (artifactName === 'x') {
        return { data: arrays.x.buffer } as ProjectionArtifactResponse;
      }
      if (artifactName === 'y') {
        return { data: arrays.y.buffer } as ProjectionArtifactResponse;
      }
      if (artifactName === 'cat') {
        return { data: arrays.category.buffer } as ProjectionArtifactResponse;
      }
      if (artifactName === 'ids') {
        return { data: arrays.ids.buffer } as ProjectionArtifactResponse;
      }
      throw new Error(`Unexpected artifact request: ${artifactName}`);
    }
  );

  return arrays;
}

function renderTab(overrides: EmbeddingVisualizationTabTestProps = {}) {
  const user = userEvent.setup();
  render(
    <EmbeddingVisualizationTab
      collectionId={collectionId}
      {...overrides}
    />
  );
  return { user };
}

async function renderTabAndLoadProjection(overrides: EmbeddingVisualizationTabTestProps = {}) {
  const { user } = renderTab(overrides);
  const viewButton = await screen.findByRole('button', { name: /view/i });
  await user.click(viewButton);
  await waitFor(() => {
    expect(lastEmbeddingViewProps).not.toBeNull();
  });
  return { user };
}

describe('EmbeddingVisualizationTab', () => {

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

  it('starts a projection with selected options and tracks pending operation state', async () => {
    const mutateAsyncMock = vi.fn(async () => ({
      id: 'projection-2',
      collection_id: collectionId,
      status: 'pending',
      operation_id: 'op-555',
      operation_status: 'processing',
      reducer: 'tsne',
      dimensionality: 2,
      created_at: new Date().toISOString(),
      meta: {},
      idempotent_reuse: false,
    }));

    const refetchMock = vi.fn();
    vi.mocked(useCollectionProjections).mockReturnValue({
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
    } as unknown as ReturnType<typeof useCollectionProjections>);

    vi.mocked(useStartProjection).mockReturnValue({
      mutateAsync: mutateAsyncMock,
      isPending: false,
    } as unknown as UseMutationResult<StartProjectionResponse, unknown, StartProjectionRequest>);

    const { user } = renderTab();

    await user.click(screen.getByRole('button', { name: /t-SNE/i }));
    const colorSelect = screen.getAllByRole('combobox')[0];
    await user.selectOptions(colorSelect, 'filetype');

    await user.click(screen.getByRole('button', { name: /Start Projection/i }));

    await waitFor(() => {
      expect(mutateAsyncMock).toHaveBeenCalledWith({ reducer: 'tsne', color_by: 'filetype' });
    });

    expect(await screen.findByText(/Projection recompute in progress/i)).toBeInTheDocument();
    expect(screen.getByText(/op-555/)).toBeInTheDocument();
    expect(lastOperationId).toBe('op-555');

    await act(async () => {
      lastOperationOptions?.onComplete?.();
    });

    expect(refetchMock).toHaveBeenCalled();
    await waitFor(() => {
      expect(screen.queryByText(/Projection recompute in progress/i)).not.toBeInTheDocument();
    });
  });

  it('shows fallback progress banner when websocket updates are unavailable', () => {
    const now = Date.now();
    vi.mocked(useCollectionProjections).mockReturnValue({
      data: [
        {
          id: 'projection-running',
          collection_id: collectionId,
          status: 'running',
          operation_id: 'op-fallback',
          operation_status: 'processing',
          reducer: 'umap',
          dimensionality: 2,
          created_at: new Date(now).toISOString(),
          meta: {},
        },
      ],
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useCollectionProjections>);

    renderTab();

    expect(screen.getByText(/Projection in progress/i)).toBeInTheDocument();
    expect(screen.getByText(/op-fallback/)).toBeInTheDocument();
    expect(
      screen.getByText(/Status from last refresh \(WebSocket unavailable\)/i)
    ).toBeInTheDocument();
  });

  it('deletes the active projection and resets the preview state', async () => {
    setupProjectionApiMocks({ pointCount: 4 });

    const deleteMock = vi.fn().mockResolvedValue('deleted');
    vi.mocked(useDeleteProjection).mockReturnValue({
      mutateAsync: deleteMock,
      isPending: false,
    } as unknown as UseMutationResult<string, unknown, string>);

    const { user } = await renderTabAndLoadProjection();

    expect(await screen.findByRole('button', { name: 'Auto' })).toBeInTheDocument();

    const deleteButton = screen.getByRole('button', { name: /Delete/i });
    await user.click(deleteButton);

    await waitFor(() => {
      expect(deleteMock).toHaveBeenCalledWith('projection-1');
    });

    await waitFor(() => {
      expect(screen.getByText(/Select a completed projection/i)).toBeInTheDocument();
    });
  });

  it('switches render modes manually and resolves query selections near the cursor', async () => {
    // Point count must exceed the DEFAULT_DENSITY_THRESHOLD (200,000 from preferences)
    // to trigger density mode by default
    setupProjectionApiMocks({ pointCount: 250_000 });

    const { user } = await renderTabAndLoadProjection();

    await waitFor(() => {
      expect(lastEmbeddingViewProps?.config?.mode).toBe('density');
    });

    await user.click(screen.getByRole('button', { name: 'Points' }));
    await waitFor(() => {
      expect(lastEmbeddingViewProps?.config?.mode).toBe('points');
    });

    await user.click(screen.getByRole('button', { name: 'Density' }));
    await waitFor(() => {
      expect(lastEmbeddingViewProps?.config?.mode).toBe('density');
    });

    const resolvedPoint = await lastEmbeddingViewProps?.querySelection?.(10, 0, 1);
    expect(resolvedPoint?.identifier).toBe(10);

    const missingPoint = await lastEmbeddingViewProps?.querySelection?.(999999, 999999, 1);
    expect(missingPoint).toBeNull();
  });

  it('surfaces selection errors when the select endpoint fails', async () => {
    setupProjectionApiMocks({ pointCount: 2 });

    vi.mocked(projectionsV2Api.select).mockRejectedValue(new Error('select failed'));

    await renderTabAndLoadProjection();

    await act(async () => {
      lastEmbeddingViewProps?.onSelection?.([0]);
    });

    expect(await screen.findByText('select failed')).toBeInTheDocument();
  });

  it('renders similar search results and allows closing the results panel', async () => {
    setupProjectionApiMocks({ pointCount: 3 });

    vi.mocked(projectionsV2Api.select).mockResolvedValue({
      data: {
        projection_id: 'projection-1',
        items: [
          {
            selected_id: 0,
            index: 0,
            document_id: 'doc-1',
            chunk_id: 10,
            chunk_index: 1,
            content_preview: 'Preview for similar search',
          },
        ],
        missing_ids: [],
        degraded: false,
      },
    } as ProjectionSelectResponse);

    vi.mocked(searchV2Api.search).mockResolvedValue({
      data: {
        results: [
          {
            document_id: 'doc-2',
            chunk_id: 'chunk-2',
            chunk_index: 3,
            file_name: 'report.pdf',
            text: 'Result text',
            score: 0.42,
          },
        ],
      },
    } as ProjectionMetadataResponse);

    const { user } = await renderTabAndLoadProjection();

    await act(async () => {
      lastEmbeddingViewProps?.onSelection?.([0]);
    });

    const findSimilarButton = await screen.findByRole('button', { name: /Find Similar/i });
    await user.click(findSimilarButton);

    const resultTitle = await screen.findByText('report.pdf');
    const resultCard = resultTitle.closest('li');
    expect(resultCard).not.toBeNull();
    if (!resultCard) {
      throw new Error('Result card not found');
    }
    const openResultButton = within(resultCard).getByRole('button', { name: /^Open$/i });
    await user.click(openResultButton);
    expect(setShowDocumentViewerMock).toHaveBeenCalledWith({
      collectionId,
      docId: 'doc-2',
      chunkId: 'chunk-2',
    });

    await user.click(screen.getByTitle('Close similar results'));
    expect(screen.queryByText(/Similar Chunks/i)).not.toBeInTheDocument();
  });

  it('shows an error toast when similar search fails', async () => {
    setupProjectionApiMocks({ pointCount: 2 });

    vi.mocked(projectionsV2Api.select).mockResolvedValue({
      data: {
        projection_id: 'projection-1',
        items: [
          {
            selected_id: 1,
            index: 0,
            document_id: 'doc-1',
            chunk_id: 7,
            chunk_index: 0,
            content_preview: 'Chunk preview content',
          },
        ],
        missing_ids: [],
        degraded: false,
      },
    } as ProjectionSelectResponse);

    vi.mocked(searchV2Api.search).mockRejectedValue(new Error('Search exploded'));

    const { user } = await renderTabAndLoadProjection();

    await act(async () => {
      lastEmbeddingViewProps?.onSelection?.([0]);
    });

    const findSimilarButton = await screen.findByRole('button', { name: /Find Similar/i });
    await user.click(findSimilarButton);

    expect(await screen.findByText('Search exploded')).toBeInTheDocument();
    expect(addToastMock).toHaveBeenCalledWith({
      type: 'error',
      message: 'Failed to find similar chunks: Search exploded',
    });
  });

  it('renders status badges and progress bars for each projection state', () => {
    const statuses = [
      { id: 'completed', status: 'completed' },
      { id: 'running', status: 'running' },
      { id: 'processing', status: 'pending', operation_status: 'processing' },
      { id: 'failed', status: 'failed' },
      { id: 'cancelled', status: 'cancelled' },
      { id: 'pending', status: 'pending' },
      { id: 'mystery', status: 'weird-status' },
    ];

    vi.mocked(useCollectionProjections).mockReturnValue({
      data: statuses.map((entry, index) => ({
        ...entry,
        collection_id: collectionId,
        reducer: 'umap',
        dimensionality: 2,
        created_at: new Date(Date.now() - index * 1_000).toISOString(),
        meta: {},
      })),
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    } as unknown as ReturnType<typeof useCollectionProjections>);

    renderTab();

    expect(screen.getByText('Completed')).toBeInTheDocument();
    expect(screen.getAllByText('Processing').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Failed')).toBeInTheDocument();
    expect(screen.getByText('Cancelled')).toBeInTheDocument();
    expect(screen.getAllByText('Pending').length).toBeGreaterThan(0);

    const rows = screen.getAllByRole('row').slice(1);
    const widths = rows
      .map((row) => row.querySelector('td:nth-child(3) .bg-purple-500') as HTMLElement | null)
      .filter((bar): bar is HTMLElement => Boolean(bar))
      .map((bar) => bar.style.width);

    expect(widths).toContain('100%');
    expect(widths).toContain('60%');
    expect(widths).toContain('10%');
    expect(widths).toContain('0%');
  });
});
