import { Suspense, useCallback, useEffect, useMemo, useRef, useState, lazy } from 'react';
import type { DataPoint } from 'embedding-atlas/react';
import { AlertCircle, Loader2, Play, Trash2, Eye, X } from 'lucide-react';
import { EmbeddingTooltipAdapter } from './EmbeddingTooltip';
import { ensureEmbeddingAtlasWebgpuCompatibility } from '../utils/embeddingAtlasWebgpuPatch';
import {
  useCollectionProjections,
  useDeleteProjection,
  useStartProjection,
} from '../hooks/useProjections';
import { useProjectionTooltip } from '../hooks/useProjectionTooltip';
import { useOperationProgress } from '../hooks/useOperationProgress';
import { projectionsV2Api } from '../services/api/v2/projections';
import { searchV2Api } from '../services/api/v2/collections';
import { useUIStore } from '../stores/uiStore';
import type {
  ProjectionArtifactName,
  ProjectionLegendItem,
  ProjectionMetadata,
  ProjectionReducer,
  ProjectionSelectionItem,
  StartProjectionRequest,
} from '../types/projection';
import type { SearchResult } from '../services/api/v2/types';
import { getErrorMessage } from '../utils/errorUtils';
import { createCategoryLabels, DEFAULT_CATEGORY_LABEL_OPTIONS } from '../utils/clusterLabels';
import type { CategoryLabel } from '../utils/clusterLabels';
import { getProjectionPointIndex } from '../utils/projectionIndex';
import { computeProjectionMetadataHash, type ProjectionMetadataHashContext } from '../utils/projectionHash';
import { trackTelemetry } from '../lib/telemetry';

const EmbeddingView = lazy(() => import('embedding-atlas/react').then((mod) => ({ default: mod.EmbeddingView })));

ensureEmbeddingAtlasWebgpuCompatibility();

/**
 * EmbeddingVisualizationTab orchestrates the Projection tab for a collection.
 *
 * Responsibilities
 * - Manage the lifecycle of projection runs for a collection (start, list, delete, recompute).
 * - Load projection artifacts (x / y coordinates, category indices, ids) and metadata from the v2 projections API.
 * - Render the current projection via EmbeddingView and manage render mode, labels, and responsive sizing.
 * - Handle hover and selection interactions and connect them to metadata tooltips, similar search, and document viewing.
 *
 * State and hooks overview
 * - projections: list of ProjectionMetadata from useCollectionProjections; this is the source of truth for runs.
 * - activeProjection: currently loaded projection arrays + ids + status used to drive EmbeddingView.
 * - activeProjectionMeta: summary of ProjectionMetadata.meta (color_by, legend, sampling counts, degraded flag).
 * - renderModeByProjection / currentRenderMode / effectiveRenderMode:
 *   per-projection render mode preference; effectiveRenderMode is the concrete
 *   'points' | 'density' mode sent to EmbeddingView.
 * - labelsEnabled / clusterLabels: toggle and cached cluster centroid labels derived from backend legend + arrays.
 * - selectionState: indices selected in EmbeddingView, resolved ProjectionSelectionItem[] from /select, and loading/error/missing.
 * - tooltipState / activeTooltip: hover state from useProjectionTooltip plus
 *   the latest DataPoint from EmbeddingView.
 * - similarSearchState: status and results for "Find similar" queries based on
 *   the current selection, via searchV2Api.
 * - recomputeDialogOpen / recomputeReducer / recomputeParams / recomputeError:
 *   UI and form state for starting a new ProjectionRun with explicit reducer configuration.
 * - pendingOperationId / currentOperationStatus: track the server-side Operation
 *   backing a projection so progress banners stay in sync.
 * - viewSize / pixelRatio: responsive canvas dimensions and device pixel ratio for EmbeddingView.
 * - setShowDocumentViewer (from uiStore): side-effect to open the document
 *   viewer for a selected chunk or similar-search result.
 *
 * Supporting hooks and services
 * - useCollectionProjections: polls the v2 projections list for a collection
 *   and drives the table of runs and their status.
 * - useStartProjection / useDeleteProjection: wrap projectionsV2Api.start /
 *   delete with optimistic cache updates and toasts.
 * - useProjectionTooltip: given collectionId, projectionId and ids, resolves
 *   hover tooltips via projectionsV2Api.select with LRU caching and debouncing.
 * - useOperationProgress: tracks the Operation backing a projection (via
 *   pendingOperationId) and calls onComplete/onError, which we use to refresh
 *   the projections list and surface recompute errors.
 * - searchV2Api.search: used to implement "Find similar" from the selection
 *   panel.
 *
 * Lifecycle (happy path)
 * - Start projection: user clicks "Start projection" → useStartProjection →
 *   projectionsV2Api.start, which creates a new ProjectionRun and Operation.
 *   The list from useCollectionProjections reflects status over time.
 * - View projection: when a run is completed, "View" calls handleViewProjection,
 *   which fetches ProjectionMetadata.meta and the x/y/cat/ids artifacts,
 *   validates lengths, then populates activeProjection and activeProjectionMeta.
 * - Hover tooltip: EmbeddingView calls onTooltip with a DataPoint; the tab
 *   forwards this to useProjectionTooltip, which maps the point index through
 *   ids and calls projectionsV2Api.select for a single id to obtain tooltip
 *   metadata. The resulting tooltipState is consumed by EmbeddingTooltipAdapter
 *   to render a React tooltip.
 * - Selection panel: EmbeddingView calls onSelection with indices (or
 *   DataPoint-like objects); the tab normalizes these to unique indices, maps
 *   them through ids, and calls projectionsV2Api.select. The
 *   ProjectionSelectionResponse items, missing_ids, and degraded flag populate
 *   selectionState and inform degraded UI.
 * - Recompute: "Recompute" opens a dialog whose form is encoded into
 *   StartProjectionRequest.config. Submitting the dialog calls
 *   startProjectionWithPayload, which always creates a new ProjectionRun rather
 *   than mutating existing runs.
 *
 * Backend contracts
 * - Projection arrays are streamed as typed buffers and must have consistent length:
 *   x: Float32Array, y: Float32Array, category: Uint8Array, ids: Int32Array.
 * - ProjectionMetadata.meta is treated as the canonical summary for
 *   visualization: color_by, legend, sampling info (sampled, shown_count,
 *   total_count), and degraded.
 * - ProjectionSelectionResponse from /select returns items, missing_ids, and a
 *   degraded boolean; when degraded is true we mirror that onto
 *   activeProjectionMeta.degraded so banners and selection panels reflect the
 *   current backend state.
 *
 * EmbeddingView contract
 * - data prop receives typed arrays ({ x, y, category }) that EmbeddingView
 *   consumes directly; lengths are validated here.
 * - config.mode receives effectiveRenderMode ('points' | 'density'), derived
 *   from user preference and point count.
 * - labels is an optional array of centroid labels
 *   ({ x, y, text, level?, priority? }) derived from the legend via
 *   createCategoryLabels and only computed when enabled to avoid unnecessary
 *   work.
 * - onTooltip and onSelection are the only interaction channels: the tab
 *   translates these callbacks into tooltip metadata, selectionState updates,
 *   similar-search queries, and document viewer navigation.
 */

type EmbeddingLabel = CategoryLabel;

interface ProjectionDataState {
  projectionId: string;
  pointCount: number;
  arrays?: {
    x: Float32Array;
    y: Float32Array;
    category: Uint8Array;
  };
  ids?: Int32Array;
  status: 'idle' | 'loading' | 'loaded' | 'error';
  error?: string;
}

interface EmbeddingVisualizationTabProps {
  collectionId: string;
  collectionEmbeddingModel?: string;
  collectionVectorCount?: number;
  collectionUpdatedAt?: string | null;
}

type RecomputeParamsState = {
  n_neighbors: string;
  min_dist: string;
  metric: string;
  sample_n: string;
  perplexity: string;
  learning_rate: string;
  n_iter: string;
  init: 'pca' | 'random';
};

const COLOR_BY_OPTIONS: Array<{ value: string; label: string }> = [
  { value: 'document_id', label: 'Document' },
  { value: 'source_dir', label: 'Source Folder' },
  { value: 'filetype', label: 'File Type' },
  { value: 'age_bucket', label: 'Age Bucket' },
];

const METRIC_OPTIONS = ['cosine', 'euclidean', 'manhattan'];
const TSNE_METRIC_OPTIONS = ['euclidean'];
const TSNE_INIT_OPTIONS: Array<'pca' | 'random'> = ['pca', 'random'];
const SAMPLE_LIMIT_CAP = 200_000;

type RenderMode = 'auto' | 'points' | 'density';

// Above this many points, "auto" mode prefers density rendering for performance.
const DENSITY_THRESHOLD = 20_000;
const RENDER_MODE_OPTIONS: RenderMode[] = ['auto', 'points', 'density'];

const REDUCER_OPTIONS: Array<{
  value: ProjectionReducer;
  label: string;
  description: string;
}> = [
    {
      value: 'umap',
      label: 'UMAP',
      description: 'Uniform Manifold Approximation and Projection (fast, good global + local structure).',
    },
    {
      value: 'tsne',
      label: 't-SNE',
      description: 't-distributed Stochastic Neighbor Embedding (great local detail, slower).',
    },
    {
      value: 'pca',
      label: 'PCA',
      description: 'Principal Component Analysis (linear baseline, very fast).',
    },
  ];

function statusBadge(status: ProjectionMetadata['status'] | string) {
  const base = 'px-2 py-1 rounded-full text-xs font-medium';
  switch (status) {
    case 'completed':
      return <span className={`${base} bg-green-100 text-green-800`}>Completed</span>;
    case 'running':
    case 'processing':
      return <span className={`${base} bg-blue-100 text-blue-800`}>Processing</span>;
    case 'failed':
      return <span className={`${base} bg-red-100 text-red-800`}>Failed</span>;
    case 'cancelled':
      return <span className={`${base} bg-gray-100 text-gray-600`}>Cancelled</span>;
    case 'pending':
      return <span className={`${base} bg-amber-100 text-amber-800`}>Pending</span>;
    default:
      return <span className={`${base} bg-amber-100 text-amber-800`}>Pending</span>;
  }
}

function projectionProgress(status: ProjectionMetadata['status'] | string) {
  if (status === 'completed') return 100;
  if (status === 'running' || status === 'processing') return 60;
  if (status === 'pending') return 10;
  return 0;
}

export function EmbeddingVisualizationTab({
  collectionId,
  collectionEmbeddingModel,
  collectionVectorCount,
  collectionUpdatedAt,
}: EmbeddingVisualizationTabProps) {
  // Projection run configuration and list state
  const [selectedReducer, setSelectedReducer] = useState<ProjectionReducer>('umap');
  const [selectedColorBy, setSelectedColorBy] = useState<string>('document_id');
  const { data: projections = [], isLoading, refetch } = useCollectionProjections(collectionId);
  const startProjection = useStartProjection(collectionId);
  const deleteProjection = useDeleteProjection(collectionId);

  // Active projection arrays + metadata used to drive EmbeddingView
  const [activeProjection, setActiveProjection] = useState<ProjectionDataState>({
    projectionId: '',
    pointCount: 0,
    status: 'idle',
  });
  const [renderModeByProjection, setRenderModeByProjection] = useState<Record<string, RenderMode>>({});
  const [activeProjectionMeta, setActiveProjectionMeta] = useState<{
    color_by?: string;
    legend?: ProjectionLegendItem[];
    sampled?: boolean;
    shown_count?: number;
    total_count?: number;
    degraded?: boolean;
  } | null>(null);
  const [labelsEnabled, setLabelsEnabled] = useState(false);

  // Selection, tooltip, and linked-document / similar-search state
  const [selectionState, setSelectionState] = useState<{
    indices: number[];
    items: ProjectionSelectionItem[];
    missing: number[];
    loading: boolean;
    error?: string;
  }>({ indices: [], items: [], missing: [], loading: false });
  const selectionRequestId = useRef(0);
  const [similarSearchState, setSimilarSearchState] = useState<{
    loading: boolean;
    error: string | null;
    results: SearchResult[];
    visible: boolean;
  }>({ loading: false, error: null, results: [], visible: false });
  const [activeTooltip, setActiveTooltip] = useState<DataPoint | null>(null);

  // Recompute dialog and long-running Operation tracking
  const [recomputeDialogOpen, setRecomputeDialogOpen] = useState(false);
  const [pendingOperationId, setPendingOperationId] = useState<string | null>(null);
  const [recomputeReducer, setRecomputeReducer] = useState<ProjectionReducer>('umap');
  const [recomputeParams, setRecomputeParams] = useState<RecomputeParamsState>({
    n_neighbors: '15',
    min_dist: '0.1',
    metric: 'cosine',
    sample_n: '',
    perplexity: '30',
    learning_rate: '200',
    n_iter: '1000',
    init: 'pca',
  });
  const [recomputeError, setRecomputeError] = useState<string | undefined>(undefined);
  const [currentOperationStatus, setCurrentOperationStatus] = useState<string | null>(null);

  // View sizing and device pixel density for EmbeddingView
  const activeRequestId = useRef(0);
  const viewContainerRef = useRef<HTMLDivElement | null>(null);
  const [viewSize, setViewSize] = useState<{ width: number; height: number }>({ width: 960, height: 540 });
  const [pixelRatio, setPixelRatio] = useState<number>(typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1);

  // Global UI store: document viewer + toasts
  const { setShowDocumentViewer, addToast } = useUIStore();

  // Tooltip metadata is fetched via /select on-demand using point ids from the active projection
  const { tooltipState, handleTooltip, handleTooltipLeave, clearTooltipCache } = useProjectionTooltip(
    collectionId ?? null,
    activeProjection.projectionId || null,
    activeProjection.ids
  );

  const { isConnected: isOperationConnected } = useOperationProgress(pendingOperationId, {
    showToasts: false,
    onComplete: () => {
      setPendingOperationId(null);
      setCurrentOperationStatus(null);
      refetch();
    },
    onError: (errorMessage) => {
      setPendingOperationId(null);
      setCurrentOperationStatus(null);
      if (errorMessage) {
        setRecomputeError(errorMessage);
      }
    },
  });

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    const handlePixelRatio = () => setPixelRatio(window.devicePixelRatio || 1);
    window.addEventListener('resize', handlePixelRatio);
    return () => {
      window.removeEventListener('resize', handlePixelRatio);
    };
  }, []);

  useEffect(() => {
    if (!viewContainerRef.current || typeof window === 'undefined' || typeof ResizeObserver === 'undefined') {
      return;
    }
    const node = viewContainerRef.current;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      const width = Math.max(360, entry.contentRect.width);
      const height = Math.max(320, Math.round(width * 0.6));
      setViewSize({ width, height });
    });
    observer.observe(node);
    return () => {
      observer.disconnect();
    };
  }, [viewContainerRef]);

  const sortedProjections = useMemo(
    () =>
      [...projections].sort((a, b) => {
        const aDate = a.created_at ? new Date(a.created_at).getTime() : 0;
        const bDate = b.created_at ? new Date(b.created_at).getTime() : 0;
        return bDate - aDate;
      }),
    [projections]
  );

  // Find in-progress projection for fallback status display (when WebSocket not available)
  const inProgressProjection = useMemo(() => {
    if (pendingOperationId) return null; // Don't show fallback if we have active pending op
    return sortedProjections.find((p) => {
      const status = p.operation_status || p.status;
      return status === 'processing' || status === 'pending' || status === 'running';
    });
  }, [sortedProjections, pendingOperationId]);

  const startProjectionWithPayload = async (payload: StartProjectionRequest) => {
    let enrichedPayload: StartProjectionRequest = payload;

    // If caller has already provided a metadata_hash (e.g. recompute dialog),
    // respect it; otherwise, compute one from the current collection context.
    if (!enrichedPayload.metadata_hash && metadataHashContext) {
      try {
        const metadataHash = await computeProjectionMetadataHash(metadataHashContext, enrichedPayload);
        enrichedPayload = {
          ...enrichedPayload,
          metadata_hash: metadataHash,
        };
      } catch (error) {
        // Hash computation failure should not block the projection; log and continue without idempotency hint.
        console.warn('Failed to compute projection metadata hash', error);
      }
    }

    const response = await startProjection.mutateAsync(enrichedPayload);
    const shouldTrackOperation =
      response?.operation_id &&
      response.operation_status &&
      !response.idempotent_reuse &&
      ['pending', 'processing', 'running'].includes(response.operation_status);

    if (shouldTrackOperation) {
      // At this point response and its operation fields are known to be present.
      setPendingOperationId(response!.operation_id!);
      setCurrentOperationStatus(response!.operation_status ?? null);
    }
    refetch();
    setActiveProjectionMeta(null);
    setSelectionState({ indices: [], items: [], missing: [], loading: false });
    return response;
  };

  const handleStartProjection = async () => {
    await startProjectionWithPayload({ reducer: selectedReducer, color_by: selectedColorBy });
  };

  const handleDeleteProjection = async (projectionId: string) => {
    await deleteProjection.mutateAsync(projectionId);
    refetch();
    setActiveProjection((prev) =>
      prev.projectionId === projectionId
        ? { projectionId: '', pointCount: 0, status: 'idle' }
        : prev
    );
    setActiveProjectionMeta(null);
  };

  const handleViewProjection = async (projection: ProjectionMetadata) => {
    if (projection.status !== 'completed') return;

    const requestId = ++activeRequestId.current;
    setActiveProjection({ projectionId: projection.id, pointCount: 0, status: 'loading' });
    setActiveProjectionMeta(null);
    setSelectionState({ indices: [], items: [], missing: [], loading: false });
    clearTooltipCache();

    try {
      const metadataResponse = await projectionsV2Api.getMetadata(collectionId, projection.id);
      const metaPayload = (metadataResponse.data?.meta as Record<string, unknown> | null) ?? {};
      const legendPayload = Array.isArray(metaPayload?.legend)
        ? (metaPayload.legend as ProjectionLegendItem[])
        : [];
      const metaColorBy = typeof metaPayload?.color_by === 'string'
        ? (metaPayload.color_by as string)
        : typeof metaPayload?.colorBy === 'string'
          ? (metaPayload.colorBy as string)
          : undefined;
      const metaSampled = Boolean(metaPayload?.sampled);
      const shownCountRaw = metaPayload?.shown_count ?? metaPayload?.shownCount ?? metaPayload?.point_count;
      const totalCountRaw = metaPayload?.total_count ?? metaPayload?.totalCount;
      const metaDegraded = Boolean(metaPayload?.degraded);

      const arrayNames: ProjectionArtifactName[] = ['x', 'y', 'cat', 'ids'];
      const responses = await Promise.all(
        arrayNames.map(async (name) => {
          const res = await projectionsV2Api.getArtifact(collectionId, projection.id, name);
          return res.data;
        })
      );

      const [xBuf, yBuf, catBuf, idsBuf] = responses;
      const x = new Float32Array(xBuf);
      const y = new Float32Array(yBuf);
      const category = new Uint8Array(catBuf);
      const ids = new Int32Array(idsBuf);

      if (x.length !== y.length || x.length !== category.length) {
        throw new Error('Projection arrays have inconsistent lengths');
      }

      if (requestId === activeRequestId.current) {
        const parsedShownCount = (() => {
          const value = Number(shownCountRaw);
          return Number.isFinite(value) && value > 0 ? value : x.length;
        })();
        const parsedTotalCount = (() => {
          const value = Number(totalCountRaw);
          if (Number.isFinite(value) && value > 0) {
            return Math.max(value, parsedShownCount);
          }
          return metaSampled ? Math.max(parsedShownCount, x.length) : x.length;
        })();

        setActiveProjection({
          projectionId: projection.id,
          pointCount: x.length,
          arrays: { x, y, category },
          ids,
          status: 'loaded',
        });
        setActiveProjectionMeta({
          color_by: metaColorBy,
          legend: legendPayload,
          sampled: metaSampled,
          shown_count: parsedShownCount,
          total_count: parsedTotalCount,
          degraded: metaDegraded,
        });
        setLabelsEnabled(legendPayload.length > 0);
      }
    } catch (error: unknown) {
      if (requestId === activeRequestId.current) {
        setActiveProjection({
          projectionId: projection.id,
          pointCount: 0,
          status: 'error',
          error: error instanceof Error ? error.message : 'Failed to load projection data',
        });
        setActiveProjectionMeta(null);
      }
    }
  };

  const handleRecomputeProjection = async () => {
    openRecomputeDialog();
  };

  const handleTooltipEvent = useCallback(
    (value: DataPoint | null) => {
      setActiveTooltip(value);
      handleTooltip(value);
    },
    [handleTooltip]
  );

  const handleTooltipLeaveEvent = useCallback(() => {
    setActiveTooltip(null);
    handleTooltipLeave();
  }, [handleTooltipLeave]);

  // User preference for render mode, persisted per projection id.
  // 'auto' lets the tab choose based on the current point count.
  const currentRenderMode: RenderMode = useMemo(() => {
    if (!activeProjection.projectionId) return 'auto';
    return renderModeByProjection[activeProjection.projectionId] ?? 'auto';
  }, [activeProjection.projectionId, renderModeByProjection]);

  // Concrete mode passed to EmbeddingView. Auto switches to 'density' for large point clouds.
  const effectiveRenderMode = useMemo<'points' | 'density'>(() => {
    if (currentRenderMode !== 'auto') {
      return currentRenderMode;
    }
    return activeProjection.pointCount >= DENSITY_THRESHOLD ? 'density' : 'points';
  }, [currentRenderMode, activeProjection.pointCount]);
  const hasLegend = Boolean(activeProjectionMeta?.legend && activeProjectionMeta.legend.length > 0);

  // Cluster labels are derived from backend legend + arrays.
  // They are relatively expensive to compute, so we:
  // - require labelsEnabled and a non-empty legend; and
  // - memoize on arrays/legend/hasLegend to avoid recomputing on every render.
  // The result is passed into EmbeddingView as EmbeddingLabel[].
  const clusterLabels: EmbeddingLabel[] = useMemo(() => {
    if (!labelsEnabled || !activeProjection.arrays || !hasLegend) {
      return [];
    }
    return createCategoryLabels({
      x: activeProjection.arrays.x,
      y: activeProjection.arrays.y,
      category: activeProjection.arrays.category,
      legend: activeProjectionMeta?.legend ?? [],
      minPoints: DEFAULT_CATEGORY_LABEL_OPTIONS.minPoints,
      maxLabels: DEFAULT_CATEGORY_LABEL_OPTIONS.maxLabels,
    });
  }, [labelsEnabled, activeProjection.arrays, activeProjectionMeta?.legend, hasLegend]);

  // Nearest-point helper for EmbeddingView. Embedding Atlas expects a querySelection
  // callback to resolve hover / click locations into concrete DataPoints. We implement
  // a simple distance-based search over the current projection arrays and use the
  // zero-based array index as the DataPoint identifier so tooltip/selection plumbing
  // can map back through `ids`.
  const querySelection = useCallback(
    async (x: number, y: number, unitDistance: number): Promise<DataPoint | null> => {
      const arrays = activeProjection.arrays;
      if (!arrays) {
        return null;
      }

      const { x: xs, y: ys, category } = arrays;
      const length = xs.length;
      if (length === 0) {
        return null;
      }

      // Allow a small radius around the cursor in data space. The library passes
      // `unitDistance` as the distance corresponding to one pixel.
      const maxDistance = unitDistance * 3;
      const maxDistanceSq = maxDistance * maxDistance;

      let bestIndex = -1;
      let bestDistanceSq = maxDistanceSq;

      for (let index = 0; index < length; index += 1) {
        const dx = xs[index] - x;
        const dy = ys[index] - y;
        const distanceSq = dx * dx + dy * dy;
        if (distanceSq < bestDistanceSq) {
          bestDistanceSq = distanceSq;
          bestIndex = index;
        }
      }

      if (bestIndex === -1) {
        return null;
      }

      const dataPoint: DataPoint = {
        x: xs[bestIndex],
        y: ys[bestIndex],
        category: category[bestIndex],
        identifier: bestIndex,
      };

      return dataPoint;
    },
    [activeProjection.arrays]
  );

  const tooltipCustomProps = useMemo(
    () => ({
      getTooltipIndex: (tooltip: DataPoint | null) => getProjectionPointIndex(tooltip),
      ids: activeProjection.ids,
      tooltipState,
    }),
    [activeProjection.ids, tooltipState]
  );

  const shownCountDisplay = activeProjectionMeta?.shown_count ?? activeProjection.pointCount;
  const totalCountDisplay = activeProjectionMeta?.total_count ?? Math.max(shownCountDisplay, activeProjection.pointCount);

  const metadataHashContext = useMemo<ProjectionMetadataHashContext | null>(() => {
    if (!collectionEmbeddingModel) {
      return null;
    }
    return {
      collectionId,
      embeddingModel: collectionEmbeddingModel,
      vectorCount: collectionVectorCount ?? 0,
      updatedAt: collectionUpdatedAt ?? null,
    };
  }, [collectionId, collectionEmbeddingModel, collectionVectorCount, collectionUpdatedAt]);

  const handleSelectionChange = async (indices: number[]) => {
    if (!activeProjection.projectionId || !activeProjection.ids) {
      setSelectionState({ indices: [], items: [], missing: [], loading: false, error: undefined });
      return;
    }

    const uniqueIndices = Array.from(new Set(indices)).filter(
      (index) => index >= 0 && index < activeProjection.ids!.length
    );

    if (uniqueIndices.length === 0) {
      setSelectionState({ indices: [], items: [], missing: [], loading: false, error: undefined });
      return;
    }

    const mappedIds = uniqueIndices
      .map((index) => activeProjection.ids![index])
      .filter((id): id is number => typeof id === 'number' && Number.isFinite(id));

    if (mappedIds.length === 0) {
      setSelectionState({ indices: [], items: [], missing: [], loading: false, error: undefined });
      return;
    }

    const requestId = ++selectionRequestId.current;
    setSelectionState((prev) => ({ ...prev, indices: uniqueIndices, loading: true, error: undefined }));
    try {
      // Selection resolution contract:
      // - send the ids corresponding to the selected indices to /select,
      // - populate selectionState with the returned ProjectionSelectionItem[] and missing_ids,
      // - mirror degraded back onto activeProjectionMeta so UI stays in sync with backend state.
      const response = await projectionsV2Api.select(collectionId, activeProjection.projectionId, mappedIds);
      if (requestId === selectionRequestId.current) {
        setSelectionState({
          indices: uniqueIndices,
          items: response.data?.items ?? [],
          missing: response.data?.missing_ids ?? [],
          loading: false,
          error: undefined,
        });
        if (response.data?.degraded && activeProjectionMeta) {
          setActiveProjectionMeta({ ...activeProjectionMeta, degraded: true });
        }
      }
    } catch (error) {
      if (requestId === selectionRequestId.current) {
        setSelectionState({
          indices: uniqueIndices,
          items: [],
          missing: [],
          loading: false,
          error: error instanceof Error ? error.message : 'Failed to resolve selection',
        });
      }
    }
  };

  const handleRenderModeChange = (mode: RenderMode) => {
    if (!activeProjection.projectionId) return;
    setRenderModeByProjection((prev) => {
      const existing = prev[activeProjection.projectionId];
      if (existing === mode) {
        return prev;
      }
      return {
        ...prev,
        [activeProjection.projectionId]: mode,
      };
    });
  };

  const handleOpenDocument = (item: ProjectionSelectionItem) => {
    if (!item.document_id) {
      addToast({ type: 'error', message: 'No document available to open' });
      return;
    }

    trackTelemetry('visualize_selection_open', {
      collection_id: collectionId,
      document_id: item.document_id,
      chunk_id: item.chunk_id,
      chunk_index: item.chunk_index,
      timestamp: new Date().toISOString(),
    });

    // Open document viewer with chunk context
    setShowDocumentViewer({
      collectionId,
      docId: item.document_id,
      chunkId: item.chunk_id !== null && item.chunk_id !== undefined ? String(item.chunk_id) : undefined,
    });
  };

  const handleFindSimilar = async (item: ProjectionSelectionItem) => {
    if (!item.content_preview) {
      addToast({ type: 'error', message: 'No content available to search with' });
      return;
    }

    // Truncate query to reasonable length (500 chars)
    const query = item.content_preview.slice(0, 500);

    setSimilarSearchState({
      loading: true,
      error: null,
      results: [],
      visible: true,
    });

    try {
      const response = await searchV2Api.search({
        query,
        collection_uuids: [collectionId],
        k: 10,
        search_type: 'semantic',
      });

      trackTelemetry('visualize_selection_find_similar', {
        collection_id: collectionId,
        chunk_id: item.chunk_id,
        query: query.slice(0, 100),
        result_count: response.data?.results?.length ?? 0,
        timestamp: new Date().toISOString(),
      });

      setSimilarSearchState({
        loading: false,
        error: null,
        results: response.data?.results ?? [],
        visible: true,
      });
    } catch (error) {
      const errorMessage = getErrorMessage(error);
      setSimilarSearchState({
        loading: false,
        error: errorMessage,
        results: [],
        visible: true,
      });
      addToast({ type: 'error', message: `Failed to find similar chunks: ${errorMessage}` });
    }
  };

  const openRecomputeDialog = () => {
    setRecomputeReducer(selectedReducer);
    setRecomputeParams({
      n_neighbors: '15',
      min_dist: '0.1',
      metric: selectedReducer === 'tsne' ? 'euclidean' : 'cosine',
      sample_n: '',
      perplexity: '30',
      learning_rate: '200',
      n_iter: '1000',
      init: 'pca',
    });
    setRecomputeError(undefined);
    setRecomputeDialogOpen(true);
  };

  const closeRecomputeDialog = () => {
    if (!startProjection.isPending) {
      setRecomputeDialogOpen(false);
      setRecomputeError(undefined);
    }
  };

  const handleRecomputeReducerChange = (value: ProjectionReducer) => {
    setRecomputeReducer(value);
    setRecomputeParams((prev) => ({
      ...prev,
      metric:
        value === 'tsne'
          ? (TSNE_METRIC_OPTIONS.includes(prev.metric) ? prev.metric : 'euclidean')
          : value === 'umap'
            ? (METRIC_OPTIONS.includes(prev.metric) ? prev.metric : 'cosine')
            : prev.metric,
    }));
  };

  const handleRecomputeSubmit = async () => {
    setRecomputeError(undefined);

    const payload: StartProjectionRequest = {
      reducer: recomputeReducer,
      color_by: selectedColorBy,
    };
    const config: Record<string, unknown> = {};
    if (recomputeReducer === 'umap') {
      const parsedNeighbors = Number(recomputeParams.n_neighbors);
      const parsedMinDist = Number(recomputeParams.min_dist);
      const metricValue = recomputeParams.metric?.trim() || 'cosine';
      const errors: string[] = [];

      if (!Number.isFinite(parsedNeighbors) || parsedNeighbors < 2) {
        errors.push('n_neighbors must be a number ≥ 2.');
      }
      if (!Number.isFinite(parsedMinDist) || parsedMinDist < 0 || parsedMinDist > 1) {
        errors.push('min_dist must be between 0 and 1.');
      }
      if (!metricValue) {
        errors.push('metric is required.');
      }

      if (errors.length > 0) {
        setRecomputeError(errors.join(' '));
        return;
      }

      config.n_neighbors = Math.floor(parsedNeighbors);
      config.min_dist = parsedMinDist;
      config.metric = metricValue;
    } else if (recomputeReducer === 'tsne') {
      const parsedPerplexity = Number(recomputeParams.perplexity);
      const parsedLearningRate = Number(recomputeParams.learning_rate);
      const parsedIterations = Number(recomputeParams.n_iter);
      const metricValue = recomputeParams.metric?.trim() || 'euclidean';
      const initValue = recomputeParams.init === 'random' ? 'random' : 'pca';
      const errors: string[] = [];

      if (!Number.isFinite(parsedPerplexity) || parsedPerplexity <= 0) {
        errors.push('perplexity must be a positive number.');
      }
      if (!Number.isFinite(parsedLearningRate) || parsedLearningRate <= 0) {
        errors.push('learning_rate must be a positive number.');
      }
      if (!Number.isFinite(parsedIterations) || parsedIterations < 250) {
        errors.push('n_iter must be an integer ≥ 250.');
      }
      if (!metricValue) {
        errors.push('metric is required.');
      }

      if (errors.length > 0) {
        setRecomputeError(errors.join(' '));
        return;
      }

      config.perplexity = parsedPerplexity;
      config.learning_rate = parsedLearningRate;
      config.n_iter = Math.floor(parsedIterations);
      config.metric = metricValue;
      config.init = initValue;
    }
    if (recomputeParams.sample_n !== '') {
      const sampleNumeric = Number(recomputeParams.sample_n);
      if (!Number.isFinite(sampleNumeric) || sampleNumeric <= 0) {
        setRecomputeError('Sample size must be a positive number.');
        return;
      }
      const sampleSize = Math.floor(sampleNumeric);
      // Encode sampling as a top-level field so the backend service can
      // normalise aliases (sample_size / sample_limit / sample_n) into the
      // immutable ProjectionRun.config used by the worker. Keeping this at the
      // top level avoids reducer-specific config normalisation from dropping
      // sampling controls (e.g. for t-SNE).
      payload.sample_size = sampleSize;
    }

    if (Object.keys(config).length > 0) {
      payload.config = config;
    }

    let metadataHash: string | undefined;
    if (metadataHashContext) {
      try {
        metadataHash = await computeProjectionMetadataHash(metadataHashContext, payload);
        payload.metadata_hash = metadataHash;
      } catch (error) {
        console.warn('Failed to compute projection metadata hash for recompute', error);
      }
    }

    trackTelemetry('visualize_recompute_start', {
      collection_id: collectionId,
      reducer: recomputeReducer,
      color_by: selectedColorBy,
      sample_size: payload.sample_size ?? payload.sample_n ?? null,
      metadata_hash: metadataHash ?? null,
      timestamp: new Date().toISOString(),
    });

    try {
      const response = await startProjectionWithPayload(payload);
      if (response?.idempotent_reuse) {
        trackTelemetry('visualize_recompute_reuse', {
          collection_id: collectionId,
          projection_id: response.id,
          metadata_hash: metadataHash ?? null,
          timestamp: new Date().toISOString(),
        });
      }
      setRecomputeDialogOpen(false);
      setRecomputeError(undefined);
    } catch (error) {
      setRecomputeError(error instanceof Error ? error.message : 'Failed to start projection');
    }
  };

  return (
    <div className="space-y-6">
      <section className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Start a new projection</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Projection method
            </label>
            <div className="grid md:grid-cols-3 gap-3">
              {REDUCER_OPTIONS.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => setSelectedReducer(option.value)}
                  className={`border rounded-lg p-3 text-left transition-colors ${selectedReducer === option.value
                    ? 'border-purple-500 bg-purple-50'
                    : 'border-gray-200 hover:border-purple-300'
                    }`}
                >
                  <div className="font-medium text-gray-900">{option.label}</div>
                  <p className="text-sm text-gray-600 mt-1">{option.description}</p>
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Color by</label>
            <select
              value={selectedColorBy}
              onChange={(event) => setSelectedColorBy(event.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 sm:text-sm"
            >
              {COLOR_BY_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <button
            type="button"
            onClick={handleStartProjection}
            disabled={startProjection.isPending}
            className="inline-flex items-center px-4 py-2 bg-purple-600 text-white rounded-md shadow-sm hover:bg-purple-700 disabled:opacity-50"
          >
            {startProjection.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Play className="h-4 w-4 mr-2" />
            )}
            Start Projection
          </button>
        </div>
      </section>

      {pendingOperationId && (
        <div className="rounded-md border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-700">
          <div className="flex items-center gap-2">
            <span className="font-medium">Projection recompute in progress…</span>
            {currentOperationStatus && (
              <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                {currentOperationStatus}
              </span>
            )}
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-blue-600">
            <span>Operation ID: {pendingOperationId}</span>
            <span>{isOperationConnected ? 'Live updates active.' : 'Connecting to progress updates…'}</span>
            {!isOperationConnected && currentOperationStatus && (
              <span className="italic">Last known status: {currentOperationStatus}</span>
            )}
          </div>
        </div>
      )}

      {!pendingOperationId && inProgressProjection && (
        <div className="rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700">
          <div className="flex items-center gap-2">
            <span className="font-medium">Projection in progress</span>
            <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
              {inProgressProjection.operation_status || inProgressProjection.status}
            </span>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-amber-600">
            <span>Projection: {inProgressProjection.reducer.toUpperCase()}</span>
            {inProgressProjection.operation_id && (
              <span>Operation ID: {inProgressProjection.operation_id}</span>
            )}
            <span className="italic">Status from last refresh (WebSocket unavailable)</span>
          </div>
        </div>
      )}

      <section className="bg-white border border-gray-200 rounded-lg shadow-sm">
        <header className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Projection runs</h3>
          {isLoading && <Loader2 className="h-4 w-4 animate-spin text-purple-600" />}
        </header>
        {sortedProjections.length === 0 ? (
          <div className="p-6 text-center text-gray-500">
            <p className="font-medium text-gray-700 mb-2">No projections yet</p>
            <p className="text-sm">Start a projection to generate a 2D representation of your embeddings.</p>
          </div>
        ) : (
          <div className="overflow-hidden">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                    Projection
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                    Progress
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                    Created
                  </th>
                  <th className="px-4 py-3" />
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {sortedProjections.map((projection) => {
                  // Prefer operation_status over projection.status for more accurate state
                  const displayStatus = projection.operation_status || projection.status;
                  const progress = projectionProgress(displayStatus);
                  return (
                    <tr key={projection.id} className="hover:bg-gray-50">
                      <td className="px-4 py-3 text-sm text-gray-900">
                        <div className="font-medium text-gray-900">{projection.reducer.toUpperCase()}</div>
                        <div className="text-xs text-gray-500">ID: {projection.id}</div>
                        {projection.message && (
                          <div className="mt-1 text-xs text-gray-500">{projection.message}</div>
                        )}
                      </td>
                      <td className="px-4 py-3">{statusBadge(displayStatus)}</td>
                      <td className="px-4 py-3">
                        <div className="h-2 rounded bg-gray-200">
                          <div
                            className="h-2 rounded bg-purple-500 transition-all"
                            style={{ width: `${progress}%` }}
                          />
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-500">
                        {projection.created_at
                          ? new Date(projection.created_at).toLocaleString()
                          : '—'}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <div className="flex items-center justify-end gap-3">
                          {projection.status === 'completed' && (
                            <button
                              type="button"
                              onClick={() => handleViewProjection(projection)}
                              className="text-sm text-purple-600 hover:text-purple-800 inline-flex items-center"
                            >
                              <Eye className="h-4 w-4 mr-1" /> View
                            </button>
                          )}
                          <button
                            type="button"
                            onClick={() => handleDeleteProjection(projection.id)}
                            disabled={deleteProjection.isPending}
                            className="text-sm text-red-600 hover:text-red-800 inline-flex items-center"
                          >
                            <Trash2 className="h-4 w-4 mr-1" /> Delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="bg-white border border-gray-200 rounded-lg shadow-sm p-4">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Projection preview</h3>
        {activeProjection.status === 'idle' && (
          <div className="text-sm text-gray-600">
            Select a completed projection to load its coordinates and visualize the embedding.
          </div>
        )}

        {activeProjection.status === 'loading' && (
          <div className="flex items-center gap-2 text-sm text-purple-700">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading projection data…
          </div>
        )}

        {activeProjection.status === 'error' && (
          <div className="flex items-center gap-2 text-sm text-red-600">
            <AlertCircle className="h-4 w-4" /> {activeProjection.error}
          </div>
        )}

        {activeProjection.status === 'loaded' && activeProjection.arrays && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-3 text-sm text-gray-700">
              <span>
                Loaded <span className="font-semibold">{activeProjection.pointCount}</span> points.
                Categories: {new Set(activeProjection.arrays.category).size}.
              </span>
              {activeProjectionMeta?.sampled && (
                <span
                  className="inline-flex items-center gap-1 rounded-full border border-amber-300 bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-700"
                  title={`Showing ${shownCountDisplay.toLocaleString()} of ${totalCountDisplay.toLocaleString()} points`}
                >
                  Sampled
                </span>
              )}
            </div>
            <div className="flex flex-col gap-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex flex-wrap items-center gap-2 text-sm text-gray-700">
                  <span className="font-medium text-gray-800">Rendering</span>
                  <div className="inline-flex overflow-hidden rounded-md border border-gray-200 bg-white">
                    {RENDER_MODE_OPTIONS.map((mode) => {
                      const isActive = currentRenderMode === mode;
                      const label =
                        mode === 'auto'
                          ? 'Auto'
                          : mode === 'density'
                            ? 'Density'
                            : 'Points';
                      return (
                        <button
                          key={mode}
                          type="button"
                          onClick={() => handleRenderModeChange(mode)}
                          className={`px-2.5 py-1 text-xs font-medium transition focus:outline-none focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-1 ${isActive
                            ? 'bg-purple-600 text-white shadow-sm'
                            : 'text-gray-600 hover:bg-purple-50'
                            }`}
                          aria-pressed={isActive}
                        >
                          {label}
                        </button>
                      );
                    })}
                  </div>
                  <span className="text-xs text-gray-500">
                    {currentRenderMode === 'auto'
                      ? `Auto switches to density at ${DENSITY_THRESHOLD.toLocaleString()}+ points`
                      : `Forced ${currentRenderMode}`}
                  </span>
                </div>
                {hasLegend ? (
                  <label className="flex items-center gap-2 text-sm text-gray-700">
                    <input
                      type="checkbox"
                      className="h-4 w-4 rounded border-gray-300 text-purple-600 focus:ring-purple-500"
                      checked={labelsEnabled}
                      onChange={(event) => setLabelsEnabled(event.target.checked)}
                    />
                    Show labels
                  </label>
                ) : null}
              </div>
              {activeProjectionMeta?.color_by && (
                <div className="flex items-center justify-between">
                  <p className="text-sm text-gray-600">
                    Colored by <span className="font-medium">{activeProjectionMeta.color_by}</span>
                  </p>
                  {(activeProjectionMeta.color_by !== selectedColorBy || activeProjectionMeta.degraded) && (
                    <button
                      type="button"
                      onClick={handleRecomputeProjection}
                      disabled={startProjection.isPending}
                      className="inline-flex items-center px-3 py-1.5 text-sm rounded-md border border-purple-500 text-purple-600 hover:bg-purple-50 disabled:opacity-50"
                    >
                      {startProjection.isPending ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      ) : null}
                      {activeProjectionMeta.degraded
                        ? 'Recompute to refresh'
                        : `Recompute with ${COLOR_BY_OPTIONS.find((opt) => opt.value === selectedColorBy)?.label ?? selectedColorBy}`}
                    </button>
                  )}
                </div>
              )}
              {activeProjectionMeta?.legend && activeProjectionMeta.legend.length > 0 && (
                <div className="bg-gray-50 border border-gray-200 rounded-md p-3">
                  <h4 className="text-sm font-semibold text-gray-700 mb-2">Legend</h4>
                  <ul className="max-h-48 overflow-y-auto text-sm text-gray-600 space-y-1">
                    {activeProjectionMeta.legend.map((entry) => (
                      <li key={entry.index} className="flex items-center justify-between">
                        <span>{entry.label}</span>
                        {typeof entry.count === 'number' && (
                          <span className="text-xs text-gray-500">{entry.count.toLocaleString()}</span>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
            <div
              className="border border-gray-200 rounded-md overflow-hidden"
              ref={viewContainerRef}
              style={{ minHeight: '320px' }}
              onPointerLeave={handleTooltipLeaveEvent}
            >
              <Suspense fallback={<div className="p-4 text-sm text-purple-700">Rendering projection…</div>}>
                {/* EmbeddingView (embedding-atlas)
                    - data: typed arrays (Float32Array / Uint8Array) for
                      x / y / category. Length consistency is validated above.
                    - tooltip: latest DataPoint from EmbeddingView; forwarded to
                      useProjectionTooltip via onTooltip.
                    - config.mode: effectiveRenderMode ('points' | 'density'),
                      derived from currentRenderMode and point count.
                    - labels: optional EmbeddingLabel[] computed from legend +
                      arrays for cluster-level annotations.
                    - onSelection: receives indices or DataPoint-like objects;
                      we normalize to indices and call /select to resolve
                      ProjectionSelectionItem[] into selectionState. */}
                <EmbeddingView
                  data={{
                    x: activeProjection.arrays.x,
                    y: activeProjection.arrays.y,
                    category: activeProjection.arrays.category,
                  }}
                  tooltip={activeTooltip}
                  width={viewSize.width}
                  height={viewSize.height}
                  pixelRatio={pixelRatio}
                  theme={{ statusBar: true }}
                  config={{ mode: effectiveRenderMode }}
                  labels={labelsEnabled && clusterLabels.length > 0 ? clusterLabels : undefined}
                  querySelection={querySelection}
                  onTooltip={handleTooltipEvent}
                  customTooltip={{
                    class: EmbeddingTooltipAdapter,
                    props: tooltipCustomProps,
                  }}
                  onSelection={(points) => {
                    // Accept both numeric indices and { index }-like objects
                    const indices = Array.isArray(points)
                      ? points
                          .map((value) => getProjectionPointIndex(value))
                          .filter((idx): idx is number => typeof idx === 'number' && Number.isInteger(idx) && idx >= 0)
                      : [];
                    void handleSelectionChange(indices);
                  }}
                />
              </Suspense>
            </div>
            {(selectionState.indices.length > 0 || selectionState.loading || selectionState.error) && (
              <div className="border border-gray-200 rounded-md p-4 bg-white shadow-sm">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-semibold text-gray-800">Selection</h4>
                  <span className="text-xs text-gray-500">
                    Indices: {selectionState.indices.length.toLocaleString()}
                  </span>
                </div>
                {activeProjectionMeta?.degraded && !selectionState.loading && (
                  <p className="mb-2 text-xs text-amber-700">
                    This projection is marked as degraded; selection results may be incomplete. Consider recomputing.
                  </p>
                )}
                {selectionState.loading && (
                  <div className="flex items-center gap-2 text-sm text-purple-600">
                    <Loader2 className="h-4 w-4 animate-spin" /> Resolving selection…
                  </div>
                )}
                {selectionState.error && !selectionState.loading && (
                  <div className="text-sm text-red-600">{selectionState.error}</div>
                )}
                {!selectionState.loading && selectionState.items.length > 0 && (
                  <>
                    {selectionState.items.length > 1 && (
                      <p className="text-xs text-gray-500 mb-2 italic">
                        Actions apply to the first selected point
                      </p>
                    )}
                    <ul className="space-y-3 text-sm text-gray-700 max-h-64 overflow-y-auto">
                      {selectionState.items.map((item) => (
                        <li key={`${item.selected_id}-${item.index}`} className="border border-gray-200 rounded-md p-3">
                          <div className="text-xs text-gray-500 mb-1">
                            Point #{item.index + 1} • ID {item.selected_id}
                          </div>
                          {item.document_id && (
                            <div className="font-medium text-gray-900">Document {item.document_id}</div>
                          )}
                          {item.chunk_index !== undefined && item.chunk_index !== null && (
                            <div className="text-xs text-gray-500">Chunk #{item.chunk_index}</div>
                          )}
                          {item.content_preview && (
                            <p className="mt-2 text-sm text-gray-600 line-clamp-3">{item.content_preview}</p>
                          )}
                          <div className="mt-3 flex gap-2">
                            <button
                              type="button"
                              onClick={() => handleOpenDocument(item)}
                              disabled={!item.document_id}
                              title="View the full document containing this chunk"
                              className="text-xs px-2 py-1 rounded border border-gray-300 text-gray-600 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                              Open
                            </button>
                            <button
                              type="button"
                              onClick={() => handleFindSimilar(item)}
                              disabled={!item.content_preview || similarSearchState.loading}
                              title="Search for semantically similar content"
                              className="text-xs px-2 py-1 rounded border border-purple-400 text-purple-600 hover:bg-purple-50 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                              {similarSearchState.loading ? 'Searching...' : 'Find Similar'}
                            </button>
                          </div>
                        </li>
                      ))}
                    </ul>
                  </>
                )}
                {!selectionState.loading &&
                  selectionState.items.length === 0 &&
                  !selectionState.error &&
                  selectionState.indices.length > 0 && (
                    <p className="text-sm text-gray-500">
                      Selected points could not be mapped to documents. Try recomputing the projection or refining your
                      selection.
                    </p>
                  )}
                {selectionState.missing.length > 0 && (
                  <p className="mt-2 text-xs text-amber-600">
                    {selectionState.missing.length.toLocaleString()} point(s) could not be resolved.
                  </p>
                )}

                {/* Similar Results Section */}
                {similarSearchState.visible && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <div className="flex items-center justify-between mb-3">
                      <h5 className="text-sm font-semibold text-gray-800">Similar Chunks</h5>
                      <button
                        type="button"
                        onClick={() => setSimilarSearchState((prev) => ({ ...prev, visible: false }))}
                        className="text-gray-500 hover:text-gray-700"
                        title="Close similar results"
                      >
                        <X className="h-4 w-4" />
                      </button>
                    </div>

                    {similarSearchState.loading && (
                      <div className="flex items-center gap-2 text-sm text-purple-600">
                        <Loader2 className="h-4 w-4 animate-spin" /> Searching for similar chunks…
                      </div>
                    )}

                    {similarSearchState.error && !similarSearchState.loading && (
                      <div className="text-sm text-red-600">{similarSearchState.error}</div>
                    )}

                    {!similarSearchState.loading && similarSearchState.results.length > 0 && (
                      <ul className="space-y-2 text-sm text-gray-700 max-h-96 overflow-y-auto">
                        {similarSearchState.results.map((result) => (
                          <li
                            key={`${result.document_id}-${result.chunk_index}`}
                            className="border border-gray-200 rounded-md p-3 hover:bg-gray-50"
                          >
                            <div className="flex items-center justify-between mb-1">
                              <div className="font-medium text-gray-900 text-xs truncate">
                                {result.file_name}
                              </div>
                              <span className="text-xs text-purple-600 font-medium ml-2">
                                {(result.score * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="text-xs text-gray-500 mb-2">
                              Chunk #{result.chunk_index}
                            </div>
                            <p className="text-sm text-gray-600 line-clamp-2 mb-2">{result.text}</p>
                            <button
                              type="button"
                              onClick={() => {
                                setShowDocumentViewer({
                                  collectionId,
                                  docId: result.document_id,
                                  chunkId: result.chunk_id,
                                });
                              }}
                              className="text-xs px-2 py-1 rounded border border-gray-300 text-gray-600 hover:bg-gray-100"
                            >
                              Open
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}

                    {!similarSearchState.loading && similarSearchState.results.length === 0 && !similarSearchState.error && (
                      <p className="text-sm text-gray-500">No similar chunks found.</p>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </section>

      {recomputeDialogOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4">
          <div className="w-full max-w-lg rounded-lg bg-white p-6 shadow-xl">
            <h3 className="text-lg font-semibold text-gray-900">Recompute Projection</h3>
            <p className="mt-2 text-sm text-gray-600">
              Choose reducer and sampling parameters for the new projection run.
            </p>

            <div className="mt-4 space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Reducer</label>
                <select
                  value={recomputeReducer}
                  onChange={(event) =>
                    handleRecomputeReducerChange(event.target.value as ProjectionReducer)
                  }
                  className="block w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 sm:text-sm"
                >
                  <option value="umap">UMAP</option>
                  <option value="tsne">t-SNE</option>
                  <option value="pca">PCA</option>
                </select>
              </div>

              {recomputeReducer === 'umap' && (
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">n_neighbors</label>
                    <input
                      type="number"
                      min={2}
                      value={recomputeParams.n_neighbors}
                      onChange={(event) =>
                        setRecomputeParams((prev) => ({ ...prev, n_neighbors: event.target.value }))
                      }
                      className="block w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 sm:text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">min_dist</label>
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.05}
                      value={recomputeParams.min_dist}
                      onChange={(event) =>
                        setRecomputeParams((prev) => ({ ...prev, min_dist: event.target.value }))
                      }
                      className="block w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 sm:text-sm"
                    />
                  </div>
                  <div className="md:col-span-2">
                    <label className="block text-sm font-medium text-gray-700 mb-1">Metric</label>
                    <select
                      value={recomputeParams.metric}
                      onChange={(event) =>
                        setRecomputeParams((prev) => ({ ...prev, metric: event.target.value }))
                      }
                      className="block w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 sm:text-sm"
                    >
                      {METRIC_OPTIONS.map((metricOption) => (
                        <option key={metricOption} value={metricOption}>
                          {metricOption}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              )}

              {recomputeReducer === 'tsne' && (
                <div className="grid gap-4 md:grid-cols-2">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">perplexity</label>
                    <input
                      type="number"
                      min={1}
                      value={recomputeParams.perplexity}
                      onChange={(event) =>
                        setRecomputeParams((prev) => ({ ...prev, perplexity: event.target.value }))
                      }
                      className="block w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 sm:text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">learning_rate</label>
                    <input
                      type="number"
                      min={1}
                      value={recomputeParams.learning_rate}
                      onChange={(event) =>
                        setRecomputeParams((prev) => ({ ...prev, learning_rate: event.target.value }))
                      }
                      className="block w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 sm:text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">n_iter</label>
                    <input
                      type="number"
                      min={250}
                      value={recomputeParams.n_iter}
                      onChange={(event) =>
                        setRecomputeParams((prev) => ({ ...prev, n_iter: event.target.value }))
                      }
                      className="block w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 sm:text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">metric</label>
                    <select
                      value={recomputeParams.metric}
                      onChange={(event) =>
                        setRecomputeParams((prev) => ({ ...prev, metric: event.target.value }))
                      }
                      className="block w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 sm:text-sm"
                    >
                      {TSNE_METRIC_OPTIONS.map((metricOption) => (
                        <option key={metricOption} value={metricOption}>
                          {metricOption}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">init</label>
                    <select
                      value={recomputeParams.init}
                      onChange={(event) =>
                        setRecomputeParams((prev) => ({ ...prev, init: event.target.value as 'pca' | 'random' }))
                      }
                      className="block w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 sm:text-sm"
                    >
                      {TSNE_INIT_OPTIONS.map((initOption) => (
                        <option key={initOption} value={initOption}>
                          {initOption}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Sample size</label>
                <input
                  type="number"
                  min={1}
                  placeholder="Optional"
                  value={recomputeParams.sample_n}
                  onChange={(event) =>
                    setRecomputeParams((prev) => ({ ...prev, sample_n: event.target.value }))
                  }
                  className="block w-full rounded-md border-gray-300 shadow-sm focus:border-purple-500 focus:ring-purple-500 sm:text-sm"
                />
                <p className="mt-1 text-xs text-gray-500">
                  Leave blank to use the default cap ({SAMPLE_LIMIT_CAP.toLocaleString()} points).
                </p>
              </div>

              {recomputeError && (
                <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                  {recomputeError}
                </div>
              )}
            </div>

            <div className="mt-6 flex justify-end gap-2">
              <button
                type="button"
                onClick={closeRecomputeDialog}
                className="inline-flex items-center rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100"
                disabled={startProjection.isPending}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleRecomputeSubmit}
                disabled={startProjection.isPending}
                className="inline-flex items-center rounded-md bg-purple-600 px-4 py-2 text-sm font-medium text-white hover:bg-purple-700 disabled:opacity-50"
              >
                {startProjection.isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Starting…
                  </>
                ) : (
                  'Start'
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default EmbeddingVisualizationTab;
