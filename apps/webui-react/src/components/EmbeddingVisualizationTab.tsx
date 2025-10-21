import { Suspense, useMemo, useRef, useState, lazy } from 'react';
import { AlertCircle, Loader2, Play, Trash2, Eye } from 'lucide-react';
import {
  useCollectionProjections,
  useDeleteProjection,
  useStartProjection,
} from '../hooks/useProjections';
import type { ProjectionMetadata, ProjectionReducer } from '../types/projection';

const EmbeddingView = lazy(() => import('embedding-atlas/react').then((mod) => ({ default: mod.EmbeddingView })));

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
}

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

function statusBadge(status: ProjectionMetadata['status']) {
  const base = 'px-2 py-1 rounded-full text-xs font-medium';
  switch (status) {
    case 'completed':
      return <span className={`${base} bg-green-100 text-green-800`}>Completed</span>;
    case 'running':
      return <span className={`${base} bg-blue-100 text-blue-800`}>Running</span>;
    case 'failed':
      return <span className={`${base} bg-red-100 text-red-800`}>Failed</span>;
    case 'cancelled':
      return <span className={`${base} bg-gray-100 text-gray-600`}>Cancelled</span>;
    default:
      return <span className={`${base} bg-amber-100 text-amber-800`}>Pending</span>;
  }
}

function projectionProgress(status: ProjectionMetadata['status']) {
  if (status === 'completed') return 100;
  if (status === 'running') return 60;
  if (status === 'pending') return 10;
  return 0;
}

export function EmbeddingVisualizationTab({ collectionId }: EmbeddingVisualizationTabProps) {
  const [selectedReducer, setSelectedReducer] = useState<ProjectionReducer>('umap');
  const [activeProjection, setActiveProjection] = useState<ProjectionDataState>({
    projectionId: '',
    pointCount: 0,
    status: 'idle',
  });
  const activeRequestId = useRef(0);
  const { data: projections = [], isLoading, refetch } = useCollectionProjections(collectionId);
  const startProjection = useStartProjection(collectionId);
  const deleteProjection = useDeleteProjection(collectionId);

  const sortedProjections = useMemo(
    () =>
      [...projections].sort((a, b) => {
        const aDate = a.created_at ? new Date(a.created_at).getTime() : 0;
        const bDate = b.created_at ? new Date(b.created_at).getTime() : 0;
        return bDate - aDate;
      }),
    [projections]
  );

  const handleStartProjection = async () => {
    await startProjection.mutateAsync({ reducer: selectedReducer });
    refetch();
  };

  const handleDeleteProjection = async (projectionId: string) => {
    await deleteProjection.mutateAsync(projectionId);
    refetch();
    setActiveProjection((prev) =>
      prev.projectionId === projectionId
        ? { projectionId: '', pointCount: 0, status: 'idle' }
        : prev
    );
  };

  const handleViewProjection = async (projection: ProjectionMetadata) => {
    if (projection.status !== 'completed') return;

    const requestId = ++activeRequestId.current;
    setActiveProjection({ projectionId: projection.id, pointCount: 0, status: 'loading' });

    try {
      const arrayNames = ['x', 'y', 'cat', 'ids'] as const;
      const responses = await Promise.all(
        arrayNames.map((name) =>
          fetch(
            `/api/v2/collections/${collectionId}/projections/${projection.id}/arrays/${name}`
          ).then((res) => {
            if (!res.ok) {
              throw new Error(`Failed to load ${name}`);
            }
            return res.arrayBuffer();
          })
        )
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
        setActiveProjection({
          projectionId: projection.id,
          pointCount: x.length,
          arrays: { x, y, category },
          ids,
          status: 'loaded',
        });
      }
    } catch (error: unknown) {
      if (requestId === activeRequestId.current) {
        setActiveProjection({
          projectionId: projection.id,
          pointCount: 0,
          status: 'error',
          error: error instanceof Error ? error.message : 'Failed to load projection data',
        });
      }
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
                  className={`border rounded-lg p-3 text-left transition-colors ${
                    selectedReducer === option.value
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
                  const progress = projectionProgress(projection.status);
                  return (
                    <tr key={projection.id} className="hover:bg-gray-50">
                      <td className="px-4 py-3 text-sm text-gray-900">
                        <div className="font-medium text-gray-900">{projection.reducer.toUpperCase()}</div>
                        <div className="text-xs text-gray-500">ID: {projection.id}</div>
                        {projection.message && (
                          <div className="mt-1 text-xs text-gray-500">{projection.message}</div>
                        )}
                      </td>
                      <td className="px-4 py-3">{statusBadge(projection.status)}</td>
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
            <div className="text-sm text-gray-700">
              Loaded <span className="font-semibold">{activeProjection.pointCount}</span> points.
              Categories: {new Set(activeProjection.arrays.category).size}.
            </div>
            <div className="border border-gray-200 rounded-md overflow-hidden">
              <Suspense fallback={<div className="p-4 text-sm text-purple-700">Rendering projection…</div>}>
                <EmbeddingView
                  data={{
                    x: activeProjection.arrays.x,
                    y: activeProjection.arrays.y,
                    category: activeProjection.arrays.category,
                  }}
                />
              </Suspense>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

export default EmbeddingVisualizationTab;
