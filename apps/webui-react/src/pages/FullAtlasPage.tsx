/**
 * FullAtlasPage - Full Embedding Atlas visualization with coordinated views.
 *
 * This page provides access to the complete Embedding Atlas experience including:
 * - Scatter plot visualization with density/points modes
 * - Interactive data table with sorting and filtering
 * - Bar charts for metadata distribution (by document, file type, etc.)
 * - Cross-filtering between all views
 * - Real-time search and nearest neighbor discovery
 *
 * Data Flow:
 * 1. Load projection arrays (x, y, category, ids) from backend API
 * 2. Fetch metadata (legend, document info) via /select endpoint
 * 3. Convert to DuckDB-WASM table via Apache Arrow
 * 4. Initialize Mosaic coordinator
 * 5. Render EmbeddingAtlas with coordinated views
 *
 * This is a "power user" feature accessible via the Visualize tab's
 * "Explore in Full Atlas" link for deep embedding exploration.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Loader2, AlertCircle, ExternalLink } from 'lucide-react';
import { Coordinator, wasmConnector } from '@uwdata/mosaic-core';
import { EmbeddingAtlas } from 'embedding-atlas/react';
import { projectionsV2Api } from '../services/api/v2/projections';
import { ensureEmbeddingAtlasWebgpuCompatibility } from '../utils/embeddingAtlasWebgpuPatch';
import type { ProjectionLegendItem } from '../types/projection';

// Apply WebGPU compatibility patch
ensureEmbeddingAtlasWebgpuCompatibility();

interface ProjectionData {
  x: Float32Array;
  y: Float32Array;
  category: Uint8Array;
  ids: Int32Array;
  pointCount: number;
}

interface ProjectionMeta {
  color_by?: string;
  legend?: ProjectionLegendItem[];
  sampled?: boolean;
  shown_count?: number;
  total_count?: number;
}

type LoadingState = 'idle' | 'loading' | 'loaded' | 'error';

const TABLE_NAME = 'projection_data';

/**
 * Convert projection arrays to SQL INSERT statements for DuckDB.
 * For large datasets, this batches inserts to avoid memory issues.
 */
function generateInsertSQL(
  data: ProjectionData,
  legend: ProjectionLegendItem[]
): string[] {
  const BATCH_SIZE = 10000;
  const statements: string[] = [];

  // Create table schema
  statements.push(`
    CREATE TABLE IF NOT EXISTS ${TABLE_NAME} (
      id INTEGER PRIMARY KEY,
      x FLOAT,
      y FLOAT,
      category INTEGER,
      category_label VARCHAR
    )
  `);

  // Build category label lookup
  const categoryLabels = new Map<number, string>();
  legend.forEach((item) => {
    categoryLabels.set(item.index, item.label);
  });

  // Batch insert rows
  for (let batchStart = 0; batchStart < data.pointCount; batchStart += BATCH_SIZE) {
    const batchEnd = Math.min(batchStart + BATCH_SIZE, data.pointCount);
    const values: string[] = [];

    for (let i = batchStart; i < batchEnd; i++) {
      const id = data.ids[i];
      const x = data.x[i];
      const y = data.y[i];
      const cat = data.category[i];
      const label = categoryLabels.get(cat) ?? `Category ${cat}`;
      // Escape single quotes in labels
      const escapedLabel = label.replace(/'/g, "''");
      values.push(`(${id}, ${x}, ${y}, ${cat}, '${escapedLabel}')`);
    }

    statements.push(`INSERT INTO ${TABLE_NAME} VALUES ${values.join(',')}`);
  }

  return statements;
}

export function FullAtlasPage() {
  const { collectionId, projectionId } = useParams<{
    collectionId: string;
    projectionId: string;
  }>();

  const [loadingState, setLoadingState] = useState<LoadingState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [projectionMeta, setProjectionMeta] = useState<ProjectionMeta | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [coordinator, setCoordinator] = useState<any>(null);
  const [isDbReady, setIsDbReady] = useState(false);

  // Load projection data and initialize DuckDB
  const initializeAtlas = useCallback(async () => {
    if (!collectionId || !projectionId) {
      setError('Missing collection or projection ID');
      setLoadingState('error');
      return;
    }

    setLoadingState('loading');
    setError(null);

    try {
      // 1. Fetch metadata
      const metadataResponse = await projectionsV2Api.getMetadata(
        collectionId,
        projectionId
      );
      const meta = metadataResponse.data?.meta as Record<string, unknown> | null;
      const legend = Array.isArray(meta?.legend)
        ? (meta.legend as ProjectionLegendItem[])
        : [];
      const colorBy = typeof meta?.color_by === 'string' ? meta.color_by : undefined;
      const sampled = Boolean(meta?.sampled);
      const shownCount =
        typeof meta?.shown_count === 'number' ? meta.shown_count : undefined;
      const totalCount =
        typeof meta?.total_count === 'number' ? meta.total_count : undefined;

      setProjectionMeta({
        color_by: colorBy,
        legend,
        sampled,
        shown_count: shownCount,
        total_count: totalCount,
      });

      // 2. Fetch projection arrays
      const [xRes, yRes, catRes, idsRes] = await Promise.all([
        projectionsV2Api.getArtifact(collectionId, projectionId, 'x'),
        projectionsV2Api.getArtifact(collectionId, projectionId, 'y'),
        projectionsV2Api.getArtifact(collectionId, projectionId, 'cat'),
        projectionsV2Api.getArtifact(collectionId, projectionId, 'ids'),
      ]);

      const projectionData: ProjectionData = {
        x: new Float32Array(xRes.data),
        y: new Float32Array(yRes.data),
        category: new Uint8Array(catRes.data),
        ids: new Int32Array(idsRes.data),
        pointCount: new Float32Array(xRes.data).length,
      };

      // 3. Initialize DuckDB-WASM and Mosaic coordinator
      const connector = wasmConnector();
      const coord = new Coordinator(connector);

      // 4. Create table and insert data
      const sqlStatements = generateInsertSQL(projectionData, legend);
      for (const sql of sqlStatements) {
        await connector.query({ type: 'exec', sql });
      }

      setCoordinator(coord);
      setIsDbReady(true);
      setLoadingState('loaded');
    } catch (err) {
      console.error('Failed to initialize Full Atlas:', err);
      setError(err instanceof Error ? err.message : 'Failed to load projection');
      setLoadingState('error');
    }
  }, [collectionId, projectionId]);

  useEffect(() => {
    initializeAtlas();

    // Cleanup coordinator on unmount
    return () => {
      coordinator?.clear();
    };
  }, [initializeAtlas]);

  // EmbeddingAtlas data configuration
  const atlasDataConfig = useMemo(
    () => ({
      table: TABLE_NAME,
      id: 'id',
      projection: { x: 'x', y: 'y' },
      text: 'category_label', // Used for labels and search
    }),
    []
  );

  // Theme based on current color scheme
  const colorScheme = useMemo(() => {
    if (typeof window === 'undefined') return 'light';
    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
  }, []);

  if (!collectionId || !projectionId) {
    return (
      <div className="min-h-screen bg-[var(--bg-primary)] p-8">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center gap-3 text-red-600 dark:text-red-400">
            <AlertCircle className="h-5 w-5" />
            <span>Missing collection or projection ID in URL</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] flex flex-col">
      {/* Header */}
      <header className="border-b border-[var(--border)] bg-[var(--bg-secondary)] px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link
              to={`/collections/${collectionId}`}
              className="flex items-center gap-2 text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Collection
            </Link>
            <div className="h-6 w-px bg-[var(--border)]" />
            <h1 className="text-lg font-semibold text-[var(--text-primary)]">
              Full Embedding Atlas
            </h1>
          </div>
          <div className="flex items-center gap-4 text-sm text-[var(--text-muted)]">
            {projectionMeta?.color_by && (
              <span>
                Colored by <span className="font-medium">{projectionMeta.color_by}</span>
              </span>
            )}
            {projectionMeta?.sampled && projectionMeta.shown_count && (
              <span className="px-2 py-1 rounded-full bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 text-xs font-medium">
                Sampled: {projectionMeta.shown_count.toLocaleString()}
                {projectionMeta.total_count && (
                  <> of {projectionMeta.total_count.toLocaleString()}</>
                )}
              </span>
            )}
            <a
              href="https://apple.github.io/embedding-atlas/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-blue-600 dark:text-blue-400 hover:underline"
            >
              <ExternalLink className="h-3 w-3" />
              Atlas Docs
            </a>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 p-6">
        {loadingState === 'loading' && (
          <div className="flex flex-col items-center justify-center h-96 gap-4">
            <Loader2 className="h-8 w-8 animate-spin text-[var(--text-secondary)]" />
            <p className="text-[var(--text-secondary)]">
              Loading projection data and initializing DuckDB...
            </p>
            <p className="text-sm text-[var(--text-muted)]">
              This may take a moment for large datasets
            </p>
          </div>
        )}

        {loadingState === 'error' && (
          <div className="flex flex-col items-center justify-center h-96 gap-4">
            <AlertCircle className="h-8 w-8 text-red-500" />
            <p className="text-red-600 dark:text-red-400 font-medium">
              Failed to load projection
            </p>
            {error && (
              <p className="text-sm text-[var(--text-muted)] max-w-md text-center">
                {error}
              </p>
            )}
            <button
              onClick={initializeAtlas}
              className="mt-2 px-4 py-2 rounded-md bg-[var(--bg-tertiary)] text-[var(--text-primary)] hover:bg-[var(--border)] transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {loadingState === 'loaded' && isDbReady && coordinator && (
          <div className="h-[calc(100vh-140px)] rounded-lg border border-[var(--border)] overflow-hidden bg-[var(--bg-secondary)]">
            <EmbeddingAtlas
              coordinator={coordinator}
              data={atlasDataConfig}
              colorScheme={colorScheme}
            />
          </div>
        )}
      </main>

      {/* Info banner */}
      {loadingState === 'loaded' && (
        <footer className="border-t border-[var(--border)] bg-[var(--bg-secondary)] px-6 py-3">
          <p className="text-xs text-[var(--text-muted)] text-center">
            Full Atlas provides cross-filtering between views. Click points to select,
            drag to brush, and use the table to filter. Data is processed locally in your browser.
          </p>
        </footer>
      )}
    </div>
  );
}

export default FullAtlasPage;
