import { useCallback, useEffect, useRef, useState } from 'react';
import type { DataPoint } from 'embedding-atlas/react';
import { projectionsV2Api } from '../services/api/v2/projections';
import { LruCache } from '../utils/lruCache';

type TooltipStatus = 'idle' | 'loading' | 'success' | 'error';

export interface TooltipMetadata {
  selectedId: number;
  documentId?: string | number | null;
  chunkIndex?: number | null;
  contentPreview?: string | null;
  source: 'cache' | 'network';
}

export interface ProjectionTooltipState {
  status: TooltipStatus;
  position: { x: number; y: number } | null;
  metadata: TooltipMetadata | null;
  error?: string;
}

export const TOOLTIP_CACHE_SIZE = 512;
export const TOOLTIP_CACHE_TTL_MS = 60_000;
export const TOOLTIP_DEBOUNCE_MS = 50;
export const TOOLTIP_MAX_INFLIGHT = 5;

function isRequestAborted(error: unknown): boolean {
  if (!error || typeof error !== 'object') {
    return false;
  }

  const maybeError = error as { code?: string; name?: string };
  return (
    maybeError.code === 'ERR_CANCELED' ||
    maybeError.name === 'CanceledError' ||
    maybeError.name === 'AbortError'
  );
}

function toTooltipMetadata(item: any, selectedId: number): TooltipMetadata {
  const preview = typeof item?.content_preview === 'string' ? item.content_preview.slice(0, 200) : null;
  return {
    selectedId,
    documentId: item?.document_id ?? null,
    chunkIndex: item?.chunk_index ?? null,
    contentPreview: preview,
    source: 'network',
  };
}

export function useProjectionTooltip(
  collectionId: string | null,
  projectionId: string | null,
  ids: Int32Array | undefined
) {
  const cacheRef = useRef(
    new LruCache<number, TooltipMetadata>({ max: TOOLTIP_CACHE_SIZE, ttlMs: TOOLTIP_CACHE_TTL_MS })
  );
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latestRequestToken = useRef(0);
  const inflightTokens = useRef<number[]>([]);
  const cancelledTokens = useRef<Set<number>>(new Set());
  const controllersRef = useRef<Map<number, AbortController>>(new Map());

  const [state, setState] = useState<ProjectionTooltipState>({
    status: 'idle',
    position: null,
    metadata: null,
  });

  const clearState = useCallback(() => {
    setState({ status: 'idle', position: null, metadata: null });
  }, []);

  useEffect(() => {
    controllersRef.current.forEach((controller) => controller.abort());
    controllersRef.current.clear();
    cacheRef.current.clear();
    cancelledTokens.current.clear();
    inflightTokens.current = [];
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
      debounceTimerRef.current = null;
    }
    clearState();
    return () => {
      controllersRef.current.forEach((controller) => controller.abort());
      controllersRef.current.clear();
    };
  }, [projectionId, clearState]);

  const scheduleFetch = useCallback(
    (point: DataPoint & { index: number }) => {
      if (!collectionId || !projectionId || !ids) {
        setState({
          status: 'error',
          position: { x: point.x, y: point.y },
          metadata: null,
          error: 'No metadata available',
        });
        return;
      }

      const pointIndex = point.index;
      if (pointIndex < 0 || pointIndex >= ids.length) {
        setState({
          status: 'error',
          position: { x: point.x, y: point.y },
          metadata: null,
          error: 'No metadata available',
        });
        return;
      }

      const selectedId = ids[pointIndex];
      if (typeof selectedId !== 'number' || Number.isNaN(selectedId)) {
        setState({
          status: 'error',
          position: { x: point.x, y: point.y },
          metadata: null,
          error: 'No metadata available',
        });
        return;
      }

      const cached = cacheRef.current.get(selectedId);
      if (cached) {
        setState({
          status: 'success',
          position: { x: point.x, y: point.y },
          metadata: { ...cached, source: 'cache' },
        });
        return;
      }

      setState({
        status: 'loading',
        position: { x: point.x, y: point.y },
        metadata: null,
      });

      const token = ++latestRequestToken.current;
      const controller = new AbortController();
      controllersRef.current.set(token, controller);
      inflightTokens.current.push(token);
      if (inflightTokens.current.length > TOOLTIP_MAX_INFLIGHT) {
        const dropped = inflightTokens.current.shift();
        if (typeof dropped === 'number') {
          cancelledTokens.current.add(dropped);
          const droppedController = controllersRef.current.get(dropped);
          if (droppedController) {
            droppedController.abort();
            controllersRef.current.delete(dropped);
          }
        }
      }

      void projectionsV2Api
        .select(collectionId, projectionId, [selectedId], { signal: controller.signal })
        .then((response) => {
          if (
            cancelledTokens.current.has(token) ||
            controller.signal.aborted ||
            latestRequestToken.current !== token
          ) {
            return;
          }

          const item = response.data?.items?.[0];
          const metadata = toTooltipMetadata(item, selectedId);
          cacheRef.current.set(selectedId, { ...metadata, source: 'network' });
          setState({
            status: 'success',
            position: { x: point.x, y: point.y },
            metadata,
          });
        })
        .catch((error) => {
          if (cancelledTokens.current.has(token) || controller.signal.aborted || isRequestAborted(error)) {
            return;
          }
          cacheRef.current.set(selectedId, {
            selectedId,
            documentId: null,
            chunkIndex: null,
            contentPreview: null,
            source: 'network',
          });
          setState({
            status: 'error',
            position: { x: point.x, y: point.y },
            metadata: null,
            error: 'No metadata available',
          });
        })
        .finally(() => {
          cancelledTokens.current.delete(token);
          controllersRef.current.delete(token);
          inflightTokens.current = inflightTokens.current.filter((t) => t !== token);
        });
    },
    [collectionId, projectionId, ids]
  );

  const handleTooltip = useCallback(
    (value: DataPoint | null) => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
        debounceTimerRef.current = null;
      }

      if (!value) {
        clearState();
        return;
      }

      const point = value as DataPoint & { index?: number };
      const index = typeof point.index === 'number' ? point.index : null;
      if (index === null) {
        setState({
          status: 'error',
          position: { x: point.x, y: point.y },
          metadata: null,
          error: 'No metadata available',
        });
        return;
      }

      debounceTimerRef.current = setTimeout(() => {
        scheduleFetch({ ...point, index });
      }, TOOLTIP_DEBOUNCE_MS);
    },
    [clearState, scheduleFetch]
  );

  const handleTooltipLeave = useCallback(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
      debounceTimerRef.current = null;
    }
    clearState();
  }, [clearState]);

  return {
    tooltipState: state,
    handleTooltip,
    handleTooltipLeave,
    clearTooltipCache: () => cacheRef.current.clear(),
  };
}
