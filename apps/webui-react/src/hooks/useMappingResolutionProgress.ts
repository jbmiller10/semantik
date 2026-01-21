/**
 * useMappingResolutionProgress - WebSocket hook for real-time mapping resolution progress
 */

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useAuthStore } from '../stores/authStore';
import { operationsV2Api } from '../services/api/v2/operations';
import { useWebSocket } from './useWebSocket';
import { datasetKeys } from './useBenchmarks';

export interface MappingResolutionProgressState {
  stage: 'starting' | 'loading_documents' | 'resolving' | 'finalizing' | 'completed' | 'failed';
  totalRefs: number;
  processedRefs: number;
  resolvedRefs: number;
  ambiguousRefs: number;
  unresolvedRefs: number;
}

interface UseMappingResolutionProgressOptions {
  datasetId: string;
  onComplete?: () => void;
  onError?: (error: string) => void;
}

export function useMappingResolutionProgress(
  operationUuid: string | null,
  options: UseMappingResolutionProgressOptions | null
) {
  const queryClient = useQueryClient();
  const token = useAuthStore((state) => state.token);

  const [progress, setProgress] = useState<MappingResolutionProgressState>({
    stage: 'starting',
    totalRefs: 0,
    processedRefs: 0,
    resolvedRefs: 0,
    ambiguousRefs: 0,
    unresolvedRefs: 0,
  });

  const connectionInfo = useMemo(() => {
    if (!token || !operationUuid) return null;
    return operationsV2Api.getGlobalWebSocketConnectionInfo(token);
  }, [token, operationUuid]);

  const wsUrl = connectionInfo?.url ?? null;
  const wsProtocols = connectionInfo?.protocols;

  const handleMessage = useCallback(
    (event: MessageEvent) => {
      if (!operationUuid || !options) return;

      try {
        const rawMessage = JSON.parse(event.data);
        const payload = rawMessage.message || rawMessage;
        const type = payload.type;
        const data = payload.data || {};

        if (type !== 'benchmark_mapping_resolution_progress') {
          return;
        }

        const messageOperationId = data.operation_id;
        if (messageOperationId !== operationUuid) {
          return;
        }

        setProgress({
          stage: (data.stage as MappingResolutionProgressState['stage']) ?? 'resolving',
          totalRefs: Number(data.total_refs ?? 0),
          processedRefs: Number(data.processed_refs ?? 0),
          resolvedRefs: Number(data.resolved_refs ?? 0),
          ambiguousRefs: Number(data.ambiguous_refs ?? 0),
          unresolvedRefs: Number(data.unresolved_refs ?? 0),
        });

        if (data.stage === 'completed') {
          queryClient.invalidateQueries({ queryKey: datasetKeys.mappings(options.datasetId) });
          options.onComplete?.();
        } else if (data.stage === 'failed') {
          queryClient.invalidateQueries({ queryKey: datasetKeys.mappings(options.datasetId) });
          options.onError?.('Mapping resolution failed');
        }
      } catch {
        // Ignore parse errors
      }
    },
    [operationUuid, options, queryClient]
  );

  const { readyState } = useWebSocket(wsUrl, {
    onMessage: handleMessage,
    protocols: wsProtocols,
    autoReconnect: true,
  });

  // Reset progress when operation changes
  useEffect(() => {
    setProgress({
      stage: 'starting',
      totalRefs: 0,
      processedRefs: 0,
      resolvedRefs: 0,
      ambiguousRefs: 0,
      unresolvedRefs: 0,
    });
  }, [operationUuid]);

  return {
    progress,
    isConnected: readyState === WebSocket.OPEN,
  };
}

