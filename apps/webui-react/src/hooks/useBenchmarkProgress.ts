/**
 * useBenchmarkProgress - WebSocket hook for real-time benchmark progress
 */

import { useState, useEffect, useMemo, useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { useAuthStore } from '../stores/authStore';
import { useWebSocket } from './useWebSocket';
import { benchmarkKeys } from './useBenchmarks';
import { operationsV2Api } from '../services/api/v2/operations';

export interface BenchmarkProgressState {
  totalRuns: number;
  completedRuns: number;
  currentRunOrder: number;
  currentRunConfig: Record<string, unknown> | null;
  stage: 'pending' | 'indexing' | 'evaluating' | 'completed' | 'failed';
  currentQueries: {
    total: number;
    processed: number;
  };
  recentMetrics: Array<{
    runOrder: number;
    config: Record<string, unknown>;
    metrics: Record<string, number>;
    timing: {
      search_ms: number | null;
      rerank_ms: number | null;
    };
  }>;
}

interface UseBenchmarkProgressOptions {
  onComplete?: () => void;
  onError?: (error: string) => void;
}

export function useBenchmarkProgress(
  benchmarkId: string | null,
  operationUuid: string | null,
  options: UseBenchmarkProgressOptions = {}
) {
  const queryClient = useQueryClient();
  const token = useAuthStore((state) => state.token);

  const [progress, setProgress] = useState<BenchmarkProgressState>({
    totalRuns: 0,
    completedRuns: 0,
    currentRunOrder: 0,
    currentRunConfig: null,
    stage: 'pending',
    currentQueries: { total: 0, processed: 0 },
    recentMetrics: [],
  });

  // Get WebSocket connection info
  const connectionInfo = useMemo(() => {
    if (!token || !operationUuid) return null;
    return operationsV2Api.getGlobalWebSocketConnectionInfo(token);
  }, [token, operationUuid]);

  const wsUrl = connectionInfo?.url ?? null;
  const wsProtocols = connectionInfo?.protocols;

  // Process incoming WebSocket messages
  const handleMessage = useCallback(
    (event: MessageEvent) => {
      try {
        const rawMessage = JSON.parse(event.data);
        const payload = rawMessage.message || rawMessage;
        const type = payload.type;
        const data = payload.data || {};

        // Filter for messages related to our benchmark
        const messageBenchmarkId = data.benchmark_id;
        const messageOperationId = data.operation_id;

        if (benchmarkId && messageBenchmarkId !== benchmarkId) {
          // Also check operation_id for backwards compatibility
          if (operationUuid && messageOperationId !== operationUuid) {
            return;
          }
        }

        // Handle different message types
        switch (type) {
          case 'benchmark_started':
            setProgress((prev) => ({
              ...prev,
              totalRuns: data.total_runs || prev.totalRuns,
              stage: 'evaluating',
            }));
            break;

          case 'benchmark_run_started':
            setProgress((prev) => ({
              ...prev,
              currentRunOrder: data.run_order || prev.currentRunOrder + 1,
              currentRunConfig: data.config || null,
              stage: 'evaluating',
              currentQueries: { total: data.total_queries || 0, processed: 0 },
            }));
            break;

          case 'benchmark_query_processed':
            setProgress((prev) => ({
              ...prev,
              currentQueries: {
                ...prev.currentQueries,
                processed: data.processed || prev.currentQueries.processed + 1,
              },
            }));
            break;

          case 'benchmark_run_completed':
            setProgress((prev) => {
              const newMetric = {
                runOrder: data.run_order || prev.currentRunOrder,
                config: data.config || prev.currentRunConfig || {},
                metrics: data.metrics || {},
                timing: {
                  search_ms: data.timing?.search_ms ?? null,
                  rerank_ms: data.timing?.rerank_ms ?? null,
                },
              };

              return {
                ...prev,
                completedRuns: data.completed_runs || prev.completedRuns + 1,
                recentMetrics: [...prev.recentMetrics.slice(-9), newMetric],
              };
            });

            // Invalidate results cache when a run completes
            if (benchmarkId) {
              queryClient.invalidateQueries({
                queryKey: benchmarkKeys.results(benchmarkId),
              });
            }
            break;

          case 'benchmark_completed':
          case 'operation_completed':
            setProgress((prev) => ({
              ...prev,
              stage: 'completed',
            }));

            // Invalidate benchmark queries
            queryClient.invalidateQueries({ queryKey: benchmarkKeys.lists() });
            if (benchmarkId) {
              queryClient.invalidateQueries({
                queryKey: benchmarkKeys.detail(benchmarkId),
              });
              queryClient.invalidateQueries({
                queryKey: benchmarkKeys.results(benchmarkId),
              });
            }

            options.onComplete?.();
            break;

          case 'benchmark_failed':
          case 'operation_failed':
            setProgress((prev) => ({
              ...prev,
              stage: 'failed',
            }));

            queryClient.invalidateQueries({ queryKey: benchmarkKeys.lists() });
            if (benchmarkId) {
              queryClient.invalidateQueries({
                queryKey: benchmarkKeys.detail(benchmarkId),
              });
            }

            options.onError?.(data.error_message || 'Benchmark failed');
            break;

          default:
            // Handle generic progress updates
            if (data.progress !== undefined || data.progress_percent !== undefined) {
              const progressPercent = data.progress ?? data.progress_percent;
              if (progress.totalRuns > 0) {
                const estimatedCompleted = Math.floor(
                  (progressPercent / 100) * progress.totalRuns
                );
                setProgress((prev) => ({
                  ...prev,
                  completedRuns: Math.max(prev.completedRuns, estimatedCompleted),
                }));
              }
            }
        }
      } catch (error) {
        console.error('Failed to parse benchmark progress message:', error);
      }
    },
    [benchmarkId, operationUuid, queryClient, options, progress.totalRuns]
  );

  // Connect to WebSocket
  const { readyState } = useWebSocket(wsUrl, {
    protocols: wsProtocols,
    onMessage: handleMessage,
  });

  // Reset progress when benchmark changes
  useEffect(() => {
    setProgress({
      totalRuns: 0,
      completedRuns: 0,
      currentRunOrder: 0,
      currentRunConfig: null,
      stage: 'pending',
      currentQueries: { total: 0, processed: 0 },
      recentMetrics: [],
    });
  }, [benchmarkId, operationUuid]);

  return {
    progress,
    isConnected: readyState === WebSocket.OPEN,
  };
}
