import { useMemo } from 'react';
import { useUIStore } from '../stores/uiStore';
import { useAuthStore } from '../stores/authStore';
import { useUpdateOperationInCache } from './useCollectionOperations';
import { useUpdateCollectionInCache } from './useCollections';
import { useWebSocket } from './useWebSocket';
import type { OperationStatus } from '../types/collection';
import { operationsV2Api } from '../services/api/v2/operations';

/** Valid operation status values */
const VALID_OPERATION_STATUSES = ['pending', 'processing', 'completed', 'failed', 'cancelled'] as const;

/** Type guard to validate operation status */
function isValidOperationStatus(status: unknown): status is OperationStatus {
    return typeof status === 'string' && VALID_OPERATION_STATUSES.includes(status as OperationStatus);
}

export function useOperationsSocket() {
    const updateOperationInCache = useUpdateOperationInCache();
    const updateCollectionInCache = useUpdateCollectionInCache();
    const { addToast } = useUIStore();
    const token = useAuthStore((state) => state.token);

    // Get WebSocket connection info with authentication via subprotocol
    const connectionInfo = useMemo(() => {
        // If there is no token, do not attempt a connection
        if (!token) return null;
        return operationsV2Api.getGlobalWebSocketConnectionInfo(token);
    }, [token]);

    const wsUrl = connectionInfo?.url ?? null;
    const wsProtocols = connectionInfo?.protocols;

    // useWebSocket handles connection/reconnection when wsUrl changes
    const { readyState } = useWebSocket(wsUrl, {
        protocols: wsProtocols,
        onMessage: (event: MessageEvent) => {
            try {
                const rawMessage = JSON.parse(event.data);

                const payload = rawMessage.message || rawMessage;
                const type = payload.type;
                const data = payload.data || {};

                const operationId: string | undefined = data.operation_id ?? data.operationId;
                const collectionId: string | undefined = data.collection_id ?? data.collectionId;

                // Update collection stats immediately when available
                const stats = data.stats;
                if (
                    collectionId &&
                    stats &&
                    typeof stats === 'object' &&
                    typeof (stats as { document_count?: unknown }).document_count === 'number' &&
                    typeof (stats as { vector_count?: unknown }).vector_count === 'number'
                ) {
                    const updates: {
                        document_count: number;
                        vector_count: number;
                        total_size_bytes?: number;
                    } = {
                        document_count: (stats as { document_count: number }).document_count,
                        vector_count: (stats as { vector_count: number }).vector_count,
                    };
                    if (typeof (stats as { total_size_bytes?: unknown }).total_size_bytes === 'number') {
                        updates.total_size_bytes = (stats as { total_size_bytes: number }).total_size_bytes;
                    }
                    updateCollectionInCache(collectionId, updates);
                }

                let status: string | undefined;
                let progress: number | undefined;
                let error: string | undefined;

                if (type === 'operation_completed') {
                    status = 'completed';
                } else if (type === 'operation_failed') {
                    status = 'failed';
                    error = data.error_message || 'Operation failed';
                } else if (type === 'operation_started') {
                    status = 'processing';
                } else if (data.status) {
                    status = data.status;
                }

                if (data) {
                    progress = data.progress;
                    error = error || data.error || data.error_message;
                }

                // Derive progress from known message shapes when backend doesn't provide a unified progress field.
                if (progress === undefined) {
                    if (typeof data.progress_percent === 'number') {
                        progress = data.progress_percent;
                    } else if (
                        type === 'document_processed' &&
                        typeof data.processed === 'number' &&
                        typeof data.total === 'number' &&
                        data.total > 0
                    ) {
                        progress = (data.processed / data.total) * 100;
                    }
                }

                if (operationId) {
                    // Validate status before using it to prevent type confusion
                    if (status && isValidOperationStatus(status)) {
                        updateOperationInCache(
                            operationId,
                            { status: status, progress: progress },
                            collectionId // Pass collection_id from message for direct cache invalidation
                        );

                        if (status === 'failed') {
                            addToast({ type: 'error', message: error || `Operation ${operationId} failed` });
                        }
                    } else if (progress !== undefined) {
                        updateOperationInCache(operationId, { progress: progress }, collectionId);
                    }
                }
            } catch (error) {
                console.error('Failed to parse operation message:', error);
            }
        },
    });

    return {
        readyState,
    };
}
