import { useMemo } from 'react';
import { useUIStore } from '../stores/uiStore';
import { useAuthStore } from '../stores/authStore';
import { useUpdateOperationInCache } from './useCollectionOperations';
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

                const operationId = data.operation_id;

                if (!operationId) {
                    return;
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

                // Validate status before using it to prevent type confusion
                if (status && isValidOperationStatus(status)) {
                    updateOperationInCache(operationId, {
                        status: status,
                        progress: progress,
                    });

                    if (status === 'failed') {
                        addToast({ type: 'error', message: error || `Operation ${operationId} failed` });
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
