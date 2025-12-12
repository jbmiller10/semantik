import { useMemo } from 'react';
import { useUIStore } from '../stores/uiStore';
import { useAuthStore } from '../stores/authStore';
import { useUpdateOperationInCache } from './useCollectionOperations';
import { useWebSocket } from './useWebSocket';
import type { OperationStatus } from '../types/collection';
import { operationsV2Api } from '../services/api/v2/operations';

export function useOperationsSocket() {
    const updateOperationInCache = useUpdateOperationInCache();
    const { addToast } = useUIStore();
    const token = useAuthStore((state) => state.token);

    const wsUrl = useMemo(() => {
        // If there is no token, do not attempt a connection
        if (!token) return null;
        return operationsV2Api.getGlobalWebSocketUrl(token);
    }, [token]);

    // useWebSocket handles connection/reconnection when wsUrl changes
    const { readyState } = useWebSocket(wsUrl, {
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

                if (status) {
                    updateOperationInCache(operationId, {
                        status: status as OperationStatus,
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
