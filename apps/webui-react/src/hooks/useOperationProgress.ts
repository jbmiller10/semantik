import { useEffect, useRef } from 'react';
import { useWebSocket } from './useWebSocket';
import { useUpdateOperationInCache } from './useCollectionOperations';
import { useUIStore } from '../stores/uiStore';
import { operationsV2Api } from '../services/api/v2/operations';
import type { OperationProgressMessage } from '../types/collection';

interface UseOperationProgressOptions {
  onComplete?: () => void;
  onError?: (error: string) => void;
  showToasts?: boolean;
}

export function useOperationProgress(
  operationId: string | null,
  options: UseOperationProgressOptions = {}
) {
  const updateOperationInCache = useUpdateOperationInCache();
  const { addToast } = useUIStore();
  const { onComplete, onError, showToasts = true } = options;
  
  // Track if we've already shown completion toast to avoid duplicates
  const hasShownComplete = useRef(false);
  
  // Construct WebSocket URL with authentication token
  const wsUrl = operationId ? operationsV2Api.getWebSocketUrl(operationId) : null;
  
  const { sendMessage, readyState } = useWebSocket(wsUrl, {
    onMessage: (event) => {
      try {
        const message: OperationProgressMessage = JSON.parse(event.data);
        
        // Update operation in React Query cache
        updateOperationInCache(message.operation_id, {
          status: message.status,
          progress: message.progress,
        });
        
        // Handle status-specific actions
        switch (message.status) {
          case 'processing':
            if (message.message && showToasts) {
              // Only show processing messages for significant updates
              if (message.progress && message.progress % 25 === 0) {
                addToast({
                  type: 'info',
                  message: message.message,
                });
              }
            }
            break;
            
          case 'completed':
            if (!hasShownComplete.current) {
              hasShownComplete.current = true;
              if (showToasts) {
                addToast({
                  type: 'success',
                  message: message.message || 'Operation completed successfully',
                });
              }
              onComplete?.();
            }
            break;
            
          case 'failed':
            if (showToasts) {
              addToast({
                type: 'error',
                message: message.error || 'Operation failed',
              });
            }
            onError?.(message.error || 'Operation failed');
            break;
            
          case 'cancelled':
            if (showToasts) {
              addToast({
                type: 'warning',
                message: 'Operation was cancelled',
              });
            }
            break;
        }
        
        // Handle metadata updates (e.g., ETA, current file)
        if (message.metadata) {
          updateOperationInCache(message.operation_id, {
            eta: message.metadata.eta,
            // Add any other metadata fields as needed
          });
        }
      } catch (error) {
        console.error('Failed to parse operation progress message:', error);
      }
    },
    onError: (event) => {
      console.error('WebSocket error for operation', operationId, event);
      if (showToasts) {
        addToast({
          type: 'error',
          message: 'Lost connection to operation progress updates',
        });
      }
    },
    onClose: () => {
      console.log('WebSocket closed for operation', operationId);
    },
  });
  
  // Reset completion tracking when operation changes
  useEffect(() => {
    hasShownComplete.current = false;
  }, [operationId]);
  
  return {
    sendMessage,
    readyState,
    isConnected: readyState === WebSocket.OPEN,
  };
}