import { useEffect, useRef } from 'react';
import { useWebSocket } from './useWebSocket';
import { useUpdateOperationInCache } from './useCollectionOperations';
import { useUIStore } from '../stores/uiStore';
import { operationsV2Api } from '../services/api/v2/operations';

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
        // Parse the raw message from the backend
        const rawMessage = JSON.parse(event.data);
        
        // Handle the backend's message format which has type and data fields
        let status: string | undefined;
        let progress: number | undefined;
        let messageText: string | undefined;
        let error: string | undefined;
        
        // Extract status based on message type
        if (rawMessage.type === 'operation_completed') {
          status = 'completed';
        } else if (rawMessage.type === 'operation_failed') {
          status = 'failed';
          error = rawMessage.data?.error_message || 'Operation failed';
        } else if (rawMessage.type === 'operation_started') {
          status = 'processing';
        } else if (rawMessage.type === 'current_state') {
          // Handle initial state message from websocket connection
          status = rawMessage.data?.status;
        } else if (rawMessage.data?.status) {
          // Generic status update
          status = rawMessage.data.status;
        }
        
        // Extract other fields from data
        if (rawMessage.data) {
          progress = rawMessage.data.progress;
          messageText = rawMessage.data.message;
          error = error || rawMessage.data.error || rawMessage.data.error_message;
        }
        
        // Only proceed if we have a status to work with
        if (status && operationId) {
          // Update operation in React Query cache
          updateOperationInCache(operationId, {
            status: status as any,
            progress: progress,
          });
          
          // Handle status-specific actions
          switch (status) {
            case 'processing':
              if (messageText && showToasts) {
                // Only show processing messages for significant updates
                if (progress && progress % 25 === 0) {
                  addToast({
                    type: 'info',
                    message: messageText,
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
                    message: messageText || 'Operation completed successfully',
                  });
                }
                onComplete?.();
              }
              break;
              
            case 'failed':
              if (showToasts) {
                addToast({
                  type: 'error',
                  message: error || 'Operation failed',
                });
              }
              onError?.(error || 'Operation failed');
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
          if (rawMessage.data?.metadata) {
            updateOperationInCache(operationId, {
              eta: rawMessage.data.metadata.eta,
              // Add any other metadata fields as needed
            });
          }
        }
      } catch (error) {
        console.error('Failed to parse operation progress message:', error, event.data);
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