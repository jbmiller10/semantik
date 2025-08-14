import { useEffect, useState, useCallback, useRef } from 'react';
import { 
  getChunkingWebSocket, 
  disconnectChunkingWebSocket,
  ChunkingMessageType,
  type WebSocketMessage,
  type ChunkingProgressData,
  type ChunkingChunkData,
  type ChunkingCompleteData,
  type ChunkingErrorData,
} from '../services/websocket';
import type { ChunkPreview, ChunkingStatistics } from '../types/chunking';

/**
 * Connection status for UI display
 */
export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error' | 'reconnecting';

/**
 * Performance data type from ChunkingCompleteData
 */
interface ChunkingPerformance {
  processingTimeMs: number;
  estimatedFullProcessingTimeMs: number;
}

/**
 * Strategy configuration for comparison
 */
interface ComparisonStrategy {
  strategy: string;
  configuration: Record<string, number | boolean | string>;
}

/**
 * Hook options
 */
interface UseChunkingWebSocketOptions {
  autoConnect?: boolean;
  onChunkReceived?: (chunk: ChunkPreview, index: number, total: number) => void;
  onProgressUpdate?: (progress: ChunkingProgressData) => void;
  onComplete?: (statistics: ChunkingStatistics, performance: ChunkingPerformance) => void;
  onError?: (error: ChunkingErrorData) => void;
  requestId?: string;
}

/**
 * Hook return type
 */
interface UseChunkingWebSocketReturn {
  // Connection management
  connectionStatus: ConnectionStatus;
  connect: () => void;
  disconnect: () => void;
  isConnected: boolean;
  reconnectAttempts: number;
  
  // Real-time data
  chunks: ChunkPreview[];
  progress: ChunkingProgressData | null;
  statistics: ChunkingStatistics | null;
  performance: ChunkingPerformance | null;
  error: ChunkingErrorData | null;
  
  // Actions
  startPreview: (documentId: string, strategy: string, configuration: Record<string, number | boolean | string>) => void;
  startComparison: (documentId: string, strategies: ComparisonStrategy[]) => void;
  clearData: () => void;
}

/**
 * Custom hook for chunking WebSocket integration
 * Provides real-time updates for chunking preview and comparison
 */
export function useChunkingWebSocket(
  options: UseChunkingWebSocketOptions = {}
): UseChunkingWebSocketReturn {
  const {
    autoConnect = true,
    onChunkReceived,
    onProgressUpdate,
    onComplete,
    onError,
    requestId,
  } = options;

  // State management
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected');
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [chunks, setChunks] = useState<ChunkPreview[]>([]);
  const [progress, setProgress] = useState<ChunkingProgressData | null>(null);
  const [statistics, setStatistics] = useState<ChunkingStatistics | null>(null);
  const [performance, setPerformance] = useState<ChunkingPerformance | null>(null);
  const [error, setError] = useState<ChunkingErrorData | null>(null);
  
  // Refs for stable callbacks
  const wsRef = useRef<ReturnType<typeof getChunkingWebSocket> | null>(null);
  const activeRequestId = useRef<string | null>(null);
  const listenersRef = useRef<Map<string, (...args: unknown[]) => void>>(new Map());

  /**
   * Initialize WebSocket connection
   */
  const connect = useCallback(() => {
    if (wsRef.current?.isConnected()) {
      console.log('WebSocket already connected');
      return;
    }

    console.log('Initializing chunking WebSocket connection');
    const ws = getChunkingWebSocket();
    wsRef.current = ws;

    // Clear any existing listeners
    listenersRef.current.clear();

    // Set up event listeners and store references
    const connectedHandler = () => {
      console.log('Chunking WebSocket connected');
      setConnectionStatus('connected');
      setReconnectAttempts(0);
      setError(null);
    };
    ws.on('connected', connectedHandler);
    listenersRef.current.set('connected', connectedHandler);

    const disconnectedHandler = (event: CloseEvent) => {
      console.log('Chunking WebSocket disconnected:', event.code, event.reason);
      setConnectionStatus('disconnected');
      
      // Clear active data on unexpected disconnect
      if (event.code !== 1000) {
        setProgress(null);
      }
    };
    ws.on('disconnected', disconnectedHandler);
    listenersRef.current.set('disconnected', disconnectedHandler);

    const reconnectingHandler = ({ attempt }: { attempt: number }) => {
      console.log(`Chunking WebSocket reconnecting (attempt ${attempt})`);
      setConnectionStatus('reconnecting');
      setReconnectAttempts(attempt);
    };
    ws.on('reconnecting', reconnectingHandler);
    listenersRef.current.set('reconnecting', reconnectingHandler);

    const reconnectFailedHandler = ({ attempts }: { attempts: number }) => {
      console.error(`Chunking WebSocket reconnection failed after ${attempts} attempts`);
      setConnectionStatus('error');
      setError({
        message: 'Failed to reconnect to server',
        code: 'RECONNECT_FAILED',
      });
    };
    ws.on('reconnect_failed', reconnectFailedHandler);
    listenersRef.current.set('reconnect_failed', reconnectFailedHandler);

    const errorHandler = (errorData: ChunkingErrorData) => {
      console.error('Chunking WebSocket error:', errorData);
      setConnectionStatus('error');
      setError({
        message: errorData.message || 'WebSocket error occurred',
        code: errorData.code,
        details: errorData.details,
      });
      onError?.(errorData);
    };
    ws.on('error', errorHandler);
    listenersRef.current.set('error', errorHandler);

    // Handle chunking-specific messages
    const previewStartHandler = (data: unknown, message: WebSocketMessage) => {
      if (message.requestId === activeRequestId.current || !activeRequestId.current) {
        console.log('Preview started:', data);
        setChunks([]);
        setProgress(null);
        setStatistics(null);
        setPerformance(null);
        setError(null);
      }
    };
    ws.on(ChunkingMessageType.PREVIEW_START, previewStartHandler);
    listenersRef.current.set(ChunkingMessageType.PREVIEW_START, previewStartHandler);

    const previewProgressHandler = (data: ChunkingProgressData, message: WebSocketMessage) => {
      if (message.requestId === activeRequestId.current || !activeRequestId.current) {
        console.log('Preview progress:', data);
        setProgress(data);
        onProgressUpdate?.(data);
      }
    };
    ws.on(ChunkingMessageType.PREVIEW_PROGRESS, previewProgressHandler);
    listenersRef.current.set(ChunkingMessageType.PREVIEW_PROGRESS, previewProgressHandler);

    const previewChunkHandler = (data: ChunkingChunkData, message: WebSocketMessage) => {
      if (message.requestId === activeRequestId.current || !activeRequestId.current) {
        console.log('Chunk received:', data.index, '/', data.total);
        
        // Add chunk to array
        setChunks(prev => {
          const newChunks = [...prev];
          newChunks[data.index] = data.chunk;
          return newChunks;
        });
        
        // Update progress
        setProgress({
          percentage: Math.round(((data.index + 1) / data.total) * 100),
          currentChunk: data.index + 1,
          totalChunks: data.total,
        });
        
        onChunkReceived?.(data.chunk, data.index, data.total);
      }
    };
    ws.on(ChunkingMessageType.PREVIEW_CHUNK, previewChunkHandler);
    listenersRef.current.set(ChunkingMessageType.PREVIEW_CHUNK, previewChunkHandler);

    const previewCompleteHandler = (data: ChunkingCompleteData, message: WebSocketMessage) => {
      if (message.requestId === activeRequestId.current || !activeRequestId.current) {
        console.log('Preview complete:', data);
        // Ensure statistics has all required fields
        const statistics: ChunkingStatistics = {
          ...data.statistics,
          sizeDistribution: data.statistics.sizeDistribution || []
        };
        setStatistics(statistics);
        setPerformance(data.performance);
        setProgress(null);
        onComplete?.(statistics, data.performance);
      }
    };
    ws.on(ChunkingMessageType.PREVIEW_COMPLETE, previewCompleteHandler);
    listenersRef.current.set(ChunkingMessageType.PREVIEW_COMPLETE, previewCompleteHandler);

    const previewErrorHandler = (data: ChunkingErrorData, message: WebSocketMessage) => {
      if (message.requestId === activeRequestId.current || !activeRequestId.current) {
        console.error('Preview error:', data);
        setError(data);
        setProgress(null);
        onError?.(data);
      }
    };
    ws.on(ChunkingMessageType.PREVIEW_ERROR, previewErrorHandler);
    listenersRef.current.set(ChunkingMessageType.PREVIEW_ERROR, previewErrorHandler);

    // Handle comparison messages
    const comparisonStartHandler = (data: unknown, message: WebSocketMessage) => {
      if (message.requestId === activeRequestId.current || !activeRequestId.current) {
        console.log('Comparison started:', data);
        setError(null);
      }
    };
    ws.on(ChunkingMessageType.COMPARISON_START, comparisonStartHandler);
    listenersRef.current.set(ChunkingMessageType.COMPARISON_START, comparisonStartHandler);

    const comparisonProgressHandler = (data: ChunkingProgressData & { currentStrategy?: number; totalStrategies?: number; estimatedTimeRemaining?: number }, message: WebSocketMessage) => {
      if (message.requestId === activeRequestId.current || !activeRequestId.current) {
        console.log('Comparison progress:', data);
        setProgress({
          percentage: data.percentage,
          currentChunk: data.currentStrategy || 0,
          totalChunks: data.totalStrategies || 0,
          estimatedTimeRemaining: data.estimatedTimeRemaining,
        });
      }
    };
    ws.on(ChunkingMessageType.COMPARISON_PROGRESS, comparisonProgressHandler);
    listenersRef.current.set(ChunkingMessageType.COMPARISON_PROGRESS, comparisonProgressHandler);

    const comparisonCompleteHandler = (data: unknown, message: WebSocketMessage) => {
      if (message.requestId === activeRequestId.current || !activeRequestId.current) {
        console.log('Comparison complete:', data);
        setProgress(null);
      }
    };
    ws.on(ChunkingMessageType.COMPARISON_COMPLETE, comparisonCompleteHandler);
    listenersRef.current.set(ChunkingMessageType.COMPARISON_COMPLETE, comparisonCompleteHandler);

    const comparisonErrorHandler = (data: ChunkingErrorData, message: WebSocketMessage) => {
      if (message.requestId === activeRequestId.current || !activeRequestId.current) {
        console.error('Comparison error:', data);
        setError(data);
        setProgress(null);
        onError?.(data);
      }
    };
    ws.on(ChunkingMessageType.COMPARISON_ERROR, comparisonErrorHandler);
    listenersRef.current.set(ChunkingMessageType.COMPARISON_ERROR, comparisonErrorHandler);

    // Connect to WebSocket
    setConnectionStatus('connecting');
    ws.connect();
  }, [onChunkReceived, onProgressUpdate, onComplete, onError]);

  /**
   * Disconnect from WebSocket
   */
  const disconnect = useCallback(() => {
    console.log('Disconnecting chunking WebSocket');
    
    // Remove all event listeners before disconnecting
    if (wsRef.current) {
      listenersRef.current.forEach((handler, event) => {
        wsRef.current?.removeListener(event, handler);
      });
      listenersRef.current.clear();
    }
    
    disconnectChunkingWebSocket();
    wsRef.current = null;
    setConnectionStatus('disconnected');
    setReconnectAttempts(0);
  }, []);

  /**
   * Start chunking preview
   */
  const startPreview = useCallback((
    documentId: string,
    strategy: string,
    configuration: Record<string, number | boolean | string>
  ) => {
    if (!wsRef.current?.isConnected()) {
      console.error('WebSocket not connected');
      setError({
        message: 'WebSocket not connected. Please wait for connection.',
        code: 'NOT_CONNECTED',
      });
      return;
    }

    const newRequestId = requestId || `preview-${Date.now()}`;
    activeRequestId.current = newRequestId;

    // Clear previous data
    setChunks([]);
    setProgress(null);
    setStatistics(null);
    setPerformance(null);
    setError(null);

    // Send preview request
    wsRef.current.send({
      type: 'preview_request',
      data: {
        documentId,
        strategy,
        configuration,
      },
      timestamp: Date.now(),
      requestId: newRequestId,
    });
  }, [requestId]);

  /**
   * Start chunking comparison
   */
  const startComparison = useCallback((
    documentId: string,
    strategies: ComparisonStrategy[]
  ) => {
    if (!wsRef.current?.isConnected()) {
      console.error('WebSocket not connected');
      setError({
        message: 'WebSocket not connected. Please wait for connection.',
        code: 'NOT_CONNECTED',
      });
      return;
    }

    const newRequestId = requestId || `comparison-${Date.now()}`;
    activeRequestId.current = newRequestId;

    // Clear previous data
    setProgress(null);
    setError(null);

    // Send comparison request
    wsRef.current.send({
      type: 'comparison_request',
      data: {
        documentId,
        strategies,
      },
      timestamp: Date.now(),
      requestId: newRequestId,
    });
  }, [requestId]);

  /**
   * Clear all data
   */
  const clearData = useCallback(() => {
    setChunks([]);
    setProgress(null);
    setStatistics(null);
    setPerformance(null);
    setError(null);
    activeRequestId.current = null;
  }, []);

  /**
   * Auto-connect on mount if enabled
   */
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      // Clean up on unmount
      if (wsRef.current) {
        disconnect();
      }
    };
  }, [autoConnect, connect, disconnect]);

  return {
    // Connection management
    connectionStatus,
    connect,
    disconnect,
    isConnected: connectionStatus === 'connected',
    reconnectAttempts,
    
    // Real-time data
    chunks,
    progress,
    statistics,
    performance,
    error,
    
    // Actions
    startPreview,
    startComparison,
    clearData,
  };
}