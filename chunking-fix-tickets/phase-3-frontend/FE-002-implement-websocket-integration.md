# FE-002: Implement WebSocket Integration

## Ticket Information
- **Priority**: CRITICAL
- **Estimated Time**: 4 hours
- **Dependencies**: FE-001 (Real API must be working first)
- **Risk Level**: HIGH - Real-time features non-functional
- **Affected Files**:
  - `apps/webui-react/src/hooks/useChunkingWebSocket.ts` (new)
  - `apps/webui-react/src/stores/chunkingStore.ts`
  - `apps/webui-react/src/components/chunking/ChunkingPreviewPanel.tsx`
  - `apps/webui-react/src/services/websocket.ts` (new)

## Context

The chunking components claim to have "real-time" updates but don't actually use WebSockets. Updates are either mocked or use polling. This means users don't see live progress during chunking operations.

### Current Problems
- No WebSocket hook for chunking operations
- Preview panel doesn't receive real-time chunk updates
- Progress bars don't update smoothly
- No reconnection logic for dropped connections

## Requirements

1. Create WebSocket service with reconnection logic
2. Implement chunking-specific WebSocket hook
3. Update components to use real-time data
4. Add connection status indicators
5. Implement graceful degradation
6. Handle WebSocket authentication

## Technical Details

### 1. Create WebSocket Service

```typescript
// apps/webui-react/src/services/websocket.ts

import { EventEmitter } from 'events';

export interface WebSocketConfig {
  url: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  protocols?: string[];
}

export interface WebSocketMessage {
  type: string;
  channel?: string;
  data: any;
  timestamp: number;
  correlation_id?: string;
}

export enum ConnectionState {
  CONNECTING = 'connecting',
  CONNECTED = 'connected',
  DISCONNECTED = 'disconnected',
  RECONNECTING = 'reconnecting',
  ERROR = 'error'
}

export class WebSocketService extends EventEmitter {
  private ws: WebSocket | null = null;
  private config: WebSocketConfig;
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private messageQueue: WebSocketMessage[] = [];
  private subscriptions = new Map<string, Set<string>>();
  private connectionState: ConnectionState = ConnectionState.DISCONNECTED;
  private lastPingTime: number = 0;
  private latency: number = 0;

  constructor(config: WebSocketConfig) {
    super();
    this.config = {
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000,
      ...config
    };
  }

  public connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      this.setConnectionState(ConnectionState.CONNECTING);

      try {
        // Add auth token to URL
        const token = localStorage.getItem('auth_token');
        const url = new URL(this.config.url);
        if (token) {
          url.searchParams.set('token', token);
        }

        this.ws = new WebSocket(url.toString(), this.config.protocols);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.setConnectionState(ConnectionState.CONNECTED);
          this.reconnectAttempts = 0;
          
          // Send queued messages
          this.flushMessageQueue();
          
          // Start heartbeat
          this.startHeartbeat();
          
          // Resubscribe to channels
          this.resubscribeToChannels();
          
          this.emit('connected');
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.setConnectionState(ConnectionState.ERROR);
          this.emit('error', error);
          reject(error);
        };

        this.ws.onclose = (event) => {
          console.log('WebSocket closed:', event.code, event.reason);
          this.setConnectionState(ConnectionState.DISCONNECTED);
          this.stopHeartbeat();
          
          if (!event.wasClean) {
            this.scheduleReconnect();
          }
          
          this.emit('disconnected', event);
        };

      } catch (error) {
        console.error('Failed to create WebSocket:', error);
        this.setConnectionState(ConnectionState.ERROR);
        reject(error);
      }
    });
  }

  public disconnect(): void {
    this.stopReconnect();
    this.stopHeartbeat();
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnecting');
      this.ws = null;
    }
    
    this.setConnectionState(ConnectionState.DISCONNECTED);
  }

  public send(message: WebSocketMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      // Queue message for later
      this.messageQueue.push(message);
    }
  }

  public subscribe(channel: string, eventTypes: string[] = ['*']): void {
    if (!this.subscriptions.has(channel)) {
      this.subscriptions.set(channel, new Set());
    }
    
    eventTypes.forEach(type => {
      this.subscriptions.get(channel)!.add(type);
    });

    // Send subscription message
    this.send({
      type: 'subscribe',
      channel,
      data: { event_types: eventTypes },
      timestamp: Date.now()
    });
  }

  public unsubscribe(channel: string): void {
    this.subscriptions.delete(channel);
    
    this.send({
      type: 'unsubscribe',
      channel,
      data: {},
      timestamp: Date.now()
    });
  }

  private handleMessage(message: WebSocketMessage): void {
    // Handle system messages
    switch (message.type) {
      case 'pong':
        this.handlePong(message);
        return;
      case 'error':
        this.emit('error', message.data);
        return;
      case 'authenticated':
        this.emit('authenticated');
        return;
    }

    // Emit channel-specific events
    if (message.channel) {
      this.emit(`channel:${message.channel}`, message);
      
      // Emit type-specific events
      this.emit(`channel:${message.channel}:${message.type}`, message.data);
    }

    // Emit global message event
    this.emit('message', message);
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.lastPingTime = Date.now();
        this.send({
          type: 'ping',
          data: { timestamp: this.lastPingTime },
          timestamp: this.lastPingTime
        });
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private handlePong(message: WebSocketMessage): void {
    const now = Date.now();
    this.latency = now - this.lastPingTime;
    this.emit('latency', this.latency);
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts!) {
      console.error('Max reconnection attempts reached');
      this.emit('max_reconnect_attempts');
      return;
    }

    this.setConnectionState(ConnectionState.RECONNECTING);
    this.reconnectAttempts++;
    
    const delay = Math.min(
      this.config.reconnectInterval! * Math.pow(1.5, this.reconnectAttempts - 1),
      30000
    );
    
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    this.reconnectTimer = setTimeout(() => {
      this.connect().catch(error => {
        console.error('Reconnection failed:', error);
      });
    }, delay);
  }

  private stopReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.reconnectAttempts = 0;
  }

  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift()!;
      this.send(message);
    }
  }

  private resubscribeToChannels(): void {
    this.subscriptions.forEach((eventTypes, channel) => {
      this.send({
        type: 'subscribe',
        channel,
        data: { event_types: Array.from(eventTypes) },
        timestamp: Date.now()
      });
    });
  }

  private setConnectionState(state: ConnectionState): void {
    this.connectionState = state;
    this.emit('connection_state_changed', state);
  }

  public getConnectionState(): ConnectionState {
    return this.connectionState;
  }

  public getLatency(): number {
    return this.latency;
  }

  public isConnected(): boolean {
    return this.connectionState === ConnectionState.CONNECTED;
  }
}

// Singleton instance
let wsInstance: WebSocketService | null = null;

export function getWebSocketService(): WebSocketService {
  if (!wsInstance) {
    const wsUrl = process.env.REACT_APP_WS_URL || 
                  `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;
    
    wsInstance = new WebSocketService({
      url: wsUrl,
      reconnectInterval: 3000,
      maxReconnectAttempts: 10,
      heartbeatInterval: 30000
    });
  }
  
  return wsInstance;
}
```

### 2. Create Chunking WebSocket Hook

```typescript
// apps/webui-react/src/hooks/useChunkingWebSocket.ts

import { useEffect, useState, useCallback, useRef } from 'react';
import { getWebSocketService, ConnectionState, WebSocketMessage } from '@/services/websocket';

export interface ChunkingProgress {
  operation_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  current_chunk: number;
  total_chunks: number;
  message?: string;
  error?: string;
}

export interface ChunkData {
  chunk_id: string;
  chunk_index: number;
  content: string;
  metadata: Record<string, any>;
  size: number;
}

export interface UseChunkingWebSocketReturn {
  connect: (operationId: string) => void;
  disconnect: () => void;
  connectionState: ConnectionState;
  progress: ChunkingProgress | null;
  chunks: ChunkData[];
  error: Error | null;
  latency: number;
  clearChunks: () => void;
}

export function useChunkingWebSocket(): UseChunkingWebSocketReturn {
  const [connectionState, setConnectionState] = useState<ConnectionState>(ConnectionState.DISCONNECTED);
  const [progress, setProgress] = useState<ChunkingProgress | null>(null);
  const [chunks, setChunks] = useState<ChunkData[]>([]);
  const [error, setError] = useState<Error | null>(null);
  const [latency, setLatency] = useState(0);
  
  const wsService = useRef(getWebSocketService());
  const currentOperationId = useRef<string | null>(null);

  const connect = useCallback((operationId: string) => {
    currentOperationId.current = operationId;
    setError(null);
    setProgress(null);
    setChunks([]);

    const ws = wsService.current;

    // Set up event listeners
    const handleConnectionChange = (state: ConnectionState) => {
      setConnectionState(state);
    };

    const handleProgress = (data: ChunkingProgress) => {
      if (data.operation_id === currentOperationId.current) {
        setProgress(data);
      }
    };

    const handleChunk = (data: ChunkData) => {
      setChunks(prev => [...prev, data]);
    };

    const handleError = (error: any) => {
      setError(new Error(error.message || 'WebSocket error'));
    };

    const handleLatency = (latency: number) => {
      setLatency(latency);
    };

    const handleComplete = (data: any) => {
      if (data.operation_id === currentOperationId.current) {
        setProgress({
          operation_id: data.operation_id,
          status: 'completed',
          progress: 100,
          current_chunk: data.total_chunks,
          total_chunks: data.total_chunks,
          message: 'Chunking completed successfully'
        });
      }
    };

    // Subscribe to events
    ws.on('connection_state_changed', handleConnectionChange);
    ws.on(`channel:chunking:${operationId}:progress`, handleProgress);
    ws.on(`channel:chunking:${operationId}:chunk`, handleChunk);
    ws.on(`channel:chunking:${operationId}:complete`, handleComplete);
    ws.on(`channel:chunking:${operationId}:error`, handleError);
    ws.on('error', handleError);
    ws.on('latency', handleLatency);

    // Connect and subscribe
    ws.connect()
      .then(() => {
        ws.subscribe(`chunking:${operationId}`, ['progress', 'chunk', 'complete', 'error']);
      })
      .catch(error => {
        setError(error);
      });

    // Cleanup function
    return () => {
      ws.off('connection_state_changed', handleConnectionChange);
      ws.off(`channel:chunking:${operationId}:progress`, handleProgress);
      ws.off(`channel:chunking:${operationId}:chunk`, handleChunk);
      ws.off(`channel:chunking:${operationId}:complete`, handleComplete);
      ws.off(`channel:chunking:${operationId}:error`, handleError);
      ws.off('error', handleError);
      ws.off('latency', handleLatency);
      
      if (currentOperationId.current) {
        ws.unsubscribe(`chunking:${currentOperationId.current}`);
      }
    };
  }, []);

  const disconnect = useCallback(() => {
    if (currentOperationId.current) {
      wsService.current.unsubscribe(`chunking:${currentOperationId.current}`);
      currentOperationId.current = null;
    }
  }, []);

  const clearChunks = useCallback(() => {
    setChunks([]);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    connect,
    disconnect,
    connectionState,
    progress,
    chunks,
    error,
    latency,
    clearChunks
  };
}
```

### 3. Update ChunkingPreviewPanel to Use WebSocket

```typescript
// apps/webui-react/src/components/chunking/ChunkingPreviewPanel.tsx

import React, { useEffect, useState } from 'react';
import { useChunkingWebSocket, ConnectionState } from '@/hooks/useChunkingWebSocket';
import { useChunkingStore } from '@/stores/chunkingStore';

export const ChunkingPreviewPanel: React.FC = () => {
  const {
    connect,
    disconnect,
    connectionState,
    progress,
    chunks,
    error,
    latency
  } = useChunkingWebSocket();

  const {
    currentOperationId,
    startChunking,
    cancelChunking
  } = useChunkingStore();

  // Connect when operation starts
  useEffect(() => {
    if (currentOperationId) {
      connect(currentOperationId);
      
      return () => {
        disconnect();
      };
    }
  }, [currentOperationId, connect, disconnect]);

  const handleStartChunking = async () => {
    try {
      const operationId = await startChunking();
      // WebSocket connection will be established via useEffect
    } catch (error) {
      console.error('Failed to start chunking:', error);
    }
  };

  const handleCancelChunking = () => {
    cancelChunking();
    disconnect();
  };

  // Connection status indicator
  const renderConnectionStatus = () => {
    const statusColors = {
      [ConnectionState.CONNECTED]: 'bg-green-500',
      [ConnectionState.CONNECTING]: 'bg-yellow-500',
      [ConnectionState.RECONNECTING]: 'bg-orange-500',
      [ConnectionState.DISCONNECTED]: 'bg-gray-500',
      [ConnectionState.ERROR]: 'bg-red-500'
    };

    return (
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${statusColors[connectionState]}`} />
        <span className="text-sm text-gray-600">
          {connectionState}
          {connectionState === ConnectionState.CONNECTED && latency > 0 && (
            <span className="ml-2 text-xs">({latency}ms)</span>
          )}
        </span>
      </div>
    );
  };

  // Progress bar with real-time updates
  const renderProgressBar = () => {
    if (!progress) return null;

    return (
      <div className="w-full">
        <div className="flex justify-between mb-2">
          <span className="text-sm font-medium">
            {progress.message || `Processing chunk ${progress.current_chunk} of ${progress.total_chunks}`}
          </span>
          <span className="text-sm text-gray-500">
            {progress.progress.toFixed(1)}%
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div 
            className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
            style={{ width: `${progress.progress}%` }}
          />
        </div>
        {progress.status === 'failed' && (
          <div className="mt-2 text-red-500 text-sm">
            Error: {progress.error}
          </div>
        )}
      </div>
    );
  };

  // Live chunk display
  const renderChunks = () => {
    if (chunks.length === 0) return null;

    return (
      <div className="mt-4 space-y-2 max-h-96 overflow-y-auto">
        <h3 className="text-lg font-semibold">
          Chunks ({chunks.length})
        </h3>
        {chunks.map((chunk, index) => (
          <div 
            key={chunk.chunk_id}
            className="p-3 bg-gray-50 rounded-lg border border-gray-200 animate-slideIn"
          >
            <div className="flex justify-between mb-1">
              <span className="text-sm font-medium">
                Chunk {chunk.chunk_index + 1}
              </span>
              <span className="text-xs text-gray-500">
                {chunk.size} characters
              </span>
            </div>
            <p className="text-sm text-gray-700 line-clamp-3">
              {chunk.content}
            </p>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">Chunking Preview</h2>
        {renderConnectionStatus()}
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-700">{error.message}</p>
        </div>
      )}

      <div className="space-y-4">
        {renderProgressBar()}
        
        <div className="flex gap-2">
          <button
            onClick={handleStartChunking}
            disabled={connectionState !== ConnectionState.CONNECTED || !!currentOperationId}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg disabled:bg-gray-300"
          >
            Start Chunking
          </button>
          
          {currentOperationId && (
            <button
              onClick={handleCancelChunking}
              className="px-4 py-2 bg-red-500 text-white rounded-lg"
            >
              Cancel
            </button>
          )}
        </div>

        {renderChunks()}
      </div>
    </div>
  );
};
```

### 4. Add CSS Animation for Smooth Updates

```css
/* apps/webui-react/src/styles/animations.css */

@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.animate-slideIn {
  animation: slideIn 0.3s ease-out;
}

/* Progress bar smooth transition */
.progress-bar-fill {
  transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Connection status pulse */
@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.connection-indicator {
  animation: pulse 2s infinite;
}
```

## Acceptance Criteria

1. **WebSocket Connection**
   - [ ] Connects successfully with auth
   - [ ] Reconnects automatically on disconnect
   - [ ] Connection status visible to user
   - [ ] Heartbeat keeps connection alive

2. **Real-time Updates**
   - [ ] Progress updates in real-time
   - [ ] Chunks appear as processed
   - [ ] No polling required
   - [ ] Smooth animations for updates

3. **Error Handling**
   - [ ] Connection errors shown to user
   - [ ] Graceful degradation without WS
   - [ ] Retry logic works
   - [ ] Error recovery implemented

4. **Performance**
   - [ ] Low latency (< 100ms)
   - [ ] No memory leaks
   - [ ] Efficient message handling
   - [ ] Proper cleanup on unmount

## Testing Requirements

1. **Unit Tests**
   ```typescript
   describe('useChunkingWebSocket', () => {
     it('should connect to WebSocket', async () => {
       const { result } = renderHook(() => useChunkingWebSocket());
       
       act(() => {
         result.current.connect('test-operation');
       });
       
       await waitFor(() => {
         expect(result.current.connectionState).toBe(ConnectionState.CONNECTED);
       });
     });
     
     it('should receive progress updates', async () => {
       const { result } = renderHook(() => useChunkingWebSocket());
       
       act(() => {
         result.current.connect('test-operation');
       });
       
       // Simulate progress message
       mockWebSocket.simulateMessage({
         type: 'progress',
         channel: 'chunking:test-operation',
         data: {
           operation_id: 'test-operation',
           progress: 50,
           current_chunk: 5,
           total_chunks: 10
         }
       });
       
       expect(result.current.progress?.progress).toBe(50);
     });
   });
   ```

2. **Integration Tests**
   - Test with real WebSocket server
   - Test reconnection scenarios
   - Test auth flow
   - Test message ordering

## Rollback Plan

1. Keep polling as fallback
2. Feature flag for WebSocket
3. Monitor connection stability
4. Quick disable if issues

## Success Metrics

- Real-time updates working for 100% of operations
- Average latency < 50ms
- Reconnection success rate > 95%
- No memory leaks after 24 hours
- User satisfaction with real-time features

## Notes for LLM Agent

- Test with network interruptions
- Ensure proper cleanup to prevent memory leaks
- Handle authentication properly
- Test with high message volume
- Consider mobile network conditions
- Add connection quality indicators