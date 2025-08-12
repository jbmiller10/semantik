import { EventEmitter } from 'events';

/**
 * WebSocket connection states
 */
export const WebSocketState = {
  CONNECTING: 0,
  OPEN: 1,
  CLOSING: 2,
  CLOSED: 3,
} as const;

export type WebSocketState = typeof WebSocketState[keyof typeof WebSocketState];

/**
 * WebSocket message types for chunking
 */
export const ChunkingMessageType = {
  // Authentication messages
  AUTH_REQUEST: 'auth_request',
  AUTH_SUCCESS: 'auth_success',
  AUTH_ERROR: 'auth_error',
  // Chunking messages
  PREVIEW_START: 'preview_start',
  PREVIEW_PROGRESS: 'preview_progress',
  PREVIEW_CHUNK: 'preview_chunk',
  PREVIEW_COMPLETE: 'preview_complete',
  PREVIEW_ERROR: 'preview_error',
  COMPARISON_START: 'comparison_start',
  COMPARISON_PROGRESS: 'comparison_progress',
  COMPARISON_RESULT: 'comparison_result',
  COMPARISON_COMPLETE: 'comparison_complete',
  COMPARISON_ERROR: 'comparison_error',
  HEARTBEAT: 'heartbeat',
  PONG: 'pong',
} as const;

export type ChunkingMessageType = typeof ChunkingMessageType[keyof typeof ChunkingMessageType];

/**
 * WebSocket message structure
 */
export interface WebSocketMessage<T = unknown> {
  type: string;
  data: T;
  timestamp: number;
  requestId?: string;
}

/**
 * Chunking-specific message data types
 */
export interface ChunkingProgressData {
  percentage: number;
  currentChunk: number;
  totalChunks: number;
  estimatedTimeRemaining?: number;
}

export interface ChunkingChunkData {
  chunk: {
    id: string;
    content: string;
    startIndex: number;
    endIndex: number;
    metadata?: Record<string, unknown>;
    tokens?: number;
    overlapWithPrevious?: number;
    overlapWithNext?: number;
  };
  index: number;
  total: number;
}

export interface ChunkingCompleteData {
  statistics: {
    totalChunks: number;
    avgChunkSize: number;
    minChunkSize: number;
    maxChunkSize: number;
    totalTokens?: number;
    avgTokensPerChunk?: number;
    overlapPercentage?: number;
    sizeDistribution?: {
      range: string;
      count: number;
      percentage: number;
    }[];
  };
  performance: {
    processingTimeMs: number;
    estimatedFullProcessingTimeMs: number;
  };
}

export interface ChunkingErrorData {
  message: string;
  code?: string;
  details?: unknown;
}

/**
 * Authentication data types
 */
export interface AuthRequestData {
  token: string;
}

export interface AuthSuccessData {
  userId?: string;
  sessionId?: string;
}

export interface AuthErrorData {
  message: string;
  code?: string;
}

/**
 * Configuration for WebSocket service
 */
export interface WebSocketConfig {
  url: string;
  reconnect?: boolean;
  reconnectInterval?: number;
  reconnectMaxAttempts?: number;
  heartbeatInterval?: number;
  connectionTimeout?: number;
}

/**
 * WebSocket service with EventEmitter pattern
 * Provides reconnection logic, heartbeat/ping-pong, and graceful degradation
 * 
 * SECURITY: Authentication is performed after connection establishment.
 * The token is sent as the first message over the secure WebSocket connection,
 * NOT in the URL query parameters. This prevents token exposure in:
 * - Server access logs
 * - Browser history
 * - Network debugging tools
 * 
 * Flow:
 * 1. Connect to WebSocket WITHOUT token in URL
 * 2. Send AUTH_REQUEST message with token after connection opens
 * 3. Wait for AUTH_SUCCESS or AUTH_ERROR response
 * 4. Only process other messages after successful authentication
 */
export class WebSocketService extends EventEmitter {
  private ws: WebSocket | null = null;
  private config: Required<WebSocketConfig>;
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private connectionTimer: NodeJS.Timeout | null = null;
  private authenticationTimer: NodeJS.Timeout | null = null;
  private isIntentionallyClosed = false;
  private messageQueue: WebSocketMessage[] = [];
  private lastPongReceived = Date.now();
  private isAuthenticated = false;
  private authenticationToken: string | null = null;
  
  constructor(config: WebSocketConfig) {
    super();
    
    // Set default configuration values
    this.config = {
      url: config.url,
      reconnect: config.reconnect ?? true,
      reconnectInterval: config.reconnectInterval ?? 3000,
      reconnectMaxAttempts: config.reconnectMaxAttempts ?? 5,
      heartbeatInterval: config.heartbeatInterval ?? 30000,
      connectionTimeout: config.connectionTimeout ?? 5000,
    };
  }

  /**
   * Connect to WebSocket server
   */
  connect(): void {
    if (this.ws?.readyState === WebSocketState.OPEN) {
      console.warn('WebSocket already connected');
      return;
    }

    this.isIntentionallyClosed = false;
    this.createConnection();
  }

  /**
   * Create WebSocket connection with authentication
   */
  private createConnection(): void {
    try {
      // Get authentication token from localStorage (Zustand store)
      const authStorage = localStorage.getItem('auth-storage');
      let token = '';
      
      if (authStorage) {
        try {
          const authState = JSON.parse(authStorage);
          token = authState.state?.token || '';
        } catch (error) {
          console.error('Failed to parse auth storage:', error);
          this.emit('error', { message: 'Authentication failed', code: 'AUTH_ERROR' });
          return;
        }
      }

      if (!token) {
        console.error('No authentication token found');
        this.emit('error', { message: 'Not authenticated', code: 'NO_AUTH' });
        return;
      }

      // Store token for authentication after connection
      this.authenticationToken = token;
      this.isAuthenticated = false;
      
      // Connect WITHOUT token in URL (SECURE)
      console.log('Connecting to WebSocket:', this.config.url);
      this.ws = new WebSocket(this.config.url);
      
      // Set up connection timeout
      this.connectionTimer = setTimeout(() => {
        if (this.ws?.readyState === WebSocketState.CONNECTING) {
          console.error('WebSocket connection timeout');
          this.ws.close();
          this.handleConnectionError(new Error('Connection timeout'));
        }
      }, this.config.connectionTimeout);

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onerror = this.handleError.bind(this);
      this.ws.onclose = this.handleClose.bind(this);
      
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      this.handleConnectionError(error as Error);
    }
  }

  /**
   * Handle WebSocket connection open
   */
  private handleOpen(): void {
    console.log('WebSocket connected, authenticating...');
    
    // Clear connection timeout
    if (this.connectionTimer) {
      clearTimeout(this.connectionTimer);
      this.connectionTimer = null;
    }
    
    // Reset reconnection attempts
    this.reconnectAttempts = 0;
    
    // Send authentication message as first message
    this.sendAuthentication();
  }
  
  /**
   * Send authentication message
   */
  private sendAuthentication(): void {
    if (!this.authenticationToken) {
      console.error('No authentication token available');
      this.ws?.close(1008, 'Authentication failed');
      return;
    }
    
    // Set authentication timeout (5 seconds)
    this.authenticationTimer = setTimeout(() => {
      console.error('Authentication timeout');
      this.isAuthenticated = false;
      this.ws?.close(1008, 'Authentication timeout');
      this.emit('error', { message: 'Authentication timeout', code: 'AUTH_TIMEOUT' });
    }, 5000);
    
    // Send authentication request
    const authMessage: WebSocketMessage<AuthRequestData> = {
      type: ChunkingMessageType.AUTH_REQUEST,
      data: { token: this.authenticationToken },
      timestamp: Date.now(),
    };
    
    // Send directly without queueing (authentication is special)
    if (this.ws?.readyState === WebSocketState.OPEN) {
      try {
        this.ws.send(JSON.stringify(authMessage));
        console.log('Authentication request sent');
      } catch (error) {
        console.error('Failed to send authentication:', error);
        this.ws?.close(1008, 'Authentication failed');
      }
    }
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      
      // Handle authentication responses first (before authentication check)
      if (message.type === ChunkingMessageType.AUTH_SUCCESS) {
        this.handleAuthSuccess(message.data as AuthSuccessData);
        return;
      }
      
      if (message.type === ChunkingMessageType.AUTH_ERROR) {
        this.handleAuthError(message.data as AuthErrorData);
        return;
      }
      
      // Check if authenticated before processing other messages
      if (!this.isAuthenticated) {
        console.warn('Received message before authentication:', message.type);
        return;
      }
      
      // Handle heartbeat/pong
      if (message.type === ChunkingMessageType.PONG) {
        this.lastPongReceived = Date.now();
        return;
      }
      
      // Emit typed message event
      this.emit('message', message);
      
      // Emit specific event for message type
      this.emit(message.type, message.data, message);
      
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error, event.data);
      this.emit('error', { message: 'Invalid message format', data: event.data });
    }
  }
  
  /**
   * Handle successful authentication
   */
  private handleAuthSuccess(data: AuthSuccessData): void {
    // Clear authentication timeout
    if (this.authenticationTimer) {
      clearTimeout(this.authenticationTimer);
      this.authenticationTimer = null;
    }
    
    console.log('WebSocket authenticated successfully', data);
    this.isAuthenticated = true;
    
    // Start heartbeat after successful authentication
    this.startHeartbeat();
    
    // Send queued messages
    this.flushMessageQueue();
    
    // Emit connected event after authentication
    this.emit('connected', { authenticated: true, ...data });
    this.emit('authenticated', data);
  }
  
  /**
   * Handle authentication error
   */
  private handleAuthError(data: AuthErrorData): void {
    // Clear authentication timeout
    if (this.authenticationTimer) {
      clearTimeout(this.authenticationTimer);
      this.authenticationTimer = null;
    }
    
    console.error('WebSocket authentication failed:', data);
    this.isAuthenticated = false;
    
    // Emit error and close connection
    this.emit('error', { message: data.message, code: data.code || 'AUTH_FAILED' });
    this.emit('authentication_failed', data);
    
    // Close connection with authentication failure code
    this.ws?.close(1008, 'Authentication failed');
  }

  /**
   * Handle WebSocket errors
   */
  private handleError(event: Event): void {
    console.error('WebSocket error:', event);
    
    // Clear connection timeout
    if (this.connectionTimer) {
      clearTimeout(this.connectionTimer);
      this.connectionTimer = null;
    }
    
    this.emit('error', event);
  }

  /**
   * Handle WebSocket connection close
   */
  private handleClose(event: CloseEvent): void {
    console.log('WebSocket closed:', event.code, event.reason);
    
    // Reset authentication state
    this.isAuthenticated = false;
    this.authenticationToken = null;
    
    // Clear timers
    this.stopHeartbeat();
    if (this.connectionTimer) {
      clearTimeout(this.connectionTimer);
      this.connectionTimer = null;
    }
    if (this.authenticationTimer) {
      clearTimeout(this.authenticationTimer);
      this.authenticationTimer = null;
    }
    
    // Emit disconnected event
    this.emit('disconnected', event);
    
    // Handle reconnection
    if (!this.isIntentionallyClosed && this.config.reconnect && event.code !== 1000) {
      this.scheduleReconnect();
    }
  }

  /**
   * Handle connection errors
   */
  private handleConnectionError(error: Error): void {
    console.error('WebSocket connection error:', error);
    this.emit('error', { message: error.message, code: 'CONNECTION_ERROR' });
    
    if (!this.isIntentionallyClosed && this.config.reconnect) {
      this.scheduleReconnect();
    }
  }

  /**
   * Schedule reconnection with exponential backoff
   */
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.config.reconnectMaxAttempts) {
      console.error('Max reconnection attempts reached');
      this.emit('reconnect_failed', { attempts: this.reconnectAttempts });
      return;
    }
    
    this.reconnectAttempts++;
    
    // Calculate delay with exponential backoff and jitter
    const baseDelay = Math.min(
      this.config.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1),
      30000 // Max 30 seconds
    );
    const jitter = Math.random() * 1000;
    const delay = baseDelay + jitter;
    
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.config.reconnectMaxAttempts})`);
    
    this.reconnectTimer = setTimeout(() => {
      this.emit('reconnecting', { attempt: this.reconnectAttempts });
      this.createConnection();
    }, delay);
  }

  /**
   * Start heartbeat mechanism
   */
  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatTimer = setInterval(() => {
      // Only send heartbeats when connected and authenticated
      if (this.ws?.readyState === WebSocketState.OPEN && this.isAuthenticated) {
        // Check if we received a pong recently
        const timeSinceLastPong = Date.now() - this.lastPongReceived;
        if (timeSinceLastPong > this.config.heartbeatInterval * 2) {
          console.warn('No pong received, connection might be dead');
          this.ws.close();
          return;
        }
        
        // Send heartbeat (will be queued if not authenticated)
        this.send({
          type: ChunkingMessageType.HEARTBEAT,
          data: { timestamp: Date.now() },
          timestamp: Date.now(),
        });
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Stop heartbeat mechanism
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * Send message through WebSocket
   */
  send(message: WebSocketMessage): boolean {
    // Check if WebSocket is open and authenticated
    if (this.ws?.readyState === WebSocketState.OPEN && this.isAuthenticated) {
      try {
        this.ws.send(JSON.stringify(message));
        return true;
      } catch (error) {
        console.error('Failed to send message:', error);
        this.emit('error', { message: 'Failed to send message', error });
        return false;
      }
    } else {
      // Queue message for later sending
      this.messageQueue.push(message);
      
      if (!this.isAuthenticated) {
        console.warn('WebSocket not authenticated yet, message queued');
      } else {
        console.warn('WebSocket not connected, message queued');
      }
      
      return false;
    }
  }

  /**
   * Flush queued messages
   */
  private flushMessageQueue(): void {
    while (this.messageQueue.length > 0 && this.ws?.readyState === WebSocketState.OPEN && this.isAuthenticated) {
      const message = this.messageQueue.shift();
      if (message) {
        // Send directly since we're already authenticated
        try {
          this.ws.send(JSON.stringify(message));
        } catch (error) {
          console.error('Failed to send queued message:', error);
          // Re-queue the message
          this.messageQueue.unshift(message);
          break;
        }
      }
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect(): void {
    this.isIntentionallyClosed = true;
    
    // Reset authentication state
    this.isAuthenticated = false;
    this.authenticationToken = null;
    
    // Clear reconnection timer
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    // Stop heartbeat
    this.stopHeartbeat();
    
    // Clear connection timer
    if (this.connectionTimer) {
      clearTimeout(this.connectionTimer);
      this.connectionTimer = null;
    }
    
    // Clear authentication timer
    if (this.authenticationTimer) {
      clearTimeout(this.authenticationTimer);
      this.authenticationTimer = null;
    }
    
    // Close WebSocket connection
    if (this.ws) {
      if (this.ws.readyState === WebSocketState.OPEN || this.ws.readyState === WebSocketState.CONNECTING) {
        this.ws.close(1000, 'Client disconnect');
      }
      this.ws = null;
    }
    
    // Clear message queue
    this.messageQueue = [];
    
    console.log('WebSocket disconnected');
  }

  /**
   * Get current connection state
   */
  getState(): WebSocketState {
    return (this.ws?.readyState ?? WebSocketState.CLOSED) as WebSocketState;
  }

  /**
   * Check if connected (but may not be authenticated)
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocketState.OPEN;
  }

  /**
   * Check if connected AND authenticated (ready for use)
   */
  isReady(): boolean {
    return this.ws?.readyState === WebSocketState.OPEN && this.isAuthenticated;
  }

  /**
   * Get authentication status
   */
  isAuthenticatedStatus(): boolean {
    return this.isAuthenticated;
  }

  /**
   * Get reconnection attempts
   */
  getReconnectAttempts(): number {
    return this.reconnectAttempts;
  }
}

// Export singleton instance for chunking WebSocket
let chunkingWebSocketInstance: WebSocketService | null = null;

export function getChunkingWebSocket(): WebSocketService {
  if (!chunkingWebSocketInstance) {
    const baseUrl = window.location.origin.replace(/^http/, 'ws');
    chunkingWebSocketInstance = new WebSocketService({
      url: `${baseUrl}/ws/chunking`,
      reconnect: true,
      reconnectInterval: 3000,
      reconnectMaxAttempts: 5,
      heartbeatInterval: 30000,
      connectionTimeout: 5000,
    });
  }
  return chunkingWebSocketInstance;
}

export function disconnectChunkingWebSocket(): void {
  if (chunkingWebSocketInstance) {
    chunkingWebSocketInstance.disconnect();
    chunkingWebSocketInstance = null;
  }
}