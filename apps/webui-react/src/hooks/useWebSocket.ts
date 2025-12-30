import { useEffect, useRef, useCallback, useState } from 'react';

interface UseWebSocketOptions {
  onOpen?: (event: Event) => void;
  onMessage?: (event: MessageEvent) => void;
  onError?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  autoReconnect?: boolean;
  reconnectInterval?: number;
  reconnectAttempts?: number;
  /** WebSocket subprotocols - used for passing auth token securely */
  protocols?: string[];
}

export function useWebSocket(
  url: string | null,
  options: UseWebSocketOptions = {}
) {
  const {
    onOpen,
    onMessage,
    onError,
    onClose,
    autoReconnect = true,
    reconnectInterval = 3000,
    reconnectAttempts = 5,
    protocols,
  } = options;

  const ws = useRef<WebSocket | null>(null);
  const reconnectCount = useRef(0);
  const reconnectTimeoutId = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [readyState, setReadyState] = useState<number>(
    WebSocket.CONNECTING
  );

  // Use refs for callbacks to avoid recreating connect on callback changes
  const onOpenRef = useRef(onOpen);
  const onMessageRef = useRef(onMessage);
  const onErrorRef = useRef(onError);
  const onCloseRef = useRef(onClose);

  // Keep refs in sync with latest callbacks
  useEffect(() => {
    onOpenRef.current = onOpen;
    onMessageRef.current = onMessage;
    onErrorRef.current = onError;
    onCloseRef.current = onClose;
  }, [onOpen, onMessage, onError, onClose]);

  const connect = useCallback(() => {
    if (!url) {
      return;
    }

    // Close existing connection before opening a new one
    if (ws.current) {
      const existingWs = ws.current;
      // Only close if not already closing/closed
      if (existingWs.readyState === WebSocket.OPEN || existingWs.readyState === WebSocket.CONNECTING) {
        existingWs.close(1000, 'Reconnecting');
      }
      ws.current = null;
    }

    try {
      // Pass protocols for authentication (token is sent via Sec-WebSocket-Protocol header)
      ws.current = protocols?.length ? new WebSocket(url, protocols) : new WebSocket(url);

      // Add a connection timeout
      const timeoutId = setTimeout(() => {
        if (ws.current?.readyState === WebSocket.CONNECTING) {
          console.error('WebSocket connection timeout');
          ws.current.close();
          onErrorRef.current?.(new Event('timeout'));
        }
      }, 5000);

      ws.current.onopen = (event) => {
        clearTimeout(timeoutId);
        setReadyState(WebSocket.OPEN);
        reconnectCount.current = 0;
        onOpenRef.current?.(event);
      };

      ws.current.onmessage = (event) => {
        onMessageRef.current?.(event);
      };

      ws.current.onerror = (event) => {
        clearTimeout(timeoutId);
        setReadyState(ws.current?.readyState || WebSocket.CLOSED);
        onErrorRef.current?.(event);
      };

      ws.current.onclose = (event) => {
        clearTimeout(timeoutId);
        setReadyState(WebSocket.CLOSED);
        onCloseRef.current?.(event);

        if (
          autoReconnect &&
          reconnectCount.current < reconnectAttempts &&
          !event.wasClean &&
          event.code !== 1000 // Normal closure
        ) {
          reconnectCount.current++;
          reconnectTimeoutId.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };
    } catch (error) {
      console.error('WebSocket connection error:', error);
    }
  }, [
    url,
    autoReconnect,
    reconnectInterval,
    reconnectAttempts,
    protocols,
  ]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutId.current) {
      clearTimeout(reconnectTimeoutId.current);
      reconnectTimeoutId.current = null;
    }
    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }
  }, []);

  const sendMessage = useCallback((data: string | object) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      const message = typeof data === 'string' ? data : JSON.stringify(data);
      ws.current.send(message);
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    sendMessage,
    readyState,
    disconnect,
    reconnect: connect,
  };
}