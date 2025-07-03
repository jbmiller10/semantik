import { useEffect, useRef, useCallback, useState } from 'react';

interface UseWebSocketOptions {
  onOpen?: (event: Event) => void;
  onMessage?: (event: MessageEvent) => void;
  onError?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  autoReconnect?: boolean;
  reconnectInterval?: number;
  reconnectAttempts?: number;
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
  } = options;

  const ws = useRef<WebSocket | null>(null);
  const reconnectCount = useRef(0);
  const reconnectTimeoutId = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [readyState, setReadyState] = useState<number>(
    WebSocket.CONNECTING
  );

  const connect = useCallback(() => {
    if (!url) {
      return;
    }

    try {
      ws.current = new WebSocket(url);
      
      // Add a connection timeout
      const timeoutId = setTimeout(() => {
        if (ws.current?.readyState === WebSocket.CONNECTING) {
          console.error('WebSocket connection timeout');
          ws.current.close();
          onError?.(new Event('timeout'));
        }
      }, 5000);
      
      ws.current.onopen = (event) => {
        clearTimeout(timeoutId);
        setReadyState(WebSocket.OPEN);
        reconnectCount.current = 0;
        onOpen?.(event);
      };

      ws.current.onmessage = (event) => {
        onMessage?.(event);
      };

      ws.current.onerror = (event) => {
        clearTimeout(timeoutId);
        setReadyState(ws.current?.readyState || WebSocket.CLOSED);
        onError?.(event);
      };

      ws.current.onclose = (event) => {
        clearTimeout(timeoutId);
        setReadyState(WebSocket.CLOSED);
        onClose?.(event);

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
    onOpen,
    onMessage,
    onError,
    onClose,
    autoReconnect,
    reconnectInterval,
    reconnectAttempts,
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