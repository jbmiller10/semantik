/**
 * Hook for handling SSE streaming from the assisted flow API.
 * Uses fetch with ReadableStream to parse Server-Sent Events.
 */

import { useCallback, useRef, useState } from 'react';
import { useAuthStore } from '../stores/authStore';
import { assistedFlowApi } from '../services/api/v2/assisted-flow';
import type {
  AssistedFlowEventType,
  AssistedFlowToolCall,
  TextEventData,
  ToolUseEventData,
  ToolResultEventData,
  DoneEventData,
  ErrorEventData,
} from '../types/assisted-flow';

export interface UseAssistedFlowStreamCallbacks {
  onText?: (text: string) => void;
  onToolUse?: (data: ToolUseEventData) => void;
  onToolResult?: (data: ToolResultEventData) => void;
  onDone?: (data: DoneEventData) => void;
  onError?: (error: string) => void;
}

export interface UseAssistedFlowStreamReturn {
  isStreaming: boolean;
  error: string | null;
  currentContent: string;
  toolCalls: AssistedFlowToolCall[];
  sendMessage: (message: string) => Promise<void>;
  cancel: () => void;
  reset: () => void;
}

/**
 * Parse a single SSE message.
 * SSE format: event: type\ndata: json\n\n
 */
function parseSSEMessage(
  raw: string
): { event: AssistedFlowEventType; data: Record<string, unknown> } | null {
  const lines = raw.split('\n');
  let eventType: string | null = null;
  let dataStr = '';

  for (const line of lines) {
    if (line.startsWith('event:')) {
      eventType = line.slice(6).trim();
    } else if (line.startsWith('data:')) {
      dataStr += line.slice(5).trim();
    }
  }

  if (!eventType || !dataStr) {
    return null;
  }

  try {
    const data = JSON.parse(dataStr);
    return { event: eventType as AssistedFlowEventType, data };
  } catch (error) {
    console.error('SSE parse failure:', dataStr, error);
    return null;
  }
}

export function useAssistedFlowStream(
  sessionId: string,
  callbacks: UseAssistedFlowStreamCallbacks = {}
): UseAssistedFlowStreamReturn {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentContent, setCurrentContent] = useState('');
  const [toolCalls, setToolCalls] = useState<AssistedFlowToolCall[]>([]);

  const abortControllerRef = useRef<AbortController | null>(null);
  const token = useAuthStore((state) => state.token);

  const reset = useCallback(() => {
    setIsStreaming(false);
    setError(null);
    setCurrentContent('');
    setToolCalls([]);
  }, []);

  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const sendMessage = useCallback(
    async (message: string) => {
      // Reset state for new message
      setIsStreaming(true);
      setError(null);
      setCurrentContent('');
      setToolCalls([]);

      // Create abort controller
      abortControllerRef.current = new AbortController();

      const url = assistedFlowApi.getStreamUrl(sessionId);

      try {
        const response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
            Accept: 'text/event-stream',
          },
          body: JSON.stringify({ message }),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Request failed' }));
          throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        if (!response.body) {
          throw new Error('No response body');
        }

        // Read the stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();

          if (done) {
            break;
          }

          // Decode chunk and add to buffer
          buffer += decoder.decode(value, { stream: true });

          // Process complete messages (separated by \n\n)
          const messages = buffer.split('\n\n');
          buffer = messages.pop() || ''; // Keep incomplete message in buffer

          for (const msg of messages) {
            if (!msg.trim()) continue;

            const parsed = parseSSEMessage(msg);
            if (!parsed) continue;

            const { event, data } = parsed;

            switch (event) {
              case 'text': {
                const textData = data as TextEventData;
                const text = textData.content || '';
                setCurrentContent((prev) => prev + text);
                callbacks.onText?.(text);
                break;
              }

              case 'tool_use': {
                const toolData = data as ToolUseEventData;
                const toolName = toolData.tool_name || 'unknown';
                const toolCallId = `${Date.now()}-${toolName}`;
                setToolCalls((prev) => [
                  ...prev,
                  {
                    id: toolCallId,
                    tool_name: toolName,
                    arguments: toolData.arguments,
                    status: 'running',
                  },
                ]);
                callbacks.onToolUse?.(toolData);
                break;
              }

              case 'tool_result': {
                const resultData = data as ToolResultEventData;
                const toolName = resultData.tool_name || 'unknown';
                setToolCalls((prev) =>
                  prev.map((tc) =>
                    tc.tool_name === toolName && tc.status === 'running'
                      ? {
                          ...tc,
                          status: resultData.success !== false ? 'success' : 'error',
                          result: resultData.result,
                        }
                      : tc
                  )
                );
                callbacks.onToolResult?.(resultData);
                break;
              }

              case 'done': {
                const doneData = data as DoneEventData;
                callbacks.onDone?.(doneData);
                break;
              }

              case 'error': {
                const errorData = data as ErrorEventData;
                setError(errorData.message || 'Unknown error');
                callbacks.onError?.(errorData.message || 'Unknown error');
                break;
              }
            }
          }
        }
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') {
          // Cancelled by user - clean up partial state
          setToolCalls((prev) =>
            prev.map((tc) =>
              tc.status === 'running'
                ? { ...tc, status: 'error' as const, error: 'Cancelled' }
                : tc
            )
          );
          return;
        }
        const errorMessage = err instanceof Error ? err.message : 'Stream failed';
        setError(errorMessage);

        // Clean up any in-progress tool calls
        setToolCalls((prev) =>
          prev.map((tc) =>
            tc.status === 'running'
              ? { ...tc, status: 'error' as const, error: 'Stream interrupted' }
              : tc
          )
        );

        callbacks.onError?.(errorMessage);
      } finally {
        setIsStreaming(false);
        abortControllerRef.current = null;
      }
    },
    [sessionId, token, callbacks]
  );

  return {
    isStreaming,
    error,
    currentContent,
    toolCalls,
    sendMessage,
    cancel,
    reset,
  };
}
