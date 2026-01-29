/**
 * Hook for handling SSE streaming from the agent API.
 * Uses fetch with ReadableStream to parse Server-Sent Events.
 */

import { useCallback, useRef, useState } from 'react';
import { useAuthStore } from '../stores/authStore';
import { agentApi } from '../services/api/v2/agent';
import {
  validateEventData,
  type AgentEventType,
} from '../schemas/agentStream';
import type {
  AgentStreamEvent,
  AgentStreamEventType,
  ToolCallState,
  SubagentState,
  Uncertainty,
  PipelineConfig,
  ToolCallStartEvent,
  ToolCallEndEvent,
  SubagentStartEvent,
  SubagentEndEvent,
  UncertaintyEvent,
  PipelineUpdateEvent,
  DoneEvent,
  StatusEvent,
  ActivityEvent,
  AgentPhase,
  QuestionEvent,
} from '../types/agent';

export interface UseAgentStreamCallbacks {
  onContent?: (text: string) => void;
  onToolCallStart?: (data: ToolCallStartEvent) => void;
  onToolCallEnd?: (data: ToolCallEndEvent) => void;
  onSubagentStart?: (data: SubagentStartEvent) => void;
  onSubagentEnd?: (data: SubagentEndEvent) => void;
  onUncertainty?: (data: UncertaintyEvent) => void;
  onPipelineUpdate?: (data: PipelineUpdateEvent) => void;
  onDone?: (data: DoneEvent) => void;
  onError?: (error: string) => void;
  onStatus?: (data: StatusEvent) => void;
  onActivity?: (data: ActivityEvent) => void;
  onQuestion?: (data: QuestionEvent) => void;
}

export interface UseAgentStreamReturn {
  isStreaming: boolean;
  error: string | null;
  currentContent: string;
  toolCalls: ToolCallState[];
  subagents: SubagentState[];
  uncertainties: Uncertainty[];
  pipeline: PipelineConfig | null;
  status: {
    phase: AgentPhase;
    message: string;
    progress?: { current: number; total: number };
  } | null;
  activities: Array<{ message: string; timestamp: string }>;
  pendingQuestions: QuestionEvent[];
  sendMessage: (message: string) => Promise<void>;
  cancel: () => void;
  reset: () => void;
  dismissQuestion: (questionId: string) => void;
}

/**
 * Parse a single SSE message.
 * SSE format: event: type\ndata: json\n\n
 *
 * @param raw - The raw SSE message string
 * @param parseFailureCount - Current count of consecutive parse failures (for tracking)
 * @returns Object with event, data, and updated failure count, or null if empty message
 * @throws Error if JSON parse fails (to allow tracking consecutive failures)
 */
function parseSSEMessage(
  raw: string,
  parseFailureCount: number
): { event: AgentStreamEventType; data: Record<string, unknown>; parseFailureCount: number } | null {
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
    // Reset failure count on successful parse
    return { event: eventType as AgentStreamEventType, data, parseFailureCount: 0 };
  } catch (error) {
    const newFailureCount = parseFailureCount + 1;
    console.error(`SSE parse failure #${newFailureCount}:`, dataStr, error);
    // Return null but with updated failure count via thrown error metadata
    // Caller should handle this and check threshold
    const parseError = new Error(`Failed to parse SSE data: ${dataStr.slice(0, 100)}`);
    (parseError as Error & { parseFailureCount: number }).parseFailureCount = newFailureCount;
    throw parseError;
  }
}

export function useAgentStream(
  conversationId: string,
  callbacks: UseAgentStreamCallbacks = {}
): UseAgentStreamReturn {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentContent, setCurrentContent] = useState('');
  const [toolCalls, setToolCalls] = useState<ToolCallState[]>([]);
  const [subagents, setSubagents] = useState<SubagentState[]>([]);
  const [uncertainties, setUncertainties] = useState<Uncertainty[]>([]);
  const [pipeline, setPipeline] = useState<PipelineConfig | null>(null);
  const [status, setStatus] = useState<{
    phase: AgentPhase;
    message: string;
    progress?: { current: number; total: number };
  } | null>(null);
  const [activities, setActivities] = useState<Array<{ message: string; timestamp: string }>>([]);
  const [pendingQuestions, setPendingQuestions] = useState<QuestionEvent[]>([]);

  const abortControllerRef = useRef<AbortController | null>(null);
  const token = useAuthStore((state) => state.token);

  const reset = useCallback(() => {
    setIsStreaming(false);
    setError(null);
    setCurrentContent('');
    setToolCalls([]);
    setSubagents([]);
    setUncertainties([]);
    setPipeline(null);
    setStatus(null);
    setActivities([]);
    setPendingQuestions([]);
  }, []);

  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  const dismissQuestion = useCallback((questionId: string) => {
    setPendingQuestions((prev) => prev.filter((q) => q.id !== questionId));
  }, []);

  const sendMessage = useCallback(
    async (message: string) => {
      // Reset state for new message
      setIsStreaming(true);
      setError(null);
      setCurrentContent('');
      setToolCalls([]);
      setSubagents([]);

      // Create abort controller
      abortControllerRef.current = new AbortController();

      const url = agentApi.getStreamUrl(conversationId);

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
        let parseFailureCount = 0;          // Consecutive failures for threshold check
        let totalParseFailures = 0;          // Total failures for diagnostics
        const PARSE_FAILURE_THRESHOLD = 3;

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

            let parsed: { event: AgentStreamEventType; data: Record<string, unknown>; parseFailureCount: number } | null = null;
            try {
              parsed = parseSSEMessage(msg, parseFailureCount);
              if (parsed) {
                parseFailureCount = parsed.parseFailureCount; // Reset on success
              }
            } catch (parseError) {
              // Track consecutive and total parse failures
              if (parseError && typeof parseError === 'object' && 'parseFailureCount' in parseError) {
                parseFailureCount = (parseError as { parseFailureCount: number }).parseFailureCount;
              }
              totalParseFailures++;
              console.warn(`SSE parse failure (total: ${totalParseFailures}, consecutive: ${parseFailureCount})`, parseError);

              // Surface warning to user on first failure (non-blocking)
              if (totalParseFailures === 1) {
                callbacks.onError?.('Some data may be missing due to a communication issue');
              }

              // Surface error to user after threshold consecutive failures (blocking)
              if (parseFailureCount >= PARSE_FAILURE_THRESHOLD) {
                const errorMsg = `Stream corrupted: ${totalParseFailures} parse failures - please refresh and try again`;
                setError(errorMsg);
                callbacks.onError?.(errorMsg);
                reader.cancel();
                return; // Stop processing - stream is corrupted
              }
              continue;
            }

            if (!parsed) continue;

            const { event, data } = parsed;

            // Validate event data against Zod schema and build typed event
            const isKnownEvent = (e: string): e is AgentEventType =>
              ['content', 'tool_call_start', 'tool_call_end', 'subagent_start', 'subagent_end',
               'uncertainty', 'pipeline_update', 'done', 'error', 'status', 'activity', 'question'].includes(e);

            if (!isKnownEvent(event)) {
              console.warn(`Unknown event type: ${event}`);
              continue;
            }

            const validatedData = validateEventData(event, data);
            if (validatedData === null) {
              // Validation failed - skip this event but don't stop streaming
              console.warn(`Skipping invalid ${event} event`);
              continue;
            }

            // Build the typed event - TypeScript narrows based on event discriminant
            const typedEvent = { event, data: validatedData } as AgentStreamEvent;

            switch (typedEvent.event) {
              case 'content': {
                setCurrentContent((prev) => prev + typedEvent.data.text);
                callbacks.onContent?.(typedEvent.data.text);
                break;
              }

              case 'tool_call_start': {
                const toolCallId = `${Date.now()}-${typedEvent.data.tool}`;
                setToolCalls((prev) => [
                  ...prev,
                  {
                    id: toolCallId,
                    tool: typedEvent.data.tool,
                    arguments: typedEvent.data.arguments,
                    status: 'running',
                  },
                ]);
                callbacks.onToolCallStart?.(typedEvent.data);
                break;
              }

              case 'tool_call_end': {
                setToolCalls((prev) =>
                  prev.map((tc) =>
                    tc.tool === typedEvent.data.tool && tc.status === 'running'
                      ? {
                          ...tc,
                          status: typedEvent.data.success ? 'success' : 'error',
                          result: typedEvent.data.result,
                          error: typedEvent.data.error,
                        }
                      : tc
                  )
                );
                callbacks.onToolCallEnd?.(typedEvent.data);
                break;
              }

              case 'subagent_start': {
                const subagentId = `${Date.now()}-${typedEvent.data.name}`;
                setSubagents((prev) => [
                  ...prev,
                  {
                    id: subagentId,
                    name: typedEvent.data.name,
                    task: typedEvent.data.task,
                    status: 'running',
                  },
                ]);
                callbacks.onSubagentStart?.(typedEvent.data);
                break;
              }

              case 'subagent_end': {
                setSubagents((prev) =>
                  prev.map((sa) =>
                    sa.name === typedEvent.data.name && sa.status === 'running'
                      ? {
                          ...sa,
                          status: typedEvent.data.success ? 'success' : 'error',
                          result: typedEvent.data.result,
                          error: typedEvent.data.error,
                        }
                      : sa
                  )
                );
                callbacks.onSubagentEnd?.(typedEvent.data);
                break;
              }

              case 'uncertainty': {
                setUncertainties((prev) => [
                  ...prev,
                  {
                    id: typedEvent.data.id,
                    severity: typedEvent.data.severity,
                    message: typedEvent.data.message,
                    resolved: false,
                    context: typedEvent.data.context,
                  },
                ]);
                callbacks.onUncertainty?.(typedEvent.data);
                break;
              }

              case 'pipeline_update': {
                setPipeline(typedEvent.data.pipeline);
                callbacks.onPipelineUpdate?.(typedEvent.data);
                break;
              }

              case 'done': {
                callbacks.onDone?.(typedEvent.data);
                break;
              }

              case 'error': {
                setError(typedEvent.data.message);
                callbacks.onError?.(typedEvent.data.message);
                break;
              }

              case 'status': {
                setStatus({
                  phase: typedEvent.data.phase,
                  message: typedEvent.data.message,
                  progress: typedEvent.data.progress,
                });
                callbacks.onStatus?.(typedEvent.data);
                break;
              }

              case 'activity': {
                setActivities((prev) => [
                  ...prev,
                  {
                    message: typedEvent.data.message,
                    timestamp: typedEvent.data.timestamp,
                  },
                ]);
                callbacks.onActivity?.(typedEvent.data);
                break;
              }

              case 'question': {
                setPendingQuestions((prev) => [...prev, typedEvent.data]);
                callbacks.onQuestion?.(typedEvent.data);
                break;
              }
            }
          }
        }
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') {
          // Cancelled by user - still clean up partial state
          setToolCalls((prev) =>
            prev.map((tc) =>
              tc.status === 'running'
                ? { ...tc, status: 'error' as const, error: 'Cancelled' }
                : tc
            )
          );
          setSubagents((prev) =>
            prev.map((sa) =>
              sa.status === 'running'
                ? { ...sa, status: 'error' as const, error: 'Cancelled' }
                : sa
            )
          );
          return;
        }
        const errorMessage = err instanceof Error ? err.message : 'Stream failed';
        setError(errorMessage);

        // Clean up any in-progress tool calls and subagents
        setToolCalls((prev) =>
          prev.map((tc) =>
            tc.status === 'running'
              ? { ...tc, status: 'error' as const, error: 'Stream interrupted' }
              : tc
          )
        );
        setSubagents((prev) =>
          prev.map((sa) =>
            sa.status === 'running'
              ? { ...sa, status: 'error' as const, error: 'Stream interrupted' }
              : sa
          )
        );

        callbacks.onError?.(errorMessage);
      } finally {
        setIsStreaming(false);
        abortControllerRef.current = null;
      }
    },
    [conversationId, token, callbacks]
  );

  return {
    isStreaming,
    error,
    currentContent,
    toolCalls,
    subagents,
    uncertainties,
    pipeline,
    status,
    activities,
    pendingQuestions,
    sendMessage,
    cancel,
    reset,
    dismissQuestion,
  };
}
