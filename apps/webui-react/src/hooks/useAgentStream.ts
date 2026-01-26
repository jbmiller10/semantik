/**
 * Hook for handling SSE streaming from the agent API.
 * Uses fetch with ReadableStream to parse Server-Sent Events.
 */

import { useCallback, useRef, useState } from 'react';
import { useAuthStore } from '../stores/authStore';
import { agentApi } from '../services/api/v2/agent';
import type {
  AgentStreamEventType,
  ToolCallState,
  SubagentState,
  Uncertainty,
  PipelineConfig,
  ContentEvent,
  ToolCallStartEvent,
  ToolCallEndEvent,
  SubagentStartEvent,
  SubagentEndEvent,
  UncertaintyEvent,
  PipelineUpdateEvent,
  DoneEvent,
  ErrorEvent,
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
 */
function parseSSEMessage(raw: string): { event: AgentStreamEventType; data: Record<string, unknown> } | null {
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
    return { event: eventType as AgentStreamEventType, data };
  } catch {
    console.error('Failed to parse SSE data:', dataStr);
    return null;
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
              case 'content': {
                const contentData = data as unknown as ContentEvent;
                setCurrentContent((prev) => prev + contentData.text);
                callbacks.onContent?.(contentData.text);
                break;
              }

              case 'tool_call_start': {
                const startData = data as unknown as ToolCallStartEvent;
                const toolCallId = `${Date.now()}-${startData.tool}`;
                setToolCalls((prev) => [
                  ...prev,
                  {
                    id: toolCallId,
                    tool: startData.tool,
                    arguments: startData.arguments,
                    status: 'running',
                  },
                ]);
                callbacks.onToolCallStart?.(startData);
                break;
              }

              case 'tool_call_end': {
                const endData = data as unknown as ToolCallEndEvent;
                setToolCalls((prev) =>
                  prev.map((tc) =>
                    tc.tool === endData.tool && tc.status === 'running'
                      ? {
                          ...tc,
                          status: endData.success ? 'success' : 'error',
                          result: endData.result,
                          error: endData.error,
                        }
                      : tc
                  )
                );
                callbacks.onToolCallEnd?.(endData);
                break;
              }

              case 'subagent_start': {
                const subagentData = data as unknown as SubagentStartEvent;
                const subagentId = `${Date.now()}-${subagentData.name}`;
                setSubagents((prev) => [
                  ...prev,
                  {
                    id: subagentId,
                    name: subagentData.name,
                    task: subagentData.task,
                    status: 'running',
                  },
                ]);
                callbacks.onSubagentStart?.(subagentData);
                break;
              }

              case 'subagent_end': {
                const subagentEndData = data as unknown as SubagentEndEvent;
                setSubagents((prev) =>
                  prev.map((sa) =>
                    sa.name === subagentEndData.name && sa.status === 'running'
                      ? {
                          ...sa,
                          status: subagentEndData.success ? 'success' : 'error',
                          result: subagentEndData.result,
                          error: subagentEndData.error,
                        }
                      : sa
                  )
                );
                callbacks.onSubagentEnd?.(subagentEndData);
                break;
              }

              case 'uncertainty': {
                const uncertaintyData = data as unknown as UncertaintyEvent;
                setUncertainties((prev) => [
                  ...prev,
                  {
                    id: uncertaintyData.id,
                    severity: uncertaintyData.severity,
                    message: uncertaintyData.message,
                    resolved: false,
                    context: uncertaintyData.context,
                  },
                ]);
                callbacks.onUncertainty?.(uncertaintyData);
                break;
              }

              case 'pipeline_update': {
                const pipelineData = data as unknown as PipelineUpdateEvent;
                setPipeline(pipelineData.pipeline);
                callbacks.onPipelineUpdate?.(pipelineData);
                break;
              }

              case 'done': {
                const doneData = data as unknown as DoneEvent;
                callbacks.onDone?.(doneData);
                break;
              }

              case 'error': {
                const errorData = data as unknown as ErrorEvent;
                setError(errorData.message);
                callbacks.onError?.(errorData.message);
                break;
              }

              case 'status': {
                const statusData = data as unknown as StatusEvent;
                setStatus({
                  phase: statusData.phase,
                  message: statusData.message,
                  progress: statusData.progress,
                });
                callbacks.onStatus?.(statusData);
                break;
              }

              case 'activity': {
                const activityData = data as unknown as ActivityEvent;
                setActivities((prev) => [
                  ...prev,
                  {
                    message: activityData.message,
                    timestamp: activityData.timestamp,
                  },
                ]);
                callbacks.onActivity?.(activityData);
                break;
              }

              case 'question': {
                const questionData = data as unknown as QuestionEvent;
                setPendingQuestions((prev) => [...prev, questionData]);
                callbacks.onQuestion?.(questionData);
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
