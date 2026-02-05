/**
 * TypeScript types for the assisted flow API.
 * Matches backend schemas from packages/webui/api/v2/assisted_flow_schemas.py
 */

// =============================================================================
// Request/Response Types
// =============================================================================

/** Configuration for a new source to be created with the session */
export interface InlineSourceConfig {
  /** Type of source (directory, git, imap) */
  source_type: string;
  /** Source-specific configuration */
  source_config: Record<string, unknown>;
}

/**
 * Request to start an assisted flow session.
 * Either source_id OR inline_source must be provided, not both.
 */
export type StartFlowRequest =
  | {
      /** Integer ID of an existing collection source */
      source_id: number;
      inline_source?: never;
      secrets?: never;
    }
  | {
      source_id?: never;
      /** Configuration for a new source (created when pipeline is applied) */
      inline_source: InlineSourceConfig;
      /** Secrets for inline_source (passwords, tokens) */
      secrets?: Record<string, string>;
    };

export interface StartFlowResponse {
  /** SDK session ID for resuming */
  session_id: string;
  /** Name of the source being configured */
  source_name: string;
}

export interface SendMessageRequest {
  /** User message to send */
  message: string;
}

// =============================================================================
// SSE Streaming Types
// =============================================================================

/** Event types from the assisted flow streaming endpoint */
export type AssistedFlowEventType =
  | 'started'
  | 'text'
  | 'tool_use'
  | 'tool_result'
  | 'question'
  | 'done'
  | 'error';

// =============================================================================
// Question Types (for AskUserQuestion handling)
// =============================================================================

/** A single option in a question */
export interface QuestionOption {
  label: string;
  description: string;
}

/** A single question item from AskUserQuestion */
export interface QuestionItem {
  question: string;
  header: string;
  options: QuestionOption[];
  multiSelect: boolean;
}

/** Data from a question SSE event */
export interface QuestionEventData {
  question_id: string;
  questions: QuestionItem[];
}

/** Text content event data from the agent */
export interface TextEventData {
  type?: 'text';
  content?: string;
}

/** Tool use event data - agent is executing a tool */
export interface ToolUseEventData {
  type?: 'tool_use';
  tool_use_id?: string;
  tool_name?: string;
  arguments?: Record<string, unknown>;
}

/** Tool result event data - tool execution completed */
export interface ToolResultEventData {
  type?: 'tool_result';
  tool_use_id?: string;
  tool_name?: string;
  result?: unknown;
  success?: boolean;
}

/** Done event data - stream complete */
export interface DoneEventData {
  status?: 'complete';
}

/** Error event data from the stream */
export interface ErrorEventData {
  message?: string;
  code?: string;
}

/**
 * Discriminated union for assisted flow stream events.
 */
export type AssistedFlowEvent =
  | { event: 'text'; data: TextEventData }
  | { event: 'tool_use'; data: ToolUseEventData }
  | { event: 'tool_result'; data: ToolResultEventData }
  | { event: 'question'; data: QuestionEventData }
  | { event: 'done'; data: DoneEventData }
  | { event: 'error'; data: ErrorEventData };

// =============================================================================
// UI State Types
// =============================================================================

export interface AssistedFlowToolCall {
  id: string;
  tool_name: string;
  arguments?: Record<string, unknown>;
  status: 'running' | 'success' | 'error';
  result?: unknown;
  error?: string;
}

export interface AssistedFlowSession {
  session_id: string;
  source_name: string;
  is_active: boolean;
  content: string;
  tool_calls: AssistedFlowToolCall[];
}
