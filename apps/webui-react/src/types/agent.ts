/**
 * TypeScript types for the agent conversation API.
 * Matches backend schemas from packages/webui/api/v2/agent_schemas.py
 */

// =============================================================================
// Enums and Literal Types
// =============================================================================

export type ConversationStatus = 'active' | 'applied' | 'abandoned';

export type UncertaintySeverity = 'blocking' | 'notable' | 'info';

export type MessageRole = 'user' | 'assistant' | 'tool' | 'subagent';

// =============================================================================
// SSE Streaming Types
// =============================================================================

export type AgentStreamEventType =
  | 'tool_call_start'
  | 'tool_call_end'
  | 'subagent_start'
  | 'subagent_end'
  | 'uncertainty'
  | 'pipeline_update'
  | 'content'
  | 'done'
  | 'error'
  | 'status'
  | 'activity'
  | 'question'; // Agent question requiring user input

export interface ToolCallStartEvent {
  tool: string;
  arguments: Record<string, unknown>;
}

export interface ToolCallEndEvent {
  tool: string;
  success: boolean;
  result?: Record<string, unknown>;
  error?: string;
}

export interface SubagentStartEvent {
  name: string;
  task: string;
}

export interface SubagentEndEvent {
  name: string;
  success: boolean;
  result?: string;
  error?: string;
}

export interface UncertaintyEvent {
  id: string;
  severity: UncertaintySeverity;
  message: string;
  context?: Record<string, unknown>;
}

export interface PipelineUpdateEvent {
  pipeline: PipelineConfig;
}

export interface ContentEvent {
  text: string;
}

export interface DoneEvent {
  pipeline_updated: boolean;
  uncertainties_added: Array<{
    id: string;
    severity: string;
    message: string;
    resolved: boolean;
    context?: Record<string, unknown>;
  }>;
  tool_calls: Array<{ tool: string; success: boolean }>;
  max_turns_reached?: boolean;
}

export interface ErrorEvent {
  message: string;
  code?: string;
}

/** Agent phase during pipeline building */
export type AgentPhase = 'idle' | 'analyzing' | 'sampling' | 'building' | 'validating' | 'ready';

/** Status update event from agent */
export interface StatusEvent {
  phase: AgentPhase;
  message: string;
  progress?: { current: number; total: number };
}

/** Activity log entry from agent */
export interface ActivityEvent {
  message: string;
  timestamp: string; // ISO 8601
}

/** Option for an agent question */
export interface QuestionOption {
  id: string;
  label: string;
  description?: string;
}

/** Question event from agent requiring user input */
export interface QuestionEvent {
  id: string;
  message: string;
  options: QuestionOption[];
  allowCustom: boolean;
}

/**
 * Discriminated union for agent stream events.
 * TypeScript will narrow the data type based on the event field,
 * eliminating the need for explicit type casts in switch statements.
 */
export type AgentStreamEvent =
  | { event: 'content'; data: ContentEvent }
  | { event: 'tool_call_start'; data: ToolCallStartEvent }
  | { event: 'tool_call_end'; data: ToolCallEndEvent }
  | { event: 'subagent_start'; data: SubagentStartEvent }
  | { event: 'subagent_end'; data: SubagentEndEvent }
  | { event: 'uncertainty'; data: UncertaintyEvent }
  | { event: 'pipeline_update'; data: PipelineUpdateEvent }
  | { event: 'done'; data: DoneEvent }
  | { event: 'error'; data: ErrorEvent }
  | { event: 'status'; data: StatusEvent }
  | { event: 'activity'; data: ActivityEvent }
  | { event: 'question'; data: QuestionEvent };

// =============================================================================
// Entity Types
// =============================================================================

export interface Uncertainty {
  id: string;
  severity: UncertaintySeverity;
  message: string;
  resolved: boolean;
  context?: Record<string, unknown>;
}

export interface AgentMessage {
  role: MessageRole;
  content: string;
  timestamp: string; // ISO 8601
  metadata?: Record<string, unknown>;
}

export interface PipelineConfig {
  embedding_model?: string;
  quantization?: string;
  chunking_strategy?: string;
  chunking_config?: Record<string, number | boolean | string>;
  sparse_index_config?: {
    enabled: boolean;
    plugin_id?: string;
    model_config_data?: Record<string, unknown>;
  };
  sync_mode?: 'one_time' | 'continuous';
  sync_interval_minutes?: number;
}

export interface SourceAnalysis {
  total_files: number;
  total_size_bytes: number;
  file_types: Record<string, number>; // e.g., { ".pdf": 10, ".txt": 5 }
  sample_files?: string[];
  warnings?: string[];
}

// Conversation summary (list view)
export interface Conversation {
  id: string;
  status: ConversationStatus;
  source_id: number | null;
  created_at: string; // ISO 8601
}

/**
 * Inline source configuration as returned from the API.
 * Note: _pending_secrets is intentionally NOT included in the response.
 */
export interface InlineSourceConfigResponse {
  source_type: string;
  source_config: Record<string, unknown>;
}

// Full conversation details
export interface ConversationDetail {
  id: string;
  status: ConversationStatus;
  /** ID of an existing source (legacy mode) */
  source_id: number | null;
  /** Inline source configuration (new source to be created on apply) */
  inline_source_config: InlineSourceConfigResponse | null;
  collection_id: string | null;
  current_pipeline: PipelineConfig | null;
  source_analysis: SourceAnalysis | null;
  uncertainties: Uncertainty[];
  messages: AgentMessage[];
  summary: string | null;
  created_at: string; // ISO 8601
  updated_at: string; // ISO 8601
}

// =============================================================================
// Request Types
// =============================================================================

/**
 * Configuration for a new source to be created with the conversation.
 * Used when the user wants to configure a source directly in the guided setup
 * rather than using a pre-existing source.
 */
export interface InlineSourceConfig {
  source_type: string;
  source_config: Record<string, unknown>;
}

/**
 * Request to create a new agent conversation.
 * Uses a discriminated union to enforce XOR: either source_id OR inline_source, not both.
 */
export type CreateConversationRequest =
  | {
      /** ID of an existing collection source to configure */
      source_id: number;
      inline_source?: never;
      secrets?: never;
    }
  | {
      source_id?: never;
      /** Configuration for a new source (created when pipeline is applied) */
      inline_source: InlineSourceConfig;
      /** Secrets for inline_source (e.g., passwords, tokens). Only used with inline_source. */
      secrets?: Record<string, string>;
    };

export interface SendMessageRequest {
  message: string;
}

export interface ApplyPipelineRequest {
  collection_name: string;
  force?: boolean;
}

export interface ResolveUncertaintyRequest {
  uncertainty_id: string;
  resolution: string;
}

// =============================================================================
// Response Types
// =============================================================================

export interface ConversationListResponse {
  conversations: Conversation[];
  total: number;
}

export interface ApplyPipelineResponse {
  collection_id: string;
  collection_name: string;
  operation_id: string | null;
  status: 'created' | 'indexing';
}

export interface AgentMessageResponse {
  response: string;
  pipeline_updated: boolean;
  uncertainties_added: Uncertainty[];
  tool_calls: Array<{ tool: string; success: boolean }>;
}

// =============================================================================
// UI State Types
// =============================================================================

export interface ToolCallState {
  id: string;
  tool: string;
  arguments: Record<string, unknown>;
  status: 'running' | 'success' | 'error';
  result?: Record<string, unknown>;
  error?: string;
}

export interface SubagentState {
  id: string;
  name: string;
  task: string;
  status: 'running' | 'success' | 'error';
  result?: string;
  error?: string;
}
