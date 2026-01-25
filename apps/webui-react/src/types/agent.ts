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
  | 'error';

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
  uncertainties_added: string[];
  tool_calls: Array<{ tool: string; success: boolean }>;
}

export interface ErrorEvent {
  message: string;
  code?: string;
}

export interface AgentStreamEvent {
  event: AgentStreamEventType;
  data: Record<string, unknown>;
}

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

// Full conversation details
export interface ConversationDetail {
  id: string;
  status: ConversationStatus;
  source_id: number | null;
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

export interface CreateConversationRequest {
  source_id: number;
}

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
