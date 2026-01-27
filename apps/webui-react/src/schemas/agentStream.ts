/**
 * Zod schemas for runtime validation of agent streaming events.
 *
 * These schemas provide runtime type validation for SSE events from the agent API,
 * catching malformed data before it causes errors in the UI.
 */

import { z } from 'zod';

// =============================================================================
// Base Schemas
// =============================================================================

export const UncertaintySeveritySchema = z.enum(['blocking', 'notable', 'info']);

export const AgentPhaseSchema = z.enum(['idle', 'analyzing', 'sampling', 'building', 'validating', 'ready']);

// =============================================================================
// Event Data Schemas
// =============================================================================

export const ContentEventSchema = z.object({
  text: z.string(),
});

export const ToolCallStartEventSchema = z.object({
  tool: z.string(),
  arguments: z.record(z.unknown()),
});

export const ToolCallEndEventSchema = z.object({
  tool: z.string(),
  success: z.boolean(),
  result: z.record(z.unknown()).optional(),
  error: z.string().optional(),
});

export const SubagentStartEventSchema = z.object({
  name: z.string(),
  task: z.string(),
});

export const SubagentEndEventSchema = z.object({
  name: z.string(),
  success: z.boolean(),
  result: z.string().optional(),
  error: z.string().optional(),
});

export const UncertaintyEventSchema = z.object({
  id: z.string(),
  severity: UncertaintySeveritySchema,
  message: z.string(),
  context: z.record(z.unknown()).optional(),
});

export const PipelineConfigSchema = z.object({
  embedding_model: z.string().optional(),
  quantization: z.string().optional(),
  chunking_strategy: z.string().optional(),
  chunking_config: z.record(z.union([z.number(), z.boolean(), z.string()])).optional(),
  sparse_index_config: z
    .object({
      enabled: z.boolean(),
      plugin_id: z.string().optional(),
      model_config_data: z.record(z.unknown()).optional(),
    })
    .optional(),
  sync_mode: z.enum(['one_time', 'continuous']).optional(),
  sync_interval_minutes: z.number().optional(),
});

export const PipelineUpdateEventSchema = z.object({
  pipeline: PipelineConfigSchema,
});

export const DoneEventSchema = z.object({
  pipeline_updated: z.boolean(),
  uncertainties_added: z.array(
    z.object({
      id: z.string(),
      severity: z.string(),
      message: z.string(),
      resolved: z.boolean(),
      context: z.record(z.unknown()).optional(),
    })
  ),
  tool_calls: z.array(
    z.object({
      tool: z.string(),
      success: z.boolean(),
    })
  ),
  max_turns_reached: z.boolean().optional(),
});

export const ErrorEventSchema = z.object({
  message: z.string(),
  code: z.string().optional(),
});

export const StatusEventSchema = z.object({
  phase: AgentPhaseSchema,
  message: z.string(),
  progress: z
    .object({
      current: z.number(),
      total: z.number(),
    })
    .optional(),
});

export const ActivityEventSchema = z.object({
  message: z.string(),
  timestamp: z.string(),
});

export const QuestionOptionSchema = z.object({
  id: z.string(),
  label: z.string(),
  description: z.string().optional(),
});

export const QuestionEventSchema = z.object({
  id: z.string(),
  message: z.string(),
  options: z.array(QuestionOptionSchema),
  allowCustom: z.boolean(),
});

// =============================================================================
// Event Type Mapping
// =============================================================================

/**
 * Map of event types to their corresponding Zod schemas.
 * Used for runtime validation of SSE event data.
 */
export const EventSchemaMap = {
  content: ContentEventSchema,
  tool_call_start: ToolCallStartEventSchema,
  tool_call_end: ToolCallEndEventSchema,
  subagent_start: SubagentStartEventSchema,
  subagent_end: SubagentEndEventSchema,
  uncertainty: UncertaintyEventSchema,
  pipeline_update: PipelineUpdateEventSchema,
  done: DoneEventSchema,
  error: ErrorEventSchema,
  status: StatusEventSchema,
  activity: ActivityEventSchema,
  question: QuestionEventSchema,
} as const;

export type AgentEventType = keyof typeof EventSchemaMap;

/**
 * Validate an SSE event's data against its expected schema.
 *
 * @param eventType - The type of event (e.g., 'content', 'status')
 * @param data - The event data to validate
 * @returns The validated data if successful, null if validation fails
 */
export function validateEventData<T extends AgentEventType>(
  eventType: T,
  data: unknown
): z.infer<(typeof EventSchemaMap)[T]> | null {
  const schema = EventSchemaMap[eventType];
  if (!schema) {
    console.warn(`No schema found for event type: ${eventType}`);
    return null;
  }

  const result = schema.safeParse(data);
  if (!result.success) {
    console.error(`Validation failed for ${eventType} event:`, result.error.errors);
    return null;
  }

  return result.data;
}

// Type exports for use with validated data
export type ValidatedContentEvent = z.infer<typeof ContentEventSchema>;
export type ValidatedToolCallStartEvent = z.infer<typeof ToolCallStartEventSchema>;
export type ValidatedToolCallEndEvent = z.infer<typeof ToolCallEndEventSchema>;
export type ValidatedSubagentStartEvent = z.infer<typeof SubagentStartEventSchema>;
export type ValidatedSubagentEndEvent = z.infer<typeof SubagentEndEventSchema>;
export type ValidatedUncertaintyEvent = z.infer<typeof UncertaintyEventSchema>;
export type ValidatedPipelineUpdateEvent = z.infer<typeof PipelineUpdateEventSchema>;
export type ValidatedDoneEvent = z.infer<typeof DoneEventSchema>;
export type ValidatedErrorEvent = z.infer<typeof ErrorEventSchema>;
export type ValidatedStatusEvent = z.infer<typeof StatusEventSchema>;
export type ValidatedActivityEvent = z.infer<typeof ActivityEventSchema>;
export type ValidatedQuestionEvent = z.infer<typeof QuestionEventSchema>;
