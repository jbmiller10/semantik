/**
 * Pipeline template type definitions.
 * These types match the backend template schemas.
 */

/**
 * A tunable parameter that can be adjusted in a template
 */
export interface TunableParameter {
  path: string;
  description: string;
  default: unknown;
  range?: [number, number] | null;
  options?: string[] | null;
}

/**
 * Node type in a pipeline DAG
 */
export type NodeType = 'parser' | 'chunker' | 'extractor' | 'embedder';

/**
 * A processing node in the pipeline DAG
 */
export interface PipelineNode {
  id: string;
  type: NodeType;
  plugin_id: string;
  config: Record<string, unknown>;
}

/**
 * An edge connecting nodes in the pipeline DAG
 */
export interface PipelineEdge {
  from_node: string;
  to_node: string;
  when?: Record<string, unknown> | null;
}

/**
 * Complete pipeline DAG definition
 */
export interface PipelineDAG {
  id: string;
  version: string;
  nodes: PipelineNode[];
  edges: PipelineEdge[];
}

/**
 * Summary information for a template (used in list views)
 */
export interface TemplateSummary {
  id: string;
  name: string;
  description: string;
  suggested_for: string[];
}

/**
 * Full template details including pipeline DAG
 */
export interface TemplateDetail extends TemplateSummary {
  pipeline: PipelineDAG;
  tunable: TunableParameter[];
}

/**
 * Response from listing templates
 */
export interface TemplateListResponse {
  templates: TemplateSummary[];
  total: number;
}
