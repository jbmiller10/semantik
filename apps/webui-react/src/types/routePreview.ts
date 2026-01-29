/**
 * TypeScript types for pipeline route preview functionality.
 * These types match the backend Pydantic schemas in pipeline_schemas.py.
 */

/**
 * Result of evaluating a single predicate field.
 */
export interface FieldEvaluationResult {
  /** Field path that was evaluated (e.g., 'mime_type', 'metadata.detected.is_code') */
  field: string;
  /** Pattern that was tested */
  pattern: unknown;
  /** Actual value from the file reference */
  value: unknown;
  /** Whether the value matched the pattern */
  matched: boolean;
}

/**
 * Result of evaluating a single edge predicate.
 */
export interface EdgeEvaluationResult {
  /** Source node ID */
  from_node: string;
  /** Target node ID */
  to_node: string;
  /** The 'when' clause predicate (null for catch-all) */
  predicate: Record<string, unknown> | null;
  /** Whether the predicate matched */
  matched: boolean;
  /** Status: 'matched' if selected (exclusive), 'matched_parallel' if parallel fan-out, 'not_matched' if evaluated, 'skipped' if not evaluated */
  status: 'matched' | 'matched_parallel' | 'not_matched' | 'skipped';
  /** Detailed field-by-field evaluation results */
  field_evaluations: FieldEvaluationResult[] | null;
  /** Whether edge has parallel=True */
  is_parallel: boolean;
  /** Path name tag from edge */
  path_name: string | null;
}

/**
 * Result of evaluating routing at a pipeline stage.
 */
export interface StageEvaluationResult {
  /** Stage identifier (e.g., 'entry_routing', 'parser_to_next') */
  stage: string;
  /** Node from which routing occurs */
  from_node: string;
  /** All edges that were evaluated at this stage */
  evaluated_edges: EdgeEvaluationResult[];
  /** The node that was selected for this stage (first/primary) */
  selected_node: string | null;
  /** All selected nodes for parallel fan-out (null if single path) */
  selected_nodes: string[] | null;
  /** Metadata state at this point in routing */
  metadata_snapshot: Record<string, unknown>;
}

/**
 * Information about a single execution path through the pipeline.
 */
export interface PathInfo {
  /** Path identifier (from edge path_name or auto-generated) */
  path_name: string;
  /** Ordered list of node IDs in this path */
  nodes: string[];
}

/**
 * Response from the route preview endpoint.
 */
export interface RoutePreviewResponse {
  /** Basic file information */
  file_info: {
    filename: string;
    extension: string | null;
    mime_type: string | null;
    size_bytes: number;
    uri: string;
  };
  /** Content detection results (detected.* fields) */
  sniff_result: Record<string, unknown> | null;
  /** Detailed evaluation results for each routing stage */
  routing_stages: StageEvaluationResult[];
  /** Ordered list of node IDs in the primary path */
  path: string[];
  /** All execution paths for parallel fan-out (null for single-path DAGs) */
  paths: PathInfo[] | null;
  /** Metadata extracted by the parser (parsed.* fields) */
  parsed_metadata: Record<string, unknown> | null;
  /** Total time taken for route preview in milliseconds */
  total_duration_ms: number;
  /** Any warnings encountered during preview */
  warnings: string[];
}

/**
 * State for the route preview hook.
 */
export interface RoutePreviewState {
  /** Whether a preview is in progress */
  isLoading: boolean;
  /** Error message if preview failed */
  error: string | null;
  /** Preview result if successful */
  result: RoutePreviewResponse | null;
  /** The file that was previewed */
  file: File | null;
}
