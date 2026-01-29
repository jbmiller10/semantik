/**
 * Editor for pipeline edge routing predicates.
 * Allows editing the 'when' clause that controls routing.
 */

import { useCallback, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import type { PipelineEdge, PipelineDAG } from '@/types/pipeline';
import { pipelineApi, type PredicateField } from '@/services/api/v2/pipeline';
import { ArrowRight, Filter, Loader2 } from 'lucide-react';

interface EdgePredicateEditorProps {
  edge: PipelineEdge;
  dag: PipelineDAG;
  fromNodeLabel: string;
  toNodeLabel: string;
  onChange: (edge: PipelineEdge) => void;
  readOnly?: boolean;
}

// Fallback predicate fields (used if API call fails)
const FALLBACK_FIELDS: PredicateField[] = [
  // Source metadata (from connector)
  { value: 'metadata.source.mime_type', label: 'MIME Type', category: 'source' },
  { value: 'metadata.source.extension', label: 'Extension', category: 'source' },
  { value: 'metadata.source.source_type', label: 'Source Type', category: 'source' },
  { value: 'metadata.source.content_type', label: 'Content Type', category: 'source' },
  // Detected metadata (from pre-routing sniff)
  { value: 'metadata.detected.is_scanned_pdf', label: 'Is Scanned PDF', category: 'detected' },
  { value: 'metadata.detected.is_code', label: 'Is Code', category: 'detected' },
  { value: 'metadata.detected.is_structured_data', label: 'Is Structured Data', category: 'detected' },
  // Parsed metadata (from parser, for mid-pipeline routing) - show common fields
  { value: 'metadata.parsed.detected_language', label: 'Detected Language', category: 'parsed' },
  { value: 'metadata.parsed.approx_token_count', label: 'Token Count', category: 'parsed' },
  { value: 'metadata.parsed.has_tables', label: 'Has Tables', category: 'parsed' },
  { value: 'metadata.parsed.has_images', label: 'Has Images', category: 'parsed' },
  { value: 'metadata.parsed.has_code_blocks', label: 'Has Code Blocks', category: 'parsed' },
  { value: 'metadata.parsed.page_count', label: 'Page Count', category: 'parsed' },
];

/**
 * Format predicate value for display (arrays become comma-separated).
 */
function formatPredicateValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.join(', ');
  }
  return String(value ?? '');
}

/**
 * Parse input value (comma-separated becomes array if multiple items).
 */
function parsePredicateValue(input: string): string | string[] {
  const items = input.split(',').map((s) => s.trim()).filter(Boolean);
  return items.length > 1 ? items : items[0] || '';
}

/**
 * Generate a stable hash of DAG structure for cache key.
 */
function getDagHash(dag: PipelineDAG): string {
  // Use node IDs and types for stability
  const nodeKeys = dag.nodes.map((n) => `${n.id}:${n.type}:${n.plugin_id}`).sort().join(',');
  return nodeKeys;
}

export function EdgePredicateEditor({
  edge,
  dag,
  fromNodeLabel,
  toNodeLabel,
  onChange,
  readOnly = false,
}: EdgePredicateEditorProps) {
  const isCatchAll = edge.when === null;

  // Fetch available predicate fields dynamically
  const dagHash = getDagHash(dag);
  const { data: availableFieldsData, isLoading: isLoadingFields } = useQuery({
    queryKey: ['predicate-fields', edge.from_node, dagHash],
    queryFn: () => pipelineApi.getAvailablePredicateFields(dag, edge.from_node),
    staleTime: 30000, // Cache for 30 seconds
    retry: 1,
  });

  // Use fetched fields or fall back to static fields
  const predicateFields = availableFieldsData?.fields ?? FALLBACK_FIELDS;

  // Extract current predicate field and value
  const { field, value } = useMemo(() => {
    if (!edge.when || Object.keys(edge.when).length === 0) {
      return { field: 'metadata.source.mime_type', value: '' };
    }
    const [f, v] = Object.entries(edge.when)[0];
    return { field: f, value: v };
  }, [edge.when]);

  // Handle catch-all toggle
  const handleCatchAllToggle = useCallback(
    (checked: boolean) => {
      if (checked) {
        onChange({ ...edge, when: null });
      } else {
        onChange({ ...edge, when: { 'metadata.source.mime_type': '' } });
      }
    },
    [edge, onChange]
  );

  // Handle field change
  const handleFieldChange = useCallback(
    (newField: string) => {
      const currentValue = edge.when ? Object.values(edge.when)[0] : '';
      onChange({
        ...edge,
        when: { [newField]: currentValue },
      });
    },
    [edge, onChange]
  );

  // Handle value change
  const handleValueChange = useCallback(
    (input: string) => {
      const parsed = parsePredicateValue(input);
      onChange({
        ...edge,
        when: { [field]: parsed },
      });
    },
    [edge, field, onChange]
  );

  // Group fields by category
  const fieldsByCategory = useMemo(() => {
    const grouped: Record<string, PredicateField[]> = {
      source: [],
      detected: [],
      parsed: [],
    };
    for (const f of predicateFields) {
      grouped[f.category]?.push(f);
    }
    return grouped;
  }, [predicateFields]);

  // Find example for currently selected field
  const currentFieldExample = useMemo(() => {
    // Simple examples based on field type
    if (field.includes('mime_type')) return 'application/pdf';
    if (field.includes('extension')) return '.pdf, .docx';
    if (field.includes('source_type')) return 'directory';
    if (field.includes('content_type')) return 'file';
    if (field.includes('is_') || field.includes('has_')) return 'true';
    if (field.includes('language')) return 'en, zh';
    if (field.includes('token_count') || field.includes('page_count') || field.includes('line_count')) return '>10000';
    return '';
  }, [field]);

  return (
    <div className="p-4 space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Filter className="w-5 h-5 text-[var(--text-muted)]" />
        <h3 className="text-lg font-semibold text-[var(--text-primary)]">
          Edge Routing
        </h3>
      </div>

      {/* Edge flow visualization */}
      <div className="flex items-center gap-3 p-3 bg-[var(--bg-tertiary)] rounded-lg">
        <span className="text-sm font-medium text-[var(--text-primary)]">
          {fromNodeLabel}
        </span>
        <ArrowRight className="w-4 h-4 text-[var(--text-muted)]" />
        <span className="text-sm font-medium text-[var(--text-primary)]">
          {toNodeLabel}
        </span>
      </div>

      {/* Catch-all toggle */}
      <div className="flex items-center gap-3">
        <input
          type="checkbox"
          id="catch-all"
          aria-label="Catch-all"
          checked={isCatchAll}
          onChange={(e) => handleCatchAllToggle(e.target.checked)}
          disabled={readOnly}
          className="h-4 w-4 rounded border-[var(--border)] bg-[var(--bg-tertiary)]"
        />
        <div>
          <label
            htmlFor="catch-all"
            className="text-sm font-medium text-[var(--text-primary)]"
          >
            Catch-all route
          </label>
          <p className="text-xs text-[var(--text-muted)]">
            Matches all files not matched by other routes
          </p>
        </div>
      </div>

      {/* Predicate editor (hidden when catch-all) */}
      {!isCatchAll && (
        <div className="space-y-4 border-t border-[var(--border)] pt-4">
          <h4 className="text-sm font-medium text-[var(--text-secondary)]">
            Route Condition
          </h4>

          {/* Field selector */}
          <div>
            <label
              htmlFor="predicate-field"
              className="block text-sm font-medium text-[var(--text-primary)] mb-1"
            >
              Field
            </label>
            {isLoadingFields ? (
              <div className="flex items-center gap-2 p-2 text-[var(--text-muted)]">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span className="text-sm">Loading fields...</span>
              </div>
            ) : (
              <select
                id="predicate-field"
                aria-label="Field"
                value={field}
                onChange={(e) => handleFieldChange(e.target.value)}
                disabled={readOnly}
                className="input-field w-full"
              >
                {fieldsByCategory.source.length > 0 && (
                  <optgroup label="Source (from connector)">
                    {fieldsByCategory.source.map((f) => (
                      <option key={f.value} value={f.value}>
                        {f.label}
                      </option>
                    ))}
                  </optgroup>
                )}
                {fieldsByCategory.detected.length > 0 && (
                  <optgroup label="Detected (pre-routing sniff)">
                    {fieldsByCategory.detected.map((f) => (
                      <option key={f.value} value={f.value}>
                        {f.label}
                      </option>
                    ))}
                  </optgroup>
                )}
                {fieldsByCategory.parsed.length > 0 && (
                  <optgroup label="Parsed (mid-pipeline routing)">
                    {fieldsByCategory.parsed.map((f) => (
                      <option key={f.value} value={f.value}>
                        {f.label}
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>
            )}
            {/* Show info message when no parsed fields available */}
            {!isLoadingFields && fieldsByCategory.parsed.length === 0 && edge.from_node === '_source' && (
              <p className="text-xs text-[var(--text-muted)] mt-1 italic">
                Parsed fields are available when routing from a parser node
              </p>
            )}
          </div>

          {/* Value input */}
          <div>
            <label
              htmlFor="predicate-value"
              className="block text-sm font-medium text-[var(--text-primary)] mb-1"
            >
              Value
            </label>
            <input
              type="text"
              id="predicate-value"
              value={formatPredicateValue(value)}
              onChange={(e) => handleValueChange(e.target.value)}
              placeholder={currentFieldExample}
              disabled={readOnly}
              className="input-field w-full"
            />
            <p className="text-xs text-[var(--text-muted)] mt-1">
              Use commas to match multiple values (OR logic)
            </p>
          </div>

          {/* Preview */}
          <div className="p-3 bg-[var(--bg-secondary)] rounded-lg border border-[var(--border)]">
            <p className="text-xs text-[var(--text-muted)] mb-1">Preview:</p>
            <code className="text-sm text-[var(--text-primary)] font-mono">
              {field}: {JSON.stringify(value)}
            </code>
          </div>
        </div>
      )}
    </div>
  );
}

export default EdgePredicateEditor;
