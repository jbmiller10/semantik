/**
 * Editor for pipeline edge routing predicates.
 * Allows editing the 'when' clause that controls routing.
 */

import { useCallback, useMemo } from 'react';
import type { PipelineEdge } from '@/types/pipeline';
import { ArrowRight, Filter } from 'lucide-react';

interface EdgePredicateEditorProps {
  edge: PipelineEdge;
  fromNodeLabel: string;
  toNodeLabel: string;
  onChange: (edge: PipelineEdge) => void;
  readOnly?: boolean;
}

// Common predicate fields for routing
const PREDICATE_FIELDS = [
  { value: 'mime_type', label: 'MIME Type', example: 'application/pdf' },
  { value: 'extension', label: 'Extension', example: '.pdf, .docx' },
  { value: 'source_type', label: 'Source Type', example: 'directory' },
  { value: 'content_type', label: 'Content Type', example: 'file' },
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

export function EdgePredicateEditor({
  edge,
  fromNodeLabel,
  toNodeLabel,
  onChange,
  readOnly = false,
}: EdgePredicateEditorProps) {
  const isCatchAll = edge.when === null;

  // Extract current predicate field and value
  const { field, value } = useMemo(() => {
    if (!edge.when || Object.keys(edge.when).length === 0) {
      return { field: 'mime_type', value: '' };
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
        onChange({ ...edge, when: { mime_type: '' } });
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
            <select
              id="predicate-field"
              aria-label="Field"
              value={field}
              onChange={(e) => handleFieldChange(e.target.value)}
              disabled={readOnly}
              className="input-field w-full"
            >
              {PREDICATE_FIELDS.map((f) => (
                <option key={f.value} value={f.value}>
                  {f.label}
                </option>
              ))}
            </select>
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
              placeholder={PREDICATE_FIELDS.find((f) => f.value === field)?.example}
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
