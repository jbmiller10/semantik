/**
 * Results display component for route preview.
 * Shows file info, detected metadata, routing path, and detailed evaluation.
 */

import { useState } from 'react';
import { FileText, Zap, Route, Clock, AlertTriangle, ChevronDown, ChevronRight } from 'lucide-react';
import type { RoutePreviewResponse } from '@/types/routePreview';
import type { PipelineDAG } from '@/types/pipeline';
import { PathVisualization } from './PathVisualization';
import { EdgeEvaluationTree } from './EdgeEvaluationTree';

interface RoutePreviewResultsProps {
  /** The preview result */
  result: RoutePreviewResponse;
  /** The pipeline DAG */
  dag: PipelineDAG;
}

/**
 * Format milliseconds for display.
 */
function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

/**
 * Collapsible section component.
 */
function Section({
  title,
  icon,
  children,
  defaultExpanded = true,
  badge,
}: {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultExpanded?: boolean;
  badge?: React.ReactNode;
}) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  return (
    <div className="border border-[var(--border)] rounded-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-2 px-3 py-2 bg-[var(--bg-secondary)] hover:bg-[var(--bg-tertiary)] transition-colors text-left"
      >
        <span className="text-[var(--text-muted)]">
          {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </span>
        <span className="text-[var(--text-muted)]">{icon}</span>
        <span className="text-sm font-medium text-[var(--text-primary)]">{title}</span>
        {badge && <span className="ml-auto">{badge}</span>}
      </button>
      {isExpanded && <div className="p-3 border-t border-[var(--border)]">{children}</div>}
    </div>
  );
}

/**
 * Key-value display row.
 */
function InfoRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-start gap-2 py-1">
      <span className="text-xs text-[var(--text-muted)] w-28 flex-shrink-0">{label}:</span>
      <span className="text-xs text-[var(--text-primary)] break-all">{value}</span>
    </div>
  );
}

/**
 * Metadata display component.
 */
function MetadataDisplay({ data, title }: { data: Record<string, unknown>; title: string }) {
  if (!data || Object.keys(data).length === 0) {
    return <p className="text-xs text-[var(--text-muted)] italic">No {title.toLowerCase()} data</p>;
  }

  return (
    <div className="space-y-1">
      {Object.entries(data).map(([key, value]) => (
        <InfoRow
          key={key}
          label={key}
          value={
            <span className="font-mono">
              {typeof value === 'boolean' ? (
                <span className={value ? 'text-green-400' : 'text-[var(--text-muted)]'}>
                  {String(value)}
                </span>
              ) : typeof value === 'object' ? (
                JSON.stringify(value)
              ) : (
                String(value)
              )}
            </span>
          }
        />
      ))}
    </div>
  );
}

export function RoutePreviewResults({ result, dag }: RoutePreviewResultsProps) {
  const hasWarnings = result.warnings.length > 0;
  const routeFound = result.path.length > 1;

  return (
    <div className="space-y-3">
      {/* Warnings banner */}
      {hasWarnings && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
          <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            {result.warnings.map((warning, idx) => (
              <p key={idx} className="text-xs text-amber-400">
                {warning}
              </p>
            ))}
          </div>
        </div>
      )}

      {/* Timing badge */}
      <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
        <Clock className="w-3 h-3" />
        <span>Preview completed in {formatDuration(result.total_duration_ms)}</span>
      </div>

      {/* Route Path (most important) */}
      <Section
        title="Routing Path"
        icon={<Route className="w-4 h-4" />}
        defaultExpanded={true}
        badge={
          routeFound ? (
            result.paths && result.paths.length > 1 ? (
              <span className="text-xs text-blue-400">{result.paths.length} paths (parallel)</span>
            ) : (
              <span className="text-xs text-green-400">{result.path.length} nodes</span>
            )
          ) : (
            <span className="text-xs text-red-400">No route</span>
          )
        }
      >
        {routeFound ? (
          <PathVisualization path={result.path} paths={result.paths} dag={dag} />
        ) : (
          <p className="text-sm text-[var(--text-muted)] italic">
            No matching route found for this file
          </p>
        )}
      </Section>

      {/* File Information */}
      <Section
        title="File Information"
        icon={<FileText className="w-4 h-4" />}
        defaultExpanded={false}
      >
        <div className="space-y-1">
          <InfoRow label="Filename" value={result.file_info.filename} />
          <InfoRow label="Extension" value={result.file_info.extension || 'none'} />
          <InfoRow label="MIME Type" value={result.file_info.mime_type || 'unknown'} />
          <InfoRow
            label="Size"
            value={`${result.file_info.size_bytes.toLocaleString()} bytes`}
          />
        </div>
      </Section>

      {/* Detected Metadata */}
      {result.sniff_result && Object.keys(result.sniff_result).length > 0 && (
        <Section
          title="Detected Content"
          icon={<Zap className="w-4 h-4" />}
          defaultExpanded={false}
        >
          <MetadataDisplay data={result.sniff_result} title="detected" />
        </Section>
      )}

      {/* Parsed Metadata */}
      {result.parsed_metadata && Object.keys(result.parsed_metadata).length > 0 && (
        <Section
          title="Parser Output"
          icon={<FileText className="w-4 h-4" />}
          defaultExpanded={false}
        >
          <MetadataDisplay data={result.parsed_metadata} title="parsed" />
        </Section>
      )}

      {/* Detailed Edge Evaluation */}
      <Section
        title="Routing Details"
        icon={<Route className="w-4 h-4" />}
        defaultExpanded={false}
        badge={
          <span className="text-xs text-[var(--text-muted)]">
            {result.routing_stages.length} stages
          </span>
        }
      >
        <EdgeEvaluationTree stages={result.routing_stages} dag={dag} />
      </Section>
    </div>
  );
}

export default RoutePreviewResults;
