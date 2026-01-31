/**
 * Displays source analysis summary when no node/edge is selected.
 * Shows file counts, types, sizes, and any warnings.
 */

import type { SourceAnalysis } from '@/types/agent';
import { FileText, AlertTriangle, HardDrive, Files } from 'lucide-react';

interface SourceAnalysisSummaryProps {
  analysis: SourceAnalysis | null;
}

/**
 * Format bytes to human-readable size.
 */
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export function SourceAnalysisSummary({ analysis }: SourceAnalysisSummaryProps) {
  // Handle null/undefined/non-object analysis
  if (!analysis || typeof analysis !== 'object') {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-6">
        <FileText className="w-12 h-12 text-[var(--text-muted)] mb-3" />
        <p className="text-[var(--text-muted)]">No source analysis available</p>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Select a node or edge to configure
        </p>
      </div>
    );
  }

  // Handle field name mismatch: backend returns 'by_extension', frontend type expects 'file_types'
  // Also handle nested structure: backend returns {count, total_size_bytes, ...}, we need flat counts
  const fileTypesRaw =
    analysis.file_types ??
    (analysis as unknown as { by_extension?: Record<string, unknown> })
      .by_extension ??
    {};

  // Defensive check: ensure fileTypesRaw is a valid object before iterating
  const safeFileTypesRaw =
    fileTypesRaw && typeof fileTypesRaw === 'object' && !Array.isArray(fileTypesRaw)
      ? fileTypesRaw
      : {};

  // Flatten if needed: {".pdf": {count: X}} → {".pdf": X}
  const fileTypes: Record<string, number> = {};
  for (const [ext, value] of Object.entries(safeFileTypesRaw)) {
    if (typeof value === 'number') {
      fileTypes[ext] = value;
    } else if (value && typeof value === 'object' && 'count' in value) {
      fileTypes[ext] = (value as { count: number }).count;
    } else {
      fileTypes[ext] = 0;
    }
  }

  // Sort file types by count descending
  const sortedFileTypes = Object.entries(fileTypes).sort(
    ([, a], [, b]) => b - a
  );

  // Calculate total files for percentage - use analysis.total_files if available, otherwise sum
  const totalFiles =
    analysis.total_files ?? Object.values(fileTypes).reduce((a, b) => a + b, 0);

  return (
    <div className="p-4 space-y-6">
      {/* Header */}
      <div>
        <h3 className="text-lg font-semibold text-[var(--text-primary)]">
          Source Analysis
        </h3>
        <p className="text-sm text-[var(--text-muted)] mt-1">
          Overview of files in your data source
        </p>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-[var(--bg-tertiary)] rounded-lg p-4">
          <div className="flex items-center gap-2 text-[var(--text-muted)] mb-1">
            <Files className="w-4 h-4" />
            <span className="text-xs uppercase tracking-wide">Files</span>
          </div>
          <p className="text-2xl font-bold text-[var(--text-primary)]">
            {totalFiles.toLocaleString()}
          </p>
        </div>

        <div className="bg-[var(--bg-tertiary)] rounded-lg p-4">
          <div className="flex items-center gap-2 text-[var(--text-muted)] mb-1">
            <HardDrive className="w-4 h-4" />
            <span className="text-xs uppercase tracking-wide">Size</span>
          </div>
          <p className="text-2xl font-bold text-[var(--text-primary)]">
            {formatBytes(analysis.total_size_bytes ?? 0)}
          </p>
        </div>
      </div>

      {/* File types */}
      <div>
        <h4 className="text-sm font-medium text-[var(--text-secondary)] mb-3">
          File Types
        </h4>
        <div className="space-y-2">
          {sortedFileTypes.map(([ext, count]) => {
            const percentage = totalFiles > 0 ? Math.round((count / totalFiles) * 100) : 0;
            return (
              <div
                key={ext}
                data-testid="file-type-item"
                className="flex items-center gap-3"
              >
                <span className="text-sm font-mono text-[var(--text-primary)] w-12">
                  {ext}
                </span>
                <div className="flex-1 h-2 bg-[var(--bg-tertiary)] rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[var(--text-muted)] rounded-full"
                    style={{ width: `${percentage}%` }}
                  />
                </div>
                <span className="text-sm text-[var(--text-muted)] w-12 text-right">
                  {count}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Warnings */}
      {analysis.warnings && analysis.warnings.length > 0 && (
        <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-4">
          <div className="flex items-start gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="text-sm font-medium text-amber-400 mb-1">
                Warnings
              </h4>
              <ul className="text-sm text-amber-300/80 space-y-1">
                {analysis.warnings.map((warning, i) => (
                  <li key={i}>{warning}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Sample files */}
      {analysis.sample_files && analysis.sample_files.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-[var(--text-secondary)] mb-2">
            Sample Files
          </h4>
          <ul className="text-sm text-[var(--text-muted)] space-y-1 font-mono">
            {analysis.sample_files.slice(0, 5).map((file, i) => (
              <li key={i} className="truncate">
                {file}
              </li>
            ))}
            {analysis.sample_files.length > 5 && (
              <li className="text-[var(--text-muted)]">
                +{analysis.sample_files.length - 5} more...
              </li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}

export default SourceAnalysisSummary;
