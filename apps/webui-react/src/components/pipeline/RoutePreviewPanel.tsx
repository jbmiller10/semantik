/**
 * Route Preview Panel component.
 * Collapsible panel for testing file routing through a pipeline DAG.
 */

import { useState, useCallback, useEffect } from 'react';
import { FlaskConical, ChevronUp, ChevronDown, AlertCircle } from 'lucide-react';
import { useRoutePreview } from '@/hooks/useRoutePreview';
import type { PipelineDAG } from '@/types/pipeline';
import { SampleFileSelector } from './SampleFileSelector';
import { RoutePreviewResults } from './RoutePreviewResults';

interface RoutePreviewPanelProps {
  /** The pipeline DAG to test routing against */
  dag: PipelineDAG;
  /** @deprecated Use onPathsHighlight instead */
  onPathHighlight?: (path: string[] | null) => void;
  /** Callback when paths are computed (for highlighting in visualization) */
  onPathsHighlight?: (paths: string[][] | null) => void;
  /** Whether the panel should be initially collapsed */
  defaultCollapsed?: boolean;
}

export function RoutePreviewPanel({
  dag,
  onPathHighlight,
  onPathsHighlight,
  defaultCollapsed = true,
}: RoutePreviewPanelProps) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);
  const { isLoading, error, result, file, previewFile, clearPreview } = useRoutePreview();

  // Handle file selection
  const handleFileSelect = useCallback(
    async (selectedFile: File) => {
      await previewFile(selectedFile, dag);
    },
    [previewFile, dag]
  );

  // Handle clear
  const handleClear = useCallback(() => {
    clearPreview();
    onPathHighlight?.(null);
    onPathsHighlight?.(null);
  }, [clearPreview, onPathHighlight, onPathsHighlight]);

  // Update path highlight when result changes
  useEffect(() => {
    if (result) {
      if (result.paths && result.paths.length > 1) {
        // Multiple parallel paths
        onPathsHighlight?.(result.paths.map(p => p.nodes));
        // Also call legacy callback with primary path
        onPathHighlight?.(result.path);
      } else if (result.path.length > 0) {
        // Single path
        onPathsHighlight?.([result.path]);
        onPathHighlight?.(result.path);
      } else {
        onPathsHighlight?.(null);
        onPathHighlight?.(null);
      }
    } else {
      onPathsHighlight?.(null);
      onPathHighlight?.(null);
    }
  }, [result, onPathHighlight, onPathsHighlight]);

  // Auto-expand when loading starts
  useEffect(() => {
    if (isLoading && isCollapsed) {
      setIsCollapsed(false);
    }
  }, [isLoading, isCollapsed]);

  return (
    <div className="border-t border-[var(--border)] bg-[var(--bg-primary)]">
      {/* Header */}
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="w-full flex items-center justify-between px-4 py-2 hover:bg-[var(--bg-secondary)] transition-colors"
      >
        <div className="flex items-center gap-2">
          <FlaskConical className="w-4 h-4 text-[var(--text-muted)]" />
          <span className="text-sm font-medium text-[var(--text-primary)]">Test Route</span>
          {result && (
            <span className="text-xs text-[var(--text-muted)]">
              ({result.path.length} nodes)
            </span>
          )}
        </div>
        <span className="text-[var(--text-muted)]">
          {isCollapsed ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </span>
      </button>

      {/* Content */}
      {!isCollapsed && (
        <div className="route-preview-content px-4 pb-4 space-y-4 max-h-[40vh] overflow-y-auto">
          {/* File selector */}
          <SampleFileSelector
            onFileSelect={handleFileSelect}
            selectedFile={file}
            onClear={handleClear}
            isLoading={isLoading}
          />

          {/* Error display */}
          {error && (
            <div className="flex items-start gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30">
              <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}

          {/* Results */}
          {result && !isLoading && <RoutePreviewResults result={result} dag={dag} />}

          {/* Loading state */}
          {isLoading && (
            <div className="flex items-center justify-center py-8">
              <div className="flex items-center gap-2 text-[var(--text-muted)]">
                <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                <span className="text-sm">Analyzing routing...</span>
              </div>
            </div>
          )}

          {/* Empty state */}
          {!file && !result && !isLoading && (
            <p className="text-xs text-[var(--text-muted)] text-center py-2">
              Upload a sample file to see how it would be routed through your pipeline
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default RoutePreviewPanel;
