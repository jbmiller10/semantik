// apps/webui-react/src/components/pipeline/PipelineBuilderFooter.tsx
import { CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

export interface PipelineBuilderFooterProps {
  isReady: boolean;
  fileCount: number;
  nodeCount: number;
  onValidate: () => void;
  onApply: () => void;
  isValidating: boolean;
  isApplying: boolean;
  validationResult?: {
    passed: number;
    failed: number;
    errors?: string[];
  };
}

export function PipelineBuilderFooter({
  isReady,
  fileCount,
  nodeCount,
  onValidate,
  onApply,
  isValidating,
  isApplying,
  validationResult,
}: PipelineBuilderFooterProps) {
  const isDisabled = isValidating || isApplying;

  return (
    <footer className="flex items-center justify-between px-4 py-3 border-t border-[var(--border)] bg-[var(--bg-secondary)]">
      {/* Left: Status and stats */}
      <div className="flex items-center gap-4">
        {/* Ready status */}
        <div className="flex items-center gap-2">
          {isReady ? (
            <>
              <CheckCircle className="w-5 h-5 text-green-500" />
              <span className="text-sm font-medium text-green-500">Pipeline ready</span>
            </>
          ) : (
            <>
              <AlertCircle className="w-5 h-5 text-amber-500" />
              <span className="text-sm font-medium text-amber-500">Pipeline incomplete</span>
            </>
          )}
        </div>

        {/* Stats */}
        <div className="flex items-center gap-3 text-sm text-[var(--text-muted)]">
          <span>{fileCount} files</span>
          <span>•</span>
          <span>{nodeCount} nodes</span>
        </div>

        {/* Validation result */}
        {validationResult && (
          <div className="flex items-center gap-2 text-sm">
            {validationResult.failed > 0 ? (
              <span className="text-amber-500">
                {validationResult.failed} of {validationResult.passed + validationResult.failed} samples failed
              </span>
            ) : (
              <span className="text-green-500">
                All {validationResult.passed} samples passed
              </span>
            )}
          </div>
        )}
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-3">
        <button
          onClick={onValidate}
          disabled={isDisabled}
          title="Validate pipeline (⌘S)"
          className="px-4 py-2 text-sm font-medium text-[var(--text-secondary)]
                     border border-[var(--border)] rounded-lg
                     hover:bg-[var(--bg-tertiary)] hover:text-[var(--text-primary)]
                     focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white
                     disabled:opacity-50 disabled:cursor-not-allowed
                     transition-colors"
        >
          {isValidating ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin inline mr-2" />
              Validating...
            </>
          ) : (
            'Validate with samples'
          )}
        </button>

        <button
          onClick={onApply}
          disabled={!isReady || isDisabled}
          title="Apply and start indexing (⌘↵)"
          className="px-4 py-2 text-sm font-bold text-gray-900 bg-gray-200 dark:bg-white
                     rounded-lg shadow-lg
                     hover:bg-gray-300 dark:hover:bg-gray-100
                     focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white
                     disabled:opacity-50 disabled:cursor-not-allowed
                     transition-all transform active:scale-95"
        >
          {isApplying ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin inline mr-2" />
              Applying...
            </>
          ) : (
            'Apply & Start Indexing'
          )}
        </button>
      </div>
    </footer>
  );
}
