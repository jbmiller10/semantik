// apps/webui-react/src/components/pipeline/PipelineBuilderHeader.tsx
import { X, Folder, GitBranch, Mail } from 'lucide-react';

export type BuilderMode = 'assisted' | 'manual';

export interface PipelineBuilderHeaderProps {
  sourceName: string;
  sourceType: string;
  mode: BuilderMode;
  onModeChange: (mode: BuilderMode) => void;
  onClose: () => void;
}

const SOURCE_ICONS: Record<string, React.ElementType> = {
  directory: Folder,
  git: GitBranch,
  imap: Mail,
};

export function PipelineBuilderHeader({
  sourceName,
  sourceType,
  mode,
  onModeChange,
  onClose,
}: PipelineBuilderHeaderProps) {
  const SourceIcon = SOURCE_ICONS[sourceType] || Folder;

  return (
    <header className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)] bg-[var(--bg-secondary)]">
      {/* Left: Title and source info */}
      <div className="flex items-center gap-4">
        <h1 className="text-lg font-semibold text-[var(--text-primary)]">
          Pipeline Builder
        </h1>
        <div className="flex items-center gap-2 text-sm text-[var(--text-secondary)]">
          <SourceIcon className="w-4 h-4" />
          <span className="max-w-xs truncate" title={sourceName}>
            {sourceName}
          </span>
        </div>
      </div>

      {/* Right: Mode toggle and close */}
      <div className="flex items-center gap-4">
        {/* Mode toggle */}
        <div className="flex rounded-lg border border-[var(--border)] overflow-hidden">
          <button
            onClick={() => onModeChange('assisted')}
            aria-pressed={mode === 'assisted'}
            className={`px-3 py-1.5 text-sm font-medium transition-colors
              ${mode === 'assisted'
                ? 'bg-[var(--bg-tertiary)] text-[var(--text-primary)]'
                : 'bg-transparent text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
              }`}
          >
            Assisted
          </button>
          <button
            onClick={() => onModeChange('manual')}
            aria-pressed={mode === 'manual'}
            className={`px-3 py-1.5 text-sm font-medium transition-colors border-l border-[var(--border)]
              ${mode === 'manual'
                ? 'bg-[var(--bg-tertiary)] text-[var(--text-primary)]'
                : 'bg-transparent text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
              }`}
          >
            Manual
          </button>
        </div>

        {/* Close button */}
        <button
          onClick={onClose}
          aria-label="Close"
          className="p-2 rounded-lg hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)]
                     focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white"
        >
          <X className="w-5 h-5" />
        </button>
      </div>
    </header>
  );
}
