/**
 * Pipeline preview sidebar component.
 * Displays current pipeline config, source analysis, and apply controls.
 */

import { useState } from 'react';
import { UncertaintyBanner } from './UncertaintyBanner';
import type { PipelineConfig, SourceAnalysis, Uncertainty } from '../../types/agent';

interface PipelinePreviewProps {
  pipeline: PipelineConfig | null;
  sourceAnalysis: SourceAnalysis | null;
  uncertainties: Uncertainty[];
  onApply: (collectionName: string) => void;
  isApplying: boolean;
  canApply: boolean;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export function PipelinePreview({
  pipeline,
  sourceAnalysis,
  uncertainties,
  onApply,
  isApplying,
  canApply,
}: PipelinePreviewProps) {
  const [collectionName, setCollectionName] = useState('');

  const hasBlockingUncertainties = uncertainties.some(
    (u) => u.severity === 'blocking' && !u.resolved
  );

  const handleApply = () => {
    if (collectionName.trim() && canApply && !hasBlockingUncertainties) {
      onApply(collectionName.trim());
    }
  };

  return (
    <div className="h-full flex flex-col bg-[var(--bg-secondary)] border-l border-[var(--border)]">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[var(--border)]">
        <h3 className="text-sm font-semibold text-[var(--text-primary)]">
          Pipeline Preview
        </h3>
        <p className="text-xs text-[var(--text-muted)] mt-0.5">
          Current configuration
        </p>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Pipeline Configuration */}
        <section>
          <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wide mb-2">
            Configuration
          </h4>
          {pipeline ? (
            <div className="space-y-3">
              {/* Embedding Model */}
              <div className="bg-[var(--bg-tertiary)] rounded-lg p-3 border border-[var(--border-subtle)]">
                <div className="flex items-center gap-2 mb-1">
                  <svg
                    className="w-4 h-4 text-[var(--text-muted)]"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z"
                    />
                  </svg>
                  <span className="text-xs text-[var(--text-muted)]">
                    Embedding
                  </span>
                </div>
                <p className="text-sm text-[var(--text-primary)] font-mono">
                  {pipeline.embedding_model || 'Not set'}
                </p>
                {pipeline.quantization && (
                  <p className="text-xs text-[var(--text-secondary)] mt-1">
                    Quantization: {pipeline.quantization}
                  </p>
                )}
              </div>

              {/* Chunking Strategy */}
              <div className="bg-[var(--bg-tertiary)] rounded-lg p-3 border border-[var(--border-subtle)]">
                <div className="flex items-center gap-2 mb-1">
                  <svg
                    className="w-4 h-4 text-[var(--text-muted)]"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 6h16M4 10h16M4 14h16M4 18h16"
                    />
                  </svg>
                  <span className="text-xs text-[var(--text-muted)]">
                    Chunking
                  </span>
                </div>
                <p className="text-sm text-[var(--text-primary)]">
                  {pipeline.chunking_strategy || 'Not set'}
                </p>
                {pipeline.chunking_config && (
                  <div className="text-xs text-[var(--text-secondary)] mt-1 space-y-0.5">
                    {Object.entries(pipeline.chunking_config).map(([key, value]) => (
                      <p key={key}>
                        {key}: {String(value)}
                      </p>
                    ))}
                  </div>
                )}
              </div>

              {/* Sparse Index */}
              {pipeline.sparse_index_config?.enabled && (
                <div className="bg-[var(--bg-tertiary)] rounded-lg p-3 border border-[var(--border-subtle)]">
                  <div className="flex items-center gap-2 mb-1">
                    <svg
                      className="w-4 h-4 text-[var(--text-muted)]"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                      />
                    </svg>
                    <span className="text-xs text-[var(--text-muted)]">
                      Hybrid Search
                    </span>
                  </div>
                  <p className="text-sm text-[var(--text-primary)]">
                    {pipeline.sparse_index_config.plugin_id || 'Enabled'}
                  </p>
                </div>
              )}

              {/* Sync Mode */}
              {pipeline.sync_mode && (
                <div className="bg-[var(--bg-tertiary)] rounded-lg p-3 border border-[var(--border-subtle)]">
                  <div className="flex items-center gap-2 mb-1">
                    <svg
                      className="w-4 h-4 text-[var(--text-muted)]"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                      />
                    </svg>
                    <span className="text-xs text-[var(--text-muted)]">
                      Sync
                    </span>
                  </div>
                  <p className="text-sm text-[var(--text-primary)] capitalize">
                    {pipeline.sync_mode.replace('_', ' ')}
                  </p>
                  {pipeline.sync_interval_minutes && (
                    <p className="text-xs text-[var(--text-secondary)] mt-1">
                      Every {pipeline.sync_interval_minutes} minutes
                    </p>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="bg-[var(--bg-tertiary)] rounded-lg p-4 border border-[var(--border-subtle)] text-center">
              <p className="text-sm text-[var(--text-muted)]">
                No pipeline configured yet
              </p>
              <p className="text-xs text-[var(--text-secondary)] mt-1">
                Chat with the assistant to configure settings
              </p>
            </div>
          )}
        </section>

        {/* Source Analysis */}
        {sourceAnalysis && (
          <section>
            <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wide mb-2">
              Source Analysis
            </h4>
            <div className="bg-[var(--bg-tertiary)] rounded-lg p-3 border border-[var(--border-subtle)]">
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <p className="text-xs text-[var(--text-muted)]">Files</p>
                  <p className="text-[var(--text-primary)] font-medium">
                    {sourceAnalysis.total_files.toLocaleString()}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-[var(--text-muted)]">Size</p>
                  <p className="text-[var(--text-primary)] font-medium">
                    {formatBytes(sourceAnalysis.total_size_bytes)}
                  </p>
                </div>
              </div>
              {sourceAnalysis.file_types &&
                Object.keys(sourceAnalysis.file_types).length > 0 && (
                  <div className="mt-3 pt-3 border-t border-[var(--border-subtle)]">
                    <p className="text-xs text-[var(--text-muted)] mb-1">
                      File types
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(sourceAnalysis.file_types)
                        .slice(0, 5)
                        .map(([ext, count]) => (
                          <span
                            key={ext}
                            className="text-xs bg-[var(--bg-primary)] px-2 py-0.5 rounded text-[var(--text-secondary)]"
                          >
                            {ext} ({count})
                          </span>
                        ))}
                      {Object.keys(sourceAnalysis.file_types).length > 5 && (
                        <span className="text-xs text-[var(--text-muted)]">
                          +{Object.keys(sourceAnalysis.file_types).length - 5} more
                        </span>
                      )}
                    </div>
                  </div>
                )}
            </div>
          </section>
        )}

        {/* Uncertainties */}
        {uncertainties.length > 0 && (
          <section>
            <h4 className="text-xs font-medium text-[var(--text-muted)] uppercase tracking-wide mb-2">
              Uncertainties
            </h4>
            <UncertaintyBanner
              uncertainties={uncertainties}
              showResolved={true}
            />
          </section>
        )}
      </div>

      {/* Apply section */}
      <div className="p-4 border-t border-[var(--border)] bg-[var(--bg-tertiary)]">
        <div className="mb-3">
          <label
            htmlFor="collection-name"
            className="block text-xs font-medium text-[var(--text-muted)] mb-1"
          >
            Collection Name
          </label>
          <input
            type="text"
            id="collection-name"
            value={collectionName}
            onChange={(e) => setCollectionName(e.target.value)}
            placeholder="Enter collection name..."
            className="input-field w-full text-sm"
            disabled={isApplying}
          />
        </div>

        <button
          onClick={handleApply}
          disabled={
            !collectionName.trim() ||
            !canApply ||
            hasBlockingUncertainties ||
            isApplying
          }
          className="btn-primary w-full justify-center disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isApplying ? (
            <>
              <svg
                className="w-4 h-4 animate-spin mr-2"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              Creating...
            </>
          ) : (
            <>
              <svg
                className="w-4 h-4 mr-2"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
              Apply Pipeline
            </>
          )}
        </button>

        {hasBlockingUncertainties && (
          <p className="text-xs text-red-400 mt-2 text-center">
            Resolve blocking issues before applying
          </p>
        )}
      </div>
    </div>
  );
}

export default PipelinePreview;
