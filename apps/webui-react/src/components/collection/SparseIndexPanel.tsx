/**
 * SparseIndexPanel Component
 *
 * Displays sparse index status and management controls for a collection.
 * Shows current status, plugin info, and allows enabling/disabling/reindexing.
 */

import { useState } from 'react';
import { Zap, Loader2, CheckCircle, XCircle, RefreshCw, Trash2, AlertCircle } from 'lucide-react';
import { useSparseIndexWithReindex, useSparseReindexProgress } from '../../hooks/useSparseIndex';
import { SparseIndexConfigModal } from './SparseIndexConfigModal';
import { SPARSE_PLUGIN_INFO } from '../../types/sparse-index';
import type { Collection } from '../../types/collection';
import type { EnableSparseIndexRequest, SparseIndexerPlugin } from '../../types/sparse-index';

interface SparseIndexPanelProps {
  /** Collection to manage sparse index for */
  collection: Collection;
}

export function SparseIndexPanel({ collection }: SparseIndexPanelProps) {
  const [showConfigModal, setShowConfigModal] = useState(false);

  const {
    status,
    isLoading,
    isError,
    enable,
    isEnabling,
    disable,
    isDisabling,
    triggerReindex,
    isReindexing,
    reindexJobId,
  } = useSparseIndexWithReindex(collection.id);

  const { data: reindexProgress } = useSparseReindexProgress(
    collection.id,
    reindexJobId
  );

  const handleEnable = (config: EnableSparseIndexRequest) => {
    enable({
      collectionUuid: collection.id,
      data: config,
    });
    setShowConfigModal(false);
  };

  const handleDisable = () => {
    if (
      window.confirm(
        'Are you sure you want to disable sparse indexing? This will delete the sparse index.'
      )
    ) {
      disable(collection.id);
    }
  };

  const handleReindex = () => {
    triggerReindex(collection.id);
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="p-6 bg-void-900/30 rounded-2xl border border-white/5">
        <div className="flex items-center justify-center py-6">
          <Loader2 className="h-6 w-6 animate-spin text-signal-500" />
          <span className="ml-2 text-sm text-gray-400">Loading sparse index status...</span>
        </div>
      </div>
    );
  }

  // Error state
  if (isError) {
    return (
      <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl">
        <div className="flex items-center gap-2 text-red-300">
          <AlertCircle className="h-5 w-5" />
          <span className="text-sm">Failed to load sparse index status</span>
        </div>
      </div>
    );
  }

  // Determine if reindex is in progress
  const isReindexInProgress =
    reindexProgress &&
    (reindexProgress.status === 'pending' || reindexProgress.status === 'processing');

  // Plugin info for display
  const pluginId = status?.plugin_id as SparseIndexerPlugin | undefined;
  const pluginInfo = pluginId ? SPARSE_PLUGIN_INFO[pluginId] : null;

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Zap className="h-5 w-5 text-signal-500" />
          <h4 className="text-sm font-bold text-gray-200 uppercase tracking-wide">Sparse Indexing</h4>
        </div>
        {status?.enabled ? (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold bg-teal-500/10 text-teal-400 border border-teal-500/20">
            <CheckCircle className="h-3 w-3 mr-1" />
            Enabled
          </span>
        ) : (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold bg-white/5 text-gray-400 border border-white/10">
            <XCircle className="h-3 w-3 mr-1" />
            Disabled
          </span>
        )}
      </div>

      {status?.enabled ? (
        /* Enabled State */
        <div className="p-6 bg-signal-500/5 border border-signal-500/20 rounded-2xl space-y-5">
          {/* Status Info */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <dt className="text-xs text-signal-400 font-bold uppercase tracking-wider">Plugin</dt>
              <dd className="mt-1 text-sm font-medium text-white">
                {pluginInfo?.name || status.plugin_id}
              </dd>
            </div>
            <div>
              <dt className="text-xs text-signal-400 font-bold uppercase tracking-wider">Vectors</dt>
              <dd className="mt-1 text-sm font-medium text-white">
                {status.document_count?.toLocaleString() || '0'}
              </dd>
            </div>
          </div>

          {/* BM25 Parameters (if applicable) */}
          {status.plugin_id === 'bm25-local' && status.model_config_data && (
            <div className="pt-4 border-t border-signal-500/10">
              <dt className="text-xs text-signal-400 font-bold uppercase tracking-wider mb-2">
                BM25 Parameters
              </dt>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-sm">
                  <span className="text-gray-400">k1:</span>{' '}
                  <span className="font-mono text-white">
                    {(status.model_config_data as { k1?: number }).k1 ?? 1.5}
                  </span>
                </div>
                <div className="text-sm">
                  <span className="text-gray-400">b:</span>{' '}
                  <span className="font-mono text-white">
                    {(status.model_config_data as { b?: number }).b ?? 0.75}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Reindex Progress */}
          {isReindexInProgress && reindexProgress && (
            <div className="pt-4 border-t border-signal-500/10">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-white">
                  Reindexing in progress
                </span>
                <span className="text-sm text-signal-400 font-mono">
                  {Math.round(reindexProgress.progress || 0)}%
                </span>
              </div>
              <div className="w-full bg-void-900 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-signal-500 h-2 rounded-full transition-all duration-300 shadow-[0_0_10px_rgba(124,58,237,0.5)]"
                  style={{ width: `${reindexProgress.progress || 0}%` }}
                />
              </div>
              {reindexProgress.current_step && (
                <p className="mt-2 text-xs text-signal-400">
                  {reindexProgress.current_step}
                </p>
              )}
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-3 pt-2">
            <button
              onClick={handleReindex}
              disabled={isReindexing || !!isReindexInProgress}
              className="flex-1 px-4 py-2 text-sm font-bold text-signal-400 bg-transparent border border-signal-500/30 rounded-xl hover:bg-signal-500/10 hover:border-signal-500/50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all"
            >
              {isReindexing || isReindexInProgress ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Reindexing...
                </>
              ) : (
                <>
                  <RefreshCw className="h-4 w-4" />
                  Reindex
                </>
              )}
            </button>
            <button
              onClick={handleDisable}
              disabled={isDisabling}
              className="px-4 py-2 text-sm font-bold text-red-400 bg-transparent border border-red-500/30 rounded-xl hover:bg-red-500/10 hover:border-red-500/50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-all"
            >
              {isDisabling ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Trash2 className="h-4 w-4" />
              )}
              Disable
            </button>
          </div>
        </div>
      ) : (
        /* Disabled State */
        <div className="p-6 bg-void-900/30 border border-white/5 rounded-2xl group">
          <p className="text-sm text-gray-400 mb-5 leading-relaxed">
            Enable sparse indexing to use hybrid search (BM25 + semantic) for this collection.
            This creates keyword-based indexes alongside your existing vector embeddings.
          </p>
          <button
            onClick={() => setShowConfigModal(true)}
            disabled={isEnabling}
            className="w-full px-4 py-2 text-sm font-bold text-white bg-signal-600 rounded-xl hover:bg-signal-500 shadow-lg shadow-signal-600/20 disabled:opacity-50 flex items-center justify-center gap-2 transition-all transform active:scale-[0.98]"
          >
            {isEnabling ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Enabling...
              </>
            ) : (
              <>
                <Zap className="h-4 w-4" />
                Enable Sparse Indexing
              </>
            )}
          </button>
        </div>
      )}

      {/* Configuration Modal */}
      <SparseIndexConfigModal
        isOpen={showConfigModal}
        onClose={() => setShowConfigModal(false)}
        onSubmit={handleEnable}
        isSubmitting={isEnabling}
        collectionName={collection.name}
        documentCount={collection.document_count || 0}
      />
    </div>
  );
}

export default SparseIndexPanel;
