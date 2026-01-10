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
      <div className="p-4 bg-gray-50 rounded-lg">
        <div className="flex items-center justify-center py-6">
          <Loader2 className="h-6 w-6 animate-spin text-gray-400" />
          <span className="ml-2 text-sm text-gray-500">Loading sparse index status...</span>
        </div>
      </div>
    );
  }

  // Error state
  if (isError) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
        <div className="flex items-center gap-2 text-red-700">
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
          <Zap className="h-5 w-5 text-purple-600" />
          <h4 className="text-sm font-medium text-gray-900">Sparse Indexing</h4>
        </div>
        {status?.enabled ? (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
            <CheckCircle className="h-3 w-3 mr-1" />
            Enabled
          </span>
        ) : (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
            <XCircle className="h-3 w-3 mr-1" />
            Disabled
          </span>
        )}
      </div>

      {status?.enabled ? (
        /* Enabled State */
        <div className="p-4 bg-purple-50 border border-purple-100 rounded-lg space-y-4">
          {/* Status Info */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <dt className="text-xs text-purple-600 uppercase tracking-wide">Plugin</dt>
              <dd className="mt-1 text-sm font-medium text-purple-900">
                {pluginInfo?.name || status.plugin_id}
              </dd>
            </div>
            <div>
              <dt className="text-xs text-purple-600 uppercase tracking-wide">Documents</dt>
              <dd className="mt-1 text-sm font-medium text-purple-900">
                {status.document_count?.toLocaleString() || '0'}
              </dd>
            </div>
          </div>

          {/* BM25 Parameters (if applicable) */}
          {status.plugin_id === 'bm25-local' && status.model_config_data && (
            <div className="pt-3 border-t border-purple-200">
              <dt className="text-xs text-purple-600 uppercase tracking-wide mb-2">
                BM25 Parameters
              </dt>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-sm">
                  <span className="text-purple-600">k1:</span>{' '}
                  <span className="font-mono text-purple-900">
                    {(status.model_config_data as { k1?: number }).k1 ?? 1.5}
                  </span>
                </div>
                <div className="text-sm">
                  <span className="text-purple-600">b:</span>{' '}
                  <span className="font-mono text-purple-900">
                    {(status.model_config_data as { b?: number }).b ?? 0.75}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Reindex Progress */}
          {isReindexInProgress && reindexProgress && (
            <div className="pt-3 border-t border-purple-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-purple-900">
                  Reindexing in progress
                </span>
                <span className="text-sm text-purple-700">
                  {Math.round(reindexProgress.progress || 0)}%
                </span>
              </div>
              <div className="w-full bg-purple-200 rounded-full h-2">
                <div
                  className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${reindexProgress.progress || 0}%` }}
                />
              </div>
              {reindexProgress.current_step && (
                <p className="mt-2 text-xs text-purple-600">
                  {reindexProgress.current_step}
                </p>
              )}
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-2 pt-2">
            <button
              onClick={handleReindex}
              disabled={isReindexing || !!isReindexInProgress}
              className="flex-1 px-3 py-2 text-sm font-medium text-purple-700 bg-white border border-purple-200 rounded-lg hover:bg-purple-50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
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
              className="px-3 py-2 text-sm font-medium text-red-700 bg-white border border-red-200 rounded-lg hover:bg-red-50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
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
        <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
          <p className="text-sm text-gray-600 mb-4">
            Enable sparse indexing to use hybrid search (BM25 + semantic) for this collection.
            This creates keyword-based indexes alongside your existing vector embeddings.
          </p>
          <button
            onClick={() => setShowConfigModal(true)}
            disabled={isEnabling}
            className="w-full px-4 py-2 text-sm font-medium text-white bg-purple-600 rounded-lg hover:bg-purple-700 disabled:opacity-50 flex items-center justify-center gap-2"
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
