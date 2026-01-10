/**
 * SparseIndexConfigModal Component
 *
 * Modal for configuring and enabling sparse indexing on a collection.
 * Allows users to select the sparse indexer plugin (BM25 or SPLADE)
 * and configure plugin-specific parameters.
 */

import React, { useState } from 'react';
import { X, Zap, AlertCircle, HelpCircle, Loader2, Cpu, Database } from 'lucide-react';
import type {
  SparseIndexerPlugin,
  BM25Config,
  EnableSparseIndexRequest,
} from '../../types/sparse-index';
import {
  DEFAULT_BM25_CONFIG,
  BM25_PARAM_RANGES,
  SPARSE_PLUGIN_INFO,
} from '../../types/sparse-index';

interface SparseIndexConfigModalProps {
  /** Whether the modal is open */
  isOpen: boolean;
  /** Callback when modal is closed */
  onClose: () => void;
  /** Callback when configuration is submitted */
  onSubmit: (config: EnableSparseIndexRequest) => void;
  /** Whether the submit is in progress */
  isSubmitting?: boolean;
  /** Collection name for display */
  collectionName: string;
  /** Number of documents to be indexed */
  documentCount?: number;
}

export function SparseIndexConfigModal({
  isOpen,
  onClose,
  onSubmit,
  isSubmitting = false,
  collectionName,
  documentCount = 0,
}: SparseIndexConfigModalProps) {
  const [selectedPlugin, setSelectedPlugin] = useState<SparseIndexerPlugin>('bm25-local');
  const [bm25Config, setBm25Config] = useState<BM25Config>(DEFAULT_BM25_CONFIG);
  const [reindexExisting, setReindexExisting] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const config: EnableSparseIndexRequest = {
      plugin_id: selectedPlugin,
      reindex_existing: reindexExisting,
    };

    if (selectedPlugin === 'bm25-local') {
      config.model_config_data = bm25Config;
    }

    onSubmit(config);
  };

  const pluginInfo = SPARSE_PLUGIN_INFO[selectedPlugin];

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-full items-center justify-center p-4">
        {/* Backdrop */}
        <div
          className="fixed inset-0 bg-black/50 transition-opacity"
          onClick={onClose}
        />

        {/* Modal */}
        <div className="relative w-full max-w-lg bg-white rounded-xl shadow-xl">
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Zap className="h-5 w-5 text-purple-600" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-gray-900">
                  Enable Sparse Indexing
                </h2>
                <p className="text-sm text-gray-500">{collectionName}</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Body */}
          <form onSubmit={handleSubmit}>
            <div className="px-6 py-4 space-y-6">
              {/* Plugin Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-3">
                  Select Sparse Indexer
                </label>
                <div className="grid grid-cols-2 gap-3">
                  {(Object.keys(SPARSE_PLUGIN_INFO) as SparseIndexerPlugin[]).map(
                    (pluginId) => {
                      const info = SPARSE_PLUGIN_INFO[pluginId];
                      const isSelected = selectedPlugin === pluginId;

                      return (
                        <button
                          key={pluginId}
                          type="button"
                          onClick={() => setSelectedPlugin(pluginId)}
                          className={`
                            relative p-4 rounded-lg border-2 text-left transition-all
                            ${
                              isSelected
                                ? 'border-purple-500 bg-purple-50'
                                : 'border-gray-200 hover:border-gray-300'
                            }
                          `}
                        >
                          <div className="flex items-start gap-3">
                            <div
                              className={`p-2 rounded-lg ${
                                isSelected ? 'bg-purple-100' : 'bg-gray-100'
                              }`}
                            >
                              {info.requiresGPU ? (
                                <Cpu
                                  className={`h-4 w-4 ${
                                    isSelected ? 'text-purple-600' : 'text-gray-500'
                                  }`}
                                />
                              ) : (
                                <Database
                                  className={`h-4 w-4 ${
                                    isSelected ? 'text-purple-600' : 'text-gray-500'
                                  }`}
                                />
                              )}
                            </div>
                            <div>
                              <p
                                className={`font-medium ${
                                  isSelected ? 'text-purple-900' : 'text-gray-900'
                                }`}
                              >
                                {info.name}
                              </p>
                              <p className="text-xs text-gray-500 mt-0.5">
                                {info.requiresGPU ? 'GPU required' : 'CPU only'}
                              </p>
                            </div>
                          </div>
                          {isSelected && (
                            <div className="absolute top-2 right-2 w-2 h-2 bg-purple-500 rounded-full" />
                          )}
                        </button>
                      );
                    }
                  )}
                </div>
                <p className="mt-2 text-sm text-gray-500">{pluginInfo.description}</p>
              </div>

              {/* BM25 Configuration */}
              {selectedPlugin === 'bm25-local' && (
                <div>
                  <button
                    type="button"
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900"
                  >
                    <span
                      className={`transform transition-transform ${
                        showAdvanced ? 'rotate-90' : ''
                      }`}
                    >
                      ▶
                    </span>
                    Advanced BM25 Parameters
                    <span className="text-xs text-gray-400">(optional)</span>
                  </button>

                  {showAdvanced && (
                    <div className="mt-4 p-4 bg-gray-50 rounded-lg space-y-4">
                      {/* k1 Parameter */}
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-sm font-medium text-gray-700 flex items-center gap-1.5">
                            k1 (Term Frequency Saturation)
                            <div className="group relative">
                              <HelpCircle className="h-3.5 w-3.5 text-gray-400" />
                              <div className="absolute left-0 bottom-full mb-2 w-48 p-2 bg-gray-900 text-white text-xs rounded opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
                                Controls how quickly term frequency saturates.
                                Higher values give more weight to repeated terms.
                              </div>
                            </div>
                          </label>
                          <span className="text-sm font-mono text-gray-600">
                            {bm25Config.k1}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={BM25_PARAM_RANGES.k1.min}
                          max={BM25_PARAM_RANGES.k1.max}
                          step={BM25_PARAM_RANGES.k1.step}
                          value={bm25Config.k1}
                          onChange={(e) =>
                            setBm25Config({
                              ...bm25Config,
                              k1: parseFloat(e.target.value),
                            })
                          }
                          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
                        />
                        <div className="flex justify-between text-xs text-gray-500 mt-1">
                          <span>Low ({BM25_PARAM_RANGES.k1.min})</span>
                          <span>Default (1.5)</span>
                          <span>High ({BM25_PARAM_RANGES.k1.max})</span>
                        </div>
                      </div>

                      {/* b Parameter */}
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-sm font-medium text-gray-700 flex items-center gap-1.5">
                            b (Length Normalization)
                            <div className="group relative">
                              <HelpCircle className="h-3.5 w-3.5 text-gray-400" />
                              <div className="absolute left-0 bottom-full mb-2 w-48 p-2 bg-gray-900 text-white text-xs rounded opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
                                Controls document length normalization.
                                0 = no normalization, 1 = full normalization.
                              </div>
                            </div>
                          </label>
                          <span className="text-sm font-mono text-gray-600">
                            {bm25Config.b}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={BM25_PARAM_RANGES.b.min}
                          max={BM25_PARAM_RANGES.b.max}
                          step={BM25_PARAM_RANGES.b.step}
                          value={bm25Config.b}
                          onChange={(e) =>
                            setBm25Config({
                              ...bm25Config,
                              b: parseFloat(e.target.value),
                            })
                          }
                          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
                        />
                        <div className="flex justify-between text-xs text-gray-500 mt-1">
                          <span>None ({BM25_PARAM_RANGES.b.min})</span>
                          <span>Default (0.75)</span>
                          <span>Full ({BM25_PARAM_RANGES.b.max})</span>
                        </div>
                      </div>

                      {/* Reset to defaults */}
                      <button
                        type="button"
                        onClick={() => setBm25Config(DEFAULT_BM25_CONFIG)}
                        className="text-xs text-purple-600 hover:text-purple-700"
                      >
                        Reset to defaults
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* Reindex Option */}
              <div className="flex items-start gap-3">
                <input
                  type="checkbox"
                  id="reindex-existing"
                  checked={reindexExisting}
                  onChange={(e) => setReindexExisting(e.target.checked)}
                  className="mt-1 h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300 rounded"
                />
                <label htmlFor="reindex-existing" className="text-sm text-gray-700">
                  <span className="font-medium">Index existing documents</span>
                  <p className="text-gray-500 mt-0.5">
                    This will create sparse vectors for all {documentCount.toLocaleString()} documents
                    in the background.
                  </p>
                </label>
              </div>

              {/* Warning */}
              <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                <div className="flex gap-2">
                  <AlertCircle className="h-5 w-5 text-yellow-600 flex-shrink-0" />
                  <div className="text-sm text-yellow-800">
                    <p className="font-medium">Note</p>
                    <p className="mt-1">
                      {selectedPlugin === 'splade-local'
                        ? 'SPLADE indexing requires GPU and may take several minutes for large collections.'
                        : 'BM25 indexing is CPU-based and typically completes quickly.'}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="px-6 py-4 border-t border-gray-200 flex justify-end gap-3">
              <button
                type="button"
                onClick={onClose}
                disabled={isSubmitting}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isSubmitting}
                className="px-4 py-2 text-sm font-medium text-white bg-purple-600 rounded-lg hover:bg-purple-700 disabled:opacity-50 flex items-center gap-2"
              >
                {isSubmitting ? (
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
          </form>
        </div>
      </div>
    </div>
  );
}

export default SparseIndexConfigModal;
