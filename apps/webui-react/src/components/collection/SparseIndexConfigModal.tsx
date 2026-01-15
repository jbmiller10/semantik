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
          className="fixed inset-0 bg-void-950/80 backdrop-blur-sm transition-opacity"
          onClick={onClose}
        />

        {/* Modal */}
        <div className="relative w-full max-w-lg glass-panel rounded-2xl shadow-2xl border border-white/10">
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 bg-void-900/50 rounded-t-2xl">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-signal-500/10 rounded-xl border border-signal-500/20">
                <Zap className="h-5 w-5 text-signal-400" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-white tracking-tight">
                  Enable Sparse Indexing
                </h2>
                <p className="text-sm text-gray-400">{collectionName}</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 text-gray-400 hover:text-white rounded-xl hover:bg-white/10 transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Body */}
          <form onSubmit={handleSubmit}>
            <div className="px-6 py-6 space-y-6">
              {/* Plugin Selection */}
              <div>
                <label className="block text-xs font-bold text-gray-400 uppercase tracking-wider mb-3">
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
                            relative p-4 rounded-xl border transition-all text-left
                            ${isSelected
                              ? 'border-signal-500 bg-signal-500/10 shadow-lg shadow-signal-500/10'
                              : 'border-white/10 bg-void-900/50 hover:bg-white/5 hover:border-white/20'
                            }
                          `}
                        >
                          <div className="flex items-start gap-3">
                            <div
                              className={`p-2 rounded-lg ${isSelected ? 'bg-signal-500/20' : 'bg-void-800'
                                }`}
                            >
                              {info.requiresGPU ? (
                                <Cpu
                                  className={`h-4 w-4 ${isSelected ? 'text-signal-400' : 'text-gray-500'
                                    }`}
                                />
                              ) : (
                                <Database
                                  className={`h-4 w-4 ${isSelected ? 'text-signal-400' : 'text-gray-500'
                                    }`}
                                />
                              )}
                            </div>
                            <div>
                              <p
                                className={`font-bold text-sm ${isSelected ? 'text-signal-300' : 'text-gray-200'
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
                            <div className="absolute top-2 right-2 w-2 h-2 bg-signal-400 rounded-full shadow-[0_0_8px_rgba(167,139,250,0.6)]" />
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
                    className="flex items-center gap-2 text-xs font-bold text-gray-400 hover:text-white uppercase tracking-wide transition-colors"
                  >
                    <span
                      className={`transform transition-transform text-signal-400 ${showAdvanced ? 'rotate-90' : ''
                        }`}
                    >
                      ▶
                    </span>
                    Advanced BM25 Parameters
                    <span className="text-xs text-gray-400">(optional)</span>
                  </button>

                  {showAdvanced && (
                    <div className="mt-4 p-4 bg-void-900/50 border border-white/5 rounded-xl space-y-4">
                      {/* k1 Parameter */}
                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-sm font-medium text-gray-300 flex items-center gap-1.5">
                            k1 (Term Frequency Saturation)
                            <div className="group relative">
                              <HelpCircle className="h-3.5 w-3.5 text-gray-500" />
                              <div className="absolute left-0 bottom-full mb-2 w-48 p-2 bg-void-950 border border-white/10 text-gray-300 text-xs rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
                                Controls how quickly term frequency saturates.
                                Higher values give more weight to repeated terms.
                              </div>
                            </div>
                          </label>
                          <span className="text-sm font-mono text-signal-400 font-bold">
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
                          className="w-full h-2 bg-void-800 rounded-lg appearance-none cursor-pointer accent-signal-500"
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
                          <label className="text-sm font-medium text-gray-300 flex items-center gap-1.5">
                            b (Length Normalization)
                            <div className="group relative">
                              <HelpCircle className="h-3.5 w-3.5 text-gray-500" />
                              <div className="absolute left-0 bottom-full mb-2 w-48 p-2 bg-void-950 border border-white/10 text-gray-300 text-xs rounded-lg shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-10">
                                Controls document length normalization.
                                0 = no normalization, 1 = full normalization.
                              </div>
                            </div>
                          </label>
                          <span className="text-sm font-mono text-signal-400 font-bold">
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
                          className="w-full h-2 bg-void-800 rounded-lg appearance-none cursor-pointer accent-signal-500"
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
                        className="text-xs font-bold text-signal-400 hover:text-signal-300 uppercase tracking-wide"
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
                  className="mt-1 h-4 w-4 bg-void-800 border-white/20 text-signal-600 focus:ring-signal-500 rounded"
                />
                <label htmlFor="reindex-existing" className="text-sm text-gray-300">
                  <span className="font-bold text-white">Index existing documents</span>
                  <p className="text-gray-400 mt-0.5">
                    This will create sparse vectors for all {documentCount.toLocaleString()} documents
                    in the background.
                  </p>
                </label>
              </div>

              {/* Warning */}
              <div className="p-4 bg-amber-500/10 border border-amber-500/20 rounded-xl">
                <div className="flex gap-3">
                  <AlertCircle className="h-5 w-5 text-amber-500 flex-shrink-0" />
                  <div className="text-sm text-amber-200">
                    <p className="font-bold">Note</p>
                    <p className="mt-1 text-amber-300/80">
                      {selectedPlugin === 'splade-local'
                        ? 'SPLADE indexing requires GPU and may take several minutes for large collections.'
                        : 'BM25 indexing is CPU-based and typically completes quickly.'}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Footer */}
            <div className="px-6 py-4 border-t border-white/10 bg-void-900/30 rounded-b-2xl flex justify-end gap-3 backdrop-blur-md">
              <button
                type="button"
                onClick={onClose}
                disabled={isSubmitting}
                className="px-4 py-2 text-sm font-medium text-gray-400 border border-white/10 rounded-xl hover:bg-white/5 hover:text-white transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={isSubmitting}
                className="px-6 py-2 text-sm font-bold text-white bg-signal-600 rounded-xl hover:bg-signal-500 shadow-lg shadow-signal-600/20 disabled:opacity-50 flex items-center gap-2 transition-all transform active:scale-95"
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
