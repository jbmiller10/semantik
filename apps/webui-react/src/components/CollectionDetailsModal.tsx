import { useState } from 'react';
import { useQuery, useQueryClient, useMutation } from '@tanstack/react-query';
import { useUIStore } from '../stores/uiStore';
import { collectionsV2Api } from '../services/api/v2/collections';
import { documentsV2Api } from '../services/api/v2/documents';
import { collectionKeys } from '../hooks/useCollections';
import { operationKeys } from '../hooks/useCollectionOperations';
import AddDataToCollectionModal from './AddDataToCollectionModal';
import RenameCollectionModal from './RenameCollectionModal';
import DeleteCollectionModal from './DeleteCollectionModal';
import ReindexCollectionModal from './ReindexCollectionModal';
import EmbeddingVisualizationTab from './EmbeddingVisualizationTab';
import { SparseIndexPanel } from './collection/SparseIndexPanel';
import type { FailedDocumentCountResponse, RetryDocumentsResponse, DocumentResponse } from '../services/api/v2/types';
import { CHUNKING_STRATEGIES } from '../types/chunking';
import type { ChunkingStrategyType } from '../types/chunking';
import {
  Type,
  GitBranch,
  FileText,
  Brain,
  Network,
  Sparkles,
  RefreshCw,
  XCircle,
  Trash2,
  Edit2,
  Plus
} from 'lucide-react';

// --- Helper Functions ---

const formatNumber = (num: number | undefined | null): string => {
  if (num === undefined || num === null) return '0';
  return num.toLocaleString();
};

const formatBytes = (bytes: number | undefined | null, decimals = 2) => {
  if (bytes === undefined || bytes === null || bytes === 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

const formatDate = (dateString: string) => {
  return new Date(dateString).toLocaleString();
};

const formatDuration = (start: string, end?: string) => {
  if (!end) return '-';
  const diff = new Date(end).getTime() - new Date(start).getTime();
  const seconds = Math.floor(diff / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
};

const formatChunkingConfig = (config: Record<string, string | number | boolean>) => {
  if (!config) return [];
  return Object.entries(config).map(([key, value]) => ({
    label: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    value: String(value)
  }));
};

// --- Component ---

function CollectionDetailsModal() {
  const queryClient = useQueryClient();
  const { showCollectionDetailsModal, setShowCollectionDetailsModal, addToast } = useUIStore();
  const [showAddDataModal, setShowAddDataModal] = useState(false);
  const [showRenameModal, setShowRenameModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'jobs' | 'files' | 'visualize' | 'settings'>('overview');
  const [filesPage, setFilesPage] = useState(1);
  const filesLimit = 50;

  // State for config changes (only instruction for now)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [configChanges, _setConfigChanges] = useState<{
    instruction?: string;
  }>({});
  const [showReindexModal, setShowReindexModal] = useState(false);

  // Close handler
  const handleClose = () => {
    setShowCollectionDetailsModal(null);
  };

  // --- Queries ---

  // Collection Details
  const { data: collection, isLoading } = useQuery({
    queryKey: collectionKeys.detail(showCollectionDetailsModal ?? ''),
    queryFn: async () => {
      if (!showCollectionDetailsModal) return null;
      const response = await collectionsV2Api.get(showCollectionDetailsModal);
      return response.data;
    },
    enabled: !!showCollectionDetailsModal,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data?.status === 'processing' || data?.activeOperation) {
        return 5000;
      }
      return 30000;
    },
  });

  // Operations (Jobs)
  const { data: operationsData } = useQuery({
    queryKey: operationKeys.list(showCollectionDetailsModal ?? ''),
    queryFn: async () => {
      if (!showCollectionDetailsModal) return null;
      const response = await collectionsV2Api.listOperations(showCollectionDetailsModal, { limit: 50 });
      return response.data;
    },
    enabled: !!showCollectionDetailsModal && activeTab === 'jobs',
  });

  // Documents (Files)
  const { data: documentsData, refetch: refetchDocuments } = useQuery({
    queryKey: [...collectionKeys.detail(showCollectionDetailsModal ?? ''), 'documents', filesPage],
    queryFn: async () => {
      if (!showCollectionDetailsModal || activeTab !== 'files') return null;
      const response = await collectionsV2Api.listDocuments(showCollectionDetailsModal, {
        page: filesPage,
        limit: filesLimit
      });
      return response.data; // Expected: { documents: [], total: number, page: number, per_page: number }
    },
    enabled: !!showCollectionDetailsModal && activeTab === 'files',
  });

  // Failed document counts
  const { data: failedCounts } = useQuery<FailedDocumentCountResponse | null>({
    queryKey: [...collectionKeys.detail(showCollectionDetailsModal ?? ''), 'failed-counts'],
    queryFn: async () => {
      if (!showCollectionDetailsModal || activeTab !== 'files') return null;
      const response = await documentsV2Api.getFailedCount(showCollectionDetailsModal);
      return response.data;
    },
    enabled: !!showCollectionDetailsModal && activeTab === 'files',
  });

  // --- Mutations ---

  const retryDocumentMutation = useMutation({
    mutationFn: (documentUuid: string) =>
      documentsV2Api.retry(showCollectionDetailsModal!, documentUuid),
    onSuccess: () => {
      addToast({ message: 'Document queued for retry', type: 'success' });
      refetchDocuments();
      queryClient.invalidateQueries({ queryKey: [...collectionKeys.detail(showCollectionDetailsModal!), 'failed-counts'] });
    },
    onError: (error: Error) => {
      addToast({ message: `Failed to retry document: ${error.message}`, type: 'error' });
    }
  });

  const retryAllFailedMutation = useMutation({
    mutationFn: () => documentsV2Api.retryFailed(showCollectionDetailsModal!),
    onSuccess: (response: { data: RetryDocumentsResponse }) => {
      addToast({ message: `Queued retry for ${response.data.reset_count} documents`, type: 'success' });
      refetchDocuments();
      queryClient.invalidateQueries({ queryKey: [...collectionKeys.detail(showCollectionDetailsModal!), 'failed-counts'] });
    },
    onError: (error: Error) => {
      addToast({ message: `Failed to retry documents: ${error.message}`, type: 'error' });
    }
  });

  // --- Handlers ---

  const handleAddDataSuccess = () => {
    setShowAddDataModal(false);
    queryClient.invalidateQueries({ queryKey: collectionKeys.detail(showCollectionDetailsModal!) });
    addToast({ message: 'Files uploaded successfully', type: 'success' });
  };

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const handleRenameSuccess = (_newName: string) => {
    setShowRenameModal(false);
    queryClient.invalidateQueries({ queryKey: collectionKeys.all });
    queryClient.invalidateQueries({ queryKey: collectionKeys.detail(showCollectionDetailsModal!) });
    addToast({ message: 'Collection renamed successfully', type: 'success' });
  };

  const handleDeleteSuccess = () => {
    setShowDeleteModal(false);
    setShowCollectionDetailsModal(null);
    queryClient.invalidateQueries({ queryKey: collectionKeys.all });
    addToast({ message: 'Collection deleted successfully', type: 'success' });
  };

  const handleReindexSuccess = () => {
    setShowReindexModal(false);
    queryClient.invalidateQueries({ queryKey: collectionKeys.detail(showCollectionDetailsModal!) });
    queryClient.invalidateQueries({ queryKey: operationKeys.list(showCollectionDetailsModal!) });
    addToast({ message: 'Re-indexing started successfully', type: 'success' });
  };

  if (!showCollectionDetailsModal) return null;

  const retryableCount = (failedCounts?.transient ?? 0) + (failedCounts?.unknown ?? 0);

  return (
    <>
      <div className="fixed inset-0 bg-[var(--bg-primary)]/80 backdrop-blur-sm z-50 transition-opacity" onClick={handleClose} />
      <div className="fixed inset-4 md:inset-[5%] panel rounded-2xl shadow-2xl border border-[var(--border)] z-50 flex flex-col max-w-6xl mx-auto overflow-hidden">

        {/* Header */}
        <div className="px-6 py-5 border-b border-[var(--border)] bg-[var(--bg-secondary)] flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-[var(--text-primary)] tracking-tight">
              {collection?.name || 'Loading...'}
            </h2>
            {collection && (
              <p className="text-sm text-[var(--text-secondary)] mt-1 font-mono">
                {operationsData?.length || 0} operations • {formatNumber(collection.document_count)} docs • {formatNumber(collection.vector_count)} vectors
              </p>
            )}
          </div>
          <button
            onClick={handleClose}
            className="text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors p-2 hover:bg-[var(--bg-tertiary)] rounded-lg"
          >
            <XCircle className="h-6 w-6" />
          </button>
        </div>

        {/* Action Buttons */}
        <div className="px-6 py-3 border-b border-[var(--border)] bg-[var(--bg-secondary)]/50 flex gap-2">
          <button
            onClick={() => setShowAddDataModal(true)}
            className="btn-primary flex items-center gap-2 px-4 py-2 text-sm font-bold rounded-xl shadow-lg transition-all transform active:scale-95"
            disabled={!collection}
          >
            <Plus className="h-4 w-4" />
            Add Data
          </button>
          <button
            onClick={() => setShowRenameModal(true)}
            className="btn-secondary flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-xl transition-colors"
            disabled={!collection}
          >
            <Edit2 className="h-4 w-4" />
            Rename
          </button>
          <button
            onClick={() => setShowDeleteModal(true)}
            className="flex items-center gap-2 px-4 py-2 border border-red-500/30 text-red-500 dark:text-red-400 text-sm font-medium rounded-xl hover:bg-red-500/10 hover:border-red-500/50 transition-colors"
            disabled={!collection}
          >
            <Trash2 className="h-4 w-4" />
            Delete
          </button>
        </div>

        {/* Tabs */}
        <div className="border-b border-[var(--border)] bg-[var(--bg-secondary)]/30">
          <nav className="flex space-x-8 px-6 overflow-x-auto" aria-label="Tabs">
            {[
              { id: 'overview', label: 'Overview', icon: Brain },
              { id: 'jobs', label: 'Jobs', icon: ActivityIcon },
              { id: 'files', label: 'Files', icon: FileText },
              { id: 'visualize', label: 'Visualize', icon: Sparkles },
              { id: 'settings', label: 'Settings', icon: Type }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as 'overview' | 'jobs' | 'files' | 'visualize' | 'settings')}
                className={`py-3 px-1 border-b-2 font-bold text-sm uppercase tracking-wide transition-colors flex items-center gap-2 ${activeTab === tab.id
                    ? 'border-[var(--accent-primary)] text-[var(--text-primary)]'
                    : 'border-transparent text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:border-[var(--border)]'
                  }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 bg-[var(--bg-primary)]">
          {isLoading && (
            <div className="flex justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[var(--accent-primary)]"></div>
            </div>
          )}

          {collection && !isLoading && (
            <div className="space-y-6">

              {/* --- OVERVIEW TAB --- */}
              {activeTab === 'overview' && (
                <div className="space-y-6">
                  {/* Stats Grid */}
                  <div>
                    <h3 className="text-lg font-bold text-[var(--text-primary)] mb-4 tracking-tight">Statistics</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-[var(--bg-secondary)] p-5 rounded-xl border border-[var(--border)]">
                        <dt className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">Documents</dt>
                        <dd className="mt-1 text-3xl font-bold text-[var(--text-primary)] tracking-tight">
                          {formatNumber(collection.document_count)}
                        </dd>
                      </div>
                      <div className="bg-[var(--bg-secondary)] p-5 rounded-xl border border-[var(--border)]">
                        <dt className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">Vectors</dt>
                        <dd className="mt-1 text-3xl font-bold text-[var(--text-primary)] tracking-tight">
                          {formatNumber(collection.vector_count)}
                        </dd>
                      </div>
                      <div className="bg-[var(--bg-secondary)] p-5 rounded-xl border border-[var(--border)]">
                        <dt className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">Total Size</dt>
                        <dd className="mt-1 text-3xl font-bold text-[var(--text-primary)] tracking-tight">
                          {formatBytes(collection.total_size_bytes || 0)}
                        </dd>
                      </div>
                      <div className="bg-[var(--bg-secondary)] p-5 rounded-xl border border-[var(--border)]">
                        <dt className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">Operations</dt>
                        <dd className="mt-1 text-3xl font-bold text-[var(--text-primary)] tracking-tight">
                          {operationsData?.length || 0}
                        </dd>
                      </div>
                    </div>
                  </div>

                  {/* Sparse Index Overview */}
                  <div>
                    <SparseIndexPanel collection={collection} />
                  </div>
                </div>
              )}

              {/* --- JOBS TAB --- */}
              {activeTab === 'jobs' && (
                <div className="space-y-8">
                  <div className="panel border border-[var(--border)] rounded-xl overflow-hidden">
                    <table className="min-w-full divide-y divide-[var(--border)]">
                      <thead className="bg-[var(--bg-secondary)]">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">Type</th>
                          <th className="px-6 py-3 text-left text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">Status</th>
                          <th className="px-6 py-3 text-left text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">Started</th>
                          <th className="px-6 py-3 text-left text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">Duration</th>
                        </tr>
                      </thead>
                      <tbody className="bg-transparent divide-y divide-[var(--border)]">
                        {!operationsData || operationsData.length === 0 ? (
                          <tr><td colSpan={4} className="px-6 py-8 text-center text-[var(--text-muted)]">No jobs found</td></tr>
                        ) : (
                          operationsData.map((job) => (
                            <tr key={job.id} className="hover:bg-[var(--bg-tertiary)] transition-colors">
                              <td className="px-6 py-4 text-sm text-[var(--text-primary)] capitalize">{job.type.replace(/_/g, ' ')}</td>
                              <td className="px-6 py-4">
                                <span className={`px-2 py-0.5 rounded-full text-xs font-bold border ${job.status === 'completed' ? 'bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/20' :
                                    job.status === 'failed' ? 'bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20' :
                                      job.status === 'processing' ? 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20' :
                                        'bg-[var(--bg-tertiary)] text-[var(--text-muted)] border-[var(--border)]'
                                  }`}>
                                  {job.status}
                                </span>
                              </td>
                              <td className="px-6 py-4 text-sm text-[var(--text-secondary)]">{formatDate(job.created_at)}</td>
                              <td className="px-6 py-4 text-sm text-[var(--text-secondary)]">
                                {job.started_at && job.completed_at ? formatDuration(job.started_at, job.completed_at) : '-'}
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* --- FILES TAB --- */}
              {activeTab === 'files' && (
                <div className="space-y-4">
                  <div className="flex justify-between items-center mb-2">
                    <h3 className="text-lg font-bold text-[var(--text-primary)] tracking-tight">Files ({documentsData?.total || 0})</h3>
                    {retryableCount > 0 && (
                      <button
                        onClick={() => retryAllFailedMutation.mutate()}
                        disabled={retryAllFailedMutation.isPending}
                        className="flex items-center gap-2 px-3 py-1.5 bg-blue-600/20 text-blue-600 dark:text-blue-400 text-xs font-bold uppercase tracking-wider border border-blue-500/30 rounded-lg hover:bg-blue-600/30 transition-colors"
                      >
                        <RefreshCw className={`h-3 w-3 ${retryAllFailedMutation.isPending ? 'animate-spin' : ''}`} />
                        Retry All Failed
                      </button>
                    )}
                  </div>

                  <div className="panel border border-[var(--border)] rounded-xl overflow-hidden">
                    <table className="min-w-full divide-y divide-[var(--border)]">
                      <thead className="bg-[var(--bg-secondary)]">
                        <tr>
                          <th className="px-6 py-3 text-left text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">Name</th>
                          <th className="px-6 py-3 text-left text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">Size</th>
                          <th className="px-6 py-3 text-left text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">Status</th>
                          <th className="px-6 py-3 text-left text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">Created</th>
                          <th className="relative px-6 py-3"><span className="sr-only">Actions</span></th>
                        </tr>
                      </thead>
                      <tbody className="bg-transparent divide-y divide-[var(--border)]">
                        {!documentsData?.documents || documentsData.documents.length === 0 ? (
                          <tr><td colSpan={5} className="px-6 py-8 text-center text-[var(--text-muted)]">No files found</td></tr>
                        ) : (
                          documentsData.documents.map((doc: DocumentResponse) => (
                            <tr key={doc.id} className="hover:bg-[var(--bg-tertiary)] transition-colors">
                              <td className="px-6 py-4 text-sm font-medium text-[var(--text-primary)] flex items-center gap-2">
                                <FileText className="h-4 w-4 text-[var(--text-muted)]" />
                                <span className="truncate max-w-xs" title={doc.file_name}>{doc.file_name}</span>
                              </td>
                              <td className="px-6 py-4 text-sm text-[var(--text-secondary)]">{formatBytes(doc.file_size)}</td>
                              <td className="px-6 py-4">
                                <span className={`px-2 py-0.5 rounded-full text-xs font-bold border ${doc.status === 'completed' ? 'bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/20' :
                                    doc.status === 'failed' ? 'bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20' :
                                      'bg-[var(--bg-tertiary)] text-[var(--text-muted)] border-[var(--border)]'
                                  }`}>
                                  {doc.status}
                                </span>
                              </td>
                              <td className="px-6 py-4 text-sm text-[var(--text-secondary)]">{formatDate(doc.created_at)}</td>
                              <td className="px-6 py-4 text-right">
                                {doc.status === 'failed' && (
                                  <button
                                    onClick={() => retryDocumentMutation.mutate(doc.id)}
                                    className="text-blue-600 dark:text-blue-400 hover:text-blue-500 dark:hover:text-blue-300"
                                    title="Retry"
                                  >
                                    <RefreshCw className="h-4 w-4" />
                                  </button>
                                )}
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>

                    {/* Pagination */}
                    {(documentsData?.total ?? 0) > filesLimit && (
                      <div className="px-6 py-3 border-t border-[var(--border)] flex items-center justify-between bg-[var(--bg-secondary)]">
                        <button
                          onClick={() => setFilesPage(p => Math.max(1, p - 1))}
                          disabled={filesPage === 1}
                          className="text-sm font-bold text-[var(--text-secondary)] hover:text-[var(--text-primary)] disabled:opacity-50"
                        >
                          Previous
                        </button>
                        <span className="text-sm text-[var(--text-muted)]">Page {filesPage}</span>
                        <button
                          onClick={() => setFilesPage(p => p + 1)}
                          disabled={filesPage * filesLimit >= (documentsData?.total ?? 0)}
                          className="text-sm font-bold text-[var(--text-secondary)] hover:text-[var(--text-primary)] disabled:opacity-50"
                        >
                          Next
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* --- VISUALIZE TAB --- */}
              {activeTab === 'visualize' && (
                <EmbeddingVisualizationTab
                  collectionId={collection.id}
                  collectionEmbeddingModel={collection.embedding_model}
                  collectionVectorCount={collection.vector_count}
                  collectionUpdatedAt={collection.updated_at}
                />
              )}

              {/* --- SETTINGS TAB --- */}
              {activeTab === 'settings' && (
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-bold text-[var(--text-primary)] mb-4 tracking-tight">Configuration</h3>

                    <div className="space-y-4">
                      {/* Read Only Model */}
                      <div>
                        <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">Embedding Model</label>
                        <div className="input-field px-4 py-2 opacity-70">
                          {collection.embedding_model}
                        </div>
                      </div>

                      {/* Chunking Config Display */}
                      {collection.chunking_strategy ? (
                        <div className="panel p-4 border border-[var(--border)] rounded-xl">
                          <div className="flex items-center gap-3 mb-4">
                            <div className="text-[var(--accent-primary)]"><Brain className="h-5 w-5" /></div>
                            <div>
                              <div className="font-bold text-[var(--text-primary)]">
                                {CHUNKING_STRATEGIES[collection.chunking_strategy as ChunkingStrategyType]?.name || collection.chunking_strategy}
                              </div>
                              <div className="text-xs text-[var(--text-secondary)]">
                                {CHUNKING_STRATEGIES[collection.chunking_strategy as ChunkingStrategyType]?.description}
                              </div>
                            </div>
                          </div>
                          <dl className="grid grid-cols-2 gap-4">
                            {formatChunkingConfig(collection.chunking_config ?? {}).map(item => (
                              <div key={item.label}>
                                <dt className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">{item.label}</dt>
                                <dd className="text-sm text-[var(--text-primary)] font-mono">{item.value}</dd>
                              </div>
                            ))}
                          </dl>
                        </div>
                      ) : (
                        <div className="bg-amber-500/10 rounded-xl p-4 border border-amber-500/20">
                          <div className="flex items-center gap-3">
                            <Type className="h-5 w-5 text-amber-600 dark:text-amber-500" />
                            <div className="flex-1">
                              <div className="font-bold text-amber-700 dark:text-amber-400">Legacy Chunking</div>
                              <div className="text-xs text-amber-600/70 dark:text-amber-300/70">Deprecated character-based chunking</div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Re-index Section */}
                  <div className="border-t border-[var(--border)] pt-6">
                    <h3 className="text-lg font-bold text-[var(--text-primary)] mb-4 tracking-tight">Re-index Collection</h3>
                    <div className="bg-amber-500/10 border border-amber-500/20 rounded-xl p-4 mb-4">
                      <div className="flex gap-3">
                        <div className="text-amber-600 dark:text-amber-500"><GitBranch className="h-5 w-5" /></div>
                        <div>
                          <h4 className="font-bold text-amber-700 dark:text-amber-400 text-sm">Action Required</h4>
                          <p className="text-xs text-amber-600/80 dark:text-amber-300/80 mt-1">Re-indexing will delete all vectors and re-process documents.</p>
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={() => setShowReindexModal(true)}
                      className="px-4 py-2 bg-amber-600/20 text-amber-700 dark:text-amber-400 border border-amber-500/30 rounded-xl text-sm font-bold hover:bg-amber-600/30 transition-colors"
                    >
                      Re-index Collection
                    </button>
                  </div>

                  {/* Sparse Index Section */}
                  <div className="border-t border-[var(--border)] pt-6">
                    <h3 className="text-lg font-bold text-[var(--text-primary)] mb-4 tracking-tight">Sparse Indexing</h3>
                    <SparseIndexPanel collection={collection} />
                  </div>
                </div>
              )}

            </div>
          )}
        </div>

      </div>

      {/* Sub-modals */}
      {showAddDataModal && collection && (
        <AddDataToCollectionModal
          collection={collection}
          onClose={() => setShowAddDataModal(false)}
          onSuccess={handleAddDataSuccess}
        />
      )}

      {showRenameModal && collection && (
        <RenameCollectionModal
          collectionId={showCollectionDetailsModal!}
          currentName={collection?.name ?? ''}
          onClose={() => setShowRenameModal(false)}
          onSuccess={handleRenameSuccess}
        />
      )}

      {showDeleteModal && collection && (
        <DeleteCollectionModal
          collectionId={showCollectionDetailsModal!}
          collectionName={collection.name}
          stats={{
            total_files: collection.document_count || 0,
            total_vectors: collection.vector_count || 0,
            total_size: collection.total_size_bytes || 0,
            job_count: operationsData?.length || 0,
          }}
          onClose={() => setShowDeleteModal(false)}
          onSuccess={handleDeleteSuccess}
        />
      )}

      {showReindexModal && collection && (
        <ReindexCollectionModal
          collection={collection}
          configChanges={{
            embedding_model: collection.embedding_model,
            instruction: configChanges.instruction
          }}
          onClose={() => setShowReindexModal(false)}
          onSuccess={handleReindexSuccess}
        />
      )}
    </>
  );
}

// Temporary icon placeholder if needed
const ActivityIcon = Network;

export default CollectionDetailsModal;
