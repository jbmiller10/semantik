import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useUIStore } from '../stores/uiStore';
import { collectionsV2Api } from '../services/api/v2/collections';
import { useCollectionStore } from '../stores/collectionStore';
import AddDataToCollectionModal from './AddDataToCollectionModal';
import RenameCollectionModal from './RenameCollectionModal';
import DeleteCollectionModal from './DeleteCollectionModal';
import ReindexCollectionModal from './ReindexCollectionModal';
import type { DocumentResponse } from '../services/api/v2/types';

// Type for aggregated source directories from documents
interface SourceInfo {
  path: string;
  document_count: number;
}

function CollectionDetailsModal() {
  const queryClient = useQueryClient();
  const { showCollectionDetailsModal, setShowCollectionDetailsModal, addToast } = useUIStore();
  const [showAddDataModal, setShowAddDataModal] = useState(false);
  const [showRenameModal, setShowRenameModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'jobs' | 'files' | 'settings'>('overview');
  const [filesPage, setFilesPage] = useState(1);
  const [configChanges, setConfigChanges] = useState<{
    chunk_size?: number;
    chunk_overlap?: number;
    instruction?: string;
  }>({});
  const [showReindexModal, setShowReindexModal] = useState(false);
  const [validationErrors, setValidationErrors] = useState<{
    chunk_size?: string;
    chunk_overlap?: string;
  }>({});

  // Fetch collection details using v2 API
  const { data: collection, isLoading, error } = useQuery({
    queryKey: ['collection-v2', showCollectionDetailsModal],
    queryFn: async () => {
      if (!showCollectionDetailsModal) return null;
      const response = await collectionsV2Api.get(showCollectionDetailsModal);
      return response.data;
    },
    enabled: !!showCollectionDetailsModal,
  });

  // Fetch operations (jobs) for the collection
  const { data: operationsData } = useQuery({
    queryKey: ['collection-operations', showCollectionDetailsModal],
    queryFn: async () => {
      if (!showCollectionDetailsModal) return null;
      const response = await collectionsV2Api.listOperations(showCollectionDetailsModal, { limit: 50 });
      return response.data;
    },
    enabled: !!showCollectionDetailsModal && activeTab === 'jobs',
  });

  // Fetch documents (files) for the collection
  const { data: documentsData } = useQuery({
    queryKey: ['collection-documents', showCollectionDetailsModal, filesPage],
    queryFn: async () => {
      if (!showCollectionDetailsModal || activeTab !== 'files') return null;
      const response = await collectionsV2Api.listDocuments(showCollectionDetailsModal, { 
        page: filesPage,
        limit: 50 
      });
      return response.data;
    },
    enabled: !!showCollectionDetailsModal && activeTab === 'files',
  });

  // Aggregate source directories from documents
  const sourceDirs = documentsData ? Array.from(
    documentsData.documents.reduce((acc, doc) => {
      if (!acc.has(doc.source_path)) {
        acc.set(doc.source_path, { path: doc.source_path, document_count: 0 });
      }
      acc.get(doc.source_path)!.document_count++;
      return acc;
    }, new Map<string, SourceInfo>())
  ).map(([, info]) => info) : [];

  const handleClose = () => {
    setShowCollectionDetailsModal(null);
    setActiveTab('overview');
    setFilesPage(1);
  };

  const formatNumber = (num: number | null | undefined) => {
    if (num === null || num === undefined) {
      return '0';
    }
    return num.toLocaleString();
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const handleAddDataSuccess = () => {
    setShowAddDataModal(false);
    queryClient.invalidateQueries({ queryKey: ['collection-v2', showCollectionDetailsModal] });
    queryClient.invalidateQueries({ queryKey: ['collection-operations', showCollectionDetailsModal] });
    queryClient.invalidateQueries({ queryKey: ['collection-documents', showCollectionDetailsModal] });
    addToast({
      type: 'success',
      message: 'Source added successfully. Check the Operations tab to monitor progress.',
    });
  };

  const handleRenameSuccess = () => {
    setShowRenameModal(false);
    // Note: with v2 API, we use UUIDs not names, so we keep the same ID
    queryClient.invalidateQueries({ queryKey: ['collection-v2'] });
    addToast({ type: 'success', message: 'Collection renamed successfully' });
  };

  const handleDeleteSuccess = () => {
    setShowDeleteModal(false);
    handleClose();
    // Refresh the collections list in the store
    const { fetchCollections } = useCollectionStore.getState();
    fetchCollections();
    addToast({ type: 'success', message: 'Collection deleted successfully' });
  };

  const handleReindexSuccess = () => {
    setShowReindexModal(false);
    setConfigChanges({});
    setValidationErrors({});
    queryClient.invalidateQueries({ queryKey: ['collection-v2', showCollectionDetailsModal] });
    queryClient.invalidateQueries({ queryKey: ['collection-operations', showCollectionDetailsModal] });
    addToast({ 
      type: 'success', 
      message: 'Re-indexing started successfully. Check the Operations tab to monitor progress.' 
    });
  };

  const validateChunkSize = (value: number): string | undefined => {
    if (value < 100) return 'Chunk size must be at least 100 tokens';
    if (value > 4000) return 'Chunk size cannot exceed 4000 tokens';
    return undefined;
  };

  const validateChunkOverlap = (value: number, chunkSize: number): string | undefined => {
    if (value < 0) return 'Chunk overlap cannot be negative';
    if (value > 200) return 'Chunk overlap cannot exceed 200 tokens';
    if (value >= chunkSize) return 'Chunk overlap must be less than chunk size';
    return undefined;
  };

  const handleChunkSizeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value) || undefined;
    setConfigChanges(prev => ({ ...prev, chunk_size: value }));
    
    if (value !== undefined) {
      const error = validateChunkSize(value);
      setValidationErrors(prev => ({ ...prev, chunk_size: error }));
      
      // Re-validate chunk overlap if it exists
      const currentOverlap = configChanges.chunk_overlap ?? collection?.chunk_overlap;
      if (currentOverlap !== undefined) {
        const overlapError = validateChunkOverlap(currentOverlap, value);
        setValidationErrors(prev => ({ ...prev, chunk_overlap: overlapError }));
      }
    }
  };

  const handleChunkOverlapChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value) || undefined;
    setConfigChanges(prev => ({ ...prev, chunk_overlap: value }));
    
    if (value !== undefined && collection) {
      const chunkSize = configChanges.chunk_size ?? collection.chunk_size;
      const error = validateChunkOverlap(value, chunkSize);
      setValidationErrors(prev => ({ ...prev, chunk_overlap: error }));
    }
  };

  if (!showCollectionDetailsModal) return null;

  return (
    <>
      <div className="fixed inset-0 bg-black bg-opacity-50 z-50" onClick={handleClose} />
      <div className="fixed inset-4 md:inset-[5%] bg-white rounded-lg shadow-xl z-50 flex flex-col max-w-6xl mx-auto">
        {/* Header */}
        <div className="px-6 py-4 border-b">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-gray-900">
                {collection?.name || 'Loading...'}
              </h2>
              {collection && (
                <p className="text-sm text-gray-500 mt-1">
                  {operationsData?.length || 0} operations • {formatNumber(collection.document_count)} documents • {formatNumber(collection.vector_count)} vectors
                </p>
              )}
            </div>
            <button
              onClick={handleClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Action Buttons */}
          <div className="mt-4 flex gap-2">
            <button
              onClick={() => setShowAddDataModal(true)}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              disabled={!collection}
            >
              Add Data
            </button>
            <button
              onClick={() => setShowRenameModal(true)}
              className="px-4 py-2 border border-gray-300 rounded hover:bg-gray-50"
              disabled={!collection}
            >
              Rename
            </button>
            <button
              onClick={() => setShowDeleteModal(true)}
              className="px-4 py-2 border border-red-300 text-red-600 rounded hover:bg-red-50"
              disabled={!collection}
            >
              Delete
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="border-b">
          <nav className="flex space-x-8 px-6" aria-label="Tabs">
            <button
              onClick={() => setActiveTab('overview')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'overview'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab('jobs')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'jobs'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Jobs
            </button>
            <button
              onClick={() => setActiveTab('files')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'files'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Files
            </button>
            <button
              onClick={() => setActiveTab('settings')}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'settings'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Settings
            </button>
          </nav>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {isLoading && (
            <div className="flex justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            </div>
          )}

          {error && (
            <div className="text-center py-12">
              <p className="text-red-600">Failed to load collection details</p>
            </div>
          )}

          {collection && activeTab === 'overview' && (
            <div className="space-y-6">
              {/* Stats */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">Statistics</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-gray-50 p-4 rounded">
                    <dt className="text-sm font-medium text-gray-500">Documents</dt>
                    <dd className="mt-1 text-2xl font-semibold text-gray-900">
                      {formatNumber(collection.document_count)}
                    </dd>
                  </div>
                  <div className="bg-gray-50 p-4 rounded">
                    <dt className="text-sm font-medium text-gray-500">Vectors</dt>
                    <dd className="mt-1 text-2xl font-semibold text-gray-900">
                      {formatNumber(collection.vector_count)}
                    </dd>
                  </div>
                  <div className="bg-gray-50 p-4 rounded">
                    <dt className="text-sm font-medium text-gray-500">Total Size</dt>
                    <dd className="mt-1 text-2xl font-semibold text-gray-900">
                      {formatBytes(collection.total_size_bytes || 0)}
                    </dd>
                  </div>
                  <div className="bg-gray-50 p-4 rounded">
                    <dt className="text-sm font-medium text-gray-500">Operations</dt>
                    <dd className="mt-1 text-2xl font-semibold text-gray-900">
                      {operationsData?.length || 0}
                    </dd>
                  </div>
                </div>
              </div>

              {/* Configuration */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">Configuration</h3>
                <dl className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Embedding Model</dt>
                    <dd className="mt-1 text-sm text-gray-900 font-mono text-xs">{collection.embedding_model}</dd>
                  </div>
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Status</dt>
                    <dd className="mt-1 text-sm text-gray-900 capitalize">
                      {collection.status}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Chunk Size</dt>
                    <dd className="mt-1 text-sm text-gray-900">{collection.chunk_size} characters</dd>
                  </div>
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Chunk Overlap</dt>
                    <dd className="mt-1 text-sm text-gray-900">{collection.chunk_overlap} characters</dd>
                  </div>
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Public</dt>
                    <dd className="mt-1 text-sm text-gray-900">{collection.is_public ? 'Yes' : 'No'}</dd>
                  </div>
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Created</dt>
                    <dd className="mt-1 text-sm text-gray-900">{formatDate(collection.created_at)}</dd>
                  </div>
                </dl>
              </div>

              {/* Source Directories */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">Source Directories</h3>
                {sourceDirs.length > 0 ? (
                  <ul className="space-y-2">
                    {sourceDirs.map((source, index) => (
                      <li key={index} className="flex items-center justify-between text-sm text-gray-900">
                        <div className="flex items-center">
                          <svg className="h-4 w-4 text-gray-400 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                          </svg>
                          {source.path}
                        </div>
                        <span className="text-gray-500">{source.document_count} documents</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-gray-500">No source directories added yet</p>
                )}
              </div>
            </div>
          )}

          {collection && activeTab === 'jobs' && operationsData && (
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Operations History</h3>
              <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 md:rounded-lg">
                <table className="min-w-full divide-y divide-gray-300">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Operation ID
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Type
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Status
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Started
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Completed
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {operationsData.map((operation) => (
                      <tr key={operation.id}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {operation.id.slice(0, 8)}...
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 capitalize">
                          {operation.type.replace('_', ' ')}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            operation.status === 'completed' ? 'bg-green-100 text-green-800' :
                            operation.status === 'failed' ? 'bg-red-100 text-red-800' :
                            operation.status === 'processing' ? 'bg-blue-100 text-blue-800' :
                            operation.status === 'cancelled' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {operation.status}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {operation.started_at ? formatDate(operation.started_at) : '-'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {operation.completed_at ? formatDate(operation.completed_at) : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {collection && activeTab === 'files' && documentsData && (
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Documents ({documentsData.total.toLocaleString()})
              </h3>
              <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 md:rounded-lg">
                <table className="min-w-full divide-y divide-gray-300">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        File Path
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Source
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Chunks
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Created
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {documentsData.documents.map((doc: DocumentResponse) => (
                      <tr key={doc.id}>
                        <td className="px-6 py-4 text-sm text-gray-900">
                          <div className="truncate max-w-md" title={doc.file_path}>
                            {doc.file_path}
                          </div>
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-500">
                          <div className="truncate max-w-xs" title={doc.source_path}>
                            {doc.source_path}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {doc.chunk_count}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {formatDate(doc.created_at)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {documentsData.total > documentsData.per_page && (
                <div className="mt-4 flex items-center justify-between">
                  <div className="text-sm text-gray-700">
                    Showing page {documentsData.page} of {Math.ceil(documentsData.total / documentsData.per_page)}
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setFilesPage(filesPage - 1)}
                      disabled={filesPage === 1}
                      className="px-3 py-1 border rounded text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Previous
                    </button>
                    <button
                      onClick={() => setFilesPage(filesPage + 1)}
                      disabled={filesPage * documentsData.per_page >= documentsData.total}
                      className="px-3 py-1 border rounded text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {collection && activeTab === 'settings' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">Collection Configuration</h3>
                <p className="text-sm text-gray-600 mb-6">
                  Adjust collection settings. Changes will require re-indexing to take effect.
                </p>
                
                <div className="space-y-4">
                  {/* Model (Read-only) */}
                  <div>
                    <label htmlFor="embedding-model" className="block text-sm font-medium text-gray-700">
                      Embedding Model
                    </label>
                    <input
                      id="embedding-model"
                      type="text"
                      value={collection.embedding_model}
                      disabled
                      className="mt-1 block w-full rounded-md border-gray-300 bg-gray-50 shadow-sm text-gray-500 sm:text-sm px-3 py-2"
                      aria-label="Embedding model (read-only)"
                      aria-describedby="model-help"
                    />
                    <p id="model-help" className="mt-1 text-xs text-gray-500">Cannot be changed after collection creation</p>
                  </div>

                  {/* Chunk Size */}
                  <div>
                    <label htmlFor="chunk-size" className="block text-sm font-medium text-gray-700">
                      Chunk Size (characters)
                    </label>
                    <input
                      id="chunk-size"
                      type="number"
                      value={configChanges.chunk_size ?? collection.chunk_size}
                      onChange={handleChunkSizeChange}
                      min="100"
                      max="4000"
                      className={`mt-1 block w-full rounded-md shadow-sm focus:ring-blue-500 sm:text-sm ${
                        validationErrors.chunk_size
                          ? 'border-red-300 focus:border-red-500'
                          : 'border-gray-300 focus:border-blue-500'
                      }`}
                      aria-label="Chunk size in tokens"
                      aria-describedby="chunk-size-help chunk-size-error"
                      aria-invalid={!!validationErrors.chunk_size}
                    />
                    {validationErrors.chunk_size ? (
                      <p id="chunk-size-error" className="mt-1 text-xs text-red-600">{validationErrors.chunk_size}</p>
                    ) : (
                      <p id="chunk-size-help" className="mt-1 text-xs text-gray-500">Recommended: 200-800 characters</p>
                    )}
                  </div>

                  {/* Chunk Overlap */}
                  <div>
                    <label htmlFor="chunk-overlap" className="block text-sm font-medium text-gray-700">
                      Chunk Overlap (characters)
                    </label>
                    <input
                      id="chunk-overlap"
                      type="number"
                      value={configChanges.chunk_overlap ?? collection.chunk_overlap}
                      onChange={handleChunkOverlapChange}
                      min="0"
                      max="200"
                      className={`mt-1 block w-full rounded-md shadow-sm focus:ring-blue-500 sm:text-sm ${
                        validationErrors.chunk_overlap
                          ? 'border-red-300 focus:border-red-500'
                          : 'border-gray-300 focus:border-blue-500'
                      }`}
                      aria-label="Chunk overlap in tokens"
                      aria-describedby="chunk-overlap-help chunk-overlap-error"
                      aria-invalid={!!validationErrors.chunk_overlap}
                    />
                    {validationErrors.chunk_overlap ? (
                      <p id="chunk-overlap-error" className="mt-1 text-xs text-red-600">{validationErrors.chunk_overlap}</p>
                    ) : (
                      <p id="chunk-overlap-help" className="mt-1 text-xs text-gray-500">Recommended: 10-20% of chunk size</p>
                    )}
                  </div>

                  {/* Instruction */}
                  <div>
                    <label htmlFor="embedding-instruction" className="block text-sm font-medium text-gray-700">
                      Embedding Instruction (Optional)
                    </label>
                    <textarea
                      id="embedding-instruction"
                      value={configChanges.instruction ?? ''}
                      onChange={(e) => setConfigChanges({
                        ...configChanges,
                        instruction: e.target.value || undefined
                      })}
                      rows={3}
                      className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                      placeholder="e.g., Represent this document for retrieval:"
                      aria-label="Custom embedding instruction"
                      aria-describedby="instruction-help"
                    />
                    <p id="instruction-help" className="mt-1 text-xs text-gray-500">
                      Custom instruction prepended to documents during embedding
                    </p>
                  </div>
                </div>
              </div>

              {/* Re-index Section */}
              <div className="border-t pt-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Re-index Collection</h3>
                <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4 mb-4" role="alert">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <svg className="h-5 w-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div className="ml-3">
                      <h3 className="text-sm font-medium text-yellow-800">
                        Re-indexing will process all documents again
                      </h3>
                      <div className="mt-2 text-sm text-yellow-700">
                        <p>This operation will:</p>
                        <ul className="list-disc list-inside mt-1">
                          <li>Delete all existing vectors</li>
                          <li>Re-process all documents with new settings</li>
                          <li>Take time depending on collection size</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => setShowReindexModal(true)}
                  disabled={
                    Object.keys(configChanges).length === 0 || 
                    Object.values(validationErrors).some(error => error !== undefined)
                  }
                  className={`px-4 py-2 rounded-md font-medium ${
                    Object.keys(configChanges).length > 0 && 
                    !Object.values(validationErrors).some(error => error !== undefined)
                      ? 'bg-yellow-600 text-white hover:bg-yellow-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500'
                      : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  }`}
                  aria-label="Re-index collection with new configuration"
                >
                  Re-index Collection
                </button>
                {Object.keys(configChanges).length === 0 && (
                  <p className="mt-2 text-sm text-gray-500">
                    Make changes to configuration above to enable re-indexing
                  </p>
                )}
                {Object.keys(configChanges).length > 0 && 
                 Object.values(validationErrors).some(error => error !== undefined) && (
                  <p className="mt-2 text-sm text-red-600">
                    Fix validation errors before re-indexing
                  </p>
                )}
              </div>
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
          collectionId={showCollectionDetailsModal}
          currentName={collection.name}
          onClose={() => setShowRenameModal(false)}
          onSuccess={handleRenameSuccess}
        />
      )}

      {showDeleteModal && collection && (
        <DeleteCollectionModal
          collectionId={showCollectionDetailsModal}
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
            ...configChanges
          }}
          onClose={() => setShowReindexModal(false)}
          onSuccess={handleReindexSuccess}
        />
      )}
    </>
  );
}

export default CollectionDetailsModal;