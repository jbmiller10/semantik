import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useUIStore } from '../stores/uiStore';
import { collectionsApi } from '../services/api';
import AddDataToCollectionModal from './AddDataToCollectionModal';
import RenameCollectionModal from './RenameCollectionModal';
import DeleteCollectionModal from './DeleteCollectionModal';
import ReindexCollectionModal from './ReindexCollectionModal';

interface CollectionDetails {
  name: string;
  stats: {
    total_files: number;
    total_vectors: number;
    total_size: number;
    job_count: number;
  };
  configuration: {
    model_name: string;
    chunk_size: number;
    chunk_overlap: number;
    quantization: string;
    vector_dim: number | null;
    instruction: string | null;
  };
  source_directories: string[];
  jobs: Array<{
    id: string;
    status: string;
    created_at: string;
    updated_at: string;
    directory_path: string;
    total_files: number;
    processed_files: number;
    failed_files: number;
    mode: string;
  }>;
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

  const { data: details, isLoading, error } = useQuery({
    queryKey: ['collection-details', showCollectionDetailsModal],
    queryFn: async () => {
      if (!showCollectionDetailsModal) return null;
      const response = await collectionsApi.getDetails(showCollectionDetailsModal);
      return response.data as CollectionDetails;
    },
    enabled: !!showCollectionDetailsModal,
  });

  const { data: filesData } = useQuery({
    queryKey: ['collection-files', showCollectionDetailsModal, filesPage],
    queryFn: async () => {
      if (!showCollectionDetailsModal || activeTab !== 'files') return null;
      const response = await collectionsApi.getFiles(showCollectionDetailsModal, filesPage);
      return response.data;
    },
    enabled: !!showCollectionDetailsModal && activeTab === 'files',
  });

  const handleClose = () => {
    setShowCollectionDetailsModal(null);
    setActiveTab('overview');
    setFilesPage(1);
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
    queryClient.invalidateQueries({ queryKey: ['collection-details', showCollectionDetailsModal] });
    queryClient.invalidateQueries({ queryKey: ['collections'] });
    addToast({
      type: 'success',
      message: 'Job created successfully. Check the Jobs tab to monitor progress.',
    });
  };

  const handleRenameSuccess = (newName: string) => {
    setShowRenameModal(false);
    setShowCollectionDetailsModal(newName);
    queryClient.invalidateQueries({ queryKey: ['collections'] });
    queryClient.invalidateQueries({ queryKey: ['collection-details'] });
    addToast({ type: 'success', message: 'Collection renamed successfully' });
  };

  const handleDeleteSuccess = () => {
    setShowDeleteModal(false);
    handleClose();
    queryClient.invalidateQueries({ queryKey: ['collections'] });
    addToast({ type: 'success', message: 'Collection deleted successfully' });
  };

  const handleReindexSuccess = () => {
    setShowReindexModal(false);
    setConfigChanges({});
    queryClient.invalidateQueries({ queryKey: ['collection-details', showCollectionDetailsModal] });
    queryClient.invalidateQueries({ queryKey: ['collections'] });
    addToast({ 
      type: 'success', 
      message: 'Re-indexing started successfully. Check the Jobs tab to monitor progress.' 
    });
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
                {details?.name || showCollectionDetailsModal}
              </h2>
              {details && (
                <p className="text-sm text-gray-500 mt-1">
                  {details.stats.job_count} jobs • {details.stats.total_files} files • {details.stats.total_vectors} vectors
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
              disabled={!details}
            >
              Add Data
            </button>
            <button
              onClick={() => setShowRenameModal(true)}
              className="px-4 py-2 border border-gray-300 rounded hover:bg-gray-50"
              disabled={!details}
            >
              Rename
            </button>
            <button
              onClick={() => setShowDeleteModal(true)}
              className="px-4 py-2 border border-red-300 text-red-600 rounded hover:bg-red-50"
              disabled={!details}
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

          {details && activeTab === 'overview' && (
            <div className="space-y-6">
              {/* Stats */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">Statistics</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-gray-50 p-4 rounded">
                    <dt className="text-sm font-medium text-gray-500">Total Files</dt>
                    <dd className="mt-1 text-2xl font-semibold text-gray-900">
                      {details.stats.total_files.toLocaleString()}
                    </dd>
                  </div>
                  <div className="bg-gray-50 p-4 rounded">
                    <dt className="text-sm font-medium text-gray-500">Total Vectors</dt>
                    <dd className="mt-1 text-2xl font-semibold text-gray-900">
                      {details.stats.total_vectors.toLocaleString()}
                    </dd>
                  </div>
                  <div className="bg-gray-50 p-4 rounded">
                    <dt className="text-sm font-medium text-gray-500">Total Size</dt>
                    <dd className="mt-1 text-2xl font-semibold text-gray-900">
                      {formatBytes(details.stats.total_size)}
                    </dd>
                  </div>
                  <div className="bg-gray-50 p-4 rounded">
                    <dt className="text-sm font-medium text-gray-500">Jobs</dt>
                    <dd className="mt-1 text-2xl font-semibold text-gray-900">
                      {details.stats.job_count}
                    </dd>
                  </div>
                </div>
              </div>

              {/* Configuration */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">Configuration</h3>
                <dl className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Model</dt>
                    <dd className="mt-1 text-sm text-gray-900">{details.configuration.model_name}</dd>
                  </div>
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Vector Dimensions</dt>
                    <dd className="mt-1 text-sm text-gray-900">
                      {details.configuration.vector_dim || 'Auto'}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Chunk Size</dt>
                    <dd className="mt-1 text-sm text-gray-900">{details.configuration.chunk_size} tokens</dd>
                  </div>
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Chunk Overlap</dt>
                    <dd className="mt-1 text-sm text-gray-900">{details.configuration.chunk_overlap} tokens</dd>
                  </div>
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Quantization</dt>
                    <dd className="mt-1 text-sm text-gray-900">{details.configuration.quantization}</dd>
                  </div>
                  {details.configuration.instruction && (
                    <div className="md:col-span-2">
                      <dt className="text-sm font-medium text-gray-500">Instruction</dt>
                      <dd className="mt-1 text-sm text-gray-900">{details.configuration.instruction}</dd>
                    </div>
                  )}
                </dl>
              </div>

              {/* Source Directories */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">Source Directories</h3>
                <ul className="space-y-2">
                  {details.source_directories.map((dir, index) => (
                    <li key={index} className="flex items-center text-sm text-gray-900">
                      <svg className="h-4 w-4 text-gray-400 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                      </svg>
                      {dir}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {details && activeTab === 'jobs' && (
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">Jobs</h3>
              <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 md:rounded-lg">
                <table className="min-w-full divide-y divide-gray-300">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Job ID
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Status
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Mode
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Files
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Created
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {details.jobs.map((job) => (
                      <tr key={job.id}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          {job.id.slice(0, 8)}...
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            job.status === 'completed' ? 'bg-green-100 text-green-800' :
                            job.status === 'failed' ? 'bg-red-100 text-red-800' :
                            job.status === 'running' ? 'bg-blue-100 text-blue-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {job.status}
                          </span>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {job.mode}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {job.processed_files}/{job.total_files}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {formatDate(job.created_at)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {details && activeTab === 'files' && filesData && (
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                Files ({filesData.total.toLocaleString()})
              </h3>
              <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 md:rounded-lg">
                <table className="min-w-full divide-y divide-gray-300">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        File Path
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Size
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Vectors
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Status
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {filesData.files.map((file: any) => (
                      <tr key={file.id}>
                        <td className="px-6 py-4 text-sm text-gray-900">
                          <div className="truncate max-w-md" title={file.path}>
                            {file.path}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {formatBytes(file.size)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {file.vectors_created}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            file.status === 'completed' ? 'bg-green-100 text-green-800' :
                            file.status === 'failed' ? 'bg-red-100 text-red-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {file.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {filesData.pages > 1 && (
                <div className="mt-4 flex items-center justify-between">
                  <div className="text-sm text-gray-700">
                    Showing page {filesData.page} of {filesData.pages}
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
                      disabled={filesPage === filesData.pages}
                      className="px-3 py-1 border rounded text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Next
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {details && activeTab === 'settings' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-4">Collection Configuration</h3>
                <p className="text-sm text-gray-600 mb-6">
                  Adjust collection settings. Changes will require re-indexing to take effect.
                </p>
                
                <div className="space-y-4">
                  {/* Model (Read-only) */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700">
                      Embedding Model
                    </label>
                    <input
                      type="text"
                      value={details.configuration.model_name}
                      disabled
                      className="mt-1 block w-full rounded-md border-gray-300 bg-gray-50 shadow-sm text-gray-500 sm:text-sm px-3 py-2"
                    />
                    <p className="mt-1 text-xs text-gray-500">Cannot be changed after collection creation</p>
                  </div>

                  {/* Chunk Size */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700">
                      Chunk Size (tokens)
                    </label>
                    <input
                      type="number"
                      value={configChanges.chunk_size ?? details.configuration.chunk_size}
                      onChange={(e) => setConfigChanges({
                        ...configChanges,
                        chunk_size: parseInt(e.target.value) || undefined
                      })}
                      min="100"
                      max="4000"
                      className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                    />
                    <p className="mt-1 text-xs text-gray-500">Recommended: 200-800 tokens</p>
                  </div>

                  {/* Chunk Overlap */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700">
                      Chunk Overlap (tokens)
                    </label>
                    <input
                      type="number"
                      value={configChanges.chunk_overlap ?? details.configuration.chunk_overlap}
                      onChange={(e) => setConfigChanges({
                        ...configChanges,
                        chunk_overlap: parseInt(e.target.value) || undefined
                      })}
                      min="0"
                      max="200"
                      className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                    />
                    <p className="mt-1 text-xs text-gray-500">Recommended: 10-20% of chunk size</p>
                  </div>

                  {/* Instruction */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700">
                      Embedding Instruction (Optional)
                    </label>
                    <textarea
                      value={configChanges.instruction ?? details.configuration.instruction ?? ''}
                      onChange={(e) => setConfigChanges({
                        ...configChanges,
                        instruction: e.target.value || undefined
                      })}
                      rows={3}
                      className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
                      placeholder="e.g., Represent this document for retrieval:"
                    />
                    <p className="mt-1 text-xs text-gray-500">
                      Custom instruction prepended to documents during embedding
                    </p>
                  </div>
                </div>
              </div>

              {/* Re-index Section */}
              <div className="border-t pt-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Re-index Collection</h3>
                <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4 mb-4">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <svg className="h-5 w-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
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
                  disabled={Object.keys(configChanges).length === 0}
                  className={`px-4 py-2 rounded-md font-medium ${
                    Object.keys(configChanges).length > 0
                      ? 'bg-yellow-600 text-white hover:bg-yellow-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500'
                      : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  }`}
                >
                  Re-index Collection
                </button>
                {Object.keys(configChanges).length === 0 && (
                  <p className="mt-2 text-sm text-gray-500">
                    Make changes to configuration above to enable re-indexing
                  </p>
                )}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Sub-modals */}
      {showAddDataModal && details && (
        <AddDataToCollectionModal
          collectionName={details.name}
          configuration={details.configuration}
          onClose={() => setShowAddDataModal(false)}
          onSuccess={handleAddDataSuccess}
        />
      )}

      {showRenameModal && details && (
        <RenameCollectionModal
          currentName={details.name}
          onClose={() => setShowRenameModal(false)}
          onSuccess={handleRenameSuccess}
        />
      )}

      {showDeleteModal && details && (
        <DeleteCollectionModal
          collectionName={details.name}
          stats={details.stats}
          onClose={() => setShowDeleteModal(false)}
          onSuccess={handleDeleteSuccess}
        />
      )}

      {showReindexModal && details && (
        <ReindexCollectionModal
          collectionName={details.name}
          configChanges={configChanges}
          onClose={() => setShowReindexModal(false)}
          onSuccess={handleReindexSuccess}
        />
      )}
    </>
  );
}

export default CollectionDetailsModal;