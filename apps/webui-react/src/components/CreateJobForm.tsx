import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useUIStore } from '../stores/uiStore';
import { useJobsStore, type Job } from '../stores/jobsStore';
import { useDirectoryScanWebSocket } from '../hooks/useDirectoryScanWebSocket';
import { jobsApi, modelsApi } from '../services/api';
import api from '../services/api';

function CreateJobForm() {
  const addToast = useUIStore((state) => state.addToast);
  const setActiveTab = useUIStore((state) => state.setActiveTab);
  const addJob = useJobsStore((state) => state.addJob);

  const [directory, setDirectory] = useState('');
  const [collectionName, setCollectionName] = useState('');
  const [creating, setCreating] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [mode, setMode] = useState<'create' | 'append'>('create');
  const [selectedCollection, setSelectedCollection] = useState('');
  const [hasConfirmedWarnings, setHasConfirmedWarnings] = useState(false);
  
  // Advanced parameters
  const [modelName, setModelName] = useState('Qwen/Qwen3-Embedding-0.6B');
  const [chunkSize, setChunkSize] = useState(600);
  const [chunkOverlap, setChunkOverlap] = useState(200);
  const [batchSize, setBatchSize] = useState(96);
  const [vectorDim, setVectorDim] = useState<number | undefined>(undefined);
  const [quantization, setQuantization] = useState('float32');
  const [instruction, setInstruction] = useState('');
  
  // Inherited settings when in append mode
  const [inheritedSettings, setInheritedSettings] = useState<any>(null);
  const [loadingSettings, setLoadingSettings] = useState(false);
  
  // Available models
  const [models, setModels] = useState<Record<string, any>>({});
  const [loadingModels, setLoadingModels] = useState(false);
  
  // Use the enhanced WebSocket-enabled directory scan hook
  const { scanning, scanResult, scanProgress, error, startScan, reset } = useDirectoryScanWebSocket();
  
  // Fetch existing collections (completed jobs)
  const { data: jobsData } = useQuery({
    queryKey: ['jobs'],
    queryFn: async () => {
      const response = await jobsApi.list();
      return response.data;
    },
  });
  
  // Filter to get only completed collections
  const completedCollections = jobsData?.filter((job: Job) => 
    job.status === 'completed' && job.total_files > 0
  ) || [];
  
  
  // Load available models on mount
  useEffect(() => {
    const loadModels = async () => {
      setLoadingModels(true);
      try {
        const response = await modelsApi.list();
        setModels(response.data.models || {});
      } catch (error) {
        console.error('Failed to load models:', error);
      } finally {
        setLoadingModels(false);
      }
    };
    loadModels();
  }, []);

  // Fetch collection metadata when a collection is selected
  useEffect(() => {
    if (mode === 'append' && selectedCollection) {
      const fetchCollectionMetadata = async () => {
        setLoadingSettings(true);
        try {
          const response = await api.get(`/api/jobs/collection-metadata/${selectedCollection}`);
          setInheritedSettings(response.data);
        } catch (error) {
          console.error('Failed to load collection settings:', error);
          addToast({ type: 'error', message: 'Failed to load collection settings' });
        } finally {
          setLoadingSettings(false);
        }
      };
      fetchCollectionMetadata();
    } else {
      setInheritedSettings(null);
    }
  }, [mode, selectedCollection]);

  const handleScan = () => {
    setHasConfirmedWarnings(false); // Reset confirmation when rescanning
    startScan(directory);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!scanResult) {
      addToast({ type: 'error', message: 'Please scan the directory first' });
      return;
    }

    if (mode === 'append' && !selectedCollection) {
      addToast({ type: 'error', message: 'Please select a collection to add to' });
      return;
    }

    // Check if there are warnings and user hasn't confirmed
    if (scanResult.warnings && scanResult.warnings.length > 0 && !hasConfirmedWarnings) {
      const warningMessages = scanResult.warnings.map(w => w.message).join('\n\n');
      const confirmed = window.confirm(
        `Warning:\n\n${warningMessages}\n\nDo you want to proceed anyway?`
      );
      
      if (!confirmed) {
        return;
      }
      
      setHasConfirmedWarnings(true);
    }

    setCreating(true);

    try {
      let response;
      
      if (mode === 'create') {
        response = await jobsApi.create({
          directory,
          collection_name: collectionName,
          model_name: modelName,
          chunk_size: chunkSize,
          chunk_overlap: chunkOverlap,
          batch_size: batchSize,
          vector_dim: vectorDim,
          quantization,
          instruction: instruction || undefined,
        });
      } else {
        // Add to existing collection
        response = await api.post('/api/jobs/add-to-collection', {
          collection_name: selectedCollection,
          directory_path: directory,
          description: `Adding documents to ${selectedCollection}`,
        });
      }
      
      addJob(response.data);
      addToast({ 
        type: 'success', 
        message: mode === 'create' 
          ? 'Job created successfully' 
          : `Adding documents to ${selectedCollection}` 
      });
      setActiveTab('jobs');
      
      // Trigger a refetch of the jobs list
      setTimeout(() => {
        window.dispatchEvent(new Event('refetch-jobs'));
      }, 100);
      
      // Reset form
      setDirectory('');
      setCollectionName('');
      setSelectedCollection('');
      setMode('create');
      setModelName('Qwen/Qwen3-Embedding-0.6B');
      setChunkSize(600);
      setChunkOverlap(200);
      setBatchSize(96);
      setVectorDim(undefined);
      setQuantization('float32');
      setInstruction('');
      setShowAdvanced(false);
      setHasConfirmedWarnings(false);
      reset();
    } catch (error: any) {
      addToast({
        type: 'error',
        message: error.response?.data?.detail || 'Failed to create job',
      });
    } finally {
      setCreating(false);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Create Embedding Job</h2>
          <p className="text-sm text-gray-600">
            Scan a directory and create embeddings for all supported documents.
          </p>
        </div>

        <div className="space-y-6">
          {/* Mode Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Collection Mode
            </label>
            <div className="flex space-x-4">
              <label className="flex items-center">
                <input
                  type="radio"
                  value="create"
                  checked={mode === 'create'}
                  onChange={(e) => setMode(e.target.value as 'create' | 'append')}
                  className="mr-2 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm">Create New Collection</span>
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  value="append"
                  checked={mode === 'append'}
                  onChange={(e) => setMode(e.target.value as 'create' | 'append')}
                  className="mr-2 text-blue-600 focus:ring-blue-500"
                />
                <span className="text-sm">Add to Existing Collection</span>
              </label>
            </div>
          </div>

          {/* Collection Name or Selection */}
          {mode === 'create' ? (
            <div>
              <label htmlFor="collection" className="block text-sm font-medium text-gray-700 mb-1">
                Job Name <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                name="collection"
                id="collection"
                value={collectionName}
                onChange={(e) => setCollectionName(e.target.value)}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                placeholder="my-documents"
              />
              <p className="mt-1 text-xs text-gray-500">
                A unique name for this collection of documents
              </p>
            </div>
          ) : (
            <div>
              <label htmlFor="existingCollection" className="block text-sm font-medium text-gray-700 mb-1">
                Select Collection <span className="text-red-500">*</span>
              </label>
              <select
                id="existingCollection"
                value={selectedCollection}
                onChange={(e) => setSelectedCollection(e.target.value)}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              >
                <option value="">Select a collection...</option>
                {completedCollections.map((job: Job) => (
                  <option key={job.id} value={job.name}>
                    {job.name} ({job.total_files} files)
                  </option>
                ))}
              </select>
              {completedCollections.length === 0 && (
                <p className="mt-1 text-xs text-gray-500">
                  No completed collections available. Create a new collection first.
                </p>
              )}
            </div>
          )}

          {/* Display inherited settings when in append mode */}
          {mode === 'append' && selectedCollection && loadingSettings && (
            <div className="bg-gray-50 border border-gray-200 rounded-md p-4">
              <p className="text-sm text-gray-600">Loading collection settings...</p>
            </div>
          )}
          
          {mode === 'append' && inheritedSettings && (
            <div className="bg-gray-50 border border-gray-200 rounded-md p-4">
              <h4 className="text-sm font-medium text-gray-900 mb-2">Inherited Settings</h4>
              <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                <div>
                  <span className="font-medium text-gray-700">Model:</span>
                  <span className="ml-2 text-gray-600">{inheritedSettings.model_name}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Chunk Size:</span>
                  <span className="ml-2 text-gray-600">{inheritedSettings.chunk_size} tokens</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Chunk Overlap:</span>
                  <span className="ml-2 text-gray-600">{inheritedSettings.chunk_overlap} tokens</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Batch Size:</span>
                  <span className="ml-2 text-gray-600">{inheritedSettings.batch_size}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Quantization:</span>
                  <span className="ml-2 text-gray-600">{inheritedSettings.quantization}</span>
                </div>
                {inheritedSettings.vector_dim && (
                  <div>
                    <span className="font-medium text-gray-700">Vector Dimension:</span>
                    <span className="ml-2 text-gray-600">{inheritedSettings.vector_dim}</span>
                  </div>
                )}
                {inheritedSettings.instruction && (
                  <div className="col-span-2">
                    <span className="font-medium text-gray-700">Instruction:</span>
                    <span className="ml-2 text-gray-600">{inheritedSettings.instruction}</span>
                  </div>
                )}
              </div>
              <p className="mt-2 text-xs text-gray-500">
                These settings will be used to ensure consistency with the existing collection.
              </p>
            </div>
          )}

          {/* Directory Path with Scan */}
          <div>
            <label htmlFor="directory" className="block text-sm font-medium text-gray-700 mb-1">
              Directory Path <span className="text-red-500">*</span>
            </label>
            <div className="flex">
              <input
                type="text"
                name="directory"
                id="directory"
                value={directory}
                onChange={(e) => setDirectory(e.target.value)}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-l-md focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                placeholder="/path/to/documents"
              />
              <button
                type="button"
                onClick={handleScan}
                disabled={scanning || !directory}
                className="px-4 py-2 border border-l-0 border-gray-300 rounded-r-md bg-gray-50 text-sm font-medium text-gray-700 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {scanning ? 'Scanning...' : 'Scan'}
              </button>
            </div>
          </div>

          {error && (
            <div className="text-sm text-red-600">{error}</div>
          )}

          {scanning && scanProgress && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
              <h4 className="text-sm font-medium text-yellow-900">Scanning...</h4>
              <div className="mt-2">
                {scanProgress.current_path && (
                  <p className="text-sm text-yellow-700 truncate" title={scanProgress.current_path}>
                    {scanProgress.current_path}
                  </p>
                )}
                {scanProgress.files_scanned !== undefined && scanProgress.total_files !== undefined && (
                  <div className="mt-2">
                    <div className="flex justify-between text-sm text-yellow-700">
                      <span>{scanProgress.files_scanned} / {scanProgress.total_files} files</span>
                      {scanProgress.percentage !== undefined && (
                        <span>{scanProgress.percentage.toFixed(0)}%</span>
                      )}
                    </div>
                    <div className="mt-1 w-full bg-yellow-200 rounded-full h-2">
                      <div
                        className="bg-yellow-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${scanProgress.percentage || 0}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {scanResult && (
            <>
              <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="text-sm font-medium text-blue-900">Scan Complete</h4>
                    <div className="mt-1 text-sm text-blue-700">
                      <span className="font-medium">{scanResult.total_files}</span> files found, 
                      <span className="font-medium ml-1">{formatBytes(scanResult.total_size)}</span> total
                    </div>
                  </div>
                  <details className="text-right">
                    <summary className="cursor-pointer text-sm text-blue-600 hover:text-blue-800">
                      View files
                    </summary>
                  </details>
                </div>
                <details open={false}>
                  <summary className="sr-only">File list</summary>
                  <div className="mt-3 border-t border-blue-200 pt-3">
                    <ul className="text-xs text-gray-600 max-h-40 overflow-y-auto space-y-1">
                      {scanResult.files.map((file, index) => (
                        <li key={index} className="truncate">
                          {file}
                        </li>
                      ))}
                    </ul>
                  </div>
                </details>
              </div>
              
              {/* Display warnings if any */}
              {scanResult.warnings && scanResult.warnings.length > 0 && (
                <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4 mt-3">
                  <h4 className="text-sm font-medium text-yellow-900 mb-2">
                    <svg className="inline-block w-5 h-5 mr-1 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    Resource Warning
                  </h4>
                  <div className="space-y-2">
                    {scanResult.warnings.map((warning, index) => (
                      <p key={index} className="text-sm text-yellow-800">
                        {warning.message}
                      </p>
                    ))}
                  </div>
                  <p className="text-xs text-yellow-700 mt-3">
                    You can still proceed, but the operation may take significant time and resources.
                  </p>
                </div>
              )}
            </>
          )}

          {/* Model Selection - Only show when creating new collection */}
          {mode === 'create' && (
            <div>
              <label htmlFor="model" className="block text-sm font-medium text-gray-700 mb-1">
                Embedding Model
              </label>
              <select
                id="model"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                disabled={loadingModels}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              >
                {Object.entries(models).map(([name, info]) => (
                  <option key={name} value={name}>
                    {name} - {info.description}
                  </option>
                ))}
              </select>
              {models[modelName] && (
                <p className="mt-1 text-xs text-gray-500">
                  Dimension: {models[modelName].dimension} | 
                  Memory: {models[modelName].memory_estimate?.[quantization] || 'N/A'} MB
                </p>
              )}
            </div>
          )}

          {/* Chunk Settings - Only show when creating new collection */}
          {mode === 'create' && (
            <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="chunkSize" className="block text-sm font-medium text-gray-700 mb-1">
                Chunk Size (tokens)
              </label>
              <input
                type="number"
                id="chunkSize"
                value={chunkSize}
                onChange={(e) => setChunkSize(Number(e.target.value))}
                min="100"
                max="50000"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              />
              <p className="mt-1 text-xs text-gray-500">
                Tokens per chunk (100-50000)
              </p>
            </div>

            <div>
              <label htmlFor="chunkOverlap" className="block text-sm font-medium text-gray-700 mb-1">
                Chunk Overlap (tokens)
              </label>
              <input
                type="number"
                id="chunkOverlap"
                value={chunkOverlap}
                onChange={(e) => setChunkOverlap(Number(e.target.value))}
                min="0"
                max={chunkSize}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
              />
              <p className="mt-1 text-xs text-gray-500">
                Overlapping tokens
              </p>
            </div>
          </div>
          )}

          {/* Advanced Options Toggle - Only show when creating new collection */}
          {mode === 'create' && (
            <>
              <div className="border-t pt-4">
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="text-sm font-medium text-blue-600 hover:text-blue-500 flex items-center"
                >
                  {showAdvanced ? 'Hide' : 'Show'} Advanced Options
                  <svg
                    className={`ml-1 h-4 w-4 transform transition-transform ${
                      showAdvanced ? 'rotate-180' : ''
                    }`}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 9l-7 7-7-7"
                    />
                  </svg>
                </button>
              </div>

              {/* Advanced Options */}
              {showAdvanced && (
                <div className="space-y-4 mt-4">
                  {/* Advanced Settings Grid */}
                  <div className="grid grid-cols-2 gap-4">
                    {/* Vector Dimension */}
                    <div>
                      <label htmlFor="vectorDim" className="block text-sm font-medium text-gray-700 mb-1">
                        Vector Dimension
                      </label>
                      <input
                        type="number"
                        id="vectorDim"
                        value={vectorDim || ''}
                        onChange={(e) => setVectorDim(e.target.value ? Number(e.target.value) : undefined)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                        placeholder="Auto"
                      />
                      <p className="mt-1 text-xs text-gray-500">
                        Leave empty for default
                      </p>
                    </div>

                    {/* Quantization */}
                    <div>
                      <label htmlFor="quantization" className="block text-sm font-medium text-gray-700 mb-1">
                        Quantization
                      </label>
                      <select
                        id="quantization"
                        value={quantization}
                        onChange={(e) => setQuantization(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                      >
                        <option value="float32">Float32</option>
                        <option value="float16">Float16</option>
                        <option value="int8">Int8</option>
                      </select>
                      <p className="mt-1 text-xs text-gray-500">
                        Model precision level
                      </p>
                    </div>

                    {/* Batch Size */}
                    <div>
                      <label htmlFor="batchSize" className="block text-sm font-medium text-gray-700 mb-1">
                        Batch Size
                      </label>
                      <input
                        type="number"
                        id="batchSize"
                        value={batchSize}
                        onChange={(e) => setBatchSize(Number(e.target.value))}
                        min="1"
                        max="1000"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                      />
                      <p className="mt-1 text-xs text-gray-500">
                        Chunks per batch
                      </p>
                    </div>
                  </div>

                  {/* Custom Instruction - Full width */}
                  <div>
                    <label htmlFor="instruction" className="block text-sm font-medium text-gray-700 mb-1">
                      Custom Instruction (optional)
                    </label>
                    <textarea
                      id="instruction"
                      value={instruction}
                      onChange={(e) => setInstruction(e.target.value)}
                      rows={2}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                      placeholder="Represent this document for searching:"
                    />
                    <p className="mt-1 text-xs text-gray-500">
                      Custom instruction for the embedding model
                    </p>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Form Actions - Right aligned like original */}
        <div className="flex justify-end space-x-3 pt-6 border-t">
          <button
            type="button"
            onClick={() => {
              setDirectory('');
              setCollectionName('');
              setSelectedCollection('');
              setMode('create');
              setModelName('Qwen/Qwen3-Embedding-0.6B');
              setChunkSize(600);
              setChunkOverlap(200);
              setBatchSize(96);
              setVectorDim(undefined);
              setQuantization('float32');
              setInstruction('');
              setShowAdvanced(false);
              setHasConfirmedWarnings(false);
              reset();
            }}
            className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
          >
            Reset
          </button>
          <button
            type="submit"
            disabled={creating || !scanResult || (mode === 'create' ? !collectionName : !selectedCollection)}
            className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {creating ? (mode === 'create' ? 'Creating...' : 'Adding...') : (mode === 'create' ? 'Create Job' : 'Add Documents')}
          </button>
        </div>
      </form>
    </div>
  );
}

export default CreateJobForm;