import { useState, useEffect, useRef } from 'react';
import { useCreateCollection } from '../hooks/useCollections';
import { useAddSource } from '../hooks/useCollectionOperations';
import { useOperationProgress } from '../hooks/useOperationProgress';
import { useEmbeddingModels } from '../hooks/useModels';
import { useUIStore } from '../stores/uiStore';
import { useChunkingStore } from '../stores/chunkingStore';
import { useNavigate } from 'react-router-dom';
import { useDirectoryScan } from '../hooks/useDirectoryScan';
import { getInputClassName, getInputClassNameWithBase } from '../utils/formStyles';
import { SimplifiedChunkingStrategySelector } from './chunking/SimplifiedChunkingStrategySelector';
import ErrorBoundary from './ErrorBoundary';
import { ConfigurationErrorFallback } from './common/ChunkingErrorFallback';
import type { CreateCollectionRequest } from '../types/collection';

interface CreateCollectionModalProps {
  onClose: () => void;
  onSuccess: () => void;
}

const DEFAULT_EMBEDDING_MODEL = 'Qwen/Qwen3-Embedding-0.6B';
const DEFAULT_QUANTIZATION = 'float16';

// Utility function to format file sizes
function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const k = 1024;
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${units[i]}`;
}

function CreateCollectionModal({ onClose, onSuccess }: CreateCollectionModalProps) {
  const createCollectionMutation = useCreateCollection();
  const addSourceMutation = useAddSource();
  const { addToast } = useUIStore();
  const { strategyConfig } = useChunkingStore();
  const navigate = useNavigate();
  const { scanning, scanResult, error: scanError, startScan, reset: resetScan } = useDirectoryScan();
  const { data: modelsData, isLoading: modelsLoading } = useEmbeddingModels();
  const formRef = useRef<HTMLFormElement>(null);
  
  const [formData, setFormData] = useState<CreateCollectionRequest>({
    name: '',
    description: '',
    embedding_model: DEFAULT_EMBEDDING_MODEL,
    quantization: DEFAULT_QUANTIZATION,
    is_public: false,
  });
  
  const [sourcePath, setSourcePath] = useState<string>('');
  const [detectedFileType, setDetectedFileType] = useState<string | undefined>(undefined);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [pendingIndexOperationId, setPendingIndexOperationId] = useState<string | null>(null);
  const [collectionIdForSource, setCollectionIdForSource] = useState<string | null>(null);
  const [sourcePathForDelayedAdd, setSourcePathForDelayedAdd] = useState<string | null>(null);

  // Monitor INDEX operation progress
  useOperationProgress(pendingIndexOperationId, {
    showToasts: false, // We'll show our own toasts
    onComplete: async () => {
      // INDEX operation completed, now we can add the source
      if (collectionIdForSource && sourcePathForDelayedAdd) {
        try {
          await addSourceMutation.mutateAsync({
            collectionId: collectionIdForSource,
            sourcePath: sourcePathForDelayedAdd,
            config: {
              chunking_strategy: strategyConfig.strategy,
              chunking_config: strategyConfig.parameters,
            }
          });
          
          // Show success with source addition
          addToast({
            message: 'Collection created and source added successfully! Navigating to collection...',
            type: 'success'
          });
          
          // Call onSuccess first, then navigate
          onSuccess();
          
          // Delay navigation slightly to let user see the success feedback
          setTimeout(() => {
            navigate(`/collections/${collectionIdForSource}`);
          }, 1000);
        } catch (sourceError) {
          // Collection was created but source addition failed
          addToast({
            message: 'Collection created but failed to add source: ' + 
                     (sourceError instanceof Error ? sourceError.message : 'Unknown error'),
            type: 'warning'
          });
          
          // Still call onSuccess since collection was created
          onSuccess();
        } finally {
          // Clean up state
          setPendingIndexOperationId(null);
          setCollectionIdForSource(null);
          setSourcePathForDelayedAdd(null);
          setIsSubmitting(false);
        }
      }
    },
    onError: (error) => {
      addToast({
        message: `INDEX operation failed: ${error}`,
        type: 'error'
      });
      setPendingIndexOperationId(null);
      setCollectionIdForSource(null);
      setSourcePathForDelayedAdd(null);
      setIsSubmitting(false);
    }
  });

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !isSubmitting) {
        onClose();
      }
    };
    
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [onClose, isSubmitting]);

  // Removed the useEffect that was preventing all form submissions
  // The form onSubmit handler already prevents default behavior

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};
    
    if (!formData.name.trim()) {
      newErrors.name = 'Collection name is required';
    } else if (formData.name.length > 100) {
      newErrors.name = 'Collection name must be 100 characters or less';
    }
    
    if (formData.description && formData.description.length > 500) {
      newErrors.description = 'Description must be 500 characters or less';
    }
    
    if (sourcePath && !sourcePath.trim()) {
      newErrors.sourcePath = 'Source path cannot be empty if provided';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    // Always prevent default first to avoid page reload
    e.preventDefault();
    e.stopPropagation();
    
    try {
      if (!validateForm()) {
        return;
      }
    } catch (error) {
      console.error('Validation error:', error);
      return;
    }
    
    setIsSubmitting(true);
    
    try {
      // Step 1: Create the collection with chunking configuration
      const response = await createCollectionMutation.mutateAsync({
        ...formData,
        chunking_strategy: strategyConfig.strategy,
        chunking_config: strategyConfig.parameters,
      });
      
      // The response should include the initial INDEX operation ID
      // Let's check if we have an operation in the response
      const indexOperationId = response.initial_operation_id;
      
      // Step 2: Handle initial source if provided
      if (sourcePath.trim() && indexOperationId) {
        // Set up state to track the INDEX operation and add source when it completes
        setPendingIndexOperationId(indexOperationId);
        setCollectionIdForSource(response.id);
        setSourcePathForDelayedAdd(sourcePath.trim());
        
        // Show progress message
        addToast({
          message: 'Collection created! Waiting for initialization before adding source...',
          type: 'info'
        });
        
        // Don't set isSubmitting to false yet - it will be done when operations complete
        return;
      } else if (sourcePath.trim() && !indexOperationId) {
        // Fallback: if we don't get an operation ID, just show success
        // This shouldn't happen but handle it gracefully
        addToast({
          message: 'Collection created! Please add the source manually.',
          type: 'warning'
        });
        onSuccess();
        setIsSubmitting(false);
        return;
      } else {
        // Show success for collection without source
        addToast({
          message: 'Collection created successfully!',
          type: 'success'
        });
        
        // Call parent's onSuccess to close modal and refresh list
        onSuccess();
      }
    } catch (error) {
      // Error handling is already done by the mutations
      // This catch block is for any unexpected errors
      if (!createCollectionMutation.isError && !addSourceMutation.isError) {
        addToast({
          message: error instanceof Error ? error.message : 'Failed to create collection',
          type: 'error'
        });
      }
      setIsSubmitting(false);
    }
  };

  const handleChange = (field: keyof CreateCollectionRequest, value: string | number | boolean | undefined) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    // Clear error when field is modified
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  const handleSourcePathChange = (value: string) => {
    setSourcePath(value);
    // Clear error when field is modified
    if (errors.sourcePath) {
      setErrors(prev => ({ ...prev, sourcePath: '' }));
    }
    // Reset scan results when path changes
    if (scanResult) {
      resetScan();
    }
    // Detect file type from path for chunking recommendations
    if (value.trim()) {
      const extension = value.split('.').pop()?.toLowerCase();
      setDetectedFileType(extension);
    } else {
      setDetectedFileType(undefined);
    }
  };

  const handleScan = async () => {
    if (sourcePath.trim()) {
      await startScan(sourcePath.trim());
    }
  };

  return (
    <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg max-w-lg md:max-w-2xl lg:max-w-4xl w-full max-h-[90vh] overflow-y-auto relative" role="dialog" aria-modal="true" aria-labelledby="modal-title">
        {/* Loading overlay */}
        {isSubmitting && (
          <div className="absolute inset-0 bg-white bg-opacity-90 flex items-center justify-center z-10 rounded-lg">
            <div className="text-center">
              <svg className="animate-spin h-8 w-8 text-blue-600 mx-auto mb-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <p className="text-gray-700 font-medium">Creating collection...</p>
            </div>
          </div>
        )}
        <form 
          ref={formRef}
          onSubmit={handleSubmit}
          action="#"
          method="POST"
          onKeyDown={(e) => {
            // Prevent form submission on Enter key in input fields
            if (e.key === 'Enter' && e.target instanceof HTMLInputElement && e.target.type !== 'submit') {
              e.preventDefault();
            }
          }}
        >
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 id="modal-title" className="text-lg font-medium text-gray-900">Create New Collection</h3>
            <p className="mt-1 text-sm text-gray-500">
              Create a new collection to store and search your documents
            </p>
          </div>
          
          <div className="px-6 py-4 space-y-4">
            {/* Validation Summary */}
            {Object.keys(errors).length > 0 && !isSubmitting && (
              <div className="bg-red-50 border border-red-200 rounded-md p-4">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-red-800">
                      Please fix the following errors:
                    </h3>
                    <ul className="mt-2 text-sm text-red-700 list-disc list-inside">
                      {Object.entries(errors).map(([field, error]) => (
                        <li key={field}>{error}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}
            {/* Collection Name */}
            <div>
              <label htmlFor="name" className="block text-sm font-medium text-gray-700">
                Collection Name <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                id="name"
                value={formData.name}
                onChange={(e) => handleChange('name', e.target.value)}
                disabled={isSubmitting}
                className={getInputClassName(!!errors.name, isSubmitting)}
                placeholder="My Documents"
                autoFocus
              />
              {errors.name && (
                <p className="mt-1 text-sm text-red-600">{errors.name}</p>
              )}
            </div>

            {/* Description */}
            <div>
              <label htmlFor="description" className="block text-sm font-medium text-gray-700">
                Description
              </label>
              <textarea
                id="description"
                value={formData.description || ''}
                onChange={(e) => handleChange('description', e.target.value)}
                disabled={isSubmitting}
                rows={3}
                className={getInputClassName(!!errors.description, isSubmitting)}
                placeholder="A collection of technical documentation..."
              />
              {errors.description && (
                <p className="mt-1 text-sm text-red-600">{errors.description}</p>
              )}
            </div>

            {/* Source Path */}
            <div>
              <label htmlFor="sourcePath" className="block text-sm font-medium text-gray-700">
                Initial Source Directory (Optional)
              </label>
              <div className="mt-1 flex rounded-md shadow-sm">
                <input
                  type="text"
                  id="sourcePath"
                  value={sourcePath}
                  onChange={(e) => handleSourcePathChange(e.target.value)}
                  disabled={isSubmitting || scanning}
                  className={getInputClassNameWithBase(!!errors.sourcePath || !!scanError, isSubmitting || scanning, 'flex-1 rounded-l-md shadow-sm sm:text-sm px-3 py-2 border appearance-none')}
                  placeholder="/path/to/documents"
                />
                <button
                  type="button"
                  onClick={handleScan}
                  disabled={!sourcePath.trim() || isSubmitting || scanning}
                  className={`inline-flex items-center px-4 py-2 border border-l-0 border-gray-300 rounded-r-md text-sm font-medium ${
                    !sourcePath.trim() || isSubmitting || scanning
                      ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                      : 'bg-white text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500'
                  }`}
                >
                  {scanning ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Scanning...
                    </>
                  ) : (
                    'Scan'
                  )}
                </button>
              </div>
              {errors.sourcePath && (
                <p className="mt-1 text-sm text-red-600">{errors.sourcePath}</p>
              )}
              {scanError && (
                <p className="mt-1 text-sm text-red-600">{scanError}</p>
              )}
              
              {/* Scan Results */}
              {scanResult && (
                <div className={`mt-3 p-3 rounded-md ${
                  scanResult.total_files > 10000 
                    ? 'bg-yellow-50 border border-yellow-200' 
                    : 'bg-blue-50 border border-blue-200'
                }`}>
                  <div className="flex items-start">
                    {scanResult.total_files > 10000 ? (
                      <svg className="h-5 w-5 text-yellow-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                    ) : (
                      <svg className="h-5 w-5 text-blue-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    )}
                    <div className="ml-3 flex-1">
                      <p className={`text-sm font-medium ${
                        scanResult.total_files > 10000 ? 'text-yellow-800' : 'text-blue-800'
                      }`}>
                        Found {scanResult.total_files.toLocaleString()} files ({formatFileSize(scanResult.total_size)})
                      </p>
                      {scanResult.total_files > 10000 && (
                        <p className="mt-1 text-sm text-yellow-700">
                          Warning: Large directory detected. Indexing may take a significant amount of time.
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              )}
              
              <p className="mt-1 text-sm text-gray-500">
                Specify a directory to start indexing immediately after collection creation
              </p>
            </div>

            {/* Embedding Model */}
            <div>
              <label htmlFor="embedding_model" className="block text-sm font-medium text-gray-700">
                Embedding Model
              </label>
              <select
                id="embedding_model"
                value={formData.embedding_model}
                onChange={(e) => handleChange('embedding_model', e.target.value)}
                disabled={modelsLoading}
                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md disabled:bg-gray-100 disabled:cursor-wait"
              >
                {modelsLoading ? (
                  <option value="">Loading models...</option>
                ) : modelsData?.models ? (
                  Object.entries(modelsData.models)
                    .sort(([a], [b]) => a.localeCompare(b))
                    .map(([modelName, config]) => (
                      <option key={modelName} value={modelName}>
                        {config.description || modelName}
                        {config.provider && config.provider !== 'dense_local' && ` (${config.provider})`}
                      </option>
                    ))
                ) : (
                  <option value="Qwen/Qwen3-Embedding-0.6B">Qwen3-Embedding-0.6B (Default - Fast)</option>
                )}
              </select>
              <p className="mt-1 text-sm text-gray-500">
                Choose the AI model for converting documents to searchable vectors
                {modelsData?.current_device && (
                  <span className="ml-1 text-gray-400">
                    (running on {modelsData.current_device})
                  </span>
                )}
              </p>
            </div>

            {/* Quantization */}
            <div>
              <label htmlFor="quantization" className="block text-sm font-medium text-gray-700">
                Model Quantization
              </label>
              <select
                id="quantization"
                value={formData.quantization}
                onChange={(e) => handleChange('quantization', e.target.value)}
                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
              >
                <option value="float32">float32 (Highest Precision)</option>
                <option value="float16">float16 (Balanced - Default)</option>
                <option value="int8">int8 (Lowest Memory Usage)</option>
              </select>
              <p className="mt-1 text-sm text-gray-500">
                Choose the precision level for the embedding model. Lower precision uses less memory but may affect accuracy
              </p>
            </div>

            {/* Chunking Strategy */}
            <ErrorBoundary 
              level="component"
              fallback={(error, resetError) => (
                <ConfigurationErrorFallback 
                  error={error} 
                  resetError={resetError}
                  onResetConfiguration={() => {
                    const chunkingStore = useChunkingStore.getState();
                    chunkingStore.resetToDefaults();
                    resetError();
                  }}
                />
              )}
            >
              <SimplifiedChunkingStrategySelector 
                disabled={isSubmitting}
                fileType={detectedFileType}
              />
            </ErrorBoundary>

            {/* Advanced Settings Accordion */}
            <div className="border-t pt-4">
              <button
                type="button"
                onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
                className="flex items-center justify-between w-full text-left focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded-md p-2 -m-2"
              >
                <h4 className="text-sm font-medium text-gray-700">Advanced Settings</h4>
                <svg
                  className={`h-5 w-5 text-gray-400 transform transition-transform ${
                    showAdvancedSettings ? 'rotate-180' : ''
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              
              {showAdvancedSettings && (
                <div className="mt-4 space-y-4">
                  {/* Public Collection */}
                  <div className="flex items-center">
                    <input
                      id="is_public"
                      type="checkbox"
                      checked={formData.is_public}
                      onChange={(e) => handleChange('is_public', e.target.checked)}
                      disabled={isSubmitting}
                      className={`h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded ${
                        isSubmitting ? 'cursor-not-allowed' : ''
                      }`}
                    />
                    <label htmlFor="is_public" className="ml-2 block text-sm text-gray-900">
                      Make this collection public
                    </label>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="px-6 py-4 bg-gray-50 border-t border-gray-200 flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting || createCollectionMutation.isPending || addSourceMutation.isPending}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting || createCollectionMutation.isPending || addSourceMutation.isPending}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting || createCollectionMutation.isPending || addSourceMutation.isPending ? (
                <>
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white inline" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Creating...
                </>
              ) : (
                'Create Collection'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default CreateCollectionModal;
export { CreateCollectionModal };