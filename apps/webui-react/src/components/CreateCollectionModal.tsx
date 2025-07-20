import { useState, useEffect, useRef } from 'react';
import { useCollectionStore } from '../stores/collectionStore';
import { useUIStore } from '../stores/uiStore';
import { useNavigate } from 'react-router-dom';
import { useDirectoryScan } from '../hooks/useDirectoryScan';
import { getInputClassName, getInputClassNameWithBase } from '../utils/formStyles';
import type { CreateCollectionRequest } from '../types/collection';

interface CreateCollectionModalProps {
  onClose: () => void;
  onSuccess: () => void;
}

const DEFAULT_EMBEDDING_MODEL = 'Qwen/Qwen3-Embedding-0.6B';
const DEFAULT_CHUNK_SIZE = 512;
const DEFAULT_CHUNK_OVERLAP = 50;
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
  const { createCollection, addSource } = useCollectionStore();
  const { addToast } = useUIStore();
  const navigate = useNavigate();
  const { scanning, scanResult, error: scanError, startScan, reset: resetScan } = useDirectoryScan();
  const formRef = useRef<HTMLFormElement>(null);
  
  const [formData, setFormData] = useState<CreateCollectionRequest>({
    name: '',
    description: '',
    embedding_model: DEFAULT_EMBEDDING_MODEL,
    quantization: DEFAULT_QUANTIZATION,
    chunk_size: DEFAULT_CHUNK_SIZE,
    chunk_overlap: DEFAULT_CHUNK_OVERLAP,
    is_public: false,
  });
  
  const [sourcePath, setSourcePath] = useState<string>('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

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
    
    if (formData.chunk_size! < 100 || formData.chunk_size! > 2000) {
      newErrors.chunk_size = 'Chunk size must be between 100 and 2000';
    }
    
    if (formData.chunk_overlap! < 0 || formData.chunk_overlap! >= formData.chunk_size!) {
      newErrors.chunk_overlap = 'Chunk overlap must be between 0 and chunk size';
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
      // Step 1: Create the collection
      const collection = await createCollection(formData);
      
      // Step 2: Add initial source if provided
      if (sourcePath.trim()) {
        try {
          await addSource(collection.id, sourcePath.trim(), {
            chunk_size: formData.chunk_size,
            chunk_overlap: formData.chunk_overlap,
          });
          
          // Show success with source addition
          addToast({
            message: 'Collection created successfully! Navigating to collection...',
            type: 'success'
          });
          
          // Call onSuccess first, then navigate
          onSuccess();
          
          // Delay navigation slightly to let user see the success feedback
          setTimeout(() => {
            navigate(`/collections/${collection.id}`);
          }, 1000);
          
          // Exit early since we already called onSuccess
          return;
        } catch (sourceError) {
          // Collection was created but source addition failed
          addToast({
            message: 'Collection created but failed to add source: ' + 
                     (sourceError instanceof Error ? sourceError.message : 'Unknown error'),
            type: 'warning'
          });
          
          // Still call onSuccess since collection was created
          onSuccess();
          return;
        }
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
      addToast({
        message: error instanceof Error ? error.message : 'Failed to create collection',
        type: 'error'
      });
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
  };

  const handleScan = async () => {
    if (sourcePath.trim()) {
      await startScan(sourcePath.trim());
    }
  };

  return (
    <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg max-w-lg w-full max-h-[90vh] overflow-y-auto relative">
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
            <h3 className="text-lg font-medium text-gray-900">Create New Collection</h3>
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
                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
              >
                <option value="Qwen/Qwen3-Embedding-0.6B">Qwen3-Embedding-0.6B (Default - Fast)</option>
                <option value="intfloat/e5-base-v2">E5-Base-v2 (Balanced)</option>
                <option value="sentence-transformers/all-MiniLM-L6-v2">All-MiniLM-L6-v2 (Lightweight)</option>
              </select>
              <p className="mt-1 text-sm text-gray-500">
                Choose the AI model for converting documents to searchable vectors
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
                  <div className="grid grid-cols-2 gap-4">
                    {/* Chunk Size */}
                    <div>
                      <label htmlFor="chunk_size" className="block text-sm font-medium text-gray-700">
                        Chunk Size
                      </label>
                      <input
                        type="number"
                        id="chunk_size"
                        value={formData.chunk_size}
                        onChange={(e) => handleChange('chunk_size', parseInt(e.target.value) || DEFAULT_CHUNK_SIZE)}
                        disabled={isSubmitting}
                        min={100}
                        max={2000}
                        className={getInputClassName(!!errors.chunk_size, isSubmitting)}
                      />
                      {errors.chunk_size && (
                        <p className="mt-1 text-sm text-red-600">{errors.chunk_size}</p>
                      )}
                    </div>

                    {/* Chunk Overlap */}
                    <div>
                      <label htmlFor="chunk_overlap" className="block text-sm font-medium text-gray-700">
                        Chunk Overlap
                      </label>
                      <input
                        type="number"
                        id="chunk_overlap"
                        value={formData.chunk_overlap}
                        onChange={(e) => handleChange('chunk_overlap', parseInt(e.target.value) || DEFAULT_CHUNK_OVERLAP)}
                        disabled={isSubmitting}
                        min={0}
                        max={formData.chunk_size! - 1}
                        className={getInputClassName(!!errors.chunk_overlap, isSubmitting)}
                      />
                      {errors.chunk_overlap && (
                        <p className="mt-1 text-sm text-red-600">{errors.chunk_overlap}</p>
                      )}
                    </div>
                  </div>
                  
                  <p className="text-sm text-gray-500">
                    Smaller chunks provide more precise results but may lose context
                  </p>

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
              disabled={isSubmitting}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting ? (
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