import { useState, useEffect } from 'react';
import { useCollectionStore } from '../stores/collectionStore';
import { useUIStore } from '../stores/uiStore';
import type { CreateCollectionRequest } from '../types/collection';

interface CreateCollectionModalProps {
  onClose: () => void;
  onSuccess: () => void;
}

const DEFAULT_EMBEDDING_MODEL = 'Qwen/Qwen3-Embedding-0.6B';
const DEFAULT_CHUNK_SIZE = 512;
const DEFAULT_CHUNK_OVERLAP = 50;

function CreateCollectionModal({ onClose, onSuccess }: CreateCollectionModalProps) {
  const { createCollection } = useCollectionStore();
  const { addToast } = useUIStore();
  
  const [formData, setFormData] = useState<CreateCollectionRequest>({
    name: '',
    description: '',
    embedding_model: DEFAULT_EMBEDDING_MODEL,
    chunk_size: DEFAULT_CHUNK_SIZE,
    chunk_overlap: DEFAULT_CHUNK_OVERLAP,
    is_public: false,
  });
  
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

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
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    setIsSubmitting(true);
    
    try {
      await createCollection(formData);
      onSuccess();
    } catch (error) {
      addToast({
        message: error instanceof Error ? error.message : 'Failed to create collection',
        type: 'error'
      });
    } finally {
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

  return (
    <div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg max-w-lg w-full max-h-[90vh] overflow-y-auto">
        <form onSubmit={handleSubmit}>
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">Create New Collection</h3>
            <p className="mt-1 text-sm text-gray-500">
              Create a new collection to store and search your documents
            </p>
          </div>
          
          <div className="px-6 py-4 space-y-4">
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
                className={`mt-1 block w-full rounded-md shadow-sm sm:text-sm ${
                  errors.name
                    ? 'border-red-300 focus:ring-red-500 focus:border-red-500'
                    : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'
                }`}
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
                rows={3}
                className={`mt-1 block w-full rounded-md shadow-sm sm:text-sm ${
                  errors.description
                    ? 'border-red-300 focus:ring-red-500 focus:border-red-500'
                    : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'
                }`}
                placeholder="A collection of technical documentation..."
              />
              {errors.description && (
                <p className="mt-1 text-sm text-red-600">{errors.description}</p>
              )}
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

            {/* Advanced Settings */}
            <div className="border-t pt-4">
              <h4 className="text-sm font-medium text-gray-700 mb-3">Advanced Settings</h4>
              
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
                    min={100}
                    max={2000}
                    className={`mt-1 block w-full rounded-md shadow-sm sm:text-sm ${
                      errors.chunk_size
                        ? 'border-red-300 focus:ring-red-500 focus:border-red-500'
                        : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'
                    }`}
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
                    min={0}
                    max={formData.chunk_size! - 1}
                    className={`mt-1 block w-full rounded-md shadow-sm sm:text-sm ${
                      errors.chunk_overlap
                        ? 'border-red-300 focus:ring-red-500 focus:border-red-500'
                        : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'
                    }`}
                  />
                  {errors.chunk_overlap && (
                    <p className="mt-1 text-sm text-red-600">{errors.chunk_overlap}</p>
                  )}
                </div>
              </div>
              
              <p className="mt-2 text-sm text-gray-500">
                Smaller chunks provide more precise results but may lose context
              </p>
            </div>

            {/* Public Collection */}
            <div className="flex items-center">
              <input
                id="is_public"
                type="checkbox"
                checked={formData.is_public}
                onChange={(e) => handleChange('is_public', e.target.checked)}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <label htmlFor="is_public" className="ml-2 block text-sm text-gray-900">
                Make this collection public
              </label>
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