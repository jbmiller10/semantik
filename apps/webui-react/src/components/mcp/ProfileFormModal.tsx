import { useState, useEffect, useCallback, useRef } from 'react';
import type {
  MCPProfile,
  MCPProfileFormData,
  MCPSearchType,
  MCPSearchMode,
} from '../../types/mcp-profile';
import {
  DEFAULT_PROFILE_FORM_DATA,
  SEARCH_TYPE_LABELS,
  SEARCH_TYPE_DESCRIPTIONS,
  SEARCH_MODE_LABELS,
  SEARCH_MODE_DESCRIPTIONS,
} from '../../types/mcp-profile';
import { useCollections } from '../../hooks/useCollections';
import {
  useCreateMCPProfile,
  useUpdateMCPProfile,
} from '../../hooks/useMCPProfiles';

interface ProfileFormModalProps {
  profile?: MCPProfile | null;
  onClose: () => void;
}

export default function ProfileFormModal({
  profile,
  onClose,
}: ProfileFormModalProps) {
  const isEditing = !!profile;
  const { data: collections, isLoading: collectionsLoading, error: collectionsError } = useCollections();
  const createProfile = useCreateMCPProfile();
  const updateProfile = useUpdateMCPProfile();

  const [formData, setFormData] = useState<MCPProfileFormData>(
    DEFAULT_PROFILE_FORM_DATA
  );
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Initialize form data when editing
  useEffect(() => {
    if (profile) {
      setFormData({
        name: profile.name,
        description: profile.description,
        collection_ids: profile.collections.map((c) => c.id),
        enabled: profile.enabled,
        search_type: profile.search_type,
        result_count: profile.result_count,
        use_reranker: profile.use_reranker,
        score_threshold: profile.score_threshold,
        hybrid_alpha: profile.hybrid_alpha,
        search_mode: profile.search_mode,
        rrf_k: profile.rrf_k,
      });
      // Show advanced if any advanced fields have values
      if (profile.score_threshold !== null || profile.hybrid_alpha !== null || profile.rrf_k !== null) {
        setShowAdvanced(true);
      }
    }
  }, [profile]);

  // Handle escape key to close modal
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !isSubmitting) {
        onClose();
      }
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [onClose, isSubmitting]);

  // Focus trap for accessibility
  const modalRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const modal = modalRef.current;
    if (!modal) return;

    const focusableElements = modal.querySelectorAll(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return;

      if (e.shiftKey && document.activeElement === firstElement) {
        e.preventDefault();
        lastElement?.focus();
      } else if (!e.shiftKey && document.activeElement === lastElement) {
        e.preventDefault();
        firstElement?.focus();
      }
    };

    // Focus first focusable element on mount
    firstElement?.focus();

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const validateForm = useCallback((): boolean => {
    const newErrors: Record<string, string> = {};

    // Name validation
    if (!formData.name.trim()) {
      newErrors.name = 'Profile name is required';
    } else if (!/^[a-z][a-z0-9_-]*$/.test(formData.name)) {
      newErrors.name =
        'Name must start with a lowercase letter and contain only lowercase letters, numbers, hyphens, and underscores';
    } else if (formData.name.length > 64) {
      newErrors.name = 'Name must be 64 characters or less';
    }

    // Description validation
    if (!formData.description.trim()) {
      newErrors.description = 'Description is required';
    } else if (formData.description.length > 1000) {
      newErrors.description = 'Description must be 1000 characters or less';
    }

    // Collections validation
    if (formData.collection_ids.length === 0) {
      newErrors.collection_ids = 'At least one collection is required';
    }

    // Result count validation
    if (formData.result_count < 1 || formData.result_count > 100) {
      newErrors.result_count = 'Result count must be between 1 and 100';
    }

    // Score threshold validation
    if (
      formData.score_threshold !== null &&
      (formData.score_threshold < 0 || formData.score_threshold > 1)
    ) {
      newErrors.score_threshold = 'Score threshold must be between 0 and 1';
    }

    // Hybrid alpha validation
    if (
      formData.hybrid_alpha !== null &&
      (formData.hybrid_alpha < 0 || formData.hybrid_alpha > 1)
    ) {
      newErrors.hybrid_alpha = 'Hybrid alpha must be between 0 and 1';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [formData]);

  const handleChange = (
    field: keyof MCPProfileFormData,
    value: unknown
  ) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
    // Clear error when field is modified
    if (errors[field]) {
      setErrors((prev) => ({ ...prev, [field]: '' }));
    }
  };

  const handleCollectionToggle = (collectionId: string) => {
    setFormData((prev) => {
      const ids = prev.collection_ids.includes(collectionId)
        ? prev.collection_ids.filter((id) => id !== collectionId)
        : [...prev.collection_ids, collectionId];
      return { ...prev, collection_ids: ids };
    });
    if (errors.collection_ids) {
      setErrors((prev) => ({ ...prev, collection_ids: '' }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitError(null);

    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);
    try {
      if (isEditing && profile) {
        await updateProfile.mutateAsync({
          profileId: profile.id,
          data: {
            name: formData.name,
            description: formData.description,
            collection_ids: formData.collection_ids,
            enabled: formData.enabled,
            search_type: formData.search_type,
            result_count: formData.result_count,
            use_reranker: formData.use_reranker,
            score_threshold: formData.score_threshold,
            hybrid_alpha: formData.hybrid_alpha,
            search_mode: formData.search_mode,
            rrf_k: formData.rrf_k,
          },
        });
      } else {
        await createProfile.mutateAsync({
          name: formData.name,
          description: formData.description,
          collection_ids: formData.collection_ids,
          enabled: formData.enabled,
          search_type: formData.search_type,
          result_count: formData.result_count,
          use_reranker: formData.use_reranker,
          score_threshold: formData.score_threshold,
          hybrid_alpha: formData.hybrid_alpha,
          search_mode: formData.search_mode,
          rrf_k: formData.rrf_k,
        });
      }
      onClose();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An error occurred';
      setSubmitError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <>
      <div
        className="fixed inset-0 bg-black bg-opacity-50 z-[60]"
        onClick={isSubmitting ? undefined : onClose}
      />
      <div
        ref={modalRef}
        className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 bg-white rounded-lg shadow-xl z-[60] w-full max-w-2xl max-h-[90vh] overflow-y-auto"
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <h2 id="modal-title" className="text-xl font-semibold text-gray-900">
            {isEditing ? 'Edit MCP Profile' : 'Create MCP Profile'}
          </h2>
          <button
            onClick={onClose}
            disabled={isSubmitting}
            className="text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 rounded"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit}>
          <div className="px-6 py-4 space-y-6">
            {/* Name */}
            <div>
              <label
                htmlFor="name"
                className="block text-sm font-medium text-gray-700"
              >
                Profile Name <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                id="name"
                value={formData.name}
                onChange={(e) => handleChange('name', e.target.value.toLowerCase())}
                disabled={isSubmitting}
                placeholder="coding"
                aria-invalid={!!errors.name}
                aria-describedby={errors.name ? 'name-error' : undefined}
                className={`mt-1 block w-full rounded-md shadow-sm sm:text-sm px-3 py-2 border ${
                  errors.name
                    ? 'border-red-300 focus:ring-red-500 focus:border-red-500'
                    : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'
                }`}
              />
              <p className="mt-1 text-xs text-gray-500">
                Used as the MCP tool name (search_{formData.name || 'name'})
              </p>
              {errors.name && (
                <p id="name-error" className="mt-1 text-sm text-red-600">{errors.name}</p>
              )}
            </div>

            {/* Description */}
            <div>
              <label
                htmlFor="description"
                className="block text-sm font-medium text-gray-700"
              >
                Description <span className="text-red-500">*</span>
              </label>
              <textarea
                id="description"
                rows={3}
                value={formData.description}
                onChange={(e) => handleChange('description', e.target.value)}
                disabled={isSubmitting}
                placeholder="Search coding documentation, API references, and technical guides"
                aria-invalid={!!errors.description}
                aria-describedby={errors.description ? 'description-error' : undefined}
                className={`mt-1 block w-full rounded-md shadow-sm sm:text-sm px-3 py-2 border ${
                  errors.description
                    ? 'border-red-300 focus:ring-red-500 focus:border-red-500'
                    : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'
                }`}
              />
              <p className="mt-1 text-xs text-gray-500">
                This is shown to the AI to help it understand when to use this profile ({formData.description.length}/1000)
              </p>
              {errors.description && (
                <p id="description-error" className="mt-1 text-sm text-red-600">{errors.description}</p>
              )}
            </div>

            {/* Collections */}
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Collections <span className="text-red-500">*</span>
              </label>
              <p className="mt-1 text-xs text-gray-500 mb-2">
                Select which collections this profile can search. Selected collections are searched together.
              </p>
              {collectionsLoading ? (
                <div className="flex items-center justify-center py-4">
                  <svg className="animate-spin h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  <span className="ml-2 text-sm text-gray-500">Loading collections...</span>
                </div>
              ) : collectionsError ? (
                <div className="text-sm text-red-600 bg-red-50 rounded-md p-4 text-center">
                  Failed to load collections. Please try again.
                </div>
              ) : !collections || collections.length === 0 ? (
                <div className="text-sm text-gray-500 bg-gray-50 rounded-md p-4 text-center">
                  No collections available. Create a collection first.
                </div>
              ) : (
                <div
                  role="group"
                  aria-describedby={errors.collection_ids ? 'collections-error' : undefined}
                  className={`border rounded-md max-h-48 overflow-y-auto ${
                    errors.collection_ids ? 'border-red-300' : 'border-gray-300'
                  }`}
                >
                  {collections.map((collection) => (
                    <label
                      key={collection.id}
                      className="flex items-center px-3 py-2 hover:bg-gray-50 cursor-pointer border-b border-gray-100 last:border-b-0"
                    >
                      <input
                        type="checkbox"
                        checked={formData.collection_ids.includes(collection.id)}
                        onChange={() => handleCollectionToggle(collection.id)}
                        disabled={isSubmitting}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <span className="ml-2 text-sm text-gray-700">
                        {collection.name}
                      </span>
                      <span className="ml-auto text-xs text-gray-400">
                        {collection.document_count} docs
                      </span>
                    </label>
                  ))}
                </div>
              )}
              {errors.collection_ids && (
                <p id="collections-error" className="mt-1 text-sm text-red-600">{errors.collection_ids}</p>
              )}
            </div>

            {/* Search Settings */}
            <div className="grid grid-cols-2 gap-4">
              {/* Search Type */}
              <div>
                <label
                  htmlFor="search_type"
                  className="block text-sm font-medium text-gray-700"
                >
                  Search Type
                </label>
                <select
                  id="search_type"
                  value={formData.search_type}
                  onChange={(e) =>
                    handleChange('search_type', e.target.value as MCPSearchType)
                  }
                  disabled={isSubmitting}
                  className="mt-1 block w-full rounded-md shadow-sm sm:text-sm px-3 py-2 border border-gray-300 focus:ring-blue-500 focus:border-blue-500"
                >
                  {Object.entries(SEARCH_TYPE_LABELS).map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
                <p className="mt-1 text-xs text-gray-500">
                  {SEARCH_TYPE_DESCRIPTIONS[formData.search_type]}
                </p>
              </div>

              {/* Result Count */}
              <div>
                <label
                  htmlFor="result_count"
                  className="block text-sm font-medium text-gray-700"
                >
                  Default Results
                </label>
                <input
                  type="number"
                  id="result_count"
                  min={1}
                  max={100}
                  value={formData.result_count}
                  onChange={(e) =>
                    handleChange('result_count', parseInt(e.target.value) || 10)
                  }
                  disabled={isSubmitting}
                  aria-invalid={!!errors.result_count}
                  aria-describedby={errors.result_count ? 'result-count-error' : 'result-count-help'}
                  className={`mt-1 block w-full rounded-md shadow-sm sm:text-sm px-3 py-2 border ${
                    errors.result_count
                      ? 'border-red-300 focus:ring-red-500 focus:border-red-500'
                      : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'
                  }`}
                />
                <p id="result-count-help" className="mt-1 text-xs text-gray-500">
                  Number of chunks to return (1-100)
                </p>
                {errors.result_count && (
                  <p id="result-count-error" className="mt-1 text-sm text-red-600">{errors.result_count}</p>
                )}
              </div>
            </div>

            {/* Search Mode Settings */}
            <div className="grid grid-cols-2 gap-4">
              {/* Search Mode */}
              <div>
                <label
                  htmlFor="search_mode"
                  className="block text-sm font-medium text-gray-700"
                >
                  Search Mode
                </label>
                <select
                  id="search_mode"
                  value={formData.search_mode}
                  onChange={(e) =>
                    handleChange('search_mode', e.target.value as MCPSearchMode)
                  }
                  disabled={isSubmitting}
                  className="mt-1 block w-full rounded-md shadow-sm sm:text-sm px-3 py-2 border border-gray-300 focus:ring-blue-500 focus:border-blue-500"
                >
                  {Object.entries(SEARCH_MODE_LABELS).map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
                <p className="mt-1 text-xs text-gray-500">
                  {SEARCH_MODE_DESCRIPTIONS[formData.search_mode]}
                </p>
              </div>

              {/* RRF K - only show when hybrid mode */}
              {formData.search_mode === 'hybrid' && (
                <div>
                  <label
                    htmlFor="rrf_k"
                    className="block text-sm font-medium text-gray-700"
                  >
                    RRF Constant (k)
                  </label>
                  <input
                    type="number"
                    id="rrf_k"
                    min={1}
                    max={1000}
                    value={formData.rrf_k ?? 60}
                    onChange={(e) =>
                      handleChange('rrf_k', e.target.value ? parseInt(e.target.value) : null)
                    }
                    disabled={isSubmitting}
                    className="mt-1 block w-full rounded-md shadow-sm sm:text-sm px-3 py-2 border border-gray-300 focus:ring-blue-500 focus:border-blue-500"
                  />
                  <p className="mt-1 text-xs text-gray-500">
                    Higher values give more weight to top-ranked results (default: 60)
                  </p>
                </div>
              )}
            </div>

            {/* Reranker Toggle */}
            <div className="flex items-center justify-between">
              <div>
                <span className="text-sm font-medium text-gray-700">
                  Use Reranker
                </span>
                <p className="text-xs text-gray-500">
                  Cross-encoder reranking improves result quality but may be slower
                </p>
              </div>
              <button
                type="button"
                onClick={() => handleChange('use_reranker', !formData.use_reranker)}
                disabled={isSubmitting}
                aria-label={formData.use_reranker ? 'Disable reranker' : 'Enable reranker'}
                className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                  formData.use_reranker ? 'bg-blue-600' : 'bg-gray-200'
                }`}
                role="switch"
                aria-checked={formData.use_reranker}
              >
                <span
                  className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                    formData.use_reranker ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>

            {/* Enabled Toggle */}
            <div className="flex items-center justify-between">
              <div>
                <span className="text-sm font-medium text-gray-700">
                  Enabled
                </span>
                <p className="text-xs text-gray-500">
                  Disabled profiles won't appear as MCP tools
                </p>
              </div>
              <button
                type="button"
                onClick={() => handleChange('enabled', !formData.enabled)}
                disabled={isSubmitting}
                aria-label={formData.enabled ? 'Disable profile' : 'Enable profile'}
                className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                  formData.enabled ? 'bg-blue-600' : 'bg-gray-200'
                }`}
                role="switch"
                aria-checked={formData.enabled}
              >
                <span
                  className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
                    formData.enabled ? 'translate-x-5' : 'translate-x-0'
                  }`}
                />
              </button>
            </div>

            {/* Advanced Settings */}
            <div className="border-t border-gray-200 pt-4">
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="flex items-center text-sm font-medium text-gray-700 hover:text-gray-900"
              >
                <svg
                  className={`w-4 h-4 mr-2 transition-transform ${
                    showAdvanced ? 'rotate-90' : ''
                  }`}
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 5l7 7-7 7"
                  />
                </svg>
                Advanced Settings
              </button>

              {showAdvanced && (
                <div className="mt-4 space-y-4 pl-6">
                  {/* Score Threshold */}
                  <div>
                    <label
                      htmlFor="score_threshold"
                      className="block text-sm font-medium text-gray-700"
                    >
                      Score Threshold
                    </label>
                    <input
                      type="number"
                      id="score_threshold"
                      min={0}
                      max={1}
                      step={0.1}
                      value={formData.score_threshold ?? ''}
                      onChange={(e) =>
                        handleChange(
                          'score_threshold',
                          e.target.value ? parseFloat(e.target.value) : null
                        )
                      }
                      disabled={isSubmitting}
                      placeholder="Optional (0-1)"
                      aria-invalid={!!errors.score_threshold}
                      aria-describedby={errors.score_threshold ? 'score-threshold-error' : undefined}
                      className={`mt-1 block w-full rounded-md shadow-sm sm:text-sm px-3 py-2 border ${
                        errors.score_threshold
                          ? 'border-red-300 focus:ring-red-500 focus:border-red-500'
                          : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'
                      }`}
                    />
                    <p className="mt-1 text-xs text-gray-500">
                      Filter out results below this relevance score (0-1). Higher values return fewer, more relevant results.
                    </p>
                    {errors.score_threshold && (
                      <p id="score-threshold-error" className="mt-1 text-sm text-red-600">{errors.score_threshold}</p>
                    )}
                  </div>

                  {/* Hybrid Alpha (only for hybrid search) */}
                  {formData.search_type === 'hybrid' && (
                    <div>
                      <label
                        htmlFor="hybrid_alpha"
                        className="block text-sm font-medium text-gray-700"
                      >
                        Hybrid Alpha
                      </label>
                      <input
                        type="number"
                        id="hybrid_alpha"
                        min={0}
                        max={1}
                        step={0.1}
                        value={formData.hybrid_alpha ?? ''}
                        onChange={(e) =>
                          handleChange(
                            'hybrid_alpha',
                            e.target.value ? parseFloat(e.target.value) : null
                          )
                        }
                        disabled={isSubmitting}
                        placeholder="Optional (0-1)"
                        aria-invalid={!!errors.hybrid_alpha}
                        aria-describedby={errors.hybrid_alpha ? 'hybrid-alpha-error' : undefined}
                        className={`mt-1 block w-full rounded-md shadow-sm sm:text-sm px-3 py-2 border ${
                          errors.hybrid_alpha
                            ? 'border-red-300 focus:ring-red-500 focus:border-red-500'
                            : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'
                        }`}
                      />
                      <p className="mt-1 text-xs text-gray-500">
                        Balance between keyword (0) and semantic (1) search
                      </p>
                      {errors.hybrid_alpha && (
                        <p id="hybrid-alpha-error" className="mt-1 text-sm text-red-600">{errors.hybrid_alpha}</p>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Submit Error */}
          {submitError && (
            <div className="mx-6 mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
              <p className="text-sm text-red-700">{submitError}</p>
            </div>
          )}

          {/* Footer */}
          <div className="px-6 py-4 bg-gray-50 border-t border-gray-200 flex justify-end gap-3 rounded-b-lg">
            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSubmitting && (
                <svg
                  className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  />
                </svg>
              )}
              {isEditing ? 'Update Profile' : 'Create Profile'}
            </button>
          </div>
        </form>
      </div>
    </>
  );
}
