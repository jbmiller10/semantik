import { useState, useEffect, useRef, useCallback } from 'react';
import { Loader2, Sparkles } from 'lucide-react';
import { useCreateCollection } from '../hooks/useCollections';
import { useAddSource } from '../hooks/useCollectionOperations';
import { useOperationProgress } from '../hooks/useOperationProgress';
import { useEmbeddingModels } from '../hooks/useModels';
import { useConnectorCatalog, useGitPreview, useImapPreview } from '../hooks/useConnectors';
import { useTemplates } from '../hooks/useTemplates';
import { useUIStore } from '../stores/uiStore';
import { useChunkingStore } from '../stores/chunkingStore';
import { usePreferences } from '../hooks/usePreferences';
import { useNavigate } from 'react-router-dom';
// import { getInputClassName } from '../utils/formStyles';
import { SimplifiedChunkingStrategySelector } from './chunking/SimplifiedChunkingStrategySelector';
import { ConnectorTypeSelector, ConnectorForm } from './connectors';
import { shouldShowField } from '../types/connector';
import ErrorBoundary from './ErrorBoundary';
import { ConfigurationErrorFallback } from './common/ChunkingErrorFallback';
import type { CreateCollectionRequest, SyncMode } from '../types/collection';
import type { GitPreviewResponse, ImapPreviewResponse } from '../types/connector';
import type { ChunkingStrategyType } from '../types/chunking';

interface CreateCollectionModalProps {
  onClose: () => void;
  onSuccess: () => void;
}

const DEFAULT_EMBEDDING_MODEL = 'Qwen/Qwen3-Embedding-0.6B';
const DEFAULT_QUANTIZATION = 'float16';

function CreateCollectionModal({ onClose, onSuccess }: CreateCollectionModalProps) {
  const createCollectionMutation = useCreateCollection();
  const addSourceMutation = useAddSource();
  const { addToast } = useUIStore();
  const { strategyConfig, setStrategy, updateConfiguration } = useChunkingStore();
  const { data: prefs } = usePreferences();
  const navigate = useNavigate();
  const { data: modelsData, isLoading: modelsLoading } = useEmbeddingModels();
  const formRef = useRef<HTMLFormElement>(null);
  const [prefsApplied, setPrefsApplied] = useState(false);

  // Connector catalog and preview hooks
  const { data: catalog, isLoading: catalogLoading } = useConnectorCatalog();
  const gitPreviewMutation = useGitPreview();
  const imapPreviewMutation = useImapPreview();

  // Pipeline templates
  const { data: templates, isLoading: templatesLoading } = useTemplates();
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);

  const [formData, setFormData] = useState<CreateCollectionRequest>({
    name: '',
    description: '',
    embedding_model: DEFAULT_EMBEDDING_MODEL,
    quantization: DEFAULT_QUANTIZATION,
    is_public: false,
    sync_mode: 'one_time',
    sync_interval_minutes: 60,
  });

  // Connector state - default to 'none' so user can skip adding a source
  const [connectorType, setConnectorType] = useState<string>('none');
  const [configValues, setConfigValues] = useState<Record<string, unknown>>({});
  const [secrets, setSecrets] = useState<Record<string, string>>({});
  const [previewResult, setPreviewResult] = useState<GitPreviewResponse | ImapPreviewResponse | null>(null);
  const [detectedFileType, setDetectedFileType] = useState<string | undefined>(undefined);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  // Sparse indexing state
  const [sparseEnabled, setSparseEnabled] = useState(false);
  const [sparsePlugin, setSparsePlugin] = useState<'bm25-local' | 'splade-local'>('bm25-local');
  const [bm25Config, setBm25Config] = useState({ k1: 1.5, b: 0.75 });
  const [pendingIndexOperationId, setPendingIndexOperationId] = useState<string | null>(null);
  const [collectionIdForSource, setCollectionIdForSource] = useState<string | null>(null);
  const [connectorDataForDelayedAdd, setConnectorDataForDelayedAdd] = useState<{
    type: string;
    config: Record<string, unknown>;
    secrets?: Record<string, string>;
    sourcePath: string;
  } | null>(null);

  // Monitor INDEX operation progress
  useOperationProgress(pendingIndexOperationId, {
    showToasts: false, // We'll show our own toasts
    onComplete: async () => {
      // INDEX operation completed, now we can add the source
      if (collectionIdForSource && connectorDataForDelayedAdd) {
        try {
          await addSourceMutation.mutateAsync({
            collectionId: collectionIdForSource,
            sourceType: connectorDataForDelayedAdd.type,
            sourceConfig: connectorDataForDelayedAdd.config,
            secrets: connectorDataForDelayedAdd.secrets,
            sourcePath: connectorDataForDelayedAdd.sourcePath,
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
          setConnectorDataForDelayedAdd(null);
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
      setConnectorDataForDelayedAdd(null);
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

  // Apply user preferences to form on first load
  useEffect(() => {
    if (prefs && !prefsApplied) {
      // Apply form defaults
      setFormData(prev => ({
        ...prev,
        embedding_model: prefs.collection_defaults.embedding_model ?? DEFAULT_EMBEDDING_MODEL,
        quantization: prefs.collection_defaults.quantization,
      }));

      // Apply sparse indexing defaults
      setSparseEnabled(prefs.collection_defaults.enable_sparse);
      setSparsePlugin(prefs.collection_defaults.sparse_type === 'splade' ? 'splade-local' : 'bm25-local');

      // Apply chunking defaults - set strategy first, then update configuration
      setStrategy(prefs.collection_defaults.chunking_strategy as ChunkingStrategyType);
      updateConfiguration({
        chunkSize: prefs.collection_defaults.chunk_size,
        chunkOverlap: prefs.collection_defaults.chunk_overlap,
      });

      setPrefsApplied(true);
    }
  }, [prefs, prefsApplied, setStrategy, updateConfiguration]);

  // Initialize default values when connector type changes
  useEffect(() => {
    // Handle 'none' type - clear all connector state
    if (connectorType === 'none') {
      setConfigValues({});
      setSecrets({});
      setPreviewResult(null);
      return;
    }

    if (!catalog) return;

    const definition = catalog[connectorType];
    if (!definition) return;

    // Set default values from field definitions
    const defaults: Record<string, unknown> = {};
    for (const field of definition.fields) {
      if (field.default !== undefined) {
        defaults[field.name] = field.default;
      }
    }

    setConfigValues(defaults);
    setSecrets({});
    setPreviewResult(null);
    // Don't clear errors on type change - only clear connector-related errors
  }, [connectorType, catalog]);

  // Handle connector type change
  const handleTypeChange = useCallback((type: string) => {
    setConnectorType(type);
    // Detect file type for directory connector
    if (type !== 'directory') {
      setDetectedFileType(undefined);
    }
  }, []);

  // Handle preview/test connection
  const handlePreview = useCallback(async () => {
    setPreviewResult(null);

    try {
      if (connectorType === 'git') {
        const result = await gitPreviewMutation.mutateAsync({
          repo_url: configValues.repo_url as string,
          ref: (configValues.ref as string) || 'main',
          auth_method: (configValues.auth_method as 'none' | 'https_token' | 'ssh_key') || 'none',
          token: secrets.token,
          ssh_key: secrets.ssh_key,
          ssh_passphrase: secrets.ssh_passphrase,
          include_globs: configValues.include_globs as string[],
          exclude_globs: configValues.exclude_globs as string[],
        });
        setPreviewResult(result);
      } else if (connectorType === 'imap') {
        const result = await imapPreviewMutation.mutateAsync({
          host: configValues.host as string,
          port: configValues.port as number,
          use_ssl: configValues.use_ssl as boolean,
          username: configValues.username as string,
          password: secrets.password,
          mailboxes: configValues.mailboxes as string[],
        });
        setPreviewResult(result);

        // If successful, populate mailboxes options from result
        if (result.valid && result.mailboxes_found.length > 0 && !configValues.mailboxes) {
          const defaultMailbox = result.mailboxes_found.includes('INBOX')
            ? ['INBOX']
            : [result.mailboxes_found[0]];
          setConfigValues((prev) => ({ ...prev, mailboxes: defaultMailbox }));
        }
      }
    } catch (error) {
      addToast({
        type: 'error',
        message: error instanceof Error ? error.message : 'Connection test failed',
      });
    }
  }, [connectorType, configValues, secrets, gitPreviewMutation, imapPreviewMutation, addToast]);

  // Build source path for display/backward compat
  const getSourcePath = useCallback((): string => {
    switch (connectorType) {
      case 'none':
        return '';
      case 'directory':
        return (configValues.path as string) || '';
      case 'git':
        return (configValues.repo_url as string) || '';
      case 'imap':
        return `${configValues.username || ''}@${configValues.host || ''}`;
      default:
        return '';
    }
  }, [connectorType, configValues]);

  // Check if user has provided any source configuration
  const hasSourceConfig = useCallback((): boolean => {
    const sourcePath = getSourcePath();
    return sourcePath.trim().length > 0;
  }, [getSourcePath]);

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

    // Validate sync configuration
    if (formData.sync_mode === 'continuous') {
      if (!formData.sync_interval_minutes || formData.sync_interval_minutes < 15) {
        newErrors.sync_interval_minutes = 'Sync interval must be at least 15 minutes for continuous sync';
      }
    }

    // Validate connector fields if user has started configuring a source
    if (catalog && hasSourceConfig()) {
      const definition = catalog[connectorType];
      if (definition) {
        // Validate required fields
        for (const field of definition.fields) {
          if (field.required && shouldShowField(field, configValues)) {
            const value = configValues[field.name];
            if (value === undefined || value === '' || value === null) {
              newErrors[field.name] = `${field.label} is required`;
            }
          }
        }

        // Validate required secrets
        for (const secret of definition.secrets) {
          if (secret.required && shouldShowField(secret, configValues)) {
            if (!secrets[secret.name]) {
              newErrors[secret.name] = `${secret.label} is required`;
            }
          }
        }
      }
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
      // Step 1: Create the collection with chunking and sparse configuration
      const createRequest: CreateCollectionRequest = {
        ...formData,
        chunking_strategy: strategyConfig.strategy,
        chunking_config: strategyConfig.parameters,
      };

      // Include sparse indexing config if enabled
      if (sparseEnabled) {
        createRequest.sparse_index_config = {
          enabled: true,
          plugin_id: sparsePlugin,
          model_config_data: sparsePlugin === 'bm25-local' ? bm25Config : undefined,
        };
      }

      const response = await createCollectionMutation.mutateAsync(createRequest);

      // The response should include the initial INDEX operation ID
      const indexOperationId = response.initial_operation_id;

      // Step 2: Handle initial source if provided
      const sourcePath = getSourcePath();
      if (sourcePath && indexOperationId) {
        // Set up state to track the INDEX operation and add source when it completes
        setPendingIndexOperationId(indexOperationId);
        setCollectionIdForSource(response.id);
        setConnectorDataForDelayedAdd({
          type: connectorType,
          config: configValues,
          secrets: Object.keys(secrets).length > 0 ? secrets : undefined,
          sourcePath,
        });

        // Show progress message
        addToast({
          message: 'Collection created! Waiting for initialization before adding source...',
          type: 'info'
        });

        // Don't set isSubmitting to false yet - it will be done when operations complete
        return;
      } else if (sourcePath && !indexOperationId) {
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
    setFormData(prev => {
      const updated = { ...prev, [field]: value };

      // When embedding model changes, adjust quantization based on model capabilities
      if (field === 'embedding_model' && typeof value === 'string' && modelsData?.models) {
        const modelConfig = modelsData.models[value];
        if (modelConfig?.supports_quantization === false) {
          // Model doesn't support quantization - clear the field
          updated.quantization = undefined;
        } else if (!prev.quantization) {
          // Model supports quantization but none selected - use default or recommended
          updated.quantization = modelConfig?.recommended_quantization || DEFAULT_QUANTIZATION;
        }
      }

      return updated;
    });
    // Clear error when field is modified
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  // Computed state for preview loading and disabled state
  const isPreviewLoading = gitPreviewMutation.isPending || imapPreviewMutation.isPending;
  const isDisabled = isSubmitting || createCollectionMutation.isPending || addSourceMutation.isPending;

  return (
    <div className="fixed inset-0 bg-black/50 dark:bg-black/80 flex items-center justify-center p-4 z-50">
      <div className="panel w-full max-w-lg md:max-w-2xl lg:max-w-4xl max-h-[90vh] overflow-y-auto relative rounded-2xl shadow-2xl" role="dialog" aria-modal="true" aria-labelledby="modal-title">
        {/* Loading overlay */}
        {isSubmitting && (
          <div className="absolute inset-0 bg-[var(--bg-secondary)]/95 backdrop-blur-sm flex items-center justify-center z-10 rounded-lg">
            <div className="text-center">
              <svg className="animate-spin h-8 w-8 text-red-400 mx-auto mb-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <p className="text-[var(--text-secondary)] font-medium">Creating collection...</p>
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
          <div className="px-6 py-5 border-b border-[var(--border)] backdrop-blur-md sticky top-0 z-10 bg-[var(--bg-secondary)]">
            <h3 id="modal-title" className="text-xl font-bold text-[var(--text-primary)] tracking-tight">Create New Collection</h3>
            <p className="mt-1 text-sm text-[var(--text-muted)]">
              Create a new collection to store and search your documents
            </p>
          </div>

          <div className="px-6 py-4 space-y-4">
            {/* Validation Summary */}
            {Object.keys(errors).length > 0 && !isSubmitting && (
              <div className="bg-red-500/10 border border-red-500/30 rounded-md p-4">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-red-300">
                      Please fix the following errors:
                    </h3>
                    <ul className="mt-2 text-sm text-red-400 list-disc list-inside">
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
              <label htmlFor="name" className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                Collection Name <span className="text-red-400">*</span>
              </label>
              <input
                type="text"
                id="name"
                value={formData.name}
                onChange={(e) => handleChange('name', e.target.value)}
                disabled={isSubmitting}
                className={`w-full px-4 py-2.5 input-field rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-gray-400/30 dark:focus:ring-white/30 transition-all duration-200 ${errors.name ? 'border-red-500/50 bg-red-500/10' : ''}`}
                placeholder="My Documents"
                autoFocus
              />
              {errors.name && (
                <p className="mt-1 text-sm text-red-500">{errors.name}</p>
              )}
            </div>

            {/* Description */}
            <div>
              <label htmlFor="description" className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                Description
              </label>
              <textarea
                id="description"
                value={formData.description || ''}
                onChange={(e) => handleChange('description', e.target.value)}
                disabled={isSubmitting}
                rows={3}
                className={`w-full px-4 py-2.5 input-field rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-gray-400/30 dark:focus:ring-white/30 transition-all duration-200 ${errors.description ? 'border-red-500/50 bg-red-500/10' : ''}`}
                placeholder="A collection of technical documentation..."
              />
              {errors.description && (
                <p className="mt-1 text-sm text-red-500">{errors.description}</p>
              )}
            </div>

            {/* Pipeline Template (Optional) */}
            <div className="border-t border-[var(--border)] pt-6 mt-6">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="h-4 w-4 text-amber-500" />
                <h4 className="text-sm font-bold text-[var(--text-primary)]">Pipeline Template (Optional)</h4>
              </div>
              <p className="text-sm text-[var(--text-muted)] mb-4">
                Select a pre-configured pipeline template optimized for your document type.
              </p>

              {templatesLoading ? (
                <div className="flex items-center py-2">
                  <Loader2 className="h-4 w-4 animate-spin text-[var(--text-muted)]" />
                  <span className="ml-2 text-sm text-[var(--text-muted)]">Loading templates...</span>
                </div>
              ) : templates && templates.length > 0 ? (
                <div className="space-y-3">
                  {/* Template Selector */}
                  <select
                    value={selectedTemplateId || ''}
                    onChange={(e) => setSelectedTemplateId(e.target.value || null)}
                    disabled={isSubmitting}
                    className="w-full px-4 py-2.5 input-field rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-gray-400/30 dark:focus:ring-white/30 transition-all duration-200"
                  >
                    <option value="">None (use defaults)</option>
                    {templates.map((template) => (
                      <option key={template.id} value={template.id}>
                        {template.name}
                      </option>
                    ))}
                  </select>

                  {/* Selected Template Info */}
                  {selectedTemplateId && (() => {
                    const selectedTemplate = templates.find(t => t.id === selectedTemplateId);
                    if (!selectedTemplate) return null;
                    return (
                      <div className="p-4 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-xl">
                        <p className="text-sm text-[var(--text-secondary)] mb-2">
                          {selectedTemplate.description}
                        </p>
                        {selectedTemplate.suggested_for.length > 0 && (
                          <div className="flex flex-wrap gap-2">
                            {selectedTemplate.suggested_for.map((tag) => (
                              <span
                                key={tag}
                                className="px-2 py-0.5 text-xs bg-amber-500/10 text-amber-500 border border-amber-500/20 rounded-full"
                              >
                                {tag}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </div>
              ) : (
                <p className="text-sm text-[var(--text-muted)]">
                  No pipeline templates available. Using default settings.
                </p>
              )}
            </div>

            {/* Initial Data Source (Optional) */}
            <div className="border-t border-[var(--border)] pt-6 mt-6">
              <h4 className="text-sm font-bold text-[var(--text-primary)] mb-2">Initial Data Source (Optional)</h4>
              <p className="text-sm text-[var(--text-muted)] mb-4">
                Optionally add an initial data source to start indexing immediately after collection creation.
              </p>

              {catalogLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin text-red-400" />
                  <span className="ml-2 text-[var(--text-muted)]">Loading connectors...</span>
                </div>
              ) : catalog ? (
                <>
                  {/* Connector Type Selector */}
                  <ConnectorTypeSelector
                    catalog={catalog}
                    selectedType={connectorType}
                    onSelect={handleTypeChange}
                    disabled={isDisabled}
                    showNoneOption={true}
                  />

                  {/* Dynamic Connector Form - only show when a connector type is selected */}
                  {connectorType !== 'none' && (
                    <ConnectorForm
                      catalog={catalog}
                      connectorType={connectorType}
                      values={configValues}
                      secrets={secrets}
                      onValuesChange={setConfigValues}
                      onSecretsChange={setSecrets}
                      errors={errors}
                      disabled={isDisabled}
                      onPreview={handlePreview}
                      previewResult={previewResult}
                      isPreviewLoading={isPreviewLoading}
                    />
                  )}
                </>
              ) : (
                <div className="text-center py-4 text-[var(--text-muted)]">
                  Failed to load connector catalog. You can add sources after creating the collection.
                </div>
              )}
            </div>

            {/* Embedding Model */}
            <div>
              <label htmlFor="embedding_model" className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                Embedding Model
              </label>
              <select
                id="embedding_model"
                value={formData.embedding_model}
                onChange={(e) => handleChange('embedding_model', e.target.value)}
                disabled={modelsLoading}
                className="w-full px-4 py-2.5 input-field rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-gray-400/30 dark:focus:ring-white/30 transition-all duration-200"
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
              <p className="mt-1 text-sm text-[var(--text-muted)]">
                Choose the AI model for converting documents to searchable vectors.
                {modelsData?.current_device && (
                  <span className="ml-1 text-[var(--text-secondary)]">
                    Running on {modelsData.current_device}.
                  </span>
                )}
                {' '}
                <a
                  href="/settings?tab=models"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] underline"
                >
                  Download additional models
                </a>
              </p>
            </div>

            {/* Quantization - only show for models that support it */}
            {formData.embedding_model && modelsData?.models && modelsData.models[formData.embedding_model]?.supports_quantization !== false && (
              <div>
                <label htmlFor="quantization" className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                  Model Quantization
                </label>
                <select
                  id="quantization"
                  value={formData.quantization}
                  onChange={(e) => handleChange('quantization', e.target.value)}
                  className="w-full px-4 py-2.5 input-field rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-gray-400/30 dark:focus:ring-white/30 transition-all duration-200"
                >
                  <option value="float32">float32 (Highest Precision)</option>
                  <option value="float16">float16 (Balanced - Default)</option>
                  <option value="int8">int8 (Lowest Memory Usage)</option>
                </select>
                <p className="mt-1 text-sm text-[var(--text-muted)]">
                  Choose the precision level for the embedding model. Lower precision uses less memory but may affect accuracy
                </p>
              </div>
            )}

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

            {/* Sync Configuration */}
            <div className="border-t border-[var(--border)] pt-6 mt-6">
              <h4 className="text-sm font-bold text-[var(--text-primary)] mb-3">Sync Configuration</h4>

              {/* Sync Mode Radio Buttons */}
              <div className="space-y-3">
                <div className="flex items-start">
                  <input
                    id="sync_mode_one_time"
                    name="sync_mode"
                    type="radio"
                    value="one_time"
                    checked={formData.sync_mode === 'one_time'}
                    onChange={() => handleChange('sync_mode', 'one_time' as SyncMode)}
                    disabled={isSubmitting}
                    className="h-4 w-4 bg-[var(--input-bg)] border-[var(--input-border)] text-gray-600 dark:text-white focus:ring-gray-400 dark:focus:ring-white mt-0.5"
                  />
                  <label htmlFor="sync_mode_one_time" className="ml-3">
                    <span className="block text-sm font-medium text-[var(--text-primary)]">One-time Import</span>
                    <span className="block text-sm text-[var(--text-muted)]">
                      Documents are imported once. Add sources manually when you want to update.
                    </span>
                  </label>
                </div>

                <div className="flex items-start">
                  <input
                    id="sync_mode_continuous"
                    name="sync_mode"
                    type="radio"
                    value="continuous"
                    checked={formData.sync_mode === 'continuous'}
                    onChange={() => handleChange('sync_mode', 'continuous' as SyncMode)}
                    disabled={isSubmitting}
                    className="h-4 w-4 bg-[var(--input-bg)] border-[var(--input-border)] text-gray-600 dark:text-white focus:ring-gray-400 dark:focus:ring-white mt-0.5"
                  />
                  <label htmlFor="sync_mode_continuous" className="ml-3">
                    <span className="block text-sm font-medium text-[var(--text-primary)]">Continuous Sync</span>
                    <span className="block text-sm text-[var(--text-muted)]">
                      Automatically check for new and updated documents at regular intervals.
                    </span>
                  </label>
                </div>
              </div>

              {/* Sync Interval (shown for continuous mode) */}
              {formData.sync_mode === 'continuous' && (
                <div className="mt-4 ml-7">
                  <label htmlFor="sync_interval_minutes" className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                    Sync Interval (minutes)
                  </label>
                  <div className="mt-1 flex items-center gap-3">
                    <input
                      type="number"
                      id="sync_interval_minutes"
                      min={15}
                      value={formData.sync_interval_minutes || 60}
                      onChange={(e) => handleChange('sync_interval_minutes', parseInt(e.target.value, 10) || 60)}
                      disabled={isSubmitting}
                      className="w-24 px-3 py-2 input-field rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-gray-400/30 dark:focus:ring-white/30 transition-all duration-200"
                    />
                    <span className="text-sm text-[var(--text-muted)]">Minimum: 15 minutes</span>
                  </div>
                  {errors.sync_interval_minutes && (
                    <p className="mt-1 text-sm text-red-500">{errors.sync_interval_minutes}</p>
                  )}
                  <p className="mt-1 text-sm text-[var(--text-muted)]">
                    How often to check sources for changes
                  </p>
                </div>
              )}
            </div>

            {/* Advanced Settings Accordion */}
            <div className="border-t border-[var(--border)] pt-6">
              <button
                type="button"
                onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
                className="flex items-center justify-between w-full text-left focus:outline-none focus:ring-2 focus:ring-gray-400/30 dark:focus:ring-white/30 focus:ring-offset-2 focus:ring-offset-[var(--bg-primary)] rounded-md p-2 -m-2 group"
              >
                <h4 className="text-sm font-bold text-[var(--text-secondary)] group-hover:text-[var(--text-primary)] transition-colors">Advanced Settings</h4>
                <svg
                  className={`h-5 w-5 text-[var(--text-muted)] transform transition-transform ${showAdvancedSettings ? 'rotate-180' : ''
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
                      className={`h-4 w-4 text-gray-600 dark:text-white focus:ring-gray-400 dark:focus:ring-white bg-[var(--input-bg)] border-[var(--input-border)] rounded ${isSubmitting ? 'cursor-not-allowed' : ''
                        }`}
                    />
                    <label htmlFor="is_public" className="ml-2 block text-sm text-[var(--text-primary)]">
                      Make this collection public
                    </label>
                  </div>

                  {/* Sparse Indexing Configuration */}
                  <div className="space-y-3">
                    <div className="flex items-center">
                      <input
                        id="sparse_enabled"
                        type="checkbox"
                        checked={sparseEnabled}
                        onChange={(e) => setSparseEnabled(e.target.checked)}
                        disabled={isSubmitting}
                        className={`h-4 w-4 text-gray-600 dark:text-white focus:ring-gray-400 dark:focus:ring-white bg-[var(--input-bg)] border-[var(--input-border)] rounded ${isSubmitting ? 'cursor-not-allowed' : ''
                          }`}
                      />
                      <label htmlFor="sparse_enabled" className="ml-2 block text-sm text-[var(--text-primary)]">
                        Enable Sparse Indexing (BM25/SPLADE for hybrid search)
                      </label>
                    </div>

                    {sparseEnabled && (
                      <div className="ml-6 space-y-3 p-4 bg-[var(--bg-tertiary)] border border-[var(--border)] rounded-xl">
                        {/* Plugin Selection */}
                        <div>
                          <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                            Sparse Indexer
                          </label>
                          <div className="grid grid-cols-2 gap-2">
                            <button
                              type="button"
                              onClick={() => setSparsePlugin('bm25-local')}
                              disabled={isSubmitting}
                              className={`p-3 text-sm rounded-xl border transition-all text-left ${sparsePlugin === 'bm25-local'
                                  ? 'border-gray-400 dark:border-white bg-gray-100 dark:bg-white/10 text-gray-800 dark:text-white'
                                  : 'border-[var(--border)] bg-[var(--bg-secondary)] text-[var(--text-muted)] hover:bg-[var(--bg-tertiary)] hover:border-[var(--border-strong)]'
                                } ${isSubmitting ? 'opacity-50 cursor-not-allowed' : ''}`}
                            >
                              <div className="font-bold uppercase tracking-wide text-xs">BM25</div>
                              <div className="text-xs text-[var(--text-muted)] mt-1">Statistical, CPU-based</div>
                            </button>
                            <button
                              type="button"
                              onClick={() => setSparsePlugin('splade-local')}
                              disabled={isSubmitting}
                              className={`p-3 text-sm rounded-xl border transition-all text-left ${sparsePlugin === 'splade-local'
                                  ? 'border-gray-400 dark:border-white bg-gray-100 dark:bg-white/10 text-gray-800 dark:text-white'
                                  : 'border-[var(--border)] bg-[var(--bg-secondary)] text-[var(--text-muted)] hover:bg-[var(--bg-tertiary)] hover:border-[var(--border-strong)]'
                                } ${isSubmitting ? 'opacity-50 cursor-not-allowed' : ''}`}
                            >
                              <div className="font-bold uppercase tracking-wide text-xs">SPLADE</div>
                              <div className="text-xs text-[var(--text-muted)] mt-1">Neural, requires GPU</div>
                            </button>
                          </div>
                        </div>

                        {/* BM25 Parameters */}
                        {sparsePlugin === 'bm25-local' && (
                          <div className="space-y-3 pt-3 border-t border-[var(--border)]">
                            <div className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">BM25 Parameters</div>
                            <div>
                              <div className="flex justify-between text-xs text-[var(--text-muted)] mb-1">
                                <span>k1 (term frequency saturation)</span>
                                <span className="font-mono text-red-400">{bm25Config.k1.toFixed(1)}</span>
                              </div>
                              <input
                                type="range"
                                min="0.5"
                                max="3.0"
                                step="0.1"
                                value={bm25Config.k1}
                                onChange={(e) => setBm25Config({ ...bm25Config, k1: parseFloat(e.target.value) })}
                                disabled={isSubmitting}
                                className="w-full h-2 bg-[var(--border)] rounded-lg appearance-none cursor-pointer accent-gray-600 dark:accent-white"
                              />
                            </div>
                            <div>
                              <div className="flex justify-between text-xs text-[var(--text-muted)] mb-1">
                                <span>b (document length normalization)</span>
                                <span className="font-mono text-red-400">{bm25Config.b.toFixed(2)}</span>
                              </div>
                              <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.05"
                                value={bm25Config.b}
                                onChange={(e) => setBm25Config({ ...bm25Config, b: parseFloat(e.target.value) })}
                                disabled={isSubmitting}
                                className="w-full h-2 bg-[var(--border)] rounded-lg appearance-none cursor-pointer accent-gray-600 dark:accent-white"
                              />
                            </div>
                          </div>
                        )}

                        {/* SPLADE Note */}
                        {sparsePlugin === 'splade-local' && (
                          <div className="text-xs text-amber-600 dark:text-amber-500 bg-amber-500/10 p-3 rounded-lg border border-amber-500/20">
                            SPLADE requires GPU and may take longer to index documents.
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="px-6 py-4 border-t border-[var(--border)] flex justify-end space-x-3 bg-[var(--bg-secondary)] backdrop-blur-md sticky bottom-0 z-10">
            <button
              type="button"
              onClick={onClose}
              disabled={isSubmitting || createCollectionMutation.isPending || addSourceMutation.isPending}
              className="px-4 py-2 text-sm font-medium text-[var(--text-secondary)] border border-[var(--border)] rounded-xl hover:bg-[var(--bg-tertiary)] hover:text-[var(--text-primary)] transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-400 dark:focus:ring-white focus:ring-offset-[var(--bg-primary)] disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting || createCollectionMutation.isPending || addSourceMutation.isPending}
              className="px-6 py-2 text-sm font-bold text-gray-900 bg-gray-200 dark:bg-white border border-transparent rounded-xl hover:bg-gray-300 dark:hover:bg-gray-100 shadow-lg focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-400 dark:focus:ring-white focus:ring-offset-[var(--bg-primary)] disabled:opacity-50 disabled:cursor-not-allowed transition-all transform active:scale-95"
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