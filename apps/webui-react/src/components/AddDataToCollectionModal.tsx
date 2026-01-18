import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Loader2 } from 'lucide-react';
import { useAddSource } from '../hooks/useCollectionOperations';
import { useConnectorCatalog, useGitPreview, useImapPreview } from '../hooks/useConnectors';
import { useUIStore } from '../stores/uiStore';
import { ConnectorTypeSelector, ConnectorForm } from './connectors';
import { shouldShowField } from '../types/connector';
import type { Collection } from '../types/collection';
import type {
  GitPreviewResponse,
  ImapPreviewResponse,
} from '../types/connector';

interface AddDataToCollectionModalProps {
  collection: Collection;
  onClose: () => void;
  onSuccess: () => void;
}

function AddDataToCollectionModal({
  collection,
  onClose,
  onSuccess,
}: AddDataToCollectionModalProps) {
  const addSourceMutation = useAddSource();
  const { addToast } = useUIStore();
  const navigate = useNavigate();

  // Connector state
  const [connectorType, setConnectorType] = useState<string>('directory');
  const [configValues, setConfigValues] = useState<Record<string, unknown>>({});
  const [secrets, setSecrets] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [previewResult, setPreviewResult] = useState<
    GitPreviewResponse | ImapPreviewResponse | null
  >(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Fetch connector catalog
  const { data: catalog, isLoading: catalogLoading } = useConnectorCatalog();

  // Preview mutations
  const gitPreviewMutation = useGitPreview();
  const imapPreviewMutation = useImapPreview();

  // Initialize default values when connector type changes
  useEffect(() => {
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
    setErrors({});
    setPreviewResult(null);
  }, [connectorType, catalog]);

  // Handle connector type change
  const handleTypeChange = useCallback((type: string) => {
    setConnectorType(type);
  }, []);

  // Validate form based on connector definition
  const validateForm = useCallback((): boolean => {
    if (!catalog) return false;

    const definition = catalog[connectorType];
    if (!definition) return false;

    const newErrors: Record<string, string> = {};

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

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [catalog, connectorType, configValues, secrets]);

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
          // Default to INBOX if available, otherwise first mailbox
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
  }, [
    connectorType,
    configValues,
    secrets,
    gitPreviewMutation,
    imapPreviewMutation,
    addToast,
  ]);

  // Build source path for display/backward compat
  const getSourcePath = useCallback((): string => {
    switch (connectorType) {
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

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      addToast({ type: 'error', message: 'Please fill in all required fields' });
      return;
    }

    setIsSubmitting(true);

    try {
      const sourcePath = getSourcePath();

      await addSourceMutation.mutateAsync({
        collectionId: collection.id,
        sourceType: connectorType,
        sourceConfig: configValues,
        secrets: Object.keys(secrets).length > 0 ? secrets : undefined,
        sourcePath,
        config: {
          chunk_size: collection.chunk_size,
          chunk_overlap: collection.chunk_overlap,
        },
      });

      // Navigate to collection detail page to show operation progress
      navigate(`/collections/${collection.id}`);
      onSuccess();
    } catch (error) {
      if (!addSourceMutation.isError) {
        addToast({
          message: error instanceof Error ? error.message : 'Failed to add data source',
          type: 'error',
        });
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const isPreviewLoading = gitPreviewMutation.isPending || imapPreviewMutation.isPending;
  const isDisabled = isSubmitting || addSourceMutation.isPending;

  return (
    <>
      <div className="fixed inset-0 bg-black/50 dark:bg-black/80 z-[60]" onClick={onClose} />
      <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 glass-panel rounded-lg shadow-xl z-[60] w-full max-w-xl max-h-[90vh] overflow-y-auto">
        <div className="px-6 py-4 border-b border-[var(--border)] sticky top-0 bg-[var(--bg-primary)]/95 backdrop-blur-sm">
          <h3 className="text-lg font-medium text-gray-100">Add Data to Collection</h3>
          <p className="mt-1 text-sm text-gray-400">
            Add new documents to "{collection.name}"
          </p>
        </div>

        {catalogLoading ? (
          <div className="px-6 py-12 flex items-center justify-center">
            <Loader2 className="h-8 w-8 animate-spin text-[var(--text-muted)]" />
            <span className="ml-2 text-gray-400">Loading connectors...</span>
          </div>
        ) : catalog ? (
          <form onSubmit={handleSubmit}>
            <div className="px-6 py-4 space-y-4">
              {/* Connector Type Selector */}
              <ConnectorTypeSelector
                catalog={catalog}
                selectedType={connectorType}
                onSelect={handleTypeChange}
                disabled={isDisabled}
              />

              {/* Dynamic Connector Form */}
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

              {/* Settings Summary */}
              <div className="bg-[var(--bg-secondary)] rounded-lg p-4 border border-[var(--border)] mt-4">
                <h4 className="text-sm font-medium text-gray-100 mb-2">Collection Settings</h4>
                <dl className="text-xs space-y-1">
                  <div className="flex justify-between">
                    <dt className="text-gray-400">Embedding Model:</dt>
                    <dd className="text-gray-200 font-mono text-xs">{collection.embedding_model}</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-gray-400">Chunk Size:</dt>
                    <dd className="text-gray-200">{collection.chunk_size} characters</dd>
                  </div>
                  <div className="flex justify-between">
                    <dt className="text-gray-400">Chunk Overlap:</dt>
                    <dd className="text-gray-200">{collection.chunk_overlap} characters</dd>
                  </div>
                </dl>
              </div>

              {/* Info Banner */}
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                <div className="flex">
                  <div className="flex-shrink-0">
                    <svg className="h-5 w-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <div className="ml-3">
                    <p className="text-sm text-blue-300">
                      Duplicate content will be automatically skipped. Only new or modified items will be processed.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="px-6 py-4 border-t border-[var(--border)] flex justify-end space-x-3 sticky bottom-0 bg-[var(--bg-primary)]/95 backdrop-blur-sm">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 border border-[var(--border)] rounded-md text-sm font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-secondary)] transition-colors"
                disabled={isDisabled}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-4 py-2 bg-gray-200 dark:bg-white text-gray-900 dark:text-gray-900 rounded-md text-sm font-medium hover:bg-gray-300 dark:hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                disabled={isDisabled}
              >
                {isDisabled ? 'Adding Source...' : 'Add Source'}
              </button>
            </div>
          </form>
        ) : (
          <div className="px-6 py-12 text-center text-gray-400">
            Failed to load connector catalog
          </div>
        )}
      </div>
    </>
  );
}

export default AddDataToCollectionModal;
