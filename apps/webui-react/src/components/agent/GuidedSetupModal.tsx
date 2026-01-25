/**
 * Modal for starting guided pipeline setup.
 * User can either:
 * 1. Configure a new data source (inline) - source created when pipeline is applied
 * 2. Use an existing source ID (legacy mode)
 */

import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useCreateConversation } from '../../hooks/useAgentConversation';
import { useConnectorCatalog, useGitPreview, useImapPreview } from '../../hooks/useConnectors';
import { ConnectorTypeSelector } from '../connectors/ConnectorTypeSelector';
import { ConnectorForm } from '../connectors/ConnectorForm';
import type { GitPreviewResponse, ImapPreviewResponse } from '../../types/connector';
import type { InlineSourceConfig } from '../../types/agent';

type SetupMode = 'new-source' | 'existing-source';

interface GuidedSetupModalProps {
  onClose: () => void;
}

export function GuidedSetupModal({ onClose }: GuidedSetupModalProps) {
  const navigate = useNavigate();
  const createConversation = useCreateConversation();
  const { data: catalog, isLoading: catalogLoading } = useConnectorCatalog();

  // Mode selection
  const [mode, setMode] = useState<SetupMode>('new-source');

  // New source mode state
  const [connectorType, setConnectorType] = useState<string>('directory');
  const [configValues, setConfigValues] = useState<Record<string, unknown>>({});
  const [secrets, setSecrets] = useState<Record<string, string>>({});
  const [formErrors, setFormErrors] = useState<Record<string, string>>({});
  const [previewResult, setPreviewResult] = useState<GitPreviewResponse | ImapPreviewResponse | null>(null);

  // Existing source mode state
  const [sourceIdInput, setSourceIdInput] = useState<string>('');

  // Preview mutations
  const gitPreviewMutation = useGitPreview();
  const imapPreviewMutation = useImapPreview();

  // Handle connector type change
  const handleConnectorTypeChange = useCallback((type: string) => {
    setConnectorType(type);
    setConfigValues({});
    setSecrets({});
    setFormErrors({});
    setPreviewResult(null);
  }, []);

  // Get source path for display
  const getSourcePath = useCallback((): string => {
    switch (connectorType) {
      case 'directory':
        return (configValues.path as string) || '';
      case 'git':
        return (configValues.repo_url as string) || (configValues.repository_url as string) || '';
      case 'imap':
        return `${configValues.username || ''}@${configValues.host || ''}`;
      default:
        return '';
    }
  }, [connectorType, configValues]);

  // Validate form
  const validateForm = useCallback((): boolean => {
    const errors: Record<string, string> = {};

    if (mode === 'new-source') {
      // Validate based on connector type
      if (!catalog || !catalog[connectorType]) {
        return false;
      }

      const definition = catalog[connectorType];

      // Check required fields
      for (const field of definition.fields) {
        if (field.required && !configValues[field.name]) {
          errors[field.name] = `${field.label || field.name} is required`;
        }
      }

      // Check required secrets
      for (const secret of definition.secrets) {
        if (secret.required && !secrets[secret.name]) {
          errors[secret.name] = `${secret.label || secret.name} is required`;
        }
      }
    } else {
      // Existing source mode
      if (!sourceIdInput.trim()) {
        errors.source_id = 'Source ID is required';
      } else if (!/^\d+$/.test(sourceIdInput.trim())) {
        errors.source_id = 'Source ID must be a number';
      }
    }

    setFormErrors(errors);
    return Object.keys(errors).length === 0;
  }, [mode, catalog, connectorType, configValues, secrets, sourceIdInput]);

  // Handle preview/test connection
  const handlePreview = useCallback(() => {
    if (!catalog) return;

    setPreviewResult(null);

    if (connectorType === 'git') {
      gitPreviewMutation.mutate(
        {
          repo_url: configValues.repo_url as string || configValues.repository_url as string,
          ref: configValues.branch as string || configValues.ref as string || undefined,
          token: secrets.password || secrets.token || undefined,
          ssh_key: secrets.ssh_key || undefined,
          ssh_passphrase: secrets.ssh_passphrase || undefined,
        },
        {
          onSuccess: (result) => setPreviewResult(result),
        }
      );
    } else if (connectorType === 'imap') {
      imapPreviewMutation.mutate(
        {
          host: configValues.host as string,
          port: configValues.port as number || 993,
          username: configValues.username as string,
          password: secrets.password,
          use_ssl: configValues.use_ssl as boolean ?? true,
        },
        {
          onSuccess: (result) => setPreviewResult(result),
        }
      );
    }
  }, [catalog, connectorType, configValues, secrets, gitPreviewMutation, imapPreviewMutation]);

  // Handle start conversation
  const handleStart = async () => {
    if (!validateForm()) return;

    if (mode === 'new-source') {
      // Create inline source config
      const inlineSource: InlineSourceConfig = {
        source_type: connectorType,
        source_config: configValues as Record<string, unknown>,
      };

      // Filter out empty secrets
      const filteredSecrets = Object.fromEntries(
        Object.entries(secrets).filter(([, value]) => value.trim().length > 0)
      );

      createConversation.mutate(
        {
          inline_source: inlineSource,
          secrets: Object.keys(filteredSecrets).length > 0 ? filteredSecrets : undefined,
        },
        {
          onSuccess: (data) => {
            onClose();
            navigate(`/agent/${data.id}`);
          },
        }
      );
    } else {
      // Existing source mode
      const sourceId = parseInt(sourceIdInput.trim(), 10);
      createConversation.mutate(
        { source_id: sourceId },
        {
          onSuccess: (data) => {
            onClose();
            navigate(`/agent/${data.id}`);
          },
        }
      );
    }
  };

  const isFormValid = mode === 'new-source'
    ? getSourcePath().trim().length > 0
    : sourceIdInput.trim().length > 0 && /^\d+$/.test(sourceIdInput.trim());

  const isPreviewLoading = gitPreviewMutation.isPending || imapPreviewMutation.isPending;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative bg-[var(--bg-primary)] rounded-xl shadow-xl w-full max-w-2xl mx-4 max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-[var(--border)] flex-shrink-0">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-[var(--text-primary)]">
                Guided Pipeline Setup
              </h2>
              <p className="text-sm text-[var(--text-secondary)] mt-1">
                Chat with an AI assistant to configure your pipeline
              </p>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)] transition-colors"
              aria-label="Close"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto flex-1">
          {/* Feature explanation */}
          <div className="mb-6 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
            <div className="flex gap-3">
              <svg
                className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <div>
                <p className="text-sm text-blue-400 font-medium mb-1">
                  How it works
                </p>
                <p className="text-sm text-blue-300/80">
                  The assistant will analyze your source documents and help you choose
                  the best embedding model, chunking strategy, and other settings through
                  a natural conversation.
                </p>
              </div>
            </div>
          </div>

          {/* Mode Selection */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-[var(--text-muted)] mb-3">
              Choose how to configure your source
            </label>
            <div className="grid grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => setMode('new-source')}
                className={`
                  p-4 rounded-lg border-2 transition-all text-left
                  ${mode === 'new-source'
                    ? 'border-gray-400 dark:border-white bg-gray-100 dark:bg-white/10'
                    : 'border-[var(--border)] hover:border-[var(--border-strong)] hover:bg-[var(--bg-tertiary)]'
                  }
                `}
              >
                <div className="flex items-center gap-2 mb-1">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  <span className={`font-medium ${mode === 'new-source' ? 'text-gray-800 dark:text-white' : 'text-[var(--text-primary)]'}`}>
                    New Source
                  </span>
                </div>
                <p className={`text-xs ${mode === 'new-source' ? 'text-gray-600 dark:text-gray-300' : 'text-[var(--text-muted)]'}`}>
                  Configure a new data source for your documents
                </p>
              </button>

              <button
                type="button"
                onClick={() => setMode('existing-source')}
                className={`
                  p-4 rounded-lg border-2 transition-all text-left
                  ${mode === 'existing-source'
                    ? 'border-gray-400 dark:border-white bg-gray-100 dark:bg-white/10'
                    : 'border-[var(--border)] hover:border-[var(--border-strong)] hover:bg-[var(--bg-tertiary)]'
                  }
                `}
              >
                <div className="flex items-center gap-2 mb-1">
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  <span className={`font-medium ${mode === 'existing-source' ? 'text-gray-800 dark:text-white' : 'text-[var(--text-primary)]'}`}>
                    Existing Source
                  </span>
                </div>
                <p className={`text-xs ${mode === 'existing-source' ? 'text-gray-600 dark:text-gray-300' : 'text-[var(--text-muted)]'}`}>
                  Use a source you've already added to a collection
                </p>
              </button>
            </div>
          </div>

          {/* New Source Mode */}
          {mode === 'new-source' && (
            <div>
              {catalogLoading ? (
                <div className="flex items-center justify-center py-8">
                  <svg className="w-6 h-6 animate-spin text-[var(--text-muted)]" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                </div>
              ) : catalog ? (
                <>
                  <ConnectorTypeSelector
                    catalog={catalog}
                    selectedType={connectorType}
                    onSelect={handleConnectorTypeChange}
                  />

                  <ConnectorForm
                    catalog={catalog}
                    connectorType={connectorType}
                    values={configValues}
                    secrets={secrets}
                    onValuesChange={setConfigValues}
                    onSecretsChange={setSecrets}
                    errors={formErrors}
                    onPreview={handlePreview}
                    previewResult={previewResult}
                    isPreviewLoading={isPreviewLoading}
                  />
                </>
              ) : (
                <div className="text-center py-8 text-[var(--text-muted)]">
                  Failed to load connectors
                </div>
              )}
            </div>
          )}

          {/* Existing Source Mode */}
          {mode === 'existing-source' && (
            <div>
              <label
                htmlFor="source-id"
                className="block text-sm font-medium text-[var(--text-primary)] mb-2"
              >
                Source ID
              </label>
              <p className="text-xs text-[var(--text-muted)] mb-3">
                Enter the ID of a source you've already added to a collection.
                The assistant will help you optimize its pipeline settings.
              </p>
              <input
                type="number"
                id="source-id"
                value={sourceIdInput}
                onChange={(e) => {
                  setSourceIdInput(e.target.value);
                  setFormErrors({});
                }}
                placeholder="Enter source ID..."
                className="input-field w-full"
                min={1}
              />
              {formErrors.source_id && (
                <p className="mt-1 text-sm text-red-400">{formErrors.source_id}</p>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-[var(--border)] flex justify-end gap-3 flex-shrink-0">
          <button
            onClick={onClose}
            className="btn-secondary"
          >
            Cancel
          </button>
          <button
            onClick={handleStart}
            disabled={!isFormValid || createConversation.isPending}
            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {createConversation.isPending ? (
              <>
                <svg
                  className="w-4 h-4 animate-spin mr-2"
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
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
                Starting...
              </>
            ) : (
              <>
                <svg
                  className="w-4 h-4 mr-2"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                  />
                </svg>
                Start Conversation
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

export default GuidedSetupModal;
