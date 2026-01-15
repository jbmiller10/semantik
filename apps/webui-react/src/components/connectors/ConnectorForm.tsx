import { CheckCircle, XCircle, Loader2 } from 'lucide-react';
import type {
  ConnectorCatalog,
  GitPreviewResponse,
  ImapPreviewResponse,
} from '../../types/connector';
import { shouldShowField } from '../../types/connector';
import { DynamicField } from './DynamicField';

interface ConnectorFormProps {
  catalog: ConnectorCatalog;
  connectorType: string;
  values: Record<string, unknown>;
  secrets: Record<string, string>;
  onValuesChange: (values: Record<string, unknown>) => void;
  onSecretsChange: (secrets: Record<string, string>) => void;
  errors: Record<string, string>;
  disabled?: boolean;
  onPreview?: () => void;
  previewResult?: GitPreviewResponse | ImapPreviewResponse | null;
  isPreviewLoading?: boolean;
}

/**
 * Dynamic form for connector configuration
 * Renders fields and secrets based on the connector definition
 */
export function ConnectorForm({
  catalog,
  connectorType,
  values,
  secrets,
  onValuesChange,
  onSecretsChange,
  errors,
  disabled = false,
  onPreview,
  previewResult,
  isPreviewLoading = false,
}: ConnectorFormProps) {
  const definition = catalog[connectorType];

  if (!definition) {
    return (
      <div className="text-sm text-red-600">
        Unknown connector type: {connectorType}
      </div>
    );
  }

  const handleFieldChange = (name: string, value: unknown) => {
    onValuesChange({ ...values, [name]: value });
  };

  const handleSecretChange = (name: string, value: string) => {
    onSecretsChange({ ...secrets, [name]: value });
  };

  // Get visible fields and secrets based on show_when conditions
  const visibleFields = definition.fields.filter((field) =>
    shouldShowField(field, values)
  );

  const visibleSecrets = definition.secrets.filter((secret) =>
    shouldShowField(secret, values)
  );

  const hasPreviewEndpoint = !!definition.preview_endpoint;
  const showPreviewButton = hasPreviewEndpoint && connectorType !== 'directory';

  return (
    <div className="space-y-4 mt-4 border-t border-white/10 pt-4">
      {/* Configuration Fields */}
      {visibleFields.map((field) => (
        <DynamicField
          key={field.name}
          field={field}
          value={values[field.name]}
          onChange={(value) => handleFieldChange(field.name, value)}
          error={errors[field.name]}
          disabled={disabled}
        />
      ))}

      {/* Secret Fields */}
      {visibleSecrets.length > 0 && (
        <div className="border-t border-white/10 pt-4 mt-4">
          <h4 className="text-sm font-medium text-gray-300 mb-3">
            Authentication
          </h4>
          {visibleSecrets.map((secret) => (
            <DynamicField
              key={secret.name}
              field={secret}
              value={secrets[secret.name]}
              onChange={(value) => handleSecretChange(secret.name, value as string)}
              error={errors[secret.name]}
              disabled={disabled}
              isSecret
            />
          ))}
        </div>
      )}

      {/* Preview/Test Connection Button */}
      {showPreviewButton && onPreview && (
        <div className="border-t border-white/10 pt-4 mt-4">
          <button
            type="button"
            onClick={onPreview}
            disabled={disabled || isPreviewLoading}
            className="inline-flex items-center px-4 py-2 border border-void-600 rounded-md shadow-sm text-sm font-medium text-gray-300 bg-void-800 hover:bg-void-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-signal-500 focus:ring-offset-void-900 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isPreviewLoading ? (
              <>
                <Loader2 className="animate-spin -ml-1 mr-2 h-4 w-4" />
                Testing...
              </>
            ) : (
              'Test Connection'
            )}
          </button>
        </div>
      )}

      {/* Preview Result */}
      {previewResult && (
        <PreviewResultDisplay
          result={previewResult}
          connectorType={connectorType}
        />
      )}
    </div>
  );
}

interface PreviewResultDisplayProps {
  result: GitPreviewResponse | ImapPreviewResponse;
  connectorType: string;
}

/**
 * Displays the result of a connector preview/validation
 */
function PreviewResultDisplay({
  result,
  connectorType,
}: PreviewResultDisplayProps) {
  if (result.valid) {
    return (
      <div className="rounded-lg bg-green-500/10 border border-green-500/30 p-4">
        <div className="flex">
          <CheckCircle className="h-5 w-5 text-green-400 flex-shrink-0" />
          <div className="ml-3">
            <h3 className="text-sm font-medium text-green-300">
              Connection successful!
            </h3>
            {connectorType === 'git' && 'refs_found' in result && (
              <div className="mt-2 text-sm text-green-400">
                <p className="font-medium">Available branches/tags:</p>
                <ul className="mt-1 list-disc list-inside">
                  {(result as GitPreviewResponse).refs_found
                    .slice(0, 5)
                    .map((ref) => (
                      <li key={ref}>{ref}</li>
                    ))}
                  {(result as GitPreviewResponse).refs_found.length > 5 && (
                    <li className="text-green-500">
                      +{(result as GitPreviewResponse).refs_found.length - 5} more
                    </li>
                  )}
                </ul>
              </div>
            )}
            {connectorType === 'imap' && 'mailboxes_found' in result && (
              <div className="mt-2 text-sm text-green-400">
                <p className="font-medium">Available mailboxes:</p>
                <ul className="mt-1 list-disc list-inside">
                  {(result as ImapPreviewResponse).mailboxes_found
                    .slice(0, 5)
                    .map((mailbox) => (
                      <li key={mailbox}>{mailbox}</li>
                    ))}
                  {(result as ImapPreviewResponse).mailboxes_found.length > 5 && (
                    <li className="text-green-500">
                      +{(result as ImapPreviewResponse).mailboxes_found.length - 5}{' '}
                      more
                    </li>
                  )}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg bg-red-500/10 border border-red-500/30 p-4">
      <div className="flex">
        <XCircle className="h-5 w-5 text-red-400 flex-shrink-0" />
        <div className="ml-3">
          <h3 className="text-sm font-medium text-red-300">Connection failed</h3>
          {result.error && (
            <p className="mt-1 text-sm text-red-400">{result.error}</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default ConnectorForm;
