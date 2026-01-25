/**
 * Editor for pipeline node configuration.
 * Shows plugin selector and config fields based on JSON schema.
 */

import { useCallback } from 'react';
import { usePlugins, usePluginConfigSchema } from '@/hooks/usePlugins';
import { nodeTypeToPluginType, NODE_TYPE_LABELS } from '@/utils/pipelinePluginMapping';
import type { PipelineNode } from '@/types/pipeline';
import type { JsonSchemaProperty, PluginConfigSchema } from '@/types/plugin';
import { Settings, Loader2 } from 'lucide-react';

interface NodeConfigEditorProps {
  node: PipelineNode;
  onChange: (node: PipelineNode) => void;
  readOnly?: boolean;
}

/**
 * Render a single config field based on its JSON Schema definition.
 */
function ConfigField({
  name,
  property,
  value,
  onChange,
  disabled,
}: {
  name: string;
  property: JsonSchemaProperty;
  value: unknown;
  onChange: (value: unknown) => void;
  disabled?: boolean;
}) {
  const label = property.title || name;
  const description = property.description;

  // Boolean field
  if (property.type === 'boolean') {
    return (
      <div className="flex items-start gap-3">
        <input
          type="checkbox"
          id={name}
          checked={value === true}
          onChange={(e) => onChange(e.target.checked)}
          disabled={disabled}
          className="mt-1 h-4 w-4 rounded border-[var(--border)] bg-[var(--bg-tertiary)] text-[var(--text-primary)]"
        />
        <div>
          <label htmlFor={name} className="text-sm font-medium text-[var(--text-primary)]">
            {label}
          </label>
          {description && (
            <p className="text-xs text-[var(--text-muted)]">{description}</p>
          )}
        </div>
      </div>
    );
  }

  // Enum/select field
  if (property.enum && property.enum.length > 0) {
    return (
      <div>
        <label htmlFor={name} className="block text-sm font-medium text-[var(--text-primary)] mb-1">
          {label}
        </label>
        {description && (
          <p className="text-xs text-[var(--text-muted)] mb-2">{description}</p>
        )}
        <select
          id={name}
          value={(value as string) ?? property.default ?? ''}
          onChange={(e) => onChange(e.target.value || undefined)}
          disabled={disabled}
          className="input-field w-full"
        >
          <option value="">Select...</option>
          {property.enum.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      </div>
    );
  }

  // Number field
  if (property.type === 'number' || property.type === 'integer') {
    return (
      <div>
        <label htmlFor={name} className="block text-sm font-medium text-[var(--text-primary)] mb-1">
          {label}
        </label>
        {description && (
          <p className="text-xs text-[var(--text-muted)] mb-2">{description}</p>
        )}
        <input
          type="number"
          id={name}
          value={(value as number) ?? property.default ?? ''}
          onChange={(e) => onChange(e.target.value ? Number(e.target.value) : undefined)}
          min={property.minimum}
          max={property.maximum}
          step={property.type === 'integer' ? 1 : 'any'}
          disabled={disabled}
          className="input-field w-full"
        />
      </div>
    );
  }

  // Default: string field
  return (
    <div>
      <label htmlFor={name} className="block text-sm font-medium text-[var(--text-primary)] mb-1">
        {label}
      </label>
      {description && (
        <p className="text-xs text-[var(--text-muted)] mb-2">{description}</p>
      )}
      <input
        type="text"
        id={name}
        value={(value as string) ?? property.default ?? ''}
        onChange={(e) => onChange(e.target.value || undefined)}
        disabled={disabled}
        className="input-field w-full"
      />
    </div>
  );
}

export function NodeConfigEditor({
  node,
  onChange,
  readOnly = false,
}: NodeConfigEditorProps) {
  const pluginType = nodeTypeToPluginType(node.type);
  const nodeTypeLabel = NODE_TYPE_LABELS[node.type];

  // Fetch available plugins for this node type
  const { data: plugins, isLoading: pluginsLoading } = usePlugins({ type: pluginType, enabled: true });

  // Fetch config schema for the current plugin
  const { data: schema, isLoading: schemaLoading } = usePluginConfigSchema(node.plugin_id);

  // Handle plugin change
  const handlePluginChange = useCallback(
    (pluginId: string) => {
      onChange({
        ...node,
        plugin_id: pluginId,
        config: {}, // Reset config when plugin changes
      });
    },
    [node, onChange]
  );

  // Handle config field change
  const handleConfigChange = useCallback(
    (fieldName: string, value: unknown) => {
      onChange({
        ...node,
        config: {
          ...node.config,
          [fieldName]: value,
        },
      });
    },
    [node, onChange]
  );

  // Loading state
  if (pluginsLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="w-6 h-6 animate-spin text-[var(--text-muted)]" />
        <span className="ml-2 text-[var(--text-muted)]">Loading plugins...</span>
      </div>
    );
  }

  const availablePlugins = plugins || [];
  const schemaProperties = (schema as PluginConfigSchema | null)?.properties || {};

  return (
    <div className="p-4 space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Settings className="w-5 h-5 text-[var(--text-muted)]" />
        <h3 className="text-lg font-semibold text-[var(--text-primary)]">
          {nodeTypeLabel}
        </h3>
      </div>

      {/* Plugin selector */}
      <div>
        <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">
          Plugin
        </label>
        <select
          value={node.plugin_id}
          onChange={(e) => handlePluginChange(e.target.value)}
          disabled={readOnly}
          className="input-field w-full"
        >
          {availablePlugins.map((plugin) => (
            <option key={plugin.id} value={plugin.id}>
              {plugin.manifest.display_name}
            </option>
          ))}
        </select>
        {availablePlugins.find((p) => p.id === node.plugin_id)?.manifest.description && (
          <p className="text-xs text-[var(--text-muted)] mt-1">
            {availablePlugins.find((p) => p.id === node.plugin_id)?.manifest.description}
          </p>
        )}
      </div>

      {/* Config fields */}
      {Object.keys(schemaProperties).length > 0 && (
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-[var(--text-secondary)] border-t border-[var(--border)] pt-4">
            Configuration
          </h4>

          {schemaLoading ? (
            <div className="flex items-center gap-2 text-[var(--text-muted)]">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-sm">Loading config...</span>
            </div>
          ) : (
            <div className="space-y-4">
              {Object.entries(schemaProperties).map(([name, property]) => (
                <ConfigField
                  key={name}
                  name={name}
                  property={property}
                  value={node.config[name]}
                  onChange={(value) => handleConfigChange(name, value)}
                  disabled={readOnly}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* No config message */}
      {!schemaLoading && Object.keys(schemaProperties).length === 0 && (
        <p className="text-sm text-[var(--text-muted)] italic">
          This plugin has no configuration options.
        </p>
      )}
    </div>
  );
}

export default NodeConfigEditor;
