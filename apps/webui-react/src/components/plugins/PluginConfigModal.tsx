import { useState, useEffect } from 'react';
import { usePluginConfigSchema, useUpdatePluginConfig } from '../../hooks/usePlugins';
import type { PluginInfo, PluginConfigSchema } from '../../types/plugin';
import PluginConfigForm from './PluginConfigForm';

interface PluginConfigModalProps {
  plugin: PluginInfo;
  onClose: () => void;
}

/**
 * Validate configuration values against schema
 */
function validateConfig(
  schema: PluginConfigSchema,
  values: Record<string, unknown>
): Record<string, string> {
  const errors: Record<string, string> = {};
  const properties = schema.properties || {};
  const required = schema.required || [];

  // Check required fields
  for (const fieldName of required) {
    const value = values[fieldName];
    if (value === undefined || value === null || value === '') {
      errors[fieldName] = 'This field is required';
    }
  }

  // Check property-specific validations
  for (const [name, property] of Object.entries(properties)) {
    const value = values[name];
    if (value === undefined || value === null || value === '') {
      continue; // Skip empty non-required fields
    }

    // String validations
    if (property.type === 'string' && typeof value === 'string') {
      if (property.minLength && value.length < property.minLength) {
        errors[name] = `Must be at least ${property.minLength} characters`;
      }
      if (property.maxLength && value.length > property.maxLength) {
        errors[name] = `Must be at most ${property.maxLength} characters`;
      }
      if (property.pattern) {
        const regex = new RegExp(property.pattern);
        if (!regex.test(value)) {
          errors[name] = 'Invalid format';
        }
      }
      if (property.enum && !property.enum.includes(value)) {
        errors[name] = `Must be one of: ${property.enum.join(', ')}`;
      }
    }

    // Number validations
    if ((property.type === 'number' || property.type === 'integer') && typeof value === 'number') {
      if (property.minimum !== undefined && value < property.minimum) {
        errors[name] = `Must be at least ${property.minimum}`;
      }
      if (property.maximum !== undefined && value > property.maximum) {
        errors[name] = `Must be at most ${property.maximum}`;
      }
      if (property.exclusiveMinimum !== undefined && value <= property.exclusiveMinimum) {
        errors[name] = `Must be greater than ${property.exclusiveMinimum}`;
      }
      if (property.exclusiveMaximum !== undefined && value >= property.exclusiveMaximum) {
        errors[name] = `Must be less than ${property.exclusiveMaximum}`;
      }
      if (property.type === 'integer' && !Number.isInteger(value)) {
        errors[name] = 'Must be an integer';
      }
    }

    // Array validations
    if (property.type === 'array' && Array.isArray(value)) {
      if (property.minItems !== undefined && value.length < property.minItems) {
        errors[name] = `Must have at least ${property.minItems} items`;
      }
      if (property.maxItems !== undefined && value.length > property.maxItems) {
        errors[name] = `Must have at most ${property.maxItems} items`;
      }
      if (property.uniqueItems && new Set(value).size !== value.length) {
        errors[name] = 'All items must be unique';
      }
    }
  }

  return errors;
}

function PluginConfigModal({ plugin, onClose }: PluginConfigModalProps) {
  const [values, setValues] = useState<Record<string, unknown>>(plugin.config || {});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [saveError, setSaveError] = useState<string | null>(null);

  // Fetch the config schema
  const {
    data: schema,
    isLoading: isLoadingSchema,
    error: schemaError,
  } = usePluginConfigSchema(plugin.id);

  // Update config mutation
  const updateConfig = useUpdatePluginConfig();

  // Reset values when plugin changes
  useEffect(() => {
    setValues(plugin.config || {});
    setErrors({});
    setSaveError(null);
  }, [plugin.id, plugin.config]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaveError(null);

    // Validate if we have a schema
    if (schema) {
      const validationErrors = validateConfig(schema, values);
      if (Object.keys(validationErrors).length > 0) {
        setErrors(validationErrors);
        return;
      }
    }

    try {
      await updateConfig.mutateAsync({
        pluginId: plugin.id,
        config: values,
      });
      onClose();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Failed to save configuration');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    }
  };

  return (
    <div
      className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full z-50 flex items-center justify-center"
      onKeyDown={handleKeyDown}
    >
      <div className="relative mx-auto w-full max-w-lg shadow-lg rounded-lg bg-white">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-medium text-gray-900">
                Configure {plugin.manifest.display_name}
              </h3>
              <p className="mt-1 text-sm text-gray-500">v{plugin.version}</p>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-500 focus:outline-none"
            >
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
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
        <form onSubmit={handleSubmit}>
          <div className="px-6 py-4 max-h-[60vh] overflow-y-auto">
            {/* Loading state */}
            {isLoadingSchema && (
              <div className="flex items-center justify-center py-8">
                <svg
                  className="animate-spin h-6 w-6 text-gray-400"
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
                <span className="ml-2 text-gray-500">Loading configuration...</span>
              </div>
            )}

            {/* Schema error */}
            {schemaError && (
              <div className="bg-red-50 border border-red-200 rounded-md p-4">
                <div className="flex">
                  <svg
                    className="h-5 w-5 text-red-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  <div className="ml-3">
                    <p className="text-sm text-red-700">Failed to load configuration schema</p>
                  </div>
                </div>
              </div>
            )}

            {/* No schema (plugin has no config) */}
            {!isLoadingSchema && !schemaError && schema === null && (
              <div className="text-center py-8 text-gray-500">
                <svg
                  className="mx-auto h-8 w-8 text-gray-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <p className="mt-2 text-sm">This plugin has no configuration options.</p>
              </div>
            )}

            {/* Config form */}
            {!isLoadingSchema && !schemaError && schema && (
              <PluginConfigForm
                schema={schema}
                values={values}
                onChange={setValues}
                errors={errors}
                disabled={updateConfig.isPending}
              />
            )}

            {/* Save error */}
            {saveError && (
              <div className="mt-4 bg-red-50 border border-red-200 rounded-md p-3">
                <p className="text-sm text-red-700">{saveError}</p>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="px-6 py-4 bg-gray-50 border-t border-gray-200 flex items-center justify-between rounded-b-lg">
            <div className="text-xs text-yellow-600 flex items-center">
              <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
              Changes require service restart
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={updateConfig.isPending || isLoadingSchema || !!schemaError}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {updateConfig.isPending ? (
                  <span className="flex items-center">
                    <svg
                      className="animate-spin h-4 w-4 mr-1.5"
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
                    Saving...
                  </span>
                ) : (
                  'Save Changes'
                )}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}

export default PluginConfigModal;
