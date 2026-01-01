import type { PluginConfigSchema, JsonSchemaProperty } from '../../types/plugin';
import { getInputClassName } from '../../utils/formStyles';

interface PluginConfigFormProps {
  schema: PluginConfigSchema;
  values: Record<string, unknown>;
  onChange: (values: Record<string, unknown>) => void;
  errors: Record<string, string>;
  disabled?: boolean;
}

/**
 * Render a form field based on JSON Schema property definition
 */
function renderField(
  name: string,
  property: JsonSchemaProperty,
  value: unknown,
  onChange: (name: string, value: unknown) => void,
  error?: string,
  required?: boolean,
  disabled?: boolean
): JSX.Element {
  const inputClassName = getInputClassName(!!error, !!disabled);
  const label = property.title || name;
  const description = property.description;
  const isEnvVar = name.endsWith('_env');

  // Boolean field
  if (property.type === 'boolean') {
    return (
      <div key={name} className="flex items-start">
        <div className="flex items-center h-5">
          <input
            type="checkbox"
            id={name}
            name={name}
            checked={value === true}
            onChange={(e) => onChange(name, e.target.checked)}
            disabled={disabled}
            className="h-4 w-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 disabled:opacity-50"
          />
        </div>
        <div className="ml-3 text-sm">
          <label htmlFor={name} className="font-medium text-gray-700">
            {label}
            {required && <span className="text-red-500 ml-1">*</span>}
          </label>
          {description && <p className="text-gray-500">{description}</p>}
          {error && <p className="text-red-600 text-xs mt-1">{error}</p>}
        </div>
      </div>
    );
  }

  // Enum/select field
  if (property.enum && property.enum.length > 0) {
    return (
      <div key={name}>
        <label htmlFor={name} className="block text-sm font-medium text-gray-700">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
        {description && <p className="text-xs text-gray-500 mt-0.5">{description}</p>}
        <select
          id={name}
          name={name}
          value={(value as string) ?? property.default ?? ''}
          onChange={(e) => onChange(name, e.target.value)}
          disabled={disabled}
          className={inputClassName}
        >
          <option value="">Select...</option>
          {property.enum.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
        {error && <p className="text-red-600 text-xs mt-1">{error}</p>}
      </div>
    );
  }

  // Number field
  if (property.type === 'number' || property.type === 'integer') {
    return (
      <div key={name}>
        <label htmlFor={name} className="block text-sm font-medium text-gray-700">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
        {description && <p className="text-xs text-gray-500 mt-0.5">{description}</p>}
        <input
          type="number"
          id={name}
          name={name}
          value={(value as number) ?? property.default ?? ''}
          onChange={(e) =>
            onChange(name, e.target.value ? Number(e.target.value) : undefined)
          }
          min={property.minimum ?? property.exclusiveMinimum}
          max={property.maximum ?? property.exclusiveMaximum}
          step={property.type === 'integer' ? 1 : 'any'}
          disabled={disabled}
          className={inputClassName}
        />
        {error && <p className="text-red-600 text-xs mt-1">{error}</p>}
      </div>
    );
  }

  // Array field (simple comma-separated)
  if (property.type === 'array') {
    const arrayValue = Array.isArray(value) ? value.join(', ') : '';
    return (
      <div key={name}>
        <label htmlFor={name} className="block text-sm font-medium text-gray-700">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
        {description && <p className="text-xs text-gray-500 mt-0.5">{description}</p>}
        <input
          type="text"
          id={name}
          name={name}
          value={arrayValue}
          onChange={(e) => {
            const items = e.target.value
              .split(',')
              .map((s) => s.trim())
              .filter(Boolean);
            onChange(name, items.length > 0 ? items : undefined);
          }}
          placeholder="item1, item2, item3"
          disabled={disabled}
          className={inputClassName}
        />
        <p className="text-xs text-gray-400 mt-0.5">Comma-separated list</p>
        {error && <p className="text-red-600 text-xs mt-1">{error}</p>}
      </div>
    );
  }

  // Default: string/text field
  // Check if it should be a textarea (long description or multiline hint)
  const isTextarea =
    (property.maxLength && property.maxLength > 200) ||
    property.description?.toLowerCase().includes('multiline');

  // Check if it's likely a secret (environment variable reference)
  const isSecret = isEnvVar || name.toLowerCase().includes('key') || name.toLowerCase().includes('secret');

  return (
    <div key={name}>
      <label htmlFor={name} className="block text-sm font-medium text-gray-700">
        {label}
        {required && <span className="text-red-500 ml-1">*</span>}
      </label>
      {description && <p className="text-xs text-gray-500 mt-0.5">{description}</p>}
      {isEnvVar && (
        <p className="text-xs text-blue-600 mt-0.5">
          Enter the name of an environment variable (e.g., OPENAI_API_KEY)
        </p>
      )}
      {isTextarea ? (
        <textarea
          id={name}
          name={name}
          value={(value as string) ?? property.default ?? ''}
          onChange={(e) => onChange(name, e.target.value || undefined)}
          disabled={disabled}
          rows={4}
          className={inputClassName}
          minLength={property.minLength}
          maxLength={property.maxLength}
        />
      ) : (
        <input
          type={isSecret && !isEnvVar ? 'password' : 'text'}
          id={name}
          name={name}
          value={(value as string) ?? property.default ?? ''}
          onChange={(e) => onChange(name, e.target.value || undefined)}
          disabled={disabled}
          className={inputClassName}
          minLength={property.minLength}
          maxLength={property.maxLength}
          pattern={property.pattern}
        />
      )}
      {error && <p className="text-red-600 text-xs mt-1">{error}</p>}
    </div>
  );
}

function PluginConfigForm({
  schema,
  values,
  onChange,
  errors,
  disabled = false,
}: PluginConfigFormProps) {
  const properties = schema.properties || {};
  const required = schema.required || [];

  const handleFieldChange = (name: string, value: unknown) => {
    onChange({
      ...values,
      [name]: value,
    });
  };

  // No properties to configure
  if (Object.keys(properties).length === 0) {
    return (
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
    );
  }

  return (
    <div className="space-y-4">
      {Object.entries(properties).map(([name, property]) =>
        renderField(
          name,
          property,
          values[name],
          handleFieldChange,
          errors[name],
          required.includes(name),
          disabled
        )
      )}
    </div>
  );
}

export default PluginConfigForm;
