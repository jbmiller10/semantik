import * as React from 'react';
import type { FieldDefinition, SecretDefinition } from '../../types/connector';
import { getInputClassName } from '../../utils/formStyles';

interface DynamicFieldProps {
  field: FieldDefinition | SecretDefinition;
  value: unknown;
  onChange: (value: unknown) => void;
  error?: string;
  disabled?: boolean;
  isSecret?: boolean;
}

/**
 * Type guard to check if field is a FieldDefinition
 */
function isFieldDefinition(
  field: FieldDefinition | SecretDefinition
): field is FieldDefinition {
  return 'type' in field;
}

/**
 * Renders a form field dynamically based on its definition
 */
export function DynamicField({
  field,
  value,
  onChange,
  error,
  disabled = false,
  isSecret = false,
}: DynamicFieldProps) {
  const inputClassName = getInputClassName(!!error, disabled);

  // For secrets, always render as password/textarea
  if (isSecret || !isFieldDefinition(field)) {
    const secretField = field as SecretDefinition;
    return (
      <div className="space-y-1">
        <label
          htmlFor={field.name}
          className="block text-xs font-bold text-gray-400 uppercase tracking-wider"
        >
          {field.label}
          {field.required && <span className="text-signal-500 ml-1">*</span>}
        </label>

        {secretField.is_multiline ? (
          <textarea
            id={field.name}
            value={(value as string) || ''}
            onChange={(e) => onChange(e.target.value)}
            disabled={disabled}
            placeholder={`Enter ${field.label.toLowerCase()}`}
            rows={6}
            className={`${inputClassName} font-mono text-sm`}
          />
        ) : (
          <input
            type="password"
            id={field.name}
            value={(value as string) || ''}
            onChange={(e) => onChange(e.target.value)}
            disabled={disabled}
            placeholder={`Enter ${field.label.toLowerCase()}`}
            className={inputClassName}
            autoComplete="off"
          />
        )}

        {field.description && (
          <p className="text-xs text-gray-500">{field.description}</p>
        )}

        {error && <p className="text-sm text-red-400">{error}</p>}
      </div>
    );
  }

  // Handle regular fields based on type
  const fieldDef = field as FieldDefinition;

  switch (fieldDef.type) {
    case 'text':
      return (
        <TextField
          field={fieldDef}
          value={value as string}
          onChange={onChange}
          error={error}
          disabled={disabled}
        />
      );

    case 'number':
      return (
        <NumberField
          field={fieldDef}
          value={value as number}
          onChange={onChange}
          error={error}
          disabled={disabled}
        />
      );

    case 'select':
      return (
        <SelectField
          field={fieldDef}
          value={value as string}
          onChange={onChange}
          error={error}
          disabled={disabled}
        />
      );

    case 'multiselect':
      return (
        <MultiselectField
          field={fieldDef}
          value={value as string[]}
          onChange={onChange}
          error={error}
          disabled={disabled}
        />
      );

    case 'textarea':
      return (
        <TextareaField
          field={fieldDef}
          value={value as string}
          onChange={onChange}
          error={error}
          disabled={disabled}
        />
      );

    case 'boolean':
      return (
        <BooleanField
          field={fieldDef}
          value={value as boolean}
          onChange={onChange}
          error={error}
          disabled={disabled}
        />
      );

    case 'glob_list':
      return (
        <GlobListField
          field={fieldDef}
          value={value as string[]}
          onChange={onChange}
          error={error}
          disabled={disabled}
        />
      );

    default:
      return (
        <TextField
          field={fieldDef}
          value={value as string}
          onChange={onChange}
          error={error}
          disabled={disabled}
        />
      );
  }
}

// Field type components

interface FieldComponentProps<T> {
  field: FieldDefinition;
  value: T;
  onChange: (value: T) => void;
  error?: string;
  disabled: boolean;
}

function TextField({
  field,
  value,
  onChange,
  error,
  disabled,
}: FieldComponentProps<string>) {
  const inputClassName = getInputClassName(!!error, disabled);

  return (
    <div className="space-y-1">
      <label
        htmlFor={field.name}
        className="block text-xs font-bold text-gray-400 uppercase tracking-wider"
      >
        {field.label}
        {field.required && <span className="text-signal-500 ml-1">*</span>}
      </label>
      <input
        type="text"
        id={field.name}
        value={value || ''}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        placeholder={field.placeholder}
        className={inputClassName}
      />
      {field.description && (
        <p className="text-xs text-gray-500">{field.description}</p>
      )}
      {error && <p className="text-sm text-red-400">{error}</p>}
    </div>
  );
}

function NumberField({
  field,
  value,
  onChange,
  error,
  disabled,
}: FieldComponentProps<number>) {
  const inputClassName = getInputClassName(!!error, disabled);

  return (
    <div className="space-y-1">
      <label
        htmlFor={field.name}
        className="block text-xs font-bold text-gray-400 uppercase tracking-wider"
      >
        {field.label}
        {field.required && <span className="text-signal-500 ml-1">*</span>}
      </label>
      <input
        type="number"
        id={field.name}
        value={value ?? field.default ?? ''}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        placeholder={field.placeholder}
        min={field.min}
        max={field.max}
        step={field.step}
        className={inputClassName}
      />
      {field.description && (
        <p className="text-xs text-gray-500">{field.description}</p>
      )}
      {error && <p className="text-sm text-red-400">{error}</p>}
    </div>
  );
}

function SelectField({
  field,
  value,
  onChange,
  error,
  disabled,
}: FieldComponentProps<string>) {
  const inputClassName = getInputClassName(!!error, disabled);

  return (
    <div className="space-y-1">
      <label
        htmlFor={field.name}
        className="block text-xs font-bold text-gray-400 uppercase tracking-wider"
      >
        {field.label}
        {field.required && <span className="text-signal-500 ml-1">*</span>}
      </label>
      <select
        id={field.name}
        value={value || (field.default as string) || ''}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className={inputClassName}
      >
        {field.options?.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      {field.description && (
        <p className="text-xs text-gray-500">{field.description}</p>
      )}
      {error && <p className="text-sm text-red-400">{error}</p>}
    </div>
  );
}

function MultiselectField({
  field,
  value,
  onChange,
  error,
  disabled,
}: FieldComponentProps<string[]>) {
  const currentValue = value || [];

  const handleToggle = (optionValue: string) => {
    if (currentValue.includes(optionValue)) {
      onChange(currentValue.filter((v) => v !== optionValue));
    } else {
      onChange([...currentValue, optionValue]);
    }
  };

  return (
    <div className="space-y-1">
      <label className="block text-xs font-bold text-gray-400 uppercase tracking-wider">
        {field.label}
        {field.required && <span className="text-signal-500 ml-1">*</span>}
      </label>
      <div className="space-y-2 mt-2">
        {field.options?.map((option) => (
          <label
            key={option.value}
            className={`flex items-center ${disabled ? 'opacity-50' : ''}`}
          >
            <input
              type="checkbox"
              checked={currentValue.includes(option.value)}
              onChange={() => handleToggle(option.value)}
              disabled={disabled}
              className="h-4 w-4 text-signal-600 focus:ring-signal-500 bg-void-800 border-white/20 rounded"
            />
            <span className="ml-2 text-sm text-gray-300">{option.label}</span>
          </label>
        ))}
      </div>
      {field.description && (
        <p className="text-xs text-gray-500">{field.description}</p>
      )}
      {error && <p className="text-sm text-red-400">{error}</p>}
    </div>
  );
}

function TextareaField({
  field,
  value,
  onChange,
  error,
  disabled,
}: FieldComponentProps<string>) {
  const inputClassName = getInputClassName(!!error, disabled);

  return (
    <div className="space-y-1">
      <label
        htmlFor={field.name}
        className="block text-xs font-bold text-gray-400 uppercase tracking-wider"
      >
        {field.label}
        {field.required && <span className="text-signal-500 ml-1">*</span>}
      </label>
      <textarea
        id={field.name}
        value={value || ''}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        placeholder={field.placeholder}
        rows={4}
        className={inputClassName}
      />
      {field.description && (
        <p className="text-xs text-gray-500">{field.description}</p>
      )}
      {error && <p className="text-sm text-red-400">{error}</p>}
    </div>
  );
}

function BooleanField({
  field,
  value,
  onChange,
  error,
  disabled,
}: FieldComponentProps<boolean>) {
  const currentValue = value ?? (field.default as boolean) ?? false;

  return (
    <div className="space-y-1">
      <label
        className={`flex items-center ${disabled ? 'opacity-50' : ''}`}
      >
        <input
          type="checkbox"
          id={field.name}
          checked={currentValue}
          onChange={(e) => onChange(e.target.checked)}
          disabled={disabled}
          className="h-4 w-4 text-signal-600 focus:ring-signal-500 bg-void-800 border-white/20 rounded"
        />
        <span className="ml-2 text-sm font-medium text-gray-300">
          {field.label}
        </span>
      </label>
      {field.description && (
        <p className="text-xs text-gray-500 ml-6">{field.description}</p>
      )}
      {error && <p className="text-sm text-red-400">{error}</p>}
    </div>
  );
}

function GlobListField({
  field,
  value,
  onChange,
  error,
  disabled,
}: FieldComponentProps<string[]>) {
  const inputClassName = getInputClassName(!!error, disabled);

  // Track the raw input string separately to allow typing commas
  const [rawInput, setRawInput] = React.useState<string>(() =>
    Array.isArray(value) ? value.join(', ') : ''
  );

  // Sync rawInput when value changes externally (e.g., form reset)
  React.useEffect(() => {
    const externalValue = Array.isArray(value) ? value.join(', ') : '';
    const externalParsed = Array.isArray(value) ? value : [];
    setRawInput((currentRaw) => {
      // Only update if the parsed values are different (not just whitespace differences)
      const currentParsed = currentRaw
        .split(',')
        .map((p) => p.trim())
        .filter((p) => p.length > 0);
      if (JSON.stringify(currentParsed) !== JSON.stringify(externalParsed)) {
        return externalValue;
      }
      return currentRaw;
    });
  }, [value]);

  const handleChange = (inputValue: string) => {
    // Update raw input immediately to allow typing commas
    setRawInput(inputValue);

    // Convert comma-separated string to array for the parent
    const patterns = inputValue
      .split(',')
      .map((p) => p.trim())
      .filter((p) => p.length > 0);
    onChange(patterns);
  };

  return (
    <div className="space-y-1">
      <label
        htmlFor={field.name}
        className="block text-xs font-bold text-gray-400 uppercase tracking-wider"
      >
        {field.label}
        {field.required && <span className="text-signal-500 ml-1">*</span>}
      </label>
      <input
        type="text"
        id={field.name}
        value={rawInput}
        onChange={(e) => handleChange(e.target.value)}
        disabled={disabled}
        placeholder={field.placeholder || 'e.g., *.md, docs/**'}
        className={inputClassName}
      />
      <p className="text-xs text-gray-500">
        {field.description || 'Comma-separated glob patterns'}
      </p>
      {error && <p className="text-sm text-red-400">{error}</p>}
    </div>
  );
}

export default DynamicField;
