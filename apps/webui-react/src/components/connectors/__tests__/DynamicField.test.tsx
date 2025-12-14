import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { DynamicField } from '../DynamicField';
import type { FieldDefinition, SecretDefinition } from '../../../types/connector';

// Mock the formStyles utility
vi.mock('../../../utils/formStyles', () => ({
  getInputClassName: (hasError: boolean, isDisabled: boolean) => {
    let className = 'base-input';
    if (hasError) className += ' error-input';
    if (isDisabled) className += ' disabled-input';
    return className;
  },
}));

describe('DynamicField', () => {
  let onChange: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    onChange = vi.fn();
  });

  describe('Secret Field Rendering', () => {
    const secretField: SecretDefinition = {
      name: 'api_token',
      label: 'API Token',
      required: true,
      description: 'Your API token for authentication',
    };

    it('renders password input for non-multiline secret', () => {
      render(
        <DynamicField
          field={secretField}
          value=""
          onChange={onChange}
          isSecret={true}
        />
      );

      const input = screen.getByLabelText(/api token/i);
      expect(input).toHaveAttribute('type', 'password');
      expect(input).toHaveAttribute('autocomplete', 'off');
    });

    it('renders textarea for multiline secret', () => {
      const multilineSecret: SecretDefinition = {
        ...secretField,
        name: 'ssh_key',
        label: 'SSH Key',
        is_multiline: true,
      };

      render(
        <DynamicField
          field={multilineSecret}
          value=""
          onChange={onChange}
          isSecret={true}
        />
      );

      const textarea = screen.getByRole('textbox');
      expect(textarea.tagName).toBe('TEXTAREA');
      expect(textarea).toHaveClass('font-mono');
    });

    it('shows required indicator (*) for required secrets', () => {
      render(
        <DynamicField
          field={secretField}
          value=""
          onChange={onChange}
          isSecret={true}
        />
      );

      const label = screen.getByLabelText(/api token/i).closest('div')?.querySelector('label');
      const asterisk = label?.querySelector('.text-red-500');
      expect(asterisk).toHaveTextContent('*');
    });

    it('shows description when provided', () => {
      render(
        <DynamicField
          field={secretField}
          value=""
          onChange={onChange}
          isSecret={true}
        />
      );

      expect(screen.getByText(/your api token for authentication/i)).toBeInTheDocument();
    });

    it('calls onChange when secret value changes', async () => {
      const user = userEvent.setup();

      render(
        <DynamicField
          field={secretField}
          value=""
          onChange={onChange}
          isSecret={true}
        />
      );

      const input = screen.getByLabelText(/api token/i);
      await user.type(input, 'secret123');

      expect(onChange).toHaveBeenCalled();
    });

    it('renders as secret when isSecret=true even if field has type property', () => {
      const fieldWithType: FieldDefinition = {
        name: 'password',
        type: 'text',
        label: 'Password',
        required: true,
      };

      render(
        <DynamicField
          field={fieldWithType}
          value=""
          onChange={onChange}
          isSecret={true}
        />
      );

      const input = screen.getByLabelText(/password/i);
      expect(input).toHaveAttribute('type', 'password');
    });
  });

  describe('TextField', () => {
    const textField: FieldDefinition = {
      name: 'username',
      type: 'text',
      label: 'Username',
      required: true,
      placeholder: 'Enter username',
      description: 'Your account username',
    };

    it('renders with correct attributes', () => {
      render(
        <DynamicField
          field={textField}
          value=""
          onChange={onChange}
        />
      );

      const input = screen.getByLabelText(/username/i);
      expect(input).toHaveAttribute('type', 'text');
      expect(input).toHaveAttribute('placeholder', 'Enter username');
    });

    it('calls onChange with string value on input', async () => {
      const user = userEvent.setup();

      render(
        <DynamicField
          field={textField}
          value=""
          onChange={onChange}
        />
      );

      const input = screen.getByLabelText(/username/i);
      await user.type(input, 'hello');

      // Each character triggers onChange
      expect(onChange).toHaveBeenCalledTimes(5);
      expect(onChange).toHaveBeenLastCalledWith('o');
    });

    it('shows error message when error prop provided', () => {
      render(
        <DynamicField
          field={textField}
          value=""
          onChange={onChange}
          error="This field is required"
        />
      );

      expect(screen.getByText('This field is required')).toBeInTheDocument();
    });

    it('shows description text', () => {
      render(
        <DynamicField
          field={textField}
          value=""
          onChange={onChange}
        />
      );

      expect(screen.getByText('Your account username')).toBeInTheDocument();
    });
  });

  describe('NumberField', () => {
    const numberField: FieldDefinition = {
      name: 'port',
      type: 'number',
      label: 'Port',
      required: true,
      min: 1,
      max: 65535,
      step: 1,
      default: 443,
    };

    it('renders with min/max/step attributes', () => {
      render(
        <DynamicField
          field={numberField}
          value={443}
          onChange={onChange}
        />
      );

      const input = screen.getByLabelText(/port/i);
      expect(input).toHaveAttribute('type', 'number');
      expect(input).toHaveAttribute('min', '1');
      expect(input).toHaveAttribute('max', '65535');
      expect(input).toHaveAttribute('step', '1');
    });

    it('converts string input to number on change', async () => {
      const user = userEvent.setup();

      render(
        <DynamicField
          field={numberField}
          value={443}
          onChange={onChange}
        />
      );

      const input = screen.getByLabelText(/port/i);
      await user.clear(input);
      await user.type(input, '8080');

      // Last call should be with number 8080
      const lastCall = onChange.mock.calls[onChange.mock.calls.length - 1][0];
      expect(typeof lastCall).toBe('number');
    });

    it('displays default value when value is undefined', () => {
      render(
        <DynamicField
          field={numberField}
          value={undefined}
          onChange={onChange}
        />
      );

      const input = screen.getByLabelText(/port/i);
      expect(input).toHaveValue(443);
    });

    it('displays provided value over default', () => {
      render(
        <DynamicField
          field={numberField}
          value={8080}
          onChange={onChange}
        />
      );

      const input = screen.getByLabelText(/port/i);
      expect(input).toHaveValue(8080);
    });
  });

  describe('SelectField', () => {
    const selectField: FieldDefinition = {
      name: 'auth_method',
      type: 'select',
      label: 'Auth Method',
      required: true,
      default: 'none',
      options: [
        { value: 'none', label: 'None' },
        { value: 'token', label: 'Token' },
        { value: 'ssh', label: 'SSH Key' },
      ],
    };

    it('renders select with all options', () => {
      render(
        <DynamicField
          field={selectField}
          value=""
          onChange={onChange}
        />
      );

      const select = screen.getByLabelText(/auth method/i);
      expect(select.tagName).toBe('SELECT');

      const options = screen.getAllByRole('option');
      expect(options).toHaveLength(3);
      expect(options[0]).toHaveTextContent('None');
      expect(options[1]).toHaveTextContent('Token');
      expect(options[2]).toHaveTextContent('SSH Key');
    });

    it('uses default value when value prop is undefined', () => {
      render(
        <DynamicField
          field={selectField}
          value={undefined}
          onChange={onChange}
        />
      );

      const select = screen.getByLabelText(/auth method/i);
      expect(select).toHaveValue('none');
    });

    it('calls onChange with selected value', async () => {
      const user = userEvent.setup();

      render(
        <DynamicField
          field={selectField}
          value="none"
          onChange={onChange}
        />
      );

      const select = screen.getByLabelText(/auth method/i);
      await user.selectOptions(select, 'token');

      expect(onChange).toHaveBeenCalledWith('token');
    });
  });

  describe('MultiselectField', () => {
    const multiselectField: FieldDefinition = {
      name: 'features',
      type: 'multiselect',
      label: 'Features',
      required: false,
      options: [
        { value: 'sync', label: 'Sync' },
        { value: 'backup', label: 'Backup' },
        { value: 'share', label: 'Share' },
      ],
    };

    it('renders checkboxes for each option', () => {
      render(
        <DynamicField
          field={multiselectField}
          value={[]}
          onChange={onChange}
        />
      );

      const checkboxes = screen.getAllByRole('checkbox');
      expect(checkboxes).toHaveLength(3);
      expect(screen.getByText('Sync')).toBeInTheDocument();
      expect(screen.getByText('Backup')).toBeInTheDocument();
      expect(screen.getByText('Share')).toBeInTheDocument();
    });

    it('adds value to array when checkbox is checked', async () => {
      const user = userEvent.setup();

      render(
        <DynamicField
          field={multiselectField}
          value={['sync']}
          onChange={onChange}
        />
      );

      const backupCheckbox = screen.getByRole('checkbox', { name: /backup/i });
      await user.click(backupCheckbox);

      expect(onChange).toHaveBeenCalledWith(['sync', 'backup']);
    });

    it('removes value from array when checkbox is unchecked', async () => {
      const user = userEvent.setup();

      render(
        <DynamicField
          field={multiselectField}
          value={['sync', 'backup']}
          onChange={onChange}
        />
      );

      const syncCheckbox = screen.getByRole('checkbox', { name: /sync/i });
      await user.click(syncCheckbox);

      expect(onChange).toHaveBeenCalledWith(['backup']);
    });

    it('handles undefined value as empty array', () => {
      render(
        <DynamicField
          field={multiselectField}
          value={undefined}
          onChange={onChange}
        />
      );

      const checkboxes = screen.getAllByRole('checkbox');
      checkboxes.forEach((checkbox) => {
        expect(checkbox).not.toBeChecked();
      });
    });
  });

  describe('TextareaField', () => {
    const textareaField: FieldDefinition = {
      name: 'description',
      type: 'textarea',
      label: 'Description',
      required: false,
      placeholder: 'Enter description',
    };

    it('renders textarea with correct attributes', () => {
      render(
        <DynamicField
          field={textareaField}
          value=""
          onChange={onChange}
        />
      );

      const textarea = screen.getByLabelText(/description/i);
      expect(textarea.tagName).toBe('TEXTAREA');
      expect(textarea).toHaveAttribute('placeholder', 'Enter description');
    });

    it('calls onChange with textarea value', async () => {
      const user = userEvent.setup();

      render(
        <DynamicField
          field={textareaField}
          value=""
          onChange={onChange}
        />
      );

      const textarea = screen.getByLabelText(/description/i);
      await user.type(textarea, 'Test');

      expect(onChange).toHaveBeenCalled();
    });
  });

  describe('BooleanField', () => {
    const booleanField: FieldDefinition = {
      name: 'enabled',
      type: 'boolean',
      label: 'Enable Feature',
      required: false,
      default: false,
      description: 'Enable this experimental feature',
    };

    it('renders checkbox with label', () => {
      render(
        <DynamicField
          field={booleanField}
          value={false}
          onChange={onChange}
        />
      );

      const checkbox = screen.getByRole('checkbox');
      expect(checkbox).not.toBeChecked();
      expect(screen.getByText('Enable Feature')).toBeInTheDocument();
    });

    it('uses default value when value is undefined', () => {
      const fieldWithDefault: FieldDefinition = {
        ...booleanField,
        default: true,
      };

      render(
        <DynamicField
          field={fieldWithDefault}
          value={undefined}
          onChange={onChange}
        />
      );

      const checkbox = screen.getByRole('checkbox');
      expect(checkbox).toBeChecked();
    });

    it('calls onChange with boolean on toggle', async () => {
      const user = userEvent.setup();

      render(
        <DynamicField
          field={booleanField}
          value={false}
          onChange={onChange}
        />
      );

      const checkbox = screen.getByRole('checkbox');
      await user.click(checkbox);

      expect(onChange).toHaveBeenCalledWith(true);
    });
  });

  describe('GlobListField', () => {
    const globListField: FieldDefinition = {
      name: 'include_patterns',
      type: 'glob_list',
      label: 'Include Patterns',
      required: false,
      placeholder: 'e.g., *.md, docs/**',
    };

    it('converts array to comma-separated string for display', () => {
      render(
        <DynamicField
          field={globListField}
          value={['*.md', 'docs/**']}
          onChange={onChange}
        />
      );

      const input = screen.getByLabelText(/include patterns/i);
      expect(input).toHaveValue('*.md, docs/**');
    });

    it('converts comma-separated input to array on change', async () => {
      const user = userEvent.setup();

      render(
        <DynamicField
          field={globListField}
          value={[]}
          onChange={onChange}
        />
      );

      const input = screen.getByLabelText(/include patterns/i);
      // Clear and paste the full value to get the final result
      await user.clear(input);
      // Simulate pasting the full value
      await user.click(input);
      await user.paste('*.js, *.ts');

      // Check the last call contains both patterns
      const lastCall = onChange.mock.calls[onChange.mock.calls.length - 1][0];
      expect(lastCall).toContain('*.js');
      expect(lastCall).toContain('*.ts');
    });

    it('filters out empty patterns', async () => {
      const user = userEvent.setup();

      render(
        <DynamicField
          field={globListField}
          value={[]}
          onChange={onChange}
        />
      );

      const input = screen.getByLabelText(/include patterns/i);
      // Simulate pasting text with empty patterns
      await user.click(input);
      await user.paste('*.js,  , *.ts');

      // Final onChange should filter empty strings
      const lastCall = onChange.mock.calls[onChange.mock.calls.length - 1][0];
      expect(lastCall).not.toContain('');
      expect(lastCall.filter((p: string) => p.length > 0)).toEqual(lastCall);
    });

    it('shows default description for glob patterns', () => {
      render(
        <DynamicField
          field={{ ...globListField, description: undefined }}
          value={[]}
          onChange={onChange}
        />
      );

      expect(screen.getByText('Comma-separated glob patterns')).toBeInTheDocument();
    });
  });

  describe('Disabled State', () => {
    it('disables text input when disabled=true', () => {
      const textField: FieldDefinition = {
        name: 'test',
        type: 'text',
        label: 'Test',
        required: false,
      };

      render(
        <DynamicField
          field={textField}
          value=""
          onChange={onChange}
          disabled={true}
        />
      );

      const input = screen.getByLabelText(/test/i);
      expect(input).toBeDisabled();
    });

    it('disables select when disabled=true', () => {
      const selectField: FieldDefinition = {
        name: 'test',
        type: 'select',
        label: 'Test',
        required: false,
        options: [{ value: 'a', label: 'A' }],
      };

      render(
        <DynamicField
          field={selectField}
          value=""
          onChange={onChange}
          disabled={true}
        />
      );

      const select = screen.getByLabelText(/test/i);
      expect(select).toBeDisabled();
    });

    it('disables checkboxes in multiselect when disabled=true', () => {
      const multiselectField: FieldDefinition = {
        name: 'test',
        type: 'multiselect',
        label: 'Test',
        required: false,
        options: [
          { value: 'a', label: 'A' },
          { value: 'b', label: 'B' },
        ],
      };

      render(
        <DynamicField
          field={multiselectField}
          value={[]}
          onChange={onChange}
          disabled={true}
        />
      );

      const checkboxes = screen.getAllByRole('checkbox');
      checkboxes.forEach((checkbox) => {
        expect(checkbox).toBeDisabled();
      });
    });
  });

  describe('Default Field Type Fallback', () => {
    it('renders as TextField for unknown field types', () => {
      const unknownField = {
        name: 'unknown',
        type: 'unknown_type' as 'text',
        label: 'Unknown',
        required: false,
      };

      render(
        <DynamicField
          field={unknownField}
          value=""
          onChange={onChange}
        />
      );

      const input = screen.getByLabelText(/unknown/i);
      expect(input).toHaveAttribute('type', 'text');
    });
  });
});
