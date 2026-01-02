import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import PluginConfigForm from '../PluginConfigForm';
import type { PluginConfigSchema } from '@/types/plugin';

describe('PluginConfigForm', () => {
  const mockOnChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('empty schema', () => {
    it('displays message when no configuration options exist', () => {
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {},
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{}}
          onChange={mockOnChange}
          errors={{}}
        />
      );

      expect(
        screen.getByText('This plugin has no configuration options.')
      ).toBeInTheDocument();
    });
  });

  describe('string fields', () => {
    it('renders text input for string type', () => {
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          api_key: {
            type: 'string',
            title: 'API Key',
            description: 'Your API key',
          },
        },
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{ api_key: 'test-key' }}
          onChange={mockOnChange}
          errors={{}}
        />
      );

      expect(screen.getByLabelText(/api key/i)).toBeInTheDocument();
      expect(screen.getByDisplayValue('test-key')).toBeInTheDocument();
      expect(screen.getByText('Your API key')).toBeInTheDocument();
    });

    it('renders password input for secret fields', () => {
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          api_secret: {
            type: 'string',
            title: 'API Secret',
          },
        },
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{}}
          onChange={mockOnChange}
          errors={{}}
        />
      );

      expect(screen.getByLabelText(/api secret/i)).toHaveAttribute(
        'type',
        'password'
      );
    });

    it('renders text input for env var fields', () => {
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          api_key_env: {
            type: 'string',
            title: 'API Key Environment Variable',
          },
        },
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{}}
          onChange={mockOnChange}
          errors={{}}
        />
      );

      expect(
        screen.getByText(/enter the name of an environment variable/i)
      ).toBeInTheDocument();
    });

    it('calls onChange when text input changes', async () => {
      const user = userEvent.setup();
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          name: {
            type: 'string',
            title: 'Name',
          },
        },
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{ name: '' }}
          onChange={mockOnChange}
          errors={{}}
        />
      );

      await user.type(screen.getByLabelText(/name/i), 'test');

      expect(mockOnChange).toHaveBeenCalled();
    });

    it('renders textarea for long text fields', () => {
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          content: {
            type: 'string',
            title: 'Content',
            maxLength: 500,
          },
        },
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{}}
          onChange={mockOnChange}
          errors={{}}
        />
      );

      const textarea = screen.getByLabelText(/content/i);
      expect(textarea.tagName.toLowerCase()).toBe('textarea');
    });
  });

  describe('boolean fields', () => {
    it('renders checkbox for boolean type', () => {
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          enabled: {
            type: 'boolean',
            title: 'Enabled',
            description: 'Enable this feature',
          },
        },
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{ enabled: true }}
          onChange={mockOnChange}
          errors={{}}
        />
      );

      const checkbox = screen.getByRole('checkbox', { name: /enabled/i });
      expect(checkbox).toBeChecked();
      expect(screen.getByText('Enable this feature')).toBeInTheDocument();
    });

    it('calls onChange when checkbox is toggled', async () => {
      const user = userEvent.setup();
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          enabled: {
            type: 'boolean',
            title: 'Enabled',
          },
        },
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{ enabled: false }}
          onChange={mockOnChange}
          errors={{}}
        />
      );

      await user.click(screen.getByRole('checkbox'));

      expect(mockOnChange).toHaveBeenCalledWith({ enabled: true });
    });
  });

  describe('number fields', () => {
    it('renders number input for number type', () => {
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          timeout: {
            type: 'number',
            title: 'Timeout',
            minimum: 0,
            maximum: 60,
          },
        },
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{ timeout: 30 }}
          onChange={mockOnChange}
          errors={{}}
        />
      );

      const input = screen.getByLabelText(/timeout/i);
      expect(input).toHaveAttribute('type', 'number');
      expect(input).toHaveAttribute('min', '0');
      expect(input).toHaveAttribute('max', '60');
      expect(input).toHaveValue(30);
    });

    it('renders integer step for integer type', () => {
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          count: {
            type: 'integer',
            title: 'Count',
          },
        },
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{ count: 5 }}
          onChange={mockOnChange}
          errors={{}}
        />
      );

      expect(screen.getByLabelText(/count/i)).toHaveAttribute('step', '1');
    });

    it('calls onChange with number value', async () => {
      const user = userEvent.setup();
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          value: {
            type: 'number',
            title: 'Value',
          },
        },
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{ value: 0 }}
          onChange={mockOnChange}
          errors={{}}
        />
      );

      const input = screen.getByLabelText(/value/i);
      await user.clear(input);
      await user.type(input, '42');

      expect(mockOnChange).toHaveBeenCalled();
    });
  });

  describe('enum fields', () => {
    it('renders select for enum type', () => {
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          model: {
            type: 'string',
            title: 'Model',
            enum: ['gpt-3.5', 'gpt-4', 'gpt-4o'],
          },
        },
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{ model: 'gpt-4' }}
          onChange={mockOnChange}
          errors={{}}
        />
      );

      const select = screen.getByLabelText(/model/i);
      expect(select.tagName.toLowerCase()).toBe('select');
      expect(screen.getByRole('option', { name: 'gpt-3.5' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'gpt-4' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'gpt-4o' })).toBeInTheDocument();
    });

    it('calls onChange when selection changes', async () => {
      const user = userEvent.setup();
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          model: {
            type: 'string',
            title: 'Model',
            enum: ['gpt-3.5', 'gpt-4'],
          },
        },
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{ model: 'gpt-3.5' }}
          onChange={mockOnChange}
          errors={{}}
        />
      );

      await user.selectOptions(screen.getByLabelText(/model/i), 'gpt-4');

      expect(mockOnChange).toHaveBeenCalledWith({ model: 'gpt-4' });
    });
  });

  describe('array fields', () => {
    it('renders comma-separated input for array type', () => {
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          tags: {
            type: 'array',
            title: 'Tags',
            items: { type: 'string' },
          },
        },
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{ tags: ['tag1', 'tag2'] }}
          onChange={mockOnChange}
          errors={{}}
        />
      );

      expect(screen.getByDisplayValue('tag1, tag2')).toBeInTheDocument();
      expect(screen.getByText('Comma-separated list')).toBeInTheDocument();
    });

    it('parses comma-separated values on change', async () => {
      const user = userEvent.setup();
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          tags: {
            type: 'array',
            title: 'Tags',
          },
        },
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{ tags: [] }}
          onChange={mockOnChange}
          errors={{}}
        />
      );

      const input = screen.getByLabelText(/tags/i);
      await user.type(input, 'a, b, c');

      expect(mockOnChange).toHaveBeenCalled();
    });
  });

  describe('required fields', () => {
    it('displays required indicator for required fields', () => {
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          name: {
            type: 'string',
            title: 'Name',
          },
        },
        required: ['name'],
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{}}
          onChange={mockOnChange}
          errors={{}}
        />
      );

      expect(screen.getByText('*')).toBeInTheDocument();
    });
  });

  describe('error handling', () => {
    it('displays error message for field with error', () => {
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          name: {
            type: 'string',
            title: 'Name',
          },
        },
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{}}
          onChange={mockOnChange}
          errors={{ name: 'Name is required' }}
        />
      );

      expect(screen.getByText('Name is required')).toBeInTheDocument();
    });
  });

  describe('disabled state', () => {
    it('disables all inputs when disabled prop is true', () => {
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          name: { type: 'string', title: 'Name' },
          enabled: { type: 'boolean', title: 'Enabled' },
          count: { type: 'number', title: 'Count' },
        },
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{}}
          onChange={mockOnChange}
          errors={{}}
          disabled={true}
        />
      );

      expect(screen.getByLabelText(/name/i)).toBeDisabled();
      expect(screen.getByRole('checkbox')).toBeDisabled();
      expect(screen.getByLabelText(/count/i)).toBeDisabled();
    });
  });

  describe('default values', () => {
    it('uses default value from schema when value is not provided', () => {
      const schema: PluginConfigSchema = {
        type: 'object',
        properties: {
          name: {
            type: 'string',
            title: 'Name',
            default: 'Default Name',
          },
        },
      };

      render(
        <PluginConfigForm
          schema={schema}
          values={{}}
          onChange={mockOnChange}
          errors={{}}
        />
      );

      expect(screen.getByDisplayValue('Default Name')).toBeInTheDocument();
    });
  });
});
