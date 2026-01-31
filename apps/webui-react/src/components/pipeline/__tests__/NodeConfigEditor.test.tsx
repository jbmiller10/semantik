import { describe, it, expect, vi, beforeEach } from 'vitest';
import React from 'react';
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import { NodeConfigEditor } from '../NodeConfigEditor';
import type { PipelineNode } from '@/types/pipeline';
import * as usePluginsModule from '@/hooks/usePlugins';
import * as useModelManagerModule from '@/hooks/useModelManager';

// Mock the usePlugins hooks
vi.mock('@/hooks/usePlugins', () => ({
  usePipelinePlugins: vi.fn(),
  usePipelinePluginConfigSchema: vi.fn(),
}));

vi.mock('@/hooks/useModelManager', () => ({
  useModelManagerModels: vi.fn(),
}));

// Mock data using the new PipelinePluginInfo structure
const mockPlugins = [
  {
    id: 'text',
    type: 'parser',
    display_name: 'Text Parser',
    description: 'Parse plain text',
    source: 'builtin',
    enabled: true,
  },
  {
    id: 'unstructured',
    type: 'parser',
    display_name: 'Unstructured',
    description: 'Parse PDFs and docs',
    source: 'builtin',
    enabled: true,
  },
];

const mockSchema = {
  type: 'object' as const,
  properties: {
    strategy: {
      type: 'string' as const,
      title: 'Strategy',
      enum: ['auto', 'fast', 'hi_res'],
      default: 'auto',
    },
  },
};

describe('NodeConfigEditor', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (usePluginsModule.usePipelinePlugins as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockPlugins,
      isLoading: false,
    });
    (usePluginsModule.usePipelinePluginConfigSchema as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockSchema,
      isLoading: false,
    });
    (useModelManagerModule.useModelManagerModels as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { models: [], cache_size: null, hf_cache_scan_error: null },
      isLoading: false,
    });
  });

  const mockNode: PipelineNode = {
    id: 'parser1',
    type: 'parser',
    plugin_id: 'unstructured',
    config: { strategy: 'auto' },
  };

  it('renders node type label', () => {
    render(
      <NodeConfigEditor
        node={mockNode}
        onChange={vi.fn()}
      />
    );
    expect(screen.getByText('Parser')).toBeInTheDocument();
  });

  it('renders plugin selector with current plugin selected', () => {
    render(
      <NodeConfigEditor
        node={mockNode}
        onChange={vi.fn()}
      />
    );
    // Get the first combobox which is the plugin selector
    const selects = screen.getAllByRole('combobox');
    const pluginSelect = selects[0];
    expect(pluginSelect).toHaveValue('unstructured');
  });

  it('shows available plugins in dropdown', async () => {
    render(
      <NodeConfigEditor
        node={mockNode}
        onChange={vi.fn()}
      />
    );

    expect(screen.getByText('Text Parser')).toBeInTheDocument();
    expect(screen.getByText('Unstructured')).toBeInTheDocument();
  });

  it('calls onChange when plugin is changed', async () => {
    const user = userEvent.setup();
    const handleChange = vi.fn();

    render(
      <NodeConfigEditor
        node={mockNode}
        onChange={handleChange}
      />
    );

    // Get the first combobox which is the plugin selector
    const selects = screen.getAllByRole('combobox');
    const pluginSelect = selects[0];
    await user.selectOptions(pluginSelect, 'text');

    expect(handleChange).toHaveBeenCalledWith({
      ...mockNode,
      plugin_id: 'text',
      config: {}, // Config reset on plugin change
    });
  });

  it('renders config fields from schema', async () => {
    render(
      <NodeConfigEditor
        node={mockNode}
        onChange={vi.fn()}
      />
    );

    await waitFor(() => {
      expect(screen.getByText('Strategy')).toBeInTheDocument();
    });
  });

  it('calls onChange when config value changes', async () => {
    const user = userEvent.setup();
    const handleChange = vi.fn();

    render(
      <NodeConfigEditor
        node={mockNode}
        onChange={handleChange}
      />
    );

    await waitFor(() => {
      expect(screen.getByLabelText('Strategy')).toBeInTheDocument();
    });

    const strategySelect = screen.getByLabelText('Strategy');
    await user.selectOptions(strategySelect, 'hi_res');

    expect(handleChange).toHaveBeenCalledWith({
      ...mockNode,
      config: { strategy: 'hi_res' },
    });
  });

  it('shows loading state while fetching plugins', () => {
    (usePluginsModule.usePipelinePlugins as ReturnType<typeof vi.fn>).mockReturnValue({
      data: undefined,
      isLoading: true,
    });

    render(
      <NodeConfigEditor
        node={mockNode}
        onChange={vi.fn()}
      />
    );

    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it('disables inputs when readOnly is true', () => {
    render(
      <NodeConfigEditor
        node={mockNode}
        onChange={vi.fn()}
        readOnly={true}
      />
    );

    // Get the first combobox which is the plugin selector
    const selects = screen.getAllByRole('combobox');
    const pluginSelect = selects[0];
    expect(pluginSelect).toBeDisabled();
  });

  it('shows the selected plugin description when available', () => {
    render(<NodeConfigEditor node={mockNode} onChange={vi.fn()} />);
    expect(screen.getByText('Parse PDFs and docs')).toBeInTheDocument();
  });

  it('shows "Loading config..." while fetching schema', () => {
    (usePluginsModule.usePipelinePluginConfigSchema as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockSchema,
      isLoading: true,
    });

    render(<NodeConfigEditor node={mockNode} onChange={vi.fn()} />);
    expect(screen.getByText(/loading config/i)).toBeInTheDocument();
  });

  it('shows a "no configuration" message when schema is empty', () => {
    (usePluginsModule.usePipelinePluginConfigSchema as ReturnType<typeof vi.fn>).mockReturnValue({
      data: { type: 'object', properties: {} },
      isLoading: false,
    });

    render(<NodeConfigEditor node={mockNode} onChange={vi.fn()} />);
    expect(screen.getByText(/no configuration options/i)).toBeInTheDocument();
  });

  it('supports boolean, number, string, and x-model-selector fields', async () => {
    const user = userEvent.setup();
    const handleChange = vi.fn();

    (usePluginsModule.usePipelinePluginConfigSchema as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        type: 'object',
        properties: {
          enabled: { type: 'boolean', title: 'Enabled' },
          max_tokens: { type: 'integer', title: 'Max tokens', minimum: 1, maximum: 10 },
          note: { type: 'string', title: 'Note' },
          model: { type: 'string', title: 'Model', 'x-model-selector': true },
        },
      },
      isLoading: false,
    });

    (useModelManagerModule.useModelManagerModels as ReturnType<typeof vi.fn>).mockReturnValue({
      data: {
        models: [
          {
            id: 'm1',
            name: 'Model 1',
            model_type: 'embedding',
            description: '',
            memory_mb: {},
            is_installed: true,
            size_on_disk_mb: null,
            used_by_collections: [],
            active_download_task_id: null,
            active_delete_task_id: null,
            embedding_details: null,
            llm_details: null,
          },
        ],
        cache_size: null,
        hf_cache_scan_error: null,
      },
      isLoading: false,
    });

    const initialNode: PipelineNode = {
      ...mockNode,
      config: { enabled: false, max_tokens: undefined, note: 'hi', model: undefined },
    };

    function Harness() {
      const [node, setNode] = React.useState<PipelineNode>(initialNode);
      return (
        <NodeConfigEditor
          node={node}
          onChange={(updated) => {
            handleChange(updated);
            setNode(updated);
          }}
        />
      );
    }

    render(<Harness />);

    await user.click(screen.getByLabelText('Enabled'));
    expect(handleChange).toHaveBeenCalledWith(
      expect.objectContaining({ config: expect.objectContaining({ enabled: true }) })
    );

    await user.type(screen.getByLabelText('Max tokens'), '7');
    expect(handleChange).toHaveBeenCalledWith(
      expect.objectContaining({ config: expect.objectContaining({ max_tokens: 7 }) })
    );

    await user.clear(screen.getByLabelText('Note'));
    await user.type(screen.getByLabelText('Note'), 'ok');
    expect(handleChange.mock.calls.at(-1)?.[0].config.note).toBe('ok');

    await user.selectOptions(screen.getByLabelText('Model'), 'm1');
    expect(handleChange.mock.calls.at(-1)?.[0].config.model).toBe('m1');
  });
});
