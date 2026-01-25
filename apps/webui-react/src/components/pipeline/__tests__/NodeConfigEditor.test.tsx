import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import { NodeConfigEditor } from '../NodeConfigEditor';
import type { PipelineNode } from '@/types/pipeline';
import * as usePluginsModule from '@/hooks/usePlugins';

// Mock the usePlugins hook
vi.mock('@/hooks/usePlugins', () => ({
  usePlugins: vi.fn(),
  usePluginConfigSchema: vi.fn(),
}));

const mockPlugins = [
  {
    id: 'text',
    type: 'parser',
    version: '1.0.0',
    manifest: { id: 'text', type: 'parser', version: '1.0.0', display_name: 'Text Parser', description: 'Parse plain text', requires: [] },
    enabled: true,
    config: {},
  },
  {
    id: 'unstructured',
    type: 'parser',
    version: '1.0.0',
    manifest: { id: 'unstructured', type: 'parser', version: '1.0.0', display_name: 'Unstructured', description: 'Parse PDFs and docs', requires: [] },
    enabled: true,
    config: {},
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
    (usePluginsModule.usePlugins as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockPlugins,
      isLoading: false,
    });
    (usePluginsModule.usePluginConfigSchema as ReturnType<typeof vi.fn>).mockReturnValue({
      data: mockSchema,
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
    (usePluginsModule.usePlugins as ReturnType<typeof vi.fn>).mockReturnValue({
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
});
