// apps/webui-react/src/components/wizard/steps/__tests__/AnalysisStep.test.tsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { AnalysisStep } from '../AnalysisStep';
import type { PipelineDAG } from '../../../../types/pipeline';
import userEvent from '@testing-library/user-event';
import type { UseAssistedFlowStreamReturn } from '../../../../hooks/useAssistedFlowStream';

const {
  mockSendMessage,
  mockResetStream,
  streamState,
} = vi.hoisted(() => ({
  mockSendMessage: vi.fn(),
  mockResetStream: vi.fn(),
  streamState: { current: {} as Partial<UseAssistedFlowStreamReturn> },
}));

// Mock useAssistedFlowStream
vi.mock('../../../../hooks/useAssistedFlowStream', () => ({
  useAssistedFlowStream: () => streamState.current,
}));

vi.mock('../../../pipeline', () => ({
  PipelineVisualization: () => <div data-testid="mock-pipeline-visualization" />,
  ConfigurationPanel: (props: { readOnly: boolean }) => (
    <div data-testid="mock-configuration-panel" data-readonly={props.readOnly ? 'true' : 'false'} />
  ),
}));

describe('AnalysisStep', () => {
  const mockDag: PipelineDAG = {
    id: 'test',
    version: '1',
    nodes: [],
    edges: [],
  };

  const defaultProps = {
    conversationId: 'conv-123',
    dag: mockDag,
    onDagChange: vi.fn(),
    onAgentComplete: vi.fn(),
    onSwitchToManual: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    streamState.current = {
      isStreaming: false,
      sendMessage: mockSendMessage,
      currentContent: '',
      toolCalls: [],
      error: null,
      reset: mockResetStream,
    };
  });

  it('renders two-column layout', () => {
    render(<AnalysisStep {...defaultProps} />);

    // Should have agent column and preview column
    expect(screen.getByTestId('agent-column')).toBeInTheDocument();
    expect(screen.getByTestId('preview-column')).toBeInTheDocument();
  });

  it('shows skip to manual option', () => {
    render(<AnalysisStep {...defaultProps} />);

    expect(screen.getByText(/skip to manual/i)).toBeInTheDocument();
  });

  it('renders agent thinking indicator initially', () => {
    // Show "Analyzing..." status when streaming
    streamState.current.isStreaming = true;

    render(<AnalysisStep {...defaultProps} />);

    expect(screen.getByText('Analyzing...')).toBeInTheDocument();
  });

  it('auto-starts analysis on mount and updates summary when content is present', async () => {
    const onSummaryChange = vi.fn();
    streamState.current.currentContent = 'Summary text';

    render(<AnalysisStep {...defaultProps} onSummaryChange={onSummaryChange} />);

    await waitFor(() => {
      expect(mockSendMessage).toHaveBeenCalledWith('Analyze my source and recommend a pipeline configuration.');
    });
    expect(onSummaryChange).toHaveBeenCalledWith('Summary text');
  });

  it('shows running tool status when a tool is executing', () => {
    streamState.current.isStreaming = true;
    streamState.current.toolCalls = [
      { id: '1', tool_name: 'list_plugins', status: 'running' },
    ];

    render(<AnalysisStep {...defaultProps} />);

    expect(screen.getByText('Running list_plugins...')).toBeInTheDocument();
    expect(screen.getByText('list_plugins')).toBeInTheDocument();
  });

  it('updates DAG when build_pipeline tool_result includes a pipeline', async () => {
    const onDagChange = vi.fn();
    const pipeline: PipelineDAG = {
      id: 'agent-recommended',
      version: '1',
      nodes: [{ id: 'parser1', type: 'parser', plugin_id: 'text', config: {} }],
      edges: [{ from_node: '_source', to_node: 'parser1', when: null }],
    };

    streamState.current.toolCalls = [
      {
        id: 'tool-1',
        tool_name: 'mcp__assisted-flow__build_pipeline',
        status: 'success',
        result: [{ type: 'text', text: JSON.stringify({ success: true, pipeline }) }],
      },
    ];

    render(<AnalysisStep {...defaultProps} dag={mockDag} onDagChange={onDagChange} />);

    await waitFor(() => {
      expect(onDagChange).toHaveBeenCalledWith(pipeline);
    });
  });

  it('resets the stream and re-triggers auto-start when clicking "Try again"', async () => {
    const user = userEvent.setup();
    streamState.current.error = 'Bad';

    render(<AnalysisStep {...defaultProps} />);

    await waitFor(() => {
      expect(mockSendMessage).toHaveBeenCalledTimes(1);
    });

    await user.click(screen.getByRole('button', { name: /try again/i }));

    expect(mockResetStream).toHaveBeenCalled();

    await waitFor(() => {
      expect(mockSendMessage).toHaveBeenCalledTimes(2);
    });
  });
});
