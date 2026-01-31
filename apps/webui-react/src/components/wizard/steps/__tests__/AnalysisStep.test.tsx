// apps/webui-react/src/components/wizard/steps/__tests__/AnalysisStep.test.tsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { AnalysisStep } from '../AnalysisStep';
import type { PipelineDAG } from '../../../../types/pipeline';
import type { UseAgentStreamReturn } from '../../../../hooks/useAgentStream';
import type { QuestionEvent } from '../../../../types/agent';
import userEvent from '@testing-library/user-event';

const {
  mockSendMessage,
  mockDismissQuestion,
  mockResetStream,
  mockAnswerQuestion,
  streamState,
} = vi.hoisted(() => ({
  mockSendMessage: vi.fn(),
  mockDismissQuestion: vi.fn(),
  mockResetStream: vi.fn(),
  mockAnswerQuestion: vi.fn(),
  streamState: { current: {} as Partial<UseAgentStreamReturn> },
}));

// Mock useAgentStream
vi.mock('../../../../hooks/useAgentStream', () => ({
  useAgentStream: () => streamState.current,
}));

// Mock agent API
vi.mock('../../../../services/api/v2/agent', () => ({
  agentApi: {
    answerQuestion: (...args: unknown[]) => mockAnswerQuestion(...args),
  },
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
      status: { phase: 'analyzing', message: 'Analyzing source...' },
      activities: [],
      pendingQuestions: [],
      dismissQuestion: mockDismissQuestion,
      isStreaming: false,
      sendMessage: mockSendMessage,
      currentContent: '',
      pipeline: null,
      error: null,
      reset: mockResetStream,
    };
    mockAnswerQuestion.mockResolvedValue({ success: true });
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
    render(<AnalysisStep {...defaultProps} />);

    // Should show some loading/thinking state - look for the status message
    expect(screen.getByText('Analyzing source...')).toBeInTheDocument();
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

  it('converts pipeline updates into a basic DAG and calls onDagChange', async () => {
    const onDagChange = vi.fn();
    streamState.current.pipeline = {
      chunking_strategy: 'semantic',
      chunking_config: { chunk_size: 123 },
      embedding_model: 'm1',
    };

    render(<AnalysisStep {...defaultProps} onDagChange={onDagChange} />);

    await waitFor(() => {
      expect(onDagChange).toHaveBeenCalled();
    });

    expect(onDagChange).toHaveBeenCalledWith({
      id: 'agent-recommended',
      version: '1',
      nodes: [
        { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
        { id: 'chunker1', type: 'chunker', plugin_id: 'semantic', config: { chunk_size: 123 } },
        { id: 'embedder1', type: 'embedder', plugin_id: 'dense_local', config: { model: 'm1' } },
      ],
      edges: [
        { from_node: '_source', to_node: 'parser1', when: null },
        { from_node: 'parser1', to_node: 'chunker1', when: null },
        { from_node: 'chunker1', to_node: 'embedder1', when: null },
      ],
    });
  });

  it('submits answers for pending questions and dismisses on success', async () => {
    const user = userEvent.setup();
    const question: QuestionEvent = {
      id: 'q1',
      message: 'Pick',
      options: [{ id: 'o1', label: 'Yes' }],
      allowCustom: false,
    };
    streamState.current.pendingQuestions = [question];

    render(<AnalysisStep {...defaultProps} />);

    await user.click(screen.getByRole('button', { name: 'Yes' }));

    await waitFor(() => {
      expect(mockAnswerQuestion).toHaveBeenCalledWith('conv-123', 'q1', 'o1', undefined);
    });
    expect(mockDismissQuestion).toHaveBeenCalledWith('q1');
  });

  it('shows an error for failed answer submissions and allows retry', async () => {
    const user = userEvent.setup();
    mockAnswerQuestion.mockRejectedValueOnce(new Error('boom'));

    streamState.current.pendingQuestions = [
      {
        id: 'q1',
        message: 'Pick',
        options: [{ id: 'o1', label: 'Yes' }],
        allowCustom: false,
      },
    ];

    render(<AnalysisStep {...defaultProps} />);

    await user.click(screen.getByRole('button', { name: 'Yes' }));

    await waitFor(() => {
      expect(screen.getByText(/could not submit your answer/i)).toBeInTheDocument();
    });

    expect(mockDismissQuestion).not.toHaveBeenCalled();
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
