// apps/webui-react/src/components/wizard/steps/__tests__/AnalysisStep.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AnalysisStep } from '../AnalysisStep';
import type { PipelineDAG } from '../../../../types/pipeline';

// Mock useAgentStream
vi.mock('../../../../hooks/useAgentStream', () => ({
  useAgentStream: () => ({
    status: { phase: 'analyzing', message: 'Analyzing source...' },
    activities: [],
    pendingQuestions: [],
    dismissQuestion: vi.fn(),
    isStreaming: true,
    sendMessage: vi.fn(),
    currentContent: '',
    pipeline: null,
  }),
}));

// Mock agent API
vi.mock('../../../../services/api/v2/agent', () => ({
  agentApiV2: {
    answerQuestion: vi.fn(),
  },
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
});
