// apps/webui-react/src/components/pipeline/__tests__/AgentStatusBar.test.tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { AgentStatusBar } from '../AgentStatusBar';

describe('AgentStatusBar', () => {
  const defaultProps = {
    status: {
      phase: 'analyzing' as const,
      message: 'Analyzing file types...',
      progress: { current: 42, total: 247 },
    },
    activities: [
      { message: 'Started analysis', timestamp: '2026-01-25T10:00:00Z' },
    ],
    isStreaming: true,
    onSend: vi.fn(),
    onStop: vi.fn(),
  };

  it('renders status message and progress', () => {
    render(<AgentStatusBar {...defaultProps} />);

    expect(screen.getByText('Analyzing file types...')).toBeInTheDocument();
    expect(screen.getByText('(42/247)')).toBeInTheDocument();
  });

  it('shows pulsing indicator when streaming', () => {
    render(<AgentStatusBar {...defaultProps} />);

    const indicator = screen.getByTestId('status-indicator');
    expect(indicator).toHaveClass('animate-pulse');
  });

  it('shows stop button when streaming', () => {
    render(<AgentStatusBar {...defaultProps} />);

    expect(screen.getByRole('button', { name: /stop/i })).toBeInTheDocument();
  });

  it('calls onStop when stop button clicked', async () => {
    const user = userEvent.setup();
    const onStop = vi.fn();
    render(<AgentStatusBar {...defaultProps} onStop={onStop} />);

    await user.click(screen.getByRole('button', { name: /stop/i }));
    expect(onStop).toHaveBeenCalled();
  });

  it('toggles expanded state', async () => {
    const user = userEvent.setup();
    render(<AgentStatusBar {...defaultProps} />);

    // Initially collapsed - activity log not visible
    expect(screen.queryByText('Activity:')).not.toBeInTheDocument();

    // Click expand
    await user.click(screen.getByRole('button', { name: /expand/i }));

    // Now activity log visible
    expect(screen.getByText('Activity:')).toBeInTheDocument();
  });

  it('shows agent input when expanded', async () => {
    const user = userEvent.setup();
    render(<AgentStatusBar {...defaultProps} />);

    await user.click(screen.getByRole('button', { name: /expand/i }));

    expect(screen.getByPlaceholderText('Tell the agent something...')).toBeInTheDocument();
  });

  it('shows idle state correctly', () => {
    render(
      <AgentStatusBar
        {...defaultProps}
        status={{ phase: 'idle', message: 'Agent idle' }}
        isStreaming={false}
      />
    );

    expect(screen.getByText('Agent idle')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /stop/i })).not.toBeInTheDocument();
  });
});
