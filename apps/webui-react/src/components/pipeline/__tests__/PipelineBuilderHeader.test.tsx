// apps/webui-react/src/components/pipeline/__tests__/PipelineBuilderHeader.test.tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { PipelineBuilderHeader } from '../PipelineBuilderHeader';

describe('PipelineBuilderHeader', () => {
  const defaultProps = {
    sourceName: '/home/john/documents',
    sourceType: 'directory',
    mode: 'assisted' as const,
    onModeChange: vi.fn(),
    onClose: vi.fn(),
  };

  it('renders title and source info', () => {
    render(<PipelineBuilderHeader {...defaultProps} />);

    expect(screen.getByText('Pipeline Builder')).toBeInTheDocument();
    expect(screen.getByText('/home/john/documents')).toBeInTheDocument();
  });

  it('shows assisted mode as active', () => {
    render(<PipelineBuilderHeader {...defaultProps} mode="assisted" />);

    const assistedButton = screen.getByRole('button', { name: /assisted/i });
    expect(assistedButton).toHaveAttribute('aria-pressed', 'true');
  });

  it('shows manual mode as active', () => {
    render(<PipelineBuilderHeader {...defaultProps} mode="manual" />);

    const manualButton = screen.getByRole('button', { name: /manual/i });
    expect(manualButton).toHaveAttribute('aria-pressed', 'true');
  });

  it('calls onModeChange when toggling', async () => {
    const user = userEvent.setup();
    const onModeChange = vi.fn();
    render(<PipelineBuilderHeader {...defaultProps} mode="assisted" onModeChange={onModeChange} />);

    await user.click(screen.getByRole('button', { name: /manual/i }));

    expect(onModeChange).toHaveBeenCalledWith('manual');
  });

  it('calls onClose when close button clicked', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<PipelineBuilderHeader {...defaultProps} onClose={onClose} />);

    await user.click(screen.getByRole('button', { name: /close/i }));

    expect(onClose).toHaveBeenCalled();
  });
});
