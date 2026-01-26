// apps/webui-react/src/components/pipeline/__tests__/PipelineBuilderFooter.test.tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { PipelineBuilderFooter } from '../PipelineBuilderFooter';

describe('PipelineBuilderFooter', () => {
  const defaultProps = {
    isReady: true,
    fileCount: 247,
    nodeCount: 3,
    onValidate: vi.fn(),
    onApply: vi.fn(),
    isValidating: false,
    isApplying: false,
  };

  it('renders summary stats', () => {
    render(<PipelineBuilderFooter {...defaultProps} />);

    expect(screen.getByText(/247 files/)).toBeInTheDocument();
    expect(screen.getByText(/3 nodes/)).toBeInTheDocument();
  });

  it('shows ready status when ready', () => {
    render(<PipelineBuilderFooter {...defaultProps} isReady={true} />);

    expect(screen.getByText(/Pipeline ready/)).toBeInTheDocument();
  });

  it('shows not ready status when not ready', () => {
    render(<PipelineBuilderFooter {...defaultProps} isReady={false} />);

    expect(screen.getByText(/Pipeline incomplete/)).toBeInTheDocument();
  });

  it('calls onValidate when validate clicked', async () => {
    const user = userEvent.setup();
    const onValidate = vi.fn();
    render(<PipelineBuilderFooter {...defaultProps} onValidate={onValidate} />);

    await user.click(screen.getByRole('button', { name: /validate/i }));

    expect(onValidate).toHaveBeenCalled();
  });

  it('calls onApply when apply clicked', async () => {
    const user = userEvent.setup();
    const onApply = vi.fn();
    render(<PipelineBuilderFooter {...defaultProps} onApply={onApply} />);

    await user.click(screen.getByRole('button', { name: /apply/i }));

    expect(onApply).toHaveBeenCalled();
  });

  it('disables apply when not ready', () => {
    render(<PipelineBuilderFooter {...defaultProps} isReady={false} />);

    expect(screen.getByRole('button', { name: /apply/i })).toBeDisabled();
  });

  it('shows loading state when validating', () => {
    render(<PipelineBuilderFooter {...defaultProps} isValidating={true} />);

    expect(screen.getByText(/Validating/)).toBeInTheDocument();
  });
});
