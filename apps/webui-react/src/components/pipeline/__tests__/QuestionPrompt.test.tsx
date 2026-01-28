// apps/webui-react/src/components/pipeline/__tests__/QuestionPrompt.test.tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { QuestionPrompt } from '../QuestionPrompt';

describe('QuestionPrompt', () => {
  const defaultProps = {
    question: {
      id: 'q1',
      message: 'I found 23 scanned PDFs. These need OCR to be searchable.',
      options: [
        { id: 'ocr', label: 'Enable OCR', description: 'Slower but searchable' },
        { id: 'skip', label: 'Skip these files' },
        { id: 'manual', label: 'Let me decide per-file' },
      ],
      allowCustom: true,
    },
    onAnswer: vi.fn(),
    onDismiss: vi.fn(),
  };

  it('renders question message', () => {
    render(<QuestionPrompt {...defaultProps} />);

    expect(screen.getByText(/I found 23 scanned PDFs/)).toBeInTheDocument();
  });

  it('renders all options', () => {
    render(<QuestionPrompt {...defaultProps} />);

    expect(screen.getByRole('button', { name: /Enable OCR/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Skip these files/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Let me decide/i })).toBeInTheDocument();
  });

  it('calls onAnswer with option when clicked', async () => {
    const user = userEvent.setup();
    const onAnswer = vi.fn();
    render(<QuestionPrompt {...defaultProps} onAnswer={onAnswer} />);

    await user.click(screen.getByRole('button', { name: /Enable OCR/i }));

    expect(onAnswer).toHaveBeenCalledWith('q1', 'ocr', undefined);
  });

  it('shows custom input when allowCustom is true', () => {
    render(<QuestionPrompt {...defaultProps} />);

    expect(screen.getByPlaceholderText(/Or type a different instruction/i)).toBeInTheDocument();
  });

  it('hides custom input when allowCustom is false', () => {
    render(
      <QuestionPrompt
        {...defaultProps}
        question={{ ...defaultProps.question, allowCustom: false }}
      />
    );

    expect(screen.queryByPlaceholderText(/Or type a different/i)).not.toBeInTheDocument();
  });

  it('calls onAnswer with custom response', async () => {
    const user = userEvent.setup();
    const onAnswer = vi.fn();
    render(<QuestionPrompt {...defaultProps} onAnswer={onAnswer} />);

    const input = screen.getByPlaceholderText(/Or type a different instruction/i);
    await user.type(input, 'Use a different approach');
    await user.click(screen.getByRole('button', { name: /send/i }));

    expect(onAnswer).toHaveBeenCalledWith('q1', undefined, 'Use a different approach');
  });

  it('calls onDismiss when close button clicked', async () => {
    const user = userEvent.setup();
    const onDismiss = vi.fn();
    render(<QuestionPrompt {...defaultProps} onDismiss={onDismiss} />);

    await user.click(screen.getByRole('button', { name: /close/i }));

    expect(onDismiss).toHaveBeenCalledWith('q1');
  });
});
