// apps/webui-react/src/components/pipeline/__tests__/QuestionQueue.test.tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { QuestionQueue } from '../QuestionQueue';

describe('QuestionQueue', () => {
  const mockQuestions = [
    {
      id: 'q1',
      message: 'First question?',
      options: [{ id: 'a', label: 'Option A' }],
      allowCustom: true,
    },
    {
      id: 'q2',
      message: 'Second question?',
      options: [{ id: 'b', label: 'Option B' }],
      allowCustom: false,
    },
  ];

  it('renders nothing when queue is empty', () => {
    const { container } = render(
      <QuestionQueue questions={[]} onAnswer={vi.fn()} onDismiss={vi.fn()} />
    );

    expect(container.firstChild).toBeNull();
  });

  it('shows first question in queue', () => {
    render(
      <QuestionQueue questions={mockQuestions} onAnswer={vi.fn()} onDismiss={vi.fn()} />
    );

    expect(screen.getByText('First question?')).toBeInTheDocument();
    expect(screen.queryByText('Second question?')).not.toBeInTheDocument();
  });

  it('shows queue count when multiple questions', () => {
    render(
      <QuestionQueue questions={mockQuestions} onAnswer={vi.fn()} onDismiss={vi.fn()} />
    );

    expect(screen.getByText(/1 of 2/)).toBeInTheDocument();
  });

  it('calls onAnswer and removes from queue', async () => {
    const user = userEvent.setup();
    const onAnswer = vi.fn();
    render(
      <QuestionQueue questions={mockQuestions} onAnswer={onAnswer} onDismiss={vi.fn()} />
    );

    await user.click(screen.getByRole('button', { name: /Option A/i }));

    expect(onAnswer).toHaveBeenCalledWith('q1', 'a', undefined);
  });
});
