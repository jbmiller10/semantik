// apps/webui-react/src/components/pipeline/__tests__/AgentInput.test.tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import { AgentInput } from '../AgentInput';

describe('AgentInput', () => {
  it('renders input and send button', () => {
    render(<AgentInput onSend={vi.fn()} />);

    expect(screen.getByPlaceholderText('Tell the agent something...')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
  });

  it('calls onSend with input value when submitted', async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<AgentInput onSend={onSend} />);

    const input = screen.getByPlaceholderText('Tell the agent something...');
    await user.type(input, 'Use semantic chunking');
    await user.click(screen.getByRole('button', { name: /send/i }));

    expect(onSend).toHaveBeenCalledWith('Use semantic chunking');
  });

  it('clears input after send', async () => {
    const user = userEvent.setup();
    render(<AgentInput onSend={vi.fn()} />);

    const input = screen.getByPlaceholderText('Tell the agent something...') as HTMLInputElement;
    await user.type(input, 'Test message');
    await user.click(screen.getByRole('button', { name: /send/i }));

    expect(input.value).toBe('');
  });

  it('disables send button when disabled prop is true', () => {
    render(<AgentInput onSend={vi.fn()} disabled />);

    expect(screen.getByRole('button', { name: /send/i })).toBeDisabled();
  });

  it('submits on Enter key', async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<AgentInput onSend={onSend} />);

    const input = screen.getByPlaceholderText('Tell the agent something...');
    await user.type(input, 'Test message{enter}');

    expect(onSend).toHaveBeenCalledWith('Test message');
  });
});
