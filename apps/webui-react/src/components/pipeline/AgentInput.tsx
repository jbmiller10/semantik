// apps/webui-react/src/components/pipeline/AgentInput.tsx
import { useState, useCallback, type KeyboardEvent } from 'react';
import { Send } from 'lucide-react';

export interface AgentInputProps {
  onSend: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export function AgentInput({
  onSend,
  disabled = false,
  placeholder = 'Tell the agent something...'
}: AgentInputProps) {
  const [value, setValue] = useState('');

  const handleSubmit = useCallback(() => {
    const trimmed = value.trim();
    if (trimmed && !disabled) {
      onSend(trimmed);
      setValue('');
    }
  }, [value, disabled, onSend]);

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }, [handleSubmit]);

  return (
    <div className="flex gap-2">
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        className="flex-1 px-3 py-2 rounded-md border border-[var(--border)]
                   bg-[var(--bg-secondary)] text-[var(--text-primary)]
                   placeholder:text-[var(--text-muted)]
                   focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white
                   disabled:opacity-50 disabled:cursor-not-allowed"
      />
      <button
        onClick={handleSubmit}
        disabled={disabled || !value.trim()}
        aria-label="Send"
        className="px-4 py-2 rounded-md bg-[var(--bg-tertiary)] border border-[var(--border)]
                   text-[var(--text-primary)] hover:bg-[var(--bg-secondary)]
                   focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white
                   disabled:opacity-50 disabled:cursor-not-allowed
                   transition-colors"
      >
        <Send className="w-4 h-4" />
      </button>
    </div>
  );
}
