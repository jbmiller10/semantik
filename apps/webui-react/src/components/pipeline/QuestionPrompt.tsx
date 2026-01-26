// apps/webui-react/src/components/pipeline/QuestionPrompt.tsx
import { useState, useCallback, useEffect, type KeyboardEvent } from 'react';
import { X, Send, HelpCircle } from 'lucide-react';
import type { QuestionEvent } from '@/types/agent';

export interface QuestionPromptProps {
  question: QuestionEvent;
  onAnswer: (questionId: string, optionId?: string, customResponse?: string) => void;
  onDismiss: (questionId: string) => void;
}

export function QuestionPrompt({ question, onAnswer, onDismiss }: QuestionPromptProps) {
  const [customInput, setCustomInput] = useState('');

  const handleOptionClick = useCallback(
    (optionId: string) => {
      onAnswer(question.id, optionId, undefined);
    },
    [question.id, onAnswer]
  );

  const handleCustomSubmit = useCallback(() => {
    const trimmed = customInput.trim();
    if (trimmed) {
      onAnswer(question.id, undefined, trimmed);
      setCustomInput('');
    }
  }, [question.id, customInput, onAnswer]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleCustomSubmit();
      }
    },
    [handleCustomSubmit]
  );

  // Handle Escape key to dismiss
  useEffect(() => {
    const handleGlobalKeyDown = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape') {
        onDismiss(question.id);
      }
    };

    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, [question.id, onDismiss]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div
        className="w-full max-w-lg mx-4 bg-[var(--bg-primary)] border border-[var(--border)]
                   rounded-lg shadow-xl overflow-hidden"
        role="dialog"
        aria-labelledby="question-title"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)] bg-[var(--bg-secondary)]">
          <div className="flex items-center gap-2">
            <HelpCircle className="w-5 h-5 text-amber-500" />
            <h2 id="question-title" className="text-sm font-medium text-[var(--text-primary)]">
              Agent Question
            </h2>
          </div>
          <button
            onClick={() => onDismiss(question.id)}
            aria-label="Close"
            className="p-1 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)]
                       focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-4">
          {/* Question message */}
          <p className="text-[var(--text-primary)]">{question.message}</p>

          {/* Options */}
          <div className="flex flex-wrap gap-2">
            {question.options.map((option) => (
              <button
                key={option.id}
                onClick={() => handleOptionClick(option.id)}
                className="px-4 py-2 rounded-md bg-[var(--bg-tertiary)] border border-[var(--border)]
                           text-[var(--text-primary)] hover:bg-[var(--bg-secondary)]
                           focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white
                           transition-colors text-sm"
                title={option.description}
              >
                {option.label}
                {option.description && (
                  <span className="block text-xs text-[var(--text-muted)] mt-0.5">
                    {option.description}
                  </span>
                )}
              </button>
            ))}
          </div>

          {/* Custom input */}
          {question.allowCustom && (
            <div className="flex gap-2 pt-2 border-t border-[var(--border-subtle)]">
              <input
                type="text"
                value={customInput}
                onChange={(e) => setCustomInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Or type a different instruction..."
                className="flex-1 px-3 py-2 rounded-md border border-[var(--border)]
                           bg-[var(--bg-secondary)] text-[var(--text-primary)]
                           placeholder:text-[var(--text-muted)]
                           focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white
                           text-sm"
              />
              <button
                onClick={handleCustomSubmit}
                disabled={!customInput.trim()}
                aria-label="Send"
                className="px-3 py-2 rounded-md bg-[var(--bg-tertiary)] border border-[var(--border)]
                           text-[var(--text-primary)] hover:bg-[var(--bg-secondary)]
                           focus:outline-none focus:ring-2 focus:ring-gray-400 dark:focus:ring-white
                           disabled:opacity-50 disabled:cursor-not-allowed
                           transition-colors"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
