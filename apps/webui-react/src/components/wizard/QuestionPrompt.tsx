/**
 * QuestionPrompt component for displaying and collecting answers to
 * questions from the AskUserQuestion tool in assisted flow.
 */

import { useState, useCallback } from 'react';
import type { QuestionEventData, QuestionItem } from '../../types/assisted-flow';

interface QuestionPromptProps {
  /** Question data from the question SSE event */
  question: QuestionEventData;
  /** Callback when user submits answers */
  onSubmit: (answers: Record<string, string>) => void;
  /** Whether submission is in progress */
  isSubmitting?: boolean;
}

/**
 * Renders a single question with radio buttons or checkboxes.
 */
function QuestionField({
  question,
  selectedValues,
  onSelect,
}: {
  question: QuestionItem;
  selectedValues: string[];
  onSelect: (questionText: string, values: string[]) => void;
}) {
  const handleChange = (optionLabel: string, checked: boolean) => {
    if (question.multiSelect) {
      // Checkbox behavior - toggle the option
      const newValues = checked
        ? [...selectedValues, optionLabel]
        : selectedValues.filter((v) => v !== optionLabel);
      onSelect(question.question, newValues);
    } else {
      // Radio behavior - replace selection
      onSelect(question.question, checked ? [optionLabel] : []);
    }
  };

  return (
    <div className="space-y-3">
      {/* Question header */}
      <div className="flex items-center gap-2">
        <span className="px-2 py-0.5 text-xs font-medium rounded bg-[var(--bg-tertiary)] text-[var(--text-secondary)]">
          {question.header}
        </span>
      </div>

      {/* Question text */}
      <p className="text-sm font-medium text-[var(--text-primary)]">{question.question}</p>

      {/* Options */}
      <div className="space-y-2">
        {question.options.map((option) => {
          const isSelected = selectedValues.includes(option.label);
          const inputType = question.multiSelect ? 'checkbox' : 'radio';
          const inputId = `${question.question}-${option.label}`;

          return (
            <label
              key={option.label}
              htmlFor={inputId}
              className={`
                flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors
                ${
                  isSelected
                    ? 'border-gray-400 dark:border-white bg-gray-100 dark:bg-white/10'
                    : 'border-[var(--border)] hover:border-gray-300 dark:hover:border-gray-600'
                }
              `}
            >
              <input
                type={inputType}
                id={inputId}
                name={question.question}
                checked={isSelected}
                onChange={(e) => handleChange(option.label, e.target.checked)}
                className="mt-0.5"
              />
              <div className="flex-1 min-w-0">
                <div
                  className={`text-sm font-medium ${isSelected ? 'text-gray-800 dark:text-white' : 'text-[var(--text-primary)]'}`}
                >
                  {option.label}
                </div>
                {option.description && (
                  <div className="text-xs text-[var(--text-muted)] mt-0.5">{option.description}</div>
                )}
              </div>
            </label>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Main question prompt component.
 * Displays all questions and collects answers before submission.
 */
export function QuestionPrompt({ question, onSubmit, isSubmitting = false }: QuestionPromptProps) {
  // Track selected values for each question (by question text)
  const [selections, setSelections] = useState<Record<string, string[]>>({});

  const handleSelect = useCallback((questionText: string, values: string[]) => {
    setSelections((prev) => ({
      ...prev,
      [questionText]: values,
    }));
  }, []);

  const handleSubmit = useCallback(() => {
    // Convert selections to the format expected by the API
    // For single-select, just use the first value
    // For multi-select, join with comma (or use first value)
    const answers: Record<string, string> = {};
    for (const q of question.questions) {
      const selected = selections[q.question] || [];
      answers[q.question] = selected[0] || '';
    }
    onSubmit(answers);
  }, [selections, question.questions, onSubmit]);

  // Check if all required questions have at least one selection
  const allAnswered = question.questions.every((q) => {
    const selected = selections[q.question] || [];
    return selected.length > 0;
  });

  return (
    <div className="p-4 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 rounded-full bg-amber-500/20 flex items-center justify-center">
          <svg
            className="w-4 h-4 text-amber-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        </div>
        <div>
          <h3 className="text-sm font-medium text-[var(--text-primary)]">Question from Assistant</h3>
          <p className="text-xs text-[var(--text-muted)]">Please select your preferences</p>
        </div>
      </div>

      {/* Questions */}
      <div className="space-y-6">
        {question.questions.map((q, index) => (
          <QuestionField
            key={`${q.question}-${index}`}
            question={q}
            selectedValues={selections[q.question] || []}
            onSelect={handleSelect}
          />
        ))}
      </div>

      {/* Submit button */}
      <div className="mt-6 flex justify-end">
        <button
          onClick={handleSubmit}
          disabled={!allAnswered || isSubmitting}
          className={`
            px-4 py-2 text-sm font-medium rounded-lg transition-colors
            ${
              allAnswered && !isSubmitting
                ? 'bg-gray-200 dark:bg-white text-gray-900 hover:bg-gray-300 dark:hover:bg-gray-100'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-400 dark:text-gray-500 cursor-not-allowed'
            }
          `}
        >
          {isSubmitting ? 'Submitting...' : 'Submit Answer'}
        </button>
      </div>
    </div>
  );
}

export default QuestionPrompt;
