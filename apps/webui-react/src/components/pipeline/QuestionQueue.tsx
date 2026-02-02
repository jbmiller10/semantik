// apps/webui-react/src/components/pipeline/QuestionQueue.tsx
import { QuestionPrompt } from './QuestionPrompt';
import type { QuestionEvent } from '@/types/agent';

export interface QuestionQueueProps {
  questions: QuestionEvent[];
  onAnswer: (questionId: string, optionId?: string, customResponse?: string) => void;
  onDismiss: (questionId: string) => void;
}

export function QuestionQueue({ questions, onAnswer, onDismiss }: QuestionQueueProps) {
  if (questions.length === 0) {
    return null;
  }

  const currentQuestion = questions[0];
  const queueCount = questions.length;

  return (
    <div className="relative">
      {/* Queue indicator */}
      {queueCount > 1 && (
        <div
          className="absolute -top-2 -right-2 z-[51] px-2 py-0.5 rounded-full
                     bg-amber-500 text-white text-xs font-medium"
        >
          1 of {queueCount}
        </div>
      )}

      <QuestionPrompt
        question={currentQuestion}
        onAnswer={onAnswer}
        onDismiss={onDismiss}
      />
    </div>
  );
}
