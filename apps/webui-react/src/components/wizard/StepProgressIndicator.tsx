// apps/webui-react/src/components/wizard/StepProgressIndicator.tsx
import { Check } from 'lucide-react';
import type { WizardStep } from '../../types/wizard';

interface StepProgressIndicatorProps {
  steps: WizardStep[];
  currentStep: number;
}

export function StepProgressIndicator({ steps, currentStep }: StepProgressIndicatorProps) {
  return (
    <div className="flex items-center justify-center gap-2 py-4">
      {steps.map((step, index) => {
        const isActive = index === currentStep;
        const isComplete = step.isComplete;
        const isFuture = index > currentStep && !isComplete;

        return (
          <div key={step.id} className="flex items-center">
            {/* Step indicator */}
            <div
              data-step={step.id}
              data-active={isActive.toString()}
              data-complete={isComplete.toString()}
              className={`
                flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium
                transition-all duration-200
                ${isActive
                  ? 'bg-gray-200 dark:bg-white/20 text-gray-900 dark:text-white'
                  : isComplete
                    ? 'bg-green-100 dark:bg-green-500/20 text-green-700 dark:text-green-400'
                    : 'text-[var(--text-muted)]'
                }
              `}
            >
              {isComplete ? (
                <Check className="w-4 h-4" />
              ) : (
                <span className={`
                  w-5 h-5 rounded-full flex items-center justify-center text-xs
                  ${isActive
                    ? 'bg-gray-800 dark:bg-white text-white dark:text-gray-900'
                    : 'border border-[var(--border)]'
                  }
                `}>
                  {index + 1}
                </span>
              )}
              <span className={isFuture ? 'opacity-50' : ''}>{step.label}</span>
            </div>

            {/* Connector line (except last) */}
            {index < steps.length - 1 && (
              <div className={`
                w-8 h-px mx-1
                ${index < currentStep ? 'bg-green-400' : 'bg-[var(--border)]'}
              `} />
            )}
          </div>
        );
      })}
    </div>
  );
}
