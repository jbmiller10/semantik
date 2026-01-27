// apps/webui-react/src/components/wizard/CollectionWizard.tsx
import { useState, useCallback, useEffect } from 'react';
import { X } from 'lucide-react';
import { StepProgressIndicator } from './StepProgressIndicator';
import { getInitialWizardState, MANUAL_STEPS, ASSISTED_STEPS } from '../../types/wizard';
import type { WizardState, WizardFlow } from '../../types/wizard';

interface CollectionWizardProps {
  onClose: () => void;
  onSuccess: () => void;
}

export function CollectionWizard({ onClose, onSuccess }: CollectionWizardProps) {
  const [wizardState, setWizardState] = useState<WizardState>(getInitialWizardState('manual'));

  // Form data for step 1 (basics)
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');

  // Determine modal size based on current step
  // Steps 0-1: compact, Step 2+: expanded
  const isExpanded = wizardState.currentStep >= 2;

  // Modal size classes
  const sizeClasses = isExpanded
    ? 'w-[90vw] max-w-7xl h-[85vh]'
    : 'w-full max-w-2xl max-h-[90vh]';

  const handleClose = useCallback(() => {
    onClose();
  }, [onClose]);

  const handleNext = useCallback(() => {
    setWizardState(prev => {
      const newSteps = [...prev.steps];
      newSteps[prev.currentStep] = { ...newSteps[prev.currentStep], isComplete: true };
      return {
        ...prev,
        currentStep: Math.min(prev.currentStep + 1, prev.steps.length - 1),
        steps: newSteps,
      };
    });
  }, []);

  const handleBack = useCallback(() => {
    setWizardState(prev => ({
      ...prev,
      currentStep: Math.max(prev.currentStep - 1, 0),
    }));
  }, []);

  const handleFlowChange = useCallback((flow: WizardFlow) => {
    setWizardState(prev => ({
      ...prev,
      flow,
      steps: flow === 'manual' ? [...MANUAL_STEPS] : [...ASSISTED_STEPS],
    }));
  }, []);

  // Keyboard: Escape to close
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        handleClose();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleClose]);

  return (
    <div className="fixed inset-0 bg-black/50 dark:bg-black/80 flex items-center justify-center p-4 z-50">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="wizard-title"
        className={`
          panel relative rounded-2xl shadow-2xl
          flex flex-col overflow-hidden
          transition-all duration-300 ease-out
          ${sizeClasses}
        `}
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-[var(--border)] bg-[var(--bg-secondary)]">
          <div className="flex items-center justify-between">
            <h2 id="wizard-title" className="text-lg font-semibold text-[var(--text-primary)]">
              Create Collection
            </h2>
            <button
              onClick={handleClose}
              aria-label="Close"
              className="p-2 rounded-lg hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)]"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Step indicator */}
          <StepProgressIndicator
            steps={wizardState.steps}
            currentStep={wizardState.currentStep}
          />
        </div>

        {/* Content area */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {/* Step 1: Basics & Source */}
          {wizardState.currentStep === 0 && (
            <div className="space-y-4">
              <div>
                <label htmlFor="name" className="block text-sm font-medium text-[var(--text-secondary)] mb-1">
                  Collection Name
                </label>
                <input
                  id="name"
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  aria-label="Collection name"
                  className="w-full px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] text-[var(--text-primary)]"
                  placeholder="My Collection"
                />
              </div>
              <div>
                <label htmlFor="description" className="block text-sm font-medium text-[var(--text-secondary)] mb-1">
                  Description (optional)
                </label>
                <textarea
                  id="description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] text-[var(--text-primary)] resize-none"
                  rows={3}
                  placeholder="Describe what this collection contains..."
                />
              </div>
              {/* Source configuration will be added here */}
              <p className="text-sm text-[var(--text-muted)]">
                Source configuration coming in next task...
              </p>
            </div>
          )}

          {/* Step 2: Mode Selection */}
          {wizardState.currentStep === 1 && (
            <div className="space-y-4">
              <p className="text-sm text-[var(--text-secondary)]">
                Choose how you want to configure your pipeline:
              </p>
              <div className="grid grid-cols-2 gap-4">
                <button
                  onClick={() => handleFlowChange('assisted')}
                  className={`
                    p-4 rounded-lg border-2 text-left transition-all
                    ${wizardState.flow === 'assisted'
                      ? 'border-gray-400 dark:border-white bg-gray-100 dark:bg-white/10'
                      : 'border-[var(--border)] hover:border-gray-300 dark:hover:border-gray-600'
                    }
                  `}
                >
                  <span className="text-lg">Assisted</span>
                  <p className="text-sm text-[var(--text-muted)] mt-1">
                    AI analyzes your source and recommends optimal settings
                  </p>
                </button>
                <button
                  onClick={() => handleFlowChange('manual')}
                  className={`
                    p-4 rounded-lg border-2 text-left transition-all
                    ${wizardState.flow === 'manual'
                      ? 'border-gray-400 dark:border-white bg-gray-100 dark:bg-white/10'
                      : 'border-[var(--border)] hover:border-gray-300 dark:hover:border-gray-600'
                    }
                  `}
                >
                  <span className="text-lg">Manual</span>
                  <p className="text-sm text-[var(--text-muted)] mt-1">
                    Configure pipeline settings yourself
                  </p>
                </button>
              </div>
            </div>
          )}

          {/* Step 3: Configure (manual) or Analysis (assisted) */}
          {wizardState.currentStep === 2 && (
            <div className="h-full">
              {wizardState.flow === 'manual' ? (
                <div className="text-center py-12">
                  <p className="text-[var(--text-muted)]">
                    DAG Editor will be integrated here in Task 5
                  </p>
                </div>
              ) : (
                <div className="text-center py-12">
                  <p className="text-[var(--text-muted)]">
                    Agent Analysis UI (Phase 2)
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Step 4: Review (assisted only) */}
          {wizardState.currentStep === 3 && wizardState.flow === 'assisted' && (
            <div className="h-full">
              <div className="text-center py-12">
                <p className="text-[var(--text-muted)]">
                  Review Pipeline UI (Phase 2)
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-[var(--border)] bg-[var(--bg-secondary)] flex justify-between">
          <button
            onClick={wizardState.currentStep === 0 ? handleClose : handleBack}
            className="px-4 py-2 rounded-lg border border-[var(--border)] hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)]"
          >
            {wizardState.currentStep === 0 ? 'Cancel' : 'Back'}
          </button>

          <button
            onClick={handleNext}
            disabled={wizardState.currentStep === 0 && !name.trim()}
            className="px-4 py-2 rounded-lg bg-gray-200 dark:bg-white text-gray-900 dark:text-gray-900 font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {wizardState.currentStep === wizardState.steps.length - 1 ? 'Create Collection' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
}
