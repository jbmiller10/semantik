// apps/webui-react/src/components/wizard/__tests__/WizardStep.test.tsx
import { describe, it, expect } from 'vitest';
import type { WizardStep, WizardState, WizardFlow } from '../../../types/wizard';

describe('WizardStep types', () => {
  it('defines step structure correctly', () => {
    const step: WizardStep = {
      id: 'basics',
      label: 'Basics & Source',
      isComplete: false,
    };
    expect(step.id).toBe('basics');
  });

  it('defines wizard state correctly', () => {
    const state: WizardState = {
      currentStep: 0,
      flow: 'manual',
      steps: [
        { id: 'basics', label: 'Basics & Source', isComplete: false },
        { id: 'mode', label: 'Mode Selection', isComplete: false },
        { id: 'configure', label: 'Configure Pipeline', isComplete: false },
      ],
    };
    expect(state.steps.length).toBe(3);
    expect(state.flow).toBe('manual');
  });

  it('defines assisted flow with 4 steps', () => {
    const state: WizardState = {
      currentStep: 2,
      flow: 'assisted',
      steps: [
        { id: 'basics', label: 'Basics & Source', isComplete: true },
        { id: 'mode', label: 'Mode Selection', isComplete: true },
        { id: 'analysis', label: 'Agent Analysis', isComplete: false },
        { id: 'review', label: 'Review Pipeline', isComplete: false },
      ],
    };
    expect(state.steps.length).toBe(4);
    expect(state.flow).toBe('assisted');
  });
});
