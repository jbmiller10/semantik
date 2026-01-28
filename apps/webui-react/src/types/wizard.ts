// apps/webui-react/src/types/wizard.ts

/** Flow type determines step sequence */
export type WizardFlow = 'manual' | 'assisted';

/** Individual wizard step */
export interface WizardStep {
  id: string;
  label: string;
  isComplete: boolean;
}

/** Complete wizard state */
export interface WizardState {
  currentStep: number;
  flow: WizardFlow;
  steps: WizardStep[];
}

/** Step IDs for type safety */
export type StepId = 'basics' | 'mode' | 'configure' | 'analysis' | 'review';

/** Steps for manual flow */
export const MANUAL_STEPS: WizardStep[] = [
  { id: 'basics', label: 'Basics & Source', isComplete: false },
  { id: 'mode', label: 'Mode Selection', isComplete: false },
  { id: 'configure', label: 'Configure Pipeline', isComplete: false },
];

/** Steps for assisted flow */
export const ASSISTED_STEPS: WizardStep[] = [
  { id: 'basics', label: 'Basics & Source', isComplete: false },
  { id: 'mode', label: 'Mode Selection', isComplete: false },
  { id: 'analysis', label: 'Agent Analysis', isComplete: false },
  { id: 'review', label: 'Review Pipeline', isComplete: false },
];

/** Get initial wizard state for a flow */
export function getInitialWizardState(flow: WizardFlow = 'manual'): WizardState {
  return {
    currentStep: 0,
    flow,
    steps: flow === 'manual' ? [...MANUAL_STEPS] : [...ASSISTED_STEPS],
  };
}
