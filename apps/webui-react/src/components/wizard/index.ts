// apps/webui-react/src/components/wizard/index.ts
export { CollectionWizard } from './CollectionWizard';
export { StepProgressIndicator } from './StepProgressIndicator';
export { BasicsStep } from './steps/BasicsStep';
export { ConfigureStep } from './steps/ConfigureStep';
export type { WizardStep, WizardState, WizardFlow, StepId } from '../../types/wizard';
export { MANUAL_STEPS, ASSISTED_STEPS, getInitialWizardState } from '../../types/wizard';
