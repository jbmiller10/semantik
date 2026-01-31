// apps/webui-react/src/components/wizard/__tests__/StepProgressIndicator.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StepProgressIndicator } from '../StepProgressIndicator';
import type { WizardStep } from '../../../types/wizard';

const mockSteps: WizardStep[] = [
  { id: 'basics', label: 'Basics', isComplete: true },
  { id: 'mode', label: 'Mode', isComplete: false },
  { id: 'configure', label: 'Configure', isComplete: false },
];

describe('StepProgressIndicator', () => {
  it('renders all step labels', () => {
    render(<StepProgressIndicator steps={mockSteps} currentStep={1} />);

    expect(screen.getByText('Basics')).toBeInTheDocument();
    expect(screen.getByText('Mode')).toBeInTheDocument();
    expect(screen.getByText('Configure')).toBeInTheDocument();
  });

  it('shows checkmark for completed steps', () => {
    render(<StepProgressIndicator steps={mockSteps} currentStep={1} />);

    // Completed step should have check icon
    const completedStep = screen.getByText('Basics').closest('[data-step]');
    expect(completedStep).toHaveAttribute('data-complete', 'true');
  });

  it('highlights current step', () => {
    render(<StepProgressIndicator steps={mockSteps} currentStep={1} />);

    const currentStep = screen.getByText('Mode').closest('[data-step]');
    expect(currentStep).toHaveAttribute('data-active', 'true');
  });

  it('dims future steps', () => {
    render(<StepProgressIndicator steps={mockSteps} currentStep={0} />);

    const futureStep = screen.getByText('Configure').closest('[data-step]');
    expect(futureStep).toHaveAttribute('data-active', 'false');
    expect(futureStep).toHaveAttribute('data-complete', 'false');
  });
});
