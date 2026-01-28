import { useEffect, useCallback, useRef } from 'react';

interface UseWizardKeyboardOptions {
  onClose: () => void;
  onNext: () => void;
  onBack: () => void;
  canAdvance: boolean;
  isSubmitting: boolean;
}

export function useWizardKeyboard({
  onClose,
  onNext,
  onBack,
  canAdvance,
  isSubmitting,
}: UseWizardKeyboardOptions) {
  const modalRef = useRef<HTMLDivElement>(null);

  // Focus trap
  const handleTabKey = useCallback((e: KeyboardEvent) => {
    if (e.key !== 'Tab' || !modalRef.current) return;

    const focusableElements = modalRef.current.querySelectorAll(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    );

    if (focusableElements.length === 0) return;

    const firstElement = focusableElements[0] as HTMLElement;
    const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

    if (e.shiftKey) {
      if (document.activeElement === firstElement) {
        e.preventDefault();
        lastElement.focus();
      }
    } else {
      if (document.activeElement === lastElement) {
        e.preventDefault();
        firstElement.focus();
      }
    }
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Escape to close
      if (e.key === 'Escape' && !isSubmitting) {
        e.preventDefault();
        onClose();
        return;
      }

      // Cmd/Ctrl+Enter to advance
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && canAdvance && !isSubmitting) {
        e.preventDefault();
        onNext();
        return;
      }

      // Cmd/Ctrl+Backspace to go back
      if ((e.metaKey || e.ctrlKey) && e.key === 'Backspace' && !isSubmitting) {
        e.preventDefault();
        onBack();
        return;
      }

      // Tab key for focus trap
      handleTabKey(e);
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose, onNext, onBack, canAdvance, isSubmitting, handleTabKey]);

  // Auto-focus first input on mount
  useEffect(() => {
    if (modalRef.current) {
      const firstInput = modalRef.current.querySelector('input, textarea') as HTMLElement;
      if (firstInput) {
        // Small delay to ensure modal is rendered
        const timeoutId = setTimeout(() => firstInput.focus(), 50);
        return () => clearTimeout(timeoutId);
      }
    }
  }, []);

  return { modalRef };
}
