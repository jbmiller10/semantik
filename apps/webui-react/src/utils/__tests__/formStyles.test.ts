import { describe, it, expect } from 'vitest';
import {
  getInputClassName,
  getInputClassNameWithBase,
  getSelectClassName,
  getTextareaClassName,
} from '../formStyles';

describe('formStyles', () => {
  describe('getInputClassName', () => {
    it('returns base classes when no error and not disabled', () => {
      const result = getInputClassName(false, false);

      expect(result).toContain('mt-1 block w-full rounded-md');
      expect(result).toContain('bg-[var(--input-bg)]');
      expect(result).toContain('border-[var(--input-border)]');
      expect(result).toContain('focus:border-[var(--input-focus)]');
      expect(result).not.toContain('border-error');
      expect(result).not.toContain('opacity-60');
      expect(result).not.toContain('cursor-not-allowed');
    });

    it('includes error classes when hasError is true', () => {
      const result = getInputClassName(true, false);

      expect(result).toContain('border-error');
      expect(result).toContain('focus:ring-error/50');
      expect(result).toContain('focus:border-error');
      expect(result).not.toContain('opacity-60');
    });

    it('includes disabled classes when isDisabled is true', () => {
      const result = getInputClassName(false, true);

      expect(result).toContain('opacity-60');
      expect(result).toContain('cursor-not-allowed');
      expect(result).not.toContain('border-error');
    });

    it('includes both error and disabled classes when both are true', () => {
      const result = getInputClassName(true, true);

      expect(result).toContain('border-error');
      expect(result).toContain('focus:ring-error/50');
      expect(result).toContain('opacity-60');
      expect(result).toContain('cursor-not-allowed');
    });

    it('uses additionalClasses instead of base classes when provided', () => {
      const additionalClasses = 'custom-class flex-1';
      const result = getInputClassName(false, false, additionalClasses);

      expect(result).toContain('custom-class');
      expect(result).toContain('flex-1');
      // Should not contain base classes
      expect(result).not.toContain('mt-1 block w-full');
      expect(result).not.toContain('bg-[var(--input-bg)]');
    });

    it('combines additionalClasses with error classes', () => {
      const additionalClasses = 'custom-class';
      const result = getInputClassName(true, false, additionalClasses);

      expect(result).toContain('custom-class');
      expect(result).toContain('border-error');
    });

    it('combines additionalClasses with disabled classes', () => {
      const additionalClasses = 'custom-class';
      const result = getInputClassName(false, true, additionalClasses);

      expect(result).toContain('custom-class');
      expect(result).toContain('opacity-60');
      expect(result).toContain('cursor-not-allowed');
    });

    it('trims whitespace from resulting class string', () => {
      const result = getInputClassName(false, false);

      expect(result).not.toMatch(/^\s/);
      expect(result).not.toMatch(/\s$/);
      expect(result).not.toContain('  '); // No double spaces
    });

    it('handles empty additionalClasses as default behavior', () => {
      const result = getInputClassName(false, false, '');

      expect(result).toContain('mt-1 block w-full');
      expect(result).toContain('bg-[var(--input-bg)]');
    });
  });

  describe('getInputClassNameWithBase', () => {
    const customBase = 'flex-1 rounded-md px-2';

    it('uses customBaseClasses as the foundation', () => {
      const result = getInputClassNameWithBase(false, false, customBase);

      expect(result).toContain('flex-1');
      expect(result).toContain('rounded-md');
      expect(result).toContain('px-2');
    });

    it('adds default border/focus classes when no error', () => {
      const result = getInputClassNameWithBase(false, false, customBase);

      expect(result).toContain('border-[var(--input-border)]');
      expect(result).toContain('focus:ring-[var(--input-focus)]/50');
      expect(result).toContain('focus:border-[var(--input-focus)]');
      expect(result).not.toContain('border-error');
    });

    it('adds error border/focus classes when hasError is true', () => {
      const result = getInputClassNameWithBase(true, false, customBase);

      expect(result).toContain('border-error');
      expect(result).toContain('focus:ring-error/50');
      expect(result).toContain('focus:border-error');
      expect(result).not.toContain('border-[var(--input-border)]');
    });

    it('adds disabled classes when isDisabled is true', () => {
      const result = getInputClassNameWithBase(false, true, customBase);

      expect(result).toContain('opacity-60');
      expect(result).toContain('cursor-not-allowed');
    });

    it('combines all class types correctly', () => {
      const result = getInputClassNameWithBase(true, true, customBase);

      expect(result).toContain('flex-1'); // base
      expect(result).toContain('border-error'); // error
      expect(result).toContain('opacity-60'); // disabled
    });

    it('trims whitespace from result', () => {
      const result = getInputClassNameWithBase(false, false, customBase);

      expect(result).not.toMatch(/^\s/);
      expect(result).not.toMatch(/\s$/);
    });
  });

  describe('getSelectClassName', () => {
    it('returns base classes for select elements', () => {
      const result = getSelectClassName(false, false);

      expect(result).toContain('mt-1 block w-full rounded-md');
      expect(result).toContain('bg-[var(--input-bg)]');
      expect(result).toContain('appearance-none');
      expect(result).toContain('border-[var(--input-border)]');
    });

    it('includes error classes when hasError is true', () => {
      const result = getSelectClassName(true, false);

      expect(result).toContain('border-error');
      expect(result).toContain('focus:ring-error/50');
      expect(result).toContain('focus:border-error');
    });

    it('includes disabled classes when isDisabled is true', () => {
      const result = getSelectClassName(false, true);

      expect(result).toContain('opacity-60');
      expect(result).toContain('cursor-not-allowed');
    });

    it('includes both error and disabled classes when both are true', () => {
      const result = getSelectClassName(true, true);

      expect(result).toContain('border-error');
      expect(result).toContain('opacity-60');
      expect(result).toContain('cursor-not-allowed');
    });

    it('does not include placeholder styling (unlike input)', () => {
      const result = getSelectClassName(false, false);

      expect(result).not.toContain('placeholder:');
    });
  });

  describe('getTextareaClassName', () => {
    it('returns base classes including resize-y', () => {
      const result = getTextareaClassName(false, false);

      expect(result).toContain('mt-1 block w-full rounded-md');
      expect(result).toContain('resize-y');
      expect(result).toContain('bg-[var(--input-bg)]');
    });

    it('includes placeholder styling', () => {
      const result = getTextareaClassName(false, false);

      expect(result).toContain('placeholder:text-[var(--text-muted)]');
    });

    it('includes error classes when hasError is true', () => {
      const result = getTextareaClassName(true, false);

      expect(result).toContain('border-error');
      expect(result).toContain('focus:ring-error/50');
      expect(result).toContain('focus:border-error');
    });

    it('includes disabled classes when isDisabled is true', () => {
      const result = getTextareaClassName(false, true);

      expect(result).toContain('opacity-60');
      expect(result).toContain('cursor-not-allowed');
    });

    it('includes both error and disabled classes when both are true', () => {
      const result = getTextareaClassName(true, true);

      expect(result).toContain('border-error');
      expect(result).toContain('opacity-60');
      expect(result).toContain('cursor-not-allowed');
    });
  });

  describe('class consistency', () => {
    it('input, select, and textarea share common base patterns', () => {
      const inputClasses = getInputClassName(false, false);
      const selectClasses = getSelectClassName(false, false);
      const textareaClasses = getTextareaClassName(false, false);

      // All should have these common patterns
      [inputClasses, selectClasses, textareaClasses].forEach((classes) => {
        expect(classes).toContain('mt-1 block w-full rounded-md');
        expect(classes).toContain('bg-[var(--input-bg)]');
        expect(classes).toContain('text-[var(--text-primary)]');
        expect(classes).toContain('border-[var(--input-border)]');
        expect(classes).toContain('focus:border-[var(--input-focus)]');
        expect(classes).toContain('transition-colors');
        expect(classes).toContain('outline-none');
      });
    });

    it('all functions use the same error classes', () => {
      const inputError = getInputClassName(true, false);
      const selectError = getSelectClassName(true, false);
      const textareaError = getTextareaClassName(true, false);

      [inputError, selectError, textareaError].forEach((classes) => {
        expect(classes).toContain('border-error');
        expect(classes).toContain('focus:ring-error/50');
        expect(classes).toContain('focus:border-error');
      });
    });

    it('all functions use the same disabled classes', () => {
      const inputDisabled = getInputClassName(false, true);
      const selectDisabled = getSelectClassName(false, true);
      const textareaDisabled = getTextareaClassName(false, true);

      [inputDisabled, selectDisabled, textareaDisabled].forEach((classes) => {
        expect(classes).toContain('opacity-60');
        expect(classes).toContain('cursor-not-allowed');
      });
    });
  });
});
