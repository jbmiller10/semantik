/**
 * Utility functions for consistent form styling across the application
 */

/**
 * Returns Tailwind CSS classes for form inputs with consistent styling
 * @param hasError - Whether the input has a validation error
 * @param isDisabled - Whether the input is disabled
 * @param additionalClasses - Additional CSS classes to apply (will override base classes)
 * @returns CSS class string for the input element
 */
export const getInputClassName = (
  hasError: boolean,
  isDisabled: boolean,
  additionalClasses = ''
): string => {
  const baseClasses = 'mt-1 block w-full rounded-md shadow-sm sm:text-sm px-3 py-2 border appearance-none ' +
    'bg-[var(--input-bg)] text-[var(--text-primary)] border-[var(--input-border)] ' +
    'focus:border-[var(--input-focus)] focus:ring-1 focus:ring-[var(--input-focus)] ' +
    'placeholder:text-[var(--text-muted)] transition-colors duration-150 outline-none';

  const errorClasses = hasError
    ? 'border-error focus:ring-error/50 focus:border-error'
    : '';
  const disabledClasses = isDisabled ? 'opacity-60 cursor-not-allowed' : '';

  // If additionalClasses are provided, they can override base classes
  if (additionalClasses) {
    return `${additionalClasses} ${errorClasses} ${disabledClasses}`.trim();
  }

  return `${baseClasses} ${errorClasses} ${disabledClasses}`.trim();
};

/**
 * Returns Tailwind CSS classes for form inputs with custom base styling
 * Useful when you need to modify the base classes (e.g., for inputs in flex containers)
 * @param hasError - Whether the input has a validation error
 * @param isDisabled - Whether the input is disabled
 * @param customBaseClasses - Custom base classes to use instead of defaults
 * @returns CSS class string for the input element
 */
export const getInputClassNameWithBase = (
  hasError: boolean,
  isDisabled: boolean,
  customBaseClasses: string
): string => {
  const errorClasses = hasError
    ? 'border-error focus:ring-error/50 focus:border-error'
    : 'border-[var(--input-border)] focus:ring-[var(--input-focus)]/50 focus:border-[var(--input-focus)]';
  const disabledClasses = isDisabled ? 'opacity-60 cursor-not-allowed' : '';

  return `${customBaseClasses} ${errorClasses} ${disabledClasses}`.trim();
};

/**
 * Returns Tailwind CSS classes for select elements
 */
export const getSelectClassName = (
  hasError: boolean,
  isDisabled: boolean
): string => {
  const baseClasses = 'mt-1 block w-full rounded-md shadow-sm sm:text-sm px-3 py-2 border appearance-none ' +
    'bg-[var(--input-bg)] text-[var(--text-primary)] border-[var(--input-border)] ' +
    'focus:border-[var(--input-focus)] focus:ring-1 focus:ring-[var(--input-focus)] ' +
    'transition-colors duration-150 outline-none';

  const errorClasses = hasError
    ? 'border-error focus:ring-error/50 focus:border-error'
    : '';
  const disabledClasses = isDisabled ? 'opacity-60 cursor-not-allowed' : '';

  return `${baseClasses} ${errorClasses} ${disabledClasses}`.trim();
};

/**
 * Returns Tailwind CSS classes for textarea elements
 */
export const getTextareaClassName = (
  hasError: boolean,
  isDisabled: boolean
): string => {
  const baseClasses = 'mt-1 block w-full rounded-md shadow-sm sm:text-sm px-3 py-2 border appearance-none ' +
    'bg-[var(--input-bg)] text-[var(--text-primary)] border-[var(--input-border)] ' +
    'focus:border-[var(--input-focus)] focus:ring-1 focus:ring-[var(--input-focus)] ' +
    'placeholder:text-[var(--text-muted)] transition-colors duration-150 outline-none resize-y';

  const errorClasses = hasError
    ? 'border-error focus:ring-error/50 focus:border-error'
    : '';
  const disabledClasses = isDisabled ? 'opacity-60 cursor-not-allowed' : '';

  return `${baseClasses} ${errorClasses} ${disabledClasses}`.trim();
};
