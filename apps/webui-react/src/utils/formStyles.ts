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
  const baseClasses = 'mt-1 block w-full rounded-md shadow-sm sm:text-sm px-3 py-2 border appearance-none bg-void-950/50 text-gray-100';
  const stateClasses = hasError
    ? 'border-red-500/50 focus:ring-red-500/50 focus:border-red-500'
    : 'border-void-700 focus:ring-signal-500/50 focus:border-signal-500';
  const disabledClasses = isDisabled ? 'bg-void-800 cursor-not-allowed opacity-60' : '';

  // If additionalClasses are provided, they can override base classes
  if (additionalClasses) {
    return `${additionalClasses} ${stateClasses} ${disabledClasses}`.trim();
  }

  return `${baseClasses} ${stateClasses} ${disabledClasses}`.trim();
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
  const stateClasses = hasError
    ? 'border-red-500/50 focus:ring-red-500/50 focus:border-red-500'
    : 'border-void-700 focus:ring-signal-500/50 focus:border-signal-500';
  const disabledClasses = isDisabled ? 'bg-void-800 cursor-not-allowed opacity-60' : '';

  return `${customBaseClasses} ${stateClasses} ${disabledClasses}`.trim();
};