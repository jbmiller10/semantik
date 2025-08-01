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
  const baseClasses = 'mt-1 block w-full rounded-md shadow-sm sm:text-sm px-3 py-2 border appearance-none';
  const stateClasses = hasError
    ? 'border-red-300 focus:ring-red-500 focus:border-red-500'
    : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500';
  const disabledClasses = isDisabled ? 'bg-gray-100 cursor-not-allowed' : '';

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
    ? 'border-red-300 focus:ring-red-500 focus:border-red-500'
    : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500';
  const disabledClasses = isDisabled ? 'bg-gray-100 cursor-not-allowed' : '';

  return `${customBaseClasses} ${stateClasses} ${disabledClasses}`.trim();
};