/**
 * Utility functions for conditional animation classes.
 * Use these to conditionally apply animation classes based on user preferences.
 */

/**
 * Conditionally return an animation class based on enabled state.
 *
 * @param enabled - Whether animations are enabled
 * @param className - The animation class to apply (e.g., 'animate-spin')
 * @returns The className if enabled, empty string otherwise
 *
 * @example
 * const animationEnabled = useAnimationEnabled();
 * <div className={`h-4 w-4 ${animateIf(animationEnabled, 'animate-spin')}`} />
 */
export function animateIf(enabled: boolean, className: string): string {
  return enabled ? className : '';
}

/**
 * Conditionally return animation classes from an object.
 *
 * @param enabled - Whether animations are enabled
 * @param classes - Object mapping class names to whether they should be applied
 * @returns Space-separated string of classes that are both enabled and have truthy values
 *
 * @example
 * const animationEnabled = useAnimationEnabled();
 * <div className={animateClasses(animationEnabled, {
 *   'animate-spin': isLoading,
 *   'animate-pulse': isPending,
 * })} />
 */
export function animateClasses(
  enabled: boolean,
  classes: Record<string, boolean>
): string {
  if (!enabled) return '';
  return Object.entries(classes)
    .filter(([, include]) => include)
    .map(([className]) => className)
    .join(' ');
}

/**
 * Create a class string with optional animation.
 * Useful for combining static classes with conditional animation.
 *
 * @param baseClasses - Classes that are always applied
 * @param animationEnabled - Whether animations are enabled
 * @param animationClass - Animation class to conditionally apply
 * @returns Combined class string
 *
 * @example
 * const animationEnabled = useAnimationEnabled();
 * <div className={withAnimation('h-4 w-4 text-blue-500', animationEnabled, 'animate-spin')} />
 */
export function withAnimation(
  baseClasses: string,
  animationEnabled: boolean,
  animationClass: string
): string {
  return animationEnabled ? `${baseClasses} ${animationClass}` : baseClasses;
}
