/**
 * Animation Context - provides animation preference to all components.
 * Reads from user preferences and provides a hook for components to consume.
 */
import { createContext, useContext, type ReactNode } from 'react';
import { usePreferences } from '../hooks/usePreferences';

interface AnimationContextValue {
  /** Whether animations are enabled (from user preferences) */
  animationEnabled: boolean;
  /** Whether preferences are still loading */
  isLoading: boolean;
}

const AnimationContext = createContext<AnimationContextValue>({
  animationEnabled: true, // Default to enabled
  isLoading: true,
});

interface AnimationProviderProps {
  children: ReactNode;
}

/**
 * Provider component that reads animation preference from user preferences.
 * Wrap your app with this to provide animation state to all components.
 */
export function AnimationProvider({ children }: AnimationProviderProps) {
  const { data: preferences, isLoading } = usePreferences();

  const animationEnabled = preferences?.interface?.animation_enabled ?? true;

  return (
    <AnimationContext.Provider value={{ animationEnabled, isLoading }}>
      {children}
    </AnimationContext.Provider>
  );
}

/**
 * Hook to access animation enabled state.
 * Falls back to true if preferences haven't loaded yet.
 *
 * @example
 * const { animationEnabled } = useAnimation();
 * return <div className={animationEnabled ? 'animate-spin' : ''}>...</div>
 */
// eslint-disable-next-line react-refresh/only-export-components
export function useAnimation(): AnimationContextValue {
  return useContext(AnimationContext);
}

/**
 * Hook that returns just the boolean animation state.
 * Convenient shorthand for components that just need the boolean.
 *
 * @example
 * const animationEnabled = useAnimationEnabled();
 * return <Loader className={animationEnabled ? 'animate-spin' : ''} />
 */
// eslint-disable-next-line react-refresh/only-export-components
export function useAnimationEnabled(): boolean {
  const { animationEnabled } = useAnimation();
  return animationEnabled;
}
