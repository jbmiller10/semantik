import { QueryClient } from '@tanstack/react-query';

/**
 * Shared QueryClient instance for the application.
 * Exported separately to avoid circular dependencies when
 * clearing cache during logout.
 */
export const queryClient = new QueryClient();
