import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { queryClient } from '../services/queryClient';

interface User {
  id: number;
  username: string;
  email: string;
  full_name?: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
  last_login?: string;
}

interface AuthState {
  token: string | null;
  refreshToken: string | null;
  user: User | null;
  setAuth: (token: string, user: User, refreshToken?: string) => void;
  setTokens: (accessToken: string, refreshToken: string) => void;
  logout: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      refreshToken: null,
      user: null,
      setAuth: (token: string, user: User, refreshToken?: string) =>
        set({ token, user, refreshToken: refreshToken || null }),
      setTokens: (accessToken: string, refreshToken: string) =>
        set({ token: accessToken, refreshToken }),
      logout: async () => {
        try {
          // Call logout API endpoint
          const token = get().token;
          const refreshToken = get().refreshToken;
          if (token) {
            await fetch('/api/auth/logout', {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ refresh_token: refreshToken }),
            });
          }
        } catch (error) {
          console.error('Logout API call failed:', error);
        } finally {
          // Always clear the state regardless of API call result
          set({ token: null, refreshToken: null, user: null });

          // Clear all possible storage keys
          localStorage.removeItem('auth-storage');
          sessionStorage.clear();

          // Invalidate all React Query caches to prevent stale data leakage
          queryClient.clear();
        }
      },
    }),
    {
      name: 'auth-storage',
    }
  )
);