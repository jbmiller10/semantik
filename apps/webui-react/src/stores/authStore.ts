import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface User {
  id: number;
  username: string;
  email: string;
  full_name?: string;
  is_active: boolean;
  created_at: string;
  last_login?: string;
}

interface AuthState {
  token: string | null;
  refreshToken: string | null;
  user: User | null;
  setAuth: (token: string, user: User, refreshToken?: string) => void;
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
          // Explicitly clear localStorage to ensure no stale auth data persists
          localStorage.removeItem('auth-storage');
        }
      },
    }),
    {
      name: 'auth-storage',
    }
  )
);