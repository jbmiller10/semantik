import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface User {
  id: number;
  username: string;
  email: string;
  full_name?: string;
  is_admin: boolean;
}

interface AuthState {
  token: string | null;
  user: User | null;
  setAuth: (token: string, user: User) => void;
  logout: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      token: null,
      user: null,
      setAuth: (token: string, user: User) => set({ token, user }),
      logout: async () => {
        try {
          // Call logout API endpoint
          const token = get().token;
          if (token) {
            await fetch('/api/auth/logout', {
              method: 'POST',
              headers: {
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json',
              },
            });
          }
        } catch (error) {
          console.error('Logout API call failed:', error);
        } finally {
          // Always clear the state regardless of API call result
          set({ token: null, user: null });
        }
      },
    }),
    {
      name: 'auth-storage',
    }
  )
);