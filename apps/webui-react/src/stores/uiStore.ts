import { create } from 'zustand';

export interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
  duration?: number;
  timerId?: ReturnType<typeof setTimeout>;
}

export type Theme = 'light' | 'dark' | 'system';

export interface UIState {
  toasts: Toast[];
  activeTab: 'search' | 'collections' | 'operations';
  showDocumentViewer: { collectionId: string; docId: string; chunkId?: string } | null;
  showCollectionDetailsModal: string | null;
  theme: Theme;
  addToast: (toast: Omit<Toast, 'id' | 'timerId'>) => void;
  removeToast: (id: string) => void;
  setActiveTab: (tab: 'search' | 'collections' | 'operations') => void;
  setShowDocumentViewer: (viewer: { collectionId: string; docId: string; chunkId?: string } | null) => void;
  setShowCollectionDetailsModal: (collectionId: string | null) => void;
  setTheme: (theme: Theme) => void;
}

// Get initial theme from localStorage or default to 'system'
const getInitialTheme = (): Theme => {
  if (typeof window === 'undefined') return 'system';
  const stored = localStorage.getItem('semantik-theme');
  if (stored === 'light' || stored === 'dark' || stored === 'system') {
    return stored;
  }
  return 'system';
};

// Apply theme to document and return resolved theme
const applyTheme = (theme: Theme): void => {
  if (typeof window === 'undefined') return;

  const root = document.documentElement;
  const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const isDark = theme === 'dark' || (theme === 'system' && systemDark);

  if (isDark) {
    root.classList.add('dark');
  } else {
    root.classList.remove('dark');
  }
};

// Initialize theme on load
const initialTheme = getInitialTheme();
if (typeof window !== 'undefined') {
  applyTheme(initialTheme);

  // Listen for system theme changes
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    const currentTheme = localStorage.getItem('semantik-theme') as Theme || 'system';
    if (currentTheme === 'system') {
      applyTheme('system');
    }
  });
}

export const useUIStore = create<UIState>((set, get) => ({
  toasts: [],
  activeTab: 'collections',
  showDocumentViewer: null,
  showCollectionDetailsModal: null,
  theme: initialTheme,
  addToast: (toast) => {
    // Use crypto.randomUUID() to avoid ID collisions from Date.now()
    const id = crypto.randomUUID();
    let timerId: ReturnType<typeof setTimeout> | undefined;

    // Auto-remove toast after duration
    if (toast.duration !== 0) {
      timerId = setTimeout(() => {
        get().removeToast(id);
      }, toast.duration || 5000);
    }

    set((state) => ({
      toasts: [...state.toasts, { ...toast, id, timerId }],
    }));
  },
  removeToast: (id) => {
    // Clear the timer to prevent memory leaks
    const toast = get().toasts.find((t) => t.id === id);
    if (toast?.timerId) {
      clearTimeout(toast.timerId);
    }
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    }));
  },
  setActiveTab: (tab) => set({ activeTab: tab }),
  setShowDocumentViewer: (viewer) => set({ showDocumentViewer: viewer }),
  setShowCollectionDetailsModal: (collectionId) => set({ showCollectionDetailsModal: collectionId }),
  setTheme: (theme) => {
    localStorage.setItem('semantik-theme', theme);
    applyTheme(theme);
    set({ theme });
  },
}));
