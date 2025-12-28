import { create } from 'zustand';

export interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
  duration?: number;
  timerId?: ReturnType<typeof setTimeout>;
}

export interface UIState {
  toasts: Toast[];
  activeTab: 'search' | 'collections' | 'operations';
  showDocumentViewer: { collectionId: string; docId: string; chunkId?: string } | null;
  showCollectionDetailsModal: string | null;
  addToast: (toast: Omit<Toast, 'id' | 'timerId'>) => void;
  removeToast: (id: string) => void;
  setActiveTab: (tab: 'search' | 'collections' | 'operations') => void;
  setShowDocumentViewer: (viewer: { collectionId: string; docId: string; chunkId?: string } | null) => void;
  setShowCollectionDetailsModal: (collectionId: string | null) => void;
}

export const useUIStore = create<UIState>((set, get) => ({
  toasts: [],
  activeTab: 'collections',
  showDocumentViewer: null,
  showCollectionDetailsModal: null,
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
}));
