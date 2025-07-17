import { create } from 'zustand';

export interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
  duration?: number;
}

interface UIState {
  toasts: Toast[];
  activeTab: 'create' | 'jobs' | 'search' | 'collections';
  showJobMetricsModal: string | null;
  showDocumentViewer: { jobId: string; docId: string; chunkId?: string } | null;
  showCollectionDetailsModal: string | null;
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  setActiveTab: (tab: 'create' | 'jobs' | 'search' | 'collections') => void;
  setShowJobMetricsModal: (jobId: string | null) => void;
  setShowDocumentViewer: (viewer: { jobId: string; docId: string; chunkId?: string } | null) => void;
  setShowCollectionDetailsModal: (collectionId: string | null) => void;
}

export const useUIStore = create<UIState>((set) => ({
  toasts: [],
  activeTab: 'collections',
  showJobMetricsModal: null,
  showDocumentViewer: null,
  showCollectionDetailsModal: null,
  addToast: (toast) => {
    const id = Date.now().toString();
    set((state) => ({ toasts: [...state.toasts, { ...toast, id }] }));
    
    // Auto-remove toast after duration
    if (toast.duration !== 0) {
      setTimeout(() => {
        set((state) => ({
          toasts: state.toasts.filter((t) => t.id !== id),
        }));
      }, toast.duration || 5000);
    }
  },
  removeToast: (id) =>
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    })),
  setActiveTab: (tab) => set({ activeTab: tab }),
  setShowJobMetricsModal: (jobId) => set({ showJobMetricsModal: jobId }),
  setShowDocumentViewer: (viewer) => set({ showDocumentViewer: viewer }),
  setShowCollectionDetailsModal: (collectionId) => set({ showCollectionDetailsModal: collectionId }),
}));