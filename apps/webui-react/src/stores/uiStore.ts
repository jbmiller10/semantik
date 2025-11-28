import { create } from 'zustand';

export interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
  duration?: number;
}

export interface DocumentViewerState {
  collectionId: string;
  docId: string;
  chunkId?: string;
  chunkContent?: string;  // Content to highlight in the document (fallback)
  startOffset?: number;   // Character offset where chunk starts in source document
  endOffset?: number;     // Character offset where chunk ends in source document
}

export interface UIState {
  toasts: Toast[];
  activeTab: 'search' | 'collections' | 'operations';
  showDocumentViewer: DocumentViewerState | null;
  showCollectionDetailsModal: string | null;
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  setActiveTab: (tab: 'search' | 'collections' | 'operations') => void;
  setShowDocumentViewer: (viewer: DocumentViewerState | null) => void;
  setShowCollectionDetailsModal: (collectionId: string | null) => void;
}

export const useUIStore = create<UIState>((set) => ({
  toasts: [],
  activeTab: 'collections',
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
  setShowDocumentViewer: (viewer) => set({ showDocumentViewer: viewer }),
  setShowCollectionDetailsModal: (collectionId) => set({ showCollectionDetailsModal: collectionId }),
}));
