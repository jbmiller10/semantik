import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

// This store now only handles client-side UI state for collections
// All server state is managed by React Query hooks in /hooks/useCollections.ts, 
// /hooks/useCollectionOperations.ts, and /hooks/useCollectionDocuments.ts

interface CollectionUIStore {
  // UI State
  selectedCollectionId: string | null;
  
  // UI Actions
  setSelectedCollection: (id: string | null) => void;
  clearStore: () => void;
}

export const useCollectionStore = create<CollectionUIStore>()(
  devtools(
    (set) => ({
      // Initial state
      selectedCollectionId: null,

      // UI actions
      setSelectedCollection: (id: string | null) => set({ selectedCollectionId: id }),
      
      clearStore: () => set({
        selectedCollectionId: null,
      }),
    }),
    {
      name: 'collection-ui-store',
    }
  )
);

// Re-export for backward compatibility during migration
// These can be removed once all components are updated
export const useCollectionUIStore = useCollectionStore;