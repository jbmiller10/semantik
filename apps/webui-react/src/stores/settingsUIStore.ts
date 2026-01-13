import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface SettingsUIState {
  /** Map of section name -> isOpen state */
  sectionStates: Record<string, boolean>;

  /** Set a specific section's open state */
  setSectionOpen: (sectionName: string, isOpen: boolean) => void;

  /** Toggle a section's open state */
  toggleSection: (sectionName: string) => void;

  /** Get whether a section is open, with fallback to defaultOpen */
  isSectionOpen: (sectionName: string, defaultOpen?: boolean) => boolean;

  /** Reset all section states */
  resetSectionStates: () => void;
}

export const useSettingsUIStore = create<SettingsUIState>()(
  persist(
    (set, get) => ({
      sectionStates: {},

      setSectionOpen: (sectionName, isOpen) =>
        set((state) => ({
          sectionStates: {
            ...state.sectionStates,
            [sectionName]: isOpen,
          },
        })),

      toggleSection: (sectionName) =>
        set((state) => ({
          sectionStates: {
            ...state.sectionStates,
            [sectionName]: !state.sectionStates[sectionName],
          },
        })),

      isSectionOpen: (sectionName, defaultOpen = true) => {
        const state = get().sectionStates[sectionName];
        return state !== undefined ? state : defaultOpen;
      },

      resetSectionStates: () => set({ sectionStates: {} }),
    }),
    {
      name: 'semantik:settings:ui',
      partialize: (state) => ({ sectionStates: state.sectionStates }),
    }
  )
);
