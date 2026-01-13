import { describe, it, expect, beforeEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useSettingsUIStore } from '../settingsUIStore';

describe('settingsUIStore', () => {
  beforeEach(() => {
    // Reset the store to initial state
    useSettingsUIStore.setState({ sectionStates: {} });
  });

  describe('isSectionOpen', () => {
    it('returns defaultOpen when section has no saved state', () => {
      const { result } = renderHook(() => useSettingsUIStore());
      const isOpen = result.current.isSectionOpen('new-section', false);
      expect(isOpen).toBe(false);
    });

    it('returns true as default when no defaultOpen provided', () => {
      const { result } = renderHook(() => useSettingsUIStore());
      const isOpen = result.current.isSectionOpen('new-section');
      expect(isOpen).toBe(true);
    });

    it('returns saved state when section has been toggled', () => {
      const { result } = renderHook(() => useSettingsUIStore());

      act(() => {
        result.current.setSectionOpen('test-section', false);
      });

      const isOpen = result.current.isSectionOpen('test-section', true);
      expect(isOpen).toBe(false);
    });

    it('returns saved state over default when saved state is false', () => {
      const { result } = renderHook(() => useSettingsUIStore());

      act(() => {
        result.current.setSectionOpen('test-section', false);
      });

      // Even though defaultOpen is true, saved state (false) takes precedence
      expect(result.current.isSectionOpen('test-section', true)).toBe(false);
    });
  });

  describe('toggleSection', () => {
    it('toggles section from undefined to true (negation of undefined)', () => {
      const { result } = renderHook(() => useSettingsUIStore());

      act(() => {
        result.current.toggleSection('test-section');
      });

      // !undefined = true
      expect(result.current.sectionStates['test-section']).toBe(true);
    });

    it('toggles section from false to true', () => {
      const { result } = renderHook(() => useSettingsUIStore());

      act(() => {
        result.current.setSectionOpen('test-section', false);
      });

      act(() => {
        result.current.toggleSection('test-section');
      });

      expect(result.current.sectionStates['test-section']).toBe(true);
    });

    it('toggles section from true to false', () => {
      const { result } = renderHook(() => useSettingsUIStore());

      act(() => {
        result.current.setSectionOpen('test-section', true);
      });

      act(() => {
        result.current.toggleSection('test-section');
      });

      expect(result.current.sectionStates['test-section']).toBe(false);
    });
  });

  describe('setSectionOpen', () => {
    it('sets section to open', () => {
      const { result } = renderHook(() => useSettingsUIStore());

      act(() => {
        result.current.setSectionOpen('test-section', true);
      });

      expect(result.current.sectionStates['test-section']).toBe(true);
    });

    it('sets section to closed', () => {
      const { result } = renderHook(() => useSettingsUIStore());

      act(() => {
        result.current.setSectionOpen('test-section', false);
      });

      expect(result.current.sectionStates['test-section']).toBe(false);
    });

    it('handles multiple sections independently', () => {
      const { result } = renderHook(() => useSettingsUIStore());

      act(() => {
        result.current.setSectionOpen('section-1', true);
        result.current.setSectionOpen('section-2', false);
        result.current.setSectionOpen('section-3', true);
      });

      expect(result.current.sectionStates['section-1']).toBe(true);
      expect(result.current.sectionStates['section-2']).toBe(false);
      expect(result.current.sectionStates['section-3']).toBe(true);
    });
  });

  describe('resetSectionStates', () => {
    it('clears all section states', () => {
      const { result } = renderHook(() => useSettingsUIStore());

      act(() => {
        result.current.setSectionOpen('section-1', true);
        result.current.setSectionOpen('section-2', false);
      });

      expect(Object.keys(result.current.sectionStates)).toHaveLength(2);

      act(() => {
        result.current.resetSectionStates();
      });

      expect(result.current.sectionStates).toEqual({});
    });

    it('sections return to default after reset', () => {
      const { result } = renderHook(() => useSettingsUIStore());

      act(() => {
        result.current.setSectionOpen('test-section', false);
      });

      expect(result.current.isSectionOpen('test-section', true)).toBe(false);

      act(() => {
        result.current.resetSectionStates();
      });

      // After reset, should return the defaultOpen value again
      expect(result.current.isSectionOpen('test-section', true)).toBe(true);
    });
  });

  describe('state sharing', () => {
    it('shares state between multiple hooks', () => {
      const { result: hook1 } = renderHook(() => useSettingsUIStore());
      const { result: hook2 } = renderHook(() => useSettingsUIStore());

      act(() => {
        hook1.current.setSectionOpen('shared-section', true);
      });

      // Both hooks should see the same state
      expect(hook1.current.sectionStates['shared-section']).toBe(true);
      expect(hook2.current.sectionStates['shared-section']).toBe(true);
    });
  });
});
