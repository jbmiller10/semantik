import { describe, it, expect, beforeEach, vi } from 'vitest';
import { 
  checkAndMigrateLocalStorage, 
  getCurrentStorageVersion,
  forceMigration,
  _internal
} from '../localStorageMigration';

const { clearLocalStorage, shouldClearKey, performMigration, STORAGE_VERSION_KEY, CURRENT_VERSION } = _internal;

// Mock window.location.reload
const mockReload = vi.fn();
Object.defineProperty(window, 'location', {
  value: { reload: mockReload },
  writable: true,
});

describe('localStorageMigration', () => {
  beforeEach(() => {
    // Clear localStorage before each test
    localStorage.clear();
    mockReload.mockClear();
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  describe('checkAndMigrateLocalStorage', () => {
    it('should not migrate if version matches current version', () => {
      localStorage.setItem(STORAGE_VERSION_KEY, CURRENT_VERSION);
      localStorage.setItem('auth-storage', 'some-auth-data');

      checkAndMigrateLocalStorage();

      expect(localStorage.getItem('auth-storage')).toBe('some-auth-data');
      expect(mockReload).not.toHaveBeenCalled();
    });

    it('should migrate and clear data if no version exists', () => {
      localStorage.setItem('auth-storage', 'old-auth-data');
      localStorage.setItem('jobs-data', 'old-jobs-data');
      localStorage.setItem('semantik_settings', 'old-settings');

      checkAndMigrateLocalStorage();

      expect(localStorage.getItem('auth-storage')).toBeNull();
      expect(localStorage.getItem('jobs-data')).toBeNull();
      expect(localStorage.getItem('semantik_settings')).toBeNull();
      expect(localStorage.getItem(STORAGE_VERSION_KEY)).toBe(CURRENT_VERSION);
      expect(mockReload).toHaveBeenCalled();
    });

    it('should migrate if version is different', () => {
      localStorage.setItem(STORAGE_VERSION_KEY, '1.0.0');
      localStorage.setItem('auth-storage', 'old-auth-data');

      checkAndMigrateLocalStorage();

      expect(localStorage.getItem('auth-storage')).toBeNull();
      expect(localStorage.getItem(STORAGE_VERSION_KEY)).toBe(CURRENT_VERSION);
      expect(mockReload).toHaveBeenCalled();
    });

    it.skip('should handle migration errors gracefully', () => {
      // Mock localStorage.getItem to throw an error on first call only
      const originalGetItem = localStorage.getItem;
      let callCount = 0;
      
      localStorage.getItem = vi.fn().mockImplementation((key) => {
        callCount++;
        if (callCount === 1) {
          throw new Error('Storage error');
        }
        return originalGetItem.call(localStorage, key);
      });

      checkAndMigrateLocalStorage();

      expect(console.error).toHaveBeenCalledWith(
        '[LocalStorage Migration] Error during migration:',
        expect.any(Error)
      );
      expect(mockReload).toHaveBeenCalled();

      // Restore original function
      localStorage.getItem = originalGetItem;
    });
  });

  describe('shouldClearKey', () => {
    it('should not clear version key', () => {
      expect(shouldClearKey(STORAGE_VERSION_KEY)).toBe(false);
    });

    it('should clear keys matching patterns', () => {
      expect(shouldClearKey('semantik_user_prefs')).toBe(true);
      expect(shouldClearKey('auth-storage')).toBe(true);
      expect(shouldClearKey('jobs-queue')).toBe(true);
      expect(shouldClearKey('search-history')).toBe(true);
      expect(shouldClearKey('collections-data')).toBe(true);
      expect(shouldClearKey('ui-theme')).toBe(true);
    });

    it('should not clear unrelated keys', () => {
      expect(shouldClearKey('other-app-data')).toBe(false);
      expect(shouldClearKey('random-key')).toBe(false);
    });
  });

  describe('clearLocalStorage', () => {
    it('should clear only app-related keys', () => {
      // Add various keys
      localStorage.setItem('semantik_settings', 'settings');
      localStorage.setItem('auth-storage', 'auth');
      localStorage.setItem('jobs-data', 'jobs');
      localStorage.setItem('other-app-data', 'other');
      localStorage.setItem(STORAGE_VERSION_KEY, CURRENT_VERSION);

      clearLocalStorage();

      expect(localStorage.getItem('semantik_settings')).toBeNull();
      expect(localStorage.getItem('auth-storage')).toBeNull();
      expect(localStorage.getItem('jobs-data')).toBeNull();
      expect(localStorage.getItem('other-app-data')).toBe('other'); // Should not be cleared
      expect(localStorage.getItem(STORAGE_VERSION_KEY)).toBe(CURRENT_VERSION); // Should not be cleared
    });
  });

  describe('performMigration', () => {
    it('should clear all data for pre-2.0 versions', () => {
      localStorage.setItem('auth-storage', 'old-data');
      localStorage.setItem('jobs-data', 'old-jobs');

      performMigration('1.5.0');

      expect(localStorage.getItem('auth-storage')).toBeNull();
      expect(localStorage.getItem('jobs-data')).toBeNull();
    });

    it('should clear all data for null version', () => {
      localStorage.setItem('auth-storage', 'old-data');

      performMigration(null);

      expect(localStorage.getItem('auth-storage')).toBeNull();
    });
  });

  describe('getCurrentStorageVersion', () => {
    it('should return current version if set', () => {
      localStorage.setItem(STORAGE_VERSION_KEY, '2.0.0');
      expect(getCurrentStorageVersion()).toBe('2.0.0');
    });

    it('should return null if not set', () => {
      expect(getCurrentStorageVersion()).toBeNull();
    });
  });

  describe('forceMigration', () => {
    it('should remove version and trigger migration', () => {
      localStorage.setItem(STORAGE_VERSION_KEY, CURRENT_VERSION);
      localStorage.setItem('auth-storage', 'data');

      forceMigration();

      expect(localStorage.getItem('auth-storage')).toBeNull();
      expect(localStorage.getItem(STORAGE_VERSION_KEY)).toBe(CURRENT_VERSION);
      expect(mockReload).toHaveBeenCalled();
    });
  });
});