/**
 * LocalStorage migration utility for handling breaking changes between versions
 * This ensures users with old data structures get a clean slate when upgrading
 */

const STORAGE_VERSION_KEY = 'semantik_storage_version';
const CURRENT_VERSION = '0.7.5';

// Keys that should be preserved across migrations
const PRESERVED_KEYS: string[] = [
  // Add any keys that should survive migrations here
  // Currently, we clear all app data on major version changes
];

// Patterns for keys that should be cleared
const CLEAR_KEY_PATTERNS = [
  'semantik_',
  'auth-storage',
  'jobs',
  'search',
  'collections',
  'settings',
  'ui-',
];

/**
 * Compare two semantic version strings.
 * Returns -1 if a < b, 0 if a === b, 1 if a > b.
 * Handles version strings like "1.0.0", "2.1.0", "10.0.0" correctly.
 */
function compareVersions(a: string, b: string): number {
  const partsA = a.split('.').map(Number);
  const partsB = b.split('.').map(Number);
  for (let i = 0; i < 3; i++) {
    const numA = partsA[i] || 0;
    const numB = partsB[i] || 0;
    if (numA < numB) return -1;
    if (numA > numB) return 1;
  }
  return 0;
}

/**
 * Check localStorage version and migrate if necessary
 * This function should be called before the app renders
 */
export function checkAndMigrateLocalStorage(): void {
  try {
    const storedVersion = localStorage.getItem(STORAGE_VERSION_KEY);
    
    // If no version or version mismatch, perform migration
    if (!storedVersion || storedVersion !== CURRENT_VERSION) {
      console.log(`[LocalStorage Migration] Migrating from version ${storedVersion || 'unknown'} to ${CURRENT_VERSION}`);
      
      // Perform migration
      performMigration(storedVersion);
      
      // Set new version
      localStorage.setItem(STORAGE_VERSION_KEY, CURRENT_VERSION);
      
      // Force reload to ensure clean state
      // This prevents any cached state from interfering with the new version
      console.log('[LocalStorage Migration] Migration complete. Reloading application...');
      window.location.reload();
    }
  } catch (error) {
    console.error('[LocalStorage Migration] Error during migration:', error);
    // In case of error, clear everything as a safety measure
    clearLocalStorage();
    localStorage.setItem(STORAGE_VERSION_KEY, CURRENT_VERSION);
    window.location.reload();
  }
}

/**
 * Perform the actual migration based on the old version
 */
function performMigration(oldVersion: string | null): void {
  // For now, we do a complete clear on any version mismatch.
  // This is the safest default when the persisted state shape is not guaranteed
  // to be forward/backward compatible.
  //
  // In the future, add version-specific migrations here instead of clearing.
  // If you add migrations, prefer:
  // - explicit oldVersion checks
  // - a "best effort" migration path
  // - a final fallback to clear on unknown/invalid versions

  if (!oldVersion) {
    clearLocalStorage();
    return;
  }

  if (oldVersion !== CURRENT_VERSION) {
    clearLocalStorage();
  }
  
  // Future migrations can be added here
  // Example:
  // if (oldVersion === '0.7.5' && CURRENT_VERSION === '0.7.6') {
  //   migrateFrom0_7_5To0_7_6();
  // }
}

/**
 * Clear all app-related localStorage keys
 */
function clearLocalStorage(): void {
  const keysToRemove: string[] = [];
  
  // Collect all keys that match our patterns
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && shouldClearKey(key)) {
      keysToRemove.push(key);
    }
  }
  
  // Remove collected keys
  keysToRemove.forEach(key => {
    console.log(`[LocalStorage Migration] Removing key: ${key}`);
    localStorage.removeItem(key);
  });
  
  console.log(`[LocalStorage Migration] Cleared ${keysToRemove.length} keys`);
}

/**
 * Determine if a key should be cleared based on patterns
 */
function shouldClearKey(key: string): boolean {
  // Never clear the version key itself
  if (key === STORAGE_VERSION_KEY) {
    return false;
  }
  
  // Check if key is in preserved list
  if (PRESERVED_KEYS.includes(key)) {
    return false;
  }
  
  // Check if key matches any clear patterns
  return CLEAR_KEY_PATTERNS.some(pattern => key.includes(pattern));
}

/**
 * Utility function to get current storage version
 */
export function getCurrentStorageVersion(): string | null {
  return localStorage.getItem(STORAGE_VERSION_KEY);
}

/**
 * Utility function to manually trigger migration (for debugging)
 */
export function forceMigration(): void {
  localStorage.removeItem(STORAGE_VERSION_KEY);
  checkAndMigrateLocalStorage();
}

// Export for testing
export const _internal = {
  clearLocalStorage,
  shouldClearKey,
  performMigration,
  compareVersions,
  STORAGE_VERSION_KEY,
  CURRENT_VERSION,
};
