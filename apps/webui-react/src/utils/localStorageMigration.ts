/**
 * LocalStorage migration utility for handling breaking changes between versions
 * This ensures users with old data structures get a clean slate when upgrading
 */

const STORAGE_VERSION_KEY = 'semantik_storage_version';
const CURRENT_VERSION = '2.0.0';

// Keys that should be preserved across migrations
const PRESERVED_KEYS: string[] = [
  // Add any keys that should survive migrations here
  // For now, we'll clear everything except the version key
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
  // For now, we're doing a complete clear for the v2.0.0 migration
  // In the future, we can add version-specific migrations here
  
  if (!oldVersion || oldVersion < '2.0.0') {
    // Complete clear for pre-2.0 or unknown versions
    clearLocalStorage();
  }
  
  // Future migrations can be added here
  // Example:
  // if (oldVersion === '2.0.0' && CURRENT_VERSION === '2.1.0') {
  //   migrateFrom2_0To2_1();
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
  STORAGE_VERSION_KEY,
  CURRENT_VERSION,
};