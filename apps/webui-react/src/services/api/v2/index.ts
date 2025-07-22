export * from './types';
export * from './collections';
export * from './auth';
export * from './settings';
export * from './directoryScan';
export * from './documents';

// Re-export the unified v2Api object as default
import { v2Api } from './collections';
export default v2Api;