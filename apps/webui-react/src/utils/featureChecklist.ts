/**
 * Feature verification checklist for React refactor
 * This utility helps track which features from the original app.js have been implemented
 */

export interface FeatureCheck {
  category: string;
  feature: string;
  implemented: boolean;
  tested: boolean;
  notes?: string;
}

export const featureChecklist: FeatureCheck[] = [
  // Authentication
  {
    category: 'Authentication',
    feature: 'Login with username/password',
    implemented: true,
    tested: false,
  },
  {
    category: 'Authentication',
    feature: 'Registration flow',
    implemented: true,
    tested: false,
  },
  {
    category: 'Authentication',
    feature: 'Token persistence in localStorage',
    implemented: true,
    tested: false,
  },
  {
    category: 'Authentication',
    feature: 'Auto-redirect when unauthorized',
    implemented: true,
    tested: false,
  },
  {
    category: 'Authentication',
    feature: 'Bearer token in API headers',
    implemented: true,
    tested: false,
  },
  {
    category: 'Authentication',
    feature: 'Logout functionality',
    implemented: true,
    tested: false,
  },

  // Job Creation
  {
    category: 'Job Creation',
    feature: 'Directory scanning with WebSocket',
    implemented: true,
    tested: false,
  },
  {
    category: 'Job Creation',
    feature: 'Real-time scan progress',
    implemented: true,
    tested: false,
  },
  {
    category: 'Job Creation',
    feature: 'Scan results display (files, size)',
    implemented: true,
    tested: false,
  },
  {
    category: 'Job Creation',
    feature: 'Collection name input',
    implemented: true,
    tested: false,
  },
  {
    category: 'Job Creation',
    feature: 'Form validation (scan required)',
    implemented: true,
    tested: false,
  },
  {
    category: 'Job Creation',
    feature: 'Job creation with API',
    implemented: true,
    tested: false,
  },
  {
    category: 'Job Creation',
    feature: 'Advanced parameters (chunk size, etc)',
    implemented: true,
    tested: false,
    notes: 'All parameters implemented with collapsible UI',
  },
  {
    category: 'Job Creation',
    feature: 'Model selection dropdown',
    implemented: true,
    tested: false,
    notes: 'Dynamically loads available models from API',
  },

  // Job Management
  {
    category: 'Job Management',
    feature: 'Job list with status grouping',
    implemented: true,
    tested: false,
  },
  {
    category: 'Job Management',
    feature: 'Real-time progress via WebSocket',
    implemented: true,
    tested: false,
  },
  {
    category: 'Job Management',
    feature: 'Progress bar and percentages',
    implemented: true,
    tested: false,
  },
  {
    category: 'Job Management',
    feature: 'Processing metrics (docs/s, ETA)',
    implemented: true,
    tested: false,
  },
  {
    category: 'Job Management',
    feature: 'Job deletion',
    implemented: true,
    tested: false,
  },
  {
    category: 'Job Management',
    feature: 'Job metrics modal',
    implemented: true,
    tested: false,
    notes: 'Modal component implemented with full metrics display',
  },
  {
    category: 'Job Management',
    feature: 'Status badges with colors',
    implemented: true,
    tested: false,
  },
  {
    category: 'Job Management',
    feature: 'Error display for failed jobs',
    implemented: true,
    tested: false,
  },

  // Search
  {
    category: 'Search',
    feature: 'Basic vector search',
    implemented: true,
    tested: false,
  },
  {
    category: 'Search',
    feature: 'Hybrid search mode',
    implemented: true,
    tested: false,
  },
  {
    category: 'Search',
    feature: 'Collection selection',
    implemented: true,
    tested: false,
  },
  {
    category: 'Search',
    feature: 'Advanced options (top_k, threshold)',
    implemented: true,
    tested: false,
  },
  {
    category: 'Search',
    feature: 'Hybrid alpha configuration',
    implemented: true,
    tested: false,
  },
  {
    category: 'Search',
    feature: 'Results grouped by document',
    implemented: true,
    tested: false,
  },
  {
    category: 'Search',
    feature: 'Chunk preview with score',
    implemented: true,
    tested: false,
  },
  {
    category: 'Search',
    feature: 'Document viewer integration',
    implemented: true,
    tested: false,
  },

  // Document Viewer
  {
    category: 'Document Viewer',
    feature: 'PDF rendering with PDF.js',
    implemented: true,
    tested: false,
  },
  {
    category: 'Document Viewer',
    feature: 'DOCX support with Mammoth',
    implemented: true,
    tested: false,
  },
  {
    category: 'Document Viewer',
    feature: 'Markdown rendering',
    implemented: true,
    tested: false,
  },
  {
    category: 'Document Viewer',
    feature: 'Search term highlighting',
    implemented: true,
    tested: false,
  },
  {
    category: 'Document Viewer',
    feature: 'Highlight navigation',
    implemented: true,
    tested: false,
  },
  {
    category: 'Document Viewer',
    feature: 'PDF page navigation',
    implemented: true,
    tested: false,
  },
  {
    category: 'Document Viewer',
    feature: 'Download functionality',
    implemented: true,
    tested: false,
  },
  {
    category: 'Document Viewer',
    feature: 'Email (EML) support',
    implemented: true,
    tested: false,
  },

  // UI/UX
  {
    category: 'UI/UX',
    feature: 'Tab navigation (Create/Jobs/Search)',
    implemented: true,
    tested: false,
  },
  {
    category: 'UI/UX',
    feature: 'Toast notifications',
    implemented: true,
    tested: false,
  },
  {
    category: 'UI/UX',
    feature: 'Loading states and spinners',
    implemented: true,
    tested: false,
  },
  {
    category: 'UI/UX',
    feature: 'Responsive design',
    implemented: true,
    tested: false,
    notes: 'Using Tailwind CSS classes',
  },
  {
    category: 'UI/UX',
    feature: 'Empty states',
    implemented: true,
    tested: false,
  },

  // WebSocket
  {
    category: 'WebSocket',
    feature: 'Auto-reconnection logic',
    implemented: true,
    tested: false,
  },
  {
    category: 'WebSocket',
    feature: 'Error handling',
    implemented: true,
    tested: false,
  },
  {
    category: 'WebSocket',
    feature: 'Clean disconnection on unmount',
    implemented: true,
    tested: false,
  },

  // API Integration
  {
    category: 'API Integration',
    feature: 'Centralized API client',
    implemented: true,
    tested: false,
  },
  {
    category: 'API Integration',
    feature: 'Auth interceptors',
    implemented: true,
    tested: false,
  },
  {
    category: 'API Integration',
    feature: 'Error response handling',
    implemented: true,
    tested: false,
  },
  {
    category: 'API Integration',
    feature: 'All original endpoints covered',
    implemented: true,
    tested: false,
  },
];

export function getImplementationStats() {
  const total = featureChecklist.length;
  const implemented = featureChecklist.filter(f => f.implemented).length;
  const tested = featureChecklist.filter(f => f.tested).length;
  
  return {
    total,
    implemented,
    tested,
    implementationPercentage: Math.round((implemented / total) * 100),
    testingPercentage: Math.round((tested / total) * 100),
  };
}

export function getMissingFeatures() {
  return featureChecklist.filter(f => !f.implemented);
}

export function getUntestedFeatures() {
  return featureChecklist.filter(f => f.implemented && !f.tested);
}