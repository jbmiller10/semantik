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

  // Collection Management
  {
    category: 'Collection Management',
    feature: 'Create new collection',
    implemented: true,
    tested: false,
  },
  {
    category: 'Collection Management',
    feature: 'List all collections',
    implemented: true,
    tested: false,
  },
  {
    category: 'Collection Management',
    feature: 'View collection details',
    implemented: true,
    tested: false,
  },
  {
    category: 'Collection Management',
    feature: 'Rename collection',
    implemented: true,
    tested: false,
  },
  {
    category: 'Collection Management',
    feature: 'Delete collection',
    implemented: true,
    tested: false,
  },
  {
    category: 'Collection Management',
    feature: 'Add data to collection',
    implemented: true,
    tested: false,
  },
  {
    category: 'Collection Management',
    feature: 'Reindex collection',
    implemented: true,
    tested: false,
  },
  {
    category: 'Collection Management',
    feature: 'Collection status display',
    implemented: true,
    tested: false,
  },

  // Operation Management
  {
    category: 'Operation Management',
    feature: 'Active operations list',
    implemented: true,
    tested: false,
  },
  {
    category: 'Operation Management',
    feature: 'Real-time progress tracking',
    implemented: true,
    tested: false,
  },
  {
    category: 'Operation Management',
    feature: 'Progress bar visualization',
    implemented: true,
    tested: false,
  },
  {
    category: 'Operation Management',
    feature: 'Processing metrics display',
    implemented: true,
    tested: false,
  },
  {
    category: 'Operation Management',
    feature: 'Operation status badges',
    implemented: true,
    tested: false,
  },
  {
    category: 'Operation Management',
    feature: 'Error display for failed operations',
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
    feature: 'Tab navigation (Collections/Search/Operations)',
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