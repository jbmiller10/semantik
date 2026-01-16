import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { operationsV2Api } from '../services/api/v2/operations';

import type { Operation } from '../types/collection';
import { RefreshCw, Activity, Clock, AlertCircle } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useUIStore } from '../stores/uiStore';
import { useCollections } from '../hooks/useCollections';
import { useAnimationEnabled } from '../contexts/AnimationContext';
import { withAnimation } from '../utils/animationClasses';
// Note: useOperationsSocket moved to Layout.tsx for global coverage

/**
 * Extract display source from operation config.
 * Prefers source_config.path when present, falls back to source_path.
 */
function getSourceDisplay(config: Record<string, unknown> | undefined): string | null {
  if (!config) return null;

  // New format: source_config.path or source_config.url
  if ('source_config' in config && typeof config.source_config === 'object' && config.source_config !== null) {
    const sourceConfig = config.source_config as Record<string, unknown>;
    if ('path' in sourceConfig && typeof sourceConfig.path === 'string') {
      return sourceConfig.path;
    }
    if ('url' in sourceConfig && typeof sourceConfig.url === 'string') {
      return sourceConfig.url;
    }
  }

  // Legacy format: source_path
  if ('source_path' in config && typeof config.source_path === 'string') {
    return config.source_path;
  }

  return null;
}

function ActiveOperationsTab() {
  const navigate = useNavigate();
  const setActiveTab = useUIStore((state) => state.setActiveTab);
  const setShowCollectionDetailsModal = useUIStore((state) => state.setShowCollectionDetailsModal);
  const { data: collections = [] } = useCollections();
  const animationEnabled = useAnimationEnabled();

  const collectionNameById = useMemo(() => {
    return collections.reduce<Record<string, string>>((acc, collection) => {
      acc[collection.id] = collection.name;
      return acc;
    }, {});
  }, [collections]);

  // Fetch active operations across all collections
  const pollingPreference =
    typeof window !== 'undefined'
      ? (window as Window & { __activeOperationsPolling?: boolean }).__activeOperationsPolling
      : undefined;

  const shouldPollActiveOperations = pollingPreference ?? true;

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['active-operations'],
    queryFn: async () => {
      const response = await operationsV2Api.list({
        status: 'processing,pending',
        page: 1,
        per_page: 100,
      });
      return response.operations;
    },
    refetchInterval: shouldPollActiveOperations ? 5000 : false,
  });

  // Get collection name for an operation
  const getCollectionName = (collectionId: string) => {
    return collectionNameById[collectionId] ?? `Collection ${collectionId}`;
  };

  // Navigate to collection details
  const navigateToCollection = (collectionId: string) => {
    setActiveTab('collections');
    setShowCollectionDetailsModal(collectionId);
    navigate(`/collections/${collectionId}`);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className={withAnimation('rounded-full h-8 w-8 border-b-2 border-blue-600', animationEnabled, 'animate-spin')}></div>
      </div>
    );
  }

  if (error) {
    const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
        <div className="flex">
          <AlertCircle className="h-5 w-5 text-red-400 mt-0.5" />
          <div className="ml-3">
            <p className="text-sm text-red-300">Failed to load active operations</p>
            <p className="text-xs text-red-400 mt-1">{errorMessage}</p>
            <button
              onClick={() => refetch()}
              className="text-sm text-red-400 hover:text-red-300 mt-2 underline"
            >
              Try again
            </button>
          </div>
        </div>
      </div>
    );
  }

  const operations = data || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Activity className="h-6 w-6 text-[var(--text-muted)]" />
            <div>
              <h2 className="text-xl font-semibold text-[var(--text-primary)]">Active Operations</h2>
              <p className="text-sm text-[var(--text-muted)] mt-1">
                Monitor all ongoing operations across your collections
              </p>
            </div>
          </div>
          <button
            onClick={() => refetch()}
            className="inline-flex items-center px-3 py-2 border border-[var(--border)] text-sm font-medium rounded-md text-[var(--text-secondary)] bg-[var(--bg-secondary)] hover:bg-[var(--bg-tertiary)]"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </button>
        </div>
      </div>

      {/* Operations List */}
      {operations.length === 0 ? (
        <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg">
          <div className="text-center py-12">
            <Activity className="mx-auto h-12 w-12 text-[var(--text-muted)]" />
            <h3 className="mt-4 text-sm font-medium text-[var(--text-primary)]">No active operations</h3>
            <p className="mt-2 text-sm text-[var(--text-muted)]">
              All operations have completed. Start a new operation from any collection.
            </p>
          </div>
        </div>
      ) : (
        <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-lg overflow-hidden">
          <ul className="divide-y divide-[var(--border)]">
            {operations.map((operation) => (
              <OperationListItem
                key={operation.id}
                operation={operation}
                collectionName={getCollectionName(operation.collection_id)}
                onNavigateToCollection={() => navigateToCollection(operation.collection_id)}
              />
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

interface OperationListItemProps {
  operation: Operation;
  collectionName: string;
  onNavigateToCollection: () => void;
}

function OperationListItem({ operation, collectionName, onNavigateToCollection }: OperationListItemProps) {
  // Progress updates handled by global useOperationsSocket in parent component
  // Removed per-operation WebSocket to avoid exceeding connection limits

  const formatOperationType = (type: string) => {
    switch (type) {
      case 'index': return 'Initial Index';
      case 'append': return 'Add Source';
      case 'reindex': return 'Re-index';
      case 'remove_source': return 'Remove Source';
      default: return type;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'processing': return 'text-blue-400 bg-blue-500/20';
      case 'pending': return 'text-yellow-400 bg-yellow-500/20';
      case 'completed': return 'text-green-400 bg-green-500/20';
      case 'failed': return 'text-red-400 bg-red-500/20';
      case 'cancelled': return 'text-gray-400 bg-gray-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  const getOperationIcon = (type: string) => {
    switch (type) {
      case 'index': return '📊';
      case 'append': return '➕';
      case 'reindex': return '🔄';
      case 'remove_source': return '➖';
      default: return '📋';
    }
  };

  return (
    <li className="px-6 py-4 hover:bg-[var(--bg-tertiary)] transition-colors">
      <div className="flex items-center justify-between">
        <div className="flex-1 min-w-0">
          <div className="flex items-center space-x-3">
            <span className="text-2xl">{getOperationIcon(operation.type)}</span>
            <div className="flex-1">
              <div className="flex items-center space-x-2">
                <h4 className="text-sm font-medium text-[var(--text-primary)]">
                  {formatOperationType(operation.type)}
                </h4>
                <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${getStatusColor(operation.status)}`}>
                  {operation.status}
                </span>
              </div>
              <div className="mt-1 flex items-center space-x-4 text-sm text-[var(--text-muted)]">
                <button
                  onClick={onNavigateToCollection}
                  className="hover:text-[var(--text-primary)] hover:underline"
                >
                  {collectionName}
                </button>
                <span className="flex items-center space-x-1">
                  <Clock className="h-3 w-3" />
                  <span>
                    Started {formatDistanceToNow(new Date(operation.created_at), { addSuffix: true })}
                  </span>
                </span>
                {(() => {
                  const sourceDisplay = getSourceDisplay(operation.config);
                  return sourceDisplay ? (
                    <span className="truncate max-w-xs" title={sourceDisplay}>
                      {sourceDisplay}
                    </span>
                  ) : null;
                })()}
              </div>
            </div>
          </div>

          {/* Progress bar for processing operations */}
          {operation.status === 'processing' && operation.progress !== undefined && (
            <div className="mt-3">
              <div className="flex justify-between text-sm text-[var(--text-secondary)] mb-1">
                <span>Progress</span>
                <span>{Math.round(operation.progress)}%</span>
              </div>
              <div className="w-full bg-[var(--border)] rounded-full h-2 overflow-hidden">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300 relative"
                  style={{ width: `${operation.progress}%` }}
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer"></div>
                </div>
              </div>
              {operation.eta && (
                <p className="mt-1 text-xs text-[var(--text-muted)]">
                  ETA: {Math.ceil(operation.eta / 60)} minutes remaining
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </li>
  );
}

export default ActiveOperationsTab;
