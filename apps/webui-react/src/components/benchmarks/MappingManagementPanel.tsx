/**
 * MappingManagementPanel - Panel for managing dataset-collection mappings
 */

import { useMemo, useState } from 'react';
import { ArrowLeft, Link2, CheckCircle, AlertCircle, Loader2, Plus, RefreshCw, Wifi, WifiOff } from 'lucide-react';
import { useDatasetMappings, useCreateMapping, useResolveMapping } from '../../hooks/useBenchmarks';
import { useCollections } from '../../hooks/useCollections';
import { useMappingResolutionProgress } from '../../hooks/useMappingResolutionProgress';
import type { BenchmarkDataset, DatasetMapping, MappingStatus } from '../../types/benchmark';

interface MappingManagementPanelProps {
  dataset: BenchmarkDataset;
  onBack: () => void;
}

function getStatusBadge(status: MappingStatus) {
  switch (status) {
    case 'resolved':
      return (
        <span className="inline-flex items-center gap-1 px-2 py-1 bg-green-500/20 text-green-400 rounded-full text-xs font-medium">
          <CheckCircle className="h-3 w-3" />
          Resolved
        </span>
      );
    case 'partial':
      return (
        <span className="inline-flex items-center gap-1 px-2 py-1 bg-amber-500/20 text-amber-400 rounded-full text-xs font-medium">
          <AlertCircle className="h-3 w-3" />
          Partial
        </span>
      );
    case 'pending':
    default:
      return (
        <span className="inline-flex items-center gap-1 px-2 py-1 bg-blue-500/20 text-blue-400 rounded-full text-xs font-medium">
          Pending
        </span>
      );
  }
}

export function MappingManagementPanel({ dataset, onBack }: MappingManagementPanelProps) {
  const [selectedCollectionId, setSelectedCollectionId] = useState<string>('');
  const [isCreating, setIsCreating] = useState(false);
  const [activeResolution, setActiveResolution] = useState<{ mappingId: number; operationUuid: string } | null>(null);

  const { data: mappings, isLoading } = useDatasetMappings(dataset.id);
  const { data: collections } = useCollections();
  const createMappingMutation = useCreateMapping();
  const resolveMappingMutation = useResolveMapping();

  const progressOptions = useMemo(() => {
    if (!activeResolution) return null;
    return {
      datasetId: dataset.id,
      onComplete: () => setActiveResolution(null),
      onError: () => setActiveResolution(null),
    };
  }, [activeResolution, dataset.id]);

  const { progress: resolutionProgress, isConnected: isResolutionConnected } = useMappingResolutionProgress(
    activeResolution?.operationUuid ?? null,
    progressOptions
  );

  // Filter out collections that already have a mapping
  const mappedCollectionIds = new Set(mappings?.map((m) => m.collection_id) ?? []);
  const availableCollections = collections?.filter((c) => !mappedCollectionIds.has(c.id)) ?? [];

  const handleCreateMapping = async () => {
    if (!selectedCollectionId) return;

    try {
      await createMappingMutation.mutateAsync({
        datasetId: dataset.id,
        data: { collection_id: selectedCollectionId },
      });
      setSelectedCollectionId('');
      setIsCreating(false);
    } catch {
      // Error handled by mutation
    }
  };

  const handleResolve = async (mapping: DatasetMapping) => {
    try {
      const result = await resolveMappingMutation.mutateAsync({
        datasetId: dataset.id,
        mappingId: mapping.id,
      });
      if (result.operation_uuid) {
        setActiveResolution({ mappingId: mapping.id, operationUuid: result.operation_uuid });
      } else {
        setActiveResolution(null);
      }
    } catch {
      // Error handled by mutation
    }
  };

  const getCollectionName = (collectionId: string) => {
    return collections?.find((c) => c.id === collectionId)?.name ?? collectionId;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button
          onClick={onBack}
          className="p-2 text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded-lg transition-colors"
        >
          <ArrowLeft className="h-5 w-5" />
        </button>
        <div>
          <h3 className="text-lg font-bold text-[var(--text-primary)]">{dataset.name}</h3>
          <p className="text-sm text-[var(--text-muted)]">
            {dataset.query_count} queries • Manage collection mappings
          </p>
        </div>
      </div>

      {/* Add Mapping */}
      <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-4">
        {isCreating ? (
          <div className="flex items-end gap-3">
            <div className="flex-1">
              <label className="block text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider mb-2">
                Select Collection
              </label>
              <select
                value={selectedCollectionId}
                onChange={(e) => setSelectedCollectionId(e.target.value)}
                disabled={createMappingMutation.isPending}
                className="w-full px-4 py-2.5 input-field rounded-xl"
              >
                <option value="">Choose a collection...</option>
                {availableCollections.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name} ({c.document_count} docs)
                  </option>
                ))}
              </select>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setIsCreating(false)}
                disabled={createMappingMutation.isPending}
                className="px-4 py-2.5 text-sm font-medium text-[var(--text-secondary)] border border-[var(--border)] rounded-xl hover:bg-[var(--bg-tertiary)] transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateMapping}
                disabled={!selectedCollectionId || createMappingMutation.isPending}
                className="px-4 py-2.5 text-sm font-bold text-gray-900 bg-gray-200 dark:bg-white rounded-xl hover:bg-gray-300 dark:hover:bg-gray-100 transition-colors disabled:opacity-50"
              >
                {createMappingMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  'Create'
                )}
              </button>
            </div>
          </div>
        ) : (
          <button
            onClick={() => setIsCreating(true)}
            disabled={availableCollections.length === 0}
            className="flex items-center gap-2 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Plus className="h-4 w-4" />
            {availableCollections.length > 0
              ? 'Add Collection Mapping'
              : 'All collections are already mapped'}
          </button>
        )}
      </div>

      {/* Mappings List */}
      {isLoading ? (
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin text-[var(--text-muted)]" />
        </div>
      ) : mappings && mappings.length > 0 ? (
        <div className="space-y-3">
          <h4 className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-wider">
            Existing Mappings
          </h4>
          <div className="space-y-2">
            {mappings.map((mapping) => (
              <div
                key={mapping.id}
                className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-4"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-[var(--bg-tertiary)] rounded-lg">
                      <Link2 className="h-5 w-5 text-[var(--text-muted)]" />
                    </div>
                    <div>
                      <p className="font-medium text-[var(--text-primary)]">
                        {getCollectionName(mapping.collection_id)}
                      </p>
                      <p className="text-sm text-[var(--text-muted)]">
                        {mapping.mapped_count}/{mapping.total_count} documents resolved
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    {getStatusBadge(mapping.mapping_status)}
                    {mapping.mapping_status !== 'resolved' && (
                      <button
                        onClick={() => handleResolve(mapping)}
                        disabled={resolveMappingMutation.isPending || activeResolution?.mappingId === mapping.id}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-[var(--text-secondary)] border border-[var(--border)] rounded-lg hover:bg-[var(--bg-tertiary)] transition-colors disabled:opacity-50"
                      >
                        {resolveMappingMutation.isPending ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : activeResolution?.mappingId === mapping.id ? (
                          <>
                            <Loader2 className="h-4 w-4 animate-spin" />
                            Resolving...
                          </>
                        ) : (
                          <>
                            <RefreshCw className="h-4 w-4" />
                            Resolve
                          </>
                        )}
                      </button>
                    )}
                  </div>
                </div>

                {/* Progress Bar */}
                {mapping.total_count > 0 && (
                  <div className="mt-3">
                    <div className="h-1.5 bg-[var(--bg-tertiary)] rounded-full overflow-hidden">
                      <div
                        className={`h-full transition-all ${
                          mapping.mapping_status === 'resolved'
                            ? 'bg-green-500'
                            : mapping.mapping_status === 'partial'
                            ? 'bg-amber-500'
                            : 'bg-blue-500'
                        }`}
                        style={{
                          width: `${(mapping.mapped_count / mapping.total_count) * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                )}

                {/* Live resolution progress (async) */}
                {activeResolution?.mappingId === mapping.id && (
                  <div className="mt-3 p-3 bg-[var(--bg-tertiary)] rounded-lg">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-[var(--text-muted)]">
                        {resolutionProgress.stage.replace('_', ' ')}
                      </span>
                      <span
                        className={`flex items-center gap-1 ${
                          isResolutionConnected ? 'text-green-400' : 'text-amber-400'
                        }`}
                      >
                        {isResolutionConnected ? <Wifi className="h-3.5 w-3.5" /> : <WifiOff className="h-3.5 w-3.5" />}
                        {isResolutionConnected ? 'Live' : 'Reconnecting...'}
                      </span>
                    </div>
                    <div className="mt-2 flex items-center justify-between text-xs text-[var(--text-secondary)]">
                      <span>
                        {resolutionProgress.processedRefs} / {resolutionProgress.totalRefs} processed
                      </span>
                      <span>
                        {resolutionProgress.resolvedRefs} resolved • {resolutionProgress.ambiguousRefs} ambiguous •{' '}
                        {resolutionProgress.unresolvedRefs} unmatched
                      </span>
                    </div>
                    <div className="mt-2 h-1.5 bg-[var(--bg-secondary)] rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500 transition-all duration-200"
                        style={{
                          width:
                            resolutionProgress.totalRefs > 0
                              ? `${Math.min(
                                  100,
                                  (resolutionProgress.processedRefs / resolutionProgress.totalRefs) * 100
                                )}%`
                              : '0%',
                        }}
                      />
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="text-center py-8 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl">
          <Link2 className="mx-auto h-10 w-10 text-[var(--text-muted)]" />
          <h4 className="mt-3 text-sm font-medium text-[var(--text-primary)]">No mappings yet</h4>
          <p className="mt-1 text-sm text-[var(--text-muted)]">
            Create a mapping to link this dataset to a collection
          </p>
        </div>
      )}
    </div>
  );
}
