/**
 * DatasetsView - Dataset list view with upload and management
 */

import { useState } from 'react';
import { Upload, Search, Database, Loader2 } from 'lucide-react';
import { useBenchmarkDatasets, useDeleteDataset, useDatasetMappings } from '../../hooks/useBenchmarks';
import { DatasetCard } from './DatasetCard';
import { DatasetUploadModal } from './DatasetUploadModal';
import { MappingManagementPanel } from './MappingManagementPanel';
import type { BenchmarkDataset } from '../../types/benchmark';

interface DatasetsViewProps {
  onDatasetSelect?: (datasetId: string) => void;
}

export function DatasetsView({ onDatasetSelect }: DatasetsViewProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<BenchmarkDataset | null>(null);

  const { data: datasetsResponse, isLoading, error } = useBenchmarkDatasets();
  const deleteDatasetMutation = useDeleteDataset();

  const datasets = datasetsResponse?.datasets ?? [];

  // Filter datasets by search query
  const filteredDatasets = datasets.filter((dataset) =>
    dataset.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    dataset.description?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleDelete = (datasetId: string) => {
    if (confirm('Are you sure you want to delete this dataset? This action cannot be undone.')) {
      deleteDatasetMutation.mutate(datasetId);
    }
  };

  const handleViewMappings = (dataset: BenchmarkDataset) => {
    setSelectedDataset(dataset);
    onDatasetSelect?.(dataset.id);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-[var(--text-muted)]" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
        <p className="text-red-400">Failed to load datasets: {error.message}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row gap-4 sm:items-center sm:justify-between">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[var(--text-muted)]" />
          <input
            type="text"
            placeholder="Search datasets..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 input-field rounded-xl"
          />
        </div>
        <button
          onClick={() => setIsUploadModalOpen(true)}
          className="inline-flex items-center gap-2 px-4 py-2 bg-gray-200 dark:bg-white text-gray-900 font-medium rounded-xl hover:bg-gray-300 dark:hover:bg-gray-100 transition-colors"
        >
          <Upload className="h-4 w-4" />
          Upload Dataset
        </button>
      </div>

      {/* Dataset Grid or Mapping Panel */}
      {selectedDataset ? (
        <MappingManagementPanel
          dataset={selectedDataset}
          onBack={() => setSelectedDataset(null)}
        />
      ) : (
        <>
          {filteredDatasets.length === 0 ? (
            <div className="text-center py-12 bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl">
              <Database className="mx-auto h-12 w-12 text-[var(--text-muted)]" />
              <h3 className="mt-4 text-sm font-medium text-[var(--text-primary)]">
                {searchQuery ? 'No datasets found' : 'No datasets yet'}
              </h3>
              <p className="mt-2 text-sm text-[var(--text-muted)]">
                {searchQuery
                  ? 'Try a different search term'
                  : 'Upload a benchmark dataset to get started'}
              </p>
              {!searchQuery && (
                <button
                  onClick={() => setIsUploadModalOpen(true)}
                  className="mt-4 inline-flex items-center gap-2 px-4 py-2 bg-[var(--bg-tertiary)] border border-[var(--border)] text-[var(--text-primary)] font-medium rounded-xl hover:bg-[var(--bg-secondary)] transition-colors"
                >
                  <Upload className="h-4 w-4" />
                  Upload Your First Dataset
                </button>
              )}
            </div>
          ) : (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {filteredDatasets.map((dataset) => (
                <DatasetCardWithMappings
                  key={dataset.id}
                  dataset={dataset}
                  onViewMappings={() => handleViewMappings(dataset)}
                  onDelete={() => handleDelete(dataset.id)}
                  isDeleting={deleteDatasetMutation.isPending && deleteDatasetMutation.variables === dataset.id}
                />
              ))}
            </div>
          )}
        </>
      )}

      {/* Upload Modal */}
      {isUploadModalOpen && (
        <DatasetUploadModal
          onClose={() => setIsUploadModalOpen(false)}
          onSuccess={() => setIsUploadModalOpen(false)}
        />
      )}
    </div>
  );
}

// Wrapper component to fetch mapping count for each dataset
function DatasetCardWithMappings({
  dataset,
  onViewMappings,
  onDelete,
  isDeleting,
}: {
  dataset: BenchmarkDataset;
  onViewMappings: () => void;
  onDelete: () => void;
  isDeleting: boolean;
}) {
  const { data: mappings } = useDatasetMappings(dataset.id);
  const mappingCount = mappings?.length ?? 0;

  return (
    <DatasetCard
      dataset={dataset}
      mappingCount={mappingCount}
      onViewMappings={onViewMappings}
      onDelete={onDelete}
      isDeleting={isDeleting}
    />
  );
}
