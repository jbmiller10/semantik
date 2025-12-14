import { useState, useCallback } from 'react';
import { Loader2, Database, Users } from 'lucide-react';
import { useGraphStats } from '@/hooks/useGraph';
import { EntityBrowser } from './EntityBrowser';
import { GraphExplorer } from './GraphExplorer';
import type { EntityResponse } from '@/types/graph';

// ============================================================================
// Types
// ============================================================================

export interface EntitiesTabProps {
  /** Collection UUID to browse entities for */
  collectionId: string;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Sub-Components
// ============================================================================

/** Loading spinner for initial load */
function LoadingState() {
  return (
    <div className="flex flex-col items-center justify-center h-[600px]">
      <Loader2 className="w-10 h-10 text-blue-500 animate-spin mb-4" />
      <p className="text-gray-600 font-medium">Loading graph data...</p>
    </div>
  );
}

/** Message when graph extraction is not enabled */
function GraphDisabledState() {
  return (
    <div className="flex flex-col items-center justify-center h-[600px] bg-gray-50 rounded-lg">
      <Database className="w-16 h-16 text-gray-300 mb-6" />
      <h3 className="text-xl font-semibold text-gray-900 mb-3">
        Graph Extraction Not Enabled
      </h3>
      <p className="text-gray-600 text-center max-w-md mb-6">
        This collection was not indexed with graph extraction enabled.
        To explore entities and relationships, re-index the collection
        with the graph extraction option turned on.
      </p>
      <div className="text-sm text-gray-500">
        Go to <span className="font-medium">Settings</span> tab to re-index with graph extraction.
      </div>
    </div>
  );
}

/** Message when graph is enabled but has no entities */
function NoEntitiesState() {
  return (
    <div className="flex flex-col items-center justify-center h-[600px] bg-gray-50 rounded-lg">
      <Users className="w-16 h-16 text-gray-300 mb-6" />
      <h3 className="text-xl font-semibold text-gray-900 mb-3">
        No Entities Found
      </h3>
      <p className="text-gray-600 text-center max-w-md">
        Graph extraction is enabled, but no entities have been extracted yet.
        This could mean the documents are still being processed, or no
        recognizable entities were found in the content.
      </p>
    </div>
  );
}

/** Panel header with title and subtitle */
interface PanelHeaderProps {
  title: string;
  subtitle: string;
}

function PanelHeader({ title, subtitle }: PanelHeaderProps) {
  return (
    <div className="pb-3 mb-3 border-b border-gray-200">
      <h3 className="text-base font-semibold text-gray-900">{title}</h3>
      <p className="text-sm text-gray-500 mt-0.5">{subtitle}</p>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

/**
 * EntitiesTab component - A split-pane interface for exploring the knowledge graph.
 *
 * Features:
 * - Left panel (1/3): EntityBrowser for searching and filtering entities
 * - Right panel (2/3): GraphExplorer for visualizing relationships
 * - State synchronization between panels
 * - Handles graph disabled, no entities, and loading states
 * - Fixed height of 600px for consistent modal layout
 */
export function EntitiesTab({ collectionId, className = '' }: EntitiesTabProps) {
  // Track the currently selected entity
  const [selectedEntityId, setSelectedEntityId] = useState<number | null>(null);

  // Fetch graph stats to check if graph is enabled and has entities
  const {
    data: stats,
    isLoading: statsLoading,
    error: statsError,
  } = useGraphStats(collectionId);

  // Handle entity selection from the browser
  const handleEntitySelect = useCallback((entity: EntityResponse) => {
    setSelectedEntityId(entity.id);
  }, []);

  // Handle entity selection from the graph (double-click on node)
  const handleGraphEntitySelect = useCallback((entityId: number) => {
    setSelectedEntityId(entityId);
  }, []);

  // Loading state
  if (statsLoading) {
    return (
      <div className={className}>
        <LoadingState />
      </div>
    );
  }

  // Error state
  if (statsError) {
    return (
      <div className={`flex flex-col items-center justify-center h-[600px] ${className}`}>
        <div className="text-red-500 mb-4">
          <Database className="w-12 h-12" />
        </div>
        <h3 className="text-lg font-medium text-gray-900 mb-2">
          Error Loading Graph Data
        </h3>
        <p className="text-sm text-red-600 text-center max-w-sm">
          {statsError instanceof Error ? statsError.message : 'An unexpected error occurred'}
        </p>
      </div>
    );
  }

  // Graph not enabled state
  if (stats && !stats.graph_enabled) {
    return (
      <div className={className}>
        <GraphDisabledState />
      </div>
    );
  }

  // No entities state (graph enabled but empty)
  if (stats && stats.graph_enabled && stats.total_entities === 0) {
    return (
      <div className={className}>
        <NoEntitiesState />
      </div>
    );
  }

  // Main split-pane layout
  return (
    <div className={`flex gap-6 h-[600px] ${className}`}>
      {/* Left Panel - Entity Browser (1/3 width) */}
      <div className="w-1/3 flex flex-col bg-white rounded-lg border border-gray-200 p-4 overflow-hidden">
        <PanelHeader
          title="Entity Browser"
          subtitle="Search and filter entities"
        />
        <div className="flex-1 overflow-hidden">
          <EntityBrowser
            collectionId={collectionId}
            onEntitySelect={handleEntitySelect}
            selectedEntityId={selectedEntityId}
            className="h-full"
          />
        </div>
      </div>

      {/* Right Panel - Graph Explorer (2/3 width) */}
      <div className="w-2/3 flex flex-col bg-white rounded-lg border border-gray-200 p-4 overflow-hidden">
        <PanelHeader
          title="Graph Explorer"
          subtitle="Visualize entity relationships"
        />
        <div className="flex-1 overflow-hidden">
          <GraphExplorer
            collectionId={collectionId}
            initialEntityId={selectedEntityId}
            onEntitySelect={handleGraphEntitySelect}
            className="h-full w-full"
            maxHops={2}
          />
        </div>
      </div>
    </div>
  );
}

export default EntitiesTab;
