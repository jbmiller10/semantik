import { useState, useCallback, useMemo, useEffect } from 'react';
import { Search, Users, Link2, Circle, AlertCircle, Database } from 'lucide-react';
import { useGraphStats, useEntityTypes, useEntitySearch } from '@/hooks/useGraph';
import { getEntityColor, getEntityBackgroundColor } from '@/types/graph';
import type { EntityResponse } from '@/types/graph';

// ============================================================================
// Types
// ============================================================================

export interface EntityBrowserProps {
  /** Collection UUID to browse entities for */
  collectionId: string;
  /** Callback when an entity is selected */
  onEntitySelect?: (entity: EntityResponse) => void;
  /** Currently selected entity ID */
  selectedEntityId?: number | null;
  /** Additional CSS classes */
  className?: string;
}

// ============================================================================
// Debounce Hook
// ============================================================================

function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(timer);
    };
  }, [value, delay]);

  return debouncedValue;
}

// ============================================================================
// Sub-Components
// ============================================================================

/** Skeleton loader for stats */
function StatsSkeleton() {
  return (
    <div className="flex gap-4 animate-pulse">
      <div className="flex items-center gap-2">
        <div className="w-4 h-4 bg-gray-200 rounded" />
        <div className="w-16 h-4 bg-gray-200 rounded" />
      </div>
      <div className="flex items-center gap-2">
        <div className="w-4 h-4 bg-gray-200 rounded" />
        <div className="w-16 h-4 bg-gray-200 rounded" />
      </div>
    </div>
  );
}

/** Skeleton loader for entity list */
function EntityListSkeleton() {
  return (
    <div className="space-y-2 animate-pulse">
      {[1, 2, 3, 4, 5].map((i) => (
        <div key={i} className="flex items-center gap-3 p-3 rounded-lg bg-gray-50">
          <div className="w-3 h-3 bg-gray-200 rounded-full" />
          <div className="flex-1">
            <div className="w-32 h-4 bg-gray-200 rounded mb-1" />
            <div className="w-16 h-3 bg-gray-200 rounded" />
          </div>
          <div className="w-10 h-4 bg-gray-200 rounded" />
        </div>
      ))}
    </div>
  );
}

/** Type filter chip component */
interface TypeChipProps {
  type: string;
  count: number;
  isSelected: boolean;
  onClick: () => void;
}

function TypeChip({ type, count, isSelected, onClick }: TypeChipProps) {
  const color = getEntityColor(type);
  const bgColor = getEntityBackgroundColor(type);

  return (
    <button
      onClick={onClick}
      className={`
        inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium
        transition-all duration-150 border
        ${isSelected
          ? 'ring-2 ring-offset-1'
          : 'hover:opacity-80'
        }
      `}
      style={{
        backgroundColor: isSelected ? color : bgColor,
        color: isSelected ? 'white' : color,
        borderColor: color,
        ...(isSelected ? { ringColor: color } : {}),
      }}
      aria-pressed={isSelected}
      aria-label={`Filter by ${type} (${count} entities)`}
    >
      <Circle
        className="w-2 h-2"
        fill={isSelected ? 'white' : color}
        stroke="none"
      />
      <span>{type}</span>
      <span className={`text-xs ${isSelected ? 'text-white/80' : 'opacity-70'}`}>
        {count}
      </span>
    </button>
  );
}

/** Single entity list item */
interface EntityItemProps {
  entity: EntityResponse;
  isSelected: boolean;
  onClick: () => void;
}

function EntityItem({ entity, isSelected, onClick }: EntityItemProps) {
  const color = getEntityColor(entity.entity_type);
  const bgColor = getEntityBackgroundColor(entity.entity_type);
  const confidencePercent = Math.round(entity.confidence * 100);

  return (
    <button
      onClick={onClick}
      className={`
        w-full text-left flex items-center gap-3 p-3 rounded-lg
        transition-all duration-150 border
        ${isSelected
          ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
          : 'border-transparent hover:bg-gray-50'
        }
      `}
      aria-pressed={isSelected}
      aria-label={`Select entity ${entity.name}`}
    >
      {/* Entity type indicator */}
      <div
        className="w-3 h-3 rounded-full flex-shrink-0"
        style={{ backgroundColor: color }}
        title={entity.entity_type}
      />

      {/* Entity info */}
      <div className="flex-1 min-w-0">
        <div className="font-medium text-gray-900 truncate">
          {entity.name}
        </div>
        <div
          className="text-xs px-1.5 py-0.5 rounded inline-block mt-0.5"
          style={{ backgroundColor: bgColor, color }}
        >
          {entity.entity_type}
        </div>
      </div>

      {/* Confidence badge */}
      <div
        className={`
          text-xs font-medium px-2 py-1 rounded
          ${confidencePercent >= 90
            ? 'bg-green-100 text-green-700'
            : confidencePercent >= 70
              ? 'bg-yellow-100 text-yellow-700'
              : 'bg-gray-100 text-gray-600'
          }
        `}
        title={`Confidence: ${confidencePercent}%`}
      >
        {confidencePercent}%
      </div>
    </button>
  );
}

// ============================================================================
// Main Component
// ============================================================================

/**
 * EntityBrowser component for browsing and searching entities in a collection.
 *
 * Features:
 * - Stats header showing entity and relationship counts
 * - Search bar with debounced filtering
 * - Type filter chips for filtering by entity type
 * - Scrollable entity list with selection support
 * - Loading, error, and empty states
 * - Graph disabled state
 */
export function EntityBrowser({
  collectionId,
  onEntitySelect,
  selectedEntityId,
  className = '',
}: EntityBrowserProps) {
  // Local state
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTypes, setSelectedTypes] = useState<string[]>([]);

  // Debounced search query (300ms)
  const debouncedQuery = useDebounce(searchQuery, 300);

  // Data fetching
  const {
    data: stats,
    isLoading: statsLoading,
    error: statsError,
  } = useGraphStats(collectionId);

  const {
    data: entityTypes,
    isLoading: typesLoading,
  } = useEntityTypes(collectionId);

  const {
    data: searchResults,
    isLoading: searchLoading,
    error: searchError,
  } = useEntitySearch(
    collectionId,
    {
      query: debouncedQuery || undefined,
      entity_types: selectedTypes.length > 0 ? selectedTypes : undefined,
      limit: 50,
    },
    stats?.graph_enabled ?? false
  );

  // Handlers
  const handleTypeToggle = useCallback((type: string) => {
    setSelectedTypes((prev) =>
      prev.includes(type)
        ? prev.filter((t) => t !== type)
        : [...prev, type]
    );
  }, []);

  const handleEntityClick = useCallback(
    (entity: EntityResponse) => {
      onEntitySelect?.(entity);
    },
    [onEntitySelect]
  );

  const handleClearFilters = useCallback(() => {
    setSearchQuery('');
    setSelectedTypes([]);
  }, []);

  // Sorted entity types by count (descending)
  const sortedEntityTypes = useMemo(() => {
    if (!entityTypes) return [];
    return Object.entries(entityTypes)
      .sort(([, a], [, b]) => b - a)
      .map(([type, count]) => ({ type, count }));
  }, [entityTypes]);

  // Check if graph is disabled
  if (stats && !stats.graph_enabled) {
    return (
      <div className={`flex flex-col items-center justify-center py-12 ${className}`}>
        <Database className="w-12 h-12 text-gray-300 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">
          Graph Not Enabled
        </h3>
        <p className="text-sm text-gray-500 text-center max-w-sm">
          This collection does not have graph extraction enabled.
          Re-index with graph extraction to explore entities and relationships.
        </p>
      </div>
    );
  }

  // Error state
  if (statsError || searchError) {
    const error = statsError || searchError;
    return (
      <div className={`flex flex-col items-center justify-center py-12 ${className}`}>
        <AlertCircle className="w-12 h-12 text-red-400 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">
          Error Loading Entities
        </h3>
        <p className="text-sm text-red-600 text-center max-w-sm">
          {error instanceof Error ? error.message : 'An unexpected error occurred'}
        </p>
      </div>
    );
  }

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Stats Header */}
      <div className="flex-shrink-0 pb-4 border-b border-gray-200">
        {statsLoading ? (
          <StatsSkeleton />
        ) : stats ? (
          <div className="flex gap-6">
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <Users className="w-4 h-4 text-blue-500" />
              <span className="font-medium">{stats.total_entities.toLocaleString()}</span>
              <span>entities</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <Link2 className="w-4 h-4 text-green-500" />
              <span className="font-medium">{stats.total_relationships.toLocaleString()}</span>
              <span>relationships</span>
            </div>
          </div>
        ) : null}
      </div>

      {/* Search Bar */}
      <div className="flex-shrink-0 py-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search entities..."
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg
                       focus:ring-2 focus:ring-blue-500 focus:border-blue-500
                       text-sm placeholder-gray-400"
            aria-label="Search entities"
          />
        </div>
      </div>

      {/* Type Filters */}
      {!typesLoading && sortedEntityTypes.length > 0 && (
        <div className="flex-shrink-0 pb-4">
          <div className="flex flex-wrap gap-2">
            {sortedEntityTypes.map(({ type, count }) => (
              <TypeChip
                key={type}
                type={type}
                count={count}
                isSelected={selectedTypes.includes(type)}
                onClick={() => handleTypeToggle(type)}
              />
            ))}
          </div>
          {(selectedTypes.length > 0 || searchQuery) && (
            <button
              onClick={handleClearFilters}
              className="mt-2 text-sm text-blue-600 hover:text-blue-800"
            >
              Clear filters
            </button>
          )}
        </div>
      )}

      {/* Entity List */}
      <div className="flex-1 overflow-y-auto min-h-0">
        {searchLoading ? (
          <EntityListSkeleton />
        ) : searchResults && searchResults.entities.length > 0 ? (
          <div className="space-y-1">
            {searchResults.entities.map((entity) => (
              <EntityItem
                key={entity.id}
                entity={entity}
                isSelected={selectedEntityId === entity.id}
                onClick={() => handleEntityClick(entity)}
              />
            ))}
            {searchResults.has_more && (
              <div className="py-3 text-center text-sm text-gray-500">
                Showing {searchResults.entities.length} of {searchResults.total} entities.
                Refine your search to see more.
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <Search className="w-10 h-10 text-gray-300 mb-3" />
            <p className="text-gray-600 font-medium">No entities found</p>
            <p className="text-sm text-gray-400 mt-1">
              {searchQuery || selectedTypes.length > 0
                ? 'Try adjusting your search or filters'
                : 'This collection has no extracted entities'}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default EntityBrowser;
