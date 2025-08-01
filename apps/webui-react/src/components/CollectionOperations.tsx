import { useCollectionOperations } from '../hooks/useCollectionOperations';
import OperationProgress from './OperationProgress';
import type { Collection } from '../types/collection';

interface CollectionOperationsProps {
  collection: Collection;
  className?: string;
  maxOperations?: number;
}

function CollectionOperations({
  collection,
  className = '',
  maxOperations = 5,
}: CollectionOperationsProps) {
  // Use React Query hook to fetch operations
  const { data: operations = [], refetch } = useCollectionOperations(collection.id);
  
  // Sort operations by creation date (newest first)
  const sortedOperations = [...(operations || [])].sort((a, b) => 
    new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  );
  
  // Get active operations (pending or processing)
  const activeOperations = sortedOperations.filter(op => 
    op.status === 'pending' || op.status === 'processing'
  );
  
  // Get recent completed operations
  const recentOperations = sortedOperations
    .filter(op => op.status === 'completed' || op.status === 'failed' || op.status === 'cancelled')
    .slice(0, maxOperations - activeOperations.length);
  
  const displayOperations = [...activeOperations, ...recentOperations].slice(0, maxOperations);
  
  if (displayOperations.length === 0) {
    return (
      <div className={`text-center py-8 text-gray-500 ${className}`}>
        <p className="text-sm">No operations yet</p>
        <p className="text-xs mt-1">Operations will appear here when you add data or re-index</p>
      </div>
    );
  }
  
  return (
    <div className={`space-y-4 ${className}`}>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-900">Operations</h3>
        {activeOperations.length > 0 && (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
            {activeOperations.length} Active
          </span>
        )}
      </div>
      
      {/* Active Operations (always show at top) */}
      {activeOperations.length > 0 && (
        <div className="space-y-4">
          <h4 className="text-xs font-medium text-gray-700 uppercase tracking-wider">
            Active Operations
          </h4>
          {activeOperations.map(operation => (
            <div 
              key={operation.id}
              className="p-4 bg-blue-50 border border-blue-200 rounded-lg"
            >
              <OperationProgress
                operation={operation}
                showDetails={true}
                onComplete={() => {
                  // Refresh operations when one completes
                  refetch();
                }}
              />
            </div>
          ))}
        </div>
      )}
      
      {/* Recent Operations */}
      {recentOperations.length > 0 && (
        <div className="space-y-4">
          {activeOperations.length > 0 && (
            <h4 className="text-xs font-medium text-gray-700 uppercase tracking-wider mt-4">
              Recent Operations
            </h4>
          )}
          {recentOperations.map(operation => (
            <div 
              key={operation.id}
              className={`p-4 border rounded-lg ${
                operation.status === 'completed' 
                  ? 'bg-gray-50 border-gray-200'
                  : operation.status === 'failed'
                  ? 'bg-red-50 border-red-200'
                  : 'bg-yellow-50 border-yellow-200'
              }`}
            >
              <OperationProgress
                operation={operation}
                showDetails={false}
              />
            </div>
          ))}
        </div>
      )}
      
      {/* Show more indicator if there are more operations */}
      {operations.length > displayOperations.length && (
        <div className="text-center pt-2">
          <p className="text-xs text-gray-500">
            {operations.length - displayOperations.length} more operations not shown
          </p>
        </div>
      )}
    </div>
  );
}

export default CollectionOperations;