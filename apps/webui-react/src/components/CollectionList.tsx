import { useQuery } from '@tanstack/react-query';
import { collectionsApi } from '../services/api';
import CollectionCard from './CollectionCard';

interface Collection {
  name: string;
  total_files: number;
  total_vectors: number;
  model_name: string;
  created_at: string;
  updated_at: string;
  job_count: number;
}

function CollectionList() {

  const { data: collections, isLoading, error, refetch } = useQuery({
    queryKey: ['collections'],
    queryFn: async () => {
      const response = await collectionsApi.list();
      return response.data as Collection[];
    },
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600 mb-4">Failed to load collections</p>
        <button
          onClick={() => refetch()}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Retry
        </button>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!collections || collections.length === 0) {
    return (
      <div className="text-center py-12">
        <svg
          className="mx-auto h-12 w-12 text-gray-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
          />
        </svg>
        <h3 className="mt-2 text-sm font-medium text-gray-900">No collections</h3>
        <p className="mt-1 text-sm text-gray-500">Get started by creating a new job.</p>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-lg font-medium text-gray-900">Document Collections</h2>
        <p className="mt-1 text-sm text-gray-500">
          Manage your indexed document collections. Each collection contains documents processed with the same embedding model and settings.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {collections.map((collection) => (
          <CollectionCard key={collection.name} collection={collection} />
        ))}
      </div>
    </div>
  );
}

export default CollectionList;