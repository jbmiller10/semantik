import { useUIStore } from '../stores/uiStore';

interface Collection {
  name: string;
  total_files: number;
  total_vectors: number;
  model_name: string;
  created_at: string;
  updated_at: string;
  job_count: number;
}

interface CollectionCardProps {
  collection: Collection;
}

function CollectionCard({ collection }: CollectionCardProps) {
  const { setShowCollectionDetailsModal } = useUIStore();

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  const formatNumber = (num: number) => {
    return num.toLocaleString();
  };

  const getModelDisplayName = (modelName: string) => {
    // Extract just the model name from the full path
    const parts = modelName.split('/');
    return parts[parts.length - 1] || modelName;
  };

  return (
    <div className="bg-white overflow-hidden shadow rounded-lg hover:shadow-lg transition-shadow">
      <div className="p-5">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h3 className="text-lg font-medium text-gray-900 truncate" title={collection.name}>
              {collection.name}
            </h3>
            <p className="mt-1 text-sm text-gray-500">
              {getModelDisplayName(collection.model_name)}
            </p>
          </div>
          <div className="ml-2 flex-shrink-0">
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
              {collection.job_count} {collection.job_count === 1 ? 'job' : 'jobs'}
            </span>
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-4">
          <div>
            <dt className="text-xs font-medium text-gray-500">Documents</dt>
            <dd className="mt-1 text-sm font-semibold text-gray-900">
              {formatNumber(collection.total_files)}
            </dd>
          </div>
          <div>
            <dt className="text-xs font-medium text-gray-500">Vectors</dt>
            <dd className="mt-1 text-sm font-semibold text-gray-900">
              {formatNumber(collection.total_vectors)}
            </dd>
          </div>
        </div>

        <div className="mt-4">
          <dt className="text-xs font-medium text-gray-500">Last updated</dt>
          <dd className="mt-1 text-sm text-gray-900">{formatDate(collection.updated_at)}</dd>
        </div>
      </div>

      <div className="bg-gray-50 px-5 py-3">
        <button
          onClick={() => setShowCollectionDetailsModal(collection.name)}
          className="w-full flex justify-center items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
        >
          <svg
            className="mr-2 -ml-1 h-4 w-4 text-gray-500"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
            />
          </svg>
          Manage
        </button>
      </div>
    </div>
  );
}

export default CollectionCard;