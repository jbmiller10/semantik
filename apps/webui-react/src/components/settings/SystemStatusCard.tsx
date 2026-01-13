/**
 * System Status Card component (System tab).
 * Displays read-only system information, limits, health status, and GPU info.
 */
import { useSystemInfo, useSystemHealth, useSystemStatus } from '../../hooks/useSystemInfo';

/**
 * Status indicator component for service health.
 */
function StatusIndicator({ status }: { status: 'healthy' | 'unhealthy' | 'loading' }) {
  const colors = {
    healthy: 'bg-green-400',
    unhealthy: 'bg-red-400',
    loading: 'bg-gray-300 animate-pulse',
  };

  return (
    <span className={`inline-block h-3 w-3 rounded-full ${colors[status]}`} />
  );
}

/**
 * Info row component for displaying key-value pairs.
 */
function InfoRow({ label, value }: { label: string; value: string | number | null | undefined }) {
  return (
    <div className="flex justify-between py-2 border-b border-gray-100 last:border-b-0">
      <span className="text-sm text-gray-500">{label}</span>
      <span className="text-sm font-medium text-gray-900">{value ?? 'N/A'}</span>
    </div>
  );
}

export default function SystemStatusCard() {
  const { data: systemInfo, isLoading: infoLoading, error: infoError } = useSystemInfo();
  const { data: systemHealth, isLoading: healthLoading } = useSystemHealth();
  const { data: systemStatus, isLoading: statusLoading } = useSystemStatus();

  // Loading state
  if (infoLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <svg className="animate-spin h-8 w-8 text-gray-400" fill="none" viewBox="0 0 24 24">
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
        <span className="ml-3 text-gray-500">Loading system information...</span>
      </div>
    );
  }

  // Error state
  if (infoError) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex">
          <svg className="h-5 w-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Error loading system info</h3>
            <p className="mt-1 text-sm text-red-700">{infoError.message}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h3 className="text-lg leading-6 font-medium text-gray-900">System Information</h3>
        <p className="mt-1 text-sm text-gray-500">
          View system configuration, resource limits, and service health status.
        </p>
      </div>

      {/* Info box */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex">
          <svg
            className="h-5 w-5 text-blue-400 flex-shrink-0"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <div className="ml-3">
            <p className="text-sm text-blue-700">
              These settings are read-only and configured by your system administrator.
              Service health auto-refreshes every 30 seconds.
            </p>
          </div>
        </div>
      </div>

      {/* System Information */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">Application</h4>
          <div className="space-y-1">
            <InfoRow label="Version" value={systemInfo?.version} />
            <InfoRow label="Environment" value={systemInfo?.environment} />
            <InfoRow label="Python Version" value={systemInfo?.python_version} />
          </div>
        </div>
      </div>

      {/* Resource Limits */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">Resource Limits</h4>
          <div className="space-y-1">
            <InfoRow
              label="Max Collections per User"
              value={systemInfo?.limits?.max_collections_per_user}
            />
            <InfoRow
              label="Max Storage per User"
              value={systemInfo?.limits?.max_storage_gb_per_user ? `${systemInfo.limits.max_storage_gb_per_user} GB` : undefined}
            />
          </div>
        </div>
      </div>

      {/* Rate Limits */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">Rate Limits</h4>
          <div className="space-y-1">
            <InfoRow label="Chunking Preview" value={systemInfo?.rate_limits?.chunking_preview} />
            <InfoRow label="Plugin Install" value={systemInfo?.rate_limits?.plugin_install} />
            <InfoRow label="LLM Test" value={systemInfo?.rate_limits?.llm_test} />
          </div>
        </div>
      </div>

      {/* GPU Status */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h4 className="text-md font-medium text-gray-900 mb-4">GPU Status</h4>

          {statusLoading ? (
            <div className="flex items-center">
              <StatusIndicator status="loading" />
              <span className="ml-2 text-sm text-gray-500">Checking GPU status...</span>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center">
                <StatusIndicator status={systemStatus?.gpu_available ? 'healthy' : 'unhealthy'} />
                <span className="ml-2 text-sm font-medium text-gray-900">
                  {systemStatus?.gpu_available ? 'GPU Available' : 'GPU Not Available'}
                </span>
              </div>

              {systemStatus?.gpu_available && (
                <>
                  <div className="space-y-1 ml-5">
                    <InfoRow label="Device" value={systemStatus?.cuda_device_name} />
                    <InfoRow label="CUDA Devices" value={systemStatus?.cuda_device_count} />
                  </div>
                </>
              )}

              <div className="flex items-center mt-2">
                <StatusIndicator status={systemStatus?.reranking_available ? 'healthy' : 'unhealthy'} />
                <span className="ml-2 text-sm font-medium text-gray-900">
                  {systemStatus?.reranking_available ? 'Reranking Available' : 'Reranking Not Available'}
                </span>
              </div>

              {systemStatus?.reranking_available && systemStatus.available_reranking_models?.length > 0 && (
                <div className="ml-5">
                  <p className="text-xs text-gray-500 mb-1">Available Models:</p>
                  <ul className="list-disc list-inside text-sm text-gray-700">
                    {systemStatus.available_reranking_models.map((model) => (
                      <li key={model} className="truncate">{model}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Service Health */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-md font-medium text-gray-900">Service Health</h4>
            {!healthLoading && (
              <span className="text-xs text-gray-400">Auto-refreshes every 30s</span>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4">
            {/* PostgreSQL */}
            <div className="flex items-center p-3 bg-gray-50 rounded-lg">
              <StatusIndicator
                status={healthLoading ? 'loading' : (systemHealth?.postgres?.status === 'healthy' ? 'healthy' : 'unhealthy')}
              />
              <div className="ml-3">
                <span className="text-sm font-medium text-gray-900">PostgreSQL</span>
                {!healthLoading && systemHealth?.postgres?.message && (
                  <p className="text-xs text-gray-500 truncate">{systemHealth.postgres.message}</p>
                )}
              </div>
            </div>

            {/* Redis */}
            <div className="flex items-center p-3 bg-gray-50 rounded-lg">
              <StatusIndicator
                status={healthLoading ? 'loading' : (systemHealth?.redis?.status === 'healthy' ? 'healthy' : 'unhealthy')}
              />
              <div className="ml-3">
                <span className="text-sm font-medium text-gray-900">Redis</span>
                {!healthLoading && systemHealth?.redis?.message && (
                  <p className="text-xs text-gray-500 truncate">{systemHealth.redis.message}</p>
                )}
              </div>
            </div>

            {/* Qdrant */}
            <div className="flex items-center p-3 bg-gray-50 rounded-lg">
              <StatusIndicator
                status={healthLoading ? 'loading' : (systemHealth?.qdrant?.status === 'healthy' ? 'healthy' : 'unhealthy')}
              />
              <div className="ml-3">
                <span className="text-sm font-medium text-gray-900">Qdrant</span>
                {!healthLoading && systemHealth?.qdrant?.message && (
                  <p className="text-xs text-gray-500 truncate">{systemHealth.qdrant.message}</p>
                )}
              </div>
            </div>

            {/* VecPipe */}
            <div className="flex items-center p-3 bg-gray-50 rounded-lg">
              <StatusIndicator
                status={healthLoading ? 'loading' : (systemHealth?.vecpipe?.status === 'healthy' ? 'healthy' : 'unhealthy')}
              />
              <div className="ml-3">
                <span className="text-sm font-medium text-gray-900">VecPipe</span>
                {!healthLoading && systemHealth?.vecpipe?.message && (
                  <p className="text-xs text-gray-500 truncate">{systemHealth.vecpipe.message}</p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
