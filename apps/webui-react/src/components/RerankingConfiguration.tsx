import { useSearchStore } from '../stores/searchStore';
import { useCallback } from 'react';
import { systemApi } from '../services/api/v2/system';

/**
 * Props for the RerankingConfiguration component
 */
interface RerankingConfigurationProps {
  /** Whether cross-encoder reranking is enabled */
  enabled: boolean;
  /** Selected reranker model (undefined means auto-select) */
  model?: string;
  /** Selected quantization method (undefined means auto) */
  quantization?: string;
  /** Callback to update reranking configuration */
  onChange: (updates: {
    useReranker?: boolean;
    rerankModel?: string;
    rerankQuantization?: string;
  }) => void;
}

/**
 * RerankingConfiguration Component
 * 
 * A reusable component for configuring cross-encoder reranking options in search.
 * Provides controls for enabling/disabling reranking, selecting models, and choosing
 * quantization methods to balance between accuracy and performance.
 * 
 * @example
 * ```tsx
 * <RerankingConfiguration
 *   enabled={searchParams.useReranker}
 *   model={searchParams.rerankModel}
 *   quantization={searchParams.rerankQuantization}
 *   onChange={(updates) => updateSearchParams(updates)}
 * />
 * ```
 */
export function RerankingConfiguration({
  enabled,
  model,
  quantization,
  onChange
}: RerankingConfigurationProps) {
  const { 
    rerankingAvailable, 
    rerankingModelsLoading,
    setRerankingAvailable,
    setRerankingModelsLoading
  } = useSearchStore();

  const handleEnabledChange = (checked: boolean) => {
    onChange({ useReranker: checked });
  };

  const handleModelChange = (value: string) => {
    onChange({ rerankModel: value === 'auto' ? undefined : value });
  };

  const handleQuantizationChange = (value: string) => {
    onChange({ rerankQuantization: value === 'auto' ? undefined : value });
  };
  
  const handleRefreshAvailability = useCallback(async () => {
    setRerankingModelsLoading(true);
    try {
      const status = await systemApi.getStatus();
      setRerankingAvailable(status.reranking_available);
    } catch (error) {
      console.error('Failed to refresh reranking availability:', error);
    } finally {
      setRerankingModelsLoading(false);
    }
  }, [setRerankingAvailable, setRerankingModelsLoading]);

  return (
    <div className="mb-4">
      <div className="bg-gray-50 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={enabled}
              onChange={(e) => handleEnabledChange(e.target.checked)}
              className="w-4 h-4 text-blue-600 rounded"
              aria-label="Enable cross-encoder reranking"
              disabled={rerankingModelsLoading || !rerankingAvailable}
            />
            <span className="text-sm font-medium text-gray-700">
              Enable Cross-Encoder Reranking
              {rerankingModelsLoading && (
                <span className="ml-2 text-xs text-gray-500">(Loading...)</span>
              )}
            </span>
          </label>
          {rerankingModelsLoading && (
            <svg className="animate-spin h-4 w-4 text-gray-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          )}
        </div>
        
        {!rerankingAvailable && !rerankingModelsLoading && (
          <div className="mt-3 ml-6">
            <div className="flex items-center justify-between">
              <p className="text-xs text-red-600">
                Reranking is not available. GPU acceleration may be required.
              </p>
              <button
                onClick={handleRefreshAvailability}
                className="text-xs text-blue-600 hover:text-blue-800 underline"
                type="button"
              >
                Check again
              </button>
            </div>
          </div>
        )}
        
        {enabled && rerankingAvailable && (
          <div className="mt-3 ml-6" role="region" aria-label="Reranking configuration options">
            <p className="text-xs text-gray-600 mb-3">
              Reranking uses a more sophisticated model to re-score the top search results, 
              improving accuracy at the cost of slightly increased latency.
            </p>
            
            <div className="space-y-3">
              {/* Reranker Model Selection */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label 
                    htmlFor="reranker-model" 
                    className="block text-xs text-gray-700 mb-1"
                  >
                    Reranker Model <span className="text-gray-500" title="VRAM requirements shown in parentheses">(VRAM)</span>
                  </label>
                  <select
                    id="reranker-model"
                    value={model || 'auto'}
                    onChange={(e) => handleModelChange(e.target.value)}
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                    aria-describedby="model-help"
                    disabled={rerankingModelsLoading}
                  >
                    <option value="auto">Auto-select</option>
                    <option value="Qwen/Qwen3-Reranker-0.6B">0.6B (Fastest, ~1GB)</option>
                    <option value="Qwen/Qwen3-Reranker-4B">4B (Balanced, ~4GB)</option>
                    <option value="Qwen/Qwen3-Reranker-8B">8B (Most Accurate, ~8GB)</option>
                  </select>
                  <span id="model-help" className="sr-only">
                    Choose a reranker model based on your performance and accuracy needs
                  </span>
                </div>
                
                <div>
                  <label 
                    htmlFor="reranker-quantization" 
                    className="block text-xs text-gray-700 mb-1"
                  >
                    Quantization
                  </label>
                  <select
                    id="reranker-quantization"
                    value={quantization || 'auto'}
                    onChange={(e) => handleQuantizationChange(e.target.value)}
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
                    aria-describedby="quantization-help"
                    disabled={rerankingModelsLoading}
                  >
                    <option value="auto">Auto (match embedding)</option>
                    <option value="float32">Float32 (Full precision)</option>
                    <option value="float16">Float16 (Balanced)</option>
                    <option value="int8">Int8 (Low memory)</option>
                  </select>
                  <span id="quantization-help" className="sr-only">
                    Select quantization method to balance memory usage and precision
                  </span>
                </div>
              </div>
              
              {/* Memory usage note */}
              <div className="mt-3 p-2 bg-blue-50 rounded text-xs text-blue-800">
                <p>
                  <strong>Memory Usage:</strong> Reranking models require GPU VRAM. 
                  If you encounter memory errors, try a smaller model or use Int8 quantization.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}