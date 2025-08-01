import { AlertTriangle, Cpu, Database, Zap, ChevronDown } from 'lucide-react';
import { useState } from 'react';

interface GPUMemoryErrorProps {
  /** Suggestion from the error response */
  suggestion: string;
  /** Currently selected model */
  currentModel?: string;
  /** Callback when user selects a smaller model */
  onSelectSmallerModel: (model: string) => void;
}

interface ModelOption {
  value: string;
  label: string;
  memory: string;
  description: string;
}

const MODEL_OPTIONS: ModelOption[] = [
  {
    value: 'Qwen/Qwen3-Reranker-0.6B',
    label: '0.6B Model',
    memory: '~1GB VRAM',
    description: 'Fastest, minimal memory usage'
  },
  {
    value: 'Qwen/Qwen3-Reranker-4B',
    label: '4B Model',
    memory: '~4GB VRAM',
    description: 'Balanced performance'
  },
  {
    value: 'Qwen/Qwen3-Reranker-8B',
    label: '8B Model',
    memory: '~8GB VRAM',
    description: 'Most accurate'
  }
];

const QUANTIZATION_TIPS = [
  { value: 'int8', label: 'Int8', savings: 'Up to 75% less memory' },
  { value: 'float16', label: 'Float16', savings: 'Up to 50% less memory' },
];

/**
 * GPUMemoryError Component
 * 
 * Displays a user-friendly error message when GPU memory is insufficient for reranking.
 * Provides actionable options to resolve the issue by switching to smaller models or
 * adjusting quantization settings.
 */
export function GPUMemoryError({ 
  suggestion, 
  currentModel, 
  onSelectSmallerModel 
}: GPUMemoryErrorProps) {
  const [showDetails, setShowDetails] = useState(false);
  
  // Find current model info and smaller alternatives
  const currentModelIndex = MODEL_OPTIONS.findIndex(m => m.value === currentModel);
  const smallerModels = currentModelIndex > 0 
    ? MODEL_OPTIONS.slice(0, currentModelIndex) 
    : MODEL_OPTIONS.slice(0, 2); // Show first two models if current model not found

  return (
    <div className="bg-amber-50 border border-amber-200 rounded-lg p-6 space-y-4">
      {/* Header with icon and title */}
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0">
          <AlertTriangle className="h-6 w-6 text-amber-600" aria-hidden="true" />
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-amber-900">
            Insufficient GPU Memory
          </h3>
          <p className="mt-1 text-sm text-amber-800">
            The selected reranking model requires more GPU memory than is currently available.
          </p>
        </div>
      </div>

      {/* Suggestion from backend */}
      {suggestion && (
        <div className="bg-amber-100 rounded-md p-3">
          <p className="text-sm text-amber-900">{suggestion}</p>
        </div>
      )}

      {/* Quick actions */}
      <div className="space-y-3">
        <p className="text-sm font-medium text-amber-900">Quick fixes:</p>
        
        {/* Model selection buttons */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {smallerModels.map((model) => (
            <button
              key={model.value}
              onClick={() => onSelectSmallerModel(model.value)}
              className="flex items-center justify-between p-3 bg-white border border-amber-300 rounded-md hover:bg-amber-50 transition-colors focus:outline-none focus:ring-2 focus:ring-amber-500"
              aria-label={`Switch to ${model.label} which uses ${model.memory}`}
            >
              <div className="flex items-center space-x-2">
                <Cpu className="h-4 w-4 text-amber-600" aria-hidden="true" />
                <div className="text-left">
                  <p className="font-medium text-sm text-gray-900">{model.label}</p>
                  <p className="text-xs text-gray-600">{model.memory}</p>
                </div>
              </div>
              <Zap className="h-4 w-4 text-amber-500" aria-hidden="true" />
            </button>
          ))}
        </div>

        {/* Disable reranking option */}
        <button
          onClick={() => onSelectSmallerModel('disabled')}
          className="w-full flex items-center justify-center p-3 bg-gray-100 border border-gray-300 rounded-md hover:bg-gray-200 transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500"
          aria-label="Disable reranking to proceed without it"
        >
          <Database className="h-4 w-4 mr-2 text-gray-600" aria-hidden="true" />
          <span className="text-sm font-medium text-gray-700">
            Proceed without reranking
          </span>
        </button>
      </div>

      {/* Expandable details section */}
      <div className="border-t border-amber-200 pt-4">
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="flex items-center text-sm font-medium text-amber-800 hover:text-amber-900 focus:outline-none focus:underline"
          aria-expanded={showDetails}
          aria-controls="memory-tips"
        >
          <ChevronDown 
            className={`h-4 w-4 mr-1 transform transition-transform ${
              showDetails ? 'rotate-180' : ''
            }`} 
            aria-hidden="true"
          />
          Tips for reducing memory usage
        </button>
        
        {showDetails && (
          <div 
            id="memory-tips" 
            className="mt-3 space-y-3 text-sm text-amber-800"
            role="region"
            aria-label="Memory reduction tips"
          >
            <div>
              <p className="font-medium mb-1">Try these quantization options:</p>
              <ul className="list-disc list-inside space-y-1 ml-2">
                {QUANTIZATION_TIPS.map((tip) => (
                  <li key={tip.value}>
                    <span className="font-medium">{tip.label}</span>: {tip.savings}
                  </li>
                ))}
              </ul>
            </div>
            
            <div>
              <p className="font-medium mb-1">Other tips:</p>
              <ul className="list-disc list-inside space-y-1 ml-2">
                <li>Close other GPU-intensive applications</li>
                <li>Reduce the number of results to rerank (lower top-K value)</li>
                <li>Consider using CPU-based reranking if available</li>
              </ul>
            </div>

            <div className="bg-amber-100 rounded-md p-2 mt-3">
              <p className="text-xs">
                <strong>Note:</strong> Model performance generally improves with size, 
                but even smaller models can significantly enhance search quality.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}