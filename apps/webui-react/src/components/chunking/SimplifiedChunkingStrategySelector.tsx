import { useState } from 'react';
import { Info, ChevronDown, Settings2, Sparkles } from 'lucide-react';
import { useChunkingStore } from '../../stores/chunkingStore';
import { CHUNKING_STRATEGIES } from '../../types/chunking';
import { ChunkingParameterTuner } from './ChunkingParameterTuner';
import { ChunkingStrategyGuide } from './ChunkingStrategyGuide';
import type { ChunkingStrategyType } from '../../types/chunking';
import './SimplifiedChunkingStrategySelector.css';

interface SimplifiedChunkingStrategySelectorProps {
  onStrategyChange?: (strategy: ChunkingStrategyType) => void;
  disabled?: boolean;
  fileType?: string;
}

export function SimplifiedChunkingStrategySelector({
  onStrategyChange,
  disabled = false,
  fileType
}: SimplifiedChunkingStrategySelectorProps) {
  const {
    selectedStrategy,
    setStrategy,
    getRecommendedStrategy
  } = useChunkingStore();

  const [showGuide, setShowGuide] = useState(false);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);

  const recommendedStrategy = fileType ? getRecommendedStrategy(fileType) : 'hybrid';

  const handleStrategyChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newStrategy = event.target.value as ChunkingStrategyType;
    if (!disabled) {
      setStrategy(newStrategy);
      onStrategyChange?.(newStrategy);
    }
  };

  const getStrategyLabel = (strategyType: ChunkingStrategyType) => {
    const strategy = CHUNKING_STRATEGIES[strategyType];
    const isRecommended = strategyType === recommendedStrategy || strategy.isRecommended;
    return `${strategy.name}${isRecommended ? ' ✨ Recommended' : ''}`;
  };

  const selectedStrategyInfo = CHUNKING_STRATEGIES[selectedStrategy];

  return (
    <div className="space-y-4">
      {/* Strategy Selector */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label htmlFor="chunking-strategy" className="block text-sm font-medium text-gray-700">
            Chunking Strategy
          </label>
          <button
            type="button"
            onClick={() => setShowGuide(true)}
            className="text-sm text-blue-600 hover:text-blue-700 flex items-center gap-1 transition-colors"
          >
            <Info className="h-3 w-3" />
            Learn more
          </button>
        </div>

        <div className="relative">
          <select
            id="chunking-strategy"
            value={selectedStrategy}
            onChange={handleStrategyChange}
            disabled={disabled}
            className="w-full pl-4 pr-10 py-2.5 text-sm border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-50 disabled:text-gray-500 appearance-none cursor-pointer"
          >
            {(Object.keys(CHUNKING_STRATEGIES) as ChunkingStrategyType[]).map((strategyType) => (
              <option key={strategyType} value={strategyType}>
                {getStrategyLabel(strategyType)}
              </option>
            ))}
          </select>
          <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400 pointer-events-none" />
        </div>

        {/* Strategy Description */}
        <p className="mt-2 text-xs text-gray-500">
          {selectedStrategyInfo.description}
        </p>

        {/* File Type Recommendation */}
        {fileType && selectedStrategy !== recommendedStrategy && (
          <div className="mt-2 p-2 bg-blue-50 border border-blue-200 rounded-md">
            <p className="text-xs text-blue-700 flex items-center gap-1">
              <Sparkles className="h-3 w-3" />
              For {fileType} files, we recommend using {CHUNKING_STRATEGIES[recommendedStrategy].name}
            </p>
          </div>
        )}
      </div>

      {/* Advanced Options Toggle */}
      <div className="border-t pt-4">
        <button
          type="button"
          onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
          disabled={disabled}
          className="flex items-center gap-2 text-sm font-medium text-gray-700 hover:text-gray-900 transition-colors disabled:opacity-50"
        >
          <Settings2 className="h-4 w-4" />
          Advanced Options
          <ChevronDown
            className={`h-4 w-4 transition-transform ${
              showAdvancedOptions ? 'rotate-180' : ''
            }`}
          />
        </button>

        {/* Collapsible Parameter Tuner */}
        {showAdvancedOptions && (
          <div className="mt-4 pl-6 space-y-4 animate-slideDown">
            <ChunkingParameterTuner
              disabled={disabled}
              showPreview={false}
              onParameterChange={() => onStrategyChange?.(selectedStrategy)}
            />
          </div>
        )}
      </div>

      {/* Strategy Guide Modal */}
      {showGuide && (
        <ChunkingStrategyGuide
          onClose={() => setShowGuide(false)}
          currentStrategy={selectedStrategy}
          fileType={fileType}
        />
      )}
    </div>
  );
}