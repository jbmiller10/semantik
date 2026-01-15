import { useState, useRef, useEffect } from 'react';
import { Info, ChevronDown, Settings2, Sparkles, Check } from 'lucide-react';
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
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const recommendedStrategy = fileType ? getRecommendedStrategy(fileType) : 'hybrid';

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleStrategySelect = (strategyType: ChunkingStrategyType) => {
    if (!disabled) {
      setStrategy(strategyType);
      onStrategyChange?.(strategyType);
      setIsDropdownOpen(false);
    }
  };

  const getStrategyLabel = (strategyType: ChunkingStrategyType) => {
    const strategy = CHUNKING_STRATEGIES[strategyType];
    const isRecommended = strategyType === recommendedStrategy || strategy.isRecommended;
    return { name: strategy.name, isRecommended };
  };

  const selectedStrategyInfo = CHUNKING_STRATEGIES[selectedStrategy];
  const selectedLabel = getStrategyLabel(selectedStrategy);

  return (
    <div className="space-y-4">
      {/* Strategy Selector */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="block text-xs font-bold text-gray-400 uppercase tracking-wider">
            Chunking Strategy
          </label>
          <button
            type="button"
            onClick={() => setShowGuide(true)}
            className="text-sm text-signal-500 hover:text-signal-400 flex items-center gap-1 transition-colors"
          >
            <Info className="h-3 w-3" />
            Learn more
          </button>
        </div>

        {/* Custom Dropdown */}
        <div className="relative" ref={dropdownRef}>
          <button
            type="button"
            onClick={() => !disabled && setIsDropdownOpen(!isDropdownOpen)}
            disabled={disabled}
            className={`
              w-full pl-4 pr-10 py-2.5 text-sm text-left border rounded-xl flex items-center justify-between
              transition-all
              ${disabled
                ? 'bg-void-900 text-gray-500 cursor-not-allowed border-white/10'
                : 'bg-void-800/50 text-white hover:bg-void-700/50 cursor-pointer border-white/10'
              }
              ${isDropdownOpen ? 'ring-2 ring-signal-500/50 border-signal-500' : ''}
            `}
            aria-expanded={isDropdownOpen}
            aria-haspopup="listbox"
          >
            <span className="flex items-center gap-2">
              {selectedLabel.name}
              {selectedLabel.isRecommended && (
                <span className="text-amber-400">✨ Recommended</span>
              )}
            </span>
            <ChevronDown className={`h-4 w-4 text-gray-400 transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} />
          </button>

          {/* Dropdown Options */}
          {isDropdownOpen && (
            <div
              className="absolute z-50 w-full mt-1 bg-void-900 border border-white/10 rounded-xl shadow-xl overflow-hidden"
              role="listbox"
            >
              {(Object.keys(CHUNKING_STRATEGIES) as ChunkingStrategyType[]).map((strategyType) => {
                const label = getStrategyLabel(strategyType);
                const isSelected = strategyType === selectedStrategy;
                return (
                  <button
                    key={strategyType}
                    type="button"
                    onClick={() => handleStrategySelect(strategyType)}
                    className={`
                      w-full px-4 py-2.5 text-sm text-left flex items-center justify-between
                      transition-colors
                      ${isSelected
                        ? 'bg-signal-500/20 text-white'
                        : 'text-gray-300 hover:bg-void-800 hover:text-white'
                      }
                    `}
                    role="option"
                    aria-selected={isSelected}
                  >
                    <span className="flex items-center gap-2">
                      {label.name}
                      {label.isRecommended && (
                        <span className="text-amber-400 text-xs">✨ Recommended</span>
                      )}
                    </span>
                    {isSelected && <Check className="h-4 w-4 text-signal-400" />}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Strategy Description */}
        <p className="mt-2 text-xs text-gray-500">
          {selectedStrategyInfo.description}
        </p>

        {/* File Type Recommendation */}
        {fileType && selectedStrategy !== recommendedStrategy && (
          <div className="mt-2 p-2 bg-signal-500/10 border border-signal-500/20 rounded-lg">
            <p className="text-xs text-signal-400 flex items-center gap-1">
              <Sparkles className="h-3 w-3" />
              For {fileType} files, we recommend using {CHUNKING_STRATEGIES[recommendedStrategy].name}
            </p>
          </div>
        )}
      </div>

      {/* Advanced Options Toggle */}
      <div className="border-t border-white/10 pt-4">
        <button
          type="button"
          onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
          disabled={disabled}
          className="flex items-center gap-2 text-sm font-medium text-gray-400 hover:text-white transition-colors disabled:opacity-50"
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
