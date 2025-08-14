import { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Sliders, 
  RotateCcw, 
  Save,
  ChevronDown,
  Eye,
  HelpCircle
} from 'lucide-react';
import { useChunkingStore } from '../../stores/chunkingStore';
import { CHUNKING_STRATEGIES, CHUNKING_PRESETS } from '../../types/chunking';
import type { ChunkingParameter } from '../../types/chunking';
import { getInputClassName } from '../../utils/formStyles';
import './ChunkingParameterTuner.css';

interface ChunkingParameterTunerProps {
  showPreview?: boolean;
  onParameterChange?: () => void;
  disabled?: boolean;
}

export function ChunkingParameterTuner({ 
  showPreview = true,
  onParameterChange,
  disabled = false 
}: ChunkingParameterTunerProps) {
  const {
    selectedStrategy,
    strategyConfig,
    updateConfiguration,
    applyPreset,
    saveCustomPreset,
    customPresets,
    selectedPreset,
    loadPreview,
    previewDocument,
    previewLoading
  } = useChunkingStore();

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [customPresetName, setCustomPresetName] = useState('');
  const [showSavePreset, setShowSavePreset] = useState(false);

  const strategy = CHUNKING_STRATEGIES[selectedStrategy];
  const basicParameters = strategy.parameters.filter(p => !p.advanced);
  const advancedParameters = strategy.parameters.filter(p => p.advanced);
  
  // Use ref to track the debounce timer
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Debounced parameter change handler
  const debouncedLoadPreview = useCallback(() => {
    // Clear any existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    // Set new timeout
    timeoutRef.current = setTimeout(() => {
      if (showPreview && previewDocument) {
        loadPreview(true);
      }
      onParameterChange?.();
      timeoutRef.current = null;
    }, 500);
  }, [showPreview, previewDocument, loadPreview, onParameterChange]);

  // Update preview when parameters change
  useEffect(() => {
    debouncedLoadPreview();
  }, [strategyConfig.parameters, debouncedLoadPreview]);
  
  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const handleParameterChange = (key: string, value: number | boolean | string) => {
    updateConfiguration({ [key]: value });
  };

  const handlePresetChange = (presetId: string) => {
    if (presetId === 'custom') {
      return;
    }
    applyPreset(presetId);
  };

  const handleSaveCustomPreset = () => {
    if (!customPresetName.trim()) return;

    saveCustomPreset({
      name: customPresetName,
      description: `Custom configuration for ${strategy.name}`,
      strategy: selectedStrategy,
      configuration: strategyConfig
    });

    setCustomPresetName('');
    setShowSavePreset(false);
  };

  const handleResetToDefaults = () => {
    const defaultConfig = CHUNKING_STRATEGIES[selectedStrategy].parameters.reduce(
      (acc, param) => ({ ...acc, [param.key]: param.defaultValue }),
      {}
    );
    updateConfiguration(defaultConfig);
  };

  const renderParameter = (param: ChunkingParameter) => {
    const value = strategyConfig.parameters[param.key];

    switch (param.type) {
      case 'number':
        return (
          <div key={param.key} className="bg-gray-50 rounded-lg p-4 space-y-3">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <label className="text-sm font-medium text-gray-900 flex items-center gap-2">
                  {param.name}
                  {param.description && (
                    <div className="group relative">
                      <HelpCircle className="h-3.5 w-3.5 text-gray-400 hover:text-gray-600 cursor-help" />
                      <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
                        <div className="bg-gray-900 text-white text-xs rounded-lg py-2 px-3 max-w-xs whitespace-normal shadow-lg">
                          {param.description}
                          <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1">
                            <div className="w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </label>
                {param.unit && (
                  <p className="text-xs text-gray-500 mt-1">Measured in {param.unit}</p>
                )}
              </div>
              <span className="text-sm font-mono font-medium text-gray-900 bg-white px-3 py-1 rounded-md border border-gray-200">
                {value}
              </span>
            </div>
            <div className="relative">
              <input
                type="range"
                min={param.min}
                max={param.max}
                step={param.step}
                value={value as number}
                onChange={(e) => handleParameterChange(param.key, Number(e.target.value))}
                disabled={disabled}
                className={`w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider
                  ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-300'}`}
                style={{
                  background: `linear-gradient(to right, #3B82F6 0%, #3B82F6 ${
                    ((Number(value) - param.min!) / (param.max! - param.min!)) * 100
                  }%, #E5E7EB ${
                    ((Number(value) - param.min!) / (param.max! - param.min!)) * 100
                  }%, #E5E7EB 100%)`
                }}
              />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>{param.min}</span>
                <span>{param.max}</span>
              </div>
            </div>
          </div>
        );

      case 'boolean':
        return (
          <div key={param.key} className="bg-gray-50 rounded-lg p-4 flex items-center justify-between">
            <div className="flex-1">
              <label className="text-sm font-medium text-gray-900 flex items-center gap-2">
                {param.name}
                {param.description && (
                  <div className="group relative">
                    <HelpCircle className="h-3.5 w-3.5 text-gray-400 hover:text-gray-600 cursor-help" />
                    <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
                      <div className="bg-gray-900 text-white text-xs rounded-lg py-2 px-3 max-w-xs whitespace-normal shadow-lg">
                        {param.description}
                        <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1">
                          <div className="w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </label>
              {param.description && (
                <p className="text-xs text-gray-500 mt-1 sm:hidden">{param.description}</p>
              )}
            </div>
            <button
              type="button"
              onClick={() => handleParameterChange(param.key, !value)}
              disabled={disabled}
              className={`relative inline-flex h-6 w-11 flex-shrink-0 rounded-full border-2 border-transparent 
                transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
                ${value ? 'bg-blue-600' : 'bg-gray-300'}
                ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
              role="switch"
              aria-checked={value as boolean}
            >
              <span
                className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 
                  transition duration-200 ease-in-out
                  ${value ? 'translate-x-5' : 'translate-x-0'}`}
              />
            </button>
          </div>
        );

      case 'select':
        return (
          <div key={param.key} className="bg-gray-50 rounded-lg p-4 space-y-2">
            <label className="text-sm font-medium text-gray-900 flex items-center gap-2">
              {param.name}
              {param.description && (
                <div className="group relative">
                  <HelpCircle className="h-3.5 w-3.5 text-gray-400 hover:text-gray-600 cursor-help" />
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
                    <div className="bg-gray-900 text-white text-xs rounded-lg py-2 px-3 max-w-xs whitespace-normal shadow-lg">
                      {param.description}
                      <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-1">
                        <div className="w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-gray-900"></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </label>
            <select
              value={value as string}
              onChange={(e) => handleParameterChange(param.key, e.target.value)}
              disabled={disabled}
              className="block w-full pl-3 pr-10 py-2 text-sm bg-white border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 rounded-md"
            >
              {param.options?.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="space-y-4">
      {/* Preset Configuration */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-medium text-gray-900 flex items-center gap-2">
            <Sliders className="h-4 w-4" />
            Configuration
          </h4>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleResetToDefaults}
              disabled={disabled}
              className="text-xs text-gray-500 hover:text-gray-700 flex items-center gap-1 transition-colors"
            >
              <RotateCcw className="h-3 w-3" />
              Reset
            </button>
            <button
              type="button"
              onClick={() => setShowSavePreset(!showSavePreset)}
              disabled={disabled}
              className="text-xs text-blue-600 hover:text-blue-700 font-medium flex items-center gap-1 transition-colors"
            >
              <Save className="h-3 w-3" />
              Save Preset
            </button>
          </div>
        </div>
        
        <select
          value={selectedPreset || 'custom'}
          onChange={(e) => handlePresetChange(e.target.value)}
          disabled={disabled}
          className="block w-full pl-3 pr-10 py-2 text-sm bg-gray-50 border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 rounded-md"
        >
          <option value="custom">Custom Configuration</option>
          {CHUNKING_PRESETS.filter(p => p.strategy === selectedStrategy).length > 0 && (
            <optgroup label="Built-in Presets">
              {CHUNKING_PRESETS.filter(p => p.strategy === selectedStrategy).map(preset => (
                <option key={preset.id} value={preset.id}>
                  {preset.name}
                </option>
              ))}
            </optgroup>
          )}
          {customPresets.filter(p => p.strategy === selectedStrategy).length > 0 && (
            <optgroup label="Custom Presets">
              {customPresets.filter(p => p.strategy === selectedStrategy).map(preset => (
                <option key={preset.id} value={preset.id}>
                  {preset.name}
                </option>
              ))}
            </optgroup>
          )}
        </select>
      </div>

      {/* Save Custom Preset Form */}
      {showSavePreset && (
        <div className="bg-blue-50 border border-blue-200 rounded-md p-3 space-y-2">
          <input
            type="text"
            value={customPresetName}
            onChange={(e) => setCustomPresetName(e.target.value)}
            placeholder="Enter preset name..."
            className={getInputClassName(false, disabled)}
            disabled={disabled}
          />
          <div className="flex justify-end space-x-2">
            <button
              type="button"
              onClick={() => {
                setShowSavePreset(false);
                setCustomPresetName('');
              }}
              className="px-3 py-1 text-sm text-gray-600 hover:text-gray-700"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleSaveCustomPreset}
              disabled={!customPresetName.trim() || disabled}
              className="px-3 py-1 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Save
            </button>
          </div>
        </div>
      )}

      {/* Parameters */}
      {basicParameters.length > 0 && (
        <div className="space-y-3">
          <h5 className="text-xs font-medium text-gray-500 uppercase tracking-wider">Parameters</h5>
          {basicParameters.map(param => renderParameter(param))}
        </div>
      )}

      {/* Advanced Parameters */}
      {advancedParameters.length > 0 && (
        <div className="pt-2">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm font-medium text-gray-700 hover:text-gray-900 transition-colors w-full justify-between p-2 -m-2 rounded-md hover:bg-gray-50"
          >
            <span>Advanced Parameters</span>
            <ChevronDown
              className={`h-4 w-4 text-gray-400 transition-transform ${
                showAdvanced ? 'rotate-180' : ''
              }`}
            />
          </button>
          
          {showAdvanced && (
            <div className="mt-3 space-y-3 animate-slideDown">
              {advancedParameters.map(param => renderParameter(param))}
            </div>
          )}
        </div>
      )}

      {/* Preview Status */}
      {showPreview && previewDocument && (
        <div className="border-t pt-4">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600 flex items-center">
              {previewLoading ? (
                <>
                  <svg className="animate-spin h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Updating preview...
                </>
              ) : (
                <>
                  <Eye className="h-4 w-4 mr-2" />
                  Preview updated
                </>
              )}
            </span>
            <button
              type="button"
              onClick={() => loadPreview(true)}
              disabled={disabled || previewLoading}
              className="text-blue-600 hover:text-blue-700 font-medium disabled:opacity-50"
            >
              Refresh
            </button>
          </div>
        </div>
      )}
    </div>
  );
}