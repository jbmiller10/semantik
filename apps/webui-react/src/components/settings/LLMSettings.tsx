/**
 * LLM Settings component for configuring LLM providers and models.
 * Allows users to set API keys, select models for quality tiers,
 * and view token usage statistics.
 */
import { useState, useEffect, useCallback } from 'react';
import {
  useLLMSettings,
  useUpdateLLMSettings,
  useLLMModels,
  useTestLLMKey,
  useLLMUsage,
  useRefreshLLMModels,
} from '../../hooks/useLLMSettings';
import { getInputClassName } from '../../utils/formStyles';
import type { LLMProviderType, AvailableModel } from '../../types/llm';

interface FormState {
  high_quality_provider: LLMProviderType | null;
  high_quality_model: string | null;
  low_quality_provider: LLMProviderType | null;
  low_quality_model: string | null;
  anthropic_api_key: string;
  openai_api_key: string;
  default_temperature: number | null;
  default_max_tokens: number | null;
}

const DEFAULT_FORM_STATE: FormState = {
  high_quality_provider: 'anthropic',
  high_quality_model: null,
  low_quality_provider: 'anthropic',
  low_quality_model: null,
  anthropic_api_key: '',
  openai_api_key: '',
  default_temperature: null,
  default_max_tokens: null,
};

export default function LLMSettings() {
  const { data: settings, isLoading, error } = useLLMSettings();
  const { data: modelsData } = useLLMModels();
  const updateMutation = useUpdateLLMSettings();
  const testKeyMutation = useTestLLMKey();
  const { data: usageData } = useLLMUsage(30, settings);
  const refreshModelsMutation = useRefreshLLMModels();

  const [formState, setFormState] = useState<FormState>(DEFAULT_FORM_STATE);
  const [showAnthropicKey, setShowAnthropicKey] = useState(false);
  const [showOpenAIKey, setShowOpenAIKey] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [testResult, setTestResult] = useState<{
    provider: LLMProviderType;
    success: boolean;
    message: string;
  } | null>(null);

  // Initialize form state from settings
  useEffect(() => {
    if (settings) {
      setFormState({
        high_quality_provider: (settings.high_quality_provider as LLMProviderType) || 'anthropic',
        high_quality_model: settings.high_quality_model,
        low_quality_provider: (settings.low_quality_provider as LLMProviderType) || 'anthropic',
        low_quality_model: settings.low_quality_model,
        anthropic_api_key: '',
        openai_api_key: '',
        default_temperature: settings.default_temperature,
        default_max_tokens: settings.default_max_tokens,
      });
    }
  }, [settings]);

  const handleChange = useCallback(
    <K extends keyof FormState>(field: K, value: FormState[K]) => {
      setFormState((prev) => ({ ...prev, [field]: value }));
    },
    []
  );

  const handleProviderChange = useCallback(
    (tier: 'high' | 'low', provider: LLMProviderType) => {
      if (tier === 'high') {
        setFormState((prev) => ({
          ...prev,
          high_quality_provider: provider,
          high_quality_model: null, // Reset model when provider changes
        }));
      } else {
        setFormState((prev) => ({
          ...prev,
          low_quality_provider: provider,
          low_quality_model: null,
        }));
      }
    },
    []
  );

  const handleSave = useCallback(async () => {
    const updateData: Record<string, unknown> = {
      high_quality_provider: formState.high_quality_provider,
      high_quality_model: formState.high_quality_model,
      low_quality_provider: formState.low_quality_provider,
      low_quality_model: formState.low_quality_model,
      default_temperature: formState.default_temperature,
      default_max_tokens: formState.default_max_tokens,
    };

    // Only include API keys if they were provided
    if (formState.anthropic_api_key) {
      updateData.anthropic_api_key = formState.anthropic_api_key;
    }
    if (formState.openai_api_key) {
      updateData.openai_api_key = formState.openai_api_key;
    }

    await updateMutation.mutateAsync(updateData);

    // Clear API key fields after successful save
    setFormState((prev) => ({
      ...prev,
      anthropic_api_key: '',
      openai_api_key: '',
    }));
  }, [formState, updateMutation]);

  const handleTestKey = useCallback(
    async (provider: LLMProviderType) => {
      const apiKey =
        provider === 'anthropic' ? formState.anthropic_api_key : formState.openai_api_key;
      if (!apiKey) return;
      setTestResult(null); // Clear previous result
      try {
        const result = await testKeyMutation.mutateAsync({ provider, api_key: apiKey });
        setTestResult({
          provider,
          success: result.success,
          message: result.message,
        });
      } catch {
        setTestResult({
          provider,
          success: false,
          message: 'Test failed - check your API key',
        });
      }
    },
    [formState.anthropic_api_key, formState.openai_api_key, testKeyMutation]
  );

  // Filter models by provider
  const getModelsForProvider = useCallback(
    (provider: LLMProviderType | null): AvailableModel[] => {
      if (!modelsData?.models || !provider) return [];
      return modelsData.models.filter((m) => m.provider === provider);
    },
    [modelsData]
  );

  // Refresh models from provider API
  const handleRefreshModels = useCallback(
    async (provider: LLMProviderType) => {
      const apiKey =
        provider === 'anthropic' ? formState.anthropic_api_key : formState.openai_api_key;

      // Check if we have an API key (either entered or already saved)
      const hasStoredKey =
        provider === 'anthropic' ? settings?.anthropic_has_key : settings?.openai_has_key;

      if (!apiKey && !hasStoredKey) {
        return; // No API key available
      }

      // If user entered a key, use that; otherwise, we can't refresh (stored keys aren't accessible)
      if (!apiKey) {
        return; // Can't use stored key for refresh - user must enter key
      }

      await refreshModelsMutation.mutateAsync({ provider, apiKey });
    },
    [formState.anthropic_api_key, formState.openai_api_key, settings, refreshModelsMutation]
  );

  // Loading state
  if (isLoading) {
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
        <span className="ml-3 text-gray-500">Loading LLM settings...</span>
      </div>
    );
  }

  // Error state (except 404 which means not configured yet)
  if (error && !error.message.includes('404')) {
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
            <h3 className="text-sm font-medium text-red-800">Error loading settings</h3>
            <p className="mt-1 text-sm text-red-700">{error.message}</p>
          </div>
        </div>
      </div>
    );
  }

  const hasAnyKey = settings?.anthropic_has_key || settings?.openai_has_key;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h3 className="text-lg leading-6 font-medium text-gray-900">LLM Configuration</h3>
        <p className="mt-1 text-sm text-gray-500">
          Configure LLM providers for AI features like HyDE search and document summarization.
        </p>
      </div>

      {/* Info box for unconfigured state */}
      {!hasAnyKey && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex">
            <svg
              className="h-5 w-5 text-yellow-400 flex-shrink-0"
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
              <h3 className="text-sm font-medium text-yellow-800">LLM Not Configured</h3>
              <p className="mt-1 text-sm text-yellow-700">
                Add an API key below to enable AI features. Your keys are encrypted and stored
                securely.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Quality Tiers Info */}
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
            <h3 className="text-sm font-medium text-blue-800">Quality Tiers</h3>
            <p className="mt-1 text-sm text-blue-700">
              <strong>High Quality:</strong> Used for complex tasks like document summarization.
              <br />
              <strong>Low Quality:</strong> Used for simple tasks like HyDE query expansion
              (faster, cheaper).
            </p>
          </div>
        </div>
      </div>

      {/* API Keys Section */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">API Keys</h3>

          <div className="space-y-4">
            {/* Anthropic API Key */}
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Anthropic API Key
                {settings?.anthropic_has_key && (
                  <span className="ml-2 text-green-600 text-xs">(configured)</span>
                )}
              </label>
              <div className="mt-1 flex rounded-md shadow-sm">
                <input
                  type={showAnthropicKey ? 'text' : 'password'}
                  value={formState.anthropic_api_key}
                  onChange={(e) => handleChange('anthropic_api_key', e.target.value)}
                  placeholder={settings?.anthropic_has_key ? '••••••••••••' : 'sk-ant-...'}
                  className={getInputClassName(false, false, 'flex-1 min-w-0 block w-full rounded-none rounded-l-md sm:text-sm px-3 py-2 border')}
                />
                <button
                  type="button"
                  onClick={() => setShowAnthropicKey(!showAnthropicKey)}
                  className="inline-flex items-center px-3 border border-l-0 border-gray-300 bg-gray-50 text-gray-500 hover:bg-gray-100"
                >
                  {showAnthropicKey ? (
                    <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                    </svg>
                  ) : (
                    <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                  )}
                </button>
                <button
                  type="button"
                  onClick={() => handleTestKey('anthropic')}
                  disabled={!formState.anthropic_api_key || testKeyMutation.isPending}
                  className="inline-flex items-center px-4 border border-l-0 border-gray-300 rounded-r-md bg-gray-50 text-sm font-medium text-gray-700 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {testKeyMutation.isPending ? 'Testing...' : 'Test'}
                </button>
              </div>
              {testResult && testResult.provider === 'anthropic' && (
                <div
                  className={`mt-2 p-2 rounded text-sm ${
                    testResult.success
                      ? 'bg-green-50 text-green-700 border border-green-200'
                      : 'bg-red-50 text-red-700 border border-red-200'
                  }`}
                >
                  {testResult.success ? '✓ ' : '✗ '}
                  {testResult.message}
                </div>
              )}
            </div>

            {/* OpenAI API Key */}
            <div>
              <label className="block text-sm font-medium text-gray-700">
                OpenAI API Key
                {settings?.openai_has_key && (
                  <span className="ml-2 text-green-600 text-xs">(configured)</span>
                )}
              </label>
              <div className="mt-1 flex rounded-md shadow-sm">
                <input
                  type={showOpenAIKey ? 'text' : 'password'}
                  value={formState.openai_api_key}
                  onChange={(e) => handleChange('openai_api_key', e.target.value)}
                  placeholder={settings?.openai_has_key ? '••••••••••••' : 'sk-...'}
                  className={getInputClassName(false, false, 'flex-1 min-w-0 block w-full rounded-none rounded-l-md sm:text-sm px-3 py-2 border')}
                />
                <button
                  type="button"
                  onClick={() => setShowOpenAIKey(!showOpenAIKey)}
                  className="inline-flex items-center px-3 border border-l-0 border-gray-300 bg-gray-50 text-gray-500 hover:bg-gray-100"
                >
                  {showOpenAIKey ? (
                    <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                    </svg>
                  ) : (
                    <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                  )}
                </button>
                <button
                  type="button"
                  onClick={() => handleTestKey('openai')}
                  disabled={!formState.openai_api_key || testKeyMutation.isPending}
                  className="inline-flex items-center px-4 border border-l-0 border-gray-300 rounded-r-md bg-gray-50 text-sm font-medium text-gray-700 hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {testKeyMutation.isPending ? 'Testing...' : 'Test'}
                </button>
              </div>
              {testResult && testResult.provider === 'openai' && (
                <div
                  className={`mt-2 p-2 rounded text-sm ${
                    testResult.success
                      ? 'bg-green-50 text-green-700 border border-green-200'
                      : 'bg-red-50 text-red-700 border border-red-200'
                  }`}
                >
                  {testResult.success ? '✓ ' : '✗ '}
                  {testResult.message}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* High Quality Tier */}
      <TierConfigCard
        title="High Quality Tier"
        description="Used for complex tasks like document summarization"
        provider={formState.high_quality_provider}
        model={formState.high_quality_model}
        models={getModelsForProvider(formState.high_quality_provider)}
        onProviderChange={(p) => handleProviderChange('high', p)}
        onModelChange={(m) => handleChange('high_quality_model', m)}
        onRefreshModels={handleRefreshModels}
        canRefresh={
          formState.high_quality_provider === 'anthropic'
            ? !!formState.anthropic_api_key
            : !!formState.openai_api_key
        }
        isRefreshing={refreshModelsMutation.isPending}
        tierRecommendation="high"
      />

      {/* Low Quality Tier */}
      <TierConfigCard
        title="Low Quality Tier"
        description="Used for simple tasks like HyDE query expansion (faster, cheaper)"
        provider={formState.low_quality_provider}
        model={formState.low_quality_model}
        models={getModelsForProvider(formState.low_quality_provider)}
        onProviderChange={(p) => handleProviderChange('low', p)}
        onModelChange={(m) => handleChange('low_quality_model', m)}
        onRefreshModels={handleRefreshModels}
        canRefresh={
          formState.low_quality_provider === 'anthropic'
            ? !!formState.anthropic_api_key
            : !!formState.openai_api_key
        }
        isRefreshing={refreshModelsMutation.isPending}
        tierRecommendation="low"
      />

      {/* Advanced Settings */}
      <div className="bg-white shadow rounded-lg">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full px-4 py-4 sm:px-6 flex items-center justify-between text-left"
        >
          <h3 className="text-lg leading-6 font-medium text-gray-900">Advanced Settings</h3>
          <svg
            className={`h-5 w-5 text-gray-400 transform transition-transform ${
              showAdvanced ? 'rotate-180' : ''
            }`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {showAdvanced && (
          <div className="px-4 pb-5 sm:px-6 border-t border-gray-200 pt-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Default Temperature
                </label>
                <input
                  type="number"
                  min="0"
                  max="2"
                  step="0.1"
                  value={formState.default_temperature ?? ''}
                  onChange={(e) =>
                    handleChange(
                      'default_temperature',
                      e.target.value ? parseFloat(e.target.value) : null
                    )
                  }
                  placeholder="0.7"
                  className={getInputClassName(false, false)}
                />
                <p className="mt-1 text-xs text-gray-500">0.0 (deterministic) to 2.0 (creative)</p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">
                  Default Max Tokens
                </label>
                <input
                  type="number"
                  min="1"
                  max="200000"
                  value={formState.default_max_tokens ?? ''}
                  onChange={(e) =>
                    handleChange(
                      'default_max_tokens',
                      e.target.value ? parseInt(e.target.value, 10) : null
                    )
                  }
                  placeholder="4096"
                  className={getInputClassName(false, false)}
                />
                <p className="mt-1 text-xs text-gray-500">Maximum tokens to generate</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Token Usage */}
      {hasAnyKey && usageData && (
        <div className="bg-white shadow rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
              Token Usage (Last 30 Days)
            </h3>

            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="bg-gray-50 p-4 rounded-lg text-center">
                <div className="text-2xl font-semibold text-gray-900">
                  {usageData.total_input_tokens.toLocaleString()}
                </div>
                <div className="text-sm text-gray-500">Input Tokens</div>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg text-center">
                <div className="text-2xl font-semibold text-gray-900">
                  {usageData.total_output_tokens.toLocaleString()}
                </div>
                <div className="text-sm text-gray-500">Output Tokens</div>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg text-center">
                <div className="text-2xl font-semibold text-gray-900">
                  {usageData.total_tokens.toLocaleString()}
                </div>
                <div className="text-sm text-gray-500">Total Tokens</div>
              </div>
            </div>

            {Object.keys(usageData.by_feature).length > 0 && (
              <div>
                <h4 className="text-sm font-medium text-gray-700 mb-2">By Feature</h4>
                <table className="min-w-full divide-y divide-gray-200">
                  <thead>
                    <tr>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                        Feature
                      </th>
                      <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">
                        Input
                      </th>
                      <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">
                        Output
                      </th>
                      <th className="px-3 py-2 text-right text-xs font-medium text-gray-500 uppercase">
                        Total
                      </th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {Object.entries(usageData.by_feature).map(([feature, tokens]) => (
                      <tr key={feature}>
                        <td className="px-3 py-2 text-sm text-gray-900 capitalize">{feature}</td>
                        <td className="px-3 py-2 text-sm text-gray-500 text-right">
                          {tokens.input_tokens.toLocaleString()}
                        </td>
                        <td className="px-3 py-2 text-sm text-gray-500 text-right">
                          {tokens.output_tokens.toLocaleString()}
                        </td>
                        <td className="px-3 py-2 text-sm text-gray-900 text-right font-medium">
                          {tokens.total_tokens.toLocaleString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Save Button */}
      <div className="flex justify-end">
        <button
          type="button"
          onClick={handleSave}
          disabled={updateMutation.isPending}
          className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {updateMutation.isPending ? (
            <>
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Saving...
            </>
          ) : (
            'Save Settings'
          )}
        </button>
      </div>
    </div>
  );
}

/**
 * Tier Configuration Card subcomponent
 */
interface TierConfigCardProps {
  title: string;
  description: string;
  provider: LLMProviderType | null;
  model: string | null;
  models: AvailableModel[];
  onProviderChange: (provider: LLMProviderType) => void;
  onModelChange: (model: string | null) => void;
  onRefreshModels: (provider: LLMProviderType) => Promise<void>;
  canRefresh: boolean;
  isRefreshing: boolean;
  tierRecommendation: 'high' | 'low';
}

function TierConfigCard({
  title,
  description,
  provider,
  model,
  models,
  onProviderChange,
  onModelChange,
  onRefreshModels,
  canRefresh,
  isRefreshing,
  tierRecommendation,
}: TierConfigCardProps) {
  // Filter to show recommended models first
  const sortedModels = [...models].sort((a, b) => {
    const aRecommended = a.tier_recommendation === tierRecommendation;
    const bRecommended = b.tier_recommendation === tierRecommendation;
    if (aRecommended && !bRecommended) return -1;
    if (!aRecommended && bRecommended) return 1;
    return a.display_name.localeCompare(b.display_name);
  });

  return (
    <div className="bg-white shadow rounded-lg">
      <div className="px-4 py-5 sm:p-6">
        <h3 className="text-lg leading-6 font-medium text-gray-900">{title}</h3>
        <p className="mt-1 text-sm text-gray-500">{description}</p>

        <div className="mt-4 space-y-4">
          {/* Provider Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Provider</label>
            <div className="flex space-x-2">
              <button
                type="button"
                onClick={() => onProviderChange('anthropic')}
                className={`flex-1 px-4 py-2 text-sm font-medium rounded-md border ${
                  provider === 'anthropic'
                    ? 'bg-blue-100 border-blue-500 text-blue-700'
                    : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                }`}
              >
                Anthropic
              </button>
              <button
                type="button"
                onClick={() => onProviderChange('openai')}
                className={`flex-1 px-4 py-2 text-sm font-medium rounded-md border ${
                  provider === 'openai'
                    ? 'bg-blue-100 border-blue-500 text-blue-700'
                    : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                }`}
              >
                OpenAI
              </button>
            </div>
          </div>

          {/* Model Selection */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-gray-700">Model</label>
              {provider && (
                <button
                  type="button"
                  onClick={() => onRefreshModels(provider)}
                  disabled={!canRefresh || isRefreshing}
                  className="text-xs text-blue-600 hover:text-blue-800 disabled:text-gray-400 disabled:cursor-not-allowed"
                  title={canRefresh ? 'Refresh models from API' : 'Enter API key above to refresh'}
                >
                  {isRefreshing ? 'Refreshing...' : 'Refresh from API'}
                </button>
              )}
            </div>
            <select
              value={model || ''}
              onChange={(e) => onModelChange(e.target.value || null)}
              className={getInputClassName(false, false)}
            >
              <option value="">Select a model...</option>
              {sortedModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.display_name}
                  {m.tier_recommendation === tierRecommendation ? ' (Recommended)' : ''}
                  {!m.is_curated ? ' *' : ''}
                </option>
              ))}
            </select>
            {sortedModels.some((m) => !m.is_curated) && (
              <p className="mt-1 text-xs text-gray-500">* Fetched from provider API</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
