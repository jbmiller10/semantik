import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import LLMSettings from '../LLMSettings';
import * as useLLMSettingsModule from '@/hooks/useLLMSettings';
import type {
  LLMSettingsResponse,
  AvailableModelsResponse,
  TokenUsageResponse,
} from '@/types/llm';

// Mock the hooks
vi.mock('@/hooks/useLLMSettings', () => ({
  useLLMSettings: vi.fn(),
  useUpdateLLMSettings: vi.fn(),
  useLLMModels: vi.fn(),
  useTestLLMKey: vi.fn(),
  useLLMUsage: vi.fn(),
  useRefreshLLMModels: vi.fn(),
}));

// Mock data
const mockSettings: LLMSettingsResponse = {
  high_quality_provider: 'anthropic',
  high_quality_model: 'claude-opus-4-5-20251101',
  low_quality_provider: 'anthropic',
  low_quality_model: 'claude-3-5-haiku-20241022',
  anthropic_has_key: true,
  openai_has_key: false,
  default_temperature: 0.7,
  default_max_tokens: 4096,
  created_at: '2025-01-01T00:00:00Z',
  updated_at: '2025-01-01T00:00:00Z',
};

const mockModels: AvailableModelsResponse = {
  models: [
    {
      id: 'claude-opus-4-5-20251101',
      name: 'Opus 4.5',
      display_name: 'Claude - Opus 4.5',
      provider: 'anthropic',
      tier_recommendation: 'high',
      context_window: 200000,
      description: 'Most capable model',
      is_curated: true,
    },
    {
      id: 'claude-3-5-haiku-20241022',
      name: 'Haiku 3.5',
      display_name: 'Claude - Haiku 3.5',
      provider: 'anthropic',
      tier_recommendation: 'low',
      context_window: 200000,
      description: 'Fast and cheap',
      is_curated: true,
    },
    {
      id: 'gpt-4o',
      name: 'GPT-4o',
      display_name: 'OpenAI - GPT-4o',
      provider: 'openai',
      tier_recommendation: 'high',
      context_window: 128000,
      description: 'OpenAI flagship',
      is_curated: true,
    },
    {
      id: 'gpt-4o-mini',
      name: 'GPT-4o Mini',
      display_name: 'OpenAI - GPT-4o Mini',
      provider: 'openai',
      tier_recommendation: 'low',
      context_window: 128000,
      description: 'Fast and affordable',
      is_curated: true,
    },
  ],
};

const mockUsage: TokenUsageResponse = {
  total_input_tokens: 58023,
  total_output_tokens: 30245,
  total_tokens: 88268,
  by_feature: {
    hyde: { input_tokens: 12345, output_tokens: 6789, total_tokens: 19134 },
    summary: { input_tokens: 45678, output_tokens: 23456, total_tokens: 69134 },
  },
  by_provider: {
    anthropic: { input_tokens: 50000, output_tokens: 25000, total_tokens: 75000 },
  },
  event_count: 156,
  period_days: 30,
};

describe('LLMSettings', () => {
  const mockMutateAsync = vi.fn();
  const mockTestMutateAsync = vi.fn();
  const mockRefreshMutateAsync = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();

    // Default mock implementations
    vi.mocked(useLLMSettingsModule.useLLMSettings).mockReturnValue({
      data: mockSettings,
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useLLMSettingsModule.useLLMSettings>);

    vi.mocked(useLLMSettingsModule.useLLMModels).mockReturnValue({
      data: mockModels,
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useLLMSettingsModule.useLLMModels>);

    vi.mocked(useLLMSettingsModule.useUpdateLLMSettings).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof useLLMSettingsModule.useUpdateLLMSettings>);

    vi.mocked(useLLMSettingsModule.useTestLLMKey).mockReturnValue({
      mutateAsync: mockTestMutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof useLLMSettingsModule.useTestLLMKey>);

    vi.mocked(useLLMSettingsModule.useLLMUsage).mockReturnValue({
      data: mockUsage,
      isLoading: false,
      error: null,
    } as unknown as ReturnType<typeof useLLMSettingsModule.useLLMUsage>);

    vi.mocked(useLLMSettingsModule.useRefreshLLMModels).mockReturnValue({
      mutateAsync: mockRefreshMutateAsync,
      isPending: false,
    } as unknown as ReturnType<typeof useLLMSettingsModule.useRefreshLLMModels>);
  });

  describe('loading state', () => {
    it('shows loading spinner when settings are loading', () => {
      vi.mocked(useLLMSettingsModule.useLLMSettings).mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
      } as unknown as ReturnType<typeof useLLMSettingsModule.useLLMSettings>);

      render(<LLMSettings />);

      expect(screen.getByText('Loading LLM settings...')).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('shows error message when loading fails', () => {
      vi.mocked(useLLMSettingsModule.useLLMSettings).mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Network error'),
      } as unknown as ReturnType<typeof useLLMSettingsModule.useLLMSettings>);

      render(<LLMSettings />);

      expect(screen.getByText('Error loading settings')).toBeInTheDocument();
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });

    it('does not show error for 404 (not configured)', () => {
      vi.mocked(useLLMSettingsModule.useLLMSettings).mockReturnValue({
        data: undefined,
        isLoading: false,
        error: new Error('Request failed with status code 404'),
      } as unknown as ReturnType<typeof useLLMSettingsModule.useLLMSettings>);

      render(<LLMSettings />);

      // Should show unconfigured state instead of error
      expect(screen.queryByText('Error loading settings')).not.toBeInTheDocument();
      expect(screen.getByText('LLM Configuration')).toBeInTheDocument();
    });
  });

  describe('unconfigured state', () => {
    it('shows info box when no API keys are configured', () => {
      vi.mocked(useLLMSettingsModule.useLLMSettings).mockReturnValue({
        data: { ...mockSettings, anthropic_has_key: false, openai_has_key: false },
        isLoading: false,
        error: null,
      } as unknown as ReturnType<typeof useLLMSettingsModule.useLLMSettings>);

      render(<LLMSettings />);

      expect(screen.getByText('LLM Not Configured')).toBeInTheDocument();
      expect(
        screen.getByText(/Add an API key below to enable AI features/)
      ).toBeInTheDocument();
    });
  });

  describe('configured state', () => {
    it('renders main sections when settings are loaded', () => {
      render(<LLMSettings />);

      expect(screen.getByText('LLM Configuration')).toBeInTheDocument();
      expect(screen.getByText('API Keys')).toBeInTheDocument();
      expect(screen.getByText('High Quality Tier')).toBeInTheDocument();
      expect(screen.getByText('Low Quality Tier')).toBeInTheDocument();
      expect(screen.getByText('Advanced Settings')).toBeInTheDocument();
    });

    it('shows configured status for Anthropic API key', () => {
      render(<LLMSettings />);

      expect(screen.getByText('Anthropic API Key')).toBeInTheDocument();
      expect(screen.getByText('(configured)')).toBeInTheDocument();
    });

    it('shows quality tiers info box', () => {
      render(<LLMSettings />);

      expect(screen.getByText('Quality Tiers')).toBeInTheDocument();
      expect(screen.getByText(/High Quality:/)).toBeInTheDocument();
      expect(screen.getByText(/Low Quality:/)).toBeInTheDocument();
    });
  });

  describe('API key testing', () => {
    it('test button is disabled when no API key is entered', () => {
      render(<LLMSettings />);

      const testButtons = screen.getAllByRole('button', { name: 'Test' });
      testButtons.forEach((button) => {
        expect(button).toBeDisabled();
      });
    });

    it('calls test mutation when API key is entered and test clicked', async () => {
      const user = userEvent.setup();
      mockTestMutateAsync.mockResolvedValue({
        success: true,
        message: 'Success',
        model_tested: 'claude-3-5-haiku-20241022',
      });

      render(<LLMSettings />);

      // Find the Anthropic API key input and enter a value
      const inputs = screen.getAllByPlaceholderText(/sk-ant-|••••••••••••/);
      const anthropicInput = inputs[0];
      await user.type(anthropicInput, 'sk-ant-test-key');

      // Click the test button
      const testButtons = screen.getAllByRole('button', { name: 'Test' });
      await user.click(testButtons[0]);

      await waitFor(() => {
        expect(mockTestMutateAsync).toHaveBeenCalledWith({
          provider: 'anthropic',
          api_key: 'sk-ant-test-key',
        });
      });
    });

    it('shows inline success message after successful test', async () => {
      const user = userEvent.setup();
      mockTestMutateAsync.mockResolvedValue({
        success: true,
        message: 'Successfully connected to Anthropic API',
        model_tested: 'claude-3-5-haiku-20241022',
      });

      render(<LLMSettings />);

      const inputs = screen.getAllByPlaceholderText(/sk-ant-|••••••••••••/);
      await user.type(inputs[0], 'sk-ant-test-key');

      const testButtons = screen.getAllByRole('button', { name: 'Test' });
      await user.click(testButtons[0]);

      await waitFor(() => {
        expect(screen.getByText(/✓.*Successfully connected to Anthropic API/)).toBeInTheDocument();
      });
    });

    it('shows inline error message after failed test', async () => {
      const user = userEvent.setup();
      mockTestMutateAsync.mockResolvedValue({
        success: false,
        message: 'Invalid API key',
        model_tested: null,
      });

      render(<LLMSettings />);

      const inputs = screen.getAllByPlaceholderText(/sk-ant-|••••••••••••/);
      await user.type(inputs[0], 'invalid-key');

      const testButtons = screen.getAllByRole('button', { name: 'Test' });
      await user.click(testButtons[0]);

      await waitFor(() => {
        expect(screen.getByText(/✗.*Invalid API key/)).toBeInTheDocument();
      });
    });
  });

  describe('provider selection', () => {
    it('renders provider buttons for both tiers', () => {
      render(<LLMSettings />);

      // Should have 2 Anthropic buttons (one per tier) + 2 OpenAI buttons
      const anthropicButtons = screen.getAllByRole('button', { name: 'Anthropic' });
      const openaiButtons = screen.getAllByRole('button', { name: 'OpenAI' });

      expect(anthropicButtons).toHaveLength(2);
      expect(openaiButtons).toHaveLength(2);
    });

    it('updates model dropdown when provider changes', async () => {
      const user = userEvent.setup();
      render(<LLMSettings />);

      // Initially on Anthropic, should show Claude models
      const selects = screen.getAllByRole('combobox');
      expect(selects[0]).toHaveTextContent('Claude - Opus 4.5');

      // Switch high quality tier to OpenAI
      const openaiButtons = screen.getAllByRole('button', { name: 'OpenAI' });
      await user.click(openaiButtons[0]);

      // The model dropdown should now be reset
      await waitFor(() => {
        expect(selects[0]).toHaveValue('');
      });
    });
  });

  describe('model selection', () => {
    it('shows models filtered by provider', () => {
      render(<LLMSettings />);

      // High quality tier with Anthropic provider
      const selects = screen.getAllByRole('combobox');
      const highQualitySelect = selects[0];

      // Should contain Anthropic models only (since provider is anthropic)
      expect(highQualitySelect).toContainHTML('Claude - Opus 4.5');
      expect(highQualitySelect).not.toContainHTML('OpenAI - GPT-4o');
    });
  });

  describe('save functionality', () => {
    it('calls update mutation when save is clicked', async () => {
      const user = userEvent.setup();
      mockMutateAsync.mockResolvedValue(mockSettings);

      render(<LLMSettings />);

      const saveButton = screen.getByRole('button', { name: 'Save Settings' });
      await user.click(saveButton);

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalled();
      });
    });

    it('shows loading state during save', () => {
      vi.mocked(useLLMSettingsModule.useUpdateLLMSettings).mockReturnValue({
        mutateAsync: mockMutateAsync,
        isPending: true,
      } as unknown as ReturnType<typeof useLLMSettingsModule.useUpdateLLMSettings>);

      render(<LLMSettings />);

      expect(screen.getByText('Saving...')).toBeInTheDocument();
    });
  });

  describe('advanced settings', () => {
    it('expands advanced settings on click', async () => {
      const user = userEvent.setup();
      render(<LLMSettings />);

      const advancedButton = screen.getByRole('button', { name: /Advanced Settings/i });
      await user.click(advancedButton);

      expect(screen.getByText('Default Temperature')).toBeInTheDocument();
      expect(screen.getByText('Default Max Tokens')).toBeInTheDocument();
    });

    it('shows temperature and max tokens inputs when expanded', async () => {
      const user = userEvent.setup();
      render(<LLMSettings />);

      await user.click(screen.getByRole('button', { name: /Advanced Settings/i }));

      const tempInput = screen.getByPlaceholderText('0.7');
      const maxTokensInput = screen.getByPlaceholderText('4096');

      expect(tempInput).toBeInTheDocument();
      expect(maxTokensInput).toBeInTheDocument();
    });
  });

  describe('token usage display', () => {
    it('shows token usage when settings are configured', () => {
      render(<LLMSettings />);

      expect(screen.getByText('Token Usage (Last 30 Days)')).toBeInTheDocument();
      expect(screen.getByText('58,023')).toBeInTheDocument(); // Total input
      expect(screen.getByText('30,245')).toBeInTheDocument(); // Total output
      expect(screen.getByText('88,268')).toBeInTheDocument(); // Total
    });

    it('shows usage breakdown by feature', () => {
      render(<LLMSettings />);

      expect(screen.getByText('By Feature')).toBeInTheDocument();
      expect(screen.getByText('hyde')).toBeInTheDocument();
      expect(screen.getByText('summary')).toBeInTheDocument();
    });

    it('does not show usage when no keys configured', () => {
      vi.mocked(useLLMSettingsModule.useLLMSettings).mockReturnValue({
        data: { ...mockSettings, anthropic_has_key: false, openai_has_key: false },
        isLoading: false,
        error: null,
      } as unknown as ReturnType<typeof useLLMSettingsModule.useLLMSettings>);

      render(<LLMSettings />);

      expect(screen.queryByText('Token Usage (Last 30 Days)')).not.toBeInTheDocument();
    });
  });

  describe('model refresh', () => {
    it('shows refresh button when API key is entered', async () => {
      const user = userEvent.setup();
      render(<LLMSettings />);

      // Enter an API key
      const inputs = screen.getAllByPlaceholderText(/sk-ant-|••••••••••••/);
      await user.type(inputs[0], 'sk-ant-test-key');

      // Refresh button should be enabled
      const refreshButtons = screen.getAllByRole('button', { name: /Refresh from API/i });
      expect(refreshButtons[0]).toBeEnabled();
    });

    it('calls refresh mutation when refresh is clicked', async () => {
      const user = userEvent.setup();
      mockRefreshMutateAsync.mockResolvedValue(mockModels);

      render(<LLMSettings />);

      // Enter an API key first
      const inputs = screen.getAllByPlaceholderText(/sk-ant-|••••••••••••/);
      await user.type(inputs[0], 'sk-ant-test-key');

      // Click refresh
      const refreshButtons = screen.getAllByRole('button', { name: /Refresh from API/i });
      await user.click(refreshButtons[0]);

      await waitFor(() => {
        expect(mockRefreshMutateAsync).toHaveBeenCalledWith({
          provider: 'anthropic',
          apiKey: 'sk-ant-test-key',
        });
      });
    });
  });
});
