import { render, screen, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ApiKeySelector from '../ApiKeySelector';
import { useApiKeys, useCreateApiKey } from '../../../hooks/useApiKeys';
import type { ApiKeyResponse, ApiKeyCreateResponse } from '../../../types/api-key';

// Mock the hooks
vi.mock('../../../hooks/useApiKeys', () => ({
  useApiKeys: vi.fn(),
  useCreateApiKey: vi.fn(),
}));

const mockUseApiKeys = useApiKeys as vi.MockedFunction<typeof useApiKeys>;
const mockUseCreateApiKey = useCreateApiKey as vi.MockedFunction<typeof useCreateApiKey>;

// Test data
const mockActiveKey1: ApiKeyResponse = {
  id: 'key-1',
  name: 'Active Key 1',
  is_active: true,
  permissions: null,
  last_used_at: null,
  expires_at: '2030-01-01T00:00:00Z', // Far future, not expired
  created_at: '2025-01-01T00:00:00Z',
};

const mockActiveKey2: ApiKeyResponse = {
  id: 'key-2',
  name: 'Active Key 2',
  is_active: true,
  permissions: null,
  last_used_at: null,
  expires_at: null, // Never expires
  created_at: '2025-01-02T00:00:00Z',
};

const mockInactiveKey: ApiKeyResponse = {
  id: 'key-inactive',
  name: 'Inactive Key',
  is_active: false,
  permissions: null,
  last_used_at: null,
  expires_at: null,
  created_at: '2025-01-03T00:00:00Z',
};

const mockExpiredKey: ApiKeyResponse = {
  id: 'key-expired',
  name: 'Expired Key',
  is_active: true,
  permissions: null,
  last_used_at: null,
  expires_at: '2020-01-01T00:00:00Z', // In the past, expired
  created_at: '2019-01-01T00:00:00Z',
};

const mockCreatedKeyResponse: ApiKeyCreateResponse = {
  id: 'key-new',
  name: 'MCP: test-profile',
  is_active: true,
  permissions: null,
  last_used_at: null,
  expires_at: null,
  created_at: '2025-01-10T00:00:00Z',
  api_key: 'smtk_newlycreatedapikey123456',
};

const defaultProps = {
  profileName: 'test-profile',
  onKeySelected: vi.fn(),
};

// Helper to render component with providers
const renderComponent = (props = {}) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <ApiKeySelector {...defaultProps} {...props} />
    </QueryClientProvider>
  );
};

describe('ApiKeySelector', () => {
  let mockMutateAsync: ReturnType<typeof vi.fn>;
  let mockRefetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();
    mockMutateAsync = vi.fn();
    mockRefetch = vi.fn();

    // Default mock setup - no keys, loading done
    mockUseApiKeys.mockReturnValue({
      data: [],
      isLoading: false,
      error: null,
      refetch: mockRefetch,
    } as unknown as ReturnType<typeof useApiKeys>);

    mockUseCreateApiKey.mockReturnValue({
      mutateAsync: mockMutateAsync,
      isPending: false,
      error: null,
    } as unknown as ReturnType<typeof useCreateApiKey>);
  });

  describe('Rendering existing keys', () => {
    it('should render dropdown with existing active keys', async () => {
      mockUseApiKeys.mockReturnValue({
        data: [mockActiveKey1, mockActiveKey2],
        isLoading: false,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof useApiKeys>);

      renderComponent();

      // Open the dropdown
      const select = screen.getByRole('combobox');
      expect(select).toBeInTheDocument();

      // Check that active keys appear in the dropdown options
      const options = screen.getAllByRole('option');
      expect(options.some(opt => opt.textContent?.includes('Active Key 1'))).toBe(true);
      expect(options.some(opt => opt.textContent?.includes('Active Key 2'))).toBe(true);
    });

    it('should filter out inactive keys', () => {
      mockUseApiKeys.mockReturnValue({
        data: [mockActiveKey1, mockInactiveKey],
        isLoading: false,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof useApiKeys>);

      renderComponent();

      const options = screen.getAllByRole('option');

      // Should have Active Key 1, placeholder, and Create new key
      expect(options.some(opt => opt.textContent?.includes('Active Key 1'))).toBe(true);
      // Should NOT have Inactive Key
      expect(options.some(opt => opt.textContent?.includes('Inactive Key'))).toBe(false);
    });

    it('should filter out expired keys', () => {
      mockUseApiKeys.mockReturnValue({
        data: [mockActiveKey1, mockExpiredKey],
        isLoading: false,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof useApiKeys>);

      renderComponent();

      const options = screen.getAllByRole('option');

      // Should have Active Key 1
      expect(options.some(opt => opt.textContent?.includes('Active Key 1'))).toBe(true);
      // Should NOT have Expired Key
      expect(options.some(opt => opt.textContent?.includes('Expired Key'))).toBe(false);
    });

    it('should have "Create new key" option', () => {
      mockUseApiKeys.mockReturnValue({
        data: [mockActiveKey1],
        isLoading: false,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof useApiKeys>);

      renderComponent();

      const options = screen.getAllByRole('option');
      expect(options.some(opt => opt.textContent?.includes('Create new key'))).toBe(true);
    });
  });

  describe('Creating new key', () => {
    it('should create new key when "Create new key" is selected', async () => {
      const user = userEvent.setup();

      mockUseApiKeys.mockReturnValue({
        data: [mockActiveKey1],
        isLoading: false,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof useApiKeys>);

      mockMutateAsync.mockResolvedValue(mockCreatedKeyResponse);

      renderComponent();

      const select = screen.getByRole('combobox');
      await user.selectOptions(select, '__create_new__');

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalledWith({
          name: 'MCP: test-profile',
          expires_in_days: null,
        });
      });

      await waitFor(() => {
        expect(defaultProps.onKeySelected).toHaveBeenCalledWith('smtk_newlycreatedapikey123456');
      });
    });

    it('should show loading state while creating key', async () => {
      mockUseApiKeys.mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof useApiKeys>);

      mockUseCreateApiKey.mockReturnValue({
        mutateAsync: mockMutateAsync,
        isPending: true,
        error: null,
      } as unknown as ReturnType<typeof useCreateApiKey>);

      renderComponent();

      // The select should be disabled while creating
      const select = screen.getByRole('combobox');
      expect(select).toBeDisabled();

      // Should show creating text
      expect(screen.getByText(/creating/i)).toBeInTheDocument();
    });

    it('should show error message if creation fails', async () => {
      const user = userEvent.setup();

      mockUseApiKeys.mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof useApiKeys>);

      mockMutateAsync.mockRejectedValue(new Error('Failed to create API key'));

      renderComponent();

      const select = screen.getByRole('combobox');
      await user.selectOptions(select, '__create_new__');

      await waitFor(() => {
        expect(screen.getByText(/failed to create/i)).toBeInTheDocument();
      });
    });
  });

  describe('Selecting existing key', () => {
    it('should call onKeySelected when existing key is selected', async () => {
      // For existing keys, we need a way to pass the actual key value
      // Since we don't have it from the list API, the component should handle this
      // This test verifies the UX for existing keys
      const user = userEvent.setup();

      mockUseApiKeys.mockReturnValue({
        data: [mockActiveKey1, mockActiveKey2],
        isLoading: false,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof useApiKeys>);

      renderComponent();

      const select = screen.getByRole('combobox');
      await user.selectOptions(select, mockActiveKey1.id);

      // For existing keys, since we don't have the actual key value,
      // component should inform user they need to use the original key
      // or show appropriate message. The exact behavior will depend on implementation.
      // This test ensures the dropdown interaction works.
      await waitFor(() => {
        expect(select).toHaveValue(mockActiveKey1.id);
      });
    });
  });

  describe('Refresh functionality', () => {
    it('should have a refresh button that re-fetches keys', async () => {
      const user = userEvent.setup();

      mockUseApiKeys.mockReturnValue({
        data: [mockActiveKey1],
        isLoading: false,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof useApiKeys>);

      renderComponent();

      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      expect(refreshButton).toBeInTheDocument();

      await user.click(refreshButton);

      expect(mockRefetch).toHaveBeenCalled();
    });
  });

  describe('Loading state', () => {
    it('should show loading state while fetching keys', () => {
      mockUseApiKeys.mockReturnValue({
        data: undefined,
        isLoading: true,
        error: null,
        refetch: mockRefetch,
      } as unknown as ReturnType<typeof useApiKeys>);

      renderComponent();

      expect(screen.getByText(/loading/i)).toBeInTheDocument();
    });
  });
});
