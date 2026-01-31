// apps/webui-react/src/components/wizard/steps/__tests__/BasicsStep.test.tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BasicsStep } from '../BasicsStep';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Mock the connector hooks
vi.mock('../../../../hooks/useConnectors', () => ({
  useConnectorCatalog: () => ({
    data: {
      directory: {
        name: 'Directory',
        fields: [{ name: 'path', label: 'Path', type: 'string', required: true }],
        secrets: [],
      },
      git: {
        name: 'Git',
        fields: [{ name: 'repo_url', label: 'Repository URL', type: 'string', required: true }],
        secrets: [],
      },
    },
    isLoading: false,
  }),
  useGitPreview: () => ({ mutateAsync: vi.fn() }),
  useImapPreview: () => ({ mutateAsync: vi.fn() }),
}));

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: false } },
});

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>
    {children}
  </QueryClientProvider>
);

describe('BasicsStep', () => {
  const defaultProps = {
    name: '',
    description: '',
    connectorType: 'none',
    configValues: {},
    secrets: {},
    syncMode: 'one_time' as const,
    onNameChange: vi.fn(),
    onDescriptionChange: vi.fn(),
    onConnectorTypeChange: vi.fn(),
    onConfigChange: vi.fn(),
    onSecretsChange: vi.fn(),
    onSyncModeChange: vi.fn(),
    errors: {},
  };

  it('renders name input', () => {
    render(<BasicsStep {...defaultProps} />, { wrapper });
    expect(screen.getByLabelText(/collection name/i)).toBeInTheDocument();
  });

  it('renders description textarea', () => {
    render(<BasicsStep {...defaultProps} />, { wrapper });
    expect(screen.getByLabelText(/description/i)).toBeInTheDocument();
  });

  it('renders source type selector', () => {
    render(<BasicsStep {...defaultProps} />, { wrapper });
    expect(screen.getByText(/select source type/i)).toBeInTheDocument();
  });

  it('calls onNameChange when name is typed', async () => {
    const onNameChange = vi.fn();
    render(<BasicsStep {...defaultProps} onNameChange={onNameChange} />, { wrapper });

    const input = screen.getByLabelText(/collection name/i);
    await userEvent.type(input, 'T');

    expect(onNameChange).toHaveBeenCalled();
  });

  it('shows validation error when provided', () => {
    render(<BasicsStep {...defaultProps} errors={{ name: 'Name is required' }} />, { wrapper });
    expect(screen.getByText('Name is required')).toBeInTheDocument();
  });

  it('shows sync mode options when source is selected', () => {
    render(<BasicsStep {...defaultProps} connectorType="directory" />, { wrapper });
    expect(screen.getByText(/one-time import/i)).toBeInTheDocument();
    expect(screen.getByText(/continuous sync/i)).toBeInTheDocument();
  });
});
