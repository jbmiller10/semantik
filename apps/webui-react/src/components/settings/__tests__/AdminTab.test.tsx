import { render, screen } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import AdminTab from '../AdminTab';

// Mock the settings components to avoid their complex dependencies
vi.mock('../ResourceLimitsSettings', () => ({
  default: () => <div data-testid="resource-limits-settings">Resource Limits Settings</div>,
}));

vi.mock('../PerformanceSettings', () => ({
  default: () => <div data-testid="performance-settings">Performance Settings</div>,
}));

vi.mock('../GpuMemorySettings', () => ({
  default: () => <div data-testid="gpu-memory-settings">GPU Memory Settings</div>,
}));

vi.mock('../SearchRerankSettings', () => ({
  default: () => <div data-testid="search-rerank-settings">Search Rerank Settings</div>,
}));

vi.mock('../DangerZoneSettings', () => ({
  default: () => <div data-testid="danger-zone-settings">Danger Zone Settings</div>,
}));

// Mock the settingsUIStore
vi.mock('../../../stores/settingsUIStore', () => ({
  useSettingsUIStore: () => ({
    isSectionOpen: (name: string, defaultOpen: boolean) => defaultOpen,
    toggleSection: vi.fn(),
  }),
}));

function renderWithProviders(component: React.ReactNode) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>{component}</BrowserRouter>
    </QueryClientProvider>
  );
}

describe('AdminTab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the admin header', () => {
    renderWithProviders(<AdminTab />);

    expect(screen.getByText('Admin Settings')).toBeInTheDocument();
  });

  it('renders the admin description', () => {
    renderWithProviders(<AdminTab />);

    expect(
      screen.getByText(
        'Administrative operations and system configuration. These settings affect all users.'
      )
    ).toBeInTheDocument();
  });

  it('renders the Shield icon in the header', () => {
    renderWithProviders(<AdminTab />);

    const header = screen.getByText('Admin Settings').closest('h3');
    expect(header?.querySelector('svg')).toBeInTheDocument();
  });

  it('renders collapsible sections with correct titles', () => {
    renderWithProviders(<AdminTab />);

    // Check that all section titles are present
    expect(screen.getByText('Resource Limits')).toBeInTheDocument();
    expect(screen.getByText('Performance Tuning')).toBeInTheDocument();
    expect(screen.getByText('GPU & Memory')).toBeInTheDocument();
    expect(screen.getByText('Search & Reranking')).toBeInTheDocument();
    expect(screen.getByText('Danger Zone')).toBeInTheDocument();
  });

  it('renders ResourceLimitsSettings component (defaultOpen=true)', () => {
    renderWithProviders(<AdminTab />);

    // ResourceLimitsSettings is default open, so it should be rendered
    expect(screen.getByTestId('resource-limits-settings')).toBeInTheDocument();
  });

  it('does not render collapsed sections by default', () => {
    renderWithProviders(<AdminTab />);

    // These sections have defaultOpen=false, so they should not render their content
    // Note: The section titles are still visible (in the collapsible header)
    // but the content is not rendered when collapsed
    expect(screen.queryByTestId('performance-settings')).not.toBeInTheDocument();
    expect(screen.queryByTestId('gpu-memory-settings')).not.toBeInTheDocument();
    expect(screen.queryByTestId('search-rerank-settings')).not.toBeInTheDocument();
    expect(screen.queryByTestId('danger-zone-settings')).not.toBeInTheDocument();
  });
});
