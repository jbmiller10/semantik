import { render, screen } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import AdminTab from '../AdminTab';

// Mock DatabaseSettings to avoid its complex dependencies
vi.mock('../DatabaseSettings', () => ({
  default: () => <div data-testid="database-settings">Database Settings</div>,
}));

describe('AdminTab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the admin header', () => {
    render(<AdminTab />);

    expect(screen.getByText('Admin Settings')).toBeInTheDocument();
  });

  it('renders the admin description', () => {
    render(<AdminTab />);

    expect(
      screen.getByText(
        'Administrative operations and system configuration. These settings affect all users.'
      )
    ).toBeInTheDocument();
  });

  it('renders the Shield icon in the header', () => {
    render(<AdminTab />);

    const header = screen.getByText('Admin Settings').closest('h3');
    expect(header?.querySelector('svg')).toBeInTheDocument();
  });

  it('renders DatabaseSettings component', () => {
    render(<AdminTab />);

    expect(screen.getByTestId('database-settings')).toBeInTheDocument();
  });
});
