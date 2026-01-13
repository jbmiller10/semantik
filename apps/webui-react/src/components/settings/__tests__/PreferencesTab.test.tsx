import { render, screen } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import PreferencesTab from '../PreferencesTab';
import { useSettingsUIStore } from '../../../stores/settingsUIStore';

// Mock the settings UI store
vi.mock('../../../stores/settingsUIStore', () => ({
  useSettingsUIStore: vi.fn(),
}));

// Mock child settings components to avoid their complex dependencies
vi.mock('../SearchPreferencesSettings', () => ({
  default: () => (
    <div data-testid="search-preferences">Search Preferences Content</div>
  ),
}));

vi.mock('../CollectionDefaultsSettings', () => ({
  default: () => (
    <div data-testid="collection-defaults">Collection Defaults Content</div>
  ),
}));

vi.mock('../LLMSettings', () => ({
  default: () => <div data-testid="llm-settings">LLM Settings Content</div>,
}));

describe('PreferencesTab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useSettingsUIStore).mockReturnValue({
      toggleSection: vi.fn(),
      isSectionOpen: vi.fn().mockReturnValue(true),
      sectionStates: {},
      setSectionOpen: vi.fn(),
      resetSectionStates: vi.fn(),
    });
  });

  it('renders the preferences header', () => {
    render(<PreferencesTab />);

    expect(screen.getByText('Preferences')).toBeInTheDocument();
    expect(
      screen.getByText(
        'Configure your search, collection defaults, and AI settings.'
      )
    ).toBeInTheDocument();
  });

  it('renders the Search Preferences section', () => {
    render(<PreferencesTab />);

    expect(screen.getByText('Search Preferences')).toBeInTheDocument();
    expect(screen.getByTestId('search-preferences')).toBeInTheDocument();
  });

  it('renders the Collection Defaults section', () => {
    render(<PreferencesTab />);

    expect(screen.getByText('Collection Defaults')).toBeInTheDocument();
    expect(screen.getByTestId('collection-defaults')).toBeInTheDocument();
  });

  it('renders the LLM Configuration section', () => {
    render(<PreferencesTab />);

    expect(screen.getByText('LLM Configuration')).toBeInTheDocument();
    expect(screen.getByTestId('llm-settings')).toBeInTheDocument();
  });

  it('renders all three CollapsibleSections', () => {
    render(<PreferencesTab />);

    // Each section has a button for expand/collapse
    const buttons = screen.getAllByRole('button');
    expect(buttons).toHaveLength(3);
  });

  describe('default open states', () => {
    it('passes defaultOpen=true to Search section', () => {
      const mockIsSectionOpen = vi.fn().mockReturnValue(true);
      vi.mocked(useSettingsUIStore).mockReturnValue({
        toggleSection: vi.fn(),
        isSectionOpen: mockIsSectionOpen,
        sectionStates: {},
        setSectionOpen: vi.fn(),
        resetSectionStates: vi.fn(),
      });

      render(<PreferencesTab />);

      // Search section should check for its open state
      expect(mockIsSectionOpen).toHaveBeenCalledWith(
        'preferences-search',
        true
      );
    });

    it('passes defaultOpen=false to Collection Defaults and LLM sections', () => {
      const mockIsSectionOpen = vi.fn().mockReturnValue(false);
      vi.mocked(useSettingsUIStore).mockReturnValue({
        toggleSection: vi.fn(),
        isSectionOpen: mockIsSectionOpen,
        sectionStates: {},
        setSectionOpen: vi.fn(),
        resetSectionStates: vi.fn(),
      });

      render(<PreferencesTab />);

      // Check Collection Defaults and LLM sections default to closed
      expect(mockIsSectionOpen).toHaveBeenCalledWith(
        'preferences-collection-defaults',
        false
      );
      expect(mockIsSectionOpen).toHaveBeenCalledWith('preferences-llm', false);
    });
  });
});
