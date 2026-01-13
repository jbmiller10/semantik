import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { CollapsibleSection } from '../CollapsibleSection';
import { useSettingsUIStore } from '../../../stores/settingsUIStore';
import { Settings } from 'lucide-react';

// Mock the store
vi.mock('../../../stores/settingsUIStore', () => ({
  useSettingsUIStore: vi.fn(),
}));

describe('CollapsibleSection', () => {
  const mockToggleSection = vi.fn();
  const mockIsSectionOpen = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useSettingsUIStore).mockReturnValue({
      toggleSection: mockToggleSection,
      isSectionOpen: mockIsSectionOpen,
      sectionStates: {},
      setSectionOpen: vi.fn(),
      resetSectionStates: vi.fn(),
    });
    // Default to open
    mockIsSectionOpen.mockReturnValue(true);
  });

  describe('rendering', () => {
    it('renders title', () => {
      render(
        <CollapsibleSection name="test" title="Test Section">
          <div>Content</div>
        </CollapsibleSection>
      );

      expect(screen.getByText('Test Section')).toBeInTheDocument();
    });

    it('renders icon when provided', () => {
      render(
        <CollapsibleSection name="test" title="Test Section" icon={Settings}>
          <div>Content</div>
        </CollapsibleSection>
      );

      // The Settings icon should be rendered
      const button = screen.getByRole('button');
      expect(button.querySelector('svg')).toBeInTheDocument();
    });

    it('renders children when expanded', () => {
      render(
        <CollapsibleSection name="test" title="Test Section">
          <div>Test Content</div>
        </CollapsibleSection>
      );

      expect(screen.getByText('Test Content')).toBeInTheDocument();
    });

    it('hides children when collapsed', () => {
      mockIsSectionOpen.mockReturnValue(false);

      render(
        <CollapsibleSection name="test" title="Test Section">
          <div>Test Content</div>
        </CollapsibleSection>
      );

      expect(screen.queryByText('Test Content')).not.toBeInTheDocument();
    });

    it('renders badge when provided', () => {
      render(
        <CollapsibleSection name="test" title="Test Section" badge={<span>3</span>}>
          <div>Content</div>
        </CollapsibleSection>
      );

      expect(screen.getByText('3')).toBeInTheDocument();
    });
  });

  describe('loading state', () => {
    it('shows spinner in header when isLoading=true', () => {
      render(
        <CollapsibleSection name="test" title="Test Section" isLoading={true}>
          <div>Content</div>
        </CollapsibleSection>
      );

      // Check for spinner SVG with animate-spin class
      const spinner = document.querySelector('.animate-spin');
      expect(spinner).toBeInTheDocument();
    });

    it('shows skeleton when isLoading=true and expanded', () => {
      render(
        <CollapsibleSection name="test" title="Test Section" isLoading={true}>
          <div>Should not show</div>
        </CollapsibleSection>
      );

      // Skeleton has animate-pulse class
      const skeleton = document.querySelector('.animate-pulse');
      expect(skeleton).toBeInTheDocument();
      // Children should not be visible
      expect(screen.queryByText('Should not show')).not.toBeInTheDocument();
    });

    it('shows children instead of skeleton when not loading', () => {
      render(
        <CollapsibleSection name="test" title="Test Section" isLoading={false}>
          <div>Visible Content</div>
        </CollapsibleSection>
      );

      expect(screen.getByText('Visible Content')).toBeInTheDocument();
      expect(document.querySelector('.animate-pulse')).not.toBeInTheDocument();
    });
  });

  describe('toggle behavior', () => {
    it('calls toggleSection when header is clicked', async () => {
      const user = userEvent.setup();

      render(
        <CollapsibleSection name="test-section" title="Test Section">
          <div>Content</div>
        </CollapsibleSection>
      );

      const header = screen.getByRole('button');
      await user.click(header);

      expect(mockToggleSection).toHaveBeenCalledWith('test-section');
    });

    it('calls onToggle callback with new state when toggling to closed', async () => {
      const user = userEvent.setup();
      const onToggle = vi.fn();
      mockIsSectionOpen.mockReturnValue(true);

      render(
        <CollapsibleSection name="test" title="Test Section" onToggle={onToggle}>
          <div>Content</div>
        </CollapsibleSection>
      );

      await user.click(screen.getByRole('button'));

      expect(onToggle).toHaveBeenCalledWith(false);
    });

    it('calls onToggle callback with new state when toggling to open', async () => {
      const user = userEvent.setup();
      const onToggle = vi.fn();
      mockIsSectionOpen.mockReturnValue(false);

      render(
        <CollapsibleSection name="test" title="Test Section" onToggle={onToggle}>
          <div>Content</div>
        </CollapsibleSection>
      );

      await user.click(screen.getByRole('button'));

      expect(onToggle).toHaveBeenCalledWith(true);
    });
  });

  describe('accessibility', () => {
    it('sets aria-expanded correctly when open', () => {
      mockIsSectionOpen.mockReturnValue(true);

      render(
        <CollapsibleSection name="test" title="Test Section">
          <div>Content</div>
        </CollapsibleSection>
      );

      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-expanded', 'true');
    });

    it('sets aria-expanded correctly when closed', () => {
      mockIsSectionOpen.mockReturnValue(false);

      render(
        <CollapsibleSection name="test" title="Test Section">
          <div>Content</div>
        </CollapsibleSection>
      );

      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-expanded', 'false');
    });

    it('sets aria-controls to link header and content', () => {
      render(
        <CollapsibleSection name="test" title="Test Section">
          <div>Content</div>
        </CollapsibleSection>
      );

      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-controls', 'section-content-test');
    });

    it('content has matching id for aria-controls', () => {
      render(
        <CollapsibleSection name="my-section" title="Test Section">
          <div>Content</div>
        </CollapsibleSection>
      );

      const content = document.getElementById('section-content-my-section');
      expect(content).toBeInTheDocument();
    });
  });

  describe('default state', () => {
    it('passes defaultOpen to isSectionOpen', () => {
      render(
        <CollapsibleSection name="test" title="Test Section" defaultOpen={false}>
          <div>Content</div>
        </CollapsibleSection>
      );

      expect(mockIsSectionOpen).toHaveBeenCalledWith('test', false);
    });

    it('passes defaultOpen=true by default', () => {
      render(
        <CollapsibleSection name="test" title="Test Section">
          <div>Content</div>
        </CollapsibleSection>
      );

      expect(mockIsSectionOpen).toHaveBeenCalledWith('test', true);
    });
  });

  describe('styling', () => {
    it('applies custom className', () => {
      const { container } = render(
        <CollapsibleSection name="test" title="Test Section" className="custom-class">
          <div>Content</div>
        </CollapsibleSection>
      );

      expect(container.firstChild).toHaveClass('custom-class');
    });

    it('has rotating chevron when open', () => {
      mockIsSectionOpen.mockReturnValue(true);

      render(
        <CollapsibleSection name="test" title="Test Section">
          <div>Content</div>
        </CollapsibleSection>
      );

      // Find the chevron icon (it's the last SVG in the button)
      const button = screen.getByRole('button');
      const svgs = button.querySelectorAll('svg');
      const chevron = svgs[svgs.length - 1];
      expect(chevron).toHaveClass('rotate-180');
    });

    it('has non-rotating chevron when closed', () => {
      mockIsSectionOpen.mockReturnValue(false);

      render(
        <CollapsibleSection name="test" title="Test Section">
          <div>Content</div>
        </CollapsibleSection>
      );

      const button = screen.getByRole('button');
      const svgs = button.querySelectorAll('svg');
      const chevron = svgs[svgs.length - 1];
      expect(chevron).not.toHaveClass('rotate-180');
    });
  });
});
