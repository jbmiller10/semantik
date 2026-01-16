import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { SparseIndexConfigModal } from '../SparseIndexConfigModal';
import type { EnableSparseIndexRequest } from '../../../types/sparse-index';

describe('SparseIndexConfigModal', () => {
  const user = userEvent.setup();
  const mockOnClose = vi.fn();
  const mockOnSubmit = vi.fn();

  const defaultProps = {
    isOpen: true,
    onClose: mockOnClose,
    onSubmit: mockOnSubmit,
    isSubmitting: false,
    collectionName: 'Test Collection',
    documentCount: 100,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('returns null when isOpen is false', () => {
      const { container } = render(
        <SparseIndexConfigModal {...defaultProps} isOpen={false} />
      );
      expect(container.firstChild).toBeNull();
    });

    it('renders modal when isOpen is true', () => {
      render(<SparseIndexConfigModal {...defaultProps} />);
      // Use heading role to be specific about which "Enable Sparse Indexing" we want
      expect(screen.getByRole('heading', { name: 'Enable Sparse Indexing' })).toBeInTheDocument();
    });

    it('displays collection name in header', () => {
      render(<SparseIndexConfigModal {...defaultProps} />);
      expect(screen.getByText('Test Collection')).toBeInTheDocument();
    });

    it('shows document count in reindex option text', () => {
      render(<SparseIndexConfigModal {...defaultProps} documentCount={1234} />);
      expect(screen.getByText(/1,234 documents/)).toBeInTheDocument();
    });
  });

  describe('plugin selection', () => {
    it('renders BM25 and SPLADE plugin options', () => {
      render(<SparseIndexConfigModal {...defaultProps} />);
      expect(screen.getByText('BM25 (Statistical)')).toBeInTheDocument();
      expect(screen.getByText('SPLADE (Neural)')).toBeInTheDocument();
    });

    it('shows BM25 as selected by default', () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      // BM25 option should have the selected styling indicator
      const bm25Button = screen.getByText('BM25 (Statistical)').closest('button');
      expect(bm25Button).toHaveClass('border-gray-400');
    });

    it('shows "CPU only" for BM25 and "GPU required" for SPLADE', () => {
      render(<SparseIndexConfigModal {...defaultProps} />);
      expect(screen.getByText('CPU only')).toBeInTheDocument();
      expect(screen.getByText('GPU required')).toBeInTheDocument();
    });

    it('updates plugin description when selection changes', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      // Initial BM25 description
      expect(
        screen.getByText(/Traditional keyword matching with TF-IDF scoring/)
      ).toBeInTheDocument();

      // Click SPLADE
      await user.click(screen.getByText('SPLADE (Neural)'));

      // SPLADE description
      expect(
        screen.getByText(/Neural sparse representations with semantic expansion/)
      ).toBeInTheDocument();
    });

    it('switches selection when clicking different plugin', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      // Click SPLADE
      await user.click(screen.getByText('SPLADE (Neural)'));

      // SPLADE option should now have selected styling
      const spladeButton = screen.getByText('SPLADE (Neural)').closest('button');
      expect(spladeButton).toHaveClass('border-gray-400');
    });
  });

  describe('BM25 advanced parameters', () => {
    it('hides advanced section by default', () => {
      render(<SparseIndexConfigModal {...defaultProps} />);
      expect(screen.queryByText('k1 (Term Frequency Saturation)')).not.toBeInTheDocument();
    });

    it('shows advanced section when toggle is clicked', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      await user.click(screen.getByText('Advanced BM25 Parameters'));

      expect(screen.getByText('k1 (Term Frequency Saturation)')).toBeInTheDocument();
      expect(screen.getByText('b (Length Normalization)')).toBeInTheDocument();
    });

    it('displays k1 slider with correct range values', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      await user.click(screen.getByText('Advanced BM25 Parameters'));

      expect(screen.getByText('Low (0.5)')).toBeInTheDocument();
      expect(screen.getByText('High (3)')).toBeInTheDocument();
    });

    it('displays b slider with correct range values', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      await user.click(screen.getByText('Advanced BM25 Parameters'));

      expect(screen.getByText('None (0)')).toBeInTheDocument();
      expect(screen.getByText('Full (1)')).toBeInTheDocument();
    });

    it('updates k1 value when slider changes', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      await user.click(screen.getByText('Advanced BM25 Parameters'));

      // Find the k1 slider by its container and input type
      const sliders = screen.getAllByRole('slider');
      const k1Slider = sliders[0]; // First slider is k1
      fireEvent.change(k1Slider, { target: { value: '2.5' } });

      expect(screen.getByText('2.5')).toBeInTheDocument();
    });

    it('updates b value when slider changes', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      await user.click(screen.getByText('Advanced BM25 Parameters'));

      const sliders = screen.getAllByRole('slider');
      const bSlider = sliders[1]; // Second slider is b
      fireEvent.change(bSlider, { target: { value: '0.5' } });

      expect(screen.getByText('0.5')).toBeInTheDocument();
    });

    it('resets parameters to defaults when Reset clicked', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      await user.click(screen.getByText('Advanced BM25 Parameters'));

      // Change values first
      const sliders = screen.getAllByRole('slider');
      fireEvent.change(sliders[0], { target: { value: '2.5' } });
      fireEvent.change(sliders[1], { target: { value: '0.5' } });

      // Reset
      await user.click(screen.getByText('Reset to defaults'));

      // Should show default values
      expect(screen.getByText('1.5')).toBeInTheDocument();
      expect(screen.getByText('0.75')).toBeInTheDocument();
    });

    it('hides parameters section when SPLADE is selected', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      // First show advanced params for BM25
      await user.click(screen.getByText('Advanced BM25 Parameters'));
      expect(screen.getByText('k1 (Term Frequency Saturation)')).toBeInTheDocument();

      // Select SPLADE
      await user.click(screen.getByText('SPLADE (Neural)'));

      // Advanced BM25 toggle should not be visible
      expect(screen.queryByText('Advanced BM25 Parameters')).not.toBeInTheDocument();
    });
  });

  describe('reindex option', () => {
    it('renders checkbox checked by default', () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      const checkbox = screen.getByRole('checkbox', { name: /index existing documents/i });
      expect(checkbox).toBeChecked();
    });

    it('toggles reindexExisting state when checkbox clicked', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      const checkbox = screen.getByRole('checkbox', { name: /index existing documents/i });
      expect(checkbox).toBeChecked();

      await user.click(checkbox);
      expect(checkbox).not.toBeChecked();

      await user.click(checkbox);
      expect(checkbox).toBeChecked();
    });

    it('displays document count in label', () => {
      render(<SparseIndexConfigModal {...defaultProps} documentCount={567} />);

      expect(
        screen.getByText(/567 documents/)
      ).toBeInTheDocument();
    });
  });

  describe('warning banner', () => {
    it('shows CPU info when BM25 is selected', () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      expect(
        screen.getByText(/BM25 indexing is CPU-based and typically completes quickly/)
      ).toBeInTheDocument();
    });

    it('shows GPU warning when SPLADE is selected', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      await user.click(screen.getByText('SPLADE (Neural)'));

      expect(
        screen.getByText(/SPLADE indexing requires GPU/)
      ).toBeInTheDocument();
    });
  });

  describe('form submission', () => {
    it('calls onSubmit with correct config for BM25', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      await user.click(screen.getByRole('button', { name: /enable sparse indexing/i }));

      expect(mockOnSubmit).toHaveBeenCalledWith({
        plugin_id: 'bm25-local',
        reindex_existing: true,
        model_config_data: { k1: 1.5, b: 0.75 },
      } satisfies EnableSparseIndexRequest);
    });

    it('calls onSubmit with correct config for SPLADE', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      await user.click(screen.getByText('SPLADE (Neural)'));
      await user.click(screen.getByRole('button', { name: /enable sparse indexing/i }));

      expect(mockOnSubmit).toHaveBeenCalledWith({
        plugin_id: 'splade-local',
        reindex_existing: true,
      } satisfies EnableSparseIndexRequest);
    });

    it('includes reindex_existing in config based on checkbox', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      // Uncheck reindex
      await user.click(screen.getByRole('checkbox', { name: /index existing documents/i }));
      await user.click(screen.getByRole('button', { name: /enable sparse indexing/i }));

      expect(mockOnSubmit).toHaveBeenCalledWith(
        expect.objectContaining({
          reindex_existing: false,
        })
      );
    });

    it('includes custom BM25 params when advanced settings are modified', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      await user.click(screen.getByText('Advanced BM25 Parameters'));

      const sliders = screen.getAllByRole('slider');
      fireEvent.change(sliders[0], { target: { value: '2.0' } });
      fireEvent.change(sliders[1], { target: { value: '0.5' } });

      await user.click(screen.getByRole('button', { name: /enable sparse indexing/i }));

      expect(mockOnSubmit).toHaveBeenCalledWith({
        plugin_id: 'bm25-local',
        reindex_existing: true,
        model_config_data: { k1: 2.0, b: 0.5 },
      });
    });

    it('prevents default form submission', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      const form = document.querySelector('form');
      const submitEvent = new Event('submit', { bubbles: true, cancelable: true });
      const preventDefaultSpy = vi.spyOn(submitEvent, 'preventDefault');

      form?.dispatchEvent(submitEvent);

      expect(preventDefaultSpy).toHaveBeenCalled();
    });
  });

  describe('modal controls', () => {
    it('calls onClose when X button clicked', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      // Find the X button (it's the one with just the X icon)
      const closeButtons = screen.getAllByRole('button');
      const xButton = closeButtons.find((btn) =>
        btn.querySelector('.lucide-x')
      );

      expect(xButton).toBeDefined();
      await user.click(xButton!);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('calls onClose when Cancel button clicked', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      await user.click(screen.getByRole('button', { name: /cancel/i }));

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('calls onClose when backdrop clicked', async () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      // Click the backdrop
      const backdrop = document.querySelector('.bg-black\\/50');
      expect(backdrop).toBeDefined();
      await user.click(backdrop!);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('disables Cancel button when isSubmitting', () => {
      render(<SparseIndexConfigModal {...defaultProps} isSubmitting={true} />);

      const cancelButton = screen.getByRole('button', { name: /cancel/i });
      expect(cancelButton).toBeDisabled();
    });

    it('disables submit button when isSubmitting', () => {
      render(<SparseIndexConfigModal {...defaultProps} isSubmitting={true} />);

      const submitButton = screen.getByRole('button', { name: /enabling/i });
      expect(submitButton).toBeDisabled();
    });

    it('shows Loader2 spinner when isSubmitting', () => {
      render(<SparseIndexConfigModal {...defaultProps} isSubmitting={true} />);

      expect(screen.getByText('Enabling...')).toBeInTheDocument();
      const spinner = document.querySelector('.animate-spin');
      expect(spinner).toBeInTheDocument();
    });

    it('shows "Enabling..." text when isSubmitting', () => {
      render(<SparseIndexConfigModal {...defaultProps} isSubmitting={true} />);

      expect(screen.getByText('Enabling...')).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('has proper checkbox labeling', () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      const checkbox = screen.getByRole('checkbox');
      expect(checkbox).toHaveAttribute('id', 'reindex-existing');

      const label = document.querySelector('label[for="reindex-existing"]');
      expect(label).toBeInTheDocument();
    });

    it('plugin buttons have correct type attribute', () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      const pluginButtons = screen
        .getAllByRole('button')
        .filter((btn) => btn.getAttribute('type') === 'button');

      // Should have at least 2 plugin buttons (BM25 and SPLADE)
      expect(pluginButtons.length).toBeGreaterThanOrEqual(2);
    });

    it('submit button has type submit', () => {
      render(<SparseIndexConfigModal {...defaultProps} />);

      const submitButton = screen.getByRole('button', { name: /enable sparse indexing/i });
      expect(submitButton).toHaveAttribute('type', 'submit');
    });
  });
});
