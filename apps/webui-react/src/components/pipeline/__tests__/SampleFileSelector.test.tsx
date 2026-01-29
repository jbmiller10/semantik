import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import { SampleFileSelector } from '../SampleFileSelector';

describe('SampleFileSelector', () => {
  const mockOnFileSelect = vi.fn();
  const mockOnClear = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  const createMockFile = (name: string, size: number, type: string = 'text/plain'): File => {
    const file = new File(['x'.repeat(size)], name, { type });
    // Mock the size property since File constructor doesn't set it correctly for tests
    Object.defineProperty(file, 'size', { value: size });
    return file;
  };

  describe('formatFileSize', () => {
    it('formats bytes (< 1KB)', () => {
      const file = createMockFile('small.txt', 512);

      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={file}
          onClear={mockOnClear}
        />
      );

      expect(screen.getByText('512 B')).toBeInTheDocument();
    });

    it('formats kilobytes (< 1MB)', () => {
      const file = createMockFile('medium.txt', 1536); // 1.5 KB

      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={file}
          onClear={mockOnClear}
        />
      );

      expect(screen.getByText('1.5 KB')).toBeInTheDocument();
    });

    it('formats megabytes (>= 1MB)', () => {
      const file = createMockFile('large.txt', 2 * 1024 * 1024); // 2 MB

      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={file}
          onClear={mockOnClear}
        />
      );

      expect(screen.getByText('2.0 MB')).toBeInTheDocument();
    });
  });

  describe('drop zone (no file selected)', () => {
    it('renders drop zone when no file selected', () => {
      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={null}
          onClear={mockOnClear}
        />
      );

      expect(screen.getByText('Click or drag a file to test routing')).toBeInTheDocument();
      expect(screen.getByText('Max 10MB')).toBeInTheDocument();
    });

    it('shows dragging state on dragover', () => {
      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={null}
          onClear={mockOnClear}
        />
      );

      const dropZone = screen.getByText('Click or drag a file to test routing').closest('div');
      expect(dropZone).toBeInTheDocument();

      fireEvent.dragOver(dropZone!);

      expect(screen.getByText('Drop file here')).toBeInTheDocument();
    });

    it('resets dragging state on dragleave', () => {
      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={null}
          onClear={mockOnClear}
        />
      );

      const dropZone = screen.getByText('Click or drag a file to test routing').closest('div');

      // Enter drag state
      fireEvent.dragOver(dropZone!);
      expect(screen.getByText('Drop file here')).toBeInTheDocument();

      // Leave drag state
      fireEvent.dragLeave(dropZone!);
      expect(screen.getByText('Click or drag a file to test routing')).toBeInTheDocument();
    });

    it('handles file drop', () => {
      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={null}
          onClear={mockOnClear}
        />
      );

      const dropZone = screen.getByText('Click or drag a file to test routing').closest('div');
      const file = createMockFile('test.txt', 100);

      const dropEvent = {
        preventDefault: vi.fn(),
        stopPropagation: vi.fn(),
        dataTransfer: {
          files: [file],
        },
      };

      fireEvent.drop(dropZone!, dropEvent);

      expect(mockOnFileSelect).toHaveBeenCalledWith(file);
    });

    it('triggers file input click on zone click', async () => {
      const user = userEvent.setup();

      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={null}
          onClear={mockOnClear}
        />
      );

      const dropZone = screen.getByText('Click or drag a file to test routing').closest('div');
      const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;

      // Mock click on file input
      const clickSpy = vi.spyOn(fileInput, 'click');

      await user.click(dropZone!);

      expect(clickSpy).toHaveBeenCalled();
    });

    it('handles file selection via input', () => {
      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={null}
          onClear={mockOnClear}
        />
      );

      const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
      const file = createMockFile('test.txt', 100);

      // Simulate file selection
      Object.defineProperty(fileInput, 'files', { value: [file] });
      fireEvent.change(fileInput);

      expect(mockOnFileSelect).toHaveBeenCalledWith(file);
    });
  });

  describe('disabled state', () => {
    it('does not trigger file selection when disabled', () => {
      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={null}
          onClear={mockOnClear}
          disabled={true}
        />
      );

      const dropZone = screen.getByText('Click or drag a file to test routing').closest('div');
      const file = createMockFile('test.txt', 100);

      const dropEvent = {
        preventDefault: vi.fn(),
        stopPropagation: vi.fn(),
        dataTransfer: {
          files: [file],
        },
      };

      fireEvent.drop(dropZone!, dropEvent);

      expect(mockOnFileSelect).not.toHaveBeenCalled();
    });

    it('applies disabled styling', () => {
      const { container } = render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={null}
          onClear={mockOnClear}
          disabled={true}
        />
      );

      // Get the outermost drop zone div which has the styling
      const dropZone = container.querySelector('.cursor-not-allowed');
      expect(dropZone).toBeInTheDocument();
    });
  });

  describe('loading state', () => {
    it('shows spinner when loading', () => {
      const file = createMockFile('test.txt', 100);

      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={file}
          onClear={mockOnClear}
          isLoading={true}
        />
      );

      // Check for spinner (Loader2 component with animate-spin)
      const spinner = document.querySelector('.animate-spin');
      expect(spinner).toBeInTheDocument();
    });

    it('does not show clear button when loading', () => {
      const file = createMockFile('test.txt', 100);

      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={file}
          onClear={mockOnClear}
          isLoading={true}
        />
      );

      // Clear button should not be visible when loading
      const clearButton = screen.queryByTitle('Clear file');
      expect(clearButton).not.toBeInTheDocument();
    });

    it('does not handle drops when loading', () => {
      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={null}
          onClear={mockOnClear}
          isLoading={true}
        />
      );

      const dropZone = screen.getByText('Click or drag a file to test routing').closest('div');
      const file = createMockFile('test.txt', 100);

      const dropEvent = {
        preventDefault: vi.fn(),
        stopPropagation: vi.fn(),
        dataTransfer: {
          files: [file],
        },
      };

      fireEvent.drop(dropZone!, dropEvent);

      expect(mockOnFileSelect).not.toHaveBeenCalled();
    });
  });

  describe('selected file state', () => {
    it('shows file name', () => {
      const file = createMockFile('my-document.pdf', 1024);

      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={file}
          onClear={mockOnClear}
        />
      );

      expect(screen.getByText('my-document.pdf')).toBeInTheDocument();
    });

    it('shows file size', () => {
      const file = createMockFile('test.txt', 2048);

      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={file}
          onClear={mockOnClear}
        />
      );

      expect(screen.getByText('2.0 KB')).toBeInTheDocument();
    });

    it('shows clear button', () => {
      const file = createMockFile('test.txt', 100);

      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={file}
          onClear={mockOnClear}
        />
      );

      expect(screen.getByTitle('Clear file')).toBeInTheDocument();
    });

    it('clear button calls onClear', async () => {
      const user = userEvent.setup();
      const file = createMockFile('test.txt', 100);

      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={file}
          onClear={mockOnClear}
        />
      );

      const clearButton = screen.getByTitle('Clear file');
      await user.click(clearButton);

      expect(mockOnClear).toHaveBeenCalled();
    });

    it('clear button stops event propagation', async () => {
      const file = createMockFile('test.txt', 100);

      render(
        <SampleFileSelector
          onFileSelect={mockOnFileSelect}
          selectedFile={file}
          onClear={mockOnClear}
        />
      );

      const clearButton = screen.getByTitle('Clear file');

      const clickEvent = new MouseEvent('click', { bubbles: true });
      const stopPropagationSpy = vi.spyOn(clickEvent, 'stopPropagation');

      fireEvent(clearButton, clickEvent);

      expect(stopPropagationSpy).toHaveBeenCalled();
    });
  });
});
