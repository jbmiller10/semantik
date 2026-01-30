/**
 * File selector component for route preview.
 * Provides drag-and-drop and click-to-upload functionality.
 */

import { useState, useRef, useCallback } from 'react';
import { Upload, File as FileIcon, X, Loader2 } from 'lucide-react';

interface SampleFileSelectorProps {
  /** Callback when a file is selected */
  onFileSelect: (file: File) => void;
  /** Currently selected file (if any) */
  selectedFile: File | null;
  /** Callback to clear the selected file */
  onClear: () => void;
  /** Whether file selection is disabled */
  disabled?: boolean;
  /** Whether a preview is in progress */
  isLoading?: boolean;
}

/**
 * Format file size for display.
 */
function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function SampleFileSelector({
  onFileSelect,
  selectedFile,
  onClear,
  disabled = false,
  isLoading = false,
}: SampleFileSelectorProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      if (disabled || isLoading) return;

      const files = Array.from(e.dataTransfer.files);
      if (files.length > 0) {
        onFileSelect(files[0]);
      }
    },
    [disabled, isLoading, onFileSelect]
  );

  const handleClick = useCallback(() => {
    if (!disabled && !isLoading) {
      fileInputRef.current?.click();
    }
  }, [disabled, isLoading]);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        onFileSelect(files[0]);
      }
      // Reset input so the same file can be selected again
      e.target.value = '';
    },
    [onFileSelect]
  );

  // Show selected file state
  if (selectedFile) {
    return (
      <div className="flex items-center gap-3 p-3 rounded-lg border border-[var(--border)] bg-[var(--bg-secondary)]">
        <FileIcon className="w-5 h-5 text-[var(--text-muted)] flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="text-sm text-[var(--text-primary)] truncate">{selectedFile.name}</p>
          <p className="text-xs text-[var(--text-muted)]">{formatFileSize(selectedFile.size)}</p>
        </div>
        {isLoading ? (
          <Loader2 className="w-4 h-4 text-[var(--text-muted)] animate-spin" />
        ) : (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onClear();
            }}
            className="p-1 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
            title="Clear file"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>
    );
  }

  // Show drop zone
  return (
    <div
      onClick={handleClick}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`
        flex flex-col items-center justify-center gap-2 p-4
        rounded-lg border-2 border-dashed transition-colors cursor-pointer
        ${
          isDragging
            ? 'border-[var(--text-primary)] bg-[var(--bg-tertiary)]'
            : 'border-[var(--border)] hover:border-[var(--text-muted)] hover:bg-[var(--bg-secondary)]'
        }
        ${disabled || isLoading ? 'opacity-50 cursor-not-allowed' : ''}
      `}
    >
      <input
        ref={fileInputRef}
        type="file"
        onChange={handleFileChange}
        className="hidden"
        disabled={disabled || isLoading}
      />
      <Upload className="w-6 h-6 text-[var(--text-muted)]" />
      <div className="text-center">
        <p className="text-sm text-[var(--text-secondary)]">
          {isDragging ? 'Drop file here' : 'Click or drag a file to test routing'}
        </p>
        <p className="text-xs text-[var(--text-muted)] mt-1">Max 10MB</p>
      </div>
    </div>
  );
}

export default SampleFileSelector;
