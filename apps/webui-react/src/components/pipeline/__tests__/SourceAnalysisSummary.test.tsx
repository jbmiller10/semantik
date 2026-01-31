import { describe, it, expect } from 'vitest';
import { render, screen } from '@/tests/utils/test-utils';
import { SourceAnalysisSummary } from '../SourceAnalysisSummary';
import type { SourceAnalysis } from '@/types/agent';

describe('SourceAnalysisSummary', () => {
  const mockAnalysis: SourceAnalysis = {
    total_files: 247,
    total_size_bytes: 52428800, // 50 MB
    file_types: {
      '.pdf': 180,
      '.md': 45,
      '.txt': 22,
    },
    sample_files: ['paper1.pdf', 'notes.md', 'readme.txt'],
    warnings: ['Some files may be password protected'],
  };

  it('renders total file count', () => {
    render(<SourceAnalysisSummary analysis={mockAnalysis} />);
    expect(screen.getByText('247')).toBeInTheDocument();
    // Check for the "Files" label specifically in the stats section
    expect(screen.getByText('Files')).toBeInTheDocument();
  });

  it('renders total size formatted', () => {
    render(<SourceAnalysisSummary analysis={mockAnalysis} />);
    expect(screen.getByText(/50.*MB/i)).toBeInTheDocument();
  });

  it('renders file type breakdown', () => {
    render(<SourceAnalysisSummary analysis={mockAnalysis} />);
    expect(screen.getByText('.pdf')).toBeInTheDocument();
    expect(screen.getByText('180')).toBeInTheDocument();
    expect(screen.getByText('.md')).toBeInTheDocument();
    expect(screen.getByText('45')).toBeInTheDocument();
  });

  it('renders warnings when present', () => {
    render(<SourceAnalysisSummary analysis={mockAnalysis} />);
    expect(screen.getByText(/password protected/i)).toBeInTheDocument();
  });

  it('renders empty state when no analysis', () => {
    render(<SourceAnalysisSummary analysis={null} />);
    expect(screen.getByText(/no source analysis/i)).toBeInTheDocument();
  });

  it('sorts file types by count descending', () => {
    render(<SourceAnalysisSummary analysis={mockAnalysis} />);
    const items = screen.getAllByTestId('file-type-item');
    // PDF should be first (180), then md (45), then txt (22)
    expect(items[0]).toHaveTextContent('.pdf');
    expect(items[1]).toHaveTextContent('.md');
    expect(items[2]).toHaveTextContent('.txt');
  });
});
