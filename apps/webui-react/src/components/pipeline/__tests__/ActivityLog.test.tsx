// apps/webui-react/src/components/pipeline/__tests__/ActivityLog.test.tsx
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ActivityLog } from '../ActivityLog';

describe('ActivityLog', () => {
  const mockActivities = [
    { message: 'Starting analysis', timestamp: '2026-01-25T10:00:00Z' },
    { message: 'Found 50 files', timestamp: '2026-01-25T10:00:05Z' },
    { message: 'Processing complete', timestamp: '2026-01-25T10:00:10Z' },
  ];

  it('renders activity entries', () => {
    render(<ActivityLog activities={mockActivities} />);

    expect(screen.getByText('Starting analysis')).toBeInTheDocument();
    expect(screen.getByText('Found 50 files')).toBeInTheDocument();
    expect(screen.getByText('Processing complete')).toBeInTheDocument();
  });

  it('renders empty state when no activities', () => {
    render(<ActivityLog activities={[]} />);

    expect(screen.getByText('No activity yet')).toBeInTheDocument();
  });

  it('displays relative timestamps', () => {
    render(<ActivityLog activities={mockActivities} />);

    // Should have timestamp elements
    const timeElements = screen.getAllByRole('time');
    expect(timeElements.length).toBe(3);
  });
});
