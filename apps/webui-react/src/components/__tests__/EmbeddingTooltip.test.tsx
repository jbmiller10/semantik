import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  TooltipContent,
  EmbeddingTooltipComponent,
  EmbeddingTooltipAdapter
} from '../EmbeddingTooltip';
import type { DataPoint } from 'embedding-atlas/react';
import type { ProjectionTooltipState } from '../../hooks/useProjectionTooltip';

// Mock DataPoint since we just need an object reference for most tests
const mockDataPoint = { x: 100, y: 100 } as unknown as DataPoint;

describe('TooltipContent', () => {
  const defaultProps = {
    tooltip: mockDataPoint,
    getTooltipIndex: vi.fn(),
    ids: new Int32Array([1, 2, 3]),
    tooltipState: {
      status: 'idle',
      position: null,
      metadata: null,
    } as ProjectionTooltipState,
  };

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('renders nothing when tooltip is null', () => {
    const { container } = render(
      <TooltipContent
        {...defaultProps}
        tooltip={null}
      />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('renders nothing when index is null', () => {
    const getTooltipIndex = vi.fn().mockReturnValue(null);
    const { container } = render(
      <TooltipContent
        {...defaultProps}
        getTooltipIndex={getTooltipIndex}
      />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('renders nothing when status is idle and no metadata', () => {
    const getTooltipIndex = vi.fn().mockReturnValue(0);
    const { container } = render(
      <TooltipContent
        {...defaultProps}
        getTooltipIndex={getTooltipIndex}
        tooltipState={{ ...defaultProps.tooltipState, status: 'idle' }}
      />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('renders loading state', () => {
    const getTooltipIndex = vi.fn().mockReturnValue(0);
    render(
      <TooltipContent
        {...defaultProps}
        getTooltipIndex={getTooltipIndex}
        tooltipState={{ ...defaultProps.tooltipState, status: 'loading' }}
      />
    );
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('renders error state', () => {
    const getTooltipIndex = vi.fn().mockReturnValue(0);
    render(
      <TooltipContent
        {...defaultProps}
        getTooltipIndex={getTooltipIndex}
        tooltipState={{ ...defaultProps.tooltipState, status: 'error' }}
      />
    );
    expect(screen.getByText('No metadata available')).toBeInTheDocument();
  });

  it('renders metadata when available', () => {
    const getTooltipIndex = vi.fn().mockReturnValue(0);
    const metadata = {
      selectedId: 1,
      originalId: 'orig-1',
      documentLabel: 'doc-label',
      chunkIndex: 5,
      contentPreview: 'some content preview',
      source: 'network' as const,
    };
    
    render(
      <TooltipContent
        {...defaultProps}
        getTooltipIndex={getTooltipIndex}
        tooltipState={{
          status: 'success',
          position: null,
          metadata,
        }}
      />
    );

    expect(screen.getByText('Point ID orig-1')).toBeInTheDocument();
    expect(screen.getByText('doc-label')).toBeInTheDocument();
    expect(screen.getByText('Chunk #5')).toBeInTheDocument();
    expect(screen.getByText('some content preview')).toBeInTheDocument();
  });

  it('renders metadata even if status is not success (e.g. loading previous data)', () => {
     const getTooltipIndex = vi.fn().mockReturnValue(0);
     const metadata = {
       selectedId: 1,
       source: 'cache' as const,
       contentPreview: 'cached preview',
     };
     
     render(
       <TooltipContent
         {...defaultProps}
         getTooltipIndex={getTooltipIndex}
         tooltipState={{
           status: 'loading',
           position: null,
           metadata,
         }}
       />
     );
 
     expect(screen.getByText('cached preview')).toBeInTheDocument();
     expect(screen.queryByText('Loading...')).not.toBeInTheDocument();
   });

   it('renders "No document metadata available" when preview is missing but originalId exists', () => {
    const getTooltipIndex = vi.fn().mockReturnValue(0);
    const metadata = {
      selectedId: 1,
      originalId: 'orig-1',
      source: 'network' as const,
      contentPreview: null,
    };
    
    render(
      <TooltipContent
        {...defaultProps}
        getTooltipIndex={getTooltipIndex}
        tooltipState={{
          status: 'success',
          position: null,
          metadata,
        }}
      />
    );

    expect(screen.getByText('No document metadata available for this point')).toBeInTheDocument();
  });

  it('renders "No metadata available" when preview and originalId are missing', () => {
    const getTooltipIndex = vi.fn().mockReturnValue(0);
    const metadata = {
      selectedId: 1,
      source: 'network' as const,
      contentPreview: null,
      originalId: null,
    };
    
    render(
      <TooltipContent
        {...defaultProps}
        getTooltipIndex={getTooltipIndex}
        tooltipState={{
          status: 'success',
          position: null,
          metadata,
        }}
      />
    );

    expect(screen.getByText('No metadata available')).toBeInTheDocument();
  });

  it('renders document ID when label is missing but ID exists', () => {
    const getTooltipIndex = vi.fn().mockReturnValue(0);
    const metadata = {
      selectedId: 1,
      documentId: 'doc-123',
      documentLabel: null,
      source: 'network' as const,
      contentPreview: 'preview',
    };
    
    render(
      <TooltipContent
        {...defaultProps}
        getTooltipIndex={getTooltipIndex}
        tooltipState={{
          status: 'success',
          position: null,
          metadata,
        }}
      />
    );

    expect(screen.getByText('Document doc-123')).toBeInTheDocument();
  });

  it('renders nothing if status is success but metadata is null (defensive check)', () => {
    const getTooltipIndex = vi.fn().mockReturnValue(0);
    const { container } = render(
      <TooltipContent
        {...defaultProps}
        getTooltipIndex={getTooltipIndex}
        tooltipState={{
          status: 'success',
          position: null,
          metadata: null,
        }}
      />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('handles ids being undefined', () => {
    const getTooltipIndex = vi.fn().mockReturnValue(0);
    const { container } = render(
      <TooltipContent
        {...defaultProps}
        getTooltipIndex={getTooltipIndex}
        ids={undefined}
        tooltipState={{ ...defaultProps.tooltipState, status: 'loading' }}
      />
    );
    // Should probably render loading or null depending on logic.
    // If selectedId is null, metadata lookup might fail or be null.
    // status is loading, metadata is null.
    // It should render loading state.
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('handles index out of bounds', () => {
    const getTooltipIndex = vi.fn().mockReturnValue(100); // Out of bounds
    const { container } = render(
      <TooltipContent
        {...defaultProps}
        getTooltipIndex={getTooltipIndex}
        tooltipState={{ ...defaultProps.tooltipState, status: 'loading' }}
      />
    );
    // selectedId becomes null.
    // status loading, metadata null -> loading state.
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('handles negative index', () => {
    const getTooltipIndex = vi.fn().mockReturnValue(-1);
    const { container } = render(
      <TooltipContent
        {...defaultProps}
        getTooltipIndex={getTooltipIndex}
        tooltipState={{ ...defaultProps.tooltipState, status: 'loading' }}
      />
    );
    // selectedId becomes null.
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });
});

describe('EmbeddingTooltipComponent', () => {
  it('creates a root and renders on init', async () => {
    const target = document.createElement('div');
    const props = {
      tooltip: mockDataPoint,
      getTooltipIndex: () => 0,
      ids: new Int32Array([1]),
      tooltipState: { status: 'loading', position: null, metadata: null } as ProjectionTooltipState,
    };

    const component = new EmbeddingTooltipComponent(target, props);
    
    await waitFor(() => {
      expect(target.innerHTML).toContain('Loading...');
    });
    
    component.destroy();
  });

  it('updates props', async () => {
    const target = document.createElement('div');
    const props = {
      tooltip: mockDataPoint,
      getTooltipIndex: () => 0,
      ids: new Int32Array([1]),
      tooltipState: { status: 'loading', position: null, metadata: null } as ProjectionTooltipState,
    };

    const component = new EmbeddingTooltipComponent(target, props);
    
    await waitFor(() => {
      expect(target.innerHTML).toContain('Loading...');
    });

    component.update({
        ...props,
        tooltipState: { status: 'error', position: null, metadata: null } as ProjectionTooltipState,
    });

    await waitFor(() => {
         expect(target.innerHTML).toContain('No metadata available');
    });
    
    component.destroy();
  });
});

describe('EmbeddingTooltipAdapter', () => {
    it('initializes and delegates calls', async () => {
        const target = document.createElement('div');
        const props = {
          tooltip: mockDataPoint,
          getTooltipIndex: () => 0,
          ids: new Int32Array([1]),
          tooltipState: { status: 'loading', position: null, metadata: null } as ProjectionTooltipState,
        };

        const adapter = new EmbeddingTooltipAdapter(target, props);
        
        await waitFor(() => {
          expect(target.innerHTML).toContain('Loading...');
        });
        
        adapter.update({
            ...props,
            tooltipState: { status: 'error', position: null, metadata: null } as ProjectionTooltipState
        });

        await waitFor(() => {
            expect(target.innerHTML).toContain('No metadata available');
       });

        adapter.destroy();
    });
});
