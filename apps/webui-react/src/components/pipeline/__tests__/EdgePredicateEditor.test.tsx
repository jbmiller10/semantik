import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import { EdgePredicateEditor } from '../EdgePredicateEditor';
import type { PipelineEdge } from '@/types/pipeline';

describe('EdgePredicateEditor', () => {
  const mockEdge: PipelineEdge = {
    from_node: '_source',
    to_node: 'parser1',
    when: { 'metadata.source.mime_type': 'application/pdf' },
  };

  it('renders edge info', () => {
    render(
      <EdgePredicateEditor
        edge={mockEdge}
        fromNodeLabel="Source"
        toNodeLabel="PDF Parser"
        onChange={vi.fn()}
      />
    );

    // Use exact text matches to avoid matching dropdown options
    expect(screen.getByText('Source')).toBeInTheDocument();
    expect(screen.getByText('PDF Parser')).toBeInTheDocument();
  });

  it('renders predicate field selector', () => {
    render(
      <EdgePredicateEditor
        edge={mockEdge}
        fromNodeLabel="Source"
        toNodeLabel="PDF Parser"
        onChange={vi.fn()}
      />
    );

    expect(screen.getByText(/MIME Type/i)).toBeInTheDocument();
  });

  it('renders predicate value', () => {
    render(
      <EdgePredicateEditor
        edge={mockEdge}
        fromNodeLabel="Source"
        toNodeLabel="PDF Parser"
        onChange={vi.fn()}
      />
    );

    expect(screen.getByDisplayValue('application/pdf')).toBeInTheDocument();
  });

  it('shows catch-all indicator for null when', () => {
    const catchAllEdge: PipelineEdge = {
      from_node: '_source',
      to_node: 'parser1',
      when: null,
    };

    render(
      <EdgePredicateEditor
        edge={catchAllEdge}
        fromNodeLabel="Source"
        toNodeLabel="Text Parser"
        onChange={vi.fn()}
      />
    );

    expect(screen.getByText(/catch-all/i)).toBeInTheDocument();
  });

  it('calls onChange when predicate field changes', async () => {
    const user = userEvent.setup();
    const handleChange = vi.fn();

    render(
      <EdgePredicateEditor
        edge={mockEdge}
        fromNodeLabel="Source"
        toNodeLabel="PDF Parser"
        onChange={handleChange}
      />
    );

    const fieldSelect = screen.getByLabelText(/field/i);
    await user.selectOptions(fieldSelect, 'metadata.source.extension');

    expect(handleChange).toHaveBeenCalled();
  });

  it('calls onChange when predicate value changes', async () => {
    const user = userEvent.setup();
    const handleChange = vi.fn();

    render(
      <EdgePredicateEditor
        edge={mockEdge}
        fromNodeLabel="Source"
        toNodeLabel="PDF Parser"
        onChange={handleChange}
      />
    );

    const valueInput = screen.getByDisplayValue('application/pdf');
    await user.clear(valueInput);
    await user.type(valueInput, 'text/plain');

    expect(handleChange).toHaveBeenCalled();
  });

  it('can toggle between predicate and catch-all', async () => {
    const user = userEvent.setup();
    const handleChange = vi.fn();

    render(
      <EdgePredicateEditor
        edge={mockEdge}
        fromNodeLabel="Source"
        toNodeLabel="PDF Parser"
        onChange={handleChange}
      />
    );

    const catchAllToggle = screen.getByRole('checkbox', { name: /catch-all/i });
    await user.click(catchAllToggle);

    expect(handleChange).toHaveBeenCalledWith({
      ...mockEdge,
      when: null,
    });
  });

  it('handles array values in predicates', () => {
    const arrayEdge: PipelineEdge = {
      from_node: '_source',
      to_node: 'parser1',
      when: { 'metadata.source.extension': ['.md', '.txt'] },
    };

    render(
      <EdgePredicateEditor
        edge={arrayEdge}
        fromNodeLabel="Source"
        toNodeLabel="Text Parser"
        onChange={vi.fn()}
      />
    );

    expect(screen.getByDisplayValue('.md, .txt')).toBeInTheDocument();
  });

  it('disables inputs when readOnly', () => {
    render(
      <EdgePredicateEditor
        edge={mockEdge}
        fromNodeLabel="Source"
        toNodeLabel="PDF Parser"
        onChange={vi.fn()}
        readOnly={true}
      />
    );

    const inputs = screen.getAllByRole('textbox');
    inputs.forEach((input) => {
      expect(input).toBeDisabled();
    });
  });
});
