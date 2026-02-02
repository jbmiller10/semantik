import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import { EdgePredicateEditor } from '../EdgePredicateEditor';
import type { PipelineEdge, PipelineDAG } from '@/types/pipeline';

describe('EdgePredicateEditor', () => {
  const mockDag: PipelineDAG = {
    id: 'test-dag',
    version: '1.0',
    nodes: [
      { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
    ],
    edges: [],
  };

  const mockEdge: PipelineEdge = {
    from_node: '_source',
    to_node: 'parser1',
    when: { 'mime_type': 'application/pdf' },
  };

  it('renders edge info', () => {
    render(
      <EdgePredicateEditor
        edge={mockEdge}
        dag={mockDag}
        fromNodeLabel="Source"
        toNodeLabel="PDF Parser"
        onChange={vi.fn()}
      />
    );

    // Use exact text matches to avoid matching dropdown options
    expect(screen.getByText('Source')).toBeInTheDocument();
    expect(screen.getByText('PDF Parser')).toBeInTheDocument();
  });

  it('renders predicate field selector', async () => {
    render(
      <EdgePredicateEditor
        edge={mockEdge}
        dag={mockDag}
        fromNodeLabel="Source"
        toNodeLabel="PDF Parser"
        onChange={vi.fn()}
      />
    );

    // Wait for the field options to load
    await waitFor(() => {
      expect(screen.getByText(/MIME Type/i)).toBeInTheDocument();
    });
  });

  it('renders predicate value', () => {
    render(
      <EdgePredicateEditor
        edge={mockEdge}
        dag={mockDag}
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
        dag={mockDag}
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
        dag={mockDag}
        fromNodeLabel="Source"
        toNodeLabel="PDF Parser"
        onChange={handleChange}
      />
    );

    // Wait for the field selector to load (API call completes)
    const fieldSelect = await waitFor(() => {
      const el = document.getElementById('predicate-field');
      expect(el).toBeTruthy();
      return el as HTMLSelectElement;
    });
    await user.selectOptions(fieldSelect, 'extension');

    expect(handleChange).toHaveBeenCalled();
  });

  it('calls onChange when predicate value changes', async () => {
    const user = userEvent.setup();
    const handleChange = vi.fn();

    render(
      <EdgePredicateEditor
        edge={mockEdge}
        dag={mockDag}
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
        dag={mockDag}
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
      when: { 'extension': ['.md', '.txt'] },
    };

    render(
      <EdgePredicateEditor
        edge={arrayEdge}
        dag={mockDag}
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
        dag={mockDag}
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

  describe('Boolean field toggle', () => {
    it('renders boolean toggle for is_* fields', () => {
      const booleanEdge: PipelineEdge = {
        from_node: '_source',
        to_node: 'parser1',
        when: { 'metadata.detected.is_scanned_pdf': true },
      };

      render(
        <EdgePredicateEditor
          edge={booleanEdge}
          dag={mockDag}
          fromNodeLabel="Source"
          toNodeLabel="OCR Parser"
          onChange={vi.fn()}
        />
      );

      // Should show a select (combobox) for boolean value
      const booleanSelect = screen.getByRole('combobox', { name: /boolean value/i });
      expect(booleanSelect).toBeInTheDocument();
      expect(booleanSelect).toHaveValue('true');
    });

    it('renders boolean toggle for has_* fields', () => {
      const booleanEdge: PipelineEdge = {
        from_node: '_source',
        to_node: 'parser1',
        when: { 'metadata.parsed.has_tables': false },
      };

      render(
        <EdgePredicateEditor
          edge={booleanEdge}
          dag={mockDag}
          fromNodeLabel="Source"
          toNodeLabel="Parser"
          onChange={vi.fn()}
        />
      );

      const booleanSelect = screen.getByRole('combobox', { name: /boolean value/i });
      expect(booleanSelect).toBeInTheDocument();
      expect(booleanSelect).toHaveValue('false');
    });

    it('calls onChange with boolean value on toggle', async () => {
      const user = userEvent.setup();
      const handleChange = vi.fn();

      const booleanEdge: PipelineEdge = {
        from_node: '_source',
        to_node: 'parser1',
        when: { 'metadata.detected.is_code': true },
      };

      render(
        <EdgePredicateEditor
          edge={booleanEdge}
          dag={mockDag}
          fromNodeLabel="Source"
          toNodeLabel="Parser"
          onChange={handleChange}
        />
      );

      const booleanSelect = screen.getByRole('combobox', { name: /boolean value/i });
      await user.selectOptions(booleanSelect, 'false');

      expect(handleChange).toHaveBeenCalledWith({
        ...booleanEdge,
        when: { 'metadata.detected.is_code': false },
      });
    });
  });

  describe('Negation checkbox', () => {
    it('renders NOT checkbox unchecked by default', () => {
      render(
        <EdgePredicateEditor
          edge={mockEdge}
          dag={mockDag}
          fromNodeLabel="Source"
          toNodeLabel="PDF Parser"
          onChange={vi.fn()}
        />
      );

      const notCheckbox = screen.getByRole('checkbox', { name: /not/i });
      expect(notCheckbox).toBeInTheDocument();
      expect(notCheckbox).not.toBeChecked();
    });

    it('shows NOT checkbox checked for negated values', () => {
      const negatedEdge: PipelineEdge = {
        from_node: '_source',
        to_node: 'parser1',
        when: { 'mime_type': '!application/pdf' },
      };

      render(
        <EdgePredicateEditor
          edge={negatedEdge}
          dag={mockDag}
          fromNodeLabel="Source"
          toNodeLabel="Parser"
          onChange={vi.fn()}
        />
      );

      const notCheckbox = screen.getByRole('checkbox', { name: /not/i });
      expect(notCheckbox).toBeChecked();
      // Value should be displayed without the ! prefix
      expect(screen.getByDisplayValue('application/pdf')).toBeInTheDocument();
    });

    it('calls onChange with negated value when NOT is toggled on', async () => {
      const user = userEvent.setup();
      const handleChange = vi.fn();

      render(
        <EdgePredicateEditor
          edge={mockEdge}
          dag={mockDag}
          fromNodeLabel="Source"
          toNodeLabel="PDF Parser"
          onChange={handleChange}
        />
      );

      const notCheckbox = screen.getByRole('checkbox', { name: /not/i });
      await user.click(notCheckbox);

      expect(handleChange).toHaveBeenCalledWith({
        ...mockEdge,
        when: { 'mime_type': '!application/pdf' },
      });
    });

    it('shows NOT indicator in preview', () => {
      const negatedEdge: PipelineEdge = {
        from_node: '_source',
        to_node: 'parser1',
        when: { 'mime_type': '!application/pdf' },
      };

      render(
        <EdgePredicateEditor
          edge={negatedEdge}
          dag={mockDag}
          fromNodeLabel="Source"
          toNodeLabel="Parser"
          onChange={vi.fn()}
        />
      );

      // Preview section should show NOT indicator (styled red)
      const previewSection = screen.getByText(/Preview:/i).closest('div');
      expect(previewSection).toBeInTheDocument();
      // The red "NOT " text in the preview
      expect(screen.getByText('NOT', { selector: '.text-red-400' })).toBeInTheDocument();
    });

    it('works with boolean fields and negation', async () => {
      const user = userEvent.setup();
      const handleChange = vi.fn();

      const booleanEdge: PipelineEdge = {
        from_node: '_source',
        to_node: 'parser1',
        when: { 'metadata.detected.is_scanned_pdf': true },
      };

      render(
        <EdgePredicateEditor
          edge={booleanEdge}
          dag={mockDag}
          fromNodeLabel="Source"
          toNodeLabel="Parser"
          onChange={handleChange}
        />
      );

      const notCheckbox = screen.getByRole('checkbox', { name: /not/i });
      await user.click(notCheckbox);

      expect(handleChange).toHaveBeenCalledWith({
        ...booleanEdge,
        when: { 'metadata.detected.is_scanned_pdf': '!true' },
      });
    });
  });

  describe('sticky header', () => {
    it('renders sticky header showing edge flow', () => {
      render(
        <EdgePredicateEditor
          edge={mockEdge}
          dag={mockDag}
          fromNodeLabel="Source"
          toNodeLabel="parser1"
          onChange={vi.fn()}
        />
      );

      const stickyHeader = document.querySelector('.edge-header-sticky');
      expect(stickyHeader).toBeInTheDocument();
      expect(stickyHeader).toHaveClass('sticky');
      expect(stickyHeader).toHaveClass('top-0');
    });
  });

  describe('parallel toggle', () => {
    it('renders parallel toggle switch', () => {
      const edge: PipelineEdge = {
        from_node: '_source',
        to_node: 'parser1',
        when: null,
      };

      render(
        <EdgePredicateEditor
          edge={edge}
          dag={mockDag}
          fromNodeLabel="Source"
          toNodeLabel="Parser"
          onChange={vi.fn()}
        />
      );

      expect(screen.getByLabelText('Parallel edge')).toBeInTheDocument();
    });

    it('shows parallel toggle as unchecked by default', () => {
      const edge: PipelineEdge = {
        from_node: '_source',
        to_node: 'parser1',
        when: null,
      };

      render(
        <EdgePredicateEditor
          edge={edge}
          dag={mockDag}
          fromNodeLabel="Source"
          toNodeLabel="Parser"
          onChange={vi.fn()}
        />
      );

      expect(screen.getByLabelText('Parallel edge')).not.toBeChecked();
    });

    it('shows parallel toggle as checked when edge.parallel is true', () => {
      const edge: PipelineEdge = {
        from_node: '_source',
        to_node: 'parser1',
        when: null,
        parallel: true,
      };

      render(
        <EdgePredicateEditor
          edge={edge}
          dag={mockDag}
          fromNodeLabel="Source"
          toNodeLabel="Parser"
          onChange={vi.fn()}
        />
      );

      expect(screen.getByLabelText('Parallel edge')).toBeChecked();
    });

    it('calls onChange with parallel: true when toggle is checked', async () => {
      const user = userEvent.setup();
      const edge: PipelineEdge = {
        from_node: '_source',
        to_node: 'parser1',
        when: null,
        parallel: false,
      };
      const onChange = vi.fn();

      render(
        <EdgePredicateEditor
          edge={edge}
          dag={mockDag}
          fromNodeLabel="Source"
          toNodeLabel="Parser"
          onChange={onChange}
        />
      );

      await user.click(screen.getByLabelText('Parallel edge'));

      expect(onChange).toHaveBeenCalledWith({
        ...edge,
        parallel: true,
      });
    });

    it('calls onChange with parallel: false when toggle is unchecked', async () => {
      const user = userEvent.setup();
      const edge: PipelineEdge = {
        from_node: '_source',
        to_node: 'parser1',
        when: null,
        parallel: true,
      };
      const onChange = vi.fn();

      render(
        <EdgePredicateEditor
          edge={edge}
          dag={mockDag}
          fromNodeLabel="Source"
          toNodeLabel="Parser"
          onChange={onChange}
        />
      );

      await user.click(screen.getByLabelText('Parallel edge'));

      expect(onChange).toHaveBeenCalledWith({
        ...edge,
        parallel: false,
      });
    });

    it('is disabled when readOnly is true', () => {
      const edge: PipelineEdge = {
        from_node: '_source',
        to_node: 'parser1',
        when: null,
      };

      render(
        <EdgePredicateEditor
          edge={edge}
          dag={mockDag}
          fromNodeLabel="Source"
          toNodeLabel="Parser"
          onChange={vi.fn()}
          readOnly
        />
      );

      expect(screen.getByLabelText('Parallel edge')).toBeDisabled();
    });
  });
});
