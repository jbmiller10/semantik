import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';

import { DragPreviewEdge } from '../DragPreviewEdge';

describe('DragPreviewEdge', () => {
  const renderInSVG = (component: React.ReactElement) => {
    return render(<svg>{component}</svg>);
  };

  it('renders a path element', () => {
    const { container } = renderInSVG(
      <DragPreviewEdge from={{ x: 100, y: 100 }} to={{ x: 200, y: 300 }} />
    );

    const path = container.querySelector('path');
    expect(path).toBeInTheDocument();
  });

  it('renders path between two points', () => {
    const from = { x: 100, y: 100 };
    const to = { x: 200, y: 300 };

    const { container } = renderInSVG(<DragPreviewEdge from={from} to={to} />);

    const path = container.querySelector('path');
    expect(path).toHaveAttribute('d');

    const d = path?.getAttribute('d');
    // Path should start at the from point
    expect(d).toContain(`M ${from.x} ${from.y}`);
    // Path should end at the to point (cubic bezier ends with the to coordinates)
    expect(d).toContain(`${to.x} ${to.y}`);
  });

  it('has dashed stroke style', () => {
    const { container } = renderInSVG(
      <DragPreviewEdge from={{ x: 100, y: 100 }} to={{ x: 200, y: 300 }} />
    );

    const path = container.querySelector('path');
    expect(path).toHaveAttribute('stroke-dasharray', '6 4');
  });

  it('has no fill', () => {
    const { container } = renderInSVG(
      <DragPreviewEdge from={{ x: 100, y: 100 }} to={{ x: 200, y: 300 }} />
    );

    const path = container.querySelector('path');
    expect(path).toHaveAttribute('fill', 'none');
  });

  it('has pointer-events none class', () => {
    const { container } = renderInSVG(
      <DragPreviewEdge from={{ x: 100, y: 100 }} to={{ x: 200, y: 300 }} />
    );

    const path = container.querySelector('path');
    expect(path).toHaveClass('pointer-events-none');
  });

  it('has preview-edge class for animation', () => {
    const { container } = renderInSVG(
      <DragPreviewEdge from={{ x: 100, y: 100 }} to={{ x: 200, y: 300 }} />
    );

    const path = container.querySelector('path');
    expect(path).toHaveClass('preview-edge');
  });

  it('renders with correct stroke width', () => {
    const { container } = renderInSVG(
      <DragPreviewEdge from={{ x: 100, y: 100 }} to={{ x: 200, y: 300 }} />
    );

    const path = container.querySelector('path');
    expect(path).toHaveAttribute('stroke-width', '2');
  });

  it('uses cubic bezier curve', () => {
    const from = { x: 100, y: 100 };
    const to = { x: 200, y: 300 };

    const { container } = renderInSVG(<DragPreviewEdge from={from} to={to} />);

    const path = container.querySelector('path');
    const d = path?.getAttribute('d');

    // Should contain 'C' for cubic bezier
    expect(d).toContain('C');
  });
});
